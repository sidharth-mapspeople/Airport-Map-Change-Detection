"""Agent 2 — semantic change detection between two snapshots of the same map.

Only meaning is compared: category group, name and position (plus relations).
Legend symbols, colours, icons and marker codes are presentation and are ignored,
so a restyled map with the same content yields no changes.

Steps per page:
  1. align   — fit a similarity transform from names present in both versions (robust to re-layout)
  2. match by identity  — same name & group anywhere on the page (detects moves)
  3. match by position  — remaining items near each other (detects replacements / repurposing)
  4. leftovers          — added / removed
  5. relations          — set difference of who-uses-which-facility facts, collapsed into 'changed' per subject
Positional matches that look like a reading difference rather than a real change (a name read in one version only,
the same name under another category, a vague category) are marked ``unconfirmed``: change scoring only keeps them
when the visual check confirms them.
Ambiguous name pairs can be adjudicated by a VLM (optional).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from mapdiff.llm.base import VLM, TextPart, ToolSpec, UserTurn
from mapdiff.models import Alignment, Change, ChangeSet, ChangeType, Entity, Point, Relation, Snapshot
from mapdiff.names import name_similarity, norm_name
from mapdiff.taxonomy import RELATION_PREDICATES, Category, group_of

# Decides whether two differently written names denote the same occupant. None = unsure.
Adjudicator = Callable[[Entity, Entity], bool | None]

_INF = 1e9
IDENTITY = (lambda p: p, lambda p: p)  # (new→old, old→new) when versions share one frame


@dataclass
class DiffConfig:
    same_name: float = 0.9  # name similarity treated as "same name"
    ambiguous_name: float = 0.6  # between this and same_name -> ask the adjudicator
    match_radius: float = 0.025  # max distance (page units) for positional matching
    move_threshold: float = 0.02  # same item displaced more than this -> MOVED
    review_confidence: float = 0.6  # changes involving less-confident entities need review


def detect_changes(
    old: Snapshot, new: Snapshot, cfg: DiffConfig | None = None, adjudicator: Adjudicator | None = None
) -> tuple[ChangeSet, dict[str, str]]:
    """Return the changeset and a mapping new_entity_id -> old_entity_id for matched entities."""
    cfg = cfg or DiffConfig()
    cs = ChangeSet(map_id=new.map_id, from_version=old.version, to_version=new.version)
    matched: dict[str, str] = {}

    for page in sorted({e.page for e in old.entities} | {e.page for e in new.entities}):
        olds = [e for e in old.entities if e.page == page]
        news = [e for e in new.entities if e.page == page]
        align, to_old = _align(olds, news, page, cfg)
        cs.alignment.append(align)
        pos = {e.id: to_old(e.anchor) for e in news}
        _match_page(olds, news, pos, cfg, adjudicator, cs, matched)

    cs.changes += _relation_changes(old.relations, new.relations)
    cs.presentation_notes = _presentation_notes(old, new)
    return cs, matched


# ---------------- alignment ----------------


def _align(olds: list[Entity], news: list[Entity], page: int, cfg: DiffConfig):
    """Fit new->old similarity transform on uniquely named items present in both versions."""

    def unique_by_name(ents: list[Entity]) -> dict[tuple[str, str], Entity]:
        buckets: dict[tuple[str, str], list[Entity]] = defaultdict(list)
        for e in ents:
            if norm_name(e.name):
                buckets[(group_of(e.category), norm_name(e.name))].append(e)
        return {k: v[0] for k, v in buckets.items() if len(v) == 1}

    uo, un = unique_by_name(olds), unique_by_name(news)
    keys = sorted(uo.keys() & un.keys())
    src = np.array([[un[k].anchor.x, un[k].anchor.y] for k in keys])
    dst = np.array([[uo[k].anchor.x, uo[k].anchor.y] for k in keys])

    identity = Alignment(page=page, method="identity", anchors=len(keys))
    if len(keys) < 3:
        return identity, IDENTITY[0]

    inliers = np.ones(len(keys), dtype=bool)
    for _ in range(3):  # fit, drop outliers (genuinely moved items), refit
        s, R, t = _umeyama(src[inliers], dst[inliers])
        resid = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
        thr = max(3 * float(np.median(resid[inliers])), 0.01)
        new_inliers = resid <= thr
        if new_inliers.sum() < 3 or (new_inliers == inliers).all():
            break
        inliers = new_inliers
    s, R, t = _umeyama(src[inliers], dst[inliers])
    resid = float(np.median(np.linalg.norm((s * src[inliers] @ R.T + t) - dst[inliers], axis=1)))

    # Only trust a transform that clearly beats identity.
    ident_resid = float(np.median(np.linalg.norm(src - dst, axis=1)))
    if ident_resid <= resid + 1e-4:
        identity.residual = ident_resid
        return identity, IDENTITY[0]

    al = Alignment(page=page, method="similarity", anchors=int(inliers.sum()), scale=float(s),
                   rotation_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])), tx=float(t[0]), ty=float(t[1]),
                   residual=resid)
    return al, alignment_transforms(al)[0]


def alignment_transforms(al: Alignment):
    """(new→old, old→new) point mappings for a stored alignment."""
    th = math.radians(al.rotation_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = np.array([al.tx, al.ty])

    def to_old(p: Point) -> Point:
        x, y = al.scale * R @ np.array([p.x, p.y]) + t
        return Point(x=float(x), y=float(y))

    def to_new(p: Point) -> Point:
        x, y = R.T @ (np.array([p.x, p.y]) - t) / al.scale
        return Point(x=float(x), y=float(y))

    return to_old, to_new


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform dst ≈ s·R·src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


# ---------------- entity matching ----------------


def _match_page(olds, news, pos, cfg: DiffConfig, adjudicator, cs: ChangeSet, matched: dict[str, str]) -> None:
    def dist(o: Entity, n: Entity) -> float:
        return o.anchor.dist(pos[n.id])

    # Pass 1: same identity (named items, same group) anywhere -> unchanged or moved.
    def identity_cost(o: Entity, n: Entity) -> float:
        if not norm_name(o.name) or group_of(o.category) != group_of(n.category):
            return _INF
        return dist(o, n) if name_similarity(o.name, n.name) >= cfg.same_name else _INF

    for o, n in _assign(olds, news, identity_cost):
        matched[n.id] = o.id
        d = dist(o, n)
        if d > cfg.move_threshold:
            cs.changes.append(_change(ChangeType.MOVED, o, n, cfg, f"{_desc(n)} moved ({d:.3f} page units)"))
        else:
            cs.unchanged += 1
    olds = [o for o in olds if o.id not in matched.values()]
    news = [n for n in news if n.id not in matched]

    # Pass 2: positional -> unchanged facility, replaced occupant, repurposed space.
    def position_cost(o: Entity, n: Entity) -> float:
        d = dist(o, n)
        if d > cfg.match_radius:
            return _INF
        return d + (0 if group_of(o.category) == group_of(n.category) else cfg.match_radius)

    for o, n in _assign(olds, news, position_cost):
        matched[n.id] = o.id
        same_group = group_of(o.category) == group_of(n.category)
        sim = name_similarity(o.name, n.name)
        if same_group and sim >= cfg.same_name:
            cs.unchanged += 1
        elif same_group and norm_name(o.name) and norm_name(n.name):
            verdict = adjudicator(o, n) if adjudicator and sim >= cfg.ambiguous_name else None
            if verdict is True:
                cs.changes.append(_change(ChangeType.RENAMED, o, n, cfg,
                                          f"'{o.name}' is now written '{n.name}'", reason="adjudicated: same occupant"))
            else:
                ch = _change(ChangeType.REPLACED, o, n, cfg, f"{o.category.value}: '{o.name}' → '{n.name}'")
                ch.needs_review |= verdict is None and sim >= cfg.ambiguous_name
                cs.changes.append(ch)
        elif same_group:
            # Name present on one side only: usually an extraction gap.
            ch = _change(ChangeType.REPLACED, o, n, cfg, f"{o.category.value}: '{o.name or '?'}' → '{n.name or '?'}'")
            cs.changes.append(_unconfirmed(ch, "name present in only one version"))
        else:
            ch = _change(ChangeType.REPURPOSED, o, n, cfg, f"{_desc(o)} → {_desc(n)}")
            doubt = _repurpose_doubt(o, n, cfg)
            cs.changes.append(_unconfirmed(ch, doubt) if doubt else ch)
    olds = [o for o in olds if o.id not in matched.values()]
    news = [n for n in news if n.id not in matched]

    cs.changes += [_change(ChangeType.REMOVED, o, None, cfg, f"{_desc(o)} removed") for o in olds]
    cs.changes += [_change(ChangeType.ADDED, None, n, cfg, f"{_desc(n)} added") for n in news]


_VAGUE = {Category.OTHER, Category.ZONE}


def _repurpose_doubt(o: Entity, n: Entity, cfg: DiffConfig) -> str:
    """Why a category change at the same spot is more likely a reading difference than a real one ('' if none)."""
    named = bool(norm_name(o.name)), bool(norm_name(n.name))
    if all(named) and name_similarity(o.name, n.name) >= cfg.same_name:
        return "same name in both versions; only the category reading differs"
    if any(named) and not all(named):
        return "name present in only one version"
    if o.category in _VAGUE or n.category in _VAGUE:
        return "vague category in one version"
    if not any(named):
        return "unnamed in both versions; the icon may have been read differently"
    return ""


def _unconfirmed(ch: Change, reason: str) -> Change:
    ch.unconfirmed = ch.needs_review = True
    ch.reason = reason
    return ch


def _assign(olds: list[Entity], news: list[Entity], cost_fn) -> list[tuple[Entity, Entity]]:
    if not olds or not news:
        return []
    cost = np.array([[cost_fn(o, n) for n in news] for o in olds])
    rows, cols = linear_sum_assignment(cost)
    return [(olds[r], news[c]) for r, c in zip(rows, cols) if cost[r, c] < _INF]


def _desc(e: Entity) -> str:
    where = f" on {e.level}" if e.level else ""
    return f"{e.category.value} '{e.name}'{where}" if e.name else f"{e.category.value}{where} at ({e.anchor.x:.2f}, {e.anchor.y:.2f})"


def _change(t: ChangeType, o: Entity | None, n: Entity | None, cfg: DiffConfig, summary: str, reason: str = "") -> Change:
    conf = min(e.confidence for e in (o, n) if e is not None)
    return Change(type=t, old=o, new=n, summary=summary, confidence=conf,
                  needs_review=conf < cfg.review_confidence, reason=reason)


# ---------------- relations ----------------


def _relation_changes(old: list[Relation], new: list[Relation]) -> list[Change]:
    def key(r: Relation) -> tuple[str, str, str]:
        return norm_name(r.subject), norm_name(r.predicate), norm_name(r.object)

    def facts(rels: list[Relation]) -> dict[tuple[str, str, str], Relation]:
        # Older snapshots may hold other predicates (titles, legend contents, ...); those are not compared.
        return {key(r): r for r in rels if r.predicate in RELATION_PREDICATES}

    ok, nk = facts(old), facts(new)
    removed = [ok[k] for k in sorted(ok.keys() - nk.keys())]
    added = [nk[k] for k in sorted(nk.keys() - ok.keys())]

    def subject(r: Relation) -> tuple[str, str]:
        return norm_name(r.subject), norm_name(r.predicate)

    changes = []
    added_by_subject = defaultdict(list)
    for r in added:
        added_by_subject[subject(r)].append(r)
    for r in removed:
        bucket = added_by_subject.get(subject(r))
        if bucket:
            a = bucket.pop(0)
            changes.append(_relation_change(ChangeType.RELATION_CHANGED, r, a,
                                            f"{r.subject} {_verb(r)}: {r.object} → {a.object}"))
        else:
            changes.append(_relation_change(ChangeType.RELATION_REMOVED, r, None,
                                            f"{r.subject} no longer {_verb(r)} {r.object}"))
    changes += [_relation_change(ChangeType.RELATION_ADDED, None, a, f"{a.subject} now {_verb(a)} {a.object}")
                for bucket in added_by_subject.values() for a in bucket]
    return changes


def _verb(r: Relation) -> str:
    return r.predicate.replace("_", " ")


def _relation_change(t: ChangeType, old: Relation | None, new: Relation | None, summary: str) -> Change:
    conf = min(r.confidence for r in (old, new) if r is not None)
    return Change(type=t, old_relation=old, new_relation=new, confidence=conf, summary=summary)


def _presentation_notes(old: Snapshot, new: Snapshot) -> list[str]:
    notes = []
    o_sym = {(l.kind, norm_name(l.symbol), norm_name(l.meaning)) for l in old.legend}
    n_sym = {(l.kind, norm_name(l.symbol), norm_name(l.meaning)) for l in new.legend}
    if o_sym != n_sym:
        notes.append(f"Legend presentation differs ({len(o_sym - n_sym)} entries only in old, "
                     f"{len(n_sym - o_sym)} only in new); styling and codes are ignored for change detection.")
    return notes


# ---------------- optional VLM adjudicator ----------------


def vlm_adjudicator(vlm: VLM) -> Adjudicator:
    """Ask the model whether two names at the same spot denote the same occupant (rebrand/spelling)."""
    spec = ToolSpec(
        "verdict",
        "Report whether the two records denote the same occupant.",
        {"type": "object", "properties": {"same": {"type": ["boolean", "null"]}, "reason": {"type": "string"}},
         "required": ["same", "reason"]},
    )

    def judge(o: Entity, n: Entity) -> bool | None:
        prompt = (
            "Two versions of the same airport floorplan show these items at the same location. Is it the same "
            "occupant (spelling variant, abbreviation, minor rebrand) or a different one (a new tenant)? "
            "Use null if you cannot tell. Answer by calling the verdict tool.\n\n"
            f"OLD: {json.dumps(o.model_dump(include={'category', 'name', 'label', 'evidence'}), ensure_ascii=False)}\n"
            f"NEW: {json.dumps(n.model_dump(include={'category', 'name', 'label', 'evidence'}), ensure_ascii=False)}"
        )
        turn = vlm.complete("You compare records from airport maps.", [UserTurn(parts=[TextPart(prompt)])], [spec])
        call = next((c for c in turn.tool_calls if c.name == "verdict"), None)
        return call.arguments.get("same") if call else None

    return judge
