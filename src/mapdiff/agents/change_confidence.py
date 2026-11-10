"""Change-confidence agent — scores every detected change before it is reported.

final = w_visual · visual + (1 − w_visual) · prior
  visual: an independent VLM looks at OLD vs NEW crops of the same spot (aligned) and judges whether
          the claimed change really happened (real → p, not real → 1 − p, unsure → 0.5)
  prior:  entity confidences × match-geometry quality (for positional matches) or relation confidence
Changes the reviewer confidently calls not real are marked ``rejected`` (reported separately, not dropped). So are
``unconfirmed`` changes (likely reading differences) unless the reviewer confidently calls them real.
A change made in the legend (code 'd' now means 'Costa' instead of 'Starbucks') looks identical at the spot on the map,
so the crops cannot show it; when both legends prove it, the legend counts as confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mapdiff.agents.change_detection import IDENTITY, DiffConfig, alignment_transforms
from mapdiff.agents.review import VerdictSchema, chunked, review_batches
from mapdiff.agents.symbols import ensure_legend_sheet
from mapdiff.agents.workspace import legend_examples, marked_crop, render_view
from mapdiff.ingest import Page
from mapdiff.llm.base import VLM, ImagePart, Part, TextPart
from mapdiff.models import FULL_PAGE, Change, ChangeSet, ChangeType, Snapshot
from mapdiff.names import name_similarity, norm_name

log = logging.getLogger(__name__)

_POSITIONAL = {ChangeType.REPLACED, ChangeType.REPURPOSED, ChangeType.RENAMED}
_RELATION = {ChangeType.RELATION_ADDED, ChangeType.RELATION_REMOVED, ChangeType.RELATION_CHANGED}


@dataclass
class ChangeConfidenceConfig:
    w_visual: float = 0.6
    review_threshold: float = 0.7  # below → needs_review
    reject_confidence: float = 0.8  # reviewer says "not real" at least this sure → rejected
    confirm_confidence: float = 0.7  # an unconfirmed change needs "real" at least this sure to be kept
    legend_confidence: float = 0.85  # confidence of a change both legends prove
    batch_size: int = 4
    crop_half: float = 0.04
    crop_px: int = 640
    workers: int = 4


REVIEW_SYSTEM = """You verify reported changes between two versions of the same airport floorplan. The detector \
that produced them can be fooled by extraction mistakes, so be sceptical.

For each numbered change you get the claim and two crops of the same location: OLD version and NEW version (the red \
circle marks the spot; the versions may be styled differently — new colours, icons or marker codes are NOT changes). \
Judge whether the claimed change really happened. The legend icons of both versions are shown: a restyled icon \
(new colour or tile) with the same shape is the same thing, not a change.
- real: the OLD crop shows the old state and the NEW crop shows the new state as claimed.
- not_real: both crops show the same thing (e.g. same shop, only restyled or misread), or the claim misdescribes it.
- unsure: the crops do not let you decide.
confidence is how sure you are of your verdict (0–1). Report every change with report_verdicts."""


_SCHEMA = VerdictSchema("change", "real", "not_real",
                        {"explanation": {"type": "string", "description": "what OLD and NEW actually show"}},
                        required=("explanation",))


def score_changes(cs: ChangeSet, old: Snapshot, new: Snapshot, old_pages: list[Page], new_pages: list[Page],
                  vlm: VLM, cfg: ChangeConfidenceConfig | None = None, dcfg: DiffConfig | None = None) -> None:
    cfg, dcfg = cfg or ChangeConfidenceConfig(), dcfg or DiffConfig()
    if not cs.changes:
        return
    transforms = {a.page: alignment_transforms(a) for a in cs.alignment}
    for snap, pages in ((old, old_pages), (new, new_pages)):
        for page in pages:
            ensure_legend_sheet(page, snap)

    def parts(batch: list[Change]) -> list[Part]:
        out: list[Part] = []
        if batch[0].type in _RELATION:  # one batch of table facts, judged against both versions' tables
            out += [TextPart("These changes come from tables/lists (not positions). Check them against both versions' tables.")]
            out += _table_views(old, old_pages, "OLD") + _table_views(new, new_pages, "NEW")
        else:  # few-shot: what each version's icons mean
            out += legend_examples(old_pages[batch[0].page], "The OLD version's") if batch[0].page < len(old_pages) else []
            out += legend_examples(new_pages[batch[0].page], "The NEW version's") if batch[0].page < len(new_pages) else []
        for k, c in enumerate(batch, 1):
            out.append(TextPart(f"Change {k} [{c.type.value}]: {c.summary}"))
            if c.type not in _RELATION:
                out += _entity_views(c, transforms.get(c.page, IDENTITY), old_pages, new_pages, cfg)
        out.append(TextPart(f"Judge all {len(batch)} changes, then call report_verdicts once."))
        return out

    relation_changes = [c for c in cs.changes if c.type in _RELATION]
    batches = chunked([c for c in cs.changes if c.type not in _RELATION], cfg.batch_size)
    batches += [relation_changes] if relation_changes else []
    log.info("change-confidence: %d changes in %d review batches", len(cs.changes), len(batches))
    verdicts = review_batches(vlm, REVIEW_SYSTEM, batches, parts, _SCHEMA, "change-confidence", cfg.workers)

    for c in cs.changes:
        v = verdicts.get(id(c))
        visual, p = _SCHEMA.score(v)
        prior = _prior(c, transforms, dcfg)
        final = cfg.w_visual * visual + (1 - cfg.w_visual) * prior
        c.confidence_breakdown = {"visual": round(visual, 3), "prior": round(prior, 3)}
        c.confidence = round(final, 3)
        c.verification_note = v.get("explanation", "") if v else "no verdict returned"
        legend = legend_change(c, old, new)
        if legend:  # the map spot looks the same; the legend proves the change
            c.confidence_breakdown["legend"] = 1.0
            c.confidence = round(max(final, cfg.legend_confidence), 3)
            c.verification_note = f"{legend}. {c.verification_note}".strip()
            c.rejected = False
            c.needs_review = c.confidence < cfg.review_threshold
            continue
        confirmed = bool(v) and v["verdict"] == "real" and p >= cfg.confirm_confidence
        c.rejected = (bool(v) and v["verdict"] == "not_real" and p >= cfg.reject_confidence) or (
            c.unconfirmed and not confirmed)
        if c.unconfirmed and not confirmed:
            c.verification_note = f"not confirmed by the visual check ({c.reason}). {c.verification_note}".strip()
        c.needs_review = not c.rejected and (final < cfg.review_threshold or c.needs_review)


def legend_change(c: Change, old: Snapshot, new: Snapshot) -> str | None:
    """'legend entry changed: X → Y' when the old name is only in the old legend and the new name only in the new one."""
    if c.type not in _POSITIONAL or not (c.old and c.new and c.old.name and c.new.name):
        return None
    before = [l.meaning for l in old.legend if l.page == c.old.page]
    after = [l.meaning for l in new.legend if l.page == c.new.page]
    o, n = c.old.name, c.new.name
    if _listed(o, before) and _listed(n, after) and not _listed(o, after) and not _listed(n, before):
        return f"legend entry changed: '{o}' → '{n}'"
    return None


def _listed(name: str, meanings: list[str]) -> bool:
    """``name`` is one of the legend's meanings (allowing a meaning like 'Exki (Food & Drinks)')."""
    key = norm_name(name)
    return bool(key) and any(name_similarity(name, m) >= 0.9 or (len(key) >= 4 and key in norm_name(m)) for m in meanings)


def _prior(c: Change, transforms, dcfg: DiffConfig) -> float:
    if c.type in _RELATION:
        return min(r.confidence for r in (c.old_relation, c.new_relation) if r is not None)
    prior = min(e.confidence for e in (c.old, c.new) if e is not None)
    if c.type in _POSITIONAL and c.old and c.new:
        to_old, _ = transforms.get(c.new.page, IDENTITY)
        d = c.old.anchor.dist(to_old(c.new.anchor))
        prior *= 1 - 0.3 * min(1.0, d / dcfg.match_radius)  # far-apart matches are less certain
    return prior


def _entity_views(c: Change, transforms, old_pages: list[Page], new_pages: list[Page],
                  cfg: ChangeConfidenceConfig) -> list[Part]:
    """OLD/NEW crops of the same physical spot (moves get both spots)."""
    page = c.page
    to_old, to_new = transforms
    if page >= len(old_pages) or page >= len(new_pages):
        return [TextPart("(page missing in one version)")]
    if c.type == ChangeType.MOVED:
        spots = [("old position", c.old.anchor, to_new(c.old.anchor)), ("new position", to_old(c.new.anchor), c.new.anchor)]
    else:
        old_at = c.old.anchor if c.old else to_old(c.new.anchor)
        new_at = c.new.anchor if c.new else to_new(c.old.anchor)
        spots = [("location", old_at, new_at)]
    parts: list[Part] = []
    for name, o, n in spots:
        parts += [TextPart(f"OLD version, {name}:"), ImagePart(marked_crop(old_pages[page].image, o, cfg.crop_half, cfg.crop_px)),
                  TextPart(f"NEW version, {name}:"), ImagePart(marked_crop(new_pages[page].image, n, cfg.crop_half, cfg.crop_px))]
    return parts


def _table_views(snap: Snapshot, pages: list[Page], which: str) -> list[Part]:
    parts: list[Part] = []
    for p in snap.panels:
        if p.kind in ("table", "legend") and p.page < len(pages):
            parts += [TextPart(f"{which} version, panel '{p.id}':"), ImagePart(render_view(pages[p.page].image, p.bbox, 1568, grid=False))]
    if not parts and pages:
        parts = [TextPart(f"{which} version, whole page:"), ImagePart(render_view(pages[0].image, FULL_PAGE, 1568, grid=False))]
    return parts
