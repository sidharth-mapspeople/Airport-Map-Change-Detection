"""Agent 3 — reporting. Kept deliberately simple for now: a Markdown report from the changeset."""

from __future__ import annotations

from mapdiff.llm.usage import total
from mapdiff.models import AgentUsage, Change, ChangeSet, ChangeType, Snapshot

_ORDER = [
    ChangeType.REPLACED, ChangeType.REPURPOSED, ChangeType.ADDED, ChangeType.REMOVED, ChangeType.MOVED,
    ChangeType.RENAMED, ChangeType.RELATION_CHANGED, ChangeType.RELATION_ADDED, ChangeType.RELATION_REMOVED,
]


def render_markdown(cs: ChangeSet, old: Snapshot, new: Snapshot) -> str:
    accepted = [c for c in cs.changes if not c.rejected]
    rejected = [c for c in cs.changes if c.rejected]
    out = [
        f"# Floorplan changes — {cs.map_id}",
        "",
        f"Version {cs.from_version} (`{old.source_file}`) → version {cs.to_version} (`{new.source_file}`)",
        "",
        f"- Changes: **{len(accepted)}** ({sum(c.needs_review for c in accepted)} need review)"
        + (f"; {len(rejected)} rejected as likely extraction noise" if rejected else ""),
        f"- Unchanged items: {cs.unchanged}",
    ]
    for label, snap in (("old", old), ("new", new)):
        q = snap.quality
        if q:
            status = "passed" if q.passed else "BELOW THRESHOLD"
            out.append(f"- Mapping confidence ({label}, v{snap.version}): {q.mapping_confidence:.3f} "
                       f"(threshold {q.threshold:.2f}, {status}, {q.corrections_used} correction round(s))")
    if old.map_date or new.map_date:
        out.append(f"- Printed map date: {old.map_date or '?'} → {new.map_date or '?'}")
    for a in cs.alignment:
        if a.method == "similarity":
            out.append(f"- Page {a.page}: layout shifted/scaled (scale {a.scale:.3f}, {a.anchors} anchors) — compensated")
    out.append("")

    if not accepted:
        out.append("No changes detected.")
    for t in _ORDER:
        group = [c for c in accepted if c.type == t]
        if group:
            out += [f"## {t.value.replace('_', ' ').capitalize()} ({len(group)})", ""]
            out += [line for c in sorted(group, key=lambda c: (c.needs_review, -c.confidence)) for line in _item(c)]
            out.append("")

    if rejected:
        out += ["## Rejected (judged not real by change-confidence review)", ""]
        out += [line for c in rejected for line in _item(c)] + [""]
    if cs.presentation_notes:
        out += ["## Ignored presentation differences", ""] + [f"- {n}" for n in cs.presentation_notes] + [""]
    usage = new.usage + cs.usage
    if usage:
        out += ["## Model usage (this version)", "", usage_table(usage, new.llm), ""]
    return "\n".join(out)


def usage_table(usage: list[AgentUsage], llm: dict[str, str] | None = None) -> str:
    """Markdown table of tokens, cost and time per agent, with a total row."""
    def money(v: float | None) -> str:
        return "n/a" if v is None else f"${v:.4f}"

    rows = ["| Agent | Model | Calls | Input tok | Output tok | Cache read | Cost | Wall time |",
            "|---|---|---|---|---|---|---|---|"]
    for u in [*usage, total(usage)]:
        model = (llm or {}).get(u.agent) or ", ".join(u.models)
        name = f"**{u.agent}**" if u.agent == "total" else u.agent
        rows.append(f"| {name} | {model if u.agent != 'total' else ''} | {u.calls} | {u.input_tokens:,} "
                    f"| {u.output_tokens:,} | {u.cache_read_tokens:,} | {money(u.cost_usd)} | {u.wall_seconds:.0f}s |")
    return "\n".join(rows)


def confidence_text(c: Change) -> str:
    """'confidence 0.83 (visual 0.90, prior 0.72)'."""
    parts = ", ".join(f"{k} {v:.2f}" for k, v in c.confidence_breakdown.items())
    return f"confidence {c.confidence:.2f}" + (f" ({parts})" if parts else "")


def _item(c: Change) -> list[str]:
    flag = " ⚠️ needs review" if c.needs_review else ""
    where = c.new or c.old
    loc = f" — at ({where.anchor.x:.3f}, {where.anchor.y:.3f}), page {where.page}" if where else ""
    lines = [f"- {c.summary}{loc} · {confidence_text(c)}{flag}"]
    lines += [f"  - {note}" for note in (c.reason, c.verification_note) if note]
    return lines
