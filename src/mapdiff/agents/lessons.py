"""Lessons agent — deterministic, no model calls: remembers the mistakes made on a map so they are not repeated.

Two kinds of lesson are kept on each snapshot and carried forward to the next version:
  * misread      — an item the reviewer doubted that the correction round then changed or removed
                   ("Check-in 14 at (0.52, 0.41) was a duplicate"). The next reading of that part of the map is
                   told about it.
  * false_change — a change claim the visual check rejected as not real ("RIA Money Transfer: currency exchange →
                   service counter"). The same claim at the same spot on a later version is only reported if the
                   visual check confirms it this time.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from mapdiff.models import BBox, Change, ChangeSet, Entity, Lesson, Point

RADIUS = 0.02  # page units: a lesson applies to things this close to where it was learned
MAX_LESSONS = 300  # most recent lessons kept per map
SAME_CLAIM = 85  # summary similarity (0–100) for "the same change claim"


def describe(e: Entity) -> str:
    return f"{e.category.value} '{e.name or '-'}'" + (f" [{e.label}]" if e.label else "")


def misreads(before: dict[str, Entity], after: list[Entity], source: str) -> list[Lesson]:
    """Lessons from the correction round: ``before`` are the doubted entities as first read, ``after`` the result."""
    now = {e.id: e for e in after}
    out = []
    for eid, old in before.items():
        new = now.get(eid)
        why = old.verification.issue if old.verification else ""
        if new is None:
            out.append(Lesson(kind="misread", page=old.page, at=old.anchor, wrong=describe(old),
                              right="nothing there (removed)", why=why, learned_from=source))
        elif describe(new) != describe(old) or new.anchor.dist(old.anchor) > 0.01:
            right = describe(new) + (f" at ({new.anchor.x:.3f}, {new.anchor.y:.3f})"
                                     if new.anchor.dist(old.anchor) > 0.01 else "")
            out.append(Lesson(kind="misread", page=old.page, at=old.anchor, wrong=describe(old), right=right, why=why,
                              learned_from=source))
    return out


def false_changes(cs: ChangeSet, source: str) -> list[Lesson]:
    """Lessons from change claims the visual check rejected (positional changes only)."""
    return [Lesson(kind="false_change", page=c.page, at=at, wrong=c.summary, right="no real change",
                   why=c.verification_note[:200], learned_from=source)
            for c in cs.changes if c.rejected and (at := _where(c)) is not None]


def carry(earlier: list[Lesson], new: list[Lesson]) -> list[Lesson]:
    """Earlier lessons plus new ones, without repeats, most recent last, at most MAX_LESSONS."""
    kept = [l for l in earlier if not any(_same(l, n) for n in new)] + new
    return kept[-MAX_LESSONS:]


def near(lessons: list[Lesson], page: int, box: BBox, kind: str = "misread") -> list[Lesson]:
    return [l for l in lessons if l.kind == kind and l.page == page and box.contains(l.at)]


def rows(lessons: list[Lesson]) -> str:
    return "\n".join(f"- at ({l.at.x:.3f}, {l.at.y:.3f}): read as {l.wrong} → really {l.right}"
                     + (f" — {l.why}" if l.why else "") for l in lessons)


def flag_repeats(cs: ChangeSet, lessons: list[Lesson]) -> int:
    """Mark changes that repeat a rejected claim as unconfirmed; returns how many were marked."""
    known = [l for l in lessons if l.kind == "false_change"]
    marked = 0
    for c in cs.changes:
        at = _where(c)
        if at is None or c.unconfirmed:
            continue
        if any(l.page == c.page and l.at.dist(at) < RADIUS and fuzz.ratio(l.wrong, c.summary) >= SAME_CLAIM
               for l in known):
            c.unconfirmed = c.needs_review = True
            c.reason = "the same change was claimed and rejected on an earlier version"
            marked += 1
    return marked


def _where(c: Change) -> Point | None:
    e = c.new or c.old
    return e.anchor if e is not None else None


def _same(a: Lesson, b: Lesson) -> bool:
    return a.kind == b.kind and a.page == b.page and a.at.dist(b.at) < RADIUS and a.wrong == b.wrong
