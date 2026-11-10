"""Lessons: mistakes found on one version are remembered and used on the next."""

from mapdiff.agents import lessons
from mapdiff.agents.change_detection import detect_changes
from mapdiff.models import BBox, ChangeSet, Entity, Lesson, Point, Snapshot, Verification
from mapdiff.taxonomy import Category


def ent(id, cat, name, x, y, issue=""):
    return Entity(id=id, category=Category(cat), name=name, anchor=Point(x=x, y=y),
                  verification=Verification(score=0.4, verdict="incorrect", method="vlm", issue=issue))


def snap(*ents, version=1):
    return Snapshot(map_id="t", version=version, source_file=f"v{version}.jpg", source_sha256="x", entities=list(ents))


def test_corrections_become_misread_lessons():
    before = {"a": ent("a", "check_in", "Check-in 14", 0.52, 0.41, "duplicate of the island label"),
              "b": ent("b", "toilet", None, 0.3, 0.3, "this is a lift"),
              "c": ent("c", "cafe", "Starbucks", 0.6, 0.6)}
    after = [ent("b", "lift", None, 0.3, 0.3), ent("c", "cafe", "Starbucks", 0.6, 0.6)]  # a removed, b fixed, c kept
    got = lessons.misreads(before, after, "v1.jpg")
    assert [(l.wrong, l.right) for l in got] == [("check_in 'Check-in 14'", "nothing there (removed)"),
                                                ("toilet '-'", "lift '-'")]
    assert got[0].why == "duplicate of the island label"


def test_a_rejected_change_is_held_back_when_claimed_again():
    old = snap(Entity(id="a", category=Category.CURRENCY_EXCHANGE, name="RIA Money Transfer", anchor=Point(x=0.4, y=0.4)))
    new = snap(Entity(id="a", category=Category.SERVICE_COUNTER, name="RIA Money Transfer", anchor=Point(x=0.4, y=0.4)),
               version=2)
    first, _ = detect_changes(old, new)
    first.changes[0].unconfirmed = False  # as if detection had not doubted it
    first.changes[0].rejected, first.changes[0].verification_note = True, "same counter in both crops"
    learned = lessons.false_changes(first, "v2.jpg")
    assert learned[0].kind == "false_change" and "RIA Money Transfer" in learned[0].wrong

    again = ChangeSet(map_id="t", from_version=2, to_version=3, changes=[c.model_copy(deep=True) for c in first.changes])
    for c in again.changes:
        c.rejected = c.unconfirmed = c.needs_review = False
    assert lessons.flag_repeats(again, learned) == 1
    assert again.changes[0].unconfirmed and "earlier version" in again.changes[0].reason


def test_lessons_carry_forward_without_repeats():
    a = Lesson(kind="misread", at=Point(x=0.1, y=0.1), wrong="x", right="y")
    b = Lesson(kind="misread", at=Point(x=0.105, y=0.1), wrong="x", right="z")  # same mistake, newer fix
    c = Lesson(kind="misread", at=Point(x=0.8, y=0.8), wrong="x", right="y")
    kept = lessons.carry([a, c], [b])
    assert [l.right for l in kept] == ["y", "z"] and kept[0].at.x == 0.8
    assert lessons.near(kept, 0, BBox(x0=0, y0=0, x1=0.2, y1=0.2)) == [b]
