import json
from pathlib import Path

from mapdiff.evaluation import score_case
from mapdiff.models import Change, ChangeSet, ChangeType, Entity, Point, Relation
from mapdiff.taxonomy import Category

MANIFEST = Path(__file__).resolve().parents[1] / "Test_files_edited" / "manifest.json"


def ent(cat, name, x, y):
    return Entity(id="e", category=Category(cat), name=name, anchor=Point(x=x, y=y))


def test_scoring_matches_types_names_locations_and_relations():
    expected = [
        {"type": "replaced", "old_name": "Starbucks", "new_name": "Costa Coffee"},
        {"type": "removed", "category": "toilet", "at": [0.59, 0.43]},
        {"type": "relation_changed", "subject": "Emirates", "old_object": "14", "new_object": "5"},
        {"type": "added", "category": "lift", "at": [0.76, 0.51]},
    ]
    cs = ChangeSet(map_id="m", from_version=1, to_version=2, changes=[
        Change(type=ChangeType.REPLACED, summary="cafe: Starbucks → Costa Coffee",
               old=ent("cafe", "Starbucks", 0.1, 0.1), new=ent("cafe", "Costa Coffee", 0.1, 0.1)),
        Change(type=ChangeType.REMOVED, summary="toilet removed", old=ent("toilet", None, 0.595, 0.425)),
        Change(type=ChangeType.RELATION_CHANGED, summary="Emirates desk",
               old_relation=Relation(subject="Emirates", predicate="checks_in_at", object="Check-in 14"),
               new_relation=Relation(subject="Emirates", predicate="checks_in_at", object="Check-in 5")),
        Change(type=ChangeType.ADDED, summary="lift added", new=ent("lift", None, 0.76, 0.51), rejected=True),
        Change(type=ChangeType.ADDED, summary="atm added", new=ent("atm", None, 0.3, 0.3)),
    ])
    s = score_case("c", expected, cs)
    assert len(s.true_positives) == 3
    assert [e["type"] for e in s.missed] == ["added"]
    assert s.wrongly_rejected[0][1] == "lift added"
    assert s.false_positives == ["atm added"]
    assert s.recall == 0.75 and s.precision == 0.75


def test_manifest_is_well_formed():
    if not MANIFEST.exists():
        return
    cases = json.loads(MANIFEST.read_text())["cases"]
    root = MANIFEST.parents[1]
    assert {c["kind"] for c in cases} == {"edit", "restyle"}
    for c in cases:
        assert (root / c["baseline"]).exists() and (root / c["variant"]).exists()
        assert (c["kind"] == "restyle") == (c["expected"] == [])
        for e in c["expected"]:
            if "category" in e:
                Category(e["category"])
            if "at" in e:
                assert all(0 <= v <= 1 for v in e["at"])
