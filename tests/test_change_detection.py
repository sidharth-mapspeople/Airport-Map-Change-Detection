from mapdiff.agents.change_detection import detect_changes
from mapdiff.models import ChangeType, Entity, LegendEntry, Point, Relation, Snapshot
from mapdiff.taxonomy import Category

BASE = [
    ("cafe", "Culto Café", "8", 0.20, 0.30),
    ("restaurant", "McDonald's", "37", 0.40, 0.35),
    ("retail", "WHSmith", "h", 0.55, 0.30),
    ("toilet", None, None, 0.30, 0.50),
    ("toilet", None, None, 0.70, 0.50),
    ("lift", None, None, 0.50, 0.60),
    ("gate", "Gate A40", "A40", 0.10, 0.80),
    ("gate", "Gate A42", "A42", 0.20, 0.80),
    ("lounge", "Al Dar Lounge", "4", 0.80, 0.20),
]


def snap(rows, version=1, relations=(), legend=(), transform=lambda x, y: (x, y)):
    ents = []
    for i, (cat, name, label, x, y) in enumerate(rows):
        tx, ty = transform(x, y)
        ents.append(Entity(id=f"e{i}", category=Category(cat), name=name, label=label, anchor=Point(x=tx, y=ty)))
    return Snapshot(map_id="t", version=version, source_file="f.pdf", source_sha256="x", entities=ents,
                    relations=list(relations), legend=list(legend))


def types(cs):
    return sorted(c.type.value for c in cs.changes)


def test_identical_content_has_no_changes():
    cs, matched = detect_changes(snap(BASE), snap(BASE, 2))
    assert cs.changes == []
    assert cs.unchanged == len(BASE)
    assert len(matched) == len(BASE)


def test_restyle_relabel_and_relayout_is_not_a_change():
    """New colours/icons/codes and a shifted, scaled layout must report nothing."""
    restyled = [(c, n, (l or "") + "-new", x, y) for c, n, l, x, y in BASE]
    old = snap(BASE, legend=[LegendEntry(kind="color", symbol="green fill", meaning="Restaurants & Cafes")])
    new = snap(restyled, 2, legend=[LegendEntry(kind="color", symbol="purple fill", meaning="Food & Drinks")],
               transform=lambda x, y: (0.05 + 0.9 * x, 0.02 + 0.9 * y))
    cs, _ = detect_changes(old, new)
    assert cs.changes == []
    assert cs.alignment[0].method == "similarity"
    assert cs.presentation_notes  # styling difference is noted, not reported as a change


def test_category_refinement_within_group_is_not_a_change():
    rows = [r if r[1] != "Culto Café" else ("restaurant", *r[1:]) for r in BASE]
    cs, _ = detect_changes(snap(BASE), snap(rows, 2))
    assert cs.changes == []


def test_spelling_variant_is_not_a_change():
    rows = [r if r[1] != "Culto Café" else ("cafe", "CULTO CAFE", *r[2:]) for r in BASE]
    cs, _ = detect_changes(snap(BASE), snap(rows, 2))
    assert cs.changes == []


def test_cafe_replaced_by_another_cafe():
    rows = [r if r[1] != "Culto Café" else ("cafe", "Starbucks", "8", 0.201, 0.299) for r in BASE]
    cs, _ = detect_changes(snap(BASE), snap(rows, 2))
    assert types(cs) == ["replaced"]
    assert "Culto Café" in cs.changes[0].summary and "Starbucks" in cs.changes[0].summary


def test_shop_repurposed_as_cafe():
    rows = [r if r[1] != "WHSmith" else ("cafe", "Costa Coffee", "h", 0.55, 0.30) for r in BASE]
    cs, _ = detect_changes(snap(BASE), snap(rows, 2))
    assert types(cs) == ["repurposed"]


def test_added_removed_and_moved():
    rows = [r for r in BASE if r[3:] != (0.70, 0.50)]  # one toilet removed
    rows.append(("atm", None, None, 0.90, 0.90))  # ATM added
    rows = [r if r[1] != "Al Dar Lounge" else (*r[:3], 0.60, 0.10) for r in rows]  # lounge moved
    cs, _ = detect_changes(snap(BASE), snap(rows, 2))
    assert types(cs) == ["added", "moved", "removed"]


def test_airline_moves_desk():
    old = snap(BASE, relations=[Relation(subject="Emirates", predicate="checks_in_at", object="Check-in 14"),
                                Relation(subject="KLM", predicate="checks_in_at", object="Check-in 5")])
    new = snap(BASE, 2, relations=[Relation(subject="Emirates", predicate="checks_in_at", object="Check-in 5"),
                                   Relation(subject="KLM", predicate="checks_in_at", object="Check-in 5"),
                                   Relation(subject="Wizz Air", predicate="checks_in_at", object="Check-in 12")])
    cs, _ = detect_changes(old, new)
    assert types(cs) == ["relation_added", "relation_changed"]
    changed = next(c for c in cs.changes if c.type == ChangeType.RELATION_CHANGED)
    assert "Check-in 14" in changed.summary and "Check-in 5" in changed.summary


def test_name_missing_in_one_version_is_unconfirmed():
    rows = [r if r[1] != "Culto Café" else ("cafe", None, "8", 0.20, 0.30) for r in BASE]
    cs, _ = detect_changes(snap(BASE), snap(rows, 2))
    assert len(cs.changes) == 1 and cs.changes[0].needs_review and cs.changes[0].unconfirmed


def test_category_reading_differences_are_unconfirmed_but_real_repurposing_is_not():
    rows = list(BASE)
    rows[2] = ("duty_free", "WHSmith", "h", 0.55, 0.30)  # same group: no change at all
    rows[8] = ("information", "Al Dar Lounge", "4", 0.80, 0.20)  # same name, other category
    rows[4] = ("escalator", None, None, 0.70, 0.50)  # unnamed icon read differently
    rows[1] = ("retail", "Boots", "37", 0.40, 0.35)  # restaurant became a named shop: a real change
    cs, _ = detect_changes(snap(BASE), snap(rows, 2))
    by_new = {(c.new.name, c.new.category.value): c for c in cs.changes}
    assert by_new[("Al Dar Lounge", "information")].unconfirmed
    assert by_new[(None, "escalator")].unconfirmed
    assert not by_new[("Boots", "retail")].unconfirmed and len(cs.changes) == 3


def test_only_facility_relations_are_compared():
    old = snap(BASE, relations=[Relation(subject="Map", predicate="is_titled", object="T1 Departures"),
                                Relation(subject="Terminal", predicate="has_car_park", object="Car park K")])
    new = snap(BASE, 2, relations=[Relation(subject="Map", predicate="is_titled", object="TERMINAL 1")])
    cs, _ = detect_changes(old, new)
    assert cs.changes == []


def test_adjudicator_marks_rebrand_as_renamed():
    rows = [r if r[1] != "WHSmith" else ("retail", "WH Smith Books", "h", 0.55, 0.30) for r in BASE]
    cs, _ = detect_changes(snap(BASE), snap(rows, 2), adjudicator=lambda o, n: True)
    assert types(cs) == ["renamed"]
