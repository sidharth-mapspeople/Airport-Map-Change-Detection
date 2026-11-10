"""Shared helpers that several agents and scripts rely on."""

from conftest import ScriptedVLM, blank_page

from mapdiff.agents.workspace import ExtractionState, PageWorkspace, entity_row
from mapdiff.evaluation import CaseScore, eval_map_id, latest_results, load_manifest, pct
from mapdiff.models import BBox, Entity, Panel, Point
from mapdiff.names import norm_code
from mapdiff.parallel import parallel_map
from mapdiff.taxonomy import Category


def test_bbox_helpers_stay_on_the_page():
    box = BBox.around(Point(x=0.01, y=0.99), 0.05)
    assert (box.x0, box.y1) == (0.0, 1.0) and round(box.width, 6) == round(box.height, 6) == 0.1
    grown = BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2).expanded(0.5)
    assert (round(grown.x0, 6), round(grown.y1, 6)) == (0.0, 0.25)
    assert BBox(x0=0, y0=0, x1=1, y1=1).center == Point(x=0.5, y=0.5)


def test_state_filters_and_rows():
    st = ExtractionState()
    st.panels = [Panel(id="m", page=0, kind="map", bbox=BBox(x0=0, y0=0, x1=1, y1=1)),
                 Panel(id="l", page=0, kind="legend", bbox=BBox(x0=0, y0=0, x1=1, y1=1)),
                 Panel(id="m2", page=1, kind="inset_map", bbox=BBox(x0=0, y0=0, x1=1, y1=1))]
    st.entities = {i: Entity(id=i, page=p, category=Category.TOILET, anchor=Point(x=0.5, y=0.5))
                   for i, p in (("b", 0), ("a", 0), ("c", 1))}
    assert [p.id for p in st.panels_on(0, maps_only=True)] == ["m"] and len(st.panels_on(0)) == 2
    assert [e.id for e in st.sorted_entities()] == ["a", "b", "c"] and len(st.entities_on(0)) == 2
    assert entity_row(st.entities["a"]) == "a | toilet | - | - | 0.500, 0.500 | 1.00"
    assert norm_code(" A1 ") == norm_code("a1")


def test_toolset_composes_tool_groups():
    ws = PageWorkspace(blank_page(), ExtractionState())
    names = [t.spec.name for t in ws.toolset(entities=True, allow_remove=True, finish=("finish_x", "done"))]
    assert names[:2] == ["view_region", "sample_colors"] and names[-1] == "finish_x"
    assert "remove_entities" in names and "update_entities" not in names
    assert "record_legend" in [t.spec.name for t in ws.toolset(layout=True)]


def test_combined_scores_and_eval_helpers():
    a = CaseScore("a", 2, true_positives=[({}, "x")], false_positives=["fp"])
    b = CaseScore("b", 2, true_positives=[({}, "y"), ({}, "z")])
    both = CaseScore.combine([a, b])
    assert both.expected == 4 and both.recall == 0.75 and both.precision == 0.75
    assert eval_map_id("Test_files/Terminal 1 map (1).pdf") == "eval-terminal-1-map-1"
    assert pct(None) == "-" and pct(0.5) == "50%"
    assert parallel_map(lambda x: x * 2, [1, 2, 3], workers=2) == [2, 4, 6]
    assert {c["kind"] for c in load_manifest()} == {"edit", "restyle"}


def test_latest_results_picks_newest_completed_run(tmp_path):
    for name in ("20261028-100000", "20261029-090000"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text("{}")
    (tmp_path / "20261029-120000").mkdir()  # still running: no summary yet
    assert latest_results(tmp_path).name == "20261029-090000"


def test_legend_codes_fill_names_and_previous_version_is_compared():
    from mapdiff.agents.understanding import previous_report, resolve_codes
    from mapdiff.models import LegendEntry

    st = ExtractionState()
    st.legend = [LegendEntry(kind="marker_code", symbol="8", meaning="Culto Café", category=Category.CAFE),
                 LegendEntry(kind="marker_code", symbol="T", meaning="Toilets", category=Category.TOILET)]
    st.entities = {
        "a": Entity(id="a", page=0, category=Category.OTHER, label="8", anchor=Point(x=0.2, y=0.2)),
        "b": Entity(id="b", page=0, category=Category.TOILET, label="T", anchor=Point(x=0.5, y=0.5)),
    }
    resolve_codes(st, 0)
    assert (st.entities["a"].category, st.entities["a"].name) == (Category.CAFE, "Culto Café")
    assert st.entities["b"].name is None  # generic facilities stay unnamed

    before = [Entity(id="x", page=0, category=Category.TOILET, anchor=Point(x=0.505, y=0.5)),
              Entity(id="y", page=0, category=Category.LIFT, anchor=Point(x=0.8, y=0.8))]
    report = previous_report(st, 0, before, 0.02)
    assert "not found now" in report and "lift" in report  # the lift may be gone, or missed
    assert "not in the previous version" in report and "Culto" in report  # the café is new


def test_same_item_cannot_be_recorded_twice():
    from mapdiff.agents.correction import correct_mapping
    from mapdiff.models import Snapshot

    desk = Entity(id="p0-e0001", page=0, category=Category.CHECK_IN, name="Check-in 1", anchor=Point(x=0.293, y=0.385),
                  confidence=0.95)
    snap = Snapshot(map_id="t", version=1, source_file="f", source_sha256="x", entities=[desk])
    again = {"category": "check_in", "name": "Check-in 1", "x": 0.292, "y": 0.385, "evidence": "box 1", "confidence": 0.9}
    other = {"category": "check_in", "name": "Check-in 2", "x": 0.317, "y": 0.396, "evidence": "box 2", "confidence": 0.9}
    def correction(parts, names, done):
        if done:
            return [("finish_correction", {})]
        return [("add_entities", {"entities": [again, other]}),
                ("update_entities", {"updates": [{"id": "p0-e0001", "note": "kept"}]})]

    vlm = ScriptedVLM(correction=correction)
    correct_mapping(snap, [blank_page()], vlm, lambda e: True)
    assert [e.name for e in snap.entities] == ["Check-in 1", "Check-in 2"]

    st = ExtractionState()  # an update that moves one item onto another is merged afterwards
    st.entities = {i: Entity(id=i, page=0, category=Category.LIFT, anchor=Point(x=0.5, y=0.5 + d), confidence=c)
                   for i, d, c in (("a", 0.0, 0.6), ("b", 0.003, 0.9))}
    st.dedupe(0)
    assert list(st.entities) == ["b"]
