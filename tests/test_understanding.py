"""Exercise ingest, tools and the full pipeline with a scripted (offline) VLM."""

from pathlib import Path

import pytest
from conftest import ScriptedVLM, item_count

from mapdiff.agents.mapping_confidence import QualityConfig
from mapdiff.agents.understanding import UnderstandingConfig, understand_file
from mapdiff.ingest import load_document
from mapdiff.pipeline import PipelineConfig, process_file
from mapdiff.store import MapStore

TEST_FILES = Path(__file__).resolve().parents[1] / "Test_files"
AUCKLAND = TEST_FILES / "auckland_Domestic-Terminal-Forecourt.pdf"
pytestmark = pytest.mark.skipif(not AUCKLAND.exists(), reason="Test_files not present")


def test_pdf_ingest_has_text_layer_and_render():
    (page,) = load_document(AUCKLAND, max_px=2000)
    assert max(page.image.size) == 2000
    assert any("Domestic Terminal" in s.text for s in page.spans)
    assert all(0 <= s.bbox.x0 <= s.bbox.x1 <= 1 for s in page.spans)


def test_raster_ingest():
    (page,) = load_document(TEST_FILES / "brussels_airport_departure.jpg", max_px=3000)
    assert max(page.image.size) == 3000 and not page.has_text_layer


def layout(parts, names, done):
    assert {"view_region", "get_text", "record_panels", "record_legend"} <= names
    if done:
        return [("finish_layout", {"title": "Domestic Terminal", "map_date": "April 2026"})]
    return [
        ("get_text", {"x0": 0, "y0": 0, "x1": 0.3, "y1": 0.3}),
        ("record_panels", {"panels": [
            {"id": "main", "kind": "map", "level": "ground", "bbox": [0.22, 0.15, 0.99, 0.95]},
            {"id": "legend", "kind": "legend", "bbox": [0.02, 0.18, 0.21, 0.92]}]}),
        ("record_legend", {"entries": [{"kind": "marker_code", "symbol": "8", "meaning": "Rental cars",
                                        "category": "car_rental"}]}),
        ("record_relations", {"relations": [
            {"subject": "Air New Zealand", "predicate": "checks_in_at", "object": "Check-in 3"},
            {"subject": "Domestic Terminal", "predicate": "has_car_park", "object": "Car park V"}]}),
    ]


def tile(parts, names, done):
    if done:
        return [("finish_tile", {})]
    # Every tile proposes the same item; only the tile whose core box contains it may keep it.
    return [("add_entities", {"entities": [{"category": "car_rental", "name": None, "label": "8", "x": 0.755,
                                            "y": 0.385, "evidence": "marker 8", "confidence": 0.95}]})]


def entity_review(parts, names, done):
    n = item_count(parts, "Item ")
    return [("report_verdicts", {"verdicts": [{"item": k, "verdict": "correct", "confidence": 0.95}
                                              for k in range(1, n + 1)]})]


SCRIPT = dict(layout=layout, tile=tile, entity_review=entity_review,
              verify=lambda p, n, d: [("finish_verification", {})])


def test_understanding_pipeline_with_scripted_vlm(tmp_path):
    vlm = ScriptedVLM(**SCRIPT)
    ucfg = UnderstandingConfig(render_px=2000, tile_px=700, workers=2)
    snap = understand_file(AUCKLAND, "akl-dom", vlm, ucfg)

    assert snap.title == "Domestic Terminal" and snap.map_date == "April 2026"
    assert [p.id for p in snap.panels] == ["main", "legend"]
    assert len(snap.entities) == 1  # proposed by every tile, kept once (core-box rule + dedupe)
    e = snap.entities[0]
    assert e.panel == "main" and e.level == "ground" and e.label == "8"
    assert [r.object for r in snap.relations] == ["Check-in 3"]  # only who-uses-which-facility facts are kept
    assert vlm.images_seen >= 3  # overview + tiles + verifier overview

    store = MapStore(tmp_path)
    cfg = PipelineConfig(understanding=ucfg, quality=QualityConfig(workers=2))
    first = process_file(AUCKLAND, "akl-dom", store, vlm, cfg)
    assert first.snapshot.version == 1
    assert first.snapshot.quality.passed and first.snapshot.quality.corrections_used == 0
    assert "correction" not in vlm.calls  # 0.95 ≥ 0.9, so no correction round
    again = process_file(AUCKLAND, "akl-dom", store, vlm, cfg)
    assert again.skipped_identical
    forced = process_file(AUCKLAND, "akl-dom", store, vlm, cfg, force=True)
    assert forced.snapshot.version == 2 and forced.changeset.changes == []
    assert forced.snapshot.entities[0].lineage_id == first.snapshot.entities[0].lineage_id
    assert "No changes detected" in forced.report_path.read_text()
