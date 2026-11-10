"""Mapping-confidence → correction loop, and change-confidence scoring."""

from conftest import ScriptedVLM, blank_page, item_count

from mapdiff.agents.change_confidence import ChangeConfidenceConfig, score_changes
from mapdiff.agents.change_detection import detect_changes
from mapdiff.agents.mapping_confidence import QualityConfig, ensure_mapping_quality, score_entities
from mapdiff.agents.reporting import render_markdown
from mapdiff.models import Entity, LegendEntry, Point, Relation, Snapshot
from mapdiff.taxonomy import Category

CFG = QualityConfig(workers=1)


def snap(*ents, legend=(), relations=(), version=1):
    return Snapshot(map_id="t", version=version, source_file="f", source_sha256="x", entities=list(ents),
                    legend=list(legend), relations=list(relations))


def ent(id, cat, name, x, y, label=None, conf=0.9):
    return Entity(id=id, category=Category(cat), name=name, label=label, anchor=Point(x=x, y=y), confidence=conf)


def reviewer(judge):
    """Entity reviewer: judge(claim_text) -> (verdict, confidence)."""
    def handle(parts, names, done):
        claims = [p.text for p in parts if getattr(p, "text", "").startswith("Item ")]
        out = []
        for k, c in enumerate(claims, 1):
            verdict, conf = judge(c)
            out.append({"item": k, "verdict": verdict, "confidence": conf, "issue": "wrong name",
                        "suggested_fix": "name is Starbucks"})
        return [("report_verdicts", {"verdicts": out})]
    return handle


def test_low_confidence_triggers_one_correction_then_passes():
    s = snap(ent("p0-e0001", "cafe", "Starbuck", 0.5, 0.5), ent("p0-e0002", "toilet", None, 0.2, 0.2))

    def correction(parts, names, done):
        assert {"update_entities", "remove_entities", "add_entities"} <= names
        task = next(p.text for p in parts if getattr(p, "text", "").startswith("Correction task"))
        assert "p0-e0001" in task and "name is Starbucks" in task and "p0-e0002" not in task
        if done:
            return [("finish_correction", {})]
        return [("update_entities", {"updates": [{"id": "p0-e0001", "name": "Starbucks", "note": "zoomed: Starbucks"}]})]

    vlm = ScriptedVLM(
        entity_review=reviewer(lambda c: ("incorrect", 0.9) if "'Starbuck'" in c else ("correct", 0.95)),
        correction=correction,
    )
    q = ensure_mapping_quality(s, [blank_page()], vlm, CFG)
    assert q.passed and q.corrections_used == 1
    assert [r.round for r in q.rounds] == [0, 1]
    assert q.rounds[0].mapping_confidence < 0.9 <= q.mapping_confidence
    fixed = next(e for e in s.entities if e.id == "p0-e0001")
    assert fixed.name == "Starbucks" and fixed.verification.verdict == "correct"
    assert vlm.calls["entity_review"] == 2  # second review only re-checks the corrected entity


def test_corrections_capped_at_one_then_continue():
    s = snap(ent("p0-e0001", "cafe", "Mystery", 0.5, 0.5))
    vlm = ScriptedVLM(
        entity_review=reviewer(lambda c: ("incorrect", 0.8)),
        correction=lambda parts, names, done: [("finish_correction", {})] if done else
        [("update_entities", {"updates": [{"id": "p0-e0001", "note": "still unsure"}]})],
    )
    q = ensure_mapping_quality(s, [blank_page()], vlm, CFG)
    assert q.corrections_used == 1 and not q.passed
    assert vlm.calls["correction"] == 2  # 1 round × (act + finish)
    assert len(s.entities) == 1  # detections are kept and sent on


def test_only_doubted_entities_are_corrected():
    sure = ent("p0-e0001", "cafe", "Starbucks", 0.5, 0.5)
    hedged = ent("p0-e0002", "toilet", None, 0.2, 0.2)
    wrong = ent("p0-e0003", "lift", None, 0.7, 0.7)
    s = snap(sure, hedged, wrong)
    verdicts = {"Starbucks": ("correct", 0.85), "toilet": ("correct", 0.6), "lift": ("incorrect", 0.9),
                "stairs": ("correct", 0.95)}
    seen = []

    def correction(parts, names, done):
        if done:
            return [("finish_correction", {})]
        seen.extend(i for i in ("p0-e0001", "p0-e0002", "p0-e0003") if any(i in getattr(p, "text", "") for p in parts))
        return [("update_entities", {"updates": [{"id": "p0-e0003", "category": "stairs"}]})]

    vlm = ScriptedVLM(entity_review=reviewer(lambda c: next(v for k, v in verdicts.items() if k in c)),
                      correction=correction)
    ensure_mapping_quality(s, [blank_page()], vlm, CFG)
    assert seen == ["p0-e0002", "p0-e0003"]  # "correct" at 0.85 is left alone


def test_unconfirmed_change_needs_a_confident_real_verdict():
    old = snap(ent("a", "lounge", "Al Dar Lounge", 0.2, 0.3, conf=1.0), ent("b", "toilet", None, 0.6, 0.3, conf=1.0))
    new = snap(ent("a", "information", "Al Dar Lounge", 0.2, 0.3, conf=1.0), ent("b", "lift", None, 0.6, 0.3, conf=1.0),
               version=2)
    cs, _ = detect_changes(old, new)
    assert all(c.unconfirmed for c in cs.changes) and len(cs.changes) == 2

    def change_review(parts, names, done):
        changes = [p.text for p in parts if getattr(p, "text", "").startswith("Change ")]
        return [("report_verdicts", {"verdicts": [
            {"change": k, "verdict": "real" if "lift" in c else "unsure", "confidence": 0.8, "explanation": "-"}
            for k, c in enumerate(changes, 1)]})]

    score_changes(cs, old, new, [blank_page()], [blank_page()], ScriptedVLM(change_review=change_review),
                  ChangeConfidenceConfig(workers=1))
    kept = [c for c in cs.changes if not c.rejected]
    assert [c.new.category.value for c in kept] == ["lift"]
    assert "not confirmed" in next(c for c in cs.changes if c.rejected).verification_note


def test_text_layer_and_legend_confirm_without_model_review():
    page = blank_page(spans=[("Culto Café", 0.5, 0.5)])
    s = snap(ent("p0-e0001", "cafe", "Culto Café", 0.5, 0.5, label="8"),
             legend=[LegendEntry(kind="marker_code", symbol="8", meaning="Culto Café")])
    vlm = ScriptedVLM()  # no handlers: any model call would fail
    score_entities(s, [page], vlm, CFG)
    assert s.entities[0].verification.method == "text_layer" and s.entities[0].confidence == 0.95


def test_legend_conflict_caps_score():
    s = snap(ent("p0-e0001", "cafe", "Starbucks", 0.5, 0.5, label="8"),
             legend=[LegendEntry(kind="marker_code", symbol="8", meaning="Culto Café")])
    vlm = ScriptedVLM(entity_review=reviewer(lambda c: ("correct", 0.99)))
    score_entities(s, [blank_page()], vlm, CFG)
    assert s.entities[0].confidence == 0.5 and "legend" in s.entities[0].verification.issue


def test_change_confidence_scores_and_rejects():
    old = snap(ent("a", "cafe", "Culto Café", 0.2, 0.3, conf=1.0), ent("b", "retail", "WHSmith", 0.6, 0.3, conf=1.0),
               relations=[Relation(subject="Emirates", predicate="checks_in_at", object="Check-in 14")])
    new = snap(ent("a", "cafe", "Starbucks", 0.2, 0.3, conf=1.0), ent("b", "retail", "Boots", 0.6, 0.3, conf=1.0),
               relations=[Relation(subject="Emirates", predicate="checks_in_at", object="Check-in 5")], version=2)
    cs, _ = detect_changes(old, new)
    assert len(cs.changes) == 3

    def change_review(parts, names, done):
        changes = [p.text for p in parts if getattr(p, "text", "").startswith("Change ")]
        out = []
        for k, c in enumerate(changes, 1):
            fake = "Boots" in c  # pretend the reviewer sees WHSmith in both crops
            out.append({"change": k, "verdict": "not_real" if fake else "real", "confidence": 0.9,
                        "explanation": "both crops show WHSmith" if fake else "confirmed"})
        if "relation" in changes[0]:
            assert item_count(parts, "OLD version") >= 1 and item_count(parts, "NEW version") >= 1
        return [("report_verdicts", {"verdicts": out})]

    score_changes(cs, old, new, [blank_page()], [blank_page()], ScriptedVLM(change_review=change_review),
                  ChangeConfidenceConfig(workers=1))
    by_summary = {c.summary: c for c in cs.changes}
    cafe = next(c for s, c in by_summary.items() if "Starbucks" in s)
    shop = next(c for s, c in by_summary.items() if "Boots" in s)
    desk = next(c for s, c in by_summary.items() if "Emirates" in s)
    assert cafe.confidence == round(0.6 * 0.9 + 0.4 * 1.0, 3) and not cafe.rejected and not cafe.needs_review
    assert shop.rejected and shop.confidence < 0.5
    assert desk.confidence_breakdown["visual"] == 0.9

    report = render_markdown(cs, old, new)
    assert "Rejected" in report and "both crops show WHSmith" in report
    assert "**2**" in report  # rejected change not counted


def test_a_change_made_in_the_legend_is_confirmed_by_the_legends():
    """'Rideshare (Uber)' → 'Rideshare (Ola)' changed only in the legend: the map spot looks the same in both crops."""
    old = snap(ent("a", "rideshare", "Rideshare (Uber)", 0.2, 0.3, label="9", conf=1.0),
               legend=[LegendEntry(kind="marker_code", symbol="9", meaning="Rideshare (Uber)")])
    new = snap(ent("a", "rideshare", "Rideshare (Ola)", 0.2, 0.3, label="9", conf=1.0),
               legend=[LegendEntry(kind="marker_code", symbol="9", meaning="Rideshare (Ola)")], version=2)
    cs, _ = detect_changes(old, new)

    def change_review(parts, names, done):
        return [("report_verdicts", {"verdicts": [{"change": 1, "verdict": "not_real", "confidence": 0.9,
                                                   "explanation": "both crops show the same marker 9"}]})]

    score_changes(cs, old, new, [blank_page()], [blank_page()], ScriptedVLM(change_review=change_review),
                  ChangeConfidenceConfig(workers=1))
    (c,) = cs.changes
    assert not c.rejected and c.confidence >= 0.85 and "legend entry changed" in c.verification_note
