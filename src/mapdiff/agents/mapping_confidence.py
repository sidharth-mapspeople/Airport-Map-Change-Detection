"""Mapping-confidence agent and the score → correct → re-score quality loop.

Self-reported confidence from the extractor is not trusted. Each entity is scored by:
  * an independent, sceptical VLM review of a close-up crop (claimed spot circled), and
  * deterministic evidence: the name printed nearby in the PDF text layer (supports),
    or a marker code whose legend meaning contradicts the name (caps the score).
The mapping confidence is the mean entity score. Below the threshold, the correction
agent fixes the entities the reviewer flagged (not "correct", or scored below ``fix_below``);
this repeats at most ``max_corrections`` times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mapdiff.agents import lessons
from mapdiff.agents.review import VerdictSchema, chunked, review_batches
from mapdiff.agents.understanding import CONVENTIONS
from mapdiff.agents.workspace import legend_examples, legend_lines, marked_crop
from mapdiff.ingest import Page
from mapdiff.llm.base import VLM, ImagePart, Part, TextPart
from mapdiff.models import Entity, MappingQuality, QualityRound, Snapshot, Verification
from mapdiff.names import name_similarity, norm_code

log = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    min_confidence: float = 0.9  # required mean entity score
    max_corrections: int = 1
    fix_below: float = 0.7  # entities judged "correct" but scored lower than this are corrected too
    batch_size: int = 8  # entities reviewed per VLM call
    crop_half: float = 0.03  # half-size of review crops (page units)
    crop_px: int = 640
    trust_text_layer: bool = True  # skip VLM review when PDF text + legend already confirm the entity
    workers: int = 4


REVIEW_SYSTEM = f"""You are a meticulous, sceptical QA reviewer of airport floorplan data. Another system extracted \
items from a map; your job is to catch its mistakes, not to agree with it. Judge it by the same conventions it was \
given:

{CONVENTIONS}

For each numbered item you get the claim and a close-up crop of the map; the red circle marks the claimed position. \
Decide independently:
- correct: there really is such an item at (or right next to) the circle, the category fits, and the name/label \
match what is printed (resolve codes with the legend codebook).
- incorrect: wrong category, wrong or misspelled name, wrong code resolution, nothing there, or a duplicate of \
something else. Say what is wrong in 'issue' and what it should be in 'suggested_fix'.
- unsure: the crop does not let you decide.
confidence is how sure you are of your verdict, on this scale:
- 0.95–1.0: the crop shows it plainly (for "correct": the icon or text is at the circle and matches the claim).
- 0.8–0.94: very likely, but one detail is hard to make out.
- 0.6–0.79: plausible, with real doubt. Below 0.6, answer unsure instead.
Use the top of the scale whenever the evidence is plain; do not hedge by habit. Report every item with \
report_verdicts."""


def mapping_confidence(snap: Snapshot) -> float:
    return sum(e.confidence for e in snap.entities) / len(snap.entities) if snap.entities else 0.0


def score_entities(snap: Snapshot, pages: list[Page], vlm: VLM, cfg: QualityConfig,
                   only_ids: set[str] | None = None) -> None:
    """Assess entities (all, or ``only_ids``) and set ``verification`` and ``confidence`` on them."""
    targets = [e for e in snap.entities if only_ids is None or e.id in only_ids]
    for e in targets:
        if e.extractor_confidence is None:
            e.extractor_confidence = e.confidence

    evidence = {e.id: _deterministic_evidence(e, pages[e.page], snap) for e in targets}
    needs_vlm = [e for e in targets if not (cfg.trust_text_layer and evidence[e.id] == "confirmed")]
    reviewed = {e.id for e in needs_vlm}
    for e in targets:
        if e.id not in reviewed:
            e.verification = Verification(score=0.95, verdict="correct", method="text_layer")

    batches = chunked(needs_vlm, cfg.batch_size)
    log.info("mapping-confidence: %d entities (%d by model review in %d batches)", len(targets), len(needs_vlm), len(batches))
    verdicts = review_batches(vlm, REVIEW_SYSTEM, batches, lambda b: _review_parts(b, pages, snap, cfg), _SCHEMA,
                              "mapping-confidence", cfg.workers)

    for e in needs_vlm:
        v = verdicts.get(id(e))
        score, _ = _SCHEMA.score(v)
        v = v or {"verdict": "unsure", "issue": "no verdict returned"}
        method = "vlm"
        if evidence[e.id] in ("confirmed", "text_supports") and v["verdict"] != "incorrect":
            score, method = max(score, 0.95), "vlm+text_layer"
        if evidence[e.id] == "legend_conflict":
            score = min(score, 0.5)
            v["issue"] = (v.get("issue") or "") + " [legend meaning of this code contradicts the name]"
        e.verification = Verification(score=round(score, 3), verdict=v["verdict"], method=method,
                                      issue=v.get("issue") or "", suggested_fix=v.get("suggested_fix") or "")
    for e in targets:
        e.confidence = e.verification.score


def _deterministic_evidence(e: Entity, page: Page, snap: Snapshot) -> str:
    """'confirmed' | 'text_supports' | 'legend_conflict' | 'none'."""
    legend_ok = None
    if e.label:
        codes = [l for l in snap.legend if l.page == e.page and l.kind == "marker_code"
                 and norm_code(l.symbol) == norm_code(e.label)]
        if codes and e.name:
            legend_ok = any(name_similarity(l.meaning, e.name) >= 0.85 for l in codes)
        if legend_ok is False:
            return "legend_conflict"
    if not (e.name and page.has_text_layer):
        return "none"
    near = [s for s in page.spans if s.bbox.center.dist(e.anchor) < 0.03]
    text_ok = any(name_similarity(s.text, e.name) >= 0.85 for s in near)
    if text_ok and legend_ok is not False:
        return "confirmed"
    if legend_ok:  # name not printed at the spot, but the code there resolves to it
        return "text_supports"
    return "none"


_SCHEMA = VerdictSchema("item", "correct", "incorrect",
                        {"issue": {"type": "string"}, "suggested_fix": {"type": "string"}})


def _review_parts(batch: list[Entity], pages: list[Page], snap: Snapshot, cfg: QualityConfig) -> list[Part]:
    legend = legend_lines([l for l in snap.legend if l.page == batch[0].page]) or "(none)"
    parts: list[Part] = [*legend_examples(pages[batch[0].page]), TextPart(f"Legend codebook:\n{legend}\n")]
    for k, e in enumerate(batch, 1):
        half = cfg.crop_half
        if e.bbox:  # areas: show the whole area
            half = max(half, e.bbox.width / 2 + 0.01, e.bbox.height / 2 + 0.01)
        claim = (f"Item {k}: category={e.category.value}, name={e.name!r}, label={e.label!r}"
                 + (f", level={e.level}" if e.level else "") + f". Extractor's evidence: {e.evidence or '-'}")
        parts += [TextPart(claim), ImagePart(marked_crop(pages[e.page].image, e.anchor, half, cfg.crop_px))]
    parts.append(TextPart(f"Review all {len(batch)} items, then call report_verdicts once."))
    return parts


def ensure_mapping_quality(snap: Snapshot, pages: list[Page], vlm: VLM, cfg: QualityConfig | None = None,
                           correction_vlm: VLM | None = None) -> MappingQuality:
    """Score; while below threshold and corrections remain, correct then re-score the touched entities.

    ``vlm`` scores; ``correction_vlm`` (default: ``vlm``) corrects.
    """
    from mapdiff.agents.correction import correct_mapping

    cfg = cfg or QualityConfig()
    correction_vlm = correction_vlm or vlm
    score_entities(snap, pages, vlm, cfg)
    rounds = [_round(snap, 0, cfg)]
    doubted = {e.id: e.model_copy(deep=True) for e in snap.entities if needs_fix(e, cfg)}
    used = 0
    while rounds[-1].mapping_confidence < cfg.min_confidence and used < cfg.max_corrections:
        used += 1
        log.info("mapping confidence %.3f < %.2f → correction %d/%d",
                 rounds[-1].mapping_confidence, cfg.min_confidence, used, cfg.max_corrections)
        touched = correct_mapping(snap, pages, correction_vlm, lambda e: needs_fix(e, cfg))
        score_entities(snap, pages, vlm, cfg, only_ids=touched)
        rounds.append(_round(snap, used, cfg, corrected=len(touched)))

    snap.lessons = lessons.misreads(doubted, snap.entities, snap.source_file) if used else []
    final = rounds[-1].mapping_confidence
    if final < cfg.min_confidence:
        log.warning("mapping confidence %.3f still below %.2f after %d corrections; continuing with current detections",
                    final, cfg.min_confidence, used)
    snap.quality = MappingQuality(threshold=cfg.min_confidence, mapping_confidence=final,
                                  passed=final >= cfg.min_confidence, corrections_used=used,
                                  max_corrections=cfg.max_corrections, rounds=rounds)
    return snap.quality


def needs_fix(e: Entity, cfg: QualityConfig) -> bool:
    """Whether the correction agent should look at ``e``: the reviewer doubted it or was not sure enough."""
    v = e.verification
    return v is None or v.verdict != "correct" or v.score < cfg.fix_below


def _round(snap: Snapshot, n: int, cfg: QualityConfig, corrected: int = 0) -> QualityRound:
    return QualityRound(round=n, mapping_confidence=round(mapping_confidence(snap), 4), entities=len(snap.entities),
                        low_confidence=sum(e.confidence < cfg.min_confidence for e in snap.entities),
                        corrected=corrected)
