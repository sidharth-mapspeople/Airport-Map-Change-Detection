"""End-to-end run:

understand → mapping confidence (≥ threshold, else correct ≤ N times) → detect changes vs previous
version → change confidence → store → report.

Each agent can use its own model (:class:`~mapdiff.settings.AgentModels`); passing a single VLM uses it for all.
The usage meter records the tokens, cost and time of each agent are stored on the snapshot (understanding, mapping
confidence, correction) and on the changeset (change confidence).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from mapdiff.agents import lessons, symbols
from mapdiff.agents.change_confidence import ChangeConfidenceConfig, score_changes
from mapdiff.agents.change_detection import Adjudicator, DiffConfig, detect_changes
from mapdiff.agents.mapping_confidence import QualityConfig, ensure_mapping_quality
from mapdiff.agents.reporting import render_markdown
from mapdiff.agents.understanding import UnderstandingConfig, understand_file
from mapdiff.icons import IconLibrary, library_dir
from mapdiff.ingest import Page, load_document, sha256_file
from mapdiff.llm.base import VLM
from mapdiff.models import ChangeSet, Snapshot
from mapdiff.settings import AgentModels
from mapdiff.store import MapStore

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    understanding: UnderstandingConfig = field(default_factory=UnderstandingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    diff: DiffConfig = field(default_factory=DiffConfig)
    change_confidence: ChangeConfidenceConfig = field(default_factory=ChangeConfidenceConfig)
    score_changes: bool = True
    icon_library: Path | None = field(default_factory=library_dir)  # learned icons; None = neither use nor learn


@dataclass
class RunResult:
    snapshot: Snapshot | None
    changeset: ChangeSet | None
    report_path: Path | None
    skipped_identical: bool = False


def _as_models(models: VLM | AgentModels) -> AgentModels:
    return models if isinstance(models, AgentModels) else AgentModels.single(models)


def _stored_pages(store: MapStore, map_id: str, version: int, cfg: PipelineConfig) -> list[Page]:
    return load_document(store.source_path(map_id, version), cfg.understanding.render_px)


def process_file(
    path: str | Path,
    map_id: str,
    store: MapStore,
    models: VLM | AgentModels,
    cfg: PipelineConfig | None = None,
    adjudicator: Adjudicator | None = None,
    force: bool = False,
) -> RunResult:
    cfg = cfg or PipelineConfig()
    m = _as_models(models)
    path = Path(path)
    prev = store.latest(map_id)
    if prev and not force and prev.source_sha256 == sha256_file(path):
        log.info("File is byte-identical to version %d; nothing to do.", prev.version)
        return RunResult(prev, None, None, skipped_identical=True)

    start = m.meter.mark()
    pages = load_document(path, cfg.understanding.render_px)
    library = IconLibrary(cfg.icon_library) if cfg.icon_library and cfg.understanding.detect_symbols else None
    snap = understand_file(path, map_id, m.understanding, cfg.understanding, pages=pages, previous=prev, library=library)
    ensure_mapping_quality(snap, pages, m.mapping_confidence, cfg.quality, correction_vlm=m.correction)
    snap.llm = m.describe()
    mapped = m.meter.mark()
    snap.usage = m.meter.summary(start, mapped)
    if prev is None:
        for e in snap.entities:
            e.lineage_id = e.id
        snap.lessons = lessons.carry([], snap.lessons)
        store.save(snap, path)
        _learn_icons(snap, pages, library)
        return RunResult(snap, None, None)

    cs, matched = detect_changes(prev, snap, cfg.diff, adjudicator)
    held = lessons.flag_repeats(cs, prev.lessons)
    old_lineage = {e.id: e.lineage_id or e.id for e in prev.entities}
    for e in snap.entities:
        e.lineage_id = old_lineage.get(matched.get(e.id, "")) or f"v{prev.version + 1}-{e.id}"
    if cfg.score_changes:
        score_changes(cs, prev, snap, _stored_pages(store, map_id, prev.version, cfg), pages, m.change_confidence, cfg.change_confidence, cfg.diff)
    cs.usage = m.meter.summary(mapped)
    snap.lessons = lessons.carry(prev.lessons, snap.lessons + lessons.false_changes(cs, snap.source_file))
    log.info("lessons: %d repeat claims held for confirmation; %d lessons kept for %s", held, len(snap.lessons), map_id)
    store.save(snap, path)
    _learn_icons(snap, pages, library)
    cs.to_version = snap.version
    report = store.save_changes(cs, render_markdown(cs, prev, snap))
    return RunResult(snap, cs, report)


def _learn_icons(snap: Snapshot, pages: list[Page], library: IconLibrary | None) -> None:
    if library is not None:
        new = symbols.learn(snap, pages, library)
        log.info("icon library: %d new icons learned from %s v%d (%d known)", new, snap.map_id, snap.version,
                 len(library.icons))


def rediff(map_id: str, store: MapStore, from_v: int, to_v: int, models: VLM | AgentModels | None = None,
           cfg: PipelineConfig | None = None, adjudicator: Adjudicator | None = None) -> tuple[ChangeSet, Path]:
    """Re-run change detection between two stored versions; change scoring only if ``models`` are given."""
    cfg = cfg or PipelineConfig()
    old, new = store.load(map_id, from_v), store.load(map_id, to_v)
    cs, _ = detect_changes(old, new, cfg.diff, adjudicator)
    lessons.flag_repeats(cs, new.lessons)
    if models is not None and cfg.score_changes:
        m = _as_models(models)
        start = m.meter.mark()
        score_changes(cs, old, new, _stored_pages(store, map_id, from_v, cfg), _stored_pages(store, map_id, to_v, cfg),
                      m.change_confidence, cfg.change_confidence, cfg.diff)
        cs.usage = m.meter.summary(start)
    return cs, store.save_changes(cs, render_markdown(cs, old, new))
