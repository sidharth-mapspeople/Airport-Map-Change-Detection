"""Correction agent — fixes the entities the mapping-confidence agent flagged.

It receives each low-scoring entity with the reviewer's issue and suggested fix, plus the
coverage checks, zooms in to look for itself, and updates / removes / adds entities.
Runs per page in bounded chunks; the quality loop decides how often it runs.
"""

from __future__ import annotations

from typing import Callable

from mapdiff.agents.understanding import SYSTEM_PROMPT, _codebook_text, coverage_report
from mapdiff.agents.workspace import ExtractionState, PageWorkspace, entity_row
from mapdiff.ingest import Page
from mapdiff.llm.base import VLM
from mapdiff.models import Entity, Snapshot

CHUNK = 40  # low-scoring entities per correction run
MAX_TURNS = 40

CORRECTION_TASK = """Correction task for page {page}. An independent reviewer doubted these extracted items. \
For each one, zoom in with view_region (and get_text where available), decide what is really there, \
and fix it:
- wrong category / name / label / position → update_entities (pass only the fields that change, plus a note)
- not a real item, or a duplicate → remove_entities
- something missing nearby → add_entities
- the original was right after all → update_entities with only the id and a note giving the evidence you saw \
(it will be re-reviewed with your note)
Do not guess: if the map genuinely does not show a name, keep name null. Call finish_correction when done.

Items to check (id | category | name | label | x, y | score | reviewer issue | suggested fix):
{items}

Coverage checks (possibly missed items):
{coverage}

Codebook:
{codebook}"""


def correct_mapping(snap: Snapshot, pages: list[Page], vlm: VLM, flagged: Callable[[Entity], bool]) -> set[str]:
    """Run the correction agent on the ``flagged`` entities; mutate ``snap``; return touched entity ids."""
    state = ExtractionState.from_snapshot(snap)
    for page in pages:
        low = sorted((e for e in state.entities_on(page.index) if flagged(e)), key=lambda e: e.confidence)
        ws = PageWorkspace(page, state)
        coverage = coverage_report(state, page)
        if not low and coverage == "No issues detected.":
            continue
        tools = ws.toolset(entities=True, allow_remove=True, allow_update=True,
                           finish=("finish_correction", "Call when every listed item has been checked."))
        for i in range(0, max(len(low), 1), CHUNK):
            task = CORRECTION_TASK.format(page=page.index, items=_rows(low[i:i + CHUNK]),
                                          coverage=coverage, codebook=_codebook_text(state, page.index))
            ws.run(vlm, SYSTEM_PROMPT, task, tools, MAX_TURNS, f"page{page.index}/correction")
        state.dedupe(page.index)  # a moved item can land on another

    snap.entities = state.sorted_entities()
    return state.touched & set(state.entities)


def _rows(ents: list[Entity]) -> str:
    def row(e: Entity) -> str:
        v = e.verification
        return f"{entity_row(e)} | {(v.issue if v else '') or '-'} | {(v.suggested_fix if v else '') or '-'}"
    return "\n".join(row(e) for e in ents) or "(none)"
