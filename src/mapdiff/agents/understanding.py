"""Agent 1 — understand a floorplan file and produce a :class:`Snapshot`.

Per page, these steps share one extraction state:
  A. layout & legend  — (model) find panels, decode this map's own legend into a codebook, read tables
  B. symbols          — (code, see symbols.py) cut out the legend icons, find them on the map, read the map's text
                        (text layer or OCR) and resolve printed legend codes; records candidate items
  C. tile scan        — (model) one bounded run per map tile (parallel): confirm or fix the candidates in the tile and
                        add everything they missed, with the legend icons shown as examples
  D. verifier         — (model) deterministic coverage checks, then a run that investigates and fixes the gaps
When the map has a previous version, its corrected inventory is shown as a checklist so unchanged items are read
the same way twice (differences between two readings would otherwise show up as false changes).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapdiff.agents import lessons, symbols
from mapdiff.agents.workspace import (
    ExtractionState,
    PageWorkspace,
    crop_page,
    entity_row,
    format_spans,
    legend_lines,
    spans_in,
)
from mapdiff.icons import IconLibrary
from mapdiff.ingest import Page, TextSpan, load_document, sha256_file
from mapdiff.llm.base import VLM
from mapdiff.models import FULL_PAGE, BBox, Entity, Panel, Snapshot
from mapdiff.names import norm_code
from mapdiff.parallel import parallel_map
from mapdiff.taxonomy import GENERIC, Category, group_of

log = logging.getLogger(__name__)


@dataclass
class UnderstandingConfig:
    render_px: int = 6000  # long side of the working page raster
    view_px: int = 1568  # long side of images sent to the model
    tile_px: int = 1400  # core tile size in working-raster pixels
    tile_margin: float = 0.12  # context margin around each tile core (fraction of core)
    workers: int = 4
    max_turns_layout: int = 40
    max_turns_tile: int = 25
    max_turns_verify: int = 30
    verify: bool = True
    use_previous: bool = True  # show the previous version's inventory as a checklist
    detect_symbols: bool = True  # find legend icons and legend codes with code before the model reads the map
    previous_radius: float = 0.02  # previous item counts as found if a same-group item is this close


# Conventions shared by the extractor, the reviewer and the correction agent, so that the reviewer judges by the
# same rules the extractor followed. Each rule answers a mistake the reviewer used to flag in the first pass.
CONVENTIONS = f"""Coordinates: every position is normalised to the whole page — x from 0 (left) to 1 (right), y from 0 \
(top) to 1 (bottom). Images carry a magenta grid labelled in page coordinates; use it to read positions.

What is an item: a place or facility a traveller can use or go to — a shop, cafe, gate, check-in island, desk, \
toilet, lift, entrance, bus stop, car park, lounge, a named zone. NOT items: walking arrows, route or line labels, \
street names, building or level titles, the legend itself, decorative photos, and text that only labels a line or \
an area outline. One physical item is one entry: an icon and the text printed beside it are the same item.

Position: anchor an item at the centre of its own icon or marker (for areas, the centre of the area). Zoom in before \
recording it — an anchor on empty floor next to the icon is a mistake.

Meaning vs presentation: every map has its own colours, icon styles, numbers and letters. That is presentation and \
may change between versions without anything real changing. What matters is the category, the specific name and the \
position. Resolve codes through the legend: if marker "8" is listed as "Culto Café" under "Restaurants & Cafes", \
record category=cafe, name="Culto Café", label="8". label is only for such legend codes — a number printed nearby \
that is not the item's own legend code (a gate number, a level, a count) is not its label.

Naming:
- name = the specific identity: brand ("Starbucks"), airline ("Emirates"), lounge ("Al Dar Lounge"), gate \
("Gate A40"), check-in island ("Check-in 5"), car park ("Car park K"), named zone ("Zone A"). Use the map's exact \
spelling; when the page has a text layer, copy spelling from it.
- When the legend gives an icon a specific meaning ("Special Assistance Desk", "Bus drop-off and pick-up"), use that \
meaning as the name.
- Generic facilities with no specific meaning in the legend (toilets, lifts, stairs, escalators, unbranded ATMs) get \
name=null.
- Never invent names. If a code cannot be resolved, keep the label, set name=null and lower the confidence.

Category: pick the closest category below; use 'other' only when none fits. The category follows from what the \
legend says the icon means, not from the icon's colour.

Categories: {", ".join(c.value for c in Category)}"""

SYSTEM_PROMPT = f"""You are an expert at reading airport floorplans (terminal maps, wayfinding maps, forecourt/parking maps). \
You build a precise, machine-readable inventory of what is where, so later versions of the same map can be compared.

{CONVENTIONS}

Coloured areas without a name still count when they denote a place: record them with their category, name=null and \
a bbox.

Confidence: ≥0.9 read clearly; 0.6–0.9 partly inferred; <0.6 a guess.

Be systematic and zoom in (view_region) whenever text or symbols are small — never guess what you could zoom to read."""


def understand_file(
    path: str | Path, map_id: str, vlm: VLM, cfg: UnderstandingConfig | None = None, pages: list[Page] | None = None,
    previous: Snapshot | None = None, library: IconLibrary | None = None,
) -> Snapshot:
    cfg = cfg or UnderstandingConfig()
    path = Path(path)
    pages = pages if pages is not None else load_document(path, cfg.render_px)
    state = ExtractionState()
    for page in pages:
        _understand_page(page, state, vlm, cfg, previous if cfg.use_previous else None, library, map_id)

    return Snapshot(
        map_id=map_id,
        source_file=path.name,
        source_sha256=sha256_file(path),
        title=state.meta.get("title"),
        map_date=state.meta.get("map_date"),
        pages=len(pages),
        panels=state.panels,
        legend=state.legend,
        entities=state.sorted_entities(),
        relations=state.relations,
        notes=state.meta["notes"],
    )


def _understand_page(page: Page, state: ExtractionState, vlm: VLM, cfg: UnderstandingConfig,
                     previous: Snapshot | None = None, library: IconLibrary | None = None,
                     map_id: str | None = None) -> None:
    ws = PageWorkspace(page, state, cfg.view_px)
    label = f"page{page.index}"
    before = [e for e in previous.entities if e.page == page.index] if previous else []

    # A. layout & legend
    text_note = (
        f"This page has an embedded text layer ({len(page.spans)} lines): use get_text / find_text for exact spelling."
        if page.has_text_layer
        else "This page is a raster image without a text layer: read text visually, zooming in as needed."
    )
    layout_task = LAYOUT_TASK.format(page=page.index, text_note=text_note)
    prev_rel = [r for r in previous.relations if r.page == page.index] if previous else []
    if prev_rel:
        layout_task += PREVIOUS_RELATIONS + "\n".join(f"- {r.subject} | {r.predicate} | {r.object}" for r in prev_rel)
    ws.run(vlm, SYSTEM_PROMPT, layout_task, ws.toolset(layout=True), cfg.max_turns_layout, f"{label}/layout")
    if not state.panels_on(page.index, maps_only=True):
        log.warning("[%s] no map panel recorded; treating the whole page as one map", label)
        state.panels.append(Panel(id=f"page{page.index}", page=page.index, kind="map", bbox=FULL_PAGE))

    # B. symbols (no model calls)
    text: list[TextSpan] = page.spans
    detected: set[str] = set()
    if cfg.detect_symbols:
        found = symbols.detect(page, state, library, map_id)
        text, detected = found.text or page.spans, set(found.added)

    # C. tile scan
    codebook = _codebook_text(state, page.index)
    jobs = [(panel, core) for panel in state.panels_on(page.index, maps_only=True)
            for core in _tile_cores(panel.bbox, page, cfg.tile_px)]
    jobs = [(p, c) for p, c in jobs if not _is_blank(page, c)]
    log.info("[%s] scanning %d tiles", label, len(jobs))

    def scan(job: tuple[Panel, BBox]) -> None:
        panel, core = job
        view = core.expanded(cfg.tile_margin)
        task = TILE_TASK.format(panel=panel.id, level=panel.level or "unknown", x0=core.x0, x1=core.x1, y0=core.y0,
                                y1=core.y1, codebook=codebook)
        if text:
            source = "text layer" if page.has_text_layer else "read by OCR, may contain misreadings"
            task += f"\n\nText lines in this view ({source}; centre x, y):\n" + (format_spans(spans_in(text, view)) or "(none)")
        candidates = [e for e in state.entities_on(page.index) if e.id in detected and core.contains(e.anchor)]
        if candidates:
            task += DETECTED_ITEMS + "\n".join(entity_row(e) for e in candidates)
        nearby = [e for e in before if core.expanded(0.25).contains(e.anchor)]
        if nearby:
            task += PREVIOUS_ITEMS + "\n".join(_previous_row(e) for e in nearby)
        mistakes = lessons.near(previous.lessons, page.index, core.expanded(0.1)) if previous else []
        if mistakes:
            task += EARLIER_MISTAKES + lessons.rows(mistakes)
        tools = ws.toolset(entities=True, core=core, allow_update=bool(candidates), allow_remove=bool(candidates),
                           finish=("finish_tile", "Call when every item in the core box is recorded."))
        ws.run(vlm, SYSTEM_PROMPT, task, tools, cfg.max_turns_tile, f"{label}/tile@{core.x0:.2f},{core.y0:.2f}",
               view=ws.view(view, core=core))

    parallel_map(scan, jobs, cfg.workers)
    state.dedupe(page.index)
    resolve_codes(state, page.index)

    # D. verify
    if cfg.verify:
        report = coverage_report(state, page, text)
        if before:
            report += "\n" + previous_report(state, page.index, before, cfg.previous_radius)
        task = VERIFY_TASK.format(page=page.index, codebook=codebook, inventory=_inventory_text(state, page.index),
                                  report=report)
        tools = ws.toolset(entities=True, allow_remove=True,
                           finish=("finish_verification", "Call when the flagged issues are resolved."))
        ws.run(vlm, SYSTEM_PROMPT, task, tools, cfg.max_turns_verify, f"{label}/verify")
        state.dedupe(page.index)
        resolve_codes(state, page.index)


LAYOUT_TASK = """Task 1 of 3 — layout and legend of page {page}. Attached: the whole page. {text_note}

1. Panels: record every distinct region with record_panels — the main floor map(s), inset maps that show other levels \
or satellites (kind inset_map, with their level), legends, tables, title blocks, and decorative photos or locator \
key-maps that only show where the terminal is (kind decoration). Give tight bboxes in page coordinates.
2. Legend: decode it completely with record_legend — every numbered/lettered marker with its exact meaning (and the \
category heading it is listed under), every colour fill, every icon, every line style. Zoom in so small icons and \
text are read exactly. Long lists: call record_legend several times.
3. Who-uses-which-facility tables with record_relations — e.g. every row of an airline → check-in desk table \
(predicate 'checks_in_at', object 'Check-in <number>'). Record every row. Titles, legend contents, insets and \
labels are not relations.
4. Call finish_layout with the map title, the printed date if any, and short notes."""

TILE_TASK = """Task 2 of 3 — inventory one tile of panel '{panel}' (level: {level}).
The image shows the tile; the red rectangle is the CORE box x {x0:.3f}–{x1:.3f}, y {y0:.3f}–{y1:.3f}. \
Record every item whose anchor lies inside the core box; anything in the margin belongs to a neighbouring tile and is \
shown only for context.

Record each legend marker/code, icon, labelled place, coloured block that denotes a place, gate, check-in \
island/counter, security lane group, passport control, entrance, lift, stair and escalator. Resolve each through the \
codebook. The tile is shown at reduced resolution, so work region by region: zoom in with view_region on each part \
of the core box, identify every symbol there, and read each item's position from the zoomed view before recording \
it. Use add_entities in batches, then call finish_tile.

Codebook (this map's legend):
{codebook}"""

DETECTED_ITEMS = """

Already recorded in this core box by automatic detection — a legend icon matched by shape, or a printed legend code \
(id | category | name | label | x, y | confidence). Zoom in and check each: wrong → update_entities or \
remove_entities; right → leave it. Do not add them again; add only what is missing.
"""

PREVIOUS_ITEMS = """

Previous version of this map — items recorded near this tile (category | name | label | x, y). Use it as a \
checklist, not as an answer: record only what this version shows. For an item that is still there, use the same \
category and name spelling as before. Where this version shows something different (another name, another kind of \
place, nothing at all), record what you see now. Positions may be shifted if the layout changed.
"""

EARLIER_MISTAKES = """

Mistakes made on earlier versions of this map in this area, found by review and corrected. Do not repeat them; \
check these spots with extra care:
"""

PREVIOUS_RELATIONS = """

Previous version of this map's table (subject | predicate | object). Spell subjects and objects the same way when the \
row is unchanged; record what this version shows.
"""

VERIFY_TASK = """Task 3 of 3 — verify the inventory of page {page}. Attached: the whole page, the codebook, the current \
inventory and automatic checks. Investigate every flagged issue by zooming in. Fix with add_entities (missing items) \
and remove_entities (duplicates or mistakes). Do not re-add items that are already in the inventory. Then call \
finish_verification.

Codebook:
{codebook}

Inventory (id | category | name | label | x, y | confidence):
{inventory}

Automatic checks:
{report}"""


def _tile_cores(box: BBox, page: Page, tile_px: int) -> list[BBox]:
    W, H = page.image.size
    nx = max(1, math.ceil(box.width * W / tile_px))
    ny = max(1, math.ceil(box.height * H / tile_px))
    dx, dy = box.width / nx, box.height / ny
    return [BBox(x0=box.x0 + i * dx, y0=box.y0 + j * dy, x1=box.x0 + (i + 1) * dx, y1=box.y0 + (j + 1) * dy)
            for j in range(ny) for i in range(nx)]


def _is_blank(page: Page, box: BBox, min_std: float = 4.0) -> bool:
    crop = crop_page(page.image, box)
    crop.thumbnail((256, 256))
    return float(np.asarray(crop.convert("L"), dtype=np.float32).std()) < min_std


def _codebook_text(state: ExtractionState, page: int) -> str:
    prow = [f"- panel '{p.id}' ({p.kind}{', level ' + p.level if p.level else ''}) "
            f"bbox [{p.bbox.x0:.3f}, {p.bbox.y0:.3f}, {p.bbox.x1:.3f}, {p.bbox.y1:.3f}]" for p in state.panels_on(page)]
    return "Panels:\n" + "\n".join(prow) + "\n\nLegend:\n" + (legend_lines(state.legend_on(page)) or "(no legend found)")


def _inventory_text(state: ExtractionState, page: int) -> str:
    ents = sorted(state.entities_on(page), key=lambda e: (round(e.anchor.y, 2), e.anchor.x))
    return "\n".join(entity_row(e) for e in ents) or "(empty)"


def coverage_report(state: ExtractionState, page: Page, text: list[TextSpan] | None = None) -> str:
    """Deterministic checks that point the verifier at likely gaps and duplicates.

    ``text`` is the map's text (default: the PDF text layer; pass OCR lines for images).
    """
    ents = state.entities_on(page.index)
    text = page.spans if text is None else text
    lines: list[str] = []

    used_labels = {norm_code(e.label) for e in ents}
    missing = [l for l in state.legend_on(page.index) if l.kind == "marker_code" and norm_code(l.symbol) not in used_labels]
    if missing:
        lines.append("Legend marker codes never found on the map (check whether they appear; some legitimately may not):")
        lines += [f"  - {l.symbol} → {l.meaning}" for l in missing[:80]]

    maps = state.panels_on(page.index, maps_only=True)
    if text:
        unexplained = []
        for s in text:
            c = s.bbox.center
            if not any(p.bbox.contains(c) for p in maps):
                continue
            near = any(abs(e.anchor.x - c.x) < 0.02 and abs(e.anchor.y - c.y) < 0.02 for e in ents) or any(
                e.bbox and e.bbox.contains(c) for e in ents)
            if not near:
                unexplained.append(s)
        if unexplained:
            lines.append(f"Text on the map with no entity nearby ({len(unexplained)}; may be street names or "
                         "decoration — record the ones that denote places/facilities):")
            lines.append(format_spans(unexplained, 120))

    low = [e for e in ents if e.confidence < 0.6]
    if low:
        lines.append("Low-confidence entities to double-check: " + ", ".join(e.id for e in low[:60]))
    unresolved = [e for e in ents if e.label and not e.name and e.category in (Category.OTHER,)]
    if unresolved:
        lines.append("Entities with a code but unresolved meaning: " + ", ".join(e.id for e in unresolved[:60]))
    return "\n".join(lines) or "No issues detected."


def _previous_row(e: Entity) -> str:
    return f"{e.category.value} | {e.name or '-'} | {e.label or '-'} | {e.anchor.x:.3f}, {e.anchor.y:.3f}"


def previous_report(state: ExtractionState, page: int, before: list[Entity], radius: float) -> str:
    """Differences from the previous version for the verifier to confirm: each is either a real change or a misreading."""
    now = state.entities_on(page)

    def found(e: Entity, among: list[Entity]) -> bool:
        return any(group_of(o.category) == group_of(e.category) and o.anchor.dist(e.anchor) < radius for o in among)

    gone = [e for e in before if not found(e, now)]
    new = [e for e in now if not found(e, before)]
    lines = []
    if gone:
        lines.append("In the previous version but not found now (zoom in: still there → add it as before; really "
                     "gone → leave it out):")
        lines += [f"  - {_previous_row(e)}" for e in gone[:80]]
    if new:
        lines.append("Found now but not in the previous version (zoom in: confirm it is really there and not a "
                     "duplicate or misread item):")
        lines += [f"  - {entity_row(e)}" for e in new[:80]]
    return "\n".join(lines) or "Same items as the previous version."


def resolve_codes(state: ExtractionState, page: int) -> None:
    """Fill what the legend already says: the name of a coded item, and the category of an 'other' item."""
    codes = {norm_code(l.symbol): l for l in state.legend_on(page) if l.kind == "marker_code"}
    with state.lock:
        for e in state.entities_on(page):
            l = codes.get(norm_code(e.label)) if e.label else None
            if l is None:
                continue
            if e.category == Category.OTHER and l.category:
                e.category = l.category
            if not e.name and e.category not in GENERIC:
                e.name = l.meaning
