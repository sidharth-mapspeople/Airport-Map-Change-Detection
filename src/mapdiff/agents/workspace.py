"""Page-level tools the understanding agent uses: zoomed views with a coordinate grid,
exact text lookup (vector PDFs), colour sampling, and recording findings."""

from __future__ import annotations

import io
import itertools
import threading
from typing import Any, get_args

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from mapdiff.agents.loop import LoopResult, Tool, ToolInputError, array_tool, run_tool_loop
from mapdiff.ingest import Page, TextSpan
from mapdiff.llm.base import VLM, ImagePart, Part, TextPart, ToolSpec
from mapdiff.models import FULL_PAGE, BBox, Entity, LegendEntry, Panel, PanelKind, Point, Relation, Snapshot
from mapdiff.names import name_similarity
from mapdiff.taxonomy import RELATION_PREDICATES, Category, group_of

_GRID_STEPS = [0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25]
_MIN_VIEW = 0.01


def _clamp_box(x0: float, y0: float, x1: float, y1: float) -> BBox:
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 < _MIN_VIEW or y1 - y0 < _MIN_VIEW:
        raise ToolInputError(f"Region too small (min {_MIN_VIEW} on each side)")
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _box_arg(a: dict[str, Any]) -> BBox:
    return _clamp_box(a["x0"], a["y0"], a["x1"], a["y1"])


def legend_examples(page: Page, which: str = "This map's") -> list[Part]:
    """The map's own legend icons as a labelled picture, for agents to compare symbols against (empty if none)."""
    if not page.legend_sheet:
        return []
    return [TextPart(f"{which} own legend icons, each labelled with its meaning. Icons that have the same shape mean "
                     "the same thing, whatever their colour:"), ImagePart(page.legend_sheet)]


def crop_page(image: Image.Image, box: BBox) -> Image.Image:
    """The part of a page image inside ``box`` (page coordinates)."""
    W, H = image.size
    return image.crop((int(box.x0 * W), int(box.y0 * H), int(box.x1 * W), int(box.y1 * H)))


def render_view(
    image: Image.Image, box: BBox, out_px: int, core: BBox | None = None, grid: bool = True, marker: Point | None = None
) -> bytes:
    """Crop ``box`` (page coords), scale its long side to ``out_px`` and overlay a labelled grid.
    ``core`` draws a red rectangle, ``marker`` a red circle around a claimed position."""
    crop = crop_page(image, box)
    cw, ch = crop.size
    scale = min(out_px / max(cw, ch), 3.0)  # never upscale more than 3x
    view = crop.resize((max(1, round(cw * scale)), max(1, round(ch * scale))), Image.Resampling.LANCZOS)
    vw, vh = view.size

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (x - box.x0) / box.width * vw, (y - box.y0) / box.height * vh

    overlay = Image.new("RGBA", view.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if grid:
        span = max(box.width, box.height)
        step = next((s for s in _GRID_STEPS if span / s <= 10), _GRID_STEPS[-1])
        font = ImageFont.load_default(size=max(11, min(vw, vh) // 70))
        decimals = 3 if step < 0.01 else 2 if step < 0.1 else 1
        for vertical, lo, hi in ((True, box.x0, box.x1), (False, box.y0, box.y1)):
            for i in itertools.count(int(lo / step) + 1):
                v = i * step
                if v >= hi:
                    break
                px, py = to_px(v, box.y0) if vertical else to_px(box.x0, v)
                line = [(px, 0), (px, vh)] if vertical else [(0, py), (vw, py)]
                draw.line(line, fill=(255, 0, 255, 90), width=1)
                draw.text((px + 2, 2) if vertical else (2, py + 2), f"{v:.{decimals}f}", fill=(200, 0, 200, 255),
                          font=font)
    if core is not None:
        (cx0, cy0), (cx1, cy1) = to_px(core.x0, core.y0), to_px(core.x1, core.y1)
        draw.rectangle([cx0, cy0, cx1, cy1], outline=(255, 0, 0, 230), width=3)
    if marker is not None:
        mx, my = to_px(marker.x, marker.y)
        r = max(12, min(vw, vh) // 14)
        draw.ellipse([mx - r, my - r, mx + r, my + r], outline=(255, 0, 0, 230), width=4)
    out = Image.alpha_composite(view.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def marked_crop(image: Image.Image, at: Point, half: float, out_px: int) -> bytes:
    """Close-up around ``at`` with the spot circled, for reviewers."""
    return render_view(image, BBox.around(at, half), out_px, grid=False, marker=at)


def entity_row(e: Entity) -> str:
    """'id | category | name | label | x, y | confidence' — the inventory format agents read and edit."""
    return (f"{e.id} | {e.category.value} | {e.name or '-'} | {e.label or '-'} | "
            f"{e.anchor.x:.3f}, {e.anchor.y:.3f} | {e.confidence:.2f}")


def legend_lines(legend: list[LegendEntry]) -> str:
    return "\n".join(f"- [{l.kind}] {l.symbol} → {l.meaning}" + (f" (category: {l.category.value})" if l.category else "")
                     for l in legend)


def spans_in(spans: list[TextSpan], box: BBox) -> list[TextSpan]:
    return [s for s in spans if box.contains(s.bbox.center)]


def format_spans(spans: list[TextSpan], limit: int = 300) -> str:
    rows = [
        f"({s.bbox.center.x:.3f}, {s.bbox.center.y:.3f}) {s.text}"
        for s in sorted(spans, key=lambda s: (round(s.bbox.y0, 2), s.bbox.x0))[:limit]
    ]
    more = f"\n… {len(spans) - limit} more lines" if len(spans) > limit else ""
    return "\n".join(rows) + more


# ---------- inputs accepted from the model ----------


class PanelIn(BaseModel):
    id: str
    kind: str
    title: str | None = None
    level: str | None = None
    bbox: list[float] = Field(min_length=4, max_length=4)


class LegendIn(BaseModel):
    kind: str
    symbol: str
    meaning: str
    category: Category | None = None


class RelationIn(BaseModel):
    subject: str
    predicate: str
    object: str
    evidence: str = ""
    confidence: float = 0.9


class EntityIn(BaseModel):
    category: Category
    name: str | None = None
    label: str | None = None
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    bbox: list[float] | None = Field(None, min_length=4, max_length=4)
    panel: str | None = None
    evidence: str = ""
    confidence: float = Field(0.9, ge=0, le=1)


class EntityUpdate(BaseModel):
    id: str
    category: Category | None = None
    name: str | None = None
    label: str | None = None
    x: float | None = Field(None, ge=0, le=1)
    y: float | None = Field(None, ge=0, le=1)
    confidence: float | None = Field(None, ge=0, le=1)
    note: str = ""


# ---------- shared, thread-safe extraction state ----------

DUPLICATE_RADIUS = 0.008  # same kind and name closer than this (page units) is the same physical item


def same_item(e: Entity, category: Category, name: str | None, anchor: Point) -> bool:
    return (group_of(e.category) == group_of(category) and e.anchor.dist(anchor) < DUPLICATE_RADIUS
            and name_similarity(e.name, name) >= 0.9)



class ExtractionState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.panels: list[Panel] = []
        self.legend: list[LegendEntry] = []
        self.relations: list[Relation] = []
        self.entities: dict[str, Entity] = {}
        self.meta: dict[str, Any] = {"notes": []}
        self.touched: set[str] = set()  # ids added or edited since the last reset (for re-scoring)
        self._ids = itertools.count(1)

    def new_entity_id(self, page: int) -> str:
        return f"p{page}-e{next(self._ids):04d}"

    def entities_on(self, page: int) -> list[Entity]:
        return [e for e in self.entities.values() if e.page == page]

    def panels_on(self, page: int, maps_only: bool = False) -> list[Panel]:
        return [p for p in self.panels if p.page == page and (p.is_map or not maps_only)]

    def legend_on(self, page: int) -> list[LegendEntry]:
        return [l for l in self.legend if l.page == page]

    def sorted_entities(self) -> list[Entity]:
        return sorted(self.entities.values(), key=lambda e: e.id)

    def duplicate_of(self, page: int, category: Category, name: str | None, anchor: Point) -> Entity | None:
        """An already recorded entity that is the same physical item. Call with the lock held."""
        return next((k for k in self.entities_on(page) if same_item(k, category, name, anchor)), None)

    def dedupe(self, page: int) -> None:
        """Merge entities that are the same physical item (e.g. one moved onto another); keep the most confident."""
        with self.lock:
            kept: list[Entity] = []
            for e in sorted(self.entities_on(page), key=lambda e: -e.confidence):
                if any(same_item(k, e.category, e.name, e.anchor) for k in kept):
                    self.entities.pop(e.id, None)
                else:
                    kept.append(e)

    @classmethod
    def from_snapshot(cls, snap: Snapshot) -> ExtractionState:
        """Reopen a finished snapshot for editing (correction agent)."""
        st = cls()
        st.panels, st.legend, st.relations = list(snap.panels), list(snap.legend), list(snap.relations)
        st.entities = {e.id: e.model_copy(deep=True) for e in snap.entities}
        last = max((int(e.id.rsplit("-e", 1)[-1]) for e in snap.entities if "-e" in e.id), default=0)
        st._ids = itertools.count(last + 1)
        return st


class PageWorkspace:
    """Binds a page and the shared state, and builds the tools for each phase."""

    def __init__(self, page: Page, state: ExtractionState, view_px: int = 1568):
        self.page = page
        self.state = state
        self.view_px = view_px

    # ----- perception -----

    def view(self, box: BBox, core: BBox | None = None) -> list[Part]:
        png = render_view(self.page.image, box, self.view_px, core=core)
        note = f"View of page region x {box.x0:.3f}–{box.x1:.3f}, y {box.y0:.3f}–{box.y1:.3f} (grid labels are page coordinates)."
        return [ImagePart(png), TextPart(note)]

    def overview(self) -> list[Part]:
        return self.view(FULL_PAGE)

    def panel_of(self, p: Point) -> Panel | None:
        hits = [pn for pn in self.state.panels_on(self.page.index, maps_only=True) if pn.bbox.contains(p)]
        return min(hits, key=lambda pn: pn.bbox.width * pn.bbox.height) if hits else None

    # ----- tool builders -----

    def perception_tools(self) -> list[Tool]:
        region = {
            "type": "object",
            "properties": {k: {"type": "number", "description": "page coordinate 0..1"} for k in ("x0", "y0", "x1", "y1")},
            "required": ["x0", "y0", "x1", "y1"],
        }
        tools = [
            Tool(
                ToolSpec(
                    "view_region",
                    "Zoom into a rectangular page region and get it as an image with a labelled page-coordinate grid. "
                    "Use small regions (e.g. 0.05–0.2 wide) to read small text and icons.",
                    region,
                ),
                lambda a: self.view(_box_arg(a)),
            ),
            Tool(
                ToolSpec(
                    "sample_colors",
                    "Dominant fill colours (hex, share of pixels) in a region. Useful to match an area to a colour legend.",
                    region,
                ),
                lambda a: self._colors(_box_arg(a)),
            ),
        ]
        if self.page.has_text_layer:
            tools += [
                Tool(
                    ToolSpec(
                        "get_text",
                        "Exact text lines from the PDF's text layer inside a region, with their centre coordinates.",
                        region,
                    ),
                    lambda a: format_spans(spans_in(self.page.spans, _box_arg(a)))
                    or "No text in this region.",
                ),
                Tool(
                    ToolSpec(
                        "find_text",
                        "Search the PDF's text layer (case-insensitive substring) and get matching lines with coordinates.",
                        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                    ),
                    lambda a: format_spans([s for s in self.page.spans if a["query"].casefold() in s.text.casefold()], 60)
                    or "No matches.",
                ),
            ]
        return tools

    def layout_tools(self) -> list[Tool]:
        def record_panels(a: dict[str, Any]) -> str:
            panels = [PanelIn.model_validate(p) for p in a["panels"]]
            with self.state.lock:
                for p in panels:
                    if p.kind not in get_args(PanelKind):
                        raise ToolInputError(f"Invalid panel kind {p.kind!r}")
                    self.state.panels.append(
                        Panel(id=p.id, page=self.page.index, kind=p.kind, title=p.title, level=p.level,
                              bbox=_clamp_box(*p.bbox))
                    )
            return f"Recorded {len(panels)} panels."

        def record_legend(a: dict[str, Any]) -> str:
            entries = [LegendIn.model_validate(e) for e in a["entries"]]
            with self.state.lock:
                self.state.legend += [LegendEntry(page=self.page.index, **e.model_dump()) for e in entries]
            return f"Recorded {len(entries)} legend entries."

        def finish_layout(a: dict[str, Any]) -> str:
            with self.state.lock:
                self.state.meta.setdefault("title", a.get("title"))
                self.state.meta.setdefault("map_date", a.get("map_date"))
                if a.get("notes"):
                    self.state.meta["notes"].append(a["notes"])
            return "Layout recorded."

        panel_item = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "short unique id, e.g. 'main', 'satellite_lower', 'legend_services'"},
                "kind": {"type": "string", "enum": ["map", "inset_map", "legend", "table", "title", "decoration", "other"]},
                "title": {"type": ["string", "null"]},
                "level": {"type": ["string", "null"], "description": "floor/level the map shows"},
                "bbox": {"type": "array", "items": {"type": "number"}, "description": "[x0, y0, x1, y1] page coords"},
            },
            "required": ["id", "kind", "bbox"],
        }
        legend_item = {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["marker_code", "color", "icon", "line", "pattern", "other"]},
                "symbol": {"type": "string", "description": "the code or a short visual description"},
                "meaning": {"type": "string", "description": "exact text the legend gives"},
                "category": {"type": ["string", "null"], "enum": [c.value for c in Category] + [None]},
            },
            "required": ["kind", "symbol", "meaning"],
        }
        return [
            array_tool("record_panels", "Record page panels (maps, insets, legends, tables, ...).", "panels",
                       panel_item, record_panels),
            array_tool("record_legend", "Record legend entries (call several times for long legends).", "entries",
                       legend_item, record_legend),
            self._relations_tool(),
            Tool(ToolSpec("finish_layout", "Finish the layout & legend task.",
                          {"type": "object", "properties": {"title": {"type": ["string", "null"]},
                                                            "map_date": {"type": ["string", "null"]},
                                                            "notes": {"type": ["string", "null"]}},
                           "required": []}), finish_layout, ends_run=True),
        ]

    def _relations_tool(self) -> Tool:
        def record_relations(a: dict[str, Any]) -> str:
            rels = [RelationIn.model_validate(r) for r in a["relations"]]
            kept = [r for r in rels if r.predicate in RELATION_PREDICATES]
            with self.state.lock:
                self.state.relations += [Relation(page=self.page.index, **r.model_dump()) for r in kept]
            skipped = len(rels) - len(kept)
            return f"Recorded {len(kept)} relations." + (
                f" Skipped {skipped} with other predicates (allowed: {', '.join(RELATION_PREDICATES)})." if skipped else "")

        item = {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "e.g. 'Emirates'"},
                "predicate": {"type": "string", "enum": list(RELATION_PREDICATES),
                              "description": "; ".join(f"{p}: {d}" for p, d in RELATION_PREDICATES.items())},
                "object": {"type": "string", "description": "e.g. 'Check-in 14'"},
                "evidence": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["subject", "predicate", "object"],
        }
        return array_tool("record_relations", "Record who-uses-which-facility facts, e.g. rows of an airline → "
                          "check-in desk table. Only the listed predicates; titles, legends and labels are not relations.",
                          "relations", item, record_relations)

    def entity_tools(self, core: BBox | None = None, allow_remove: bool = False, allow_update: bool = False) -> list[Tool]:
        def add_entities(a: dict[str, Any]) -> str:
            items = [EntityIn.model_validate(e) for e in a["entities"]]
            added, skipped, duplicates = [], [], []
            with self.state.lock:
                for e in items:
                    anchor = Point(x=e.x, y=e.y)
                    what = e.name or e.label or e.category.value
                    if core is not None and not core.contains(anchor, margin=0.002):
                        skipped.append(what)
                        continue
                    dup = self.state.duplicate_of(self.page.index, e.category, e.name, anchor)
                    if dup is not None:
                        duplicates.append(f"{what} = {dup.id}")
                        continue
                    panel = self.panel_of(anchor)
                    eid = self.state.new_entity_id(self.page.index)
                    self.state.entities[eid] = Entity(
                        id=eid, page=self.page.index, panel=e.panel or (panel.id if panel else None),
                        level=panel.level if panel else None, category=e.category, name=e.name, label=e.label,
                        anchor=anchor, bbox=_clamp_box(*e.bbox) if e.bbox else None,
                        evidence=e.evidence, confidence=e.confidence, extractor_confidence=e.confidence,
                    )
                    self.state.touched.add(eid)
                    added.append(eid)
            msg = f"Added {len(added)} entities ({', '.join(added)})."
            if skipped:
                msg += f" Skipped {len(skipped)} outside the core box (another tile covers them): {', '.join(skipped)}."
            if duplicates:
                msg += (f" Not added, already recorded (use update_entities to change them): {', '.join(duplicates)}.")
            return msg

        item = {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": [c.value for c in Category]},
                "name": {"type": ["string", "null"], "description": "specific identity; null for generic facilities"},
                "label": {"type": ["string", "null"], "description": "code/marker printed on the map, if any"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "bbox": {"type": ["array", "null"], "items": {"type": "number"}, "description": "[x0,y0,x1,y1] for areas"},
                "panel": {"type": ["string", "null"]},
                "evidence": {"type": "string", "description": "how you know, e.g. 'marker 8 → legend Restaurants & Cafes'"},
                "confidence": {"type": "number"},
            },
            "required": ["category", "x", "y", "evidence", "confidence"],
        }
        tools = [array_tool("add_entities", "Record places/facilities on the map (batch many per call).", "entities",
                            item, add_entities)]
        if allow_remove:
            def remove_entities(a: dict[str, Any]) -> str:
                with self.state.lock:
                    gone = [i for i in a["ids"] if self.state.entities.pop(i, None) is not None]
                return f"Removed {len(gone)}: {', '.join(gone)}"

            tools.append(array_tool("remove_entities", "Delete wrong or duplicate entities by id.", "ids",
                                    {"type": "string"}, remove_entities))
        if allow_update:
            tools.append(self._update_tool())
        return tools

    def _update_tool(self) -> Tool:
        def update_entities(a: dict[str, Any]) -> str:
            items = [EntityUpdate.model_validate(u) for u in a["updates"]]
            done, missing = [], []
            with self.state.lock:
                for u in items:
                    e = self.state.entities.get(u.id)
                    if e is None:
                        missing.append(u.id)
                        continue
                    changes = u.model_dump(include=u.model_fields_set - {"id", "x", "y", "note"})
                    if {"x", "y"} & u.model_fields_set:
                        changes["anchor"] = Point(x=u.x if u.x is not None else e.anchor.x,
                                                  y=u.y if u.y is not None else e.anchor.y)
                    if u.note:
                        changes["evidence"] = f"{e.evidence} | corrected: {u.note}".strip(" |")
                    self.state.entities[u.id] = e.model_copy(update=changes | {"verification": None})
                    self.state.touched.add(u.id)
                    done.append(u.id)
            msg = f"Updated {len(done)}: {', '.join(done)}."
            return msg + (f" Unknown ids: {', '.join(missing)}." if missing else "")

        item = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "category": {"type": "string", "enum": [c.value for c in Category]},
                "name": {"type": ["string", "null"]},
                "label": {"type": ["string", "null"]},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "confidence": {"type": "number"},
                "note": {"type": "string", "description": "what you corrected and why"},
            },
            "required": ["id", "note"],
        }
        return array_tool("update_entities", "Correct fields of existing entities. Only the fields you pass change.",
                          "updates", item, update_entities)

    def toolset(self, *, layout: bool = False, entities: bool = False, core: BBox | None = None,
                allow_remove: bool = False, allow_update: bool = False,
                finish: tuple[str, str] | None = None) -> list[Tool]:
        """Perception tools plus the requested recording tools and an optional (name, description) finish tool."""
        tools = self.perception_tools()
        if layout:
            tools += self.layout_tools()
        if entities:
            tools += self.entity_tools(core, allow_remove, allow_update)
        if finish:
            tools.append(self.finish_tool(*finish))
        return tools

    def run(self, vlm: VLM, system: str, task: str, tools: list[Tool], max_turns: int, label: str,
            view: list[Part] | None = None) -> LoopResult:
        """One agent run on this page: ``view`` (default: the whole page), the legend icons, then the task text."""
        first = [*(view if view is not None else self.overview()), *legend_examples(self.page), TextPart(task)]
        return run_tool_loop(vlm, system, first, tools, max_turns, label)

    @staticmethod
    def finish_tool(name: str, description: str) -> Tool:
        return Tool(ToolSpec(name, description,
                             {"type": "object", "properties": {"notes": {"type": ["string", "null"]}}, "required": []}),
                    lambda a: "Done.", ends_run=True)

    def _colors(self, box: BBox) -> str:
        crop = crop_page(self.page.image, box)
        crop.thumbnail((200, 200))
        q = crop.quantize(colors=8)
        pal = q.getpalette()
        counts = sorted(q.getcolors(), reverse=True)
        total = sum(c for c, _ in counts)
        return "\n".join(
            f"#{pal[3 * i]:02x}{pal[3 * i + 1]:02x}{pal[3 * i + 2]:02x}  {c / total:.0%}" for c, i in counts[:6]
        )
