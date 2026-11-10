"""Symbol detection agent — deterministic, no model calls: finds what the map's own legend says to look for.

Per page, right after the legend is read:
  1. cut each legend icon out of the legend; fill a missing category from the learned icon library;
  2. find every instance of those icons (plus icons learned from earlier versions of the same map) on the map
     panels, by template matching with a shape check, so colour and restyling do not matter;
  3. read the text on the map (PDF text layer, or OCR on images) and resolve printed legend codes
     ("d" → Starbucks, "5" → Check-in 5).
What it finds is recorded as candidate entities with their evidence. The model then confirms, corrects and completes
them, and the mapping-confidence reviewer checks every one, so nothing here is taken as final.
After a version is stored, :func:`learn` adds its legend icons to the library for later maps.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

from PIL import Image, ImageDraw

from mapdiff.agents.workspace import ExtractionState
from mapdiff.icons import IconLibrary, LegendIcon, SymbolHit, find_symbols, legend_icons
from mapdiff.ingest import Page, TextSpan
from mapdiff.models import Entity, LegendEntry, Snapshot
from mapdiff.ocr import page_text
from mapdiff.taxonomy import GENERIC, Category

log = logging.getLogger(__name__)

LIBRARY_HINT = 0.8  # library similarity needed to give an uncategorised legend icon a category
VERIFIED_SCORE = 0.9  # an icon counts as verified once an entity of its kind was reviewed "correct" at this score


@dataclass
class PageSymbols:
    icons: list[LegendIcon]
    text: list[TextSpan]  # text lines on the map panels (text layer or OCR)
    added: list[str]  # ids of the candidate entities recorded


def detect(page: Page, state: ExtractionState, library: IconLibrary | None = None,
           map_id: str | None = None) -> PageSymbols:
    """Record candidate entities for ``page`` from its legend icons and legend codes; set ``page.legend_sheet``."""
    legend = state.legend_on(page.index)
    icons = legend_icons(page, legend, state.panels)
    if library is not None:
        _fill_categories(icons, library)
    page.legend_sheet = legend_sheet(icons) if icons else None

    maps = state.panels_on(page.index, maps_only=True)
    others = [p.bbox.expanded(0.05) for p in state.panels_on(page.index) if not p.is_map]  # tables, legends
    templates = icons + (_earlier_icons(library, map_id, icons) if library is not None and map_id else [])
    hits = find_symbols(page, templates, [p.bbox for p in maps]) if templates else []
    everything = [s for p in maps for s in page_text(page, p.bbox)]  # incl. tables drawn inside a map's box
    text = [s for s in everything if not any(o.contains(s.bbox.center) for o in others)]

    added: list[str] = []
    with state.lock:
        for hit in hits:
            if not any(o.contains(hit.anchor) for o in others):
                added += _record(state, page, _from_hit(hit))
        for span, entry, code, ranged in _coded(text, everything, legend):
            added += _record(state, page, _from_code(span, entry, code, ranged))
    log.info("[page%d/symbols] %d legend icons, %d icon matches, %d text lines → %d candidates",
             page.index, len(icons), len(hits), len(text), len(added))
    return PageSymbols(icons, text, added)


def ensure_legend_sheet(page: Page, snap: Snapshot) -> None:
    """Give a page loaded from the store its legend icon sheet (computed from the stored legend; no model calls)."""
    if page.legend_sheet is None:
        icons = legend_icons(page, [l for l in snap.legend if l.page == page.index], snap.panels)
        page.legend_sheet = legend_sheet(icons) if icons else None


def legend_sheet(icons: list[LegendIcon], cell: int = 96, cols: int = 6) -> bytes:
    """One image of the legend's icons, each labelled with its meaning: few-shot examples for the agents."""
    rows = (len(icons) + cols - 1) // cols
    label_h = 28
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, icon in enumerate(icons):
        x, y = (n % cols) * cell, (n // cols) * (cell + label_h)
        img = icon.image.copy()
        img.thumbnail((cell - 12, cell - 12))
        sheet.paste(img, (x + (cell - img.width) // 2, y + 6))
        draw.text((x + 4, y + cell), _short(icon.meaning, 16), fill="black")
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def learn(snap: Snapshot, pages: list[Page], library: IconLibrary) -> int:
    """Add this version's legend icons to the library; returns how many icons are new."""
    before = len(library.icons)
    source = f"{snap.map_id} v{snap.version}"
    for page in pages:
        for icon in legend_icons(page, [l for l in snap.legend if l.page == page.index], snap.panels):
            known = library.add(icon, source)
            if known is None:  # rejected by a person before
                continue
            known.verified |= any(e.category == icon.category and e.verification is not None
                                  and e.verification.verdict == "correct" and e.verification.score >= VERIFIED_SCORE
                                  for e in snap.entities if e.page == page.index)
    library.save()
    return len(library.icons) - before


# ---------------- helpers ----------------


def _fill_categories(icons: list[LegendIcon], library: IconLibrary) -> None:
    for icon in icons:
        if icon.category in (None, Category.OTHER):
            best = library.match(icon.silhouette, 1)
            if best and best[0][1] >= LIBRARY_HINT and best[0][0].category:
                icon.category = Category(best[0][0].category)


def _earlier_icons(library: IconLibrary, map_id: str, current: list[LegendIcon]) -> list[LegendIcon]:
    """Icons learned from earlier versions of this map that this version's legend does not show."""
    have = {i.meaning.lower() for i in current}
    out = []
    for known in library.icons:
        if known.meaning.lower() not in have and any(s.startswith(f"{map_id} v") for s in known.sources):
            img = library.image(known)
            if img is not None:
                cat = Category(known.category) if known.category else None
                out.append(LegendIcon(known.meaning, cat, None, img))
    return out


def _from_hit(hit: SymbolHit) -> dict:
    cat = hit.category or Category.OTHER
    return dict(category=cat, name=None if cat in GENERIC else hit.meaning, label=None, anchor=hit.anchor,
                evidence=f"legend icon '{hit.meaning}' found here (match {hit.score:.2f})",
                confidence=round(0.5 + 0.4 * hit.score, 2))


def _coded(text: list[TextSpan], everything: list[TextSpan], legend: list[LegendEntry]):
    """(text line, legend entry, code, part of a numbered range) for text that is exactly a legend marker code."""
    codes: dict[str, tuple[LegendEntry, bool]] = {}
    for entry in (l for l in legend if l.kind == "marker_code"):
        for code, ranged in _codes(entry.symbol):
            codes.setdefault(code, (entry, ranged))
    def is_code(sp: TextSpan) -> bool:
        return sp.text.strip().strip(".").lower() in codes

    for span in text:
        if is_code(span) and _stands_alone(span, [t for t in everything if not is_code(t)]):
            key = span.text.strip().strip(".").lower()
            entry, ranged = codes[key]
            yield span, entry, (span.text.strip() if ranged else key), ranged


def _stands_alone(span: TextSpan, words: list[TextSpan], reach: float = 12) -> bool:
    """False for a code printed in a row of text (a table: 'Emirates ... 5'); a marker on the map stands alone."""
    h = span.bbox.height
    for w in words:
        same_row = min(span.bbox.y1, w.bbox.y1) - max(span.bbox.y0, w.bbox.y0) > 0.5 * min(h, w.bbox.height)
        gap = max(w.bbox.x0 - span.bbox.x1, span.bbox.x0 - w.bbox.x1)
        if same_row and gap < reach * h:
            return False
    return True


def _codes(symbol: str) -> list[tuple[str, bool]]:
    """Codes a legend symbol stands for: 'd' → [d], '1-14 (number boxes)' → [1 … 14] (marked as a range)."""
    m = re.match(r"\s*([A-Za-z0-9]{1,4})\s*(?:[-–]\s*([A-Za-z0-9]{1,4}))?", symbol)
    if not m:
        return []
    a, b = m.group(1), m.group(2)
    if b and a.isdigit() and b.isdigit() and int(a) < int(b) <= int(a) + 200:
        return [(str(n), True) for n in range(int(a), int(b) + 1)]
    return [(a.lower(), False)]


def _from_code(span: TextSpan, entry: LegendEntry, code: str, ranged: bool) -> dict:
    cat = entry.category or Category.OTHER
    name = None if cat in GENERIC else (f"{entry.meaning} {code}" if ranged else entry.meaning)
    return dict(category=cat, name=name, label=code, anchor=span.bbox.center,
                evidence=f"printed code '{code}' → legend '{entry.meaning}'", confidence=0.8)


def _record(state: ExtractionState, page: Page, item: dict) -> list[str]:
    """Add a candidate unless the same item is already recorded. Call with the lock held."""
    if state.duplicate_of(page.index, item["category"], item["name"], item["anchor"]) is not None:
        return []
    panel = next((p for p in state.panels_on(page.index, maps_only=True) if p.bbox.contains(item["anchor"])), None)
    eid = state.new_entity_id(page.index)
    state.entities[eid] = Entity(id=eid, page=page.index, panel=panel.id if panel else None,
                                 level=panel.level if panel else None, extractor_confidence=item["confidence"], **item)
    state.touched.add(eid)
    return [eid]


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"
