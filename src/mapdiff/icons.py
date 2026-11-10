"""Icons: cut them out of a map's legend, compare them by shape, learn them across maps, find them on the map.

* :func:`silhouette` reduces an icon to the shape of its glyph, so colour does not matter: a white toilet figure on a
  blue tile and a black one on yellow give the same silhouette. Tiles and circles around a glyph are removed.
* :func:`legend_icons` locates each legend icon next to its meaning text (PDF text layer or OCR).
* :class:`IconLibrary` keeps the icons learned from processed maps (``airport-icons/``), each with its meaning and
  category, and returns the most similar ones for a new icon.
* :func:`find_symbols` finds every instance of a map's own legend icons on that map by template matching.
All of it is deterministic and local: no model calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rapidfuzz import fuzz
from scipy import ndimage

from mapdiff.ingest import Page, TextSpan
from mapdiff.models import BBox, LegendEntry, Panel, Point
from mapdiff.ocr import page_text
from mapdiff.settings import PROJECT_ROOT
from mapdiff.taxonomy import Category

SIZE = 48  # silhouettes are SIZE x SIZE


def library_dir() -> Path:
    """Where learned icons live: ``MAPDIFF_ICON_LIBRARY`` or ``airport-icons/`` in the project."""
    return Path(os.environ.get("MAPDIFF_ICON_LIBRARY") or PROJECT_ROOT / "airport-icons")


# ---------------- silhouettes ----------------


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB on a white background: transparent parts of downloaded icons must not turn black."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.split()[-1])
        return flat
    return img.convert("RGB")


def silhouette(img: Image.Image) -> np.ndarray:
    """The glyph's shape as a SIZE x SIZE float mask, independent of colour and of any tile or circle behind it."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    fg = _differs(rgb, _border_colour(rgb))
    if fg.any():
        fill = np.median(rgb[fg], axis=0)  # the dominant ink: a tile's or circle's fill if there is one
        container = _container(fg & (np.abs(rgb - fill).max(axis=2) < 60))
        if container is not None:  # glyph drawn inside the fill: the glyph is what is not fill-coloured
            inside = ndimage.binary_erosion(ndimage.binary_fill_holes(container), iterations=2)
            fg = inside & (np.abs(rgb - fill).max(axis=2) >= 60)
    return _normalise(_drop_edge_slivers(fg))


def has_shape(sil: np.ndarray) -> bool:
    """False for empty crops and plain colour swatches (a filled square has no glyph to compare)."""
    return 0.03 < float(sil.mean()) < 0.7


def shape_vector(sil: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    """Outline descriptor of a silhouette (histograms of edge directions per cell, HOG-style), unit length.

    Edge directions describe a glyph's drawing better than its filled area, so different icons that happen to cover
    the same area (a coffee cup and a lift) score low.
    """
    gx, gy = cv2.Sobel(sil, cv2.CV_32F, 1, 0), cv2.Sobel(sil, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    ang = np.minimum(((np.arctan2(gy, gx) % np.pi) / np.pi * bins).astype(int), bins - 1)
    cells = []
    for y in range(0, SIZE, cell):
        for x in range(0, SIZE, cell):
            h = np.bincount(ang[y:y + cell, x:x + cell].ravel(), mag[y:y + cell, x:x + cell].ravel(), bins)
            cells.append(h / (np.linalg.norm(h) + 1e-6))
    v = np.concatenate(cells)
    return v / (np.linalg.norm(v) + 1e-9)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """0..1 shape similarity of two silhouettes."""
    return float(shape_vector(a) @ shape_vector(b))


def _border_colour(rgb: np.ndarray) -> np.ndarray:
    frame = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    return np.median(frame, axis=0)


def _differs(rgb: np.ndarray, colour: np.ndarray, floor: float = 40.0) -> np.ndarray:
    diff = np.abs(rgb - colour).max(axis=2)
    if diff.max() <= floor:
        return np.zeros(diff.shape, bool)
    thr, _ = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return diff > max(float(thr), floor)


def _container(fg: np.ndarray) -> np.ndarray | None:
    """The largest blob if it is a solid tile or circle filling most of the crop, else None."""
    lab, n = ndimage.label(fg)
    if n == 0:
        return None
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    blob = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(blob)
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    solid = ndimage.binary_fill_holes(blob)[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean()
    big = h * w >= 0.4 * fg.size
    has_glyph = ndimage.binary_fill_holes(blob).sum() > 1.05 * blob.sum()  # holes = a glyph cut out of the fill
    return blob if big and solid > 0.7 and has_glyph else None


def _drop_edge_slivers(fg: np.ndarray) -> np.ndarray:
    """Remove thin strips along the crop's edge: background caught when a box is a few pixels off the icon."""
    lab, n = ndimage.label(fg)
    thin = max(2, round(0.08 * min(fg.shape)))
    H, W = fg.shape
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        on_edge = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W
        if on_edge and min(h, w) <= thin:
            fg[lab == k] = False
    return fg


def _normalise(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((SIZE, SIZE), np.float32)
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return out
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    side = max(crop.shape)
    sq = np.zeros((side, side), np.float32)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    sq[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    inner = SIZE - 4
    out[2:2 + inner, 2:2 + inner] = cv2.resize(sq, (inner, inner), interpolation=cv2.INTER_AREA)
    return out


# ---------------- legend icons ----------------


@dataclass
class LegendIcon:
    meaning: str
    category: Category | None
    bbox: BBox | None  # where the icon is drawn on the page (None for an icon taken from the library)
    image: Image.Image = field(repr=False)

    @property
    def silhouette(self) -> np.ndarray:
        return silhouette(self.image)


def legend_icons(page: Page, legend: list[LegendEntry], panels: list[Panel], min_match: int = 80) -> list[LegendIcon]:
    """Cut out the icon drawn next to each icon-type legend entry's meaning text."""
    W, H = page.image.size
    rgb = np.asarray(page.image, dtype=np.float32)
    lines: list[tuple[TextSpan, np.ndarray, tuple[int, int, int, int]]] = []
    for panel in (p for p in panels if p.page == page.index and p.kind == "legend"):
        area = panel.bbox.expanded(0.1)  # recorded legend boxes are often a little tight
        spans = page_text(page, area)
        px = _px(area, W, H)
        region = rgb[px[1]:px[3], px[0]:px[2]]
        ink = _differs(region, _border_colour(region), floor=30.0)
        for (lx0, ly0, lx1, ly1) in (_px(sp.bbox, W, H) for sp in spans):  # text is not an icon
            ink[max(0, ly0 - px[1]):max(0, ly1 - px[1]), max(0, lx0 - px[0]):max(0, lx1 - px[0])] = False
        lines += [(sp, ink, px) for sp in spans]
    out: list[LegendIcon] = []
    for entry in (l for l in legend if l.page == page.index and l.kind == "icon"):
        for line, ink, px in _candidate_lines(entry.meaning, lines, min_match):
            box = _icon_beside(line, ink, px, W, H)
            if box is None:
                continue
            icon = LegendIcon(entry.meaning, entry.category, box, page.image.crop(_px(box, W, H)))
            if has_shape(icon.silhouette):
                out.append(icon)
                break
    return out


def _key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _candidate_lines(meaning: str, lines: list, min_match: int) -> list:
    """Lines that read like ``meaning``, best first; taller lines first among equals (a label, not text in an icon)."""
    want = _key(meaning)
    # Whole-text similarity first ('Toilet - Male' must not take 'Toilet - Female'); partial matches still find
    # the first line of a meaning that wraps onto two lines.
    scored = [(max(fuzz.ratio(want, k), 0.95 * fuzz.partial_ratio(want, k)), sp.bbox.height, item) for item in lines
              if len(k := _key((sp := item[0]).text)) >= 2]
    return [item for score, _, item in sorted(scored, key=lambda t: (-t[0], -t[1])) if score >= min_match]


def _icon_beside(line: TextSpan, ink: np.ndarray, px: tuple[int, int, int, int], W: int, H: int) -> BBox | None:
    """The nearest icon-sized group of ink left or right of a text line, on the same row."""
    lx0, ly0, lx1, ly1 = _px(line.bbox, W, H)
    h = max(4, ly1 - ly0)
    cy = (ly0 + ly1) / 2
    lab, _ = ndimage.label(ink)
    parts = []
    for sl in ndimage.find_objects(lab):
        y0, y1, x0, x1 = sl[0].start + px[1], sl[0].stop + px[1], sl[1].start + px[0], sl[1].stop + px[0]
        if (y1 - y0) * (x1 - x0) >= 4 and y0 < cy + 2.5 * h and y1 > cy - 2.5 * h:
            parts.append((x0, y0, x1, y1))

    def gap(b) -> float | None:
        x0, _, x1, _ = b
        if x1 <= lx0 + 0.2 * h:
            return lx0 - x1
        return x0 - lx1 if x0 >= lx1 - 0.2 * h else None

    seeds = [b for b in parts if abs((b[1] + b[3]) / 2 - cy) <= 1.2 * h and (g := gap(b)) is not None and g <= 9 * h
             and b[3] - b[1] >= 0.5 * h]
    if not seeds:
        return None
    # Nearest in the text's own row: legends stack rows tightly, so being level with the text counts most.
    x0, y0, x1, y1 = min(seeds, key=lambda b: gap(b) + 4 * abs((b[1] + b[3]) / 2 - cy))
    grown = True
    while grown:  # add nearby pieces of the same glyph (an 'i' dot, the second toilet figure), not the next row
        grown = False
        for b in parts:
            near = b[0] < x1 + 0.3 * h and b[2] > x0 - 0.3 * h and b[1] < y1 + 0.15 * h and b[3] > y0 - 0.15 * h
            inside_row = abs((b[1] + b[3]) / 2 - cy) <= 1.6 * h
            if near and inside_row and not (x0 <= b[0] and b[2] <= x1 and y0 <= b[1] and b[3] <= y1):
                x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
                grown = True
    if not (0.8 * h <= y1 - y0 <= 5 * h and 0.5 * h <= x1 - x0 <= 5 * h):
        return None
    return BBox(x0=x0 / W, y0=y0 / H, x1=x1 / W, y1=y1 / H)


def _px(b: BBox, W: int, H: int) -> tuple[int, int, int, int]:
    return int(b.x0 * W), int(b.y0 * H), int(np.ceil(b.x1 * W)), int(np.ceil(b.y1 * H))


# ---------------- learned library ----------------


@dataclass
class LibraryIcon:
    id: str
    file: str  # relative to the library directory
    meaning: str
    category: str | None
    sources: list[str]  # "<map_id> v<version>" it was learned from
    verified: bool = False  # True once a reviewer confirmed the meaning on a map


class IconLibrary:
    """Icons learned from legends of processed maps, stored as images plus ``library.json``.

    With ``root=None`` the library lives in memory only (for experiments and tests). ``reference=True`` also loads
    the downloaded reference pictograms (``reference/manifest.json``); their sources start with ``reference:``.
    Icons a person rejected (``rejected.json`` plus their images in ``rejected/``) are never learned again, and a
    category a person corrected (``recategorised.json``: meaning → category) is used whenever that meaning is learned.
    """

    def __init__(self, root: Path | None, reference: bool = False):
        self.root = root
        self.icons: list[LibraryIcon] = []
        self._vec: dict[str, np.ndarray] = {}
        self._img: dict[str, Image.Image] = {}
        self.fixed: dict[str, str] = json.loads((root / "recategorised.json").read_text()) \
            if root and (root / "recategorised.json").exists() else {}
        self.rejected: list[dict] = []
        self._rejected_vec: list[np.ndarray] = []
        if root and (root / "rejected.json").exists():
            for d in json.loads((root / "rejected.json").read_text()):
                self.rejected.append(d)
                self._rejected_vec.append(shape_vector(silhouette(to_rgb(Image.open(root / d["file"])))))
        index = root / "library.json" if root else None
        if index and index.exists():
            for d in json.loads(index.read_text()):
                self._load(LibraryIcon(**d), root / d["file"])
        ref = root / "reference" / "manifest.json" if root and reference else None
        if ref and ref.exists():
            for d in json.loads(ref.read_text()):
                self._load(LibraryIcon(f"ref:{d['file']}", f"reference/{d['file']}", d["meaning"], d["category"],
                                       [f"reference:{d['set']}"]), root / "reference" / d["file"])

    def _load(self, icon: LibraryIcon, path: Path) -> None:
        img = to_rgb(Image.open(path))
        sil = silhouette(img)
        if has_shape(sil):
            self.icons.append(icon)
            self._img[icon.id] = img
            self._vec[icon.id] = shape_vector(sil)

    def add(self, icon: LegendIcon, source: str, same: float = 0.92) -> LibraryIcon | None:
        """Learn ``icon``; an icon already known with the same category and shape only gains a source.

        Returns None for an icon a person rejected before (same meaning and shape).
        """
        sil = icon.silhouette
        vec = shape_vector(sil)
        cat = self.fixed.get(icon.meaning, icon.category.value if icon.category else None)
        if any(r["meaning"] == icon.meaning and float(vec @ rv) >= same for r, rv in zip(self.rejected, self._rejected_vec)):
            return None
        for known in self.icons:
            if known.category == cat and float(vec @ self._vec[known.id]) >= same:
                if source not in known.sources:
                    known.sources.append(source)
                return known
        iid = hashlib.sha1(sil.tobytes() + (cat or "").encode()).hexdigest()[:12]  # same shape may differ in category
        rel = f"{cat or 'unknown'}/{iid}.png"
        if self.root:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            icon.image.save(self.root / rel)
        known = LibraryIcon(iid, rel, icon.meaning, cat, [source])
        self.icons.append(known)
        self._vec[iid] = vec
        self._img[iid] = icon.image
        return known

    def image(self, icon: LibraryIcon) -> Image.Image | None:
        return self._img.get(icon.id)

    def reject(self, icon_id: str, reason: str) -> None:
        """Remove a learned icon for good: its image moves to ``rejected/`` and it will not be learned again."""
        icon = next(i for i in self.icons if i.id == icon_id)
        rel = f"rejected/{icon.id}.png"
        (self.root / "rejected").mkdir(parents=True, exist_ok=True)
        (self.root / icon.file).rename(self.root / rel)
        self.rejected.append(dict(file=rel, meaning=icon.meaning, category=icon.category, sources=icon.sources,
                                  reason=reason))
        self._rejected_vec.append(self._vec.pop(icon.id))
        self._img.pop(icon.id, None)
        self.icons.remove(icon)
        (self.root / "rejected.json").write_text(json.dumps(self.rejected, indent=1, ensure_ascii=False))

    def recategorise(self, meaning: str, category: str) -> None:
        """Correct the category of every learned icon with this meaning, now and whenever it is learned again."""
        for icon in [i for i in self.icons if i.meaning == meaning and i.category != category]:
            rel = f"{category}/{icon.id}.png"
            (self.root / category).mkdir(parents=True, exist_ok=True)
            (self.root / icon.file).rename(self.root / rel)
            icon.file, icon.category = rel, category
        self.fixed[meaning] = category
        (self.root / "recategorised.json").write_text(json.dumps(self.fixed, indent=1, ensure_ascii=False))

    def match(self, sil: np.ndarray, k: int = 3, exclude_source: str | None = None) -> list[tuple[LibraryIcon, float]]:
        """The ``k`` most similar learned icons (optionally ignoring those learned only from ``exclude_source``)."""
        vec = shape_vector(sil)
        pool = [i for i in self.icons if not exclude_source or any(not s.startswith(exclude_source) for s in i.sources)]
        scored = sorted(((i, float(vec @ self._vec[i.id])) for i in pool), key=lambda t: -t[1])
        return scored[:k]

    def save(self) -> None:
        assert self.root, "an in-memory library has nowhere to be saved"
        self.root.mkdir(parents=True, exist_ok=True)
        icons = sorted((i for i in self.icons if not i.id.startswith("ref:")), key=lambda i: (i.category or "", i.meaning))
        (self.root / "library.json").write_text(json.dumps([asdict(i) for i in icons], indent=1, ensure_ascii=False))


# ---------------- finding a map's own icons on the map ----------------


@dataclass
class SymbolHit:
    meaning: str
    category: Category | None
    anchor: Point
    bbox: BBox
    score: float


def find_symbols(page: Page, icons: list[LegendIcon], regions: list[BBox], threshold: float = 0.7,
                 scales: tuple[float, ...] = (0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.55, 1.85),
                 work_px: int = 18, min_shape: float = 0.5) -> list[SymbolHit]:
    """Every place in ``regions`` where one of the map's own legend icons is drawn.

    Template matching proposes places; each is then kept only if its glyph has the icon's shape, because a plain
    tile (an "i" on yellow) otherwise also matches any window pane or number box of the same size.
    """
    W, H = page.image.size
    gray = cv2.cvtColor(np.asarray(page.image), cv2.COLOR_RGB2GRAY)
    hits: list[SymbolHit] = []
    shapes = {id(i): shape_vector(i.silhouette) for i in icons}
    for icon in icons:
        templ = cv2.cvtColor(np.asarray(icon.image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        if min(templ.shape) < 6:
            continue
        f = work_px / min(templ.shape)  # work at a resolution where the smallest searched icon is ~work_px
        for region in regions:
            rx0, ry0, rx1, ry1 = _px(region, W, H)
            area = cv2.resize(gray[ry0:ry1, rx0:rx1], None, fx=f * scales[0] * 2, fy=f * scales[0] * 2,
                              interpolation=cv2.INTER_AREA)
            k = f * scales[0] * 2  # area scale factor
            for s in scales:
                t = cv2.resize(templ, None, fx=k * s, fy=k * s, interpolation=cv2.INTER_AREA)
                if t.shape[0] < 8 or t.shape[1] < 8 or t.shape[0] >= area.shape[0] or t.shape[1] >= area.shape[1]:
                    continue
                res = cv2.matchTemplate(area, t, cv2.TM_CCOEFF_NORMED)
                for y, x in zip(*np.nonzero(res >= threshold)):
                    bx0, by0 = rx0 + x / k, ry0 + y / k
                    bx1, by1 = bx0 + t.shape[1] / k, by0 + t.shape[0] / k
                    box = BBox(x0=bx0 / W, y0=by0 / H, x1=bx1 / W, y1=by1 / H)
                    hits.append(SymbolHit(icon.meaning, icon.category, box.center, box, float(res[y, x])))
    hits = _suppress(hits)
    verified = []
    for h in hits:
        icon = next(i for i in icons if i.meaning == h.meaning)
        shape = float(shape_vector(silhouette(page.image.crop(_px(h.bbox, W, H)))) @ shapes[id(icon)])
        if shape >= min_shape:
            h.score = round(min(h.score, shape), 3)
            verified.append(h)
    return verified


def _suppress(hits: list[SymbolHit], iou: float = 0.3) -> list[SymbolHit]:
    """Keep the best-scoring hit where several overlap (same icon at nearby offsets, or similar icons)."""
    kept: list[SymbolHit] = []
    for h in sorted(hits, key=lambda h: -h.score):
        if all(_iou(h.bbox, k.bbox) < iou for k in kept):
            kept.append(h)
    return kept


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0
