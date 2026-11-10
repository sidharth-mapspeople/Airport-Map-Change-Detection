"""Deterministic ingest: render pages to images and pull exact text from vector PDFs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image

from mapdiff.models import BBox

Image.MAX_IMAGE_PIXELS = None  # airport maps are legitimately huge

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class TextSpan:
    text: str
    bbox: BBox  # normalised page coordinates


@dataclass
class Page:
    index: int
    image: Image.Image  # RGB
    source_kind: str  # "pdf" | "image"
    spans: list[TextSpan] = field(default_factory=list)
    legend_sheet: bytes | None = None  # PNG of this map's legend icons, shown to agents as examples (set by symbols)

    @property
    def has_text_layer(self) -> bool:
        return bool(self.spans)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_document(path: str | Path, max_px: int = 6000) -> list[Page]:
    """Load a PDF or raster image into pages rendered with the long side at most ``max_px``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, max_px)
    if suffix in IMAGE_SUFFIXES:
        return [_load_image(path, max_px)]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_pdf(path: Path, max_px: int) -> list[Page]:
    pages = []
    with pymupdf.open(path) as doc:
        for i, pg in enumerate(doc):
            rect = pg.rect
            zoom = max_px / max(rect.width, rect.height)
            image = pixmap_image(pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False))
            pages.append(Page(i, image, "pdf", _pdf_lines(pg)))
    return pages


def pixmap_image(pix: pymupdf.Pixmap) -> Image.Image:
    """A PyMuPDF render (no alpha) as a PIL image."""
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _pdf_lines(pg: pymupdf.Page) -> list[TextSpan]:
    """One span per text line, so multi-word labels ('Prayer Room') stay together."""
    w, h = pg.rect.width, pg.rect.height
    out = []
    for block in pg.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = " ".join(s["text"].strip() for s in line["spans"] if s["text"].strip())
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            out.append(TextSpan(text, BBox(x0=x0 / w, y0=y0 / h, x1=x1 / w, y1=y1 / h)))
    return out


def _load_image(path: Path, max_px: int) -> Page:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_px:
        image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return Page(0, image, "image")
