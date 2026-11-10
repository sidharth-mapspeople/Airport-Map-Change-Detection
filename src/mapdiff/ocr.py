"""Local text detection for raster pages (no text layer), so labels and numbers get exact positions for free.

Wraps RapidOCR (ONNX models bundled with the package). Large regions are read in overlapping tiles at their native
resolution, so small map labels stay legible; results come back as :class:`~mapdiff.ingest.TextSpan` in page
coordinates, the same shape as a PDF text layer.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
from PIL import Image

from mapdiff.ingest import Page, TextSpan
from mapdiff.models import FULL_PAGE, BBox

TILE_PX = 1600  # OCR tile side in page pixels
OVERLAP_PX = 160  # tiles overlap so a label on a seam is read whole by one of them
MIN_SCORE = 0.5
_lock = threading.Lock()  # the ONNX session is shared; RapidOCR is not documented as thread-safe


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def page_text(page: Page, box: BBox = FULL_PAGE) -> list[TextSpan]:
    """Text lines in ``box``: the PDF text layer when there is one, otherwise OCR."""
    if page.has_text_layer:
        return [s for s in page.spans if box.contains(s.bbox.center)]
    return ocr_spans(page.image, box)


def ocr_spans(image: Image.Image, box: BBox = FULL_PAGE, upscale: float = 1.5) -> list[TextSpan]:
    """OCR ``box`` of ``image``, enlarged by ``upscale`` first (1.5 reads small map labels and codes best)."""
    W, H = image.size
    x0, y0, x1, y1 = int(box.x0 * W), int(box.y0 * H), int(box.x1 * W), int(box.y1 * H)
    step = TILE_PX - OVERLAP_PX
    found: list[TextSpan] = []
    for ty in range(y0, max(y0 + 1, y1 - OVERLAP_PX), step):
        for tx in range(x0, max(x0 + 1, x1 - OVERLAP_PX), step):
            tile = image.crop((tx, ty, min(tx + TILE_PX, x1), min(ty + TILE_PX, y1)))
            if upscale != 1.0:
                tile = tile.resize((round(tile.width * upscale), round(tile.height * upscale)), Image.Resampling.LANCZOS)
            with _lock:
                result, _ = _engine()(np.asarray(tile))
            for quad, text, score in result or []:
                if float(score) < MIN_SCORE or not text.strip():
                    continue
                xs = [p[0] / upscale + tx for p in quad]
                ys = [p[1] / upscale + ty for p in quad]
                found.append(TextSpan(text.strip(), BBox(x0=min(xs) / W, y0=min(ys) / H, x1=max(xs) / W, y1=max(ys) / H)))
    return _merge_overlaps(found)


def _merge_overlaps(spans: list[TextSpan]) -> list[TextSpan]:
    """Join pieces of one line that were read on both sides of a tile seam."""
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(spans):
            j = next((j for j in range(i + 1, len(spans)) if _same_line(a.bbox, spans[j].bbox)), None)
            if j is not None:
                spans[i] = _join(a, spans.pop(j))
                merged = True
                break
    return spans


def _same_line(a: BBox, b: BBox) -> bool:
    v = min(a.y1, b.y1) - max(a.y0, b.y0)
    return v > 0.6 * min(a.height, b.height) and min(a.x1, b.x1) > max(a.x0, b.x0)


def _join(a: TextSpan, b: TextSpan) -> TextSpan:
    left, right = sorted((a, b), key=lambda s: s.bbox.x0)
    box = BBox(x0=left.bbox.x0, y0=min(a.bbox.y0, b.bbox.y0), x1=max(a.bbox.x1, b.bbox.x1), y1=max(a.bbox.y1, b.bbox.y1))
    if right.bbox.x0 - left.bbox.x0 < 0.5 * left.bbox.height or right.bbox.x1 <= left.bbox.x1:
        return TextSpan(max(a.text, b.text, key=len), box)  # the same text read twice
    overlap = next((k for k in range(min(len(left.text), len(right.text)), 0, -1)
                    if left.text.endswith(right.text[:k])), 0)
    text = left.text + right.text[overlap:] if overlap else max(a.text, b.text, key=len)
    return TextSpan(text, box)
