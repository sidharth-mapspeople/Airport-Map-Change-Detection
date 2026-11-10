"""Create artificially edited copies of Test_files/ with known ground truth.

Writes to Test_files_edited/:
  <variant files>       edited PDFs / images
  manifest.json         per case: baseline, variant, expected changes
  previews/             before/after crops of every edit, for eyeballing

Two kinds of case:
  edit     real content changes (renamed tenants, removed/added facilities, airline desk moves)
  restyle  presentation-only changes (colours, format, scale) — the expected change list is EMPTY

Run:  PYTHONNOUSERSITE=1 python scripts/make_test_edits.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from mapdiff.ingest import pixmap_image
from mapdiff.settings import PROJECT_ROOT

Image.MAX_IMAGE_PIXELS = None

ROOT = PROJECT_ROOT
SRC = ROOT / "Test_files"
OUT = ROOT / "Test_files_edited"
PREVIEWS = OUT / "previews"

AUH = SRC / "Terminal 1 map_2015_R3_tcm13-4100 (1).pdf"
AKL_DOM = SRC / "auckland_Domestic-Terminal-Forecourt.pdf"
AKL_OVW = SRC / "auckland_airport.pdf"
BRU_DEP = SRC / "brussels_airport_departure.jpg"
BRU_OVW = SRC / "brussels_airport.jpg"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


# ---------------------------------------------------------------- PDF helpers


def pdf_replace_text(page: pymupdf.Page, old: str, new: str, near: tuple[float, float] | None = None) -> pymupdf.Rect:
    """Replace the text ``old`` (first match, or the match nearest ``near``) with ``new`` in the same size/colour."""
    rects = page.search_for(old)
    if not rects:
        raise ValueError(f"{old!r} not found")
    if near:
        rects.sort(key=lambda r: abs(r.x0 - near[0]) + abs(r.y0 - near[1]))
    r = rects[0]
    span = next(s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]
                if old.split()[0] in s["text"] and pymupdf.Rect(s["bbox"]).intersects(r))
    size, color, baseline = span["size"], pymupdf.sRGB_to_pdf(span["color"]), span["origin"][1]
    page.add_redact_annot(pymupdf.Rect(r.x0, r.y0 + 1, r.x1, r.y1 - 1), fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    if new:
        page.insert_text((r.x0, baseline), new, fontname="helv", fontsize=size, color=color)
    return r


def pdf_icon_drawings(page: pymupdf.Page, center: tuple[float, float], radius: float = 4.0) -> list[dict]:
    cx, cy = center
    return [d for d in page.get_drawings()
            if d["rect"].width <= 2 * radius + 1 and d["rect"].height <= 2 * radius + 1
            and abs((d["rect"].x0 + d["rect"].x1) / 2 - cx) <= radius
            and abs((d["rect"].y0 + d["rect"].y1) / 2 - cy) <= radius]


def pdf_copy_drawings(page: pymupdf.Page, drawings: list[dict], dx: float, dy: float) -> None:
    def P(p):
        return pymupdf.Point(p.x + dx, p.y + dy)

    for d in drawings:
        sh = page.new_shape()
        for it in d["items"]:
            if it[0] == "l":
                sh.draw_line(P(it[1]), P(it[2]))
            elif it[0] == "c":
                sh.draw_bezier(P(it[1]), P(it[2]), P(it[3]), P(it[4]))
            elif it[0] == "re":
                sh.draw_rect(it[1] + (dx, dy, dx, dy))
            elif it[0] == "qu":
                q = it[1]
                sh.draw_quad(pymupdf.Quad(P(q.ul), P(q.ur), P(q.ll), P(q.lr)))
        sh.finish(fill=d.get("fill"), color=d.get("color"), width=d.get("width") or 0,
                  even_odd=bool(d.get("even_odd")), closePath=bool(d.get("closePath")),
                  fill_opacity=d.get("fill_opacity") or 1, stroke_opacity=d.get("stroke_opacity") or 1)
        sh.commit()


def pdf_remove_drawings(page: pymupdf.Page, drawings: list[dict]) -> pymupdf.Rect:
    box = pymupdf.Rect(drawings[0]["rect"])
    for d in drawings[1:]:
        box |= d["rect"]
    box = box + (-0.3, -0.3, 0.3, 0.3)
    page.add_redact_annot(box, fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                          text=pymupdf.PDF_REDACT_TEXT_NONE)
    return box


def pdf_preview(before: Path, after: Path, rect: pymupdf.Rect, name: str, pad: float = 25) -> None:
    clip = rect + (-pad, -pad, pad, pad)
    imgs = []
    for p in (before, after):
        with pymupdf.open(p) as doc:
            imgs.append(pixmap_image(doc[0].get_pixmap(clip=clip, dpi=300)))
    _side_by_side(imgs, name)


# ---------------------------------------------------------------- raster helpers


def ring_median(im: Image.Image, box: tuple[int, int, int, int], border: int = 8) -> tuple[int, ...]:
    """Median colour of a thin ring just outside ``box`` (the local background)."""
    x0, y0, x1, y1 = box
    strips = [im.crop((x0 - border, y0, x0, y1)), im.crop((x1, y0, x1 + border, y1)),
              im.crop((x0, y0 - border, x1, y0)), im.crop((x0, y1, x1, y1 + border))]
    px = np.concatenate([np.asarray(s).reshape(-1, 3) for s in strips])
    return tuple(int(v) for v in np.median(px, axis=0))


def img_erase(im: Image.Image, box: tuple[int, int, int, int], color=None) -> None:
    ImageDraw.Draw(im).rectangle(box, fill=color or ring_median(im, box))


def img_text(im: Image.Image, xy: tuple[int, int], text: str, size: int, anchor: str = "ls") -> None:
    font = ImageFont.truetype(FONT, size) if Path(FONT).exists() else ImageFont.load_default(size=size)
    ImageDraw.Draw(im).text(xy, text, fill=(0, 0, 0), font=font, anchor=anchor)


def img_copy_icon(im: Image.Image, src: tuple[int, int, int, int], center: tuple[int, int], radius: int = 8) -> tuple:
    patch = im.crop(src)
    mask = Image.new("L", patch.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((1, 1, patch.width - 2, patch.height - 2), radius=radius, fill=255)
    x0, y0 = center[0] - patch.width // 2, center[1] - patch.height // 2
    im.paste(patch, (x0, y0), mask)
    return x0, y0, x0 + patch.width, y0 + patch.height


def img_preview(before: Image.Image, after: Image.Image, box: tuple[int, int, int, int], name: str, pad: int = 80) -> None:
    x0, y0, x1, y1 = box
    clip = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    _side_by_side([before.crop(clip), after.crop(clip)], name)


def hue_shift(im: Image.Image, shift: int) -> Image.Image:
    h, s, v = im.convert("HSV").split()
    return Image.merge("HSV", (h.point(lambda x: (x + shift) % 256), s, v)).convert("RGB")


def _side_by_side(imgs: list[Image.Image], name: str) -> None:
    h = max(i.height for i in imgs)
    canvas = Image.new("RGB", (sum(i.width for i in imgs) + 20, h + 30), "white")
    d = ImageDraw.Draw(canvas)
    x = 0
    for label, i in zip(("BEFORE", "AFTER"), imgs):
        canvas.paste(i, (x, 30))
        d.text((x + 4, 6), label, fill=(200, 0, 0), font=ImageFont.load_default(size=18))
        x += i.width + 20
    canvas.thumbnail((1600, 1600))
    canvas.save(PREVIEWS / f"{name}.png")


# ---------------------------------------------------------------- cases


def auh_edits() -> dict:
    out = OUT / "auh_t1_edited.pdf"
    doc = pymupdf.open(AUH)
    pg = doc[0]
    W, H = pg.rect.width, pg.rect.height
    r1 = pdf_replace_text(pg, "Culto Café", "Costa Coffee")
    r2 = pdf_replace_text(pg, "McDonald's", "Burger King")
    r3 = pdf_replace_text(pg, "Etihad Transfer Desk", "Qatar Transfer Desk")
    icon = pdf_icon_drawings(pg, (705.9, 503.3))
    assert len(icon) == 6, f"expected 6 toilet-icon paths, got {len(icon)}"
    pdf_copy_drawings(pg, icon, dx=753.0 - 705.9, dy=467.0 - 503.3)  # add a toilet before removing the original
    removed = pdf_remove_drawings(pg, icon)
    doc.save(out, garbage=3, deflate=True)
    for i, r in enumerate((r1, r2, r3), 1):
        pdf_preview(AUH, out, r, f"auh_t1_edit{i}_legend", pad=12)
    pdf_preview(AUH, out, removed | pymupdf.Rect(750, 464, 756, 470), "auh_t1_edit4_toilet_moved")
    return dict(
        id="auh-t1-edits", kind="edit", baseline=AUH, variant=out,
        description="Legend renames (2 cafés/restaurants, 1 transfer desk); one toilet icon removed; one added elsewhere.",
        expected=[
            {"type": "replaced", "old_name": "Culto Café", "new_name": "Costa Coffee"},
            {"type": "replaced", "old_name": "McDonald's", "new_name": "Burger King"},
            {"type": "replaced", "old_name": "Etihad Transfer Desk", "new_name": "Qatar Transfer Desk"},
            {"type": "removed", "category": "toilet", "at": [705.9 / W, 503.3 / H]},
            {"type": "added", "category": "toilet", "at": [753.0 / W, 467.0 / H]},
        ],
    )


def auh_restyle() -> dict:
    """Same content, different format: vector PDF rasterised to a JPEG (no text layer)."""
    out = OUT / "auh_t1_rasterised.jpg"
    with pymupdf.open(AUH) as doc:
        pixmap_image(doc[0].get_pixmap(dpi=250)).save(out, quality=90)
    return dict(id="auh-t1-restyle", kind="restyle", baseline=AUH, variant=out,
                description="Vector PDF rasterised to JPEG: no text layer, identical content.", expected=[])


def akl_dom_edits() -> dict:
    out = OUT / "akl_domestic_edited.pdf"
    doc = pymupdf.open(AKL_DOM)
    pg = doc[0]
    r1 = pdf_replace_text(pg, "Rideshare (Uber)", "Rideshare (Ola)")
    r2 = pdf_replace_text(pg, "Supershuttle", "SkyBus")
    r3 = pdf_replace_text(pg, "April 2026", "October 2026")
    doc.save(out, garbage=3, deflate=True)
    for i, r in enumerate((r1, r2, r3), 1):
        pdf_preview(AKL_DOM, out, r, f"akl_domestic_edit{i}", pad=15)
    return dict(
        id="akl-domestic-edits", kind="edit", baseline=AKL_DOM, variant=out,
        description="Legend renames of two transport services; printed date April → October 2026 (metadata only).",
        expected=[
            {"type": "replaced", "old_name": "Rideshare (Uber)", "new_name": "Rideshare (Ola)"},
            {"type": "replaced", "old_name": "Supershuttle", "new_name": "SkyBus"},
        ],
        expected_map_date="October 2026",
    )


def akl_ovw_edits() -> dict:
    out = OUT / "akl_overview_edited.pdf"
    doc = pymupdf.open(AKL_OVW)
    pg = doc[0]
    r1 = pdf_replace_text(pg, "Ibis Budget Hotel", "Holiday Inn Express", near=(576.8, 57.9))
    r2 = pdf_replace_text(pg, "Kuriri Cafe and te Kaahu", "Jamie's Italian")
    pdf_replace_text(pg, "Restaurant", "", near=(576.8, 223.8))
    doc.save(out, garbage=3, deflate=True)
    pdf_preview(AKL_OVW, out, r1, "akl_overview_edit1", pad=15)
    pdf_preview(AKL_OVW, out, r2 | pymupdf.Rect(576.8, 223.8, 613.3, 232.9), "akl_overview_edit2", pad=15)
    return dict(
        id="akl-overview-edits", kind="edit", baseline=AKL_OVW, variant=out,
        description="Amenity legend: hotel and restaurant replaced by new tenants.",
        expected=[
            {"type": "replaced", "old_name": "Ibis Budget Hotel", "new_name": "Holiday Inn Express"},
            {"type": "replaced", "old_name": "Kuriri Cafe and te Kaahu Restaurant", "new_name": "Jamie's Italian"},
        ],
    )


def bru_dep_edits() -> dict:
    out = OUT / "bru_departure_edited.jpg"
    before = Image.open(BRU_DEP).convert("RGB")
    im = before.copy()
    W, H = im.size
    # 1. legend: d. Starbucks -> d. Costa Coffee
    b1 = (385, 3200, 760, 3283)
    img_erase(im, b1)
    img_text(im, (395, 3265), "Costa Coffee", 52)
    # 2. airline table: Emirates desk 14 -> 5
    b2 = (4975, 880, 5040, 930)
    img_erase(im, b2)
    img_text(im, (5030, 921), "5", 46, anchor="rs")
    # 3. airline table: TAROM row removed
    b3 = (5130, 1493, 6175, 1550)
    img_erase(im, b3)
    # 4. a toilet icon (incl. drop shadow) removed from the map
    b4 = (3780, 2170, 3902, 2295)
    img_erase(im, b4)
    # 5. a lift icon added on the floor (copied from an existing one)
    b5 = img_copy_icon(im, (3591, 2629, 3673, 2709), (4930, 2650))
    im.save(out, quality=92)
    for i, b in enumerate((b1, b2, b3, b4, b5), 1):
        img_preview(before, im, b, f"bru_departure_edit{i}")
    return dict(
        id="bru-departure-edits", kind="edit", baseline=BRU_DEP, variant=out,
        description="Café rename in legend; Emirates moved desk 14 → 5; TAROM removed from desk table; "
                    "one toilet icon removed; one lift icon added.",
        expected=[
            {"type": "replaced", "old_name": "Starbucks", "new_name": "Costa Coffee"},
            {"type": "relation_changed", "subject": "Emirates", "old_object": "14", "new_object": "5"},
            {"type": "relation_removed", "subject": "TAROM"},
            {"type": "removed", "category": "toilet", "at": [3837 / W, 2227 / H]},
            {"type": "added", "category": "lift", "at": [4930 / W, 2650 / H]},
        ],
    )


def bru_dep_restyle() -> dict:
    """Recoloured, shrunk and shifted: everything moves and changes colour, nothing changes meaning."""
    out = OUT / "bru_departure_restyled.jpg"
    im = hue_shift(Image.open(BRU_DEP).convert("RGB"), 40)
    small = im.resize((round(im.width * 0.92), round(im.height * 0.92)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", im.size, "white")
    canvas.paste(small, (260, 120))
    canvas.save(out, quality=88)
    img_preview(Image.open(BRU_DEP).convert("RGB"), canvas, (3700, 2100, 3950, 2350), "bru_departure_restyle")
    return dict(id="bru-departure-restyle", kind="restyle", baseline=BRU_DEP, variant=out,
                description="Hue rotated 56°, scaled to 92 % and shifted; content identical.", expected=[])


def bru_ovw_edits() -> dict:
    out = OUT / "bru_overview_edited.jpg"
    before = Image.open(BRU_OVW).convert("RGB")
    im = before.copy()
    W, H = im.size
    b1 = (4092, 3040, 4178, 3126)
    img_erase(im, b1)
    im.save(out, quality=92)
    img_preview(before, im, b1, "bru_overview_edit1")
    return dict(
        id="bru-overview-edits", kind="edit", baseline=BRU_OVW, variant=out,
        description="One toilet icon removed from the isometric overview.",
        expected=[{"type": "removed", "category": "toilet", "at": [4131 / W, 3079 / H]}],
    )


def bru_ovw_restyle() -> dict:
    out = OUT / "bru_overview_restyled.jpg"
    hue_shift(Image.open(BRU_OVW).convert("RGB"), -30).save(out, quality=85)
    return dict(id="bru-overview-restyle", kind="restyle", baseline=BRU_OVW, variant=out,
                description="Hue rotated −42° and recompressed; content identical.", expected=[])


def main() -> None:
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    cases = [auh_edits(), auh_restyle(), akl_dom_edits(), akl_ovw_edits(),
             bru_dep_edits(), bru_dep_restyle(), bru_ovw_edits(), bru_ovw_restyle()]
    for c in cases:
        c["baseline"] = str(Path(c["baseline"]).relative_to(ROOT))
        c["variant"] = str(Path(c["variant"]).relative_to(ROOT))
        for e in c["expected"]:
            if "at" in e:
                e["at"] = [round(v, 4) for v in e["at"]]
    (OUT / "manifest.json").write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False))
    print(f"Wrote {len(cases)} cases to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
