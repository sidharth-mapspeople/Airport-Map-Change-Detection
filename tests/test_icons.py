"""Icon silhouettes, legend icon cut-out, symbol search, the learned library, legend codes — all offline."""

from conftest import blank_page
from PIL import Image, ImageDraw

from mapdiff.agents.symbols import _codes, detect
from mapdiff.agents.workspace import ExtractionState
from mapdiff.icons import IconLibrary, LegendIcon, find_symbols, legend_icons, silhouette, similarity
from mapdiff.ingest import TextSpan
from mapdiff.models import BBox, LegendEntry, Panel
from mapdiff.ocr import _join
from mapdiff.taxonomy import Category


def tile(glyph: str, fill: str, ink: str, size: int = 60) -> Image.Image:
    """A legend-style icon: a filled tile with a glyph drawn in another colour."""
    img = Image.new("RGB", (size + 8, size + 8), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((4, 4, size + 4, size + 4), fill=fill)
    if glyph == "person":
        d.ellipse((size * 0.4, size * 0.15, size * 0.6, size * 0.35), fill=ink)
        d.rectangle((size * 0.38, size * 0.4, size * 0.62, size * 0.85), fill=ink)
    elif glyph == "stairs":
        for k in range(4):
            d.rectangle((size * (0.15 + 0.18 * k), size * (0.75 - 0.15 * k), size * 0.85, size * (0.85 - 0.15 * k)), fill=ink)
    else:  # arrow
        d.polygon([(size * 0.15, size * 0.5), (size * 0.6, size * 0.2), (size * 0.6, size * 0.8)], fill=ink)
        d.rectangle((size * 0.55, size * 0.42, size * 0.88, size * 0.58), fill=ink)
    return img


def test_silhouette_ignores_colour_but_not_shape():
    a = silhouette(tile("person", "#1f3a93", "white"))
    b = silhouette(tile("person", "#ffd400", "black"))  # restyled: other tile colour, inverted glyph colour
    c = silhouette(tile("stairs", "#1f3a93", "white"))
    assert similarity(a, b) > 0.9
    assert similarity(a, c) < similarity(a, b) - 0.2


def legend_page():
    """A page with a legend (icons left of their meaning) and a map that shows the icons, one of them restyled."""
    page = blank_page()
    img = page.image
    W, H = img.size
    for k, (glyph, meaning) in enumerate((("person", "Toilets"), ("stairs", "Stairs"))):
        y = int(H * (0.72 + 0.1 * k))
        img.paste(tile(glyph, "#333333", "white", 40), (int(W * 0.05), y))
        page.spans.append(TextSpan(meaning, BBox(x0=0.15, y0=(y + 14) / H, x1=0.3, y1=(y + 34) / H)))
    img.paste(tile("person", "#333333", "white", 40), (int(W * 0.5), int(H * 0.2)))
    img.paste(tile("stairs", "#333333", "white", 40), (int(W * 0.7), int(H * 0.3)))
    d = ImageDraw.Draw(img)
    d.rectangle((int(W * 0.3), int(H * 0.4), int(W * 0.3) + 48, int(H * 0.4) + 48), fill="#333333")  # plain tile
    legend = [LegendEntry(kind="icon", symbol="person", meaning="Toilets", category=Category.TOILET),
              LegendEntry(kind="icon", symbol="stairs", meaning="Stairs", category=Category.STAIRS)]
    panels = [Panel(id="map", kind="map", bbox=BBox(x0=0.0, y0=0.0, x1=1.0, y1=0.6)),
              Panel(id="legend", kind="legend", bbox=BBox(x0=0.02, y0=0.68, x1=0.4, y1=0.95))]
    return page, legend, panels


def test_legend_icons_are_cut_out_and_found_on_the_map():
    page, legend, panels = legend_page()
    icons = legend_icons(page, legend, panels)
    assert [i.meaning for i in icons] == ["Toilets", "Stairs"]
    hits = find_symbols(page, icons, [panels[0].bbox])
    found = sorted((h.meaning, round(h.anchor.x, 1), round(h.anchor.y, 1)) for h in hits)
    assert found == [("Stairs", 0.7, 0.3), ("Toilets", 0.5, 0.2)]  # the plain dark tile is not taken for an icon


def test_detect_records_icons_and_printed_codes_as_candidates():
    page, legend, panels = legend_page()
    legend.append(LegendEntry(kind="marker_code", symbol="1-14 (number boxes)", meaning="Check-in",
                              category=Category.CHECK_IN))
    legend.append(LegendEntry(kind="marker_code", symbol="d", meaning="Starbucks", category=Category.CAFE))
    page.spans += [TextSpan("5", BBox(x0=0.2, y0=0.1, x1=0.21, y1=0.12)),
                   TextSpan("d", BBox(x0=0.8, y0=0.5, x1=0.81, y1=0.52))]
    st = ExtractionState()
    st.legend, st.panels = legend, panels
    found = detect(page, st, IconLibrary(None))
    got = sorted((e.category.value, e.name, e.label) for e in st.entities.values())
    assert got == [("cafe", "Starbucks", "d"), ("check_in", "Check-in 5", "5"), ("stairs", None, None),
                   ("toilet", None, None)]
    assert len(found.added) == 4 and page.legend_sheet  # the sheet of legend icons is ready for the agents
    assert _codes("A40") == [("a40", False)] and len(_codes("1 - 14")) == 14


def test_library_learns_once_and_matches_a_restyled_icon(tmp_path):
    lib = IconLibrary(tmp_path)
    icon = LegendIcon("Toilets", Category.TOILET, None, tile("person", "#333333", "white"))
    lib.add(icon, "bru v1")
    lib.add(LegendIcon("Toilets", Category.TOILET, None, tile("person", "#555555", "white")), "bru v2")
    lib.add(LegendIcon("Stairs", Category.STAIRS, None, tile("stairs", "#333333", "white")), "bru v1")
    lib.save()
    again = IconLibrary(tmp_path)
    assert len(again.icons) == 2 and again.icons[1].sources == ["bru v1", "bru v2"]
    (best, score), *_ = again.match(silhouette(tile("person", "#00aa55", "black")))
    assert best.meaning == "Toilets" and score > 0.8


def test_ocr_pieces_read_across_a_tile_seam_are_joined():
    a = TextSpan("Brussels Airport Informat", BBox(x0=0.1, y0=0.5, x1=0.3, y1=0.51))
    b = TextSpan("nformation", BBox(x0=0.27, y0=0.5, x1=0.35, y1=0.51))
    assert _join(a, b).text == "Brussels Airport Information"


def test_rejected_and_recategorised_icons_stay_that_way(tmp_path):
    lib = IconLibrary(tmp_path)
    swatch = lib.add(LegendIcon("Check-In", Category.CHECK_IN, None, tile("arrow", "#333333", "white")), "map v1")
    park = lib.add(LegendIcon("Accessibility park", Category.TOILET_ACCESSIBLE, None,
                              tile("person", "#333333", "white")), "map v1")
    lib.reject(swatch.id, "number tile, not an icon")
    lib.recategorise(park.meaning, "parking")
    lib.save()

    again = IconLibrary(tmp_path)  # e.g. a rebuild that learns the same legend once more
    assert again.add(LegendIcon("Check-In", Category.CHECK_IN, None, tile("arrow", "#555555", "white")), "map v2") is None
    relearned = again.add(LegendIcon("Accessibility park", Category.TOILET_ACCESSIBLE, None,
                                     tile("person", "#444444", "white")), "map v2")
    assert relearned.category == "parking" and [i.meaning for i in again.icons] == ["Accessibility park"]
