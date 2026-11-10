"""Measure the icon and text detectors on stored map versions — free, no model calls.

1. Legend icons: how many icon-type legend entries are cut out of each map's legend.
2. Learning across versions: icons learned from a map's first version identify the icons of its later versions
   (edited, restyled, rasterised), whose colours may differ.
3. Learning across airports: icons learned from the other airports identify this airport's icons.
4. Symbol search: a map's own legend icons found on the map by template matching, compared with the items stored
   for that version (which are themselves imperfect, so read both numbers as estimates).
5. OCR on raster maps: how many stored names and codes are found by local OCR near their position.
The learned library is written to airport-icons/ (images + library.json), as the pipeline would learn it.

Run:  PYTHONNOUSERSITE=1 python scripts/evaluate_icons.py [--data-dir eval_data/economy] [--no-save]
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from mapdiff.agents.symbols import learn
from mapdiff.icons import IconLibrary, find_symbols, legend_icons, library_dir
from mapdiff.ingest import load_document
from mapdiff.ocr import ocr_spans
from mapdiff.settings import PROJECT_ROOT
from mapdiff.store import MapStore
from mapdiff.taxonomy import group_of

NEAR = 0.015  # page units: a detection "is" a stored item when it is this close
ACCEPT = 0.55  # library matches below this similarity are treated as "don't know"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "eval_data" / "economy"))
    ap.add_argument("--library", default=str(library_dir()))
    ap.add_argument("--no-save", action="store_true", help="measure only; do not write the library")
    args = ap.parse_args()
    store = MapStore(args.data_dir)
    maps = sorted(p.name for p in Path(args.data_dir).iterdir() if p.is_dir())

    versions = {}  # (map_id, v) -> (snapshot, page, icons)
    print("## 1. Legend icons cut out")
    for map_id in maps:
        for v in store.versions(map_id):
            snap = store.load(map_id, v)
            page = load_document(store.source_path(map_id, v), 6000)[0]
            icons = legend_icons(page, snap.legend, snap.panels)
            versions[(map_id, v)] = (snap, page, icons)
            want = sum(l.kind == "icon" for l in snap.legend)
            print(f"  {map_id} v{v} ({snap.source_file}): {len(icons)}/{want}")

    print("\n## 2. Learning across versions (library from v1 → later versions)")
    tally = defaultdict(lambda: [0, 0, 0])  # correct, wrong, unknown
    for map_id in maps:
        lib = IconLibrary(None)
        for icon in versions[(map_id, 1)][2]:
            lib.add(icon, f"{map_id} v1")
        for v in store.versions(map_id)[1:]:
            snap, _, icons = versions[(map_id, v)]
            for icon in icons:
                tally[f"{map_id} v{v}"][_judge(lib, icon, None)] += 1
    _print_tally(tally)

    print("\n## 3. Learning across airports (library from the other airports' maps)")
    lib = IconLibrary(None)
    for (map_id, v), (_, _, icons) in versions.items():
        for icon in icons:
            lib.add(icon, f"{map_id} v{v}")
    tally = defaultdict(lambda: [0, 0, 0])
    examples = []
    for map_id in maps:
        for icon in versions[(map_id, 1)][2]:
            r = _judge(lib, icon, map_id)
            tally[map_id][r] += 1
            best = lib.match(icon.silhouette, 1, exclude_source=map_id)
            if best and r != 2:
                examples.append((map_id, icon.meaning, best[0][0].meaning, best[0][1], r == 0))
    _print_tally(tally)
    for m, q, got, sc, ok in sorted(examples, key=lambda e: -e[3])[:25]:
        print(f"    {'✓' if ok else '✗'} {sc:.2f}  {q!r} ← {got!r}  ({m})")

    print("\n## 4. Symbol search on the map (own legend icons)")
    for (map_id, v), (snap, page, icons) in versions.items():
        regions = [p.bbox for p in snap.panels if p.page == page.index and p.is_map]
        t = time.perf_counter()
        hits = find_symbols(page, icons, regions)
        secs = time.perf_counter() - t
        groups = {group_of(i.category) for i in icons if i.category}
        stored = [e for e in snap.entities if group_of(e.category) in groups]
        found = sum(any(h.category and group_of(h.category) == group_of(e.category)
                        and h.anchor.dist(e.anchor) < NEAR for h in hits) for e in stored)
        right = sum(any(h.category and group_of(h.category) == group_of(e.category)
                        and h.anchor.dist(e.anchor) < NEAR for e in snap.entities) for h in hits)
        print(f"  {map_id} v{v}: {len(hits)} hits in {secs:.0f}s; stored items of these kinds found "
              f"{found}/{len(stored)}; hits on a stored item {right}/{len(hits)}")

    print("\n## 5. OCR on raster maps (stored names / codes read near their position)")
    for (map_id, v), (snap, page, _) in versions.items():
        if page.has_text_layer:
            continue
        t = time.perf_counter()
        spans = [s for p in snap.panels if p.page == page.index and p.is_map for s in ocr_spans(page.image, p.bbox)]
        secs = time.perf_counter() - t
        texts = [e for e in snap.entities if e.name or e.label]
        hit = sum(any(s.bbox.center.dist(e.anchor) < 0.03 and any(
            t and fuzz.partial_ratio(t.lower(), s.text.lower()) >= 85 for t in (e.name, e.label)) for s in spans)
            for e in texts)
        print(f"  {map_id} v{v}: {len(spans)} text lines in {secs:.0f}s; stored names/codes found {hit}/{len(texts)}")

    print("\n## 6. Reference pictograms from the web (airport-icons/reference) identifying each map's legend icons")
    ref = IconLibrary(Path(args.library), reference=True)
    ref.icons = [i for i in ref.icons if i.id.startswith("ref:")]
    by_set = defaultdict(list)
    for i in ref.icons:
        by_set[i.sources[0]].append(i)
    queries = [i for (m, v), (_, _, icons) in versions.items() if v == 1 for i in icons if i.category]
    pools = {"all reference sets": ref.icons, **by_set}
    for name, pool in pools.items():
        sub = IconLibrary(None)
        sub.icons, sub._vec = pool, {i.id: ref._vec[i.id] for i in pool}
        top1 = top3 = sure = sure_right = 0
        for q in queries:
            ranked = sub.match(q.silhouette, 3)
            ok = [bool(k.category) and group_of(q.category) == group_of(type(q.category)(k.category)) for k, _ in ranked]
            top1 += ok[0] if ok else 0
            top3 += any(ok)
            if ranked and ranked[0][1] >= 0.8:
                sure += 1
                sure_right += ok[0]
        n = len(queries) or 1
        print(f"  {name} ({len(pool)} icons): top-1 {top1 / n:.0%}, top-3 {top3 / n:.0%}; "
              f"similarity ≥ 0.8 on {sure} icons, right {sure_right}/{sure}")

    if not args.no_save:
        lib = IconLibrary(Path(args.library))
        for snap, page, _ in versions.values():
            learn(snap, [page], lib)
        print(f"\nLibrary: {len(lib.icons)} icons ({sum(i.verified for i in lib.icons)} verified) → {lib.root}")
    return 0


def _judge(lib: IconLibrary, icon, exclude: str | None) -> int:
    """0 = right meaning or category, 1 = wrong, 2 = below the acceptance threshold (would ask the model)."""
    best = lib.match(icon.silhouette, 1, exclude_source=exclude)
    if not best or best[0][1] < ACCEPT:
        return 2
    known, _ = best[0]
    same_meaning = fuzz.token_set_ratio(known.meaning.lower(), icon.meaning.lower()) >= 85
    same_group = bool(known.category and icon.category) and group_of(icon.category) == group_of(
        type(icon.category)(known.category))
    return 0 if same_meaning or same_group else 1


def _print_tally(tally) -> None:
    tot = [0, 0, 0]
    for name, (ok, bad, unk) in tally.items():
        print(f"  {name}: right {ok}, wrong {bad}, unsure {unk}")
        tot = [a + b for a, b in zip(tot, (ok, bad, unk))]
    n = sum(tot) or 1
    print(f"  TOTAL: right {tot[0]} ({tot[0] / n:.0%}), wrong {tot[1]} ({tot[1] / n:.0%}), unsure {tot[2]} ({tot[2] / n:.0%})")


if __name__ == "__main__":
    raise SystemExit(main())
