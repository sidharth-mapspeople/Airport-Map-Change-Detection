# airport-icons — icons learned from processed floorplans

Each icon here was cut out of the legend of a map that mapdiff processed, together with the meaning the legend
gives it. The pipeline adds to this library after every stored version
([symbols.py](../src/mapdiff/agents/symbols.py) `learn`), and uses it in two ways:

- **Same map, later versions:** icons learned from earlier versions are also searched for on a new version, so an
  icon still counts when a new legend omits or restyles it. Recognition across versions measured 97% on the test maps.
- **Missing categories:** a legend icon that the legend reader left without a category gets the category of a
  closely matching learned icon (similarity ≥ 0.8).

Icons are compared by the shape of their glyph, not their colour ([icons.py](../src/mapdiff/icons.py)). Across
different airports, shapes differ too much to label icons automatically (about 20–35% top-1 on the test maps), so the
library is not used for that.

## Layout

- `<category>/<id>.png` — the icon as cut from the legend (original colours).
- `library.json` — one entry per icon: `meaning`, `category`, `sources` (map id and version it was learned from), and
  `verified` (an entity of that kind was reviewed "correct" at score ≥ 0.9 on that map).
- `rejected.json` + `rejected/` — icons a person removed, with the reason. They are never learned again.
- `recategorised.json` — categories a person corrected (meaning → category), applied whenever that meaning is learned.

## Rebuild

```bash
PYTHONNOUSERSITE=1 python scripts/evaluate_icons.py   # measures the detectors and relearns from eval_data/economy
```

The icons are crops of the airport maps in `Test_files/` and `Test_files_edited/`, kept for this project's own
recognition; they are not redistributable artwork.

## reference/ — open pictogram sets from the web

299 airport-relevant pictograms from six open sets, reviewed by eye, rendered to PNG, each with its source and licence
in `reference/manifest.json`:

| Set | Licence | Icons | Identifies the test maps' legend icons (top-1 / top-3) |
|---|---|---|---|
| ISO 7001 (Wikimedia Commons) | CC0 | 72 | 35% / 56% |
| AIGA / US DOT (Wikimedia Commons) | public domain | 44 | 40% / 40% |
| Google Material Symbols | Apache 2.0 | 51 | 21% / 42% |
| Mapbox Maki | CC0 | 46 | 21% / 33% |
| Temaki | CC0 | 45 | 19% / 27% |
| OpenStreetMap Carto | CC0 | 41 | 15% / 23% |
| **All together** | | 299 | **48% / 60%** |

Each map draws its own icons, so generic pictograms are still wrong about half the time; only very close matches
(similarity ≥ 0.8, 7 of 8 right) are reliable, and those are rare. The pipeline therefore does **not** use them; they
are kept as a reference collection. `IconLibrary(root, reference=True)` loads them for experiments.

Curation: `reference/rejected.json` lists the 76 pictograms removed on review (not airport facilities, a wrong meaning
for their category, or unreadable) with the reason, and `reference/recategorised.json` the 9 moved to a better
category. `python scripts/fetch_reference_icons.py` skips the rejected ones and keeps the corrections when it
re-downloads. Measure with `python scripts/evaluate_icons.py --no-save` (section 6).
