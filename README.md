# mapdiff — airport floorplan understanding & change detection

Feed it a floorplan (PDF or image) under a stable `--map-id`. It builds a semantic inventory of
what is where, stores it as a new version, and reports what changed since the previous version.

```
                         icon library (airport-icons/) ◄── learn legend icons ──────────────┐
                                  │                                                         │
file ─► ingest ─► understanding (layout/legend → symbols → tiles → coverage verifier)       │
                       ▲          │     (legend icons shown to every agent as examples)     │
   lessons of vN-1 ────┤          ▼                                                         │
   (earlier mistakes)  │   ┌──► mapping-confidence ── mean ≥ 0.9 ──┐                        │
                       │   │          │ < 0.9 (one retry)          │                        │
                       │   └─ re-score ◄─ correction ─► misread lessons                     │
                       │                                           ▼                        │
                       │  snapshot vN-1 ──► change detection ◄── snapshot vN ───────────────┘
                       │                         │  (repeats of rejected claims held back)
                       │                         ▼
                       └── false-change lessons ◄─ change-confidence (old + new legend icons) ─► report
```

## Agents

| Agent | Module | What it does |
|---|---|---|
| Understanding | [understanding.py](src/mapdiff/agents/understanding.py) | Decodes this map's legend, scans tiles in parallel, fills names from legend codes, verifies coverage. For a new version, the previous version's inventory is a checklist so unchanged items are read the same way, and the verifier confirms every item that appeared or disappeared |
| Symbol detection | [symbols.py](src/mapdiff/agents/symbols.py) | No model calls. Cuts each legend icon out of the legend, finds every instance on the map by template matching plus a glyph-shape check (colour-independent), reads the map's text (PDF text layer, or local OCR for images) and resolves printed legend codes ("d" → Starbucks). Records these as candidates that the model then confirms, corrects and completes. The legend icons are also shown to every agent as a labelled picture (few-shot examples) |
| Lessons | [lessons.py](src/mapdiff/agents/lessons.py) | No model calls. Remembers each map's mistakes on its snapshot: misreadings the correction round fixed, and change claims the visual check rejected. The next version's map reading is told about earlier misreadings nearby, and a change that repeats a rejected claim is only reported if the visual check confirms it again |
| Mapping confidence | [mapping_confidence.py](src/mapdiff/agents/mapping_confidence.py) | Independently scores every entity: sceptical VLM review of a close-up crop + PDF text-layer / legend evidence. Mapping confidence = mean score |
| Correction | [correction.py](src/mapdiff/agents/correction.py) | Runs only when mapping confidence < threshold; fixes the entities the reviewer doubted (not "correct", or scored below 0.7), plus coverage gaps. At most `--max-corrections` (1) round, then the pipeline continues with what it has; the result is recorded in `snapshot.quality` |
| Change detection | [change_detection.py](src/mapdiff/agents/change_detection.py) | Deterministic semantic diff (+ optional VLM adjudication of near-identical names). Only who-uses-which-facility relations are compared (`checks_in_at`, `boards_at`, `handled_by`). Matches that look like a reading difference (name read in one version only, same name under another category, vague category) are marked `unconfirmed` |
| Change confidence | [change_confidence.py](src/mapdiff/agents/change_confidence.py) | Scores each change: `0.6 × visual + 0.4 × prior`. Visual = VLM judgement of OLD vs NEW crops of the same aligned spot (tables for relation changes); prior = entity scores × match geometry. Confidently "not real" → `rejected` (reported separately). `unconfirmed` changes are rejected unless the visual check confidently calls them real |
| Reporting | [reporting.py](src/mapdiff/agents/reporting.py) | Markdown + JSON report incl. mapping confidence of both versions and per-change confidence |

## Icon library

[airport-icons/](airport-icons/) holds icons learned from processed maps (meaning, category, source map). After every
stored version its legend icons are added; later versions of the same map also search for them. See
[airport-icons/README.md](airport-icons/README.md). `python scripts/evaluate_icons.py` measures the icon and text
detectors on stored maps without model calls and rebuilds the library.

## Design principles

- **Meaning, not presentation.** Each map has its own legend (colours, icons, numbers, letters).
  The understanding agent decodes it per map and resolves every marker to a fixed
  [taxonomy](src/mapdiff/taxonomy.py) + specific name. Change detection compares only category
  group, name, position and relations — a restyled map with the same content reports no changes.
- **Align versions on content, not pixels.** New-version coordinates are mapped into the old frame
  with a similarity transform fitted on names found in both versions (robust to re-layout/re-crop;
  genuinely moved items are rejected as outliers).
- **Exact text wherever possible.** Vector PDFs give exact text and positions; the model gets them
  alongside the images.
- **Bounded agent runs.** Per page: a layout & legend run → parallel per-tile scan runs (each
  limited to its core box) → a verifier run fed by deterministic coverage checks (legend codes never
  found, map text with no entity nearby, low-confidence items).
- **Vendor-neutral model access.** Agents depend only on the `VLM` protocol
  ([llm/base.py](src/mapdiff/llm/base.py)); [`VLMClient`](src/mapdiff/llm/vlm_client.py) selects a
  wire adapter by `--provider`. Adding a provider = one adapter class + one registry entry.

## Change types

`added`, `removed`, `replaced` (same spot, different occupant — cafe X → cafe Y), `repurposed`
(same spot, different kind of place), `moved`, `renamed` (VLM-adjudicated spelling/rebrand),
`relation_added/removed/changed` (e.g. airline moved to another check-in desk).
Changes involving low-confidence or half-named items are flagged `needs_review`.

## Setup

```bash
conda activate change_detect_py312
PYTHONNOUSERSITE=1 pip install -e ".[dev]"
cp .env.example .env           # then set ANTHROPIC_API_KEY in .env (gitignored)
mapdiff models --ping          # shows each agent's model and checks access (costs < $0.01)
```

`PYTHONNOUSERSITE=1` keeps packages from `~/.local` out of the env.

## Model and cost

Every agent that calls a model uses Haiku 5.5, the cheapest model, with its own effort
([settings.py](src/mapdiff/settings.py)): medium for understanding, mapping confidence, correction and
change confidence, low for the adjudicator. In the latest evaluation, a map cost
$0.11 on average to understand and check: $0.05–0.07 for small maps (40–80 entities), $0.10–0.21 for large ones
(200+). Each comparison cost about $0.007, so one map update (read the new version, compare it with the last)
is about $0.11.

Change detection and reporting are deterministic and cost nothing. Override one agent with
`MAPDIFF_<AGENT>_EFFORT` (or `MAPDIFF_<AGENT>_MODEL`), e.g. `MAPDIFF_UNDERSTANDING_EFFORT=high`.

Every request's tokens, cost and time are recorded per agent ([usage.py](src/mapdiff/llm/usage.py),
prices in [pricing.py](src/mapdiff/llm/pricing.py)). The record is stored in each `snapshot.json` and
changeset, printed after `mapdiff run`, and added to each change report.
`mapdiff versions` shows the cost of each version.

## Usage

```bash
mapdiff -v run Test_files/brussels_airport_departure.jpg --map-id bru-departures   # baseline v1
mapdiff -v run new_version.jpg --map-id bru-departures                             # v2 + report
mapdiff versions --map-id bru-departures
mapdiff diff --map-id bru-departures --from 1 --to 2      # re-diff stored versions, no model calls
```

Options: `--model/--effort` (one model or effort for all agents), `--workers` (parallel tiles), `--no-verify`,
`--min-confidence` (default 0.9), `--max-corrections` (default 1), `--no-change-scoring`,
`--adjudicate` (ask the VLM about ambiguous name changes), `--force` (re-process identical files;
byte-identical files are otherwise skipped). `mapdiff diff --no-llm` re-diffs without any model calls.

Output lives in `data/<map-id>/versions/vNNNN/snapshot.json` and
`data/<map-id>/changes/vNNNN_to_vMMMM.{md,json}`.

## Tests

```bash
PYTHONNOUSERSITE=1 python -m pytest
```

Tests run offline: the change-detection guarantees (restyle ⇒ no change, replacement, repurpose,
move, airline desk reassignment, …) and the full understanding pipeline driven by a scripted VLM.

## Evaluation (real models)

```bash
PYTHONNOUSERSITE=1 python scripts/evaluate_edits.py --name haiku-v2                # all cases
PYTHONNOUSERSITE=1 python scripts/evaluate_edits.py --name haiku-v2 --cases bru-departure-edits bru-departure-restyle
```

The script scores the pipeline on [Test_files_edited/](Test_files_edited/) (recall, false positives,
mapping confidence), plus cost and time per agent, per map and per diff. It keeps spending low:

- It starts with a smoke pair. If recall is below `--gate-recall`, the other cases are skipped.
- Understood maps are cached in `eval_data/<name>/`. Use a new `--name` after changing the pipeline, so the maps
  are understood again with the new code.
- `--budget-usd` (default $3) stops the run.

Results go to `eval_results/<timestamp>/`: `summary.md`, `summary.json` and `visual/index.html`.
The visual page opens in your browser when the run ends (`--no-visual` / `--no-open` to skip). It shows each map
before and after, the planted and detected changes with crops, and scores, cost and time per agent.
`python scripts/visual_report.py` rebuilds the page for the latest (or a given) run without model calls.
