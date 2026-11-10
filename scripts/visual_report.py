"""Rebuild the visual results page for an evaluation run (the evaluation builds it automatically).

Run:  PYTHONNOUSERSITE=1 python scripts/visual_report.py [eval_results/<timestamp>] [--no-open]
Without a folder, uses the latest run. No model calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mapdiff.evaluation import EVAL_DATA_DIR, latest_results
from mapdiff.visual_report import build, show


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", help="eval_results/<timestamp> folder (default: latest run)")
    ap.add_argument("--data-dir", default=str(EVAL_DATA_DIR))
    ap.add_argument("--no-open", dest="open", action="store_false", help="don't open the page in a browser")
    args = ap.parse_args()
    page = build(Path(args.results) if args.results else latest_results(), Path(args.data_dir))
    print(f"Visual report → {page}")
    if args.open:
        show(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
