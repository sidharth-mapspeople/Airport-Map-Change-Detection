"""Command line: ``mapdiff run``, ``mapdiff diff``, ``mapdiff versions``, ``mapdiff models``."""

from __future__ import annotations

import argparse
import io
import sys

from mapdiff import settings
from mapdiff.agents.mapping_confidence import QualityConfig
from mapdiff.agents.reporting import usage_table
from mapdiff.agents.understanding import UnderstandingConfig
from mapdiff.llm import LLMError
from mapdiff.llm.base import ImagePart, TextPart, ToolSpec, UserTurn
from mapdiff.llm.usage import cost_of
from mapdiff.pipeline import PipelineConfig, process_file, rediff
from mapdiff.settings import AgentModels
from mapdiff.store import MapStore


def main(argv: list[str] | None = None) -> int:
    settings.load_env()
    ap = argparse.ArgumentParser(prog="mapdiff", description="Understand airport floorplans and detect changes.")
    ap.add_argument("--data-dir", default="data", help="version store root (default: data)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=None, help=f"use this model for every agent (default: {settings.MODEL})")
        p.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                       help="use this effort for every agent (default: per agent, see settings.py)")

    run = sub.add_parser("run", help="understand a file, store it as a new version and report changes")
    run.add_argument("file")
    run.add_argument("--map-id", required=True, help="stable id for this floorplan, e.g. auh-t1-departures")
    run.add_argument("--workers", type=int, default=4, help="parallel tile scans")
    run.add_argument("--no-verify", action="store_true", help="skip the verification sub-agent")
    run.add_argument("--adjudicate", action="store_true", help="let the VLM judge ambiguous name changes")
    run.add_argument("--force", action="store_true", help="process even if the file is byte-identical")
    run.add_argument("--min-confidence", type=float, default=0.9, help="required mean mapping confidence (default 0.9)")
    run.add_argument("--max-corrections", type=int, default=1, help="max correction rounds (default 1)")
    run.add_argument("--no-change-scoring", action="store_true", help="skip the change-confidence agent")
    add_model_args(run)

    diff = sub.add_parser("diff", help="re-run change detection between two stored versions")
    diff.add_argument("--map-id", required=True)
    diff.add_argument("--from", dest="from_v", type=int, required=True)
    diff.add_argument("--to", dest="to_v", type=int, required=True)
    diff.add_argument("--adjudicate", action="store_true")
    diff.add_argument("--no-llm", action="store_true", help="deterministic diff only (no change scoring)")
    add_model_args(diff)

    ver = sub.add_parser("versions", help="list stored versions of a map")
    ver.add_argument("--map-id", required=True)

    mod = sub.add_parser("models", help="show which model each agent uses")
    mod.add_argument("--ping", action="store_true", help="send one tiny image request to each model to check access")
    add_model_args(mod)

    args = ap.parse_args(argv)
    settings.configure_logging(args.verbose)
    store = MapStore(args.data_dir)

    try:
        if args.cmd == "versions":
            for v in store.versions(args.map_id):
                s = store.load(args.map_id, v)
                cost = cost_of(s.usage) if s.usage else None
                print(f"v{v}: {s.source_file}  {s.created_at:%Y-%m-%d %H:%M}  "
                      f"{len(s.entities)} entities, {len(s.relations)} relations"
                      + (f", ${cost:.4f}" if cost is not None else ""))
            return 0

        models = AgentModels.from_settings(args.model, args.effort)
        if args.cmd == "models":
            for agent, desc in models.describe().items():
                print(f"  {agent:<20} {desc}")
            return _ping(models) if args.ping else 0

        if args.cmd == "diff":
            adjudicator = models.adjudicator_fn(args.adjudicate and not args.no_llm)
            cs, report = rediff(args.map_id, store, args.from_v, args.to_v, None if args.no_llm else models,
                                adjudicator=adjudicator)
            print(f"{sum(not c.rejected for c in cs.changes)} changes → {report}")
            _print_usage(models)
            return 0

        adjudicator = models.adjudicator_fn(args.adjudicate)
        cfg = PipelineConfig(
            understanding=UnderstandingConfig(workers=args.workers, verify=not args.no_verify),
            quality=QualityConfig(min_confidence=args.min_confidence, max_corrections=args.max_corrections,
                                  workers=args.workers),
            score_changes=not args.no_change_scoring,
        )
        res = process_file(args.file, args.map_id, store, models, cfg, adjudicator=adjudicator, force=args.force)
        if res.skipped_identical:
            print(f"Identical to stored version {res.snapshot.version}; no changes.")
            return 0
        q = res.snapshot.quality
        print(f"Mapping confidence {q.mapping_confidence:.3f} ({'passed' if q.passed else 'below threshold'}, "
              f"{q.corrections_used} correction round(s)); {len(res.snapshot.entities)} entities, "
              f"{len(res.snapshot.relations)} relations.")
        if res.changeset is None:
            print(f"Stored baseline v{res.snapshot.version}.")
        else:
            n = sum(not c.rejected for c in res.changeset.changes)
            print(f"Stored v{res.snapshot.version}: {n} changes → {res.report_path}")
        _print_usage(models)
        return 0
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _print_usage(models: AgentModels) -> None:
    usage = models.meter.summary()
    if usage:
        print("\n" + usage_table(usage, models.describe()))


def _ping(models: AgentModels) -> int:
    """One small image + tool request per distinct (model, effort), with the same settings the agents use."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 80), "white")
    ImageDraw.Draw(img).text((20, 30), "Gate 12", fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    spec = ToolSpec("answer", "Report the text you read.", {
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    msg = UserTurn(parts=[ImagePart(buf.getvalue()), TextPart("Read the text in the image and call `answer`.")])
    seen, failed = set(), 0
    for agent in settings.AGENTS:
        vlm = getattr(models, agent)
        if vlm is None or vlm.describe() in seen:
            continue
        seen.add(vlm.describe())
        try:
            turn = vlm.complete("You read airport signage.", [msg], [spec])
            got = turn.tool_calls[0].arguments.get("text") if turn.tool_calls else turn.text.strip()
            print(f"  ok    {vlm.describe():<36} read {got!r}")
        except LLMError as e:
            failed += 1
            print(f"  FAIL  {vlm.describe():<36} {e}")
    _print_usage(models)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
