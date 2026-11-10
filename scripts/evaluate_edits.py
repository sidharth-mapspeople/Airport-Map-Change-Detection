"""Evaluate the pipeline on Test_files_edited/: accuracy, cost and time per agent.

Cost-aware by design:
  * The run starts with a smoke pair (one edit case + one restyle case of the same map). If it fails the gate
    (--gate-recall), the remaining cases are skipped instead of paying for the full set.
  * Understood maps are stored in eval_data/<name>/ and reused across runs and cases; their recorded usage still
    counts toward per-map cost, but not toward this run's spend. Use a new --name after changing the pipeline,
    so maps are understood again with the new code.
  * --budget-usd stops the run once actual spend reaches it; results so far are still written.

Run:  PYTHONNOUSERSITE=1 python scripts/evaluate_edits.py [--name haiku-v2] [--cases ...] [-v]
Results: eval_results/<timestamp>/summary.md, summary.json and visual/index.html, which opens in the browser
when the run ends (--no-visual / --no-open to skip).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from mapdiff import settings
from mapdiff.agents.mapping_confidence import QualityConfig
from mapdiff.agents.reporting import usage_table
from mapdiff.agents.understanding import UnderstandingConfig
from mapdiff.evaluation import EVAL_DATA_DIR, EVAL_RESULTS_DIR, MANIFEST, CaseScore, describe, eval_map_id, load_manifest, pct, score_case
from mapdiff.ingest import sha256_file
from mapdiff.llm import LLMError
from mapdiff.llm.usage import cost_of, merge
from mapdiff.models import AgentUsage
from mapdiff.pipeline import PipelineConfig, process_file, rediff
from mapdiff.settings import AgentModels
from mapdiff.store import MapStore
from mapdiff.visual_report import build as build_visual
from mapdiff.visual_report import show

ROOT = settings.PROJECT_ROOT
SMOKE = ["bru-departure-edits", "bru-departure-restyle"]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class MapRun:
    """One file understood + quality-checked (a baseline or a variant)."""

    file: str
    cached: bool
    seconds: float
    usage: list[AgentUsage]
    entities: int
    mapping_confidence: float | None
    corrections: int


@dataclass
class CaseResult:
    case: dict
    score: CaseScore
    report: str
    variant: MapRun
    scoring_usage: list[AgentUsage]
    scoring_seconds: float
    mapping_conf: tuple[float | None, float | None]
    map_date: str | None


@dataclass
class RunResult:
    name: str
    agents: dict[str, str]
    cases: list[CaseResult] = field(default_factory=list)
    map_runs: list[MapRun] = field(default_factory=list)
    spent_usd: float = 0.0
    stopped: str = ""
    errors: list[str] = field(default_factory=list)

    def score(self, kind: str | None = None) -> CaseScore:
        """All cases (or one kind: 'edit' / 'restyle') pooled into one score."""
        return CaseScore.combine([c.score for c in self.cases if kind in (None, c.case["kind"])])

    def recall(self) -> float | None:
        return self.score("edit").recall

    def precision(self) -> float | None:
        return self.score().precision

    def false_positives(self) -> int:
        return len(self.score().false_positives)

    def restyle_false_positives(self) -> int:
        return len(self.score("restyle").false_positives)

    def all_usage(self) -> list[AgentUsage]:
        return merge(*(r.usage for r in self.map_runs), *(c.scoring_usage for c in self.cases))


def main() -> int:
    settings.load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--data-dir", default=str(EVAL_DATA_DIR))
    ap.add_argument("--out", default=str(EVAL_RESULTS_DIR))
    ap.add_argument("--no-visual", dest="visual", action="store_false", help="skip the visual results page")
    ap.add_argument("--no-open", dest="open", action="store_false", help="build the page but don't open a browser")
    ap.add_argument("--name", default=settings.MODEL,
                    help="name of this pipeline version; its maps are stored in <data-dir>/<name>")
    ap.add_argument("--cases", nargs="*", help="case ids to run (default: all)")
    ap.add_argument("--smoke", nargs="*", default=SMOKE, help="cases run first as a gate (empty: no gate)")
    ap.add_argument("--gate-recall", type=float, default=0.4, help="min smoke recall to run the other cases")
    ap.add_argument("--budget-usd", type=float, default=3.0, help="stop once this run has spent this much")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    settings.configure_logging(args.verbose)

    cases = load_manifest(args.manifest)
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    smoke = [c for c in cases if c["id"] in (args.smoke or [])]
    rest = [c for c in cases if c not in smoke]

    models = AgentModels.from_settings()
    run = RunResult(args.name, models.describe())
    print(f"\n######## {run.name}: " + ", ".join(f"{a}={m}" for a, m in run.agents.items()))
    store = MapStore(Path(args.data_dir) / run.name)
    cfg = PipelineConfig(understanding=UnderstandingConfig(workers=args.workers),
                         quality=QualityConfig(workers=args.workers), score_changes=False)

    def spend_check() -> None:
        run.spent_usd = cost_of(models.meter.summary()) or 0.0
        if run.spent_usd >= args.budget_usd:
            raise BudgetExceeded(f"budget ${args.budget_usd:.2f} reached (spent ${run.spent_usd:.2f})")

    def attempt(case: dict) -> None:
        try:
            run.cases.append(run_case(case, store, models, cfg, run, spend_check))
        except LLMError as e:  # one failed case should not lose the rest of the run
            run.errors.append(f"{case['id']}: {e}")
            print(f"   ERROR {case['id']}: {e}")

    try:
        for case in smoke:
            attempt(case)
        if smoke and rest:
            s_recall = run.score("edit").recall
            if s_recall is not None and s_recall < args.gate_recall:
                run.stopped = f"smoke recall {s_recall:.0%} < gate {args.gate_recall:.0%}; rest skipped"
                print(f"   {run.stopped}")
        if not run.stopped:
            for case in rest:
                attempt(case)
    except BudgetExceeded as e:
        print(f"\nSTOPPED: {e}")
        run.stopped = str(e)
    run.spent_usd = cost_of(models.meter.summary()) or 0.0
    print(f"   → recall {pct(run.recall())}, false positives {run.false_positives()}")

    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(render(run, args))
    (out / "summary.json").write_text(json.dumps(to_json(run), indent=2, ensure_ascii=False))
    print(f"\nSpent this run: ${run.spent_usd:.2f}\nSummary → {out / 'summary.md'}")
    if args.visual and run.cases:
        page = build_visual(out, Path(args.data_dir))
        print(f"Visual report → {page}")
        if args.open:
            show(page)
    return 0


def run_case(case: dict, store: MapStore, models: AgentModels, cfg: PipelineConfig, pr: RunResult,
             spend_check) -> CaseResult:
    map_id = eval_map_id(case["baseline"])
    print(f"\n== [{pr.name}] {case['id']} ({case['kind']}) — map {map_id}")
    if store.versions(map_id):  # the baseline is always v1 of its eval map
        base = store.load(map_id, 1)
        if base.source_sha256 != sha256_file(ROOT / case["baseline"]):
            raise SystemExit(f"{store.root}/{map_id} v1 is not {case['baseline']}; use a fresh --data-dir")
        if not any(r.file == case["baseline"] for r in pr.map_runs):
            pr.map_runs.append(_map_run(case["baseline"], base, cached=True, seconds=0.0))
    else:
        t = time.perf_counter()
        res = process_file(ROOT / case["baseline"], map_id, store, models, cfg)
        pr.map_runs.append(_map_run(case["baseline"], res.snapshot, cached=False, seconds=time.perf_counter() - t))
        spend_check()

    # A variant already understood in this run's store (e.g. by an interrupted run) is reused, not re-paid.
    var_sha = sha256_file(ROOT / case["variant"])
    var_v = next((v for v in store.versions(map_id)[1:] if store.load(map_id, v).source_sha256 == var_sha), None)
    if var_v:
        variant = _map_run(case["variant"], store.load(map_id, var_v), cached=True, seconds=0.0)
    else:
        t = time.perf_counter()
        var = process_file(ROOT / case["variant"], map_id, store, models, cfg, force=True)
        var_v = var.snapshot.version
        variant = _map_run(case["variant"], var.snapshot, cached=False, seconds=time.perf_counter() - t)
    pr.map_runs.append(variant)
    spend_check()

    t = time.perf_counter()
    cs, report = rediff(map_id, store, 1, var_v, models, PipelineConfig())
    scoring_seconds = time.perf_counter() - t
    spend_check()

    score = score_case(case["id"], case["expected"], cs)
    old, new = store.load(map_id, 1), store.load(map_id, var_v)
    print(f"   expected {score.expected}, found {len(score.true_positives)}, missed {len(score.missed)}, "
          f"false positives {len(score.false_positives)}; variant ${_cost(variant.usage)} "
          f"in {variant.seconds:.0f}s → {report}")
    return CaseResult(case, score, str(report), variant, cs.usage, scoring_seconds,
                      (old.quality.mapping_confidence if old.quality else None,
                       new.quality.mapping_confidence if new.quality else None), new.map_date)


def _map_run(file: str, snap, cached: bool, seconds: float) -> MapRun:
    q = snap.quality
    if cached:  # time is not stored on the snapshot; use the summed per-agent wall time instead
        seconds = sum(u.wall_seconds for u in snap.usage)
    return MapRun(file, cached, seconds, snap.usage, len(snap.entities),
                  q.mapping_confidence if q else None, q.corrections_used if q else 0)


def _cost(usage: list[AgentUsage]) -> str:
    c = cost_of(usage)
    return "n/a" if c is None else f"{c:.4f}"


def _avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def render(run: RunResult, args) -> str:
    maps = [r for r in run.map_runs if r.usage]
    cost_map = _avg([cost_of(r.usage) or 0.0 for r in maps])
    time_map = _avg([r.seconds for r in maps])
    conf = _avg([r.mapping_confidence for r in run.map_runs if r.mapping_confidence is not None])
    cost_diff = _avg([cost_of(c.scoring_usage) or 0.0 for c in run.cases if c.scoring_usage])
    lines = [
        f"# Evaluation: {run.name}", "",
        "Models: " + ", ".join(f"{a} `{m}`" for a, m in run.agents.items()), "",
        f"Spent this run: **${run.spent_usd:.2f}** (budget ${args.budget_usd:.2f})."
        + (f" Stopped: {run.stopped}." if run.stopped else ""), "",
        "| Cases | Recall | Precision | False pos. (restyle) | Mapping conf. | Cost / map | Time / map | Cost / diff scoring |",
        "|---|---|---|---|---|---|---|---|",
        f"| {len(run.cases)} | {pct(run.recall())} | {pct(run.precision())} "
        f"| {run.false_positives()} ({run.restyle_false_positives()}) | {'-' if conf is None else f'{conf:.3f}'} "
        f"| {'-' if cost_map is None else f'${cost_map:.3f}'} | {'-' if time_map is None else f'{time_map:.0f}s'} "
        f"| {'-' if cost_diff is None else f'${cost_diff:.4f}'} |",
        "", "*Cost / map* is understanding + mapping confidence + correction for one file (cached maps included "
        "from their stored usage). *Time / map* is end-to-end for fresh runs.", "",
        "## Cost and time per agent (all maps and diffs)", "",
    ]
    usage = run.all_usage()
    lines += [usage_table(usage, run.agents) if usage else "No model calls recorded.", ""]
    tot = cost_of(usage)
    if tot:
        lines += ["Share of cost: " + ", ".join(
            f"{u.agent} {100 * (u.cost_usd or 0) / tot:.0f}%" for u in sorted(usage, key=lambda u: -(u.cost_usd or 0))), ""]
    lines += [f"- ⛔ {e}" for e in run.errors] + ([""] if run.errors else [])
    lines += ["## Maps processed", "", "| File | Cached | Entities | Mapping conf. | Corrections | Cost | Time |",
              "|---|---|---|---|---|---|---|"]
    for r in run.map_runs:
        lines.append(f"| {r.file} | {'yes' if r.cached else 'no'} | {r.entities} "
                     f"| {'-' if r.mapping_confidence is None else f'{r.mapping_confidence:.3f}'} | {r.corrections} "
                     f"| ${_cost(r.usage)} | {r.seconds:.0f}s |")
    lines += ["", "## Cases", "",
              "| Case | Kind | Expected | Found | Missed | False pos. | Wrongly rejected | Mapping conf. (old → new) |",
              "|---|---|---|---|---|---|---|---|"]
    for c in run.cases:
        s, (mo, mn) = c.score, c.mapping_conf
        conf = f"{mo:.2f} → {mn:.2f}" if mo is not None and mn is not None else "-"
        lines.append(f"| {s.case_id} | {c.case['kind']} | {s.expected} | {len(s.true_positives)} | {len(s.missed)} "
                     f"| {len(s.false_positives)} | {len(s.wrongly_rejected)} | {conf} |")
    for c in run.cases:
        s = c.score
        lines += ["", f"### {s.case_id}", "", c.case["description"], "", f"Report: `{c.report}`", ""]
        lines += [f"- ✅ {describe(e)} — reported as: {summary}" for e, summary in s.true_positives]
        lines += [f"- ❌ missed: {describe(e)}" for e in s.missed]
        lines += [f"- 🚫 wrongly rejected: {summary}" for _, summary in s.wrongly_rejected]
        lines += [f"- ⚠️ false positive: {fp}" for fp in s.false_positives]
        if "expected_map_date" in c.case:
            ok = (c.map_date or "").strip() == c.case["expected_map_date"]
            lines.append(f"- {'✅' if ok else '❌'} printed date read as {c.map_date!r} "
                         f"(expected {c.case['expected_map_date']!r})")
    return "\n".join(lines) + "\n"


def to_json(run: RunResult) -> dict:
    return dict(
        name=run.name, agents=run.agents, recall=run.recall(), precision=run.precision(),
        false_positives=run.false_positives(), spent_usd=round(run.spent_usd, 4), stopped=run.stopped,
        errors=run.errors,
        usage=[u.model_dump() for u in run.all_usage()],
        maps=[dict(file=r.file, cached=r.cached, seconds=round(r.seconds, 1), entities=r.entities,
                   mapping_confidence=r.mapping_confidence, corrections=r.corrections,
                   usage=[u.model_dump() for u in r.usage]) for r in run.map_runs],
        cases=[dict(case=c.case["id"], kind=c.case["kind"], expected=c.score.expected,
                    true_positives=len(c.score.true_positives), missed=[describe(e) for e in c.score.missed],
                    wrongly_rejected=[s for _, s in c.score.wrongly_rejected],
                    false_positives=c.score.false_positives, recall=c.score.recall, precision=c.score.precision,
                    mapping_confidence=c.mapping_conf, report=c.report,
                    change_scoring_usage=[u.model_dump() for u in c.scoring_usage],
                    change_scoring_seconds=round(c.scoring_seconds, 1)) for c in run.cases])


if __name__ == "__main__":
    raise SystemExit(main())
