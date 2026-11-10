"""Score a changeset against ground-truth expected changes (see Test_files_edited/manifest.json)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mapdiff.models import Change, ChangeSet, Point
from mapdiff.names import name_similarity, norm_name
from mapdiff.settings import PROJECT_ROOT
from mapdiff.taxonomy import Category, group_of

# Reported change types that satisfy an expected type.
_COMPATIBLE = {
    "replaced": {"replaced", "repurposed", "renamed"},
    "repurposed": {"repurposed", "replaced"},
    "added": {"added"},
    "removed": {"removed"},
    "moved": {"moved"},
    "relation_changed": {"relation_changed"},
    "relation_removed": {"relation_removed", "relation_changed"},
    "relation_added": {"relation_added", "relation_changed"},
}
LOCATION_TOLERANCE = 0.04
EDITS_DIR = PROJECT_ROOT / "Test_files_edited"
MANIFEST = EDITS_DIR / "manifest.json"
EVAL_DATA_DIR = PROJECT_ROOT / "eval_data"  # stored versions per pipeline version (--name), reused across runs
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"  # one timestamped folder per run


@dataclass
class CaseScore:
    case_id: str
    expected: int
    true_positives: list[tuple[dict[str, Any], str]] = field(default_factory=list)  # (expected, reported summary)
    missed: list[dict[str, Any]] = field(default_factory=list)
    wrongly_rejected: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    # (expected, matching reported change or None, "found" | "rejected" | "missed") in manifest order
    pairs: list[tuple[dict[str, Any], Change | None, str]] = field(default_factory=list)
    unmatched: list[Change] = field(default_factory=list)  # accepted changes that match nothing (false positives)

    @property
    def recall(self) -> float | None:
        return len(self.true_positives) / self.expected if self.expected else None

    @property
    def precision(self) -> float | None:
        reported = len(self.true_positives) + len(self.false_positives)
        return len(self.true_positives) / reported if reported else None

    @classmethod
    def combine(cls, scores: list[CaseScore], case_id: str = "all") -> CaseScore:
        """Pool several cases into one score, so recall/precision are computed over all of them."""
        out = cls(case_id, sum(s.expected for s in scores))
        for s in scores:
            for name in ("true_positives", "missed", "wrongly_rejected", "false_positives", "pairs", "unmatched"):
                getattr(out, name).extend(getattr(s, name))
        return out


def load_manifest(path: str | Path = MANIFEST) -> list[dict[str, Any]]:
    """Test cases from Test_files_edited/manifest.json (paths relative to the project root)."""
    return json.loads(Path(path).read_text())["cases"]


def latest_results(results_root: Path = EVAL_RESULTS_DIR) -> Path:
    """The most recent evaluation run folder (one containing summary.json)."""
    runs = sorted(p.parent for p in results_root.glob("*/summary.json"))
    if not runs:
        raise FileNotFoundError(f"No evaluation runs in {results_root}")
    return runs[-1]


def eval_map_id(baseline: str) -> str:
    """Stable map id for a baseline file, e.g. 'eval-brussels-airport-departure'."""
    return "eval-" + re.sub(r"[^a-z0-9]+", "-", Path(baseline).stem.lower()).strip("-")


def names_match(a: str | None, b: str | None) -> bool:
    na, nb = norm_name(a), norm_name(b)
    return bool(na and nb) and (name_similarity(a, b) >= 0.8 or na in nb or nb in na)


def matches(exp: dict[str, Any], c: Change) -> bool:
    if c.type.value not in _COMPATIBLE[exp["type"]]:
        return False
    if exp["type"].startswith("relation"):
        rel = c.old_relation or c.new_relation
        if not rel or not names_match(rel.subject, exp["subject"]):
            return False
        digits = lambda s: re.findall(r"\d+", s or "")
        if "old_object" in exp and (not c.old_relation or digits(c.old_relation.object) != digits(exp["old_object"])):
            return False
        if "new_object" in exp and (not c.new_relation or digits(c.new_relation.object) != digits(exp["new_object"])):
            return False
        return True
    if "old_name" in exp and not (c.old and names_match(c.old.name, exp["old_name"])):
        return False
    if "new_name" in exp and not (c.new and names_match(c.new.name, exp["new_name"])):
        return False
    ent = c.old if exp["type"] == "removed" else (c.new or c.old)
    if "category" in exp and group_of(ent.category) != group_of(Category(exp["category"])):
        return False
    if "at" in exp and ent.anchor.dist(Point(x=exp["at"][0], y=exp["at"][1])) > LOCATION_TOLERANCE:
        return False
    return True


def score_case(case_id: str, expected: list[dict[str, Any]], cs: ChangeSet) -> CaseScore:
    score = CaseScore(case_id, len(expected))
    accepted = [c for c in cs.changes if not c.rejected]
    rejected = [c for c in cs.changes if c.rejected]
    used: set[int] = set()
    for exp in expected:
        hit = next((c for c in accepted if id(c) not in used and matches(exp, c)), None)
        if hit:
            used.add(id(hit))
            score.true_positives.append((exp, hit.summary))
            score.pairs.append((exp, hit, "found"))
            continue
        bad = next((c for c in rejected if id(c) not in used and matches(exp, c)), None)
        if bad:
            used.add(id(bad))
            score.wrongly_rejected.append((exp, bad.summary))
        score.missed.append(exp)
        score.pairs.append((exp, bad, "rejected" if bad else "missed"))
    score.unmatched = [c for c in accepted if id(c) not in used]
    score.false_positives = [c.summary for c in score.unmatched]
    return score


def pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def describe(exp: dict[str, Any]) -> str:
    if exp["type"].startswith("relation"):
        return f"{exp['type']}: {exp['subject']} {exp.get('old_object', '')}→{exp.get('new_object', '')}".rstrip("→")
    names = " → ".join(n for n in (exp.get("old_name"), exp.get("new_name")) if n)
    return f"{exp['type']}: {names or exp.get('category', '?')}" + (f" at {exp['at']}" if "at" in exp else "")
