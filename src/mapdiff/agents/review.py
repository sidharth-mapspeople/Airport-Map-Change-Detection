"""Batched, independent model review shared by the confidence agents.

Items are reviewed in small batches (in parallel). Each batch is one short tool run in which the reviewer reports a
verdict per numbered item through ``report_verdicts``. A verdict converts to a score: positive → p, negative → 1 − p,
unsure or missing → 0.5, where p is the reviewer's stated confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from mapdiff.agents.loop import array_tool, run_tool_loop
from mapdiff.llm.base import VLM, Part
from mapdiff.parallel import parallel_map

T = TypeVar("T")


@dataclass(frozen=True)
class VerdictSchema:
    key: str  # name of the item-number field, e.g. "item" or "change"
    positive: str
    negative: str
    fields: dict[str, Any]  # extra per-verdict fields, e.g. issue / explanation
    required: tuple[str, ...] = ()
    unsure: str = "unsure"

    @property
    def verdicts(self) -> tuple[str, str, str]:
        return self.positive, self.negative, self.unsure

    def score(self, v: dict[str, Any] | None) -> tuple[float, float]:
        """(score, reviewer confidence p) for a verdict, or (0.5, 0.5) if none was returned."""
        if v is None:
            return 0.5, 0.5
        p = float(min(1.0, max(0.0, v.get("confidence", 0.5))))
        return {self.positive: p, self.negative: 1 - p}.get(v["verdict"], 0.5), p


def chunked(items: Sequence[T], size: int) -> list[list[T]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def review_batches(vlm: VLM, system: str, batches: list[list[T]], build_parts: Callable[[list[T]], list[Part]],
                   schema: VerdictSchema, label: str, workers: int) -> dict[int, dict[str, Any]]:
    """Review every batch; return verdicts keyed by ``id()`` of the reviewed item."""

    def run(batch: list[T]) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}

        def report(a: dict[str, Any]) -> str:
            for v in a["verdicts"]:
                k = int(v[schema.key])
                if 1 <= k <= len(batch) and v.get("verdict") in schema.verdicts:
                    out[id(batch[k - 1])] = v
            return "Recorded."

        item = {"type": "object",
                "properties": {schema.key: {"type": "integer"},
                               "verdict": {"type": "string", "enum": list(schema.verdicts)},
                               "confidence": {"type": "number"}, **schema.fields},
                "required": [schema.key, "verdict", "confidence", *schema.required]}
        tool = array_tool("report_verdicts", f"Report a verdict for every {schema.key}.", "verdicts", item, report,
                          ends_run=True)
        run_tool_loop(vlm, system, build_parts(batch), [tool], max_turns=3, label=label)
        return out

    verdicts: dict[int, dict[str, Any]] = {}
    for part in parallel_map(run, batches, workers):
        verdicts.update(part)
    return verdicts
