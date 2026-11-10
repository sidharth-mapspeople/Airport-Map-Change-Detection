"""Per-agent accounting of model calls: tokens, cost and time.

Every :class:`~mapdiff.llm.vlm_client.VLMClient` built with a meter records each request under its agent name.
Wall time per agent is the union of its request intervals, so parallel calls are not double counted.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from mapdiff.llm.pricing import cost_usd
from mapdiff.models import AgentUsage


@dataclass(frozen=True)
class CallRecord:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    start: float
    end: float
    cost_usd: float | None


class UsageMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[CallRecord] = []

    def record(self, agent: str, model: str, usage: dict[str, int], start: float, end: float) -> None:
        inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cr, cw = usage.get("cache_read_input_tokens", 0), usage.get("cache_creation_input_tokens", 0)
        rec = CallRecord(agent, model, inp, out, cr, cw, start, end, cost_usd(model, inp, out, cr, cw))
        with self._lock:
            self.records.append(rec)

    def mark(self) -> int:
        """A position to summarise from later, e.g. one pipeline run within a longer session."""
        with self._lock:
            return len(self.records)

    def summary(self, since: int = 0, until: int | None = None) -> list[AgentUsage]:
        with self._lock:
            recs = self.records[since:until]
        by_agent: dict[str, list[CallRecord]] = {}
        for r in recs:
            by_agent.setdefault(r.agent, []).append(r)
        return [_aggregate(agent, rs) for agent, rs in by_agent.items()]


def total(usages: list[AgentUsage], agent: str = "total") -> AgentUsage:
    """Add up per-agent usage. Wall time is summed, which is exact when agents run one after another."""
    costs = [u.cost_usd for u in usages]
    return AgentUsage(
        agent=agent, models=sorted({m for u in usages for m in u.models}),
        calls=sum(u.calls for u in usages),
        input_tokens=sum(u.input_tokens for u in usages), output_tokens=sum(u.output_tokens for u in usages),
        cache_read_tokens=sum(u.cache_read_tokens for u in usages),
        cache_write_tokens=sum(u.cache_write_tokens for u in usages),
        cost_usd=None if any(c is None for c in costs) else round(sum(costs), 6),
        api_seconds=round(sum(u.api_seconds for u in usages), 2),
        wall_seconds=round(sum(u.wall_seconds for u in usages), 2),
    )


def cost_of(usages: list[AgentUsage]) -> float | None:
    """Total cost of a usage list: 0 for none, None if any model's price is unknown."""
    return total(usages).cost_usd if usages else 0.0


def merge(*groups: list[AgentUsage]) -> list[AgentUsage]:
    """Combine usage lists from several runs, agent by agent."""
    by_agent: dict[str, list[AgentUsage]] = {}
    for g in groups:
        for u in g:
            by_agent.setdefault(u.agent, []).append(u)
    return [total(us, agent) for agent, us in by_agent.items()]


def _aggregate(agent: str, rs: list[CallRecord]) -> AgentUsage:
    calls = [AgentUsage(agent=agent, models=[r.model], calls=1, input_tokens=r.input_tokens,
                        output_tokens=r.output_tokens, cache_read_tokens=r.cache_read_tokens,
                        cache_write_tokens=r.cache_write_tokens, cost_usd=r.cost_usd, api_seconds=r.end - r.start)
             for r in rs]
    u = total(calls, agent)
    u.wall_seconds = round(_union_length([(r.start, r.end) for r in rs]), 2)
    return u


def _union_length(intervals: list[tuple[float, float]]) -> float:
    length, cur_start, cur_end = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_end is None or s > cur_end:
            if cur_end is not None:
                length += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    if cur_end is not None:
        length += cur_end - cur_start
    return length
