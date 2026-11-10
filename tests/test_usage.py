"""Per-agent model routing, usage metering and pricing."""

import os

import pytest
from conftest import ScriptedVLM
from test_understanding import AUCKLAND, SCRIPT

from mapdiff import settings
from mapdiff.agents.mapping_confidence import QualityConfig
from mapdiff.agents.understanding import UnderstandingConfig
from mapdiff.llm.pricing import cost_usd
from mapdiff.llm.usage import UsageMeter, merge, total
from mapdiff.pipeline import PipelineConfig, process_file
from mapdiff.settings import AgentModels
from mapdiff.store import MapStore


class MeteredScriptedVLM(ScriptedVLM):
    """Scripted VLM that reports fixed token usage to a meter, like VLMClient does."""

    def __init__(self, agent, model, meter, **handlers):
        super().__init__(**handlers)
        self.agent, self.model, self.meter = agent, model, meter

    def complete(self, system, messages, tools):
        turn = super().complete(system, messages, tools)
        self.meter.record(self.agent, self.model, {"input_tokens": 1000, "output_tokens": 100}, 0.0, 1.0)
        return turn


def test_pricing_including_long_prompt_tier():
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert cost_usd("claude-sonnet-5-5", 0, 1_000_000, cache_read=1_000_000) == pytest.approx(10.10)
    assert cost_usd("claude-haiku-5-5", 100_000, 1000) == pytest.approx(0.0105)
    # Over 100k prompt tokens (cache reads count too), the whole request is billed at the higher tier.
    assert cost_usd("claude-haiku-5-5", 50_000, 1000, cache_read=60_000) == pytest.approx(0.0305)
    assert cost_usd("some-unknown-model", 10, 10) is None


def test_meter_wall_time_counts_parallel_calls_once():
    m = UsageMeter()
    m.record("a", "claude-haiku-5-5", {"input_tokens": 10, "output_tokens": 5}, 0.0, 10.0)
    m.record("a", "claude-haiku-5-5", {"input_tokens": 10, "output_tokens": 5}, 5.0, 12.0)  # overlaps
    m.record("a", "claude-haiku-5-5", {"input_tokens": 10, "output_tokens": 5}, 20.0, 21.0)
    (u,) = m.summary()
    assert u.calls == 3 and u.input_tokens == 30
    assert u.api_seconds == 18.0 and u.wall_seconds == 13.0
    mark = m.mark()
    m.record("b", "unknown", {"input_tokens": 1}, 0.0, 1.0)
    assert [x.agent for x in m.summary(mark)] == ["b"] and m.summary(mark)[0].cost_usd is None
    assert total(m.summary()).cost_usd is None  # an unknown price makes the total unknown, not understated
    assert merge(m.summary(0, mark), m.summary(0, mark))[0].calls == 6


def test_haiku_defaults_and_env_overrides(monkeypatch):
    monkeypatch.setenv("MAPDIFF_CHANGE_CONFIDENCE_EFFORT", "high")
    s = settings.agent_settings()
    assert set(s) == set(settings.AGENTS) and {m for m, _ in s.values()} == {"claude-haiku-5-5"}
    assert s["understanding"] == ("claude-haiku-5-5", "medium") and s["change_confidence"][1] == "high"
    assert {m for m, _ in settings.agent_settings(model="x").values()} == {"x"}


def test_load_env_does_not_override(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("# c\nMAPDIFF_T1=from_file\nexport MAPDIFF_T2='quoted'\nMAPDIFF_T3=file\n")
    monkeypatch.setenv("MAPDIFF_T3", "env")
    monkeypatch.delenv("MAPDIFF_T1", raising=False)
    monkeypatch.delenv("MAPDIFF_T2", raising=False)
    assert settings.load_env(tmp_path / ".env") == tmp_path / ".env"
    assert (os.environ["MAPDIFF_T1"], os.environ["MAPDIFF_T2"], os.environ["MAPDIFF_T3"]) == ("from_file", "quoted", "env")


@pytest.mark.skipif(not AUCKLAND.exists(), reason="Test_files not present")
def test_each_agent_uses_its_own_model_and_usage_is_stored(tmp_path):
    meter = UsageMeter()
    reader = MeteredScriptedVLM("understanding", "claude-sonnet-5-5", meter, **SCRIPT)
    checker = MeteredScriptedVLM("mapping_confidence", "claude-haiku-5-5", meter, entity_review=SCRIPT["entity_review"])
    unused = ScriptedVLM()  # correction and change scoring must not be called here
    models = AgentModels(reader, checker, unused, unused, None, meter)
    cfg = PipelineConfig(understanding=UnderstandingConfig(render_px=2000, tile_px=700, workers=2),
                         quality=QualityConfig(workers=2))
    snap = process_file(AUCKLAND, "akl", MapStore(tmp_path), models, cfg).snapshot

    assert "entity_review" not in reader.calls and set(checker.calls) == {"entity_review"}
    assert unused.calls == {}
    by_agent = {u.agent: u for u in snap.usage}
    assert set(by_agent) == {"understanding", "mapping_confidence"}
    assert by_agent["understanding"].calls == sum(reader.calls.values())
    assert by_agent["mapping_confidence"].cost_usd == pytest.approx(
        by_agent["mapping_confidence"].calls * cost_usd("claude-haiku-5-5", 1000, 100))
    assert snap.llm["understanding"] == "scripted/claude-sonnet-5-5"
    assert MapStore(tmp_path).load("akl", 1).usage == snap.usage  # persisted with the version
