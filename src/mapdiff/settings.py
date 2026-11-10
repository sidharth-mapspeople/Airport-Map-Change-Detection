"""Which model each agent uses, read from the environment and an optional ``.env`` file.

Every agent runs on Haiku 5.5, the cheapest model, with its own effort. Override single agents with
``MAPDIFF_<AGENT>_MODEL`` / ``MAPDIFF_<AGENT>_EFFORT``, e.g. ``MAPDIFF_UNDERSTANDING_EFFORT``.
``.env`` is looked up in the current directory, then the project root; real environment variables win.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from mapdiff.agents.change_detection import Adjudicator, vlm_adjudicator
from mapdiff.llm import VLM, UsageMeter, VLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Agents that call a model. Change detection and reporting are deterministic.
AGENTS = ("understanding", "mapping_confidence", "correction", "change_confidence", "adjudicator")

MODEL = "claude-haiku-5-5"

# Effort per agent. The adjudicator answers one short yes/no question, so it needs little thinking.
EFFORTS: dict[str, str] = {
    "understanding": "medium",
    "mapping_confidence": "medium",
    "correction": "medium",
    "change_confidence": "medium",
    "adjudicator": "low",
}


def load_env(path: str | Path | None = None) -> Path | None:
    """Load KEY=VALUE lines from a .env file into os.environ without overriding. Returns the file used."""
    candidates = [Path(path)] if path else [Path.cwd() / ".env", PROJECT_ROOT / ".env"]
    for p in candidates:
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.removeprefix("export ").strip(), value.strip().strip("'\""))
            return p
    return None


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def _env(name: str, default: str | None = None) -> str | None:
    """An environment setting; unset and empty both mean ``default``."""
    return os.environ.get(name) or default


def provider() -> str:
    return _env("MAPDIFF_PROVIDER", "claude")


def agent_settings(model: str | None = None, effort: str | None = None) -> dict[str, tuple[str, str]]:
    """(model, effort) per agent: defaults, then per-agent env overrides, then ``model``/``effort`` for all."""
    out = {}
    for agent in AGENTS:
        key = f"MAPDIFF_{agent.upper()}"
        out[agent] = (model or _env(f"{key}_MODEL", MODEL), effort or _env(f"{key}_EFFORT", EFFORTS[agent]))
    return out


@dataclass
class AgentModels:
    """The model client each agent uses."""

    understanding: VLM
    mapping_confidence: VLM
    correction: VLM
    change_confidence: VLM
    adjudicator: VLM | None = None
    meter: UsageMeter = field(default_factory=UsageMeter)  # stays empty for clients that don't report usage

    @classmethod
    def single(cls, vlm: VLM) -> AgentModels:
        """One model for every agent (tests, or a deliberate single-model run)."""
        return cls(vlm, vlm, vlm, vlm, vlm, getattr(vlm, "meter", None) or UsageMeter())

    @classmethod
    def from_settings(cls, model: str | None = None, effort: str | None = None,
                      meter: UsageMeter | None = None) -> AgentModels:
        meter = meter or UsageMeter()
        prov = provider()
        clients = {agent: VLMClient(prov, m, agent=agent, meter=meter, effort=e)
                   for agent, (m, e) in agent_settings(model, effort).items()}
        return cls(**clients, meter=meter)

    def adjudicator_fn(self, enabled: bool = True) -> Adjudicator | None:
        """Name-change adjudicator backed by the adjudicator model, or None when disabled or unset."""
        return vlm_adjudicator(self.adjudicator) if enabled and self.adjudicator is not None else None

    def describe(self) -> dict[str, str]:
        return {a: (v.describe() if hasattr(v, "describe") else f"{v.provider}/{v.model}")
                for a in AGENTS if (v := getattr(self, a)) is not None}
