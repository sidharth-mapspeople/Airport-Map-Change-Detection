"""Offline scripted VLM: routes each call to a handler chosen by the tools on offer."""

from __future__ import annotations

from typing import Callable

import pytest
from PIL import Image

from mapdiff.ingest import Page, TextSpan
from mapdiff.llm.base import AssistantTurn, ImagePart, TextPart, ToolCall
from mapdiff.models import BBox

@pytest.fixture(autouse=True)
def _icon_library(tmp_path, monkeypatch):
    """Tests never read or write the project's learned icon library."""
    monkeypatch.setenv("MAPDIFF_ICON_LIBRARY", str(tmp_path / "airport-icons"))


Handler = Callable[[list, set, bool], list[tuple[str, dict]]]  # (first-message parts, tool names, done) -> calls


class ScriptedVLM:
    provider, model = "scripted", "none"

    def __init__(self, **handlers: Handler):
        self.handlers = handlers
        self.calls: dict[str, int] = {}
        self.images_seen = 0

    def complete(self, system, messages, tools):
        names = {t.name for t in tools}
        phase = self._phase(names, messages[0].parts)
        self.calls[phase] = self.calls.get(phase, 0) + 1
        if len(messages) == 1:
            self.images_seen += sum(isinstance(p, ImagePart) for p in messages[0].parts)
        calls = self.handlers[phase](messages[0].parts, names, len(messages) > 1)
        return AssistantTurn("", [ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)], "tool_use")

    @staticmethod
    def _phase(names: set, parts: list) -> str:
        text = " ".join(p.text for p in parts if isinstance(p, TextPart))
        if "report_verdicts" in names:
            return "change_review" if "Change 1" in text else "entity_review"
        if "finish_layout" in names:
            return "layout"
        if "finish_tile" in names:
            return "tile"
        if "finish_verification" in names:
            return "verify"
        if "finish_correction" in names:
            return "correction"
        raise AssertionError(f"unknown phase for tools {names}")


def item_count(parts: list, prefix: str) -> int:
    return sum(isinstance(p, TextPart) and p.text.startswith(prefix) for p in parts)


def blank_page(index: int = 0, spans: list[tuple[str, float, float]] = ()) -> Page:
    """A white page with optional text-layer lines centred at (x, y)."""
    return Page(index, Image.new("RGB", (800, 600), "white"), "pdf" if spans else "image",
                [TextSpan(t, BBox(x0=x - 0.02, y0=y - 0.005, x1=x + 0.02, y1=y + 0.005)) for t, x, y in spans])
