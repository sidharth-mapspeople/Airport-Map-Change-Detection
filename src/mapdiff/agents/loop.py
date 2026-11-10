"""Provider-neutral tool-use loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from mapdiff.llm.base import VLM, AssistantTurn, LLMError, Message, Part, TextPart, ToolResult, ToolSpec, UserTurn

log = logging.getLogger(__name__)


class ToolInputError(ValueError):
    """Raised by a tool handler for bad arguments; reported back to the model so it can retry."""


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], list[Part] | str]
    ends_run: bool = False


def array_tool(name: str, description: str, key: str, item: dict[str, Any],
               handler: Callable[[dict[str, Any]], list[Part] | str], ends_run: bool = False) -> Tool:
    """A tool whose single argument ``key`` is a list of ``item`` objects (the batch-record pattern)."""
    schema = {"type": "object", "properties": {key: {"type": "array", "items": item}}, "required": [key]}
    return Tool(ToolSpec(name, description, schema), handler, ends_run)


@dataclass
class LoopResult:
    finished: bool
    turns: int
    final_text: str


def run_tool_loop(
    vlm: VLM,
    system: str,
    first_message: list[Part],
    tools: list[Tool],
    max_turns: int = 40,
    label: str = "agent",
) -> LoopResult:
    """Drive the model until it calls a tool with ``ends_run`` or stops on its own.

    History is append-only so provider-side prompt caching and reasoning replay stay valid.
    """
    by_name = {t.spec.name: t for t in tools}
    specs = [t.spec for t in tools]
    messages: list[Message] = [UserTurn(parts=list(first_message))]
    last: AssistantTurn | None = None

    for turn in range(1, max_turns + 1):
        last = vlm.complete(system, messages, specs)
        messages.append(last)

        if last.stop_reason == "refusal":
            raise LLMError(f"[{label}] model declined the request")
        if last.stop_reason == "max_tokens":
            log.warning("[%s] hit max_tokens on turn %d; asking it to continue more briefly", label, turn)
            messages.append(UserTurn(parts=[TextPart("Your last reply was cut off. Continue, using smaller tool calls.")]))
            continue
        if not last.tool_calls:
            if last.stop_reason == "end_turn":
                return LoopResult(False, turn, last.text)
            messages.append(UserTurn(parts=[TextPart("Continue with the task using the tools.")]))
            continue

        results, finished = [], False
        for call in last.tool_calls:
            tool = by_name.get(call.name)
            log.info("[%s] turn %d -> %s %s", label, turn, call.name, _short(call.arguments))
            if tool is None:
                results.append(ToolResult(call.id, [TextPart(f"Unknown tool {call.name!r}")], is_error=True))
                continue
            try:
                out = tool.handler(call.arguments)
                content = [TextPart(out)] if isinstance(out, str) else out
                results.append(ToolResult(call.id, content))
                finished |= tool.ends_run
            except (ToolInputError, ValidationError, KeyError, TypeError, ValueError) as e:
                results.append(ToolResult(call.id, [TextPart(f"Error: {e}")], is_error=True))
        messages.append(UserTurn(tool_results=results))
        if finished:
            return LoopResult(True, turn, last.text)

    log.warning("[%s] stopped after max_turns=%d", label, max_turns)
    return LoopResult(False, max_turns, last.text if last else "")


def _short(args: dict[str, Any], limit: int = 160) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"
