"""Provider-neutral types for a tool-using vision LLM.

Agents only talk to the :class:`VLM` protocol (implemented by
:class:`mapdiff.llm.vlm_client.VLMClient`). Supporting another provider means
adding a wire adapter that translates these types to and from its API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/png"


Part = TextPart | ImagePart


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    content: list[Part]
    is_error: bool = False


@dataclass
class UserTurn:
    parts: list[Part] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    raw: Any = None  # provider-native content, replayed verbatim on the next request
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None  # model that actually answered (may differ after a server-side fallback)


Message = UserTurn | AssistantTurn


class LLMError(RuntimeError):
    pass


class VLM(Protocol):
    provider: str
    model: str

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """Run one model turn. Must never mutate ``messages``."""
        ...
