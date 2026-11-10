"""Provider-agnostic vision-language model client.

``VLMClient(provider=..., model=...)`` picks a wire adapter for the provider's
API. Each adapter translates the neutral types in :mod:`mapdiff.llm.base` to
and from one API protocol. To add a provider, write an adapter with the same
``complete`` signature and add it to ``_ADAPTERS`` and ``DEFAULT_MODELS``.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from mapdiff.llm.base import (
    AssistantTurn,
    ImagePart,
    LLMError,
    Message,
    Part,
    TextPart,
    ToolCall,
    ToolSpec,
    UserTurn,
)
from mapdiff.llm.usage import UsageMeter

DEFAULT_MODELS = {"claude": "claude-haiku-5-5"}
log = logging.getLogger(__name__)


class VLMClient:
    """The single entry point agents use to call a vision-language model.

    ``agent`` names the agent this client serves; with a ``meter``, every request's tokens, cost and
    duration are recorded under that name.
    """

    def __init__(self, provider: str = "claude", model: str | None = None, *, agent: str = "default",
                 meter: UsageMeter | None = None, **options: Any):
        try:
            adapter_cls = _ADAPTERS[provider]
        except KeyError:
            raise ValueError(f"Unknown VLM provider {provider!r}; available: {sorted(_ADAPTERS)}") from None
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.agent = agent
        self.meter = meter
        self.options = {k: v for k, v in options.items() if v is not None}
        self._adapter = adapter_cls(self.model, **self.options)

    def describe(self) -> str:
        effort = self.options.get("effort")
        return f"{self.provider}/{self.model}" + (f"@{effort}" if effort else "")

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        start = time.time()
        turn = self._adapter.complete(system, messages, tools)
        if self.meter is not None:
            self.meter.record(self.agent, turn.model or self.model, turn.usage, start, time.time())
        return turn


class MessagesApiAdapter:
    """Adapter for the Messages API (``POST /v1/messages``) with vision and client tools."""

    _STOP_REASONS = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal"}

    def __init__(self, model: str, effort: str = "high", max_tokens: int = 64000, server_fallback: bool = True,
                 retries: int = 4):
        import anthropic

        self._sdk = anthropic
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.server_fallback = server_fallback
        self.retries = retries
        # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`.
        self.client = anthropic.Anthropic()

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[self._to_wire(m) for m in messages],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # history is append-only, so the prefix caches
        )
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "eager_input_streaming": True,
                }
                for t in tools
            ]
        if self.server_fallback:
            # On a safety-classifier decline, re-run server-side on the recommended fallback model.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        msg = self._stream_with_retry(kwargs)

        text = "".join(b.text for b in msg.content if b.type == "text")
        calls = []
        if msg.stop_reason == "tool_use":
            calls = [ToolCall(b.id, b.name, b.input) for b in msg.content if b.type == "tool_use"]
        usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cache_read_input_tokens": msg.usage.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": msg.usage.cache_creation_input_tokens or 0,
        }
        return AssistantTurn(
            text=text,
            tool_calls=calls,
            stop_reason=self._STOP_REASONS.get(msg.stop_reason, "other"),
            raw=msg.content,
            usage=usage,
            model=getattr(msg, "model", None),
        )

    def _stream_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Stream one request. The SDK retries failures before the stream starts; this also retries a stream
        that breaks midway (dropped connection, overloaded/server error event) with exponential backoff."""
        for attempt in range(self.retries + 1):
            try:
                with self.client.beta.messages.stream(**kwargs) as stream:
                    return stream.get_final_message()
            except self._sdk.APIStatusError as e:
                if e.status_code not in (408, 409, 429) and e.status_code < 500 or attempt == self.retries:
                    raise LLMError(f"Model API error {e.status_code}: {e.message}") from e
                reason = f"status {e.status_code}"
            except Exception as e:
                transient = isinstance(e, self._sdk.APIConnectionError) or type(e).__module__.split(".")[0] in (
                    "httpx", "httpx2", "httpcore")
                if not transient:
                    raise
                if attempt == self.retries:
                    raise LLMError(f"Could not reach the model API: {e}") from e
                reason = type(e).__name__
            delay = 2 ** (attempt + 1)
            log.warning("model request failed (%s); retry %d/%d in %ds", reason, attempt + 1, self.retries, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _to_wire(m: Message) -> dict[str, Any]:
        if not isinstance(m, UserTurn):
            # Replay the exact content blocks (incl. thinking) the API returned.
            return {"role": "assistant", "content": m.raw}
        content: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": [_wire_part(p) for p in r.content],
                "is_error": r.is_error,
            }
            for r in m.tool_results
        ]
        content += [_wire_part(p) for p in m.parts]
        return {"role": "user", "content": content}


def _wire_part(p: Part) -> dict[str, Any]:
    if isinstance(p, TextPart):
        return {"type": "text", "text": p.text}
    if isinstance(p, ImagePart):
        data = base64.standard_b64encode(p.data).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": p.media_type, "data": data}}
    raise TypeError(f"Unknown part type: {type(p).__name__}")


_ADAPTERS: dict[str, type] = {
    "claude": MessagesApiAdapter,
}
