"""Token prices in USD per million tokens, used to report what each agent costs.

Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-11-10). Thinking tokens are
billed as output and are included in ``output_tokens``. Cache writes are priced at the 5-minute rate, which is
what the adapter's ``cache_control={"type": "ephemeral"}`` uses. Models not listed here report no cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    cache_write: float
    cache_read: float
    output: float
    long_prompt_over: int | None = None  # prompts over this many tokens are billed at ``long``
    long: Price | None = None


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.125, 0.01, 0.50, long_prompt_over=100_000,
                              long=Price(0.50, 0.625, 0.05, 2.50)),
    "claude-haiku-4-5": Price(1.00, 1.25, 0.10, 5.00),
    "claude-sonnet-5-5": Price(2.00, 2.50, 0.20, 10.00),
    "claude-sonnet-5": Price(2.00, 2.50, 0.20, 10.00),
    "claude-opus-5-5": Price(4.00, 5.00, 0.20, 20.00),
    "claude-opus-5": Price(5.00, 6.25, 0.50, 25.00),
    "claude-fable-5-1": Price(10.00, 12.50, 0.25, 50.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0,
             cache_write: int = 0) -> float | None:
    """Cost of one request, or None if the model's price is unknown."""
    p = PRICES.get(model)
    if p is None:
        return None
    if p.long and p.long_prompt_over and input_tokens + cache_read + cache_write > p.long_prompt_over:
        p = p.long
    return (input_tokens * p.input + output_tokens * p.output
            + cache_read * p.cache_read + cache_write * p.cache_write) / 1e6
