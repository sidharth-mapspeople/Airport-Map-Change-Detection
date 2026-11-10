"""Run independent model calls concurrently (they are I/O-bound, so threads suffice)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    """``[fn(x) for x in items]`` with up to ``workers`` running at once; order is preserved."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(fn, items))
