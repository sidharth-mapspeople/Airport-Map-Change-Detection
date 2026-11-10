"""Name normalisation shared by de-duplication and change detection."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def norm_name(name: str | None) -> str:
    """'Culto Café ' -> 'culto cafe'; '' / None -> ''."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNCT.sub(" ", s.casefold())
    return _SPACE.sub(" ", s).strip()


def norm_code(code: str | None) -> str:
    """Marker codes / legend symbols compare case- and whitespace-insensitively: ' A1' == 'a1'."""
    return (code or "").strip().casefold()


def name_similarity(a: str | None, b: str | None) -> float:
    """0..1. Both empty counts as identical (generic facilities)."""
    na, nb = norm_name(a), norm_name(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    # Spacing-insensitive comparison catches 'WHSmith' vs 'WH Smith', 'Mc Donalds' vs 'McDonalds'.
    compact = fuzz.ratio(na.replace(" ", ""), nb.replace(" ", ""))
    return max(fuzz.token_sort_ratio(na, nb), compact) / 100.0
