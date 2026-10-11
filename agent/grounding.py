"""Check that every number in LLM-written text exists in the facts it was given.

Used for sales insights and competitor insights: the LLM may round ($1.4M for 1395310.17) but
may not compute or invent figures.
"""
from __future__ import annotations

import re

_NUM_RE = re.compile(r"\$?(\d[\d,]*(?:\.\d+)?)\s*([kKmM](?![a-z]))?")
SMALL_COUNT = 12  # "top 3", "5 channels", "two-step" are always allowed


def numbers_in(text: str) -> list[tuple[str, float]]:
    return [(m.group(0).strip(), float(m.group(1).replace(",", ""))) for m in _NUM_RE.finditer(text)]


def fact_numbers(facts) -> set[float]:
    """All numbers in a nested facts structure, including those in keys and strings, plus k/M scalings."""
    out: set[float] = set()

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(k)  # keys carry numbers too, e.g. "skus_making_80pct_of_revenue"
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.update({float(v), abs(float(v)), float(v) / 1e3, float(v) / 1e6})
        elif isinstance(v, str):
            out.update(n for _, n in numbers_in(v))

    walk(facts)
    return out


def ungrounded(text: str, facts) -> list[str]:
    known = fact_numbers(facts)
    return [
        token for token, value in numbers_in(text)
        if not (value <= SMALL_COUNT and value.is_integer()) and not any(_close(value, k) for k in known)
    ]


def _close(a: float, b: float) -> bool:
    """Allow rounding (14% for 14.4, $1.4M for 1.395M) but not new figures."""
    if b == 0:
        return a == 0
    rounded_to_int = a.is_integer() and abs(a - b) < 0.5
    return rounded_to_int or abs(a - b) < 0.051 or abs(a - b) / abs(b) <= 0.015
