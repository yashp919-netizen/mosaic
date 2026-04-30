"""Shared utilities used across Mosaic agents."""

from __future__ import annotations

import time
from pathlib import Path


def llm_call_with_retry(fn, *args, max_retries: int = 4, base_wait: float = 15.0, **kwargs):
    """Call fn(*args, **kwargs), retrying on 429 RESOURCE_EXHAUSTED with backoff.

    Waits base_wait * (attempt+1) seconds between retries (15s, 30s, 45s, 60s).
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                last_exc = exc
                wait = base_wait * (attempt + 1)
                time.sleep(wait)
            else:
                raise
    raise last_exc  # type: ignore[misc]


def detect_delimiter(csv_path: Path) -> str:
    """Sniff CSV delimiter from first 5 lines. Returns ',' or ';'."""
    semicolons = 0
    commas = 0
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            semicolons += line.count(";")
            commas += line.count(",")
    return ";" if semicolons > commas else ","
