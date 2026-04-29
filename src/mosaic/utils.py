"""Shared utilities used across Mosaic agents."""

from __future__ import annotations

from pathlib import Path


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
