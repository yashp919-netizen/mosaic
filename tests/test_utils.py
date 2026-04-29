"""Tests for src/mosaic/utils.py shared utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.utils import detect_delimiter

_SYNTH = Path(__file__).parent.parent / "data" / "synth"


def test_detect_delimiter_uk_comma():
    assert detect_delimiter(_SYNTH / "market_uk.csv") == ","


def test_detect_delimiter_br_semicolon():
    assert detect_delimiter(_SYNTH / "market_br.csv") == ";"
