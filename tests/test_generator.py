"""Tests for data/synth/generator.py.

Covers: reproducibility, row counts, planted-issue rates, ground_truth validity.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Make the generator importable (it lives outside src/)
_SYNTH_DIR = Path(__file__).parent.parent / "data" / "synth"
if str(_SYNTH_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNTH_DIR))

from generator import generate  # noqa: E402  (after sys.path patch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(n: int, seed: int, tmp: Path) -> Path:
    """Run generator and return output dir."""
    generate(n=n, seed=seed, out_dir=tmp)
    return tmp


def _read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _load_gt(out_dir: Path) -> dict:
    with open(out_dir / "ground_truth.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Reproducibility — same seed → identical file bytes
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_identical_bytes(self, tmp_path):
        d1 = tmp_path / "run1"
        d2 = tmp_path / "run2"
        d1.mkdir(); d2.mkdir()
        generate(n=200, seed=42, out_dir=d1)
        generate(n=200, seed=42, out_dir=d2)

        for fname in ("market_uk.csv", "market_in.csv", "market_br.csv", "ground_truth.json"):
            assert (d1 / fname).read_bytes() == (d2 / fname).read_bytes(), \
                f"{fname} differs between runs with same seed"

    def test_different_seeds_differ(self, tmp_path):
        d1 = tmp_path / "s42"
        d2 = tmp_path / "s99"
        d1.mkdir(); d2.mkdir()
        generate(n=200, seed=42, out_dir=d1)
        generate(n=200, seed=99, out_dir=d2)

        uk42 = (d1 / "market_uk.csv").read_bytes()
        uk99 = (d2 / "market_uk.csv").read_bytes()
        assert uk42 != uk99, "Different seeds should produce different data"


# ---------------------------------------------------------------------------
# 2. Row counts — each CSV should have exactly n rows (header excluded)
# ---------------------------------------------------------------------------

class TestRowCounts:
    @pytest.fixture(scope="class")
    def out(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("counts")
        generate(n=500, seed=42, out_dir=d)
        return d

    def test_uk_row_count(self, out):
        rows = _read_csv(out / "market_uk.csv", ",")
        assert len(rows) == 500

    def test_in_row_count(self, out):
        rows = _read_csv(out / "market_in.csv", ",")
        assert len(rows) == 500

    def test_br_row_count(self, out):
        rows = _read_csv(out / "market_br.csv", ";")
        assert len(rows) == 500

    def test_br_semicolon_delimiter(self, out):
        # If we accidentally read with comma, row count or column count will be wrong
        rows_semi = _read_csv(out / "market_br.csv", ";")
        assert len(rows_semi[0]) == 9, "BR CSV should have 9 columns when read with ';'"


# ---------------------------------------------------------------------------
# 3. Planted-issue rates — within ±0.5 percentage points of target
# ---------------------------------------------------------------------------

class TestPlantedIssueRates:
    N = 1000  # large enough for reliable rate checks

    @pytest.fixture(scope="class")
    def gt(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("rates")
        generate(n=self.N, seed=42, out_dir=d)
        return _load_gt(d)

    def _assert_rate(self, gt: dict, issue_type: str, target_pct: float):
        issue = next(i for i in gt["planted_issues"] if i["issue_type"] == issue_type)
        actual_pct = len(issue["row_indices"]) / self.N * 100
        assert abs(actual_pct - target_pct) <= 0.5, (
            f"{issue_type}: expected ~{target_pct}%, got {actual_pct:.2f}%"
        )

    def test_duplicate_sku_rate(self, gt):
        self._assert_rate(gt, "duplicate_sku", 3.0)

    def test_allergen_mismatch_rate(self, gt):
        self._assert_rate(gt, "allergen_mismatch", 5.0)

    def test_encoding_glitch_rate(self, gt):
        self._assert_rate(gt, "encoding_glitch", 2.0)

    def test_inconsistent_units_rate(self, gt):
        self._assert_rate(gt, "inconsistent_units", 1.0)

    def test_test_data_poison_rate(self, gt):
        self._assert_rate(gt, "test_data_poison", 4.0)


# ---------------------------------------------------------------------------
# 4. ground_truth.json validity — structure + columns reference real headers
# ---------------------------------------------------------------------------

class TestGroundTruth:
    @pytest.fixture(scope="class")
    def data(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("gt")
        generate(n=200, seed=42, out_dir=d)
        return d, _load_gt(d)

    def test_top_level_keys(self, data):
        _, gt = data
        assert "mappings" in gt
        assert "planted_issues" in gt
        assert "generator_seed" in gt

    def test_all_five_issue_types_present(self, data):
        _, gt = data
        types = {i["issue_type"] for i in gt["planted_issues"]}
        expected = {
            "duplicate_sku", "allergen_mismatch",
            "encoding_glitch", "inconsistent_units", "test_data_poison",
        }
        assert types == expected

    def test_mapping_source_columns_exist_in_csvs(self, data):
        out_dir, gt = data
        uk_headers = set(_read_csv(out_dir / "market_uk.csv")[0].keys())
        in_headers = set(_read_csv(out_dir / "market_in.csv")[0].keys())
        br_headers = set(_read_csv(out_dir / "market_br.csv", ";")[0].keys())

        market_headers = {
            "market_uk": uk_headers,
            "market_in": in_headers,
            "market_br": br_headers,
        }

        for mapping in gt["mappings"]:
            market = mapping["market"]
            col = mapping["source_column"]
            assert col in market_headers[market], \
                f"source_column '{col}' not found in {market} CSV headers"

    def test_all_markets_have_mappings(self, data):
        _, gt = data
        markets_in_gt = {m["market"] for m in gt["mappings"]}
        assert markets_in_gt == {"market_uk", "market_in", "market_br"}

    def test_row_indices_are_valid(self, data):
        out_dir, gt = data
        for issue in gt["planted_issues"]:
            for idx in issue["row_indices"]:
                assert 0 <= idx < 200, f"row index {idx} out of range for n=200"

    def test_no_real_brand_names(self, data):
        out_dir, _ = data
        forbidden = {"dove", "hellmann", "unilever", "loreal", "pantene", "head shoulders"}
        for fname, delim in [("market_uk.csv", ","), ("market_in.csv", ","), ("market_br.csv", ";")]:
            rows = _read_csv(out_dir / fname, delim)
            for row in rows:
                for val in row.values():
                    low = val.lower()
                    for brand in forbidden:
                        assert brand not in low, \
                            f"Real brand name '{brand}' found in {fname}: {val!r}"
