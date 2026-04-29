"""Tests for SCOUT statistical profiling layer (no LLM calls)."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest

from mosaic.agents.scout import (
    profile_market,
    _infer_dtype,
    _detect_language,
    _detect_market_issues,
)
from mosaic.schemas import MarketProfile

import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures — tiny synthetic CSVs written to tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_csv(tmp_path) -> Path:
    """A small clean CSV with known column types."""
    p = tmp_path / "clean.csv"
    p.write_text(
        textwrap.dedent("""\
        sku_id,product_name,size_ml,weight_g,launch_date,in_stock
        UK-001,Zephyr Shampoo 250ml,250,300,2022-03-15,true
        UK-002,Luminos Body Wash 500ml,500,550,2021-11-01,false
        UK-003,Vanta Conditioner 400ml,400,420,2023-06-20,true
        UK-004,Oralix Face Wash 150ml,150,180,2022-09-10,false
        UK-005,Purevex Moisturiser 200ml,200,230,2023-01-05,true
    """),
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def high_null_csv(tmp_path) -> Path:
    """CSV where one column is >50% null."""
    p = tmp_path / "high_null.csv"
    p.write_text(
        textwrap.dedent("""\
        sku_id,product_name,allergens
        UK-001,Zephyr Shampoo,
        UK-002,Luminos Wash,
        UK-003,Vanta Gel,
        UK-004,Oralix Cream,milk
        UK-005,Purevex Foam,
    """),
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def mojibake_csv(tmp_path) -> Path:
    """CSV containing mojibake characters in a string column."""
    p = tmp_path / "mojibake.csv"
    rows = [
        ["sku_id", "nome_produto"],
        ["BR-001", "ZÃ©phyr ShampÃ´o"],  # mojibake: é → Ã©, ô → Ã´
        ["BR-002", "Luminos Body Wash"],
        ["BR-003", "Vanta CondicionÃ§ador"],
        ["BR-004", "Oralix Creme Facial"],
        ["BR-005", "Purevex HidrataÃ§Ã£o"],
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return p


@pytest.fixture()
def mixed_lang_csv(tmp_path) -> Path:
    """CSV with Devanagari in one column (simulating market_in PROD_NM)."""
    p = tmp_path / "mixed_lang.csv"
    p.write_text(
        textwrap.dedent("""\
        SKU_CD,PROD_NM,BRND
        IN001,Zephyr शैम्पू 250ml,Zephyr
        IN002,Luminos साबुन Body Wash,Luminos
        IN003,Vanta क्रीम Moisturiser,Vanta
        IN004,Oralix Face Wash Gel,Oralix
        IN005,Purevex शैम्पू Repair,Purevex
    """),
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def semicolon_csv(tmp_path) -> Path:
    """CSV using semicolon delimiter (simulating market_br)."""
    p = tmp_path / "br.csv"
    p.write_text(
        textwrap.dedent("""\
        codigo_sku;nome_produto;peso_oz
        BR-001;Zephyr Shampoo;3.53
        BR-002;Luminos Body Wash;7.05
        BR-003;Vanta Conditioner;4.94
    """),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# _infer_dtype unit tests
# ---------------------------------------------------------------------------


class TestInferDtype:
    def test_integer_column(self):
        s = pd.Series(["100", "200", "300", "400"])
        assert _infer_dtype(s) == "integer"

    def test_float_column(self):
        s = pd.Series(["3.53", "7.05", "4.94", "10.58"])
        assert _infer_dtype(s) == "float"

    def test_string_column(self):
        s = pd.Series(["Zephyr Shampoo", "Luminos Wash", "Vanta Gel"])
        assert _infer_dtype(s) == "string"

    def test_iso_date_column(self):
        s = pd.Series(["2022-03-15", "2021-11-01", "2023-06-20"])
        assert _infer_dtype(s) == "date"

    def test_ddmmyy_date_column(self):
        s = pd.Series(["15/03/22", "01/11/21", "20/06/23"])
        assert _infer_dtype(s) == "date"

    def test_boolean_column(self):
        s = pd.Series(["true", "false", "true", "false"])
        assert _infer_dtype(s) == "boolean"

    def test_all_null_column(self):
        s = pd.Series([None, None, None])
        assert _infer_dtype(s) == "string"


# ---------------------------------------------------------------------------
# _detect_language unit tests
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_english_column(self):
        # Use real English sentences — invented brand names confuse langdetect
        s = pd.Series(
            [
                "This product moisturises and hydrates dry skin effectively.",
                "Apply gently to hair and rinse with warm water.",
                "Suitable for daily use on all skin types.",
                "Contains natural extracts for a refreshing cleanse.",
                "Dermatologically tested and approved for sensitive skin.",
            ]
            * 4
        )
        lang = _detect_language(s)
        assert lang == "en"

    def test_non_string_column_returns_none(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _detect_language(s) is None

    def test_all_null_returns_none(self):
        s = pd.Series([None, None, None], dtype=object)
        assert _detect_language(s) is None


# ---------------------------------------------------------------------------
# _detect_market_issues unit tests
# ---------------------------------------------------------------------------


class TestDetectMarketIssues:
    def test_high_null_detected(self):
        df = pd.DataFrame(
            {
                "sku_id": ["A", "B", "C", "D", "E"],
                "allergens": [None, None, None, None, "milk"],  # 80% null
            }
        )
        issues = _detect_market_issues(df)
        assert any("high_null_rate:allergens" in i for i in issues)

    def test_mojibake_detected(self):
        df = pd.DataFrame(
            {
                "sku_id": ["BR-001", "BR-002"],
                "nome_produto": ["ZÃ©phyr ShampÃ´o", "Luminos Wash"],
            }
        )
        issues = _detect_market_issues(df)
        assert any("suspected_mojibake:nome_produto" in i for i in issues)

    def test_test_poison_detected(self):
        df = pd.DataFrame(
            {
                "sku_id": ["A", "B"],
                "product_name": ["TEST-Shampoo", "Normal Product"],
            }
        )
        issues = _detect_market_issues(df)
        assert any("test_data_poison" in i for i in issues)

    def test_clean_data_no_issues(self):
        df = pd.DataFrame(
            {
                "sku_id": ["UK-001", "UK-002"],
                "product_name": ["Zephyr Shampoo", "Luminos Body Wash"],
            }
        )
        issues = _detect_market_issues(df)
        assert issues == []


# ---------------------------------------------------------------------------
# profile_market integration tests
# ---------------------------------------------------------------------------


class TestProfileMarket:
    def test_returns_market_profile(self, clean_csv):
        mp = profile_market(clean_csv, "test_market")
        assert isinstance(mp, MarketProfile)

    def test_correct_row_count(self, clean_csv):
        mp = profile_market(clean_csv, "test_market")
        assert mp.row_count == 5

    def test_correct_column_count(self, clean_csv):
        mp = profile_market(clean_csv, "test_market")
        assert mp.column_count == 6

    def test_null_rate_calculation(self, high_null_csv):
        mp = profile_market(high_null_csv, "test")
        allergens = next(c for c in mp.columns if c.column_name == "allergens")
        assert allergens.null_rate == pytest.approx(0.8, abs=0.01)

    def test_high_null_flagged_in_issues(self, high_null_csv):
        mp = profile_market(high_null_csv, "test")
        assert any("high_null_rate" in i for i in mp.data_quality_issues)

    def test_mojibake_flagged(self, mojibake_csv):
        mp = profile_market(mojibake_csv, "market_br")
        assert any("suspected_mojibake" in i for i in mp.data_quality_issues)

    def test_semicolon_delimiter_auto_detected(self, semicolon_csv):
        mp = profile_market(semicolon_csv, "market_br")
        assert mp.column_count == 3
        assert mp.row_count == 3

    def test_profiling_runtime_populated(self, clean_csv):
        mp = profile_market(clean_csv, "test")
        assert mp.profiling_runtime_seconds >= 0.0

    def test_value_samples_max_10(self, clean_csv):
        mp = profile_market(clean_csv, "test")
        for col in mp.columns:
            assert len(col.value_samples) <= 10

    def test_dtype_inference_on_known_columns(self, clean_csv):
        mp = profile_market(clean_csv, "test")
        col_map = {c.column_name: c for c in mp.columns}
        assert col_map["size_ml"].inferred_dtype in ("integer", "float")
        assert col_map["launch_date"].inferred_dtype == "date"
        assert col_map["product_name"].inferred_dtype == "string"
