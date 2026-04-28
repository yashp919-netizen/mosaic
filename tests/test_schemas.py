"""Tests for src/mosaic/schemas.py — one model per section."""

from __future__ import annotations

import pytest
from datetime import datetime
from pydantic import ValidationError

from mosaic.schemas import (
    ColumnProfile,
    LineageRecord,
    MarketProfile,
    MappingProposal,
    SkuRecord,
    TargetSchemaField,
)


# ---------------------------------------------------------------------------
# TargetSchemaField
# ---------------------------------------------------------------------------

class TestTargetSchemaField:
    def test_valid(self):
        f = TargetSchemaField(
            name="sku_id",
            description="Unique product identifier",
            dtype="string",
            required=True,
            example_values=["UK-0001", "UK-0002"],
        )
        assert f.name == "sku_id"
        assert f.dtype == "string"

    def test_invalid_dtype(self):
        with pytest.raises(ValidationError):
            TargetSchemaField(
                name="x", description="y", dtype="blob", required=False, example_values=[]
            )

    def test_json_round_trip(self):
        f = TargetSchemaField(
            name="weight_grams",
            description="Weight in grams",
            dtype="float",
            required=True,
            example_values=["100.0", "250.5"],
        )
        assert TargetSchemaField.model_validate_json(f.model_dump_json()) == f


# ---------------------------------------------------------------------------
# SkuRecord
# ---------------------------------------------------------------------------

class TestSkuRecord:
    def test_valid_with_extra_fields(self):
        r = SkuRecord.model_validate(
            {"_market_id": "uk", "_row_index": 0, "product_name": "Zephyr Shampoo"}
        )
        assert r.model_extra["product_name"] == "Zephyr Shampoo"

    def test_json_round_trip(self):
        r = SkuRecord.model_validate({"_market_id": "in", "_row_index": 5, "PROD_NM": "लुमिनोस"})
        restored = SkuRecord.model_validate_json(r.model_dump_json())
        assert restored.model_extra["PROD_NM"] == "लुमिनोस"


# ---------------------------------------------------------------------------
# ColumnProfile
# ---------------------------------------------------------------------------

class TestColumnProfile:
    def test_valid(self):
        cp = ColumnProfile(
            column_name="product_name",
            inferred_dtype="string",
            null_rate=0.02,
            unique_count=4800,
            detected_language="en",
            value_samples=["Zephyr Shampoo 250ml"],
            llm_description="Product display name.",
            data_quality_flags=[],
        )
        assert cp.null_rate == 0.02

    def test_null_rate_above_1_fails(self):
        with pytest.raises(ValidationError):
            ColumnProfile(
                column_name="x", inferred_dtype="string",
                null_rate=1.5, unique_count=0,
            )

    def test_null_rate_below_0_fails(self):
        with pytest.raises(ValidationError):
            ColumnProfile(
                column_name="x", inferred_dtype="string",
                null_rate=-0.1, unique_count=0,
            )

    def test_value_samples_max_10(self):
        # Exactly 10 is fine
        cp = ColumnProfile(
            column_name="x", inferred_dtype="string", null_rate=0.0, unique_count=20,
            value_samples=[str(i) for i in range(10)],
        )
        assert len(cp.value_samples) == 10
        # 11 items exceeds max_length=10 → ValidationError
        with pytest.raises(ValidationError):
            ColumnProfile(
                column_name="x", inferred_dtype="string", null_rate=0.0, unique_count=20,
                value_samples=[str(i) for i in range(11)],
            )

    def test_json_round_trip(self):
        cp = ColumnProfile(
            column_name="barcode", inferred_dtype="string",
            null_rate=0.0, unique_count=5000,
            value_samples=["5000112637922"],
        )
        assert ColumnProfile.model_validate_json(cp.model_dump_json()) == cp


# ---------------------------------------------------------------------------
# MarketProfile
# ---------------------------------------------------------------------------

class TestMarketProfile:
    def _make(self) -> MarketProfile:
        col = ColumnProfile(
            column_name="sku_id", inferred_dtype="string",
            null_rate=0.0, unique_count=5000,
        )
        return MarketProfile(
            market_id="market_uk",
            row_count=5000,
            column_count=9,
            columns=[col],
            data_quality_issues=["high null rate in allergens"],
            profiling_runtime_seconds=1.23,
        )

    def test_valid(self):
        mp = self._make()
        assert mp.market_id == "market_uk"
        assert mp.profiling_runtime_seconds == 1.23

    def test_negative_row_count_fails(self):
        with pytest.raises(ValidationError):
            MarketProfile(
                market_id="x", row_count=-1, column_count=0,
                columns=[], profiling_runtime_seconds=0.0,
            )

    def test_json_round_trip(self):
        mp = self._make()
        assert MarketProfile.model_validate_json(mp.model_dump_json()) == mp


# ---------------------------------------------------------------------------
# MappingProposal
# ---------------------------------------------------------------------------

class TestMappingProposal:
    def test_valid(self):
        mp = MappingProposal(
            source_column="PROD_NM",
            target_field="global_product_name",
            confidence=0.91,
            reasoning="Name and samples match.",
            requires_human_approval=False,
            candidate_alternatives=[("brand", 0.45)],
        )
        assert mp.confidence == 0.91

    def test_confidence_above_1_fails(self):
        with pytest.raises(ValidationError):
            MappingProposal(
                source_column="x", target_field="y", confidence=1.1,
            )

    def test_confidence_below_0_fails(self):
        with pytest.raises(ValidationError):
            MappingProposal(
                source_column="x", target_field="y", confidence=-0.1,
            )

    def test_target_field_nullable(self):
        mp = MappingProposal(
            source_column="unknown_col", target_field=None, confidence=0.1,
        )
        assert mp.target_field is None

    def test_json_round_trip(self):
        mp = MappingProposal(
            source_column="SKU_CD", target_field="sku_id", confidence=0.97,
            candidate_alternatives=[("barcode", 0.30)],
        )
        assert MappingProposal.model_validate_json(mp.model_dump_json()) == mp


# ---------------------------------------------------------------------------
# LineageRecord
# ---------------------------------------------------------------------------

class TestLineageRecord:
    def test_valid(self):
        lr = LineageRecord(
            source_sku_id="UK-00001",
            source_market="market_uk",
            target_sku_id="MSC-uk-000001",
            field_lineage={"global_product_name": "product_name"},
            agents_involved=["scout", "atlas", "scribe"],
            transformations_applied=[],
            timestamp=datetime(2026, 4, 28, 12, 0, 0),
        )
        assert lr.source_market == "market_uk"

    def test_default_timestamp(self):
        lr = LineageRecord(
            source_sku_id="x", source_market="y", target_sku_id="z",
            field_lineage={}, agents_involved=[],
        )
        assert isinstance(lr.timestamp, datetime)

    def test_json_round_trip(self):
        lr = LineageRecord(
            source_sku_id="BR-00001",
            source_market="market_br",
            target_sku_id="MSC-br-000001",
            field_lineage={"global_product_name": "nome_produto", "brand": "marca"},
            agents_involved=["scout", "atlas", "scribe"],
            transformations_applied=["date_format: DD/MM/YY -> ISO8601"],
            timestamp=datetime(2026, 4, 28, 9, 0, 0),
        )
        assert LineageRecord.model_validate_json(lr.model_dump_json()) == lr
