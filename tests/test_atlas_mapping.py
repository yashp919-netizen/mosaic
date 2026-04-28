"""Tests for the ATLAS mapping layer (embedding + heuristics, no LLM)."""

from __future__ import annotations

import pytest

from mosaic.agents.atlas import load_target_schema, propose_mappings
from mosaic.retrieval.heuristics import score_heuristics, normalise_name
from mosaic.schemas import ColumnProfile, MarketProfile, MappingProposal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def target_schema():
    return load_target_schema()


def _make_profile(columns: list[tuple[str, str, list[str]]]) -> MarketProfile:
    """Build a minimal MarketProfile from (col_name, dtype, samples) tuples."""
    cols = [
        ColumnProfile(
            column_name=name,
            inferred_dtype=dtype,
            null_rate=0.0,
            unique_count=100,
            value_samples=samples,
        )
        for name, dtype, samples in columns
    ]
    return MarketProfile(
        market_id="test",
        row_count=100,
        column_count=len(cols),
        columns=cols,
        profiling_runtime_seconds=0.1,
    )


# ---------------------------------------------------------------------------
# Heuristics unit tests
# ---------------------------------------------------------------------------

class TestHeuristics:
    def test_exact_match_boost(self):
        boost = score_heuristics("sku_id", "string", "sku_id", "string")
        assert boost >= 0.30

    def test_abbreviation_boost_prod_nm(self):
        boost = score_heuristics("PROD_NM", "string", "global_product_name", "string")
        assert boost >= 0.20, f"Expected abbreviation boost, got {boost}"

    def test_abbreviation_boost_wt_gms(self):
        boost = score_heuristics("WT_GMS", "integer", "weight_grams", "float")
        assert boost >= 0.20

    def test_abbreviation_boost_dt_lnch(self):
        boost = score_heuristics("DT_LNCH", "date", "launch_date", "date")
        assert boost >= 0.20

    def test_substring_boost(self):
        # "brand" is substring of "brand" — but also test partial match
        boost = score_heuristics("brand_name", "string", "brand", "string")
        assert boost >= 0.15

    def test_dtype_compat_boost(self):
        boost = score_heuristics("weight_g", "float", "weight_grams", "float")
        # Should get dtype boost (both numeric) on top of substring/abbrev
        assert boost >= 0.05

    def test_no_boost_for_unrelated(self):
        boost = score_heuristics("ZZZ_UNKNOWN", "string", "launch_date", "date")
        assert boost == 0.0

    def test_normalise_name(self):
        assert normalise_name("PROD NM") == "prod_nm"
        assert normalise_name("DT-LNCH") == "dt_lnch"
        assert normalise_name("weight.grams") == "weight_grams"


# ---------------------------------------------------------------------------
# Trivial mapping — exact name matches should score > 0.9
# ---------------------------------------------------------------------------

class TestExactNameMappings:
    def test_sku_id_maps_to_sku_id(self, target_schema):
        profile = _make_profile([("sku_id", "string", ["UK-00001", "UK-00002"])])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.source_column == "sku_id"
        assert p.target_field == "sku_id"
        assert p.confidence > 0.90, f"Expected >0.90, got {p.confidence}"
        assert not p.requires_human_approval

    def test_brand_maps_to_brand(self, target_schema):
        profile = _make_profile([("brand", "string", ["Zephyr", "Luminos"])])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.target_field == "brand"
        assert p.confidence > 0.90

    def test_barcode_maps_to_barcode(self, target_schema):
        profile = _make_profile([("barcode", "string", ["5000112637922"])])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.target_field == "barcode"
        assert p.confidence > 0.90

    def test_launch_date_maps_correctly(self, target_schema):
        profile = _make_profile([("launch_date", "date", ["2022-03-15", "2021-11-01"])])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.target_field == "launch_date"
        assert p.confidence > 0.90


# ---------------------------------------------------------------------------
# Abbreviation mappings — Indian market columns should still hit high confidence
# ---------------------------------------------------------------------------

class TestAbbreviationMappings:
    def test_prod_nm_maps_to_global_product_name(self, target_schema):
        profile = _make_profile([
            ("PROD_NM", "string", ["Luminos Tomato Ketchup", "Hevara Laundry Powder"])
        ])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.target_field == "global_product_name", \
            f"Expected global_product_name, got {p.target_field} ({p.confidence:.3f})"
        assert p.confidence > 0.75

    def test_wt_gms_maps_to_weight_grams(self, target_schema):
        profile = _make_profile([("WT_GMS", "integer", ["150", "300", "500"])])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.target_field == "weight_grams", \
            f"Expected weight_grams, got {p.target_field} ({p.confidence:.3f})"
        assert p.confidence > 0.75

    def test_dt_lnch_maps_to_launch_date(self, target_schema):
        # DT_LNCH is a highly abbreviated column — embeddings + heuristics give
        # the correct target but confidence is intentionally < 0.75 (human approval
        # is correct for this case). Day 4 LLM layer will boost it further.
        profile = _make_profile([("DT_LNCH", "date", ["19/04/18", "10/01/22"])])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.target_field == "launch_date", \
            f"Expected launch_date, got {p.target_field} ({p.confidence:.3f})"
        assert p.confidence > 0.50, f"Expected >0.50, got {p.confidence}"

    def test_sku_cd_maps_to_sku_id(self, target_schema):
        profile = _make_profile([("SKU_CD", "string", ["IN00001", "IN00002"])])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.target_field == "sku_id", \
            f"Expected sku_id, got {p.target_field} ({p.confidence:.3f})"


# ---------------------------------------------------------------------------
# No-match case — junk column should produce low confidence + require approval
# ---------------------------------------------------------------------------

class TestNoMatch:
    def test_junk_column_low_confidence(self, target_schema):
        profile = _make_profile([
            ("ZZZZZZ_MYSTERY_COL", "string", ["abc123", "xyz456", "qrs789"])
        ])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        assert p.confidence < 0.75, f"Expected low confidence, got {p.confidence}"
        assert p.requires_human_approval

    def test_junk_column_has_candidates(self, target_schema):
        profile = _make_profile([
            ("ZZZZZZ_MYSTERY_COL", "string", ["abc123"])
        ])
        proposals = propose_mappings(profile, target_schema)
        p = proposals[0]
        # Should still provide alternatives for human review
        assert len(p.candidate_alternatives) > 0


# ---------------------------------------------------------------------------
# Multi-column profile — verify sorting and structure
# ---------------------------------------------------------------------------

class TestMultiColumn:
    def test_sorted_by_confidence_descending(self, target_schema):
        profile = _make_profile([
            ("sku_id", "string", ["UK-001"]),          # should be high confidence
            ("ZZZZZZ_MYSTERY", "string", ["abc123"]),   # should be low confidence
            ("brand", "string", ["Zephyr"]),            # should be high confidence
        ])
        proposals = propose_mappings(profile, target_schema)
        confidences = [p.confidence for p in proposals]
        assert confidences == sorted(confidences, reverse=True)

    def test_one_proposal_per_column(self, target_schema):
        profile = _make_profile([
            ("sku_id", "string", ["UK-001"]),
            ("brand", "string", ["Zephyr"]),
            ("barcode", "string", ["5000112637922"]),
        ])
        proposals = propose_mappings(profile, target_schema)
        assert len(proposals) == 3
