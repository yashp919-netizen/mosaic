"""Tests for src/mosaic/eval_utils.py — ground truth field translation."""

from __future__ import annotations

from mosaic.eval_utils import GT_TO_CANONICAL, canonicalize_gt_mapping


def test_four_renamed_fields_resolve_correctly():
    raw = {
        "product_name": "product_name",
        "size_ml": "size_ml",
        "weight_g": "weight_g",
        "category": "category",
    }
    result = canonicalize_gt_mapping(raw)
    assert result["product_name"] == "global_product_name"
    assert result["size_ml"] == "size_value"
    assert result["weight_g"] == "weight_grams"
    assert result["category"] == "category_l1"


def test_passthrough_fields_unchanged():
    raw = {"sku_id": "sku_id", "brand": "brand", "allergens": "allergens"}
    result = canonicalize_gt_mapping(raw)
    assert result == raw


def test_gt_to_canonical_covers_all_four_drifted_fields():
    assert "product_name" in GT_TO_CANONICAL
    assert "size_ml" in GT_TO_CANONICAL
    assert "weight_g" in GT_TO_CANONICAL
    assert "category" in GT_TO_CANONICAL
