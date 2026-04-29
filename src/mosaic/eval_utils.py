"""Shared eval utilities — ground truth translation and helpers."""

from __future__ import annotations

GT_TO_CANONICAL: dict[str, str] = {
    "product_name": "global_product_name",
    "size_ml": "size_value",
    "weight_g": "weight_grams",
    "category": "category_l1",
    # remaining fields are 1-to-1 (sku_id, brand, allergens, barcode, launch_date)
}


def canonicalize_gt_mapping(mappings: dict[str, str]) -> dict[str, str]:
    """Apply GT_TO_CANONICAL translation to a ground truth mappings dict.

    Args:
        mappings: {source_column: target_field} from ground_truth.json for one market.

    Returns:
        New dict with target_field values translated to canonical schema names.
    """
    return {src: GT_TO_CANONICAL.get(tgt, tgt) for src, tgt in mappings.items()}
