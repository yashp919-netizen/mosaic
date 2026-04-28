"""Data-driven heuristics for ATLAS column mapping.

Abbreviation patterns are stored here so atlas.py stays free of hardcoded
lookup tables. Add new patterns as new source markets are onboarded.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Abbreviation map: normalised source token -> target field name fragment
#
# Keys are lowercase, stripped tokens that appear in source column names.
# Values are fragments (or full names) from the target schema.
# Atlas uses this to add a confidence boost when a match is found.
# ---------------------------------------------------------------------------

ABBREVIATION_MAP: dict[str, str] = {
    # India market abbreviations (market_in)
    "prod_nm":   "global_product_name",
    "brnd":      "brand",
    "sz":        "size_value",
    "wt_gms":    "weight_grams",
    "cat_cd":    "category_l1",
    "sku_cd":    "sku_id",
    "allrgy":    "allergens",
    "ean":       "barcode",
    "dt_lnch":   "launch_date",
    # Brazil market abbreviations (market_br)
    "codigo_sku":       "sku_id",
    "nome_produto":     "global_product_name",
    "marca":            "brand",
    "tamanho":          "size_value",
    "peso_oz":          "weight_grams",
    "categoria":        "category_l1",
    "alergenos":        "allergens",
    "codigo_barras":    "barcode",
    "data_lancamento":  "launch_date",
    # UK market (mostly exact, but these help with partial matches)
    "product_name":  "global_product_name",
    "size_ml":       "size_value",
    "weight_g":      "weight_grams",
    "category":      "category_l1",
    # Generic abbreviations
    "nm":    "name",
    "cd":    "code",
    "dt":    "date",
    "wt":    "weight",
    "qty":   "quantity",
    "amt":   "amount",
    "desc":  "description",
}

# Numeric dtype kinds (for dtype-compatibility boost)
_NUMERIC_DTYPES = {"integer", "float"}


def normalise_name(name: str) -> str:
    """Lowercase, strip, replace common separators with underscore."""
    return re.sub(r"[\s\-\.]+", "_", name.strip().lower())


def abbreviation_target(source_col: str) -> str | None:
    """Return the target field fragment for a known abbreviation, or None."""
    key = normalise_name(source_col)
    return ABBREVIATION_MAP.get(key)


def score_heuristics(
    source_col: str,
    source_dtype: str,
    target_field: str,
    target_dtype: str,
) -> float:
    """Return the total heuristic boost for a (source_col, target_field) pair.

    Boosts (additive, capped at 1.0 by the caller):
      +0.30  exact name match (case-insensitive, normalised)
      +0.20  known abbreviation pattern maps to target_field
      +0.15  substring match in either direction
      +0.05  dtype compatibility (both numeric)
    """
    boost = 0.0
    src_norm = normalise_name(source_col)
    tgt_norm = normalise_name(target_field)

    # Exact match
    if src_norm == tgt_norm:
        boost += 0.30

    # Known abbreviation
    abbrev_target = abbreviation_target(source_col)
    if abbrev_target and normalise_name(abbrev_target) in tgt_norm:
        boost += 0.20

    # Substring in either direction (only if not already exact)
    if boost < 0.30:
        if src_norm in tgt_norm or tgt_norm in src_norm:
            boost += 0.15

    # Dtype compatibility
    if source_dtype in _NUMERIC_DTYPES and target_dtype in _NUMERIC_DTYPES:
        boost += 0.05

    return boost
