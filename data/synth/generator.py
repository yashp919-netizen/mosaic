#!/usr/bin/env python3
"""
Mosaic synthetic data generator.

Produces market_uk.csv, market_in.csv, market_br.csv and ground_truth.json
inside --out-dir. All output is deterministic for a given --seed.

Usage:
    python data/synth/generator.py [--seed 42] [--out-dir data/synth]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    from faker import Faker
except ImportError:
    print("faker not installed. Run: pip install faker", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Static lookup tables
# ---------------------------------------------------------------------------

BRANDS = [
    "Zephyr", "Luminos", "Vanta", "Oralix", "Purevex",
    "Helix", "Solven", "Glova", "Tresol", "Brightway",
    "Nexpur", "Kaleo", "Serafix", "Aurova", "Plexum",
    "Cristalix", "Verdant", "Nuvelo", "Elaris", "Frostine",
    "Toreva", "Qualis", "Zenpur", "Vivara", "Motivo",
    "Cladis", "Purelix", "Sovara", "Orinum", "Cryston",
    "Velaris", "Solaris", "Primus", "Alteva", "Novix",
    "Luxara", "Clarity", "Frosten", "Aevum", "Lumeva",
    "Teraxis", "Gleam", "Polvera", "Solvix", "Nuarix",
    "Hevara", "Glicen", "Tolvex", "Krisol", "Arovia",
]

CATEGORIES_UK = [
    "Personal Care > Body Wash",
    "Personal Care > Shampoo",
    "Personal Care > Conditioner",
    "Personal Care > Face Wash",
    "Personal Care > Moisturiser",
    "Personal Care > Deodorant",
    "Personal Care > Hand Wash",
    "Household > Surface Cleaner",
    "Household > Laundry",
    "Household > Dishwash",
    "Food & Beverage > Condiments",
    "Food & Beverage > Snacks",
]

CATEGORY_TO_IN = {
    "Personal Care > Body Wash":      "PC-BW",
    "Personal Care > Shampoo":        "PC-SH",
    "Personal Care > Conditioner":    "PC-CD",
    "Personal Care > Face Wash":      "PC-FW",
    "Personal Care > Moisturiser":    "PC-MS",
    "Personal Care > Deodorant":      "PC-DO",
    "Personal Care > Hand Wash":      "PC-HW",
    "Household > Surface Cleaner":    "HH-SC",
    "Household > Laundry":            "HH-LN",
    "Household > Dishwash":           "HH-DW",
    "Food & Beverage > Condiments":   "FB-CN",
    "Food & Beverage > Snacks":       "FB-SN",
}

CATEGORY_TO_BR = {
    "Personal Care > Body Wash":      "Cuidados Pessoais > Banho",
    "Personal Care > Shampoo":        "Cuidados Pessoais > Cabelo > Shampoo",
    "Personal Care > Conditioner":    "Cuidados Pessoais > Cabelo > Condicionador",
    "Personal Care > Face Wash":      "Cuidados Pessoais > Rosto > Limpeza",
    "Personal Care > Moisturiser":    "Cuidados Pessoais > Rosto > Hidratação",
    "Personal Care > Deodorant":      "Cuidados Pessoais > Desodorante",
    "Personal Care > Hand Wash":      "Cuidados Pessoais > Sabonete Líquido",
    "Household > Surface Cleaner":    "Casa > Limpeza > Superfícies",
    "Household > Laundry":            "Casa > Lavanderia",
    "Household > Dishwash":           "Casa > Limpeza > Louças",
    "Food & Beverage > Condiments":   "Alimentos > Condimentos",
    "Food & Beverage > Snacks":       "Alimentos > Petiscos",
}

PRODUCT_TYPES: dict[str, list[str]] = {
    "Personal Care > Body Wash":    ["Moisturising Body Wash", "Refreshing Shower Gel", "Nourishing Body Cleanser", "Deep Clean Body Wash", "Sensitive Skin Body Wash"],
    "Personal Care > Shampoo":      ["Hydrating Shampoo", "Volumising Shampoo", "Anti-Dandruff Shampoo", "Repair Shampoo", "Colour Protect Shampoo"],
    "Personal Care > Conditioner":  ["Deep Conditioner", "Leave-In Conditioner", "Repair Conditioner", "Volumising Conditioner", "Smooth & Shine Conditioner"],
    "Personal Care > Face Wash":    ["Gentle Face Wash", "Oil Control Face Wash", "Brightening Face Wash", "Exfoliating Face Scrub", "Hydrating Face Wash"],
    "Personal Care > Moisturiser":  ["Daily Moisturiser", "Night Cream", "SPF Moisturiser", "Intensive Moisturiser", "Lightweight Lotion"],
    "Personal Care > Deodorant":    ["24hr Roll-On Deodorant", "48hr Spray Deodorant", "Sport Deodorant", "Sensitive Deodorant", "Whitening Deodorant"],
    "Personal Care > Hand Wash":    ["Antibacterial Hand Wash", "Moisturising Hand Wash", "Fragrance-Free Hand Wash", "Foaming Hand Wash", "Aloe Hand Wash"],
    "Household > Surface Cleaner":  ["Multi-Surface Spray", "Kitchen Cleaner", "Bathroom Cleaner", "Glass Cleaner", "Disinfectant Spray"],
    "Household > Laundry":          ["Liquid Laundry Detergent", "Laundry Powder", "Fabric Softener", "Laundry Capsules", "Stain Remover"],
    "Household > Dishwash":         ["Dish Liquid", "Dishwasher Tablets", "Rinse Aid", "Heavy Duty Dish Soap", "Eco Dish Liquid"],
    "Food & Beverage > Condiments": ["Tomato Ketchup", "Mayonnaise", "Mustard Sauce", "Hot Chilli Sauce", "Vinegar Dressing"],
    "Food & Beverage > Snacks":     ["Salted Crackers", "Mixed Nuts Blend", "Granola Bar", "Rice Cakes", "Oat Protein Bar"],
}

ALLERGENS_EN = ["milk", "eggs", "wheat", "soy", "peanuts", "tree nuts", "fish", "shellfish", "sesame", "gluten"]

ALLERGEN_TO_PT = {
    "milk": "leite", "eggs": "ovos", "wheat": "trigo", "soy": "soja",
    "peanuts": "amendoim", "tree nuts": "castanhas", "fish": "peixe",
    "shellfish": "crustáceos", "sesame": "gergelim", "gluten": "glúten",
}

SIZE_ML_OPTIONS = [50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000]
WEIGHT_G_OPTIONS = [50, 75, 100, 150, 200, 250, 300, 400, 500]

DEVANAGARI_WORDS = [
    "शैम्पू", "साबुन", "क्रीम", "लोशन", "जेल",
    "पाउडर", "तेल", "फोम", "स्प्रे", "सीरम",
]

TEST_POISON_PREFIXES = ["TEST-", "XXX-", "DUMMY-"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_allergens(rng: random.Random) -> list[str]:
    k = rng.randint(0, 4)
    return rng.sample(ALLERGENS_EN, k)


def _random_date(rng: random.Random) -> date:
    start = date(2018, 1, 1)
    return start + timedelta(days=rng.randint(0, 365 * 6))


def _mojibake(s: str) -> str:
    """Simulate UTF-8 bytes read as Latin-1 (classic CPG data pipeline glitch)."""
    try:
        return s.encode("utf-8").decode("latin-1")
    except Exception:
        return s


def _fl_oz_from_ml(ml: float) -> float:
    """Convert ml to fl oz, rounded to 2 dp."""
    return round(ml / 29.5735, 2)


def _g_to_oz(g: float) -> float:
    return round(g / 28.3495, 2)


def _ean13(rng: random.Random) -> str:
    digits = [rng.randint(0, 9) for _ in range(12)]
    check = (10 - sum((3 if i % 2 else 1) * d for i, d in enumerate(digits)) % 10) % 10
    return "".join(map(str, digits)) + str(check)


# ---------------------------------------------------------------------------
# Concept generation
# ---------------------------------------------------------------------------

def build_concepts(n: int, rng: random.Random) -> list[dict]:
    """Generate n canonical product concepts shared across all markets."""
    concepts = []
    for i in range(n):
        category = rng.choice(CATEGORIES_UK)
        product_type = rng.choice(PRODUCT_TYPES[category])
        brand = rng.choice(BRANDS)
        size_ml = rng.choice(SIZE_ML_OPTIONS)
        weight_g = rng.choice(WEIGHT_G_OPTIONS)
        allergens = _random_allergens(rng)
        barcode = _ean13(rng)
        launch_date = _random_date(rng)
        concepts.append({
            "idx": i,
            "category": category,
            "product_name": f"{brand} {product_type}",
            "brand": brand,
            "size_ml": size_ml,
            "weight_g": weight_g,
            "allergens": allergens,
            "barcode": barcode,
            "launch_date": launch_date,
        })
    return concepts


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def build_uk_row(concept: dict, sku_id: str) -> dict:
    allergens_str = ",".join(concept["allergens"]) if concept["allergens"] else ""
    return {
        "sku_id": sku_id,
        "product_name": concept["product_name"],
        "brand": concept["brand"],
        "size_ml": concept["size_ml"],
        "weight_g": concept["weight_g"],
        "category": concept["category"],
        "allergens": allergens_str,
        "barcode": concept["barcode"],
        "launch_date": concept["launch_date"].isoformat(),
    }


def build_in_row(concept: dict, sku_id: str, rng: random.Random, use_devanagari: bool) -> dict:
    prod_nm = concept["product_name"]
    if use_devanagari:
        dv = rng.choice(DEVANAGARI_WORDS)
        words = prod_nm.split()
        insert_pos = rng.randint(1, max(1, len(words) - 1))
        words.insert(insert_pos, dv)
        prod_nm = " ".join(words)

    allergens_str = ",".join(concept["allergens"]) if concept["allergens"] else ""
    d = concept["launch_date"]
    date_str = f"{d.day:02d}/{d.month:02d}/{str(d.year)[2:]}"

    return {
        "SKU_CD": sku_id,
        "PROD_NM": prod_nm,
        "BRND": concept["brand"],
        "SZ": f"{concept['size_ml']}ml",
        "WT_GMS": concept["weight_g"],
        "CAT_CD": CATEGORY_TO_IN[concept["category"]],
        "ALLRGY": allergens_str,
        "EAN": concept["barcode"],
        "DT_LNCH": date_str,
    }


def build_br_row(concept: dict, sku_id: str) -> dict:
    allergens_pt = ",".join(ALLERGEN_TO_PT[a] for a in concept["allergens"]) if concept["allergens"] else ""
    d = concept["launch_date"]
    date_str = f"{d.day:02d}/{d.month:02d}/{d.year}"

    return {
        "codigo_sku": sku_id,
        "nome_produto": concept["product_name"],
        "marca": concept["brand"],
        "tamanho": f"{concept['size_ml']} mL",
        "peso_oz": _g_to_oz(concept["weight_g"]),
        "categoria": CATEGORY_TO_BR[concept["category"]],
        "alergenos": allergens_pt,
        "codigo_barras": concept["barcode"],
        "data_lancamento": date_str,
    }


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate(n: int, seed: int, out_dir: Path) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = build_concepts(n, rng)

    # --- Decide which rows get which planted issues (non-overlapping sets) ---
    all_indices = list(range(n))
    rng.shuffle(all_indices)

    n_dup      = math.ceil(n * 0.03)   # 3%  duplicate SKUs
    n_allergen = math.ceil(n * 0.05)   # 5%  allergen mismatches
    n_glitch   = math.ceil(n * 0.02)   # 2%  encoding glitches (BR only)
    n_units    = math.ceil(n * 0.01)   # 1%  inconsistent units (UK only)
    n_poison   = math.ceil(n * 0.04)   # 4%  test-data poison

    offset = 0
    dup_indices     = set(all_indices[offset : offset + n_dup]);    offset += n_dup
    allergen_indices= set(all_indices[offset : offset + n_allergen]); offset += n_allergen
    glitch_indices  = set(all_indices[offset : offset + n_glitch]); offset += n_glitch
    units_indices   = set(all_indices[offset : offset + n_units]);  offset += n_units
    poison_indices  = set(all_indices[offset : offset + n_poison])

    # --- Build rows ---
    uk_rows, in_rows, br_rows = [], [], []

    for i, concept in enumerate(concepts):
        # SKU IDs — duplicate issue shares a raw numeric ID across markets
        if i in dup_indices:
            raw_id = f"{i:05d}"
            uk_id = raw_id
            in_id = raw_id
            br_id = raw_id
        else:
            uk_id = f"UK-{i:05d}"
            in_id = f"IN{i:05d}"
            br_id = f"BR-{i:05d}"

        # UK row
        uk = build_uk_row(concept, uk_id)
        if i in units_indices:
            uk["size_ml"] = _fl_oz_from_ml(concept["size_ml"])
        if i in poison_indices:
            prefix = rng.choice(TEST_POISON_PREFIXES)
            uk["product_name"] = prefix + uk["product_name"]
        uk_rows.append(uk)

        # IN row
        use_dv = rng.random() < 0.30
        in_row = build_in_row(concept, in_id, rng, use_dv)
        if i in allergen_indices:
            mismatched = _random_allergens(rng)
            in_row["ALLRGY"] = ",".join(mismatched)
        if i in poison_indices:
            prefix = rng.choice(TEST_POISON_PREFIXES)
            in_row["PROD_NM"] = prefix + in_row["PROD_NM"]
        in_rows.append(in_row)

        # BR row
        br = build_br_row(concept, br_id)
        if i in allergen_indices:
            mismatched_pt = [ALLERGEN_TO_PT[a] for a in _random_allergens(rng)]
            br["alergenos"] = ",".join(mismatched_pt)
        if i in glitch_indices:
            br["nome_produto"] = _mojibake(br["nome_produto"])
        if i in poison_indices:
            prefix = rng.choice(TEST_POISON_PREFIXES)
            br["nome_produto"] = prefix + br["nome_produto"]
        br_rows.append(br)

    # --- Write CSVs ---
    _write_csv(out_dir / "market_uk.csv", uk_rows, delimiter=",")
    _write_csv(out_dir / "market_in.csv", in_rows, delimiter=",")
    _write_csv(out_dir / "market_br.csv", br_rows, delimiter=";")

    # --- Build and write ground truth ---
    ground_truth = {
        "generator_seed": seed,
        "generated_at": date.today().isoformat(),
        "mappings": [
            # UK
            {"market": "market_uk", "source_column": "sku_id",       "target_field": "sku_id"},
            {"market": "market_uk", "source_column": "product_name",  "target_field": "product_name"},
            {"market": "market_uk", "source_column": "brand",         "target_field": "brand"},
            {"market": "market_uk", "source_column": "size_ml",       "target_field": "size_ml"},
            {"market": "market_uk", "source_column": "weight_g",      "target_field": "weight_g"},
            {"market": "market_uk", "source_column": "category",      "target_field": "category"},
            {"market": "market_uk", "source_column": "allergens",     "target_field": "allergens"},
            {"market": "market_uk", "source_column": "barcode",       "target_field": "barcode"},
            {"market": "market_uk", "source_column": "launch_date",   "target_field": "launch_date"},
            # IN
            {"market": "market_in", "source_column": "SKU_CD",  "target_field": "sku_id"},
            {"market": "market_in", "source_column": "PROD_NM", "target_field": "product_name"},
            {"market": "market_in", "source_column": "BRND",    "target_field": "brand"},
            {"market": "market_in", "source_column": "SZ",      "target_field": "size_ml"},
            {"market": "market_in", "source_column": "WT_GMS",  "target_field": "weight_g"},
            {"market": "market_in", "source_column": "CAT_CD",  "target_field": "category"},
            {"market": "market_in", "source_column": "ALLRGY",  "target_field": "allergens"},
            {"market": "market_in", "source_column": "EAN",     "target_field": "barcode"},
            {"market": "market_in", "source_column": "DT_LNCH", "target_field": "launch_date"},
            # BR
            {"market": "market_br", "source_column": "codigo_sku",        "target_field": "sku_id"},
            {"market": "market_br", "source_column": "nome_produto",       "target_field": "product_name"},
            {"market": "market_br", "source_column": "marca",              "target_field": "brand"},
            {"market": "market_br", "source_column": "tamanho",            "target_field": "size_ml"},
            {"market": "market_br", "source_column": "peso_oz",            "target_field": "weight_g"},
            {"market": "market_br", "source_column": "categoria",          "target_field": "category"},
            {"market": "market_br", "source_column": "alergenos",          "target_field": "allergens"},
            {"market": "market_br", "source_column": "codigo_barras",      "target_field": "barcode"},
            {"market": "market_br", "source_column": "data_lancamento",    "target_field": "launch_date"},
        ],
        "planted_issues": [
            {
                "issue_type": "duplicate_sku",
                "market": "all",
                "row_indices": sorted(dup_indices),
                "description": (
                    f"{len(dup_indices)} SKUs share a raw numeric ID across all three markets "
                    "(no market prefix), simulating a cross-market duplicate that harmonization must resolve."
                ),
            },
            {
                "issue_type": "allergen_mismatch",
                "market": "market_in+market_br",
                "row_indices": sorted(allergen_indices),
                "description": (
                    f"{len(allergen_indices)} rows have allergen lists in market_in and market_br "
                    "that differ from the canonical concept, simulating regulatory transcription errors."
                ),
            },
            {
                "issue_type": "encoding_glitch",
                "market": "market_br",
                "row_indices": sorted(glitch_indices),
                "description": (
                    f"{len(glitch_indices)} rows in market_br have mojibake in nome_produto "
                    "(UTF-8 bytes decoded as Latin-1), simulating a legacy ETL pipeline encoding bug."
                ),
            },
            {
                "issue_type": "inconsistent_units",
                "market": "market_uk",
                "row_indices": sorted(units_indices),
                "description": (
                    f"{len(units_indices)} rows in market_uk have size_ml values stored in fl oz "
                    "(no flag or column rename), simulating a unit conversion error from a regional feed."
                ),
            },
            {
                "issue_type": "test_data_poison",
                "market": "all",
                "row_indices": sorted(poison_indices),
                "description": (
                    f"{len(poison_indices)} rows across all markets have TEST-, XXX-, or DUMMY- "
                    "prefixed product names, simulating test records that leaked into production data."
                ),
            },
        ],
    }

    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    print(f"Generated {n} concepts x 3 markets = {n * 3:,} total rows")
    print(f"  market_uk.csv : {len(uk_rows):,} rows")
    print(f"  market_in.csv : {len(in_rows):,} rows")
    print(f"  market_br.csv : {len(br_rows):,} rows")
    print(f"  ground_truth.json written")
    print(f"\nPlanted issues (seed={seed}):")
    print(f"  duplicate_sku        : {len(dup_indices):>4} rows  (3%)")
    print(f"  allergen_mismatch    : {len(allergen_indices):>4} rows  (5%)")
    print(f"  encoding_glitch      : {len(glitch_indices):>4} rows  (2%, BR only)")
    print(f"  inconsistent_units   : {len(units_indices):>4} rows  (1%, UK only)")
    print(f"  test_data_poison     : {len(poison_indices):>4} rows  (4%, all markets)")


def _write_csv(path: Path, rows: list[dict], delimiter: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mosaic synthetic data generator")
    parser.add_argument("--seed",    type=int, default=42,            help="Random seed (default: 42)")
    parser.add_argument("--n",       type=int, default=5000,          help="Number of product concepts (default: 5000)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/synth"), help="Output directory")
    args = parser.parse_args()

    generate(n=args.n, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
