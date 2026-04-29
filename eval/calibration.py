"""Day-4 calibration sanity check.

Buckets Atlas mapping proposals by predicted confidence and computes
actual accuracy per bucket. This is NOT the final eval harness (Day 8);
it's a quick sanity check on whether confidence semantics are meaningful.

Usage:
    python eval/calibration.py [--market uk|in|br|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_GT_PATH = _ROOT / "data" / "synth" / "ground_truth.json"
_MARKET_CSV = {
    "market_uk": _ROOT / "data" / "synth" / "market_uk.csv",
    "market_in": _ROOT / "data" / "synth" / "market_in.csv",
    "market_br": _ROOT / "data" / "synth" / "market_br.csv",
}

sys.path.insert(0, str(_ROOT / "src"))
from mosaic.eval_utils import GT_TO_CANONICAL, canonicalize_gt_mapping  # noqa: E402


def _canonicalise(field: str) -> str:
    return GT_TO_CANONICAL.get(field, field)


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------


def _load_ground_truth() -> dict[str, dict[str, str]]:
    """Return {market_id: {source_column: canonical_target_field}}."""
    raw = json.loads(_GT_PATH.read_text(encoding="utf-8"))
    raw_by_market: dict[str, dict[str, str]] = defaultdict(dict)
    for entry in raw["mappings"]:
        raw_by_market[entry["market"]][entry["source_column"]] = entry["target_field"]
    return {market: canonicalize_gt_mapping(m) for market, m in raw_by_market.items()}


# ---------------------------------------------------------------------------
# Run Atlas (stats-only, no LLM, for speed)
# ---------------------------------------------------------------------------


def _run_atlas(market_id: str) -> list:
    """Return list of MappingProposal for a market using heuristic Atlas only."""
    from mosaic.agents.scout import profile_market
    from mosaic.agents.atlas import propose_mappings

    csv_path = _MARKET_CSV[market_id]
    profile = profile_market(csv_path, market_id, characterize=False)
    return propose_mappings(profile)


# ---------------------------------------------------------------------------
# Calibration bucketing
# ---------------------------------------------------------------------------

_BUCKETS = [(i / 10, (i + 1) / 10) for i in range(10)]  # (0.0,0.1) ... (0.9,1.0)


def _bucket_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}-{hi:.1f}"


def calibrate(markets: list[str]) -> list[dict]:
    """Run Atlas on each market, compare to ground truth, bucket by confidence.

    Returns list of bucket dicts: {bucket, n, correct, accuracy}.
    """
    gt = _load_ground_truth()

    # Accumulate (confidence, correct) pairs across all markets
    pairs: list[tuple[float, bool]] = []
    for market_id in markets:
        gt_market = gt.get(market_id, {})
        proposals = _run_atlas(market_id)
        for p in proposals:
            if p.target_field is None:
                continue
            true_field = gt_market.get(p.source_column)
            if true_field is None:
                continue
            correct = p.target_field == true_field
            pairs.append((p.confidence, correct))

    # Bucket
    results = []
    for lo, hi in _BUCKETS:
        in_bucket = [(conf, ok) for conf, ok in pairs if lo <= conf < hi]
        # Edge-case: include conf=1.0 in the top bucket
        if hi == 1.0:
            in_bucket = [(conf, ok) for conf, ok in pairs if lo <= conf <= hi]
        n = len(in_bucket)
        midpoint = round(lo + 0.05, 2)
        if n == 0:
            results.append({"bucket": _bucket_label(lo, hi), "midpoint": midpoint, "n": 0, "accuracy": None})
        else:
            accuracy = sum(ok for _, ok in in_bucket) / n
            results.append(
                {
                    "bucket": _bucket_label(lo, hi),
                    "midpoint": midpoint,
                    "n": n,
                    "accuracy": round(accuracy, 4),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict], markets: list[str]) -> None:
    header = f"{'Confidence bucket':<20} {'N':>5}  {'Actual accuracy':>16}"
    sep = "-" * len(header)
    print(f"\nCalibration sanity check | markets: {', '.join(markets)}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        if r["n"] == 0:
            acc_str = "     n/a"
        else:
            acc_str = f"{r['accuracy']:.4f}"
        print(f"{r['bucket']:<20} {r['n']:>5}  {acc_str:>16}")
    print(sep)

    # Filled-bucket summary
    filled = [r for r in rows if r["n"] > 0 and r["accuracy"] is not None]
    if filled:
        total_n = sum(r["n"] for r in filled)
        overall_acc = sum(r["accuracy"] * r["n"] for r in filled) / total_n
        print(f"\nOverall accuracy (weighted): {overall_acc:.4f}  |  Total proposals: {total_n}")

    # ECE — weighted average of |actual_accuracy - bucket_midpoint|
    ece_num = sum(
        abs(r["accuracy"] - r["midpoint"]) * r["n"]
        for r in filled
    )
    ece = ece_num / total_n if total_n else float("nan")
    print(f"ECE (Expected Calibration Error):  {ece:.4f}")
    if ece < 0.10:
        print("  Calibration is acceptable for v1 (ECE < 0.10).")
    elif ece < 0.15:
        print("  Calibration is borderline (0.10 <= ECE < 0.15). Monitor in Week 2.")
    else:
        print("  Calibration problem (ECE >= 0.15). Revisit confidence scoring in Week 2.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas calibration sanity check (Day 4)")
    parser.add_argument(
        "--market",
        choices=["uk", "in", "br", "all"],
        default="all",
        help="Which market(s) to evaluate (default: all)",
    )
    args = parser.parse_args()

    if args.market == "all":
        markets = ["market_uk", "market_in", "market_br"]
    else:
        markets = [f"market_{args.market}"]

    print(f"Running Atlas (heuristic-only) on: {', '.join(markets)} ...")
    rows = calibrate(markets)
    _print_table(rows, markets)


if __name__ == "__main__":
    sys.path.insert(0, str(_ROOT / "src"))
    main()
