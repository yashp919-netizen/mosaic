"""Mosaic full evaluation harness.

Runs the complete Scout → Atlas → Scribe pipeline on all three synthetic markets,
computes mapping precision/recall/F1, confidence calibration ECE, data quality
detection rate, and per-agent wall-clock runtime.

Usage:
    python -m eval.run_eval [--market uk|in|br|all] [--llm ollama|gemini]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mosaic.eval_utils import GT_TO_CANONICAL, canonicalize_gt_mapping  # noqa: E402

_GT_PATH = _ROOT / "data" / "synth" / "ground_truth.json"
_MARKET_CSV: dict[str, Path] = {
    "market_uk": _ROOT / "data" / "synth" / "market_uk.csv",
    "market_in": _ROOT / "data" / "synth" / "market_in.csv",
    "market_br": _ROOT / "data" / "synth" / "market_br.csv",
}
_REPORTS_DIR = _ROOT / "eval" / "reports"

# Issue types Scout can detect per-market, and the matching signal prefix
_DETECTABLE_SIGNALS: dict[str, str] = {
    "encoding_glitch": "suspected_mojibake:",
    "test_data_poison": "test_data_poison:",
}


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------


def _load_ground_truth() -> tuple[dict[str, dict[str, str]], list[dict]]:
    """Return (canonical_gt_by_market, planted_issues).

    canonical_gt_by_market: {market_id: {source_column: canonical_target_field}}
    planted_issues: raw list from ground_truth.json
    """
    raw = json.loads(_GT_PATH.read_text(encoding="utf-8"))

    raw_by_market: dict[str, dict[str, str]] = defaultdict(dict)
    for entry in raw["mappings"]:
        raw_by_market[entry["market"]][entry["source_column"]] = entry["target_field"]

    canonical = {market: canonicalize_gt_mapping(m) for market, m in raw_by_market.items()}
    return canonical, raw["planted_issues"]


# ---------------------------------------------------------------------------
# Per-market pipeline runner (direct agent calls for clean timing)
# ---------------------------------------------------------------------------


def _run_market_pipeline(market_id: str, llm_provider: str) -> dict:
    """Run Scout → Atlas → (auto-approve) → Scribe for one market.

    Returns a dict with profile, proposals, final_mappings, lineage, and timings.
    """
    import pandas as pd

    from mosaic.agents.atlas import load_target_schema, propose_mappings
    from mosaic.agents.scout import profile_market
    from mosaic.agents.scribe import generate_lineage, persist_lineage, persist_summary
    from mosaic.utils import detect_delimiter

    llm_client = None
    if llm_provider:
        from mosaic.llm.factory import get_llm
        llm_client = get_llm(llm_provider)

    csv_path = _MARKET_CSV[market_id]
    target_schema = load_target_schema()

    # ---- Scout ----
    t0 = time.perf_counter()
    profile = profile_market(
        csv_path, market_id,
        characterize=llm_client is not None,
        llm_client=llm_client,
    )
    scout_time = time.perf_counter() - t0

    # ---- Atlas ----
    t0 = time.perf_counter()
    proposals = propose_mappings(profile, target_schema, llm_client=llm_client)
    atlas_time = time.perf_counter() - t0

    # ---- Auto-approve: pick top proposed target_field for each flagged item ----
    human_decisions: dict[str, str | None] = {
        p.source_column: p.target_field
        for p in proposals
        if p.requires_human_approval
    }

    # Finalize (mirrors finalize_node logic)
    human_reviewed_cols = set(human_decisions.keys())
    final_mappings = []
    for p in proposals:
        if p.source_column not in human_reviewed_cols:
            final_mappings.append(p)
        else:
            decision = human_decisions[p.source_column]
            if decision is not None:
                final_mappings.append(p if decision == p.target_field else
                                      p.model_copy(update={"target_field": decision}))

    # Build minimal state dict for Scribe
    state: dict = {
        "market_id": market_id,
        "csv_path": str(csv_path),
        "proposals": proposals,
        "final_mappings": final_mappings,
        "human_decisions": human_decisions,
        "profile": profile,
    }

    # ---- Scribe (lineage only — skip LLM summary for eval speed) ----
    sep = detect_delimiter(csv_path)
    source_df = pd.read_csv(csv_path, sep=sep, low_memory=False)

    t0 = time.perf_counter()
    lineage = generate_lineage(state, source_df)
    synth_dir = _ROOT / "data" / "synth"
    short = market_id.split("_", 1)[1] if "_" in market_id else market_id
    persist_lineage(lineage, synth_dir, short)
    scribe_time = time.perf_counter() - t0

    return {
        "market_id": market_id,
        "profile": profile,
        "proposals": proposals,
        "final_mappings": final_mappings,
        "human_decisions": human_decisions,
        "scout_time": scout_time,
        "atlas_time": atlas_time,
        "scribe_time": scribe_time,
        "token_count": llm_client.token_count if llm_client else 0,
    }


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------


def _compute_prf(
    final_mappings: list,
    gt_market: dict[str, str],
) -> dict[str, float]:
    """Compute precision, recall, F1 against ground truth for one market."""
    proposal_map = {p.source_column: p.target_field for p in final_mappings if p.target_field}

    tp = fp = fn = 0
    for src_col, gt_target in gt_market.items():
        predicted = proposal_map.get(src_col)
        if predicted is None:
            fn += 1
        elif predicted == gt_target:
            tp += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Calibration ECE
# ---------------------------------------------------------------------------


def _compute_ece(market_ids: list[str]) -> dict[str, float]:
    """Run calibration buckets and return ECE per market and overall."""
    from eval.calibration import calibrate

    ece_by_market: dict[str, float] = {}
    for market_id in market_ids:
        rows = calibrate([market_id])
        filled = [r for r in rows if r["n"] > 0 and r["accuracy"] is not None]
        total_n = sum(r["n"] for r in filled)
        if total_n == 0:
            ece_by_market[market_id] = float("nan")
        else:
            ece_by_market[market_id] = round(
                sum(abs(r["accuracy"] - r["midpoint"]) * r["n"] for r in filled) / total_n, 4
            )

    # Overall ECE across all markets
    all_rows = calibrate(market_ids)
    filled = [r for r in all_rows if r["n"] > 0 and r["accuracy"] is not None]
    total_n = sum(r["n"] for r in filled)
    overall = (
        round(sum(abs(r["accuracy"] - r["midpoint"]) * r["n"] for r in filled) / total_n, 4)
        if total_n else float("nan")
    )
    ece_by_market["overall"] = overall
    return ece_by_market


# ---------------------------------------------------------------------------
# Data quality detection rate
# ---------------------------------------------------------------------------


def _compute_dq_detection(profile, market_id: str, planted_issues: list[dict]) -> dict:
    """Compute per-issue-type detection rate for one market.

    Only issue types that Scout can detect (encoding_glitch, test_data_poison)
    are reported with 0/1 detection; cross-market issues (duplicate_sku,
    allergen_mismatch) and inconsistent_units are flagged as 'not_detectable'.
    """
    all_signals = profile.data_quality_issues + [
        flag for col in profile.columns for flag in col.data_quality_flags
    ]

    # Filter issues that apply to this market
    relevant = [
        iss for iss in planted_issues
        if market_id in iss["market"] or iss["market"] == "all"
    ]

    detection: dict[str, dict] = {}
    seen_types: set[str] = set()

    for iss in relevant:
        itype = iss["issue_type"]
        if itype in seen_types:
            continue
        seen_types.add(itype)

        if itype in _DETECTABLE_SIGNALS:
            prefix = _DETECTABLE_SIGNALS[itype]
            detected = any(s.startswith(prefix) for s in all_signals)
            detection[itype] = {"detected": detected, "rate": 1.0 if detected else 0.0}
        else:
            detection[itype] = {"detected": None, "rate": None, "note": "not_detectable_per_market"}

    return detection


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------


def run_full_eval(market_ids: list[str], llm_provider: str = "ollama") -> dict:
    """Run full eval across given markets. Returns results dict."""
    gt_by_market, planted_issues = _load_ground_truth()

    market_results: dict[str, dict] = {}

    print(f"\nRunning Mosaic eval on markets: {', '.join(m.split('_',1)[1] for m in market_ids)}")
    print(f"LLM provider: {llm_provider or 'none (heuristics only)'}\n")

    for market_id in market_ids:
        short = market_id.split("_", 1)[1]
        print(f"  [{short.upper()}] Running pipeline...", flush=True)

        run = _run_market_pipeline(market_id, llm_provider)
        gt_market = gt_by_market.get(market_id, {})

        prf = _compute_prf(run["final_mappings"], gt_market)
        dq = _compute_dq_detection(run["profile"], market_id, planted_issues)

        market_results[market_id] = {
            "market_id": market_id,
            "mapping": prf,
            "runtime": {
                "scout_s": round(run["scout_time"], 2),
                "atlas_s": round(run["atlas_time"], 2),
                "scribe_s": round(run["scribe_time"], 2),
            },
            "dq_detection": dq,
            "scout_profiling_runtime_s": run["profile"].profiling_runtime_seconds,
            "token_count": run.get("token_count", 0),
        }

        print(
            f"  [{short.upper()}] precision={prf['precision']:.2f}  "
            f"recall={prf['recall']:.2f}  f1={prf['f1']:.2f}  "
            f"scout={run['scout_time']:.1f}s  atlas={run['atlas_time']:.1f}s  "
            f"scribe={run['scribe_time']:.1f}s"
        )

    # ECE (uses heuristic Atlas for calibration, independent of llm_provider)
    print("\n  Computing calibration ECE...", flush=True)
    ece = _compute_ece(market_ids)

    # Attach ECE to results
    for market_id in market_ids:
        market_results[market_id]["ece"] = ece.get(market_id, float("nan"))

    total_tokens = sum(market_results[m].get("token_count", 0) for m in market_ids)
    results = {
        "market_results": market_results,
        "overall_ece": ece.get("overall", float("nan")),
        "llm_provider": llm_provider,
        "market_ids": market_ids,
        "total_tokens": total_tokens,
    }

    # Save raw results
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = _REPORTS_DIR / "raw_v1.json"
    raw_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n  Raw results saved: {raw_path}")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_table(results: dict) -> None:
    market_ids = results["market_ids"]
    mr = results["market_results"]

    col_w = 8

    def fmt(v, decimals=2):
        if v is None or (isinstance(v, float) and v != v):  # nan check
            return " " * col_w
        return f"{v:.{decimals}f}".rjust(col_w)

    def fmt1(v):
        return fmt(v, 1)

    short_labels = [m.split("_", 1)[1].upper() for m in market_ids]

    # Overall mapping metrics (simple average)
    precisions = [mr[m]["mapping"]["precision"] for m in market_ids]
    recalls = [mr[m]["mapping"]["recall"] for m in market_ids]
    f1s = [mr[m]["mapping"]["f1"] for m in market_ids]
    eces = [mr[m].get("ece", float("nan")) for m in market_ids]

    def mean(vals):
        clean = [v for v in vals if isinstance(v, float) and v == v]
        return sum(clean) / len(clean) if clean else float("nan")

    # DQ detection rates (only detectable types)
    dq_rates_by_market = {}
    for m in market_ids:
        detectable = [
            v["rate"] for v in mr[m]["dq_detection"].values()
            if v.get("rate") is not None
        ]
        dq_rates_by_market[m] = mean(detectable) if detectable else float("nan")

    header_cols = "  ".join(f"{lbl:>{col_w}}" for lbl in short_labels)
    overall_col = f"{'Overall':>{col_w}}"
    sep = "-" * (32 + (col_w + 2) * len(market_ids) + col_w + 2)

    print()
    print(sep)
    print(f"{'Metric':<32}  {header_cols}  {overall_col}")
    print(sep)

    rows = [
        ("Mapping precision",  precisions, mean(precisions)),
        ("Mapping recall",     recalls,    mean(recalls)),
        ("Mapping F1",         f1s,        mean(f1s)),
        ("ECE",                eces,       mean(eces)),
        ("DQ detection rate",  [dq_rates_by_market[m] for m in market_ids], mean(list(dq_rates_by_market.values()))),
    ]

    for label, vals, overall in rows:
        cells = "  ".join(fmt(v) for v in vals)
        print(f"{label:<32}  {cells}  {fmt(overall)}")

    print(sep)

    # Runtime rows (no overall)
    for agent, key in [("Scout runtime (s)", "scout_s"), ("Atlas runtime (s)", "atlas_s"), ("Scribe runtime (s)", "scribe_s")]:
        cells = "  ".join(fmt1(mr[m]["runtime"][key]) for m in market_ids)
        blank = " " * col_w
        print(f"{agent:<32}  {cells}  {blank}")

    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mosaic full evaluation harness")
    parser.add_argument(
        "--market",
        choices=["uk", "in", "br", "all"],
        default="all",
        help="Market(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--llm",
        choices=["ollama", "gemini", "none"],
        default="ollama",
        help="LLM provider (default: ollama). Use 'none' for heuristics only.",
    )
    args = parser.parse_args()

    if args.market == "all":
        market_ids = ["market_uk", "market_in", "market_br"]
    else:
        market_ids = [f"market_{args.market}"]

    llm_provider = "" if args.llm == "none" else args.llm

    results = run_full_eval(market_ids, llm_provider)
    _print_table(results)


if __name__ == "__main__":
    main()
