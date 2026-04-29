# Evaluation Methodology

## Overview

Mosaic ships with a custom evaluation harness (`eval/run_eval.py`) designed for the specific task of source-to-target column mapping, where standard RAG metrics (faithfulness, answer relevancy) do not apply.

---

## Metrics

### Mapping Accuracy (Precision / Recall / F1)

For each market, the eval compares Atlas's final proposals against `data/synth/ground_truth.json`:

- **True Positive (TP):** A proposal whose `target_field` matches the ground truth for that `source_column`.
- **False Positive (FP):** A proposal produced but with the wrong `target_field`.
- **False Negative (FN):** An expected mapping (in ground truth) that the system did not produce.

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
```

Human-review items are auto-approved at their top proposed `target_field` for the eval run (we measure system accuracy, not human behaviour). A separate human-review simulation is out of scope for v1.

Ground truth field names are translated from source-friendly aliases to canonical target schema names via `GT_TO_CANONICAL` in `src/mosaic/eval_utils.py`. This translation is the single source of truth — the ground truth JSON is not modified.

### Confidence Calibration (ECE)

Proposals are bucketed by predicted confidence in 0.1-width bins (0.0–0.1, 0.1–0.2, ..., 0.9–1.0). For each bucket:

- **Actual accuracy:** fraction of proposals in that bucket whose `target_field` matches ground truth.
- **Expected accuracy:** bucket midpoint (0.05, 0.15, ..., 0.95).
- **Gap:** actual − expected.

**Expected Calibration Error (ECE)** = weighted average of |gap| across buckets, weighted by bucket count N.

A perfectly calibrated system has ECE = 0. ECE < 0.10 is acceptable for v1. ECE > 0.15 warrants investigation.

Calibration plots (reliability diagrams) are saved to `eval/reports/calibration_v1.png`.

### Data Quality Detection Rate

For each planted issue type in `ground_truth.json`, the eval checks whether Scout flagged it in `MarketProfile.data_quality_issues` or any `ColumnProfile.data_quality_flags`. Reported as a fraction per issue type.

Note: `duplicate_sku` and `allergen_mismatch` are cross-market issues that cannot be detected by per-market profiling. These are reported as `not_detectable_per_market`.

### Runtime

Per-market wall-clock runtime broken down by agent:
- **Scout runtime:** `MarketProfile.profiling_runtime_seconds` (statistical + LLM characterization combined).
- **Atlas runtime:** `time.perf_counter()` wrapper in `eval/run_eval.py`.
- **Scribe runtime:** `time.perf_counter()` wrapper in `eval/run_eval.py`.

---

## Reproducibility

The eval is deterministic for a given LLM provider and seed:

```bash
python -m eval.run_eval --market all --llm ollama
```

The synthetic data generator is deterministic for `--seed 42` (the default). Re-running the generator with the same seed produces byte-identical CSVs and the same `ground_truth.json`.

LLM outputs are not deterministic across runs (temperature > 0). Mapping accuracy results may vary by ±1–2 percentage points between runs due to LLM non-determinism in the tie-breaking step. The heuristic-only baseline (no LLM) is fully deterministic.

---

## v1 Results

See `eval/reports/raw_v1.json` for the full numerical results and `eval/reports/comparison_v1.md` for the Llama 3.2 8B vs Gemini 2.0 Flash comparison.

| Market | Mapping F1 | ECE |
|---|---|---|
| UK (clean schema, English) | 1.00 | 0.05 |
| India (abbreviated columns, mixed language) | 0.94 | 0.19 |
| Brazil (Portuguese, semicolons, mojibake) | 0.94 | 0.15 |
| **Overall ECE** | — | **0.1315** |

The system is underconfident (ECE positive direction), which is the correct failure mode for enterprise data migration. See `docs/architecture.md` Section 5 for analysis.
