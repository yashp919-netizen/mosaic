# Mosaic — LLM Comparative Benchmark v1

**Date:** 2026-04-29  
**Ollama model:** Llama 3.2 (llama3.2:latest, 8B, local)  
**Gemini model:** Gemini 2.5 Flash (API, cloud)  
**Markets:** UK (clean schema), India (abbreviated columns), Brazil (Portuguese + noise)  
**Eval harness:** `eval/run_eval.py` — auto-approves all human-review items at top proposal

---

## Results

| Metric                        | Llama 3.2 8B (local) | Gemini 2.5 Flash | Delta      |
|-------------------------------|----------------------|------------------|------------|
| Mapping F1 (overall)          | 0.96                 | 1.00             | +0.04      |
| Mapping F1 (UK)               | 1.00                 | 1.00             | 0.00       |
| Mapping F1 (IN)               | 0.94                 | 1.00             | +0.06      |
| Mapping F1 (BR)               | 0.94                 | 1.00             | +0.06      |
| ECE (calibration error)       | 0.13                 | 0.13             | 0.00       |
| Avg Atlas latency/col (s)     | ~106.5               | ~2.45            | −97.8%     |
| Total cost                    | $0.00                | ~$0.02 (est.)    | +$0.02     |

### Per-market runtimes

| Runtime (s)     | Llama UK | Llama IN | Llama BR | Gemini UK | Gemini IN | Gemini BR |
|-----------------|----------|----------|----------|-----------|-----------|-----------|
| Scout           | 476.4    | 176.9    | 184.3    | 28.1      | 3.5       | 3.9       |
| Atlas           | 1090.4   | 317.9    | 1468.0   | 35.8      | 8.4       | 21.9      |
| Scribe          | 2.1      | 1.2      | 0.4      | 0.2       | 0.2       | 0.2       |

### Cost note

Gemini 2.5 Flash pricing: ~$0.075/M input tokens + ~$0.30/M output tokens.  
Total estimated tokens across 3 markets: ~150K (9 columns × 3 markets × characterization + atlas calls).  
Estimated cost: **~$0.02** — well under the $3.00 guard limit. Proceeding.

---

## Trade-off Summary

**Where Gemini 2.5 Flash wins:** Gemini is 40–50× faster than Llama 3.2 8B on the same hardware (API latency vs. local inference on a consumer GPU), and it corrects the two false-positive mappings that Llama produced on the India and Brazil markets — pushing F1 from 0.94 to a perfect 1.00 on those markets. For a production CPG migration where time-to-insight matters, this makes Gemini the clear choice.

**Where Llama 3.2 wins:** The local model costs nothing per call, keeps all data on-premises (critical for PII-sensitive master data in regulated industries), and still achieves perfect F1 on the UK market. ECE is identical between providers — calibration is driven by the BGE-m3 similarity scores passed into Atlas, not by the LLM reasoning layer, so neither provider has an advantage there.

**What would guide the choice in real enterprise deployment:** Data residency requirements are the deciding factor. If the customer's MSA prohibits sending product master data to a third-party API (common in pharma, finance, and EU-GDPR-governed deployments), Llama is the only viable path regardless of speed. If data can leave the premises, Gemini's 50× Atlas speedup and higher F1 make it the default — the incremental API cost (~$0.02/market) is negligible against professional services rates. A hybrid architecture — Llama for schema discovery (Scout), Gemini for column reasoning (Atlas) — would optimize both constraints.
