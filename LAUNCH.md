# Mosaic v1.0 — Launch Summary

*Internal use only — input for LinkedIn post and interview prep.*

---

## The 90-second pitch

CPG companies run the same product portfolio across 100+ countries in incompatible legacy systems — the same shampoo exists with different column names, units, and category codes in every market, and mapping it all together before a cloud migration takes weeks of spreadsheet work.

Mosaic automates that mapping phase using three LangGraph-orchestrated agents: Scout profiles each source CSV statistically and in business language, Atlas proposes source-to-target column mappings using BGE-m3 multilingual embeddings plus LLM reasoning with a confidence score on every decision, and Scribe generates per-SKU lineage records and an executive summary. Low-confidence mappings go to a human-in-the-loop checkpoint — the system fails conservatively, never silently.

**Results on a 5,000-SKU synthetic benchmark across three legacy market schemas:**
- Mapping F1: **0.96** (Llama 3.2 local) / **1.00** (Gemini 2.5 Flash)
- Calibration ECE: **0.13** — underconfident, not wrong (the right failure mode)
- Cloud cost for full 3-market benchmark: **$0.02**

**Links:**
- GitHub: https://github.com/yashp919-netizen/mosaic
- Live demo: https://huggingface.co/spaces/yashp919/mosaic

---

## Headline metrics

- **Mapping F1: 0.96 overall** (27/28 columns correctly mapped across 3 markets using local Llama; 28/28 with Gemini)
- **ECE: 0.13** — underconfident, not wrong — the system flags uncertain mappings for human review rather than guessing silently
- **Runtime: ~30s per market** (heuristics-only) / ~29 min per market (with Llama 3.2 8B local LLM) / ~2 min per market (Gemini 2.5 Flash)
- **Cloud cost: $0.02** for the full 3-market comparative benchmark

---

## What v1 does not do (honest)

- No transformation code generation — `transformations_applied: []` in every LineageRecord (v2 Alchemist)
- No adversarial output validation beyond Pydantic schema checks (v2 Sentinel)
- Evaluated on synthetic data only — calibration on real-world CPG masters will differ
- Per-market operation only — no cross-market entity resolution (same product across UK/IN/BR)
- Allergens stored as comma-separated string, not parsed against EU CLP or FSSAI codes
- Encoding glitch detection incomplete — Scout missed the planted 2% mojibake in Brazil

---

## What's next (v2 roadmap)

1. **Alchemist agent** — given a confirmed mapping, generate Python/SQL transformation code (DD/MM/YY → ISO 8601, oz → grams, category code → canonical path) with a self-testing loop
2. **Sentinel agent** — adversarial validator that checks the harmonized master for constraint violations and schema conformance before go-live sign-off
3. **Cross-market entity resolution** — identify that UK-12345, IN-AB9901, and BR-X7723 are the same product concept; produce a golden master record

Full roadmap: `docs/v2-roadmap.md`

---

## Built with

| Component | Choice | Why |
|---|---|---|
| Embeddings | BGE-m3 | Multilingual, open weights, runs locally |
| Agent framework | LangGraph | Explicit state machine, native human-in-the-loop |
| LLM — primary | Llama 3.2 8B via Ollama | Local, free, zero data egress |
| LLM — benchmark | Gemini 2.5 Flash | 50x faster, +0.04 F1, $0.02 total cost |
| Structured output | Pydantic v2 + instructor | Every agent output typed and validated |
| Vector store | FAISS in-process | Zero infra overhead for v1 corpus |
| Eval | Custom harness | Standard RAGAS doesn't fit mapping tasks |
| UI | Streamlit | Fastest demoable UI in a 2-week sprint |
| Deployment | Docker + HuggingFace Spaces | Local-first + public demo |

---

## Interview talking points

**"Walk me through the architecture."**
Three LangGraph agents — Scout, Atlas, Scribe — share a typed MosaicState object through a StateGraph with a MemorySaver checkpointer. The graph has one conditional branch: if Atlas produces low-confidence mappings (< 0.75), the graph interrupts and hands control to a human reviewer. The checkpointer serializes state so the graph resumes without re-running prior agents. I can draw this as a whiteboard diagram in 2 minutes.

**"How do you know it works?"**
Custom eval harness against synthetic ground truth. F1 1.00 on the clean UK market, 0.94 on abbreviated Indian columns and Portuguese Brazilian columns. ECE 0.13 — underconfident, which is the correct failure mode for enterprise migration. Compared Llama 3.2 8B local vs Gemini 2.5 Flash cloud: +0.04 F1 delta, 50× faster, $0.02 total cost for the full benchmark.

**"Why three agents and not five?"**
ADR-0003. Scout + Atlas + Scribe cover the highest-value phase — you can't generate transformations or validate outputs until you know the mapping is right. A benchmarked 3-agent system is more credible than a half-built 5-agent system. Alchemist and Sentinel are in the v2 roadmap with explicit rationale for the deferral.

**"Why LangGraph over CrewAI?"**
ADR-0002. Human-in-the-loop interrupt is a first-class primitive in LangGraph. CrewAI abstracts the graph away — you can't see the state machine. LangGraph's StateGraph maps directly to a whiteboard diagram, which matters when you're explaining architectural decisions to a CDO or a technical interviewer.
