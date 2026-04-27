# Mosaic
**A Multi-Agent AI Reference Architecture for CPG Product Master Harmonization During Cloud Migration**

> *Tagline:* "Many markets, many systems, one product truth — harmonized by agents, validated by humans."

---

## 1. Problem Statement

Consumer Packaged Goods companies operate the same brand portfolio across 100+ countries through a patchwork of inherited ERP, e-commerce, and trade-promotion systems — each with its own product master schema, language, units, and regulatory codes. The same shampoo SKU exists in 40 country systems with different names, inconsistent allergen flags, mismatched category taxonomies, and conflicting packaging hierarchies.

When a CPG enterprise consolidates onto a modern AI-first cloud data platform — exactly the kind of program publicly announced by Unilever and Google Cloud in February 2026 — the migration team faces a brutal week-one problem: producing a unified, governed product master that downstream AI capabilities (agentic commerce, digital twins, supply chain forecasting) can actually trust.

Mid-program data engineers, MDM leads, and migration consultants typically spend weeks producing source-to-target column mappings, weeks more authoring transformations, and never get a fully transparent audit trail. **Mosaic compresses the analysis-and-mapping phase from weeks to hours, with full lineage and a quantified confidence score on every decision.**

## 2. Target User Persona

**Primary:** Master Data Management lead at a CPG company (or the consultant doing this work for them) during an SAP ECC to cloud-platform migration. 5-10 years of experience. Reports into a Chief Data Officer or Chief Supply Chain Officer.

**Secondary:** Solutions architects at consulting firms (Accenture, Deloitte, IBM Consulting, Capgemini) running CPG migration practices.

**Why they'd use it:** They are currently doing source profiling and column mapping in spreadsheets and bespoke Python scripts, with no shared agent-driven layer that combines statistical profiling, semantic understanding, and audit-grade lineage.

## 3. Why This Is a Benchmark Project (Resume Angle)

**Domain timeliness.** The Unilever-Google Cloud partnership announced in February 2026 makes CPG cloud-migration plus agentic AI the most-discussed enterprise-tech story of the quarter.

**Architectural depth.** Three specialized LangGraph-orchestrated agents with shared state, structured I/O via Pydantic, an evaluation harness with custom domain metrics, and a deliberate human-in-the-loop checkpoint pattern. Each piece is individually defensible in a 30-minute interview.

**Quantified rigor.** Ships with a benchmark report on a synthetic but realistic 5,000-SKU dataset across three markets, with mapping accuracy, confidence calibration, and time-to-harmonized-master metrics.

**Resume bullet (target):**
> Designed and shipped Mosaic, a multi-agent AI reference architecture for CPG product-master harmonization during cloud migrations. Three LangGraph-orchestrated agents (profiling, semantic mapping, audit reporting) achieved [X]% mapping accuracy on a 5,000-SKU synthetic benchmark across three legacy market schemas, with sub-minute end-to-end runtime per market. Inspired by publicly disclosed CPG transformation programs (Unilever x Google Cloud, February 2026). Stack: LangGraph, BGE-m3 multilingual embeddings, Llama 3.2 (local) with Gemini 2.0 Flash comparison, Pydantic structured outputs, custom eval harness.

## 4. Architecture Overview

Three agents orchestrated by a LangGraph state machine:

- SCOUT: Data profiler. Reads source CSV, produces statistical profile + LLM business-language characterization. Emits a MarketProfile Pydantic object.
- ATLAS: Semantic mapper. Takes Scout's profile + target schema, proposes source-to-target column mappings with confidence scores using BGE-m3 embeddings + heuristics + LLM reasoning. Low-confidence mappings go to human-in-the-loop queue.
- SCRIBE: Auditor. Generates per-SKU lineage records and an LLM-written executive summary grounded in structured data only.

Deferred to v2: Alchemist (transformation generator), Sentinel (output validator).

Flow: Scout -> Atlas -> [human checkpoint if needed] -> Scribe

## 5. Tech Stack with Justifications

| Component | Choice | Why |
|---|---|---|
| Embeddings | BGE-m3 | Multilingual, open weights, runs locally, better than OpenAI ada for noisy CPG text |
| Agent framework | LangGraph | Explicit state machine, drawable in interview, native human-in-the-loop support |
| LLM primary | Llama 3.2 8B via Ollama | Local, free, privacy-aligned |
| LLM eval comparison | Gemini 2.0 Flash via Vertex AI | Aligned with Unilever-Google Cloud announced stack |
| Structured output | Pydantic + instructor | Every agent output typed and validated |
| Vector store | FAISS in-process | No infra overhead for MVP. ADR-001 documents why not Qdrant |
| Eval | Custom harness | Standard RAGAS does not fit mapping tasks |
| Synthetic data | Faker-based generator | Controlled ground truth, reproducible from seed |
| API | FastAPI | Async, auto OpenAPI docs |
| UI | Streamlit | Fastest demoable UI in 2 weeks |
| Deployment | HuggingFace Spaces + Dockerfile | Free public demo + portability |

## 6. Synthetic Data Specification

Three market CSVs, 5,000 product concepts each expressed three ways. Use invented brand names — do NOT use real brand names like Dove or Hellmann's.

Market 1 - market_uk.csv (clean, comma-separated):
Columns: sku_id, product_name, brand, size_ml, weight_g, category, allergens, barcode, launch_date
Dates in ISO format. Allergens as comma-separated string.

Market 2 - market_in.csv (mixed-language, abbreviated, comma-separated):
Columns: SKU_CD, PROD_NM, BRND, SZ, WT_GMS, CAT_CD, ALLRGY, EAN, DT_LNCH
~30% of PROD_NM values mix Devanagari + Latin script. Dates as DD/MM/YY. Categories as short codes like PC-BW.

Market 3 - market_br.csv (Portuguese, semicolon-separated):
Columns: codigo_sku, nome_produto, marca, tamanho, peso_oz, categoria, alergenos, codigo_barras, data_lancamento
Weight in ounces. Size as string like "250 mL". Category as path like "Cuidados Pessoais > Banho".

Planted issues:
- 3% cross-market duplicate SKUs
- 5% allergen mismatches between markets
- 2% encoding glitches (mojibake) in market_br.csv only
- 1% inconsistent units in market_uk.csv (size_ml values actually in fl oz, no flag)
- 4% test-data poison (rows with TEST, XXX, DUMMY in product name)

Also emit data/synth/ground_truth.json with keys: mappings (per-market source column to target field), planted_issues (list with issue_type, market, row_indices, description).

Generator must be deterministic for a given --seed (default 42).

## 7. Repository Structure

mosaic/
├── README.md
├── LICENSE
├── pyproject.toml
├── .github/workflows/eval.yml
├── docs/
│   ├── architecture.md
│   ├── eval-methodology.md
│   ├── public-sources.md
│   └── adr/
│       ├── 0001-faiss-vs-qdrant.md
│       ├── 0002-langgraph-vs-crewai.md
│       └── 0003-three-agents-not-five.md
├── data/
│   ├── synth/
│   │   ├── generator.py
│   │   ├── ground_truth.json
│   │   ├── market_uk.csv
│   │   ├── market_in.csv
│   │   └── market_br.csv
│   └── target_schema.yaml
├── src/mosaic/
│   ├── schemas.py
│   ├── orchestrator/
│   │   ├── graph.py
│   │   └── state.py
│   ├── agents/
│   │   ├── scout.py
│   │   ├── atlas.py
│   │   └── scribe.py
│   ├── llm/
│   │   ├── local.py
│   │   ├── gemini.py
│   │   └── factory.py
│   ├── prompts/
│   │   ├── scout_v1.md
│   │   ├── atlas_v1.md
│   │   └── scribe_v1.md
│   └── api/
│       └── main.py
├── eval/
│   ├── run_eval.py
│   ├── metrics.py
│   └── reports/
│       └── benchmark_v1.md
├── ui/streamlit_app.py
├── deploy/
│   ├── Dockerfile
│   └── docker-compose.yml
└── tests/
    ├── test_scout.py
    ├── test_atlas.py
    └── test_orchestrator.py

## 8. Sprint Plan

Week 1 (Build): Day 1 scaffolding + schemas + data generator, Day 2 Scout agent, Day 3 Atlas part 1 embeddings, Day 4 Atlas part 2 LLM reasoning, Day 5 LangGraph orchestrator, Day 6 Scribe agent, Day 7 polish + tag week1-complete.

Week 2 (Evaluate + Ship): Day 8 eval harness, Day 9 comparative benchmark, Day 10 Streamlit UI, Day 11 architecture docs + ADRs, Day 12 README rewrite, Day 13 Dockerfile + HuggingFace deploy, Day 14 launch + tag v1.0.

Detailed daily prompts live in daily-prompts.md.

## 9. Evaluation Methodology

Mapping accuracy: precision, recall, F1 per market vs ground_truth.json.
Confidence calibration: bucket proposals by predicted confidence, compute actual accuracy per bucket, plot reliability diagram, report Expected Calibration Error (ECE).
Data quality detection rate: fraction of planted issues Scout flagged, by issue type.
Runtime: per-market wall-clock time broken down by agent.
Reproducibility: python -m eval.run_eval --version v1 reproduces full benchmark on clean checkout.

## 10. Definition of Done for v1

A recruiter who knows nothing about me can in 5 minutes:
1. Read the README and understand what it does and how well it works.
2. Watch a 2-minute demo (gif or HuggingFace Spaces).
3. See the benchmark report with mapping accuracy, calibration plot, Llama-vs-Gemini comparison.
4. Find at least three Architecture Decision Records.
5. Run docker compose up and have it work.

## 11. v2 Stretch Goals

- Alchemist agent: LLM-driven transformation code generation with self-tests
- Sentinel agent: adversarial validator
- Allergen detection: free-text ingredients to standardized allergen tags
- Cross-market entity resolution: golden record logic
- Vertex AI deployment path ADR
- Compliance hooks: EU CLP Regulation, Indian FSSAI

## 12. Public-Source Disclaimer

Mosaic is a public-source-informed reference architecture. It is not built with, endorsed by, or representative of any specific company's systems, decisions, or data. The framing is informed by publicly disclosed CPG industry transformation programs, including the Unilever-Google Cloud partnership announced February 17, 2026, publicly reported Unilever scale details, and broader 2026 commentary on SAP S/4HANA migration patterns. All synthetic data, schemas, and agent designs are original to this project.

## 13. Pending Items

1. Author certifications - to be added to README certified competencies section mapping each cert to a specific architectural choice.
2. Local-first vs cloud-first preference - defaulted to local-first with Gemini eval comparison.
3. Codename Mosaic - keep or rename.

---

This brief is the source of truth for v1. If the work diverges, update the brief or change course consciously. Do not drift.
