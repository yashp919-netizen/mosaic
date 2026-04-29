# ADR-0001: FAISS vs Qdrant for Vector Storage

**Status:** Accepted

---

## Context

Mosaic's Atlas agent uses BGE-m3 embeddings to compute cosine similarity between source column descriptors and target schema field descriptors. This requires a vector store capable of nearest-neighbour search.

The corpus at v1 scale is tiny: approximately 9 column descriptors per market × 3 markets = ~27 source vectors, against 12 target schema fields. Total indexed vectors: ~40. The store is rebuilt per pipeline run; there is no cross-run persistence requirement.

Options evaluated:

| Option | Operational model | Pros | Cons |
|---|---|---|---|
| **FAISS** (in-process) | Python library, no server | Zero infra overhead; pip install; IVF-PQ scales to 100K+ | No cross-process sharing; index rebuilt per run |
| **Qdrant** | Docker service | Persistent; filtering; REST API; production-ready | Docker dependency; overkill for 40 vectors; two services to manage |
| **Pinecone** | Managed cloud | Fully managed; scales horizontally | Paid; network latency; external data dependency; offline-hostile |
| **Weaviate** | Docker service | Rich schema support; hybrid search | Even heavier than Qdrant for this use case |

---

## Decision

Use **FAISS in-process** for v1.

---

## Rationale

At v1 scale, any server-based vector store introduces infrastructure complexity that is not justified by the corpus size. A FAISS flat index over 40 vectors completes a similarity search in microseconds; the embedding step (BGE-m3 forward pass) dominates runtime by three orders of magnitude.

FAISS ships as a pip package (`faiss-cpu`), which means `uv sync` is sufficient to run the full pipeline with no Docker or network dependencies. This is a meaningful property for a demo-first project targeting HuggingFace Spaces and `docker compose up` as deployment targets.

The `IVF-PQ` index type used by FAISS supports scaling to 100K+ vectors if the target schema grows (e.g., a real enterprise product master with hundreds of attributes). Migrating from FAISS to Qdrant requires changing only `src/mosaic/retrieval/store.py` — the Atlas agent consumes an abstract interface.

Continuity with earlier work (Dumbledore AI project) also influenced this choice: BGE-m3 + FAISS is a validated local-embedding pattern.

---

## Alternatives Considered

**Qdrant:** The most likely v2 choice if Mosaic needs to serve multiple concurrent pipeline runs or share a vector index across processes (e.g., a FastAPI backend). The operational overhead is manageable in a proper deployment. Not justified in a 14-day sprint.

**Pinecone:** Rejected for v1 because it adds a network dependency, a paid account, and an external data transfer — all incompatible with the local-first, demo-first requirements.

**Weaviate:** Heavier than Qdrant, no additional features relevant to this use case.

---

## Consequences

- The vector index is rebuilt on every pipeline run. Cost is negligible (~40 vectors, milliseconds).
- No cross-process or cross-run vector sharing. If the same pipeline is called twice concurrently, each call builds its own in-memory index.
- Replacing FAISS with Qdrant in v2 requires changing `src/mosaic/retrieval/store.py` only. Atlas does not need to change.
- IVF-PQ index parameters may need tuning if corpus grows significantly (>10K vectors).

**Revisit trigger:** v2, if the FastAPI REST layer requires shared vector state, or if the target schema grows beyond ~1,000 fields.
