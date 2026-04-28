# ADR-0001: FAISS vs Qdrant for Vector Store

**Status:** Accepted

## Context

Mosaic needs a vector store for BGE-m3 embeddings used by ATLAS during semantic column mapping.

## Decision

Use FAISS in-process rather than Qdrant.

## Reasons

- Zero infrastructure overhead for MVP — no Docker service to manage during development
- Index size is small (target schema has <100 fields; source schemas have <20 columns each)
- FAISS ships as a pip package; the entire index fits in memory and is rebuilt per run
- Qdrant's persistence and filtering features are not needed at this scale

## Consequences

If the target schema grows beyond ~10,000 fields or multi-tenant isolation is needed, migrate to Qdrant. ADR to be revisited at v2.
