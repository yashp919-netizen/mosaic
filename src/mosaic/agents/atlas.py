"""ATLAS agent — semantic column mapper.

Combines BGE-m3 embedding similarity with heuristic boosters to propose
source-to-target field mappings with calibrated confidence scores.

Day 3: embedding + heuristics only. LLM reasoning layer added on Day 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from mosaic.retrieval.embeddings import (
    cosine_similarity_matrix,
    embed_column,
    embed_target_field,
)
from mosaic.retrieval.heuristics import score_heuristics
from mosaic.schemas import ColumnProfile, MarketProfile, MappingProposal, TargetSchemaField

_TARGET_SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "target_schema.yaml"

_HUMAN_APPROVAL_THRESHOLD = 0.75
_TOP_K = 3  # candidate alternatives to store


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

def load_target_schema(path: Path | None = None) -> list[TargetSchemaField]:
    """Load target schema fields from YAML."""
    schema_path = path or _TARGET_SCHEMA_PATH
    raw = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    return [TargetSchemaField(**f) for f in raw["fields"]]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _build_target_embeddings(fields: list[TargetSchemaField]) -> np.ndarray:
    """Embed all target fields — shape (n_fields, 1024)."""
    return np.stack([
        embed_target_field(f.name, f.description, f.example_values)
        for f in fields
    ])


def _build_source_embedding(col: ColumnProfile) -> np.ndarray:
    """Embed one source column using its name and value samples — shape (1024,)."""
    return embed_column(col.column_name, col.value_samples)


# ---------------------------------------------------------------------------
# Core mapping logic
# ---------------------------------------------------------------------------

def _map_one_column(
    col: ColumnProfile,
    source_vec: np.ndarray,
    target_fields: list[TargetSchemaField],
    target_vecs: np.ndarray,
) -> MappingProposal:
    """Propose a mapping for one source column."""
    # Cosine similarities: shape (n_targets,)
    sims = cosine_similarity_matrix(source_vec[np.newaxis], target_vecs)[0]

    # Apply heuristic boosts
    boosted = np.array([
        min(1.0, float(sims[i]) + score_heuristics(
            col.column_name,
            col.inferred_dtype,
            target_fields[i].name,
            target_fields[i].dtype,
        ))
        for i in range(len(target_fields))
    ], dtype=np.float32)

    # Top-K candidates
    top_k_idx = np.argsort(boosted)[::-1][:_TOP_K]
    best_idx = int(top_k_idx[0])
    best_score = float(boosted[best_idx])

    candidate_alternatives = [
        (target_fields[int(i)].name, float(boosted[int(i)]))
        for i in top_k_idx[1:]
    ]

    return MappingProposal(
        source_column=col.column_name,
        target_field=target_fields[best_idx].name,
        confidence=round(best_score, 4),
        reasoning="",  # LLM fills this on Day 4
        requires_human_approval=best_score < _HUMAN_APPROVAL_THRESHOLD,
        candidate_alternatives=candidate_alternatives,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose_mappings(
    market_profile: MarketProfile,
    target_schema: list[TargetSchemaField] | None = None,
) -> list[MappingProposal]:
    """Propose source-to-target column mappings for one market.

    Args:
        market_profile: Output of SCOUT's profile_market().
        target_schema: Target fields to map to. Loads from YAML if None.

    Returns:
        One MappingProposal per source column, sorted by confidence descending.
    """
    if target_schema is None:
        target_schema = load_target_schema()

    target_vecs = _build_target_embeddings(target_schema)

    proposals = []
    for col in market_profile.columns:
        source_vec = _build_source_embedding(col)
        proposal = _map_one_column(col, source_vec, target_schema, target_vecs)
        proposals.append(proposal)

    proposals.sort(key=lambda p: p.confidence, reverse=True)
    return proposals
