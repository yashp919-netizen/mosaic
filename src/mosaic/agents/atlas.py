"""ATLAS agent — semantic column mapper.

Combines BGE-m3 embedding similarity with heuristic boosters to propose
source-to-target field mappings with calibrated confidence scores.

Day 3: embedding + heuristics only.
Day 4: LLM tie-breaking for ambiguous cases + reasoning for all proposals.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml  # type: ignore[import-untyped]  # types-PyYAML not in dev deps
from pydantic import BaseModel, Field, field_validator

from mosaic.prompts.loader import load_prompt
from mosaic.retrieval.embeddings import (
    cosine_similarity_matrix,
    embed_column,
    embed_target_field,
)
from mosaic.retrieval.heuristics import score_heuristics
from mosaic.schemas import ColumnProfile, MarketProfile, MappingProposal, TargetSchemaField

if TYPE_CHECKING:
    from mosaic.llm.factory import MosaicLLMClient

_TARGET_SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "target_schema.yaml"

_HUMAN_APPROVAL_THRESHOLD = 0.75
_TOP_K = 3  # candidate alternatives to store

# Ambiguity thresholds
_AMBIGUOUS_GAP = 0.10  # top-1 minus top-2 < this → ambiguous
_AMBIGUOUS_SCORE_LOW = 0.55  # top-1 in this range → ambiguous
_AMBIGUOUS_SCORE_HIGH = 0.80


# ---------------------------------------------------------------------------
# LLM structured response models
# ---------------------------------------------------------------------------


class AtlasResolutionResponse(BaseModel):
    """Structured LLM output for ambiguous mapping resolution."""

    chosen_field: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AtlasReasoningResponse(BaseModel):
    """Structured LLM output for reasoning-only (unambiguous case)."""

    reasoning: str

    @field_validator("reasoning", mode="before")
    @classmethod
    def coerce_to_str(cls, v: object) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            # llama3.2 sometimes returns {"label": "..."} — extract any string value
            for val in v.values():
                if isinstance(val, str) and val:
                    return val
        return str(v)


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------


def load_target_schema(path: Path | None = None) -> list[TargetSchemaField]:
    """Load target schema fields from YAML."""
    schema_path = path or _TARGET_SCHEMA_PATH
    raw = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    return [TargetSchemaField(**f) for f in raw["fields"]]


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _build_target_embeddings(fields: list[TargetSchemaField]) -> np.ndarray:
    """Embed all target fields — shape (n_fields, 1024)."""
    return np.stack([embed_target_field(f.name, f.description, f.example_values) for f in fields])


def _build_source_embedding(col: ColumnProfile) -> np.ndarray:
    """Embed one source column using its name and value samples — shape (1024,)."""
    return embed_column(col.column_name, col.value_samples)


# ---------------------------------------------------------------------------
# Core heuristic mapping logic
# ---------------------------------------------------------------------------


def _map_one_column(
    col: ColumnProfile,
    source_vec: np.ndarray,
    target_fields: list[TargetSchemaField],
    target_vecs: np.ndarray,
) -> tuple[MappingProposal, list[tuple[str, float]]]:
    """Propose a mapping for one source column. Returns (proposal, all_candidates)."""
    # Cosine similarities: shape (n_targets,)
    sims = cosine_similarity_matrix(source_vec[np.newaxis], target_vecs)[0]

    # Apply heuristic boosts
    boosted = np.array(
        [
            min(
                1.0,
                float(sims[i])
                + score_heuristics(
                    col.column_name,
                    col.inferred_dtype,
                    target_fields[i].name,
                    target_fields[i].dtype,
                ),
            )
            for i in range(len(target_fields))
        ],
        dtype=np.float32,
    )

    # All candidates sorted desc
    all_idx = np.argsort(boosted)[::-1]
    top_k_idx = all_idx[:_TOP_K]
    best_idx = int(top_k_idx[0])
    best_score = float(boosted[best_idx])

    all_candidates = [(target_fields[int(i)].name, float(boosted[int(i)])) for i in all_idx]

    candidate_alternatives = [
        (target_fields[int(i)].name, float(boosted[int(i)])) for i in top_k_idx[1:]
    ]

    proposal = MappingProposal(
        source_column=col.column_name,
        target_field=target_fields[best_idx].name,
        confidence=round(best_score, 4),
        reasoning="",
        requires_human_approval=best_score < _HUMAN_APPROVAL_THRESHOLD,
        candidate_alternatives=candidate_alternatives,
    )
    return proposal, all_candidates


# ---------------------------------------------------------------------------
# Ambiguity detection
# ---------------------------------------------------------------------------


def _is_ambiguous(proposal: MappingProposal) -> bool:
    """True if this mapping is ambiguous and needs LLM tie-breaking."""
    top1 = proposal.confidence
    top2 = proposal.candidate_alternatives[0][1] if proposal.candidate_alternatives else 0.0
    gap = top1 - top2
    return gap < _AMBIGUOUS_GAP or (_AMBIGUOUS_SCORE_LOW <= top1 <= _AMBIGUOUS_SCORE_HIGH)


# ---------------------------------------------------------------------------
# LLM resolution
# ---------------------------------------------------------------------------


def llm_resolve_ambiguous(
    source_col: ColumnProfile,
    candidates: list[tuple[str, float]],
    target_fields: dict[str, TargetSchemaField],
    llm_client: MosaicLLMClient,
    embed_score: float,
) -> tuple[str, float, str]:
    """Ask LLM to pick the best target field for an ambiguous source column.

    Returns (chosen_field, blended_confidence, reasoning).
    Confidence is blended: 0.6 * llm_conf + 0.4 * embed_conf.
    """
    top3 = candidates[:3]
    candidates_text = "\n".join(
        f"{i + 1}. {name} — {target_fields[name].description} "
        f"(examples: {', '.join(target_fields[name].example_values[:3])}) "
        f"[embedding score: {score:.3f}]"
        for i, (name, score) in enumerate(top3)
        if name in target_fields
    )

    prompt = load_prompt(
        "atlas_resolve_v1",
        column_name=source_col.column_name,
        value_samples=", ".join(source_col.value_samples[:5]),
        llm_description=source_col.llm_description or "(no LLM description available)",
        candidates=candidates_text,
    )

    response: AtlasResolutionResponse = llm_client.create(
        response_model=AtlasResolutionResponse,
        messages=[{"role": "user", "content": prompt}],
    )

    blended_conf = round(0.6 * response.confidence + 0.4 * embed_score, 4)
    return response.chosen_field, blended_conf, response.reasoning


def _fill_reasoning_batch(
    proposals: list[MappingProposal],
    col_map: dict[str, ColumnProfile],
    target_fields: dict[str, TargetSchemaField],
    llm_client: MosaicLLMClient,
) -> list[MappingProposal]:
    """Fill in reasoning for unambiguous proposals without changing their confidence.

    Runs each call sequentially ("batch" here means we isolate and group them
    before calling, rather than interleaving with resolution calls).
    """
    updated = []
    for p in proposals:
        col = col_map[p.source_column]
        tf = target_fields.get(p.target_field or "")
        if tf is None:
            updated.append(p)
            continue

        prompt = load_prompt(
            "atlas_reasoning_v1",
            column_name=col.column_name,
            value_samples=", ".join(col.value_samples[:5]),
            llm_description=col.llm_description or "(none)",
            target_field_name=tf.name,
            target_field_description=tf.description,
            target_field_examples=", ".join(tf.example_values[:3]),
        )

        try:
            reasoning = llm_client.complete(
                messages=[{"role": "user", "content": prompt}]
            ).strip()
        except Exception as exc:  # noqa: BLE001
            reasoning = f"[LLM reasoning failed: {exc!s:.120}]"

        updated.append(p.model_copy(update={"reasoning": reasoning}))

    return updated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def propose_mappings(
    market_profile: MarketProfile,
    target_schema: list[TargetSchemaField] | None = None,
    llm_client: MosaicLLMClient | None = None,
) -> list[MappingProposal]:
    """Propose source-to-target column mappings for one market.

    Args:
        market_profile: Output of SCOUT's profile_market().
        target_schema: Target fields to map to. Loads from YAML if None.
        llm_client: When provided, ambiguous proposals are LLM-resolved and
            all proposals get a reasoning string filled in.

    Returns:
        One MappingProposal per source column, sorted by confidence descending.
    """
    if target_schema is None:
        target_schema = load_target_schema()

    target_fields_dict: dict[str, TargetSchemaField] = {f.name: f for f in target_schema}
    target_vecs = _build_target_embeddings(target_schema)

    proposals: list[MappingProposal] = []
    all_candidates_map: dict[str, list[tuple[str, float]]] = {}

    for col in market_profile.columns:
        source_vec = _build_source_embedding(col)
        proposal, all_candidates = _map_one_column(col, source_vec, target_schema, target_vecs)
        proposals.append(proposal)
        all_candidates_map[col.column_name] = all_candidates

    if llm_client is not None:
        col_map = {col.column_name: col for col in market_profile.columns}
        ambiguous_indices = [i for i, p in enumerate(proposals) if _is_ambiguous(p)]
        unambiguous_indices = [i for i, p in enumerate(proposals) if not _is_ambiguous(p)]

        # Resolve ambiguous cases: LLM picks field + confidence + reasoning
        for i in ambiguous_indices:
            p = proposals[i]
            col = col_map[p.source_column]
            candidates = all_candidates_map[p.source_column]
            try:
                chosen, blended_conf, reasoning = llm_resolve_ambiguous(
                    col, candidates, target_fields_dict, llm_client, p.confidence
                )
                # If chosen field is valid, update proposal
                if chosen in target_fields_dict:
                    new_alts = [
                        (name, score) for name, score in candidates[:_TOP_K] if name != chosen
                    ]
                    proposals[i] = p.model_copy(
                        update={
                            "target_field": chosen,
                            "confidence": blended_conf,
                            "reasoning": reasoning,
                            "requires_human_approval": blended_conf < _HUMAN_APPROVAL_THRESHOLD,
                            "candidate_alternatives": new_alts[:2],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                # LLM call failed — keep heuristic result, leave reasoning empty
                proposals[i] = proposals[i].model_copy(
                    update={"reasoning": f"[LLM resolution failed: {exc}]"}
                )

        # Fill reasoning for unambiguous cases in batch
        unambiguous = [proposals[i] for i in unambiguous_indices]
        filled = _fill_reasoning_batch(unambiguous, col_map, target_fields_dict, llm_client)
        for idx, i in enumerate(unambiguous_indices):
            proposals[i] = filled[idx]

    proposals.sort(key=lambda p: p.confidence, reverse=True)
    return proposals
