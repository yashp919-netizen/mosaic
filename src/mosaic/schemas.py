"""Pydantic v2 data models for the Mosaic pipeline.

All agents read and write these types. Field names here are the contract —
downstream days depend on exact names, do not rename without updating agents.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Target schema definition (loaded from data/target_schema.yaml)
# ---------------------------------------------------------------------------

class TargetSchemaField(BaseModel):
    """One field in the unified target product master schema."""

    name: str
    description: str
    dtype: Literal["string", "integer", "float", "date", "boolean"]
    required: bool
    example_values: list[str]


# ---------------------------------------------------------------------------
# Raw source row (dynamic columns per market)
# ---------------------------------------------------------------------------

class SkuRecord(BaseModel):
    """One row from a source market CSV. Extra columns are allowed and preserved."""

    model_config = ConfigDict(extra="allow")

    _market_id: str
    _row_index: int


# ---------------------------------------------------------------------------
# SCOUT outputs
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    """Statistical + semantic profile of one source column."""

    column_name: str
    inferred_dtype: str
    null_rate: float = Field(ge=0.0, le=1.0)
    unique_count: int = Field(ge=0)
    detected_language: str | None = None
    value_samples: list[str] = Field(default_factory=list, max_length=10)
    llm_description: str = ""
    data_quality_flags: list[str] = Field(default_factory=list)

    @field_validator("null_rate")
    @classmethod
    def clamp_null_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"null_rate must be between 0 and 1, got {v}")
        return v


class MarketProfile(BaseModel):
    """Full profile of one source market CSV. Primary output of SCOUT."""

    market_id: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnProfile]
    data_quality_issues: list[str] = Field(default_factory=list)
    profiling_runtime_seconds: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# ATLAS outputs
# ---------------------------------------------------------------------------

class MappingProposal(BaseModel):
    """Proposed mapping from one source column to one target field."""

    source_column: str
    target_field: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    requires_human_approval: bool = False
    candidate_alternatives: list[tuple[str, float]] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0 and 1, got {v}")
        return v


# ---------------------------------------------------------------------------
# SCRIBE outputs
# ---------------------------------------------------------------------------

class LineageRecord(BaseModel):
    """Audit record linking one source SKU row to its harmonized target."""

    source_sku_id: str
    source_market: str
    target_sku_id: str
    field_lineage: dict[str, str]
    agents_involved: list[str]
    transformations_applied: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
