from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared enumerations
# ---------------------------------------------------------------------------

class Market(str, Enum):
    UK = "market_uk"
    IN = "market_in"
    BR = "market_br"


class ConfidenceTier(str, Enum):
    HIGH = "high"        # >= 0.85
    MEDIUM = "medium"    # 0.60 – 0.84
    LOW = "low"          # < 0.60  → goes to human-in-the-loop queue


class IssueType(str, Enum):
    DUPLICATE_SKU = "duplicate_sku"
    ALLERGEN_MISMATCH = "allergen_mismatch"
    ENCODING_GLITCH = "encoding_glitch"
    INCONSISTENT_UNITS = "inconsistent_units"
    TEST_DATA_POISON = "test_data_poison"


# ---------------------------------------------------------------------------
# SCOUT outputs
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    """Statistical + semantic profile of one source column."""

    column_name: str
    dtype: str
    null_pct: float = Field(ge=0.0, le=1.0)
    unique_count: int = Field(ge=0)
    sample_values: list[str] = Field(default_factory=list, max_length=10)
    min_value: str | None = None
    max_value: str | None = None
    business_description: str = Field(
        default="",
        description="LLM-generated plain-English description of what this column likely contains",
    )


class DataQualityFlag(BaseModel):
    """A single data quality issue detected by SCOUT."""

    issue_type: IssueType
    affected_column: str | None = None
    row_count: int = Field(ge=0)
    sample_row_indices: list[int] = Field(default_factory=list, max_length=5)
    description: str


class MarketProfile(BaseModel):
    """Full profile of one source market CSV. Primary output of SCOUT."""

    market: Market
    source_file: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    delimiter: str = ","
    encoding: str = "utf-8"
    columns: list[ColumnProfile]
    quality_flags: list[DataQualityFlag] = Field(default_factory=list)
    profiled_at: datetime = Field(default_factory=datetime.utcnow)
    llm_summary: str = Field(
        default="",
        description="LLM-written paragraph summarising the market dataset in business language",
    )


# ---------------------------------------------------------------------------
# ATLAS outputs
# ---------------------------------------------------------------------------

class ColumnMapping(BaseModel):
    """Proposed mapping from one source column to one target field."""

    source_column: str
    target_field: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_tier: ConfidenceTier
    embedding_similarity: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(
        default="",
        description="LLM-generated explanation for why this mapping was chosen",
    )
    needs_human_review: bool = False
    transformation_hint: str = Field(
        default="",
        description="Short note on any unit conversion or format change required (e.g. 'DD/MM/YY → ISO 8601')",
    )

    @field_validator("confidence_tier", mode="before")
    @classmethod
    def derive_tier(cls, v: Any, info: Any) -> ConfidenceTier:
        if isinstance(v, ConfidenceTier):
            return v
        confidence = info.data.get("confidence", 0.0)
        if confidence >= 0.85:
            return ConfidenceTier.HIGH
        if confidence >= 0.60:
            return ConfidenceTier.MEDIUM
        return ConfidenceTier.LOW


class MappingProposal(BaseModel):
    """Full set of column mappings for one market. Primary output of ATLAS."""

    market: Market
    mappings: list[ColumnMapping]
    unmapped_source_columns: list[str] = Field(default_factory=list)
    unmapped_target_fields: list[str] = Field(default_factory=list)
    human_review_queue: list[str] = Field(
        default_factory=list,
        description="source_column names that need human confirmation before proceeding",
    )
    proposed_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def overall_confidence(self) -> float:
        if not self.mappings:
            return 0.0
        return sum(m.confidence for m in self.mappings) / len(self.mappings)


# ---------------------------------------------------------------------------
# Human-in-the-loop checkpoint
# ---------------------------------------------------------------------------

class HumanReviewDecision(BaseModel):
    """Reviewer's verdict on a single low-confidence mapping."""

    source_column: str
    approved_target_field: str
    reviewer_note: str = ""
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)


class HumanReviewResponse(BaseModel):
    """Collected decisions from a human reviewer. Feeds back into ATLAS."""

    market: Market
    decisions: list[HumanReviewDecision]


# ---------------------------------------------------------------------------
# SCRIBE outputs
# ---------------------------------------------------------------------------

class LineageRecord(BaseModel):
    """Audit trail for a single mapped field in a single SKU row."""

    sku_id: str
    market: Market
    source_column: str
    source_value: str
    target_field: str
    mapped_value: str
    confidence: float = Field(ge=0.0, le=1.0)
    transformation_applied: str = ""


class MarketLineage(BaseModel):
    """All lineage records for one market."""

    market: Market
    records: list[LineageRecord]
    total_skus: int = Field(ge=0)
    total_fields_mapped: int = Field(ge=0)


class HarmonizationReport(BaseModel):
    """Executive summary across all markets. Primary output of SCRIBE."""

    run_id: str
    markets_processed: list[Market]
    market_profiles: list[MarketProfile]
    mapping_proposals: list[MappingProposal]
    lineages: list[MarketLineage]
    executive_summary: str = Field(
        default="",
        description="LLM-written executive summary grounded in structured data only",
    )
    overall_mapping_accuracy: float | None = None
    total_skus_harmonized: int = Field(ge=0, default=0)
    total_quality_flags: int = Field(ge=0, default=0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Ground truth (used by eval harness)
# ---------------------------------------------------------------------------

class GroundTruthMapping(BaseModel):
    """Known correct source→target mapping for one market."""

    market: Market
    source_column: str
    target_field: str


class PlantedIssue(BaseModel):
    """A deliberately seeded data quality issue for eval measurement."""

    issue_type: IssueType
    market: Market
    row_indices: list[int]
    description: str


class GroundTruth(BaseModel):
    """Contents of data/synth/ground_truth.json."""

    mappings: list[GroundTruthMapping]
    planted_issues: list[PlantedIssue]
    generator_seed: int = 42
    generated_at: date = Field(default_factory=date.today)


# ---------------------------------------------------------------------------
# LangGraph pipeline state
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    """Shared state object threaded through the LangGraph state machine."""

    run_id: str
    markets: list[Market]
    market_profiles: dict[str, MarketProfile] = Field(default_factory=dict)
    mapping_proposals: dict[str, MappingProposal] = Field(default_factory=dict)
    human_review_responses: dict[str, HumanReviewResponse] = Field(default_factory=dict)
    lineages: dict[str, MarketLineage] = Field(default_factory=dict)
    report: HarmonizationReport | None = None
    current_market: Market | None = None
    awaiting_human_review: bool = False
    errors: list[str] = Field(default_factory=list)
