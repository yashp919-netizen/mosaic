"""LLM characterization layer for SCOUT.

Separated from scout.py to keep the stats layer importable without an LLM client.
"""

from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field, model_validator

from mosaic.prompts.loader import load_prompt
from mosaic.schemas import ColumnProfile, MarketProfile

_VALID_FLAGS = {
    "mixed_languages",
    "abbreviated_codes",
    "free_text",
    "high_cardinality_categorical",
    "numeric_as_string",
    "date_format_variation",
    "high_null_rate",
    "test_data_present",
    "encoding_issues",
    "unit_ambiguity",
}


# ---------------------------------------------------------------------------
# Structured output schema for the LLM
# ---------------------------------------------------------------------------


class ColumnCharacterization(BaseModel):
    """Structured LLM output for one column.

    A model_validator normalises common key variants that llama3.2 returns
    instead of the exact field names ('description', 'business_description',
    'column_description', etc.) so we don't need retries for name drift.
    """

    model_config = ConfigDict(populate_by_name=True)

    business_description: str = Field(
        description="1-2 sentence plain-English description of this column's business meaning"
    )
    quality_flags: list[str] = Field(
        default_factory=list,
        description="Data quality flags that apply; use only the valid flag strings",
    )

    @model_validator(mode="before")
    @classmethod
    def normalise_keys(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        # Accept any *_description key as business_description
        if "business_description" not in data:
            for key in ("description", "column_description", "business_desc", "desc"):
                if key in data:
                    data["business_description"] = data.pop(key)
                    break
        # Accept flags / data_quality_flags as quality_flags
        if "quality_flags" not in data:
            for key in ("flags", "data_quality_flags", "quality_flag"):
                if key in data:
                    data["quality_flags"] = data.pop(key)
                    break
        # Last resort — if still missing, use empty string rather than fail
        if "business_description" not in data:
            data["business_description"] = ""
        return data


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------


def _load_prompt(col: ColumnProfile) -> str:
    samples_str = ", ".join(f'"{v}"' for v in col.value_samples[:5]) or "(none)"
    return load_prompt(
        "scout_characterize_v1",
        column_name=col.column_name,
        inferred_dtype=col.inferred_dtype,
        null_rate_pct=f"{col.null_rate * 100:.1f}",
        detected_language=col.detected_language or "unknown",
        value_samples=samples_str,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def characterize_columns(
    profile: MarketProfile,
    llm_client,
    rate_limit_delay: float = 0.0,
) -> MarketProfile:
    """Fill llm_description and data_quality_flags for each column in profile.

    Returns a new MarketProfile with the characterization fields populated.
    rate_limit_delay: seconds to sleep after each LLM call. Set to 5.0 for
    Gemini free-tier (15 req/min) to avoid 429 RESOURCE_EXHAUSTED errors.
    """
    import time

    updated_columns: list[ColumnProfile] = []

    for col in profile.columns:
        prompt = _load_prompt(col)
        try:
            result: ColumnCharacterization = llm_client.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=ColumnCharacterization,
            )
            clean_flags = [f for f in result.quality_flags if f in _VALID_FLAGS]
            updated_col = col.model_copy(
                update={
                    "llm_description": result.business_description,
                    "data_quality_flags": clean_flags,
                }
            )
        except Exception as exc:
            # Degrade gracefully — leave description empty, flag the error
            updated_col = col.model_copy(
                update={
                    "llm_description": f"[characterization failed: {exc}]",
                    "data_quality_flags": [],
                }
            )

        updated_columns.append(updated_col)

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    return profile.model_copy(update={"columns": updated_columns})
