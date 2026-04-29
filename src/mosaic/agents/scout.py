"""SCOUT agent — statistical profiler and LLM characterizer for source market CSVs."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from mosaic.schemas import ColumnProfile, MarketProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOJIBAKE_PATTERNS = re.compile(r"Ã§|Ã©|Ã£|Ã¨|Ã¬|Ã|Â£|Â°|Ã‰|â€™|â€œ|â€")
_TEST_POISON_PATTERN = re.compile(r"\b(?:TEST|XXX|DUMMY)\b", re.IGNORECASE)

_SAMPLE_SIZE_FOR_LANGDETECT = 20  # rows sampled for language detection


# ---------------------------------------------------------------------------
# Dtype inference
# ---------------------------------------------------------------------------


def _infer_dtype(series: pd.Series) -> str:
    """Infer a Mosaic dtype string from a pandas Series."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return "string"

    # pandas already typed it as numeric
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"

    # For object columns, probe with string heuristics
    sample = non_null.astype(str).head(50)

    # Date heuristic — try common formats
    date_patterns = [
        r"^\d{4}-\d{2}-\d{2}$",  # ISO 8601
        r"^\d{2}/\d{2}/\d{2,4}$",  # DD/MM/YY or DD/MM/YYYY
        r"^\d{2}-\d{2}-\d{4}$",  # DD-MM-YYYY
    ]
    date_re = re.compile("|".join(date_patterns))
    if sample.str.match(date_re).mean() > 0.8:
        return "date"

    # Boolean heuristic
    bool_values = {"true", "false", "yes", "no", "1", "0"}
    if set(sample.str.lower().unique()) <= bool_values:
        return "boolean"

    # Numeric stored as string
    try:
        pd.to_numeric(non_null.astype(str).str.replace(",", ""))
        # Check if all are integers
        if non_null.astype(str).str.match(r"^-?\d+$").all():
            return "integer"
        return "float"
    except (ValueError, TypeError):
        pass

    return "string"


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _is_string_series(series: pd.Series) -> bool:
    """True for object or pandas 3 str dtype columns."""
    return pd.api.types.is_object_dtype(series) or str(series.dtype) in ("str", "string")


def _detect_language(series: pd.Series) -> str | None:
    """Return detected language code for string columns, None for non-string."""
    if not _is_string_series(series):
        return None

    try:
        from langdetect import detect
    except ImportError:
        return None

    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return None

    sample = non_null.sample(min(_SAMPLE_SIZE_FOR_LANGDETECT, len(non_null)), random_state=42)
    # Concatenate a block of text for more reliable detection
    text = " ".join(sample.tolist())
    try:
        return detect(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data quality checks (market-level)
# ---------------------------------------------------------------------------


def _detect_market_issues(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate > 0.50:
            issues.append(f"high_null_rate:{col}:{null_rate:.1%}")

    # Mojibake scan — check all string columns
    for col in df.select_dtypes(include=["object", "str"]).columns:
        sample_text = " ".join(df[col].dropna().astype(str).head(200).tolist())
        if _MOJIBAKE_PATTERNS.search(sample_text):
            issues.append(f"suspected_mojibake:{col}")

    # Test-data poison scan
    poison_count = 0
    for col in df.select_dtypes(include=["object", "str"]).columns:
        poison_count += (
            df[col].dropna().astype(str).str.contains(_TEST_POISON_PATTERN, regex=True).sum()
        )
    if poison_count > 0:
        issues.append(f"test_data_poison:{poison_count}_rows")

    return issues


# ---------------------------------------------------------------------------
# Core profiling
# ---------------------------------------------------------------------------


def _profile_column(col: str, series: pd.Series) -> ColumnProfile:
    non_null = series.dropna()
    null_rate = float(series.isna().mean())
    unique_count = int(series.nunique(dropna=True))

    # Value samples — up to 10 distinct non-null values
    distinct_vals = non_null.astype(str).unique().tolist()
    value_samples = distinct_vals[:10]

    return ColumnProfile(
        column_name=col,
        inferred_dtype=_infer_dtype(series),
        null_rate=null_rate,
        unique_count=unique_count,
        detected_language=_detect_language(series),
        value_samples=value_samples,
        llm_description="",
        data_quality_flags=[],
    )


def profile_market(
    csv_path: Path,
    market_id: str,
    *,
    characterize: bool = False,
    llm_client=None,
) -> MarketProfile:
    """Profile a source market CSV and return a MarketProfile.

    Args:
        csv_path: Path to the CSV file.
        market_id: Identifier string for the market (e.g. 'market_uk').
        characterize: If True, call the LLM to fill llm_description and
            data_quality_flags on each ColumnProfile. Requires llm_client.
        llm_client: An instructor-wrapped LLM client. Required when characterize=True.
    """
    t0 = time.perf_counter()

    from mosaic.utils import detect_delimiter
    delimiter = detect_delimiter(csv_path)

    df = pd.read_csv(csv_path, delimiter=delimiter, dtype=str, keep_default_na=True)

    column_profiles = [_profile_column(col, df[col]) for col in df.columns]
    data_quality_issues = _detect_market_issues(df)

    runtime = time.perf_counter() - t0

    profile = MarketProfile(
        market_id=market_id,
        row_count=len(df),
        column_count=len(df.columns),
        columns=column_profiles,
        data_quality_issues=data_quality_issues,
        profiling_runtime_seconds=round(runtime, 3),
    )

    if characterize:
        if llm_client is None:
            raise ValueError("llm_client must be provided when characterize=True")
        from mosaic.agents._scout_llm import characterize_columns

        profile = characterize_columns(profile, llm_client)

    return profile
