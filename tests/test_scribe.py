"""Tests for the SCRIBE agent — lineage generation and executive summary."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from mosaic.agents.scribe import (
    _build_run_summary,
    generate_lineage,
    generate_summary,
    persist_lineage,
    persist_summary,
)
from mosaic.schemas import ColumnProfile, MappingProposal, MarketProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(market_id: str = "uk", human_decisions: dict | None = None) -> dict:
    proposals = [
        MappingProposal(
            source_column="sku_id",
            target_field="sku_id",
            confidence=0.98,
            requires_human_approval=False,
        ),
        MappingProposal(
            source_column="product_name",
            target_field="global_product_name",
            confidence=0.91,
            requires_human_approval=False,
        ),
        MappingProposal(
            source_column="brand",
            target_field="brand",
            confidence=0.95,
            requires_human_approval=False,
        ),
    ]
    return {
        "market_id": market_id,
        "csv_path": "",
        "target_schema": [],
        "profile": None,
        "proposals": proposals,
        "pending_human_review": [],
        "human_decisions": human_decisions or {},
        "final_mappings": proposals,
        "agent_log": [],
    }


def _make_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": [f"SKU-{i:04d}" for i in range(n)],
            "product_name": [f"Product {i}" for i in range(n)],
            "brand": ["BrandA"] * n,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_lineage_row_count():
    """One LineageRecord per source row."""
    state = _make_state()
    df = _make_df(50)
    records = generate_lineage(state, df)
    assert len(records) == 50


def test_generate_lineage_target_sku_id_format():
    """target_sku_id follows MSC-{MARKET}-{row:06d} pattern."""
    state = _make_state("uk")
    df = _make_df(3)
    records = generate_lineage(state, df)
    assert records[0].target_sku_id == "MSC-UK-000000"
    assert records[2].target_sku_id == "MSC-UK-000002"


def test_generate_lineage_source_sku_id_from_mapped_column():
    """source_sku_id should be the value from the column mapped to sku_id."""
    state = _make_state()
    df = _make_df(5)
    records = generate_lineage(state, df)
    assert records[0].source_sku_id == "SKU-0000"
    assert records[4].source_sku_id == "SKU-0004"


def test_generate_lineage_field_lineage():
    """field_lineage maps target_field → source_column (inverse of MappingProposal)."""
    state = _make_state()
    df = _make_df(1)
    records = generate_lineage(state, df)
    fl = records[0].field_lineage
    assert fl["sku_id"] == "sku_id"
    assert fl["global_product_name"] == "product_name"
    assert fl["brand"] == "brand"


def test_generate_lineage_no_human_review():
    """agents_involved should NOT include 'human' when no human decisions."""
    state = _make_state(human_decisions={})
    df = _make_df(3)
    records = generate_lineage(state, df)
    assert "human" not in records[0].agents_involved


def test_generate_lineage_with_human_review():
    """agents_involved SHOULD include 'human' when human decisions were made."""
    state = _make_state(human_decisions={"product_name": "global_product_name"})
    df = _make_df(3)
    records = generate_lineage(state, df)
    assert "human" in records[0].agents_involved


def test_persist_lineage_creates_valid_jsonl():
    """Persisted JSONL must have one valid JSON object per line."""
    state = _make_state()
    df = _make_df(20)
    records = generate_lineage(state, df)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = persist_lineage(records, Path(tmp), "uk")
        assert out_path.exists()
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 20
        for line in lines:
            obj = json.loads(line)  # must not raise
            assert "source_sku_id" in obj
            assert "target_sku_id" in obj
            assert "field_lineage" in obj


def test_persist_lineage_one_record_per_row():
    """JSONL line count must equal source DataFrame row count."""
    state = _make_state()
    df = _make_df(7)
    records = generate_lineage(state, df)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = persist_lineage(records, Path(tmp), "uk")
        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(df)


# ---------------------------------------------------------------------------
# Task 2 — Executive summary tests
# ---------------------------------------------------------------------------


def _make_state_with_profile(**kwargs) -> dict:
    """State with a real MarketProfile so _build_run_summary can read quality issues."""
    col = ColumnProfile(
        column_name="sku_id",
        inferred_dtype="string",
        null_rate=0.0,
        unique_count=100,
    )
    profile = MarketProfile(
        market_id="uk",
        row_count=100,
        column_count=1,
        columns=[col],
        data_quality_issues=["high-null: some_col", "mojibake detected in desc"],
        profiling_runtime_seconds=2.5,
    )
    base = _make_state(**kwargs)
    base["profile"] = profile
    return base


def test_build_run_summary_keys():
    """Structured input dict must have all required keys for the prompt."""
    state = _make_state_with_profile()
    df = _make_df(10)
    records = generate_lineage(state, df)
    summary_input = _build_run_summary(state, records)

    required_keys = {
        "market_id",
        "row_count",
        "total_columns",
        "auto_approved_count",
        "human_reviewed_count",
        "rejected_count",
        "final_mapping_count",
        "top_quality_issues",
        "profiling_runtime_seconds",
        "unmapped_columns",
    }
    assert required_keys.issubset(summary_input.keys())


def test_build_run_summary_counts():
    """Counts in the structured dict must be arithmetically consistent."""
    state = _make_state_with_profile(human_decisions={"product_name": "global_product_name"})
    df = _make_df(10)
    records = generate_lineage(state, df)
    s = _build_run_summary(state, records)

    assert s["row_count"] == 10
    assert s["human_reviewed_count"] == 1
    assert s["rejected_count"] == 0
    # auto_approved = total_columns - len(human_decisions)
    assert s["auto_approved_count"] == s["total_columns"] - 1


def test_build_run_summary_quality_issues_capped_at_3():
    """top_quality_issues is capped at 3 entries (CDO brief stays concise)."""
    state = _make_state_with_profile()
    state["profile"].data_quality_issues = ["issue1", "issue2", "issue3", "issue4"]
    df = _make_df(5)
    records = generate_lineage(state, df)
    s = _build_run_summary(state, records)
    assert len(s["top_quality_issues"]) <= 3


def test_generate_summary_uses_llm_client():
    """generate_summary must call llm_client.create with the _SummaryOutput model."""
    from mosaic.agents.scribe import _SummaryOutput

    state = _make_state_with_profile()
    df = _make_df(5)
    records = generate_lineage(state, df)

    mock_client = MagicMock()
    mock_client.create.return_value = _SummaryOutput(
        summary="Migration of UK market completed with 3 columns auto-approved."
    )

    result = generate_summary(state, records, mock_client)

    assert mock_client.create.called
    call_kwargs = mock_client.create.call_args
    assert call_kwargs.kwargs.get("response_model") is _SummaryOutput or (
        call_kwargs.args and call_kwargs.args[0] is _SummaryOutput
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_summary_prompt_contains_run_data():
    """The message sent to the LLM must include key metrics from the run."""
    from mosaic.agents.scribe import _SummaryOutput

    state = _make_state_with_profile()
    df = _make_df(5)
    records = generate_lineage(state, df)

    captured_messages = []

    def capture_create(*args, **kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return _SummaryOutput(summary="Test summary.")

    mock_client = MagicMock()
    mock_client.create.side_effect = capture_create

    generate_summary(state, records, mock_client)

    assert captured_messages, "No messages were sent to LLM"
    combined = " ".join(m["content"] for m in captured_messages)
    assert "market_id" in combined
    assert "row_count" in combined
    assert "auto_approved_count" in combined


def test_persist_summary_creates_markdown():
    """persist_summary must write a .md file with the summary content."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = persist_summary("This is the summary.", Path(tmp), "uk")
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "This is the summary." in content
        assert "UK" in content  # market_id uppercased in header
