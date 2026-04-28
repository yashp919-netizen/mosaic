"""SCRIBE agent — lineage records and executive summary.

Runs after finalize_node. Reads the source DataFrame directly from csv_path
(DataFrames are not checkpointer-serializable, so we re-read rather than carry
them through LangGraph state).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from mosaic.orchestrator.state import MosaicState
from mosaic.schemas import LineageRecord, MappingProposal


# ---------------------------------------------------------------------------
# Lineage generation
# ---------------------------------------------------------------------------

def generate_lineage(
    state: MosaicState,
    source_df: pd.DataFrame,
) -> list[LineageRecord]:
    """Create one LineageRecord per row in source_df.

    field_lineage is keyed by target_field, value is source_column (the inverse
    of what Atlas stores in MappingProposal).

    transformations_applied is always empty in v1 — Alchemist (v2) will fill this
    once it generates transformation code.
    """
    final_mappings: list[MappingProposal] = state.get("final_mappings", [])
    market_id: str = state["market_id"]
    human_decisions: dict[str, str | None] = state.get("human_decisions", {})

    # Build field_lineage dict: target_field → source_column
    field_lineage: dict[str, str] = {}
    for mapping in final_mappings:
        if mapping.target_field:
            field_lineage[mapping.target_field] = mapping.source_column

    # Determine if any column was human-reviewed
    human_reviewed_cols = {col for col, decision in human_decisions.items() if decision is not None}
    any_human_reviewed = bool(human_reviewed_cols)

    agents_involved = ["scout", "atlas", "scribe"]
    if any_human_reviewed:
        agents_involved = ["scout", "atlas", "human", "scribe"]

    # Find the source column that maps to sku_id
    sku_source_col = field_lineage.get("sku_id")

    # Strip "market_" prefix once so IDs read MSC-UK-000000 not MSC-MARKET_UK-000000
    short_market = market_id.split("_", 1)[1] if market_id.startswith("market_") else market_id

    now = datetime.datetime.utcnow()
    records: list[LineageRecord] = []

    for row_index, row in source_df.iterrows():
        # source_sku_id: value from the column mapped to sku_id, fallback to row index
        if sku_source_col and sku_source_col in source_df.columns:
            source_sku_id = str(row[sku_source_col])
        else:
            source_sku_id = f"unknown-{row_index}"

        target_sku_id = f"MSC-{short_market.upper()}-{int(row_index):06d}"

        record = LineageRecord(
            source_sku_id=source_sku_id,
            source_market=market_id,
            target_sku_id=target_sku_id,
            field_lineage=field_lineage,
            agents_involved=agents_involved,
            transformations_applied=[],  # Alchemist (v2) will populate this
            timestamp=now,
        )
        records.append(record)

    return records


def persist_lineage(records: list[LineageRecord], output_dir: Path, market_id: str) -> Path:
    """Write lineage records to a JSONL file, one record per line."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"lineage_{market_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

class _SummaryOutput(BaseModel):
    """Structured LLM output for the executive summary. Single field keeps the
    LLM grounded — it cannot add fields or stray from the provided data."""

    summary: str


def _load_summary_prompt() -> str:
    """Load the summary prompt template from disk."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "scribe_summary_v1.md"
    return prompt_path.read_text(encoding="utf-8")


def _build_run_summary(state: MosaicState, lineage: list[LineageRecord]) -> dict[str, Any]:
    """Build the structured input dict that gets injected into the LLM prompt."""
    proposals: list[MappingProposal] = state.get("proposals", [])
    final_mappings: list[MappingProposal] = state.get("final_mappings", [])
    human_decisions: dict[str, str | None] = state.get("human_decisions", {})
    profile = state.get("profile")

    auto_approved = len(proposals) - len(human_decisions)
    human_reviewed = sum(1 for v in human_decisions.values() if v is not None)
    rejected = sum(1 for v in human_decisions.values() if v is None)
    unmapped = [p.source_column for p in proposals if p.target_field is None]
    quality_issues = (profile.data_quality_issues[:3] if profile else [])
    runtime = (profile.profiling_runtime_seconds if profile else 0.0)

    return {
        "market_id": state["market_id"],
        "row_count": len(lineage),
        "total_columns": len(proposals),
        "auto_approved_count": auto_approved,
        "human_reviewed_count": human_reviewed,
        "rejected_count": rejected,
        "final_mapping_count": len(final_mappings),
        "top_quality_issues": quality_issues,
        "profiling_runtime_seconds": round(runtime, 2),
        "unmapped_columns": unmapped,
    }


def generate_summary(
    state: MosaicState,
    lineage: list[LineageRecord],
    llm_client: Any,
) -> str:
    """Generate a 200-word CDO-ready executive summary grounded in run metrics.

    Uses instructor + a Pydantic output model to prevent the LLM from inventing
    numbers not present in the structured input.
    """
    run_summary = _build_run_summary(state, lineage)
    prompt_template = _load_summary_prompt()

    # Inject the structured data into the prompt
    data_block = json.dumps(run_summary, indent=2)
    user_content = prompt_template + f"\n\n---\nRUN DATA:\n```json\n{data_block}\n```"

    output: _SummaryOutput = llm_client.create(
        response_model=_SummaryOutput,
        messages=[{"role": "user", "content": user_content}],
    )
    return output.summary


def persist_summary(summary: str, output_dir: Path, market_id: str) -> Path:
    """Write the executive summary to a markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"summary_{market_id}.md"
    out_path.write_text(f"# Mosaic Executive Summary — {market_id.upper()}\n\n{summary}\n", encoding="utf-8")
    return out_path
