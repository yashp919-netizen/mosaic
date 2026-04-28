"""MosaicState — shared state TypedDict threaded through the LangGraph pipeline."""

from __future__ import annotations

from typing import TypedDict

from mosaic.schemas import MarketProfile, MappingProposal, TargetSchemaField


class MosaicState(TypedDict):
    """All data passed between nodes in the Mosaic LangGraph state machine.

    Lifecycle:
      scout_node   → fills profile, source_df
      atlas_node   → fills proposals, pending_human_review
      human_review → fills human_decisions (via interrupt)
      finalize     → fills final_mappings
    """

    # Inputs (set by caller before graph.invoke)
    market_id: str
    csv_path: str
    target_schema: list[TargetSchemaField]

    # SCOUT output
    profile: MarketProfile | None
    # source_df is NOT stored here — DataFrames are not checkpointer-serializable.
    # Day 6 (Scribe) will re-read csv_path directly.

    # ATLAS output
    proposals: list[MappingProposal]
    pending_human_review: list[MappingProposal]  # subset where requires_human_approval is True

    # Human review
    # source_column → approved target_field string, or None to reject
    human_decisions: dict[str, str | None]

    # Final merged output after human review
    final_mappings: list[MappingProposal]

    # SCRIBE output
    lineage_path: str     # path to the persisted lineage_{market}.jsonl
    summary_path: str     # path to the persisted summary_{market}.md
    executive_summary: str  # raw summary text (also in the .md file)

    # Audit trail — each agent appends a short message
    agent_log: list[str]
