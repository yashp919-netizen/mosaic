"""Mosaic LangGraph state machine.

Nodes: scout → atlas → [human_review] → finalize
The human_review node uses LangGraph's interrupt() so the caller can gather
decisions externally and resume.  The graph is compiled with a MemorySaver
checkpointer so the interrupt is resumable across calls.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

import pandas as pd

from mosaic.agents.atlas import load_target_schema, propose_mappings
from mosaic.agents.scout import profile_market
from mosaic.agents.scribe import (
    generate_lineage,
    generate_summary,
    persist_lineage,
    persist_summary,
)
from mosaic.orchestrator.state import MosaicState
from mosaic.schemas import MappingProposal


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def scout_node(state: MosaicState) -> dict[str, Any]:
    """Profile the source CSV and populate state.profile."""
    csv_path = Path(state["csv_path"])
    market_id = state["market_id"]

    profile = profile_market(csv_path, market_id, characterize=False)

    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    log_entry = f"[{ts}] scout: profiled {market_id} — {profile.row_count} rows, {profile.column_count} cols"

    return {
        "profile": profile,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def atlas_node(state: MosaicState) -> dict[str, Any]:
    """Propose column mappings and split into auto-approved vs pending review."""
    profile = state["profile"]
    target_schema = state.get("target_schema") or load_target_schema()

    proposals = propose_mappings(profile, target_schema, llm_client=None)
    pending = [p for p in proposals if p.requires_human_approval]

    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    log_entry = (
        f"[{ts}] atlas: {len(proposals)} proposals — "
        f"{len(proposals) - len(pending)} auto-approved, {len(pending)} pending review"
    )

    return {
        "proposals": proposals,
        "pending_human_review": pending,
        "human_decisions": {},
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def human_review_node(state: MosaicState) -> dict[str, Any]:
    """Interrupt and hand control back to the caller to gather human decisions.

    The caller resumes the graph by invoking it again with the same thread_id
    and passing human_decisions in the state update.

    The interrupt payload is the list of MappingProposal dicts that need review,
    so the caller (CLI or UI) can present them to the user.
    """
    pending = state["pending_human_review"]

    # interrupt() suspends the graph here and returns the payload to the caller.
    # When resumed, execution continues from the line after interrupt().
    decisions: dict[str, str | None] = interrupt(
        {
            "pending_human_review": [p.model_dump() for p in pending],
            "message": (
                f"{len(pending)} mapping(s) need human review. "
                "Resume the graph with human_decisions populated."
            ),
        }
    )

    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    log_entry = f"[{ts}] human_review: received decisions for {len(decisions)} columns"

    return {
        "human_decisions": decisions,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def scribe_node(state: MosaicState) -> dict[str, Any]:
    """Generate per-row lineage records and an executive summary.

    Re-reads the source CSV from csv_path rather than carrying a DataFrame in
    state — DataFrames are not checkpointer-serializable.
    """
    csv_path = Path(state["csv_path"])
    market_id = state["market_id"]

    # BR market uses semicolons; UK and IN use commas
    sep = ";" if market_id == "market_br" else ","
    source_df = pd.read_csv(csv_path, sep=sep, low_memory=False)

    # Persist lineage
    project_root = Path(__file__).parent.parent.parent.parent
    synth_dir = project_root / "data" / "synth"
    short_market = market_id.split("_", 1)[1] if "_" in market_id else market_id

    records = generate_lineage(state, source_df)
    lineage_path = persist_lineage(records, synth_dir, short_market)

    # Executive summary — try LLM, fall back to structured plain text
    executive_summary = ""
    try:
        from mosaic.llm.factory import get_llm

        llm_client = get_llm("ollama")
        executive_summary = generate_summary(state, records, llm_client)
    except Exception:
        # LLM unavailable — build a minimal plain-text summary from the run data
        proposals = state.get("proposals", [])
        final = state.get("final_mappings", [])
        hd = state.get("human_decisions", {})
        auto = len(proposals) - len(hd)
        reviewed = sum(1 for v in hd.values() if v is not None)
        rejected = sum(1 for v in hd.values() if v is None)
        executive_summary = (
            f"Mosaic pipeline completed for market {short_market.upper()}. "
            f"{len(source_df):,} source rows processed across {len(proposals)} columns. "
            f"{auto} columns mapped automatically, {reviewed} required human review, "
            f"{rejected} were rejected. "
            f"{len(final)} final mappings produced. "
            f"(LLM summary unavailable — Ollama not reachable at generation time.)"
        )

    summary_path = persist_summary(executive_summary, synth_dir, short_market)

    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    log_entry = (
        f"[{ts}] scribe: {len(records)} lineage records -> {lineage_path.name}; "
        f"summary -> {summary_path.name}"
    )

    return {
        "lineage_path": str(lineage_path),
        "summary_path": str(summary_path),
        "executive_summary": executive_summary,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


def finalize_node(state: MosaicState) -> dict[str, Any]:
    """Merge human decisions into final_mappings.

    - auto-approved proposals → kept as-is
    - human-approved (same field) → kept as-is, adds 'human' to a note in reasoning
    - human-overridden (different field) → target_field replaced
    - human-rejected (None) → dropped
    """
    proposals: list[MappingProposal] = state["proposals"]
    human_decisions: dict[str, str | None] = state.get("human_decisions", {})
    human_reviewed_cols = set(human_decisions.keys())

    final: list[MappingProposal] = []

    for p in proposals:
        if p.source_column not in human_reviewed_cols:
            # Auto-approved — no human review needed
            final.append(p)
            continue

        decision = human_decisions[p.source_column]
        if decision is None:
            # Rejected — drop
            continue

        if decision == p.target_field:
            # Approved as proposed
            final.append(
                p.model_copy(
                    update={
                        "reasoning": (p.reasoning + " [human-approved]").strip(),
                    }
                )
            )
        else:
            # Human chose a different field
            final.append(
                p.model_copy(
                    update={
                        "target_field": decision,
                        "reasoning": (
                            f"Human override: changed from '{p.target_field}' to '{decision}'. "
                            + p.reasoning
                        ).strip(),
                        "requires_human_approval": False,
                    }
                )
            )

    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
    auto = len(proposals) - len(human_reviewed_cols)
    reviewed = len(human_reviewed_cols)
    rejected = sum(1 for v in human_decisions.values() if v is None)
    log_entry = (
        f"[{ts}] finalize: {len(final)} final mappings "
        f"(auto={auto}, reviewed={reviewed}, rejected={rejected})"
    )

    return {
        "final_mappings": final,
        "agent_log": state.get("agent_log", []) + [log_entry],
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def _needs_human_review(state: MosaicState) -> str:
    """Route to human_review if any proposals need it, otherwise skip to finalize."""
    if state.get("pending_human_review"):
        return "human_review"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()


def build_graph() -> Any:
    """Build and compile the Mosaic LangGraph state machine."""
    builder = StateGraph(MosaicState)

    builder.add_node("scout", scout_node)
    builder.add_node("atlas", atlas_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("scribe", scribe_node)

    builder.set_entry_point("scout")
    builder.add_edge("scout", "atlas")
    builder.add_conditional_edges(
        "atlas",
        _needs_human_review,
        {"human_review": "human_review", "finalize": "finalize"},
    )
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", "scribe")
    builder.add_edge("scribe", END)

    return builder.compile(checkpointer=_checkpointer)
