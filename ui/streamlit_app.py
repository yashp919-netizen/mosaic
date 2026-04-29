"""Mosaic Streamlit UI — Day 10."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src/ is on the path when running as a script
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKET_OPTIONS = ["UK", "India", "Brazil"]
MARKET_ID_MAP = {"UK": "market_uk", "India": "market_in", "Brazil": "market_br"}
MARKET_KEY_MAP = {"UK": "uk", "India": "in", "Brazil": "br"}

LLM_OPTIONS = ["Local (Ollama)", "Cloud (Gemini)", "No LLM (heuristics only)"]
LLM_PROVIDER_MAP = {
    "Local (Ollama)": "ollama",
    "Cloud (Gemini)": "gemini",
    "No LLM (heuristics only)": "",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Mosaic",
    page_icon="🔮",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: dict = {
    "stage": "idle",          # idle | running | awaiting_review | resuming | complete | error
    "thread_id": None,
    "graph": None,
    "graph_config": None,
    "result": None,
    "pending_decisions": None,
    "error_msg": None,
    "last_market": "UK",
    "last_llm": "No LLM (heuristics only)",
}

for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _run_phase1(market: str, llm_option: str) -> None:
    """Run Scout + Atlas and store result in session_state."""
    from mosaic.agents.atlas import load_target_schema
    from mosaic.orchestrator.graph import build_graph

    market_id = MARKET_ID_MAP[market]
    market_key = MARKET_KEY_MAP[market]
    csv_path = PROJECT_ROOT / "data" / "synth" / f"market_{market_key}.csv"
    llm_provider = LLM_PROVIDER_MAP[llm_option]

    target_schema = load_target_schema()
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "market_id": market_id,
        "csv_path": str(csv_path),
        "target_schema": target_schema,
        "llm_provider": llm_provider,
        "profile": None,
        "proposals": [],
        "pending_human_review": [],
        "human_decisions": {},
        "final_mappings": [],
        "lineage_path": "",
        "summary_path": "",
        "executive_summary": "",
        "agent_log": [],
    }

    result = graph.invoke(initial_state, config)

    st.session_state["thread_id"] = thread_id
    st.session_state["graph"] = graph
    st.session_state["graph_config"] = config
    st.session_state["result"] = result

    if result.get("__interrupt__"):
        st.session_state["stage"] = "awaiting_review"
    else:
        st.session_state["stage"] = "complete"


def _run_phase2(decisions: dict) -> None:
    """Resume the interrupted graph with human decisions."""
    from langgraph.types import Command

    graph = st.session_state["graph"]
    config = st.session_state["graph_config"]
    result = graph.invoke(Command(resume=decisions), config)
    st.session_state["result"] = result
    st.session_state["stage"] = "complete"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> tuple[str, str, bool]:
    with st.sidebar:
        st.title("🔮 Mosaic")
        st.caption("Multi-agent CPG product master harmonization")
        st.divider()

        market = st.radio("**Market**", MARKET_OPTIONS, index=0)
        st.divider()
        llm_option = st.radio("**LLM Provider**", LLM_OPTIONS, index=2)
        st.divider()

        run_clicked = st.button(
            "▶ Run Mosaic",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["stage"] in ("running", "resuming"),
        )

        if st.session_state["stage"] not in ("idle",):
            if st.button("↺ Reset", use_container_width=True):
                for k, v in _STATE_DEFAULTS.items():
                    st.session_state[k] = v
                st.rerun()

        st.divider()
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-yashp919--netizen-181717?logo=github)]"
            "(https://github.com/yashp919-netizen)"
        )

    return market, llm_option, run_clicked


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_scout(result: dict) -> None:
    profile = result.get("profile")
    if not profile:
        return

    with st.expander("🔍 Scout Results", expanded=True):
        rows = []
        for col in profile.columns:
            rows.append(
                {
                    "Column Name": col.column_name,
                    "Dtype": col.inferred_dtype,
                    "Null Rate": f"{col.null_rate:.1%}",
                    "Language": col.detected_language or "—",
                    "Flags": ", ".join(col.data_quality_flags) if col.data_quality_flags else "—",
                    "Description": col.llm_description or "—",
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if profile.data_quality_issues:
            st.markdown("**Market-level data quality issues:**")
            for issue in profile.data_quality_issues:
                st.markdown(f"- ⚠️ {issue}")
        else:
            st.success("No market-level data quality issues detected.")

        st.caption(f"Profiling runtime: {profile.profiling_runtime_seconds:.2f}s")


def _render_atlas(result: dict) -> None:
    proposals = result.get("proposals") or []
    if not proposals:
        return

    pending = result.get("pending_human_review") or []
    auto_count = len(proposals) - len(pending)

    with st.expander("🗺️ Atlas Mappings", expanded=True):
        # Build table
        rows = []
        for p in proposals:
            rows.append(
                {
                    "Source Column": p.source_column,
                    "Target Field": p.target_field or "(unmapped)",
                    "Confidence": p.confidence,
                    "Review?": "⚠️ Yes" if p.requires_human_approval else "✅ No",
                }
            )
        df = pd.DataFrame(rows)

        def _color_conf(val: float) -> str:
            if val >= 0.85:
                return "background-color:#d4edda;color:#155724"
            if val >= 0.65:
                return "background-color:#fff3cd;color:#856404"
            return "background-color:#f8d7da;color:#721c24"

        styled = df.style.map(_color_conf, subset=["Confidence"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Confidence distribution
        bins = {"High ≥0.85": 0, "Mid 0.65–0.84": 0, "Low <0.65": 0}
        for p in proposals:
            if p.confidence >= 0.85:
                bins["High ≥0.85"] += 1
            elif p.confidence >= 0.65:
                bins["Mid 0.65–0.84"] += 1
            else:
                bins["Low <0.65"] += 1
        chart_df = pd.DataFrame(
            {"Bucket": list(bins.keys()), "Count": list(bins.values())}
        ).set_index("Bucket")
        st.bar_chart(chart_df)

        st.markdown(f"**{auto_count}** auto-approved &nbsp;|&nbsp; **{len(pending)}** need review")


def _render_human_review(result: dict) -> dict | None:
    """Render the human review form. Returns decisions dict when submitted, else None."""
    pending = result.get("pending_human_review") or []
    if not pending:
        return None

    # Build a lookup: source_column → ColumnProfile (for samples)
    profile = result.get("profile")
    col_profile_map = {}
    if profile:
        col_profile_map = {c.column_name: c for c in profile.columns}

    with st.expander("👤 Human Review", expanded=True):
        st.warning(f"{len(pending)} mapping(s) require your decision before Scribe can run.")

        decisions: dict[str, str | None] = {}

        for i, proposal in enumerate(pending):
            src = proposal.source_column
            cp = col_profile_map.get(src)

            st.markdown(f"#### Column: `{src}`")
            left, right = st.columns([1, 2])

            with left:
                if cp:
                    st.caption(f"Samples: {', '.join(cp.value_samples[:3])}")
                    if cp.llm_description:
                        st.caption(f"Description: {cp.llm_description[:180]}")
                if proposal.reasoning:
                    st.caption(f"Reasoning: {proposal.reasoning[:200]}")

            with right:
                # Build ordered candidate list (top + up to 2 alternatives)
                candidates: list[tuple[str, float]] = []
                if proposal.target_field:
                    candidates.append((proposal.target_field, proposal.confidence))
                for alt_name, alt_score in (proposal.candidate_alternatives or [])[:2]:
                    if alt_name and alt_name != proposal.target_field:
                        candidates.append((alt_name, alt_score))

                option_labels = [
                    f"{'✅' if j == 0 else '🔹'} {name}  ({score:.3f})"
                    for j, (name, score) in enumerate(candidates)
                ] + ["✏️ Custom field", "❌ Reject"]

                choice = st.radio(
                    "Decision",
                    option_labels,
                    key=f"hr_choice_{i}",
                    label_visibility="collapsed",
                )

                if choice == "❌ Reject":
                    decisions[src] = None
                elif choice == "✏️ Custom field":
                    custom = st.text_input("Custom target field:", key=f"hr_custom_{i}")
                    decisions[src] = custom.strip() or None
                else:
                    idx = option_labels.index(choice)
                    decisions[src] = candidates[idx][0]

            st.divider()

        submitted = st.button("✅ Submit Decisions", type="primary")
        if submitted:
            return decisions

    return None


def _render_scribe(result: dict) -> None:
    executive_summary = result.get("executive_summary", "")
    lineage_path_str = result.get("lineage_path", "")
    proposals = result.get("proposals") or []
    human_decisions = result.get("human_decisions") or {}
    profile = result.get("profile")

    if not executive_summary and not lineage_path_str:
        return

    with st.expander("📋 Scribe Output", expanded=True):
        if executive_summary:
            st.markdown("**Executive Summary**")
            st.markdown(executive_summary)

        st.divider()

        # Download button
        if lineage_path_str:
            lp = Path(lineage_path_str)
            if lp.exists():
                st.download_button(
                    "⬇️ Download Lineage JSONL",
                    data=lp.read_bytes(),
                    file_name=lp.name,
                    mime="application/jsonlines",
                )

        # Run statistics
        auto_count = len(proposals) - len(human_decisions)
        reviewed_count = sum(1 for v in human_decisions.values() if v is not None)
        c1, c2, c3 = st.columns(3)
        if profile:
            c1.metric("Source Rows", f"{profile.row_count:,}")
        c2.metric("Auto-approved", auto_count)
        c3.metric("Human-reviewed", reviewed_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    market, llm_option, run_clicked = _render_sidebar()

    st.title("🔮 Mosaic")
    st.markdown(
        "*Multi-agent AI reference architecture for CPG product master harmonization "
        "during cloud migration*"
    )
    st.divider()

    stage = st.session_state["stage"]
    result = st.session_state.get("result") or {}

    # ---- Trigger: Run Mosaic button ----
    if run_clicked:
        for k, v in _STATE_DEFAULTS.items():
            st.session_state[k] = v
        st.session_state["stage"] = "running"
        st.session_state["last_market"] = market
        st.session_state["last_llm"] = llm_option
        st.rerun()

    # ---- Stage: running ----
    if stage == "running":
        with st.status(
            f"Running Mosaic · {st.session_state['last_market']} · "
            f"{st.session_state['last_llm']}",
            expanded=True,
        ) as status:
            st.write("🔍 Scout: profiling source data…")
            st.write("🗺️ Atlas: proposing column mappings…")
            try:
                _run_phase1(
                    st.session_state["last_market"],
                    st.session_state["last_llm"],
                )
                new_stage = st.session_state["stage"]
                if new_stage == "awaiting_review":
                    status.update(label="Scout + Atlas complete — human review needed", state="complete")
                else:
                    status.update(label="Pipeline complete!", state="complete")
            except Exception as exc:
                st.session_state["stage"] = "error"
                st.session_state["error_msg"] = str(exc)
                status.update(label="Pipeline error", state="error")
        st.rerun()

    # ---- Stage: resuming ----
    elif stage == "resuming":
        with st.status("Finalizing decisions and running Scribe…", expanded=True) as status:
            try:
                _run_phase2(st.session_state["pending_decisions"] or {})
                status.update(label="Pipeline complete!", state="complete")
            except Exception as exc:
                st.session_state["stage"] = "error"
                st.session_state["error_msg"] = str(exc)
                status.update(label="Error", state="error")
        st.rerun()

    # ---- Stage: error ----
    elif stage == "error":
        st.error(f"**Pipeline error:** {st.session_state.get('error_msg', 'Unknown error')}")
        st.info("Click **↺ Reset** in the sidebar to start over.")

    # ---- Stages: awaiting_review / complete ----
    elif stage in ("awaiting_review", "complete"):
        _render_scout(result)
        _render_atlas(result)

        if stage == "awaiting_review":
            decisions = _render_human_review(result)
            if decisions is not None:
                st.session_state["pending_decisions"] = decisions
                st.session_state["stage"] = "resuming"
                st.rerun()
        else:
            # Section 3 only shown if there were human reviews
            if result.get("human_decisions"):
                _render_human_review_summary(result)
            _render_scribe(result)

    # ---- Stage: idle ----
    else:
        st.info(
            "Select a **Market** and **LLM Provider** in the sidebar, "
            "then click **▶ Run Mosaic** to start the pipeline."
        )


def _render_human_review_summary(result: dict) -> None:
    """After completion, show a collapsed summary of human review decisions."""
    human_decisions = result.get("human_decisions") or {}
    if not human_decisions:
        return

    with st.expander("👤 Human Review (completed)", expanded=False):
        rows = [
            {
                "Column": col,
                "Decision": target if target else "❌ Rejected",
            }
            for col, target in human_decisions.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
