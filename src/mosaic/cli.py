"""Mosaic CLI — entry point for all agent subcommands.

Usage:
    python -m mosaic.cli scout --market <uk|in|br> [--no-llm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Market helpers
# ---------------------------------------------------------------------------

_MARKET_MAP: dict[str, tuple[str, str]] = {
    "uk": ("market_uk", "data/synth/market_uk.csv"),
    "in": ("market_in", "data/synth/market_in.csv"),
    "br": ("market_br", "data/synth/market_br.csv"),
}


def _resolve_market(key: str) -> tuple[str, Path]:
    """Return (market_id, csv_path) for the given market key."""
    if key not in _MARKET_MAP:
        print(f"Unknown market '{key}'. Choose from: {', '.join(_MARKET_MAP)}", file=sys.stderr)
        sys.exit(1)
    market_id, rel_path = _MARKET_MAP[key]
    # Resolve relative to project root (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / rel_path
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print("Run: python data/synth/generator.py", file=sys.stderr)
        sys.exit(1)
    return market_id, csv_path


# ---------------------------------------------------------------------------
# scout subcommand
# ---------------------------------------------------------------------------

def cmd_scout(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    from mosaic.agents.scout import profile_market

    console = Console()
    market_id, csv_path = _resolve_market(args.market)

    llm_client = None
    if not args.no_llm:
        from mosaic.llm.factory import get_llm
        llm_client = get_llm("ollama")

    with console.status(f"[bold cyan]Profiling {csv_path.name}...", spinner="dots"):
        profile = profile_market(
            csv_path,
            market_id,
            characterize=not args.no_llm,
            llm_client=llm_client,
        )

    # ---- Header panel ----
    header = (
        f"[bold]{profile.market_id}[/bold]  "
        f"[dim]|[/dim]  {profile.row_count:,} rows  "
        f"[dim]|[/dim]  {profile.column_count} columns  "
        f"[dim]|[/dim]  profiled in {profile.profiling_runtime_seconds:.2f}s"
    )
    console.print(Panel(header, title="[bold blue]SCOUT - Market Profile", expand=False))

    # ---- Quality issues ----
    if profile.data_quality_issues:
        console.print("\n[bold yellow]Data Quality Issues Detected:[/bold yellow]")
        for issue in profile.data_quality_issues:
            console.print(f"  [yellow](!)[/yellow]  {issue}")
    else:
        console.print("\n[green](ok) No market-level quality issues detected[/green]")

    # ---- Column profiles table ----
    table = Table(
        title="\nColumn Profiles",
        box=box.ROUNDED,
        show_lines=True,
        highlight=True,
    )
    table.add_column("Column", style="bold cyan", no_wrap=True)
    table.add_column("Dtype", style="dim")
    table.add_column("Null %", justify="right")
    table.add_column("Unique", justify="right")
    table.add_column("Lang", justify="center")
    table.add_column("Samples", overflow="fold", max_width=30)
    table.add_column("Flags", overflow="fold", max_width=22)
    table.add_column("LLM Description", overflow="fold", max_width=50)

    for col in profile.columns:
        null_pct = f"{col.null_rate * 100:.1f}%"
        null_style = "red" if col.null_rate > 0.5 else ("yellow" if col.null_rate > 0.2 else "")
        samples_str = ", ".join(col.value_samples[:3])
        flags_str = "\n".join(col.data_quality_flags) if col.data_quality_flags else "[dim]-[/dim]"
        desc = col.llm_description[:120] + "…" if len(col.llm_description) > 120 else col.llm_description
        if not desc:
            desc = "[dim](no LLM)[/dim]"

        table.add_row(
            col.column_name,
            col.inferred_dtype,
            Text(null_pct, style=null_style),
            str(col.unique_count),
            col.detected_language or "—",
            samples_str,
            flags_str,
            desc,
        )

    console.print(table)
    console.print(f"\n[dim]CSV:[/dim] {csv_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def cmd_atlas(args: argparse.Namespace) -> None:
    import json
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel

    from mosaic.agents.scout import profile_market
    from mosaic.agents.atlas import propose_mappings, load_target_schema

    console = Console()
    market_id, csv_path = _resolve_market(args.market)

    llm_client = None
    if not args.no_llm:
        from mosaic.llm.factory import get_llm
        llm_client = get_llm("ollama")

    with console.status(f"[bold cyan]Profiling {csv_path.name} (stats only)...", spinner="dots"):
        profile = profile_market(csv_path, market_id, characterize=False)

    llm_label = "embeddings + heuristics + LLM reasoning" if llm_client else "embeddings + heuristics"
    with console.status(f"[bold cyan]Running ATLAS ({llm_label})...", spinner="dots"):
        target_schema = load_target_schema()
        proposals = propose_mappings(profile, target_schema, llm_client=llm_client)

    auto = sum(1 for p in proposals if not p.requires_human_approval)
    review = len(proposals) - auto

    header = (
        f"[bold]{market_id}[/bold]  [dim]|[/dim]  "
        f"{len(proposals)} columns  [dim]|[/dim]  "
        f"[green]{auto} auto-approved[/green]  [dim]|[/dim]  "
        f"[yellow]{review} need review[/yellow]"
    )
    console.print(Panel(header, title="[bold blue]ATLAS - Mapping Proposals", expand=False))

    table = Table(box=box.ROUNDED, show_lines=True, highlight=True)
    table.add_column("Source Column", style="bold cyan", no_wrap=True)
    table.add_column("Target Field", style="bold green", no_wrap=True)
    table.add_column("Confidence", justify="right")
    table.add_column("Review?", justify="center")
    table.add_column("Alternatives", overflow="fold", max_width=40)
    table.add_column("Reasoning", overflow="fold", max_width=50)

    for p in proposals:
        conf_str = f"{p.confidence:.3f}"
        if p.confidence >= 0.85:
            conf_style = "green"
        elif p.confidence >= 0.65:
            conf_style = "yellow"
        else:
            conf_style = "red"

        review_str = "[yellow](!)  yes[/yellow]" if p.requires_human_approval else "[green]no[/green]"
        alts = "  ".join(f"{n} ({s:.2f})" for n, s in p.candidate_alternatives)
        reasoning = (p.reasoning[:100] + "…") if len(p.reasoning) > 100 else (p.reasoning or "[dim](none)[/dim]")

        table.add_row(
            p.source_column,
            p.target_field or "[red](unmapped)[/red]",
            f"[{conf_style}]{conf_str}[/{conf_style}]",
            review_str,
            alts,
            reasoning,
        )

    console.print(table)

    # Persist JSON output
    project_root = Path(__file__).parent.parent.parent
    out_dir = project_root / "data" / "synth"
    out_path = out_dir / f"atlas_output_{args.market}.json"
    out_path.write_text(
        json.dumps([p.model_dump() for p in proposals], indent=2, default=str),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Saved:[/dim] {out_path}")


# ---------------------------------------------------------------------------
# run subcommand (full pipeline with human-in-the-loop review)
# ---------------------------------------------------------------------------

def _prompt_decision(
    console,
    source_col: str,
    top_candidate: str,
    alternatives: list[tuple[str, float]],
    confidence: float,
    reasoning: str,
    valid_target_names: set[str],
) -> str | None:
    """Interactively prompt the user for a review decision. Returns chosen target or None."""
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    # Build candidate list for display
    all_candidates = [(top_candidate, confidence)] + list(alternatives)

    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    tbl.add_column("#", style="dim", width=3)
    tbl.add_column("Target Field", style="cyan")
    tbl.add_column("Confidence", justify="right")
    for i, (name, score) in enumerate(all_candidates[:3], start=1):
        color = "green" if score >= 0.85 else ("yellow" if score >= 0.65 else "red")
        tbl.add_row(str(i), name, f"[{color}]{score:.3f}[/{color}]")

    body = f"[bold]Source column:[/bold] [cyan]{source_col}[/cyan]\n"
    if reasoning:
        body += f"[dim]Reasoning:[/dim] {reasoning[:200]}\n"
    console.print(Panel(body + "\n", title="[yellow]Human Review Required[/yellow]", expand=False))
    console.print(tbl)
    console.print(
        "  [green][a][/green] approve top candidate  "
        "[cyan][1-3][/cyan] pick nth candidate  "
        "[magenta][c][/magenta] type custom field  "
        "[red][s][/red] skip / reject"
    )

    while True:
        try:
            raw = input("  Decision > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw in ("a", "1"):
            return all_candidates[0][0]
        if raw == "2" and len(all_candidates) >= 2:
            return all_candidates[1][0]
        if raw == "3" and len(all_candidates) >= 3:
            return all_candidates[2][0]
        if raw in ("s", "skip", "reject"):
            return None
        if raw == "c":
            custom = input("  Custom field name > ").strip()
            if custom in valid_target_names:
                return custom
            console.print(f"  [red]'{custom}' not in target schema. Valid fields:[/red]")
            console.print("  " + ", ".join(sorted(valid_target_names)))
        else:
            console.print("  [dim]Enter a, 1, 2, 3, c, or s[/dim]")


def cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline: Scout → Atlas → human review (if needed) → finalize → Scribe."""
    import uuid
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel

    from langgraph.types import Command

    from mosaic.orchestrator.graph import build_graph
    from mosaic.agents.atlas import load_target_schema

    console = Console()
    market_id, csv_path = _resolve_market(args.market)
    target_schema = load_target_schema()
    valid_target_names = {f.name for f in target_schema}

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "market_id": market_id,
        "csv_path": str(csv_path),
        "target_schema": target_schema,
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

    # ---- Phase 1: Scout + Atlas ----
    with console.status("[bold cyan]Running Scout + Atlas...[/bold cyan]", spinner="dots"):
        result = graph.invoke(initial_state, config)

    # ---- Phase 2: Human review (if interrupted) ----
    human_decisions: dict[str, str | None] = {}

    if result.get("__interrupt__"):
        pending = result.get("pending_human_review", [])
        console.print(
            Panel(
                f"[yellow]{len(pending)} mapping(s) flagged for human review[/yellow]",
                title="[bold]Human Review[/bold]",
                expand=False,
            )
        )

        for proposal in pending:
            src = proposal.source_column
            top = proposal.target_field or ""
            alts = proposal.candidate_alternatives or []
            conf = proposal.confidence
            reason = proposal.reasoning or ""
            decision = _prompt_decision(
                console, src, top, alts, conf, reason, valid_target_names
            )
            human_decisions[src] = decision

        # Resume the graph — human_review_node gets decisions as interrupt()'s return value
        with console.status("[bold cyan]Finalizing...[/bold cyan]", spinner="dots"):
            result = graph.invoke(Command(resume=human_decisions), config)
    else:
        # No review needed — graph ran straight to finalize
        pass

    # ---- Phase 3: Print results ----
    final_mappings = result.get("final_mappings", [])
    proposals = result.get("proposals", [])

    total = len(proposals)
    reviewed = len(human_decisions)
    rejected = sum(1 for v in human_decisions.values() if v is None)
    auto_approved = total - reviewed
    n_final = len(final_mappings)

    # Summary header
    header = (
        f"[bold]{market_id}[/bold]  [dim]|[/dim]  "
        f"Processed [bold]{total}[/bold] columns  [dim]|[/dim]  "
        f"auto-approved [green]{auto_approved}[/green]  [dim]|[/dim]  "
        f"human-reviewed [yellow]{reviewed}[/yellow]  [dim]|[/dim]  "
        f"rejected [red]{rejected}[/red]  [dim]|[/dim]  "
        f"final mappings [bold]{n_final}[/bold]"
    )
    console.print(Panel(header, title="[bold blue]Mosaic — Run Complete[/bold blue]", expand=False))

    # Final mapping table
    tbl = Table(box=box.ROUNDED, show_lines=True, highlight=True)
    tbl.add_column("Source Column", style="bold cyan", no_wrap=True)
    tbl.add_column("Target Field", style="bold green", no_wrap=True)
    tbl.add_column("Confidence", justify="right")
    tbl.add_column("Note", overflow="fold", max_width=30)

    for m in sorted(final_mappings, key=lambda p: p.confidence, reverse=True):
        conf = m.confidence
        color = "green" if conf >= 0.85 else ("yellow" if conf >= 0.65 else "red")
        note = ""
        if m.source_column in human_decisions:
            note = "[yellow]human-reviewed[/yellow]"
        tbl.add_row(
            m.source_column,
            m.target_field or "[red](unmapped)[/red]",
            f"[{color}]{conf:.3f}[/{color}]",
            note,
        )

    console.print(tbl)

    # ---- Lineage preview ----
    lineage_path = result.get("lineage_path", "")
    if lineage_path:
        from pathlib import Path as _Path
        lp = _Path(lineage_path)
        if lp.exists():
            lines = lp.read_text(encoding="utf-8").splitlines()
            preview = "\n".join(lines[:3])
            console.print(
                Panel(
                    f"[dim]{preview}[/dim]",
                    title=f"[bold blue]SCRIBE — Lineage Preview (first 3 of {len(lines)} records)[/bold blue]",
                    expand=False,
                )
            )
            console.print(f"[dim]Full lineage:[/dim] {lp}")

    # ---- Executive summary ----
    executive_summary = result.get("executive_summary", "")
    summary_path = result.get("summary_path", "")
    if executive_summary:
        console.print(
            Panel(
                executive_summary,
                title="[bold blue]SCRIBE — Executive Summary[/bold blue]",
                expand=False,
            )
        )
    if summary_path:
        console.print(f"[dim]Summary saved:[/dim] {summary_path}")

    # Agent log
    if args.verbose:
        console.print("\n[bold]Agent Log:[/bold]")
        for entry in result.get("agent_log", []):
            console.print(f"  [dim]{entry}[/dim]")

    console.print(
        f"\n[bold]Summary:[/bold] Processed {total} columns, "
        f"auto-approved {auto_approved}, human-reviewed {reviewed}, rejected {rejected}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosaic",
        description="Mosaic — CPG product master harmonization agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scout
    scout_p = sub.add_parser("scout", help="Profile a source market CSV (SCOUT agent)")
    scout_p.add_argument(
        "--market", required=True, choices=["uk", "in", "br"],
        help="Market to profile: uk, in, or br",
    )
    scout_p.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM characterization (stats only, much faster)",
    )
    scout_p.set_defaults(func=cmd_scout)

    # atlas
    atlas_p = sub.add_parser("atlas", help="Propose column mappings (ATLAS agent)")
    atlas_p.add_argument(
        "--market", required=True, choices=["uk", "in", "br"],
        help="Market to map: uk, in, or br",
    )
    atlas_p.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM reasoning (heuristics only, much faster)",
    )
    atlas_p.set_defaults(func=cmd_atlas)

    # run
    run_p = sub.add_parser("run", help="Full pipeline: Scout → Atlas → human review → finalize")
    run_p.add_argument(
        "--market", required=True, choices=["uk", "in", "br"],
        help="Market to process: uk, in, or br",
    )
    run_p.add_argument(
        "--verbose", action="store_true",
        help="Print agent log at end of run",
    )
    run_p.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
