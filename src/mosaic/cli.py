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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
