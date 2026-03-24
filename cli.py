"""
modelsafe CLI — scan HuggingFace models for security threats.

Usage:
    modelsafe scan gpt2
    modelsafe scan microsoft/phi-2 --output json --verbose
    modelsafe scan /path/to/local/model --output markdown
    modelsafe list-threats

This CLI is installable as a console_scripts entry point via pyproject.toml.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from modelsafe.scanner import ModelScanner, ScanResult
from modelsafe.threat_db import ThreatDatabase

console = Console()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="modelsafe")
def cli() -> None:
    """modelsafe — ML Model Supply Chain Scanner.

    Detect backdoored and malicious HuggingFace models before loading them.
    """


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("model_id")
@click.option(
    "--output",
    "-o",
    default="terminal",
    type=click.Choice(["terminal", "json", "markdown"], case_sensitive=False),
    help="Output format.",
    show_default=True,
)
@click.option(
    "--local-path",
    "-l",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Local model directory (skips HF Hub download for weight analysis).",
)
@click.option(
    "--skip-activation-scan",
    is_flag=True,
    default=False,
    help="Skip activation scan (faster, misses behavioural anomalies).",
)
@click.option(
    "--hf-token",
    envvar="HF_TOKEN",
    default=None,
    help="HuggingFace API token (or set HF_TOKEN env var).",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose output.")
def scan(
    model_id: str,
    output: str,
    local_path: Optional[str],
    skip_activation_scan: bool,
    hf_token: Optional[str],
    verbose: bool,
) -> None:
    """Scan a HuggingFace model for security threats.

    MODEL_ID can be a HuggingFace model identifier (e.g. 'gpt2',
    'microsoft/phi-2') or a local path label when --local-path is provided.

    Examples:

    \b
        modelsafe scan gpt2
        modelsafe scan microsoft/phi-2 --output json
        modelsafe scan my-model --local-path /models/my-model
    """
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if not model_id:
        console.print("[red]Error: MODEL_ID cannot be empty.[/red]")
        sys.exit(1)

    scanner = ModelScanner(
        skip_activation_scan=skip_activation_scan,
        hf_token=hf_token,
    )

    with console.status(f"[bold cyan]Scanning {model_id}...[/bold cyan]"):
        try:
            result = scanner.scan(model_id, local_path=local_path)
        except Exception as exc:
            console.print(f"[red]Scan failed: {exc}[/red]")
            sys.exit(2)

    _render_result(result, output.lower())

    # Exit code 1 if unsafe
    if not result.safe:
        sys.exit(1)


def _render_result(result: ScanResult, output_format: str) -> None:
    """Render a scan result in the requested format.

    Args:
        result: The completed scan result.
        output_format: One of "terminal", "json", "markdown".
    """
    if output_format == "json":
        click.echo(result.to_json())
        return

    if output_format == "markdown":
        click.echo(_to_markdown(result))
        return

    # Terminal (rich) output
    _render_terminal(result)


def _render_terminal(result: ScanResult) -> None:
    """Render a rich-formatted terminal report.

    Args:
        result: Completed scan result.
    """
    # Header
    status_color = "green" if result.safe else "red"
    status_text = "SAFE" if result.safe else "UNSAFE"
    console.print()
    console.print(
        Panel(
            f"[bold {status_color}]{status_text}[/bold {status_color}]  "
            f"Risk score: [bold]{result.risk_score:.3f}[/bold]  |  "
            f"Model: [cyan]{result.model_id}[/cyan]",
            title="modelsafe Scan Result",
            border_style=status_color,
        )
    )

    if result.threat_db_hit and result.threat_record:
        console.print(
            Panel(
                f"[bold red]KNOWN THREAT MATCH[/bold red]\n"
                f"CVE: {result.threat_record.get('cve_id', 'N/A')}\n"
                f"Type: {result.threat_record.get('threat_type', '?')}\n"
                f"Effect: {result.threat_record.get('effect', '?')}",
                border_style="red",
            )
        )

    # Scores table
    scores_table = Table(title="Check Scores", show_header=True, header_style="bold cyan")
    scores_table.add_column("Check", style="cyan")
    scores_table.add_column("Score / Status", justify="right")
    scores_table.add_column("Verdict", justify="center")

    def _score_cell(score: float) -> str:
        if score < 0.3:
            return f"[green]{score:.3f}[/green]"
        if score < 0.7:
            return f"[yellow]{score:.3f}[/yellow]"
        return f"[red]{score:.3f}[/red]"

    prov_status = "[green]Verified[/green]" if result.provenance_verified else "[red]Issues found[/red]"
    scores_table.add_row("Provenance", prov_status, "OK" if result.provenance_verified else "WARN")
    scores_table.add_row("Weight Analysis", _score_cell(result.weight_anomaly_score),
                         "OK" if result.weight_anomaly_score < 0.5 else "FAIL")
    scores_table.add_row("Activation Scan", _score_cell(result.activation_anomaly_score),
                         "OK" if result.activation_anomaly_score < 0.5 else "FAIL")
    scores_table.add_row("[bold]Composite Risk[/bold]", _score_cell(result.risk_score),
                         "[green]PASS[/green]" if result.safe else "[red]FAIL[/red]")
    console.print(scores_table)

    # Findings table
    if result.findings:
        findings_table = Table(title="Findings", show_header=True, header_style="bold")
        findings_table.add_column("Check", style="cyan", no_wrap=True)
        findings_table.add_column("Severity", justify="center")
        findings_table.add_column("Detail")

        sev_colors = {"critical": "bold red", "high": "red", "medium": "yellow",
                      "low": "blue", "info": "dim"}
        for f in result.findings:
            sev = f.get("severity", "info")
            color = sev_colors.get(sev, "white")
            findings_table.add_row(
                f.get("check", "?"),
                f"[{color}]{sev.upper()}[/{color}]",
                f.get("detail", ""),
            )
        console.print(findings_table)

    console.print(
        f"\n[dim]Scanned in {result.scan_duration_s:.2f}s  |  "
        f"Timestamp: {result.timestamp}[/dim]\n"
    )


def _to_markdown(result: ScanResult) -> str:
    """Generate a Markdown-formatted scan report.

    Args:
        result: Completed scan result.

    Returns:
        Markdown string.
    """
    status = "SAFE" if result.safe else "UNSAFE"
    lines = [
        f"# modelsafe Scan Report — {result.model_id}",
        "",
        f"**Status:** {status}  ",
        f"**Risk Score:** {result.risk_score:.3f}  ",
        f"**Timestamp:** {result.timestamp}  ",
        f"**Duration:** {result.scan_duration_s:.2f}s",
        "",
        "## Scores",
        "",
        "| Check | Score | Verdict |",
        "|---|---|---|",
        f"| Provenance | {'Verified' if result.provenance_verified else 'Issues'} | {'OK' if result.provenance_verified else 'WARN'} |",
        f"| Weight Analysis | {result.weight_anomaly_score:.3f} | {'OK' if result.weight_anomaly_score < 0.5 else 'FAIL'} |",
        f"| Activation Scan | {result.activation_anomaly_score:.3f} | {'OK' if result.activation_anomaly_score < 0.5 else 'FAIL'} |",
        f"| **Composite Risk** | **{result.risk_score:.3f}** | **{'PASS' if result.safe else 'FAIL'}** |",
        "",
    ]

    if result.findings:
        lines += ["## Findings", "", "| Check | Severity | Detail |", "|---|---|---|"]
        for f in result.findings:
            lines.append(f"| {f.get('check', '?')} | {f.get('severity', 'info').upper()} | {f.get('detail', '')} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# list-threats command
# ---------------------------------------------------------------------------


@cli.command("list-threats")
@click.option(
    "--db-path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Custom threat database path.",
)
@click.option(
    "--output",
    "-o",
    default="terminal",
    type=click.Choice(["terminal", "json"], case_sensitive=False),
    help="Output format.",
)
def list_threats(db_path: Optional[str], output: str) -> None:
    """List all known threats in the threat database."""
    db = ThreatDatabase(db_path=db_path) if db_path else ThreatDatabase()
    threats = db.list_threats()

    if output == "json":
        click.echo(json.dumps({"threats": threats, "count": len(threats)}, indent=2))
        return

    if not threats:
        console.print("[yellow]No known threats in database.[/yellow]")
        return

    table = Table(
        title=f"Known Threats ({len(threats)} total, v{db.database_version()})",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("CVE ID", style="bold red", no_wrap=True)
    table.add_column("Model ID", style="cyan")
    table.add_column("Type")
    table.add_column("Reported")
    table.add_column("Hash (first 16)")

    for t in threats:
        table.add_row(
            t.get("cve_id", "N/A"),
            t.get("model_id", "?"),
            t.get("threat_type", "?"),
            t.get("reported", "?"),
            t.get("hash", "")[:16] + "...",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
