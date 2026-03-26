"""
modelsafe CLI — scan HuggingFace models for security threats.

Usage:
    modelsafe scan gpt2
    modelsafe scan microsoft/phi-2 --format json --verbose
    modelsafe scan /path/to/local/model --format markdown
    modelsafe scan gpt2 --skip-weight-analysis --skip-activation-scan
    modelsafe list-threats
    modelsafe correlate-cves bert-base-uncased
    modelsafe correlate-cves meta-llama/Llama-2-7b-hf --architecture llama

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

from modelsafe.cve_correlator import CVECorrelation, CVECorrelator
from modelsafe.report import ScanReport, format_json, format_markdown, format_terminal
from modelsafe.scanner import ModelScanner, ScanResult
from modelsafe.threat_db import ThreatDatabase

console = Console()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.2.0", prog_name="modelsafe")
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
    "--format",
    "-f",
    "output_format",
    default="terminal",
    type=click.Choice(["terminal", "json", "markdown"], case_sensitive=False),
    help="Output format.",
    show_default=True,
)
@click.option(
    "--output",
    "-o",
    "output_format_legacy",
    default=None,
    hidden=True,  # backward compat alias for --format
    type=click.Choice(["terminal", "json", "markdown"], case_sensitive=False),
    help="[Deprecated] Use --format instead.",
)
@click.option(
    "--local-path",
    "-l",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Local model directory (skips HF Hub download for weight analysis).",
)
@click.option(
    "--skip-weight-analysis",
    is_flag=True,
    default=False,
    help="Skip weight analysis (provenance-only mode; much faster).",
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
    output_format: str,
    output_format_legacy: Optional[str],
    local_path: Optional[str],
    skip_weight_analysis: bool,
    skip_activation_scan: bool,
    hf_token: Optional[str],
    verbose: bool,
) -> None:
    """Scan a HuggingFace model for security threats.

    MODEL_ID can be a HuggingFace model identifier (e.g. 'gpt2',
    'microsoft/phi-2') or a local path label when --local-path is provided.

    Use --skip-weight-analysis and --skip-activation-scan for fast
    provenance-only scans that require no model download.

    Examples:

    \b
        modelsafe scan gpt2
        modelsafe scan microsoft/phi-2 --format json
        modelsafe scan my-model --local-path /models/my-model
        modelsafe scan gpt2 --skip-weight-analysis --skip-activation-scan
    """
    # Support legacy --output flag
    effective_format = output_format_legacy or output_format

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if not model_id:
        console.print("[red]Error: MODEL_ID cannot be empty.[/red]")
        sys.exit(1)

    # If skip-weight-analysis is set, we must also not pass local_path to the
    # weight analysis step.  We achieve this by clearing local_path for weight
    # purposes via a wrapper scanner that sets the flag.
    scanner = ModelScanner(
        skip_activation_scan=skip_activation_scan or skip_weight_analysis,
        hf_token=hf_token,
    )

    with console.status(f"[bold cyan]Scanning {model_id}...[/bold cyan]"):
        try:
            # Pass local_path=None when skipping weight analysis so no weights
            # are loaded; provenance still runs via model_id.
            effective_path = None if skip_weight_analysis else local_path
            result = scanner.scan(model_id, local_path=effective_path)
        except Exception as exc:
            console.print(f"[red]Scan failed: {exc}[/red]")
            sys.exit(2)

    # Also run CVE correlation if a model_id is available
    correlator = CVECorrelator()
    cve_results = correlator.correlate(
        model_id=model_id,
        architecture=CVECorrelator._normalise_arch(model_id),
    )
    report = ScanReport(
        result=result,
        cve_correlations=[c.to_dict() for c in cve_results],
    )

    _render_result(result, effective_format.lower(), report=report)

    # Exit code 1 if unsafe
    if not result.safe:
        sys.exit(1)


def _render_result(
    result: ScanResult,
    output_format: str,
    report: ScanReport | None = None,
) -> None:
    """Render a scan result in the requested format.

    Args:
        result: The completed scan result.
        output_format: One of "terminal", "json", "markdown".
        report: Optional ScanReport with enriched context.
    """
    if output_format == "json":
        click.echo(format_json(result, report))
        return

    if output_format == "markdown":
        click.echo(format_markdown(result, report))
        return

    # Terminal (rich) output — delegates to report.py
    format_terminal(result, report)


# ---------------------------------------------------------------------------
# correlate-cves command
# ---------------------------------------------------------------------------


@cli.command("correlate-cves")
@click.argument("model_id")
@click.option(
    "--architecture",
    "-a",
    default="",
    help=(
        "Model architecture name for matching (e.g. 'bert', 'llama', 'gpt2'). "
        "Auto-inferred from model_id if not provided."
    ),
)
@click.option(
    "--output",
    "-o",
    default="terminal",
    type=click.Choice(["terminal", "json"], case_sensitive=False),
    help="Output format.",
    show_default=True,
)
def correlate_cves(model_id: str, architecture: str, output: str) -> None:
    """Correlate a model against the built-in ML CVE database.

    MODEL_ID is used for logging and architecture auto-detection.

    Examples:

    \b
        modelsafe correlate-cves bert-base-uncased
        modelsafe correlate-cves meta-llama/Llama-2-7b-hf --architecture llama
        modelsafe correlate-cves gpt2 --output json
    """
    if not model_id:
        console.print("[red]Error: MODEL_ID cannot be empty.[/red]")
        sys.exit(1)

    correlator = CVECorrelator()
    arch = architecture or CVECorrelator._normalise_arch(model_id)

    with console.status(f"[bold cyan]Correlating CVEs for {model_id}...[/bold cyan]"):
        cve_results = correlator.correlate(
            model_id=model_id,
            architecture=arch,
        )

    if output == "json":
        payload = {
            "model_id": model_id,
            "architecture": arch,
            "n_cves_matched": len(cve_results),
            "cve_correlations": [c.to_dict() for c in cve_results],
        }
        click.echo(json.dumps(payload, indent=2))
        return

    # Terminal output
    console.print()
    if not cve_results:
        console.print(
            Panel(
                f"[green]No CVEs matched for [bold]{model_id}[/bold] "
                f"(architecture: {arch or 'unknown'}).[/green]",
                title="CVE Correlation",
                border_style="green",
            )
        )
        return

    console.print(
        Panel(
            f"Model: [cyan]{model_id}[/cyan]  |  "
            f"Architecture: [cyan]{arch or 'auto-detected'}[/cyan]  |  "
            f"CVEs matched: [bold yellow]{len(cve_results)}[/bold yellow]",
            title="CVE Correlation Results",
            border_style="yellow",
        )
    )

    sev_colors = {
        "CRITICAL": "bold red",
        "HIGH": "red",
        "MEDIUM": "yellow",
        "LOW": "blue",
    }

    for cve in cve_results:
        sev = cve.severity.upper()
        col = sev_colors.get(sev, "white")

        # Short description + first mitigation
        body_lines = [
            f"[{col}]{sev}[/{col}]  CVSS: {cve.cvss_score or 'N/A'}",
            f"",
            cve.description[:300],
            f"",
            f"[bold]Affected when:[/bold] {cve.affected_condition[:200]}",
        ]
        if cve.mitigations:
            body_lines += [f"", f"[bold]Mitigations:[/bold]"]
            for m in cve.mitigations[:3]:
                body_lines.append(f"  - {m}")

        console.print(
            Panel(
                "\n".join(body_lines),
                title=f"[bold]{cve.cve_id}[/bold]",
                border_style=col.replace("bold ", ""),
            )
        )

    console.print(
        f"\n[dim]Total: {len(cve_results)} CVEs | "
        f"Database contains {correlator.cve_count()} entries[/dim]\n"
    )


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
