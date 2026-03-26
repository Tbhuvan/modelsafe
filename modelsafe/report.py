"""
Rich scan report generation for modelsafe.

Provides three output formats for ScanResult objects:
- Terminal: Rich-formatted tables for interactive CLI use.
- JSON: Fully structured machine-readable output for CI/CD pipelines.
- Markdown: PR-comment-ready report for GitHub Actions and GitLab CI.

Risk level labels:
    0.0 – 0.2  CLEAN
    0.2 – 0.4  LOW
    0.4 – 0.6  MEDIUM
    0.6 – 0.8  HIGH
    0.8 – 1.0  CRITICAL

References:
    OWASP AI Security Top 10 — ML05:2023 Supply Chain Vulnerabilities
    arXiv:2409.09368 — Models Are Codes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .scanner import ScanResult

# ---------------------------------------------------------------------------
# Risk level constants
# ---------------------------------------------------------------------------

_RISK_BANDS: list[tuple[float, str, str]] = [
    # (upper_bound, label, rich_color)
    (0.2, "CLEAN", "green"),
    (0.4, "LOW", "cyan"),
    (0.6, "MEDIUM", "yellow"),
    (0.8, "HIGH", "red"),
    (1.01, "CRITICAL", "bold red"),
]


def risk_label(score: float) -> str:
    """Map a risk score to a human-readable label.

    Args:
        score: Risk score in [0.0, 1.0].

    Returns:
        One of: CLEAN, LOW, MEDIUM, HIGH, CRITICAL.
    """
    for upper, label, _ in _RISK_BANDS:
        if score < upper:
            return label
    return "CRITICAL"


def risk_color(score: float) -> str:
    """Map a risk score to a Rich markup color string.

    Args:
        score: Risk score in [0.0, 1.0].

    Returns:
        Rich-compatible color/style string.
    """
    for upper, _, color in _RISK_BANDS:
        if score < upper:
            return color
    return "bold red"


# ---------------------------------------------------------------------------
# ScanReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScanReport:
    """Wraps a ScanResult with additional reporting context.

    Attributes:
        result: The underlying ScanResult from ModelScanner.
        scanner_version: Version string of modelsafe used for the scan.
        cve_correlations: Optional list of CVECorrelation dicts from CVECorrelator.
        weight_summary: Optional WeightSummary dict for richer weight findings.
        extra_context: Free-form dict for any additional metadata.
    """

    result: "ScanResult"
    scanner_version: str = "0.2.0"
    cve_correlations: list[dict[str, Any]] = field(default_factory=list)
    weight_summary: dict[str, Any] | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_level(self) -> str:
        """Human-readable risk level label for the wrapped result.

        Returns:
            Risk level string: CLEAN, LOW, MEDIUM, HIGH, or CRITICAL.
        """
        return risk_label(self.result.risk_score)


# ---------------------------------------------------------------------------
# Terminal format
# ---------------------------------------------------------------------------


def format_terminal(result: "ScanResult", report: ScanReport | None = None) -> None:
    """Print a rich-formatted scan report to the terminal.

    Uses Rich tables, panels, and colour-coded scores.  Writes directly to
    stdout via a Rich Console so that colour codes are handled automatically.

    Args:
        result: Completed ScanResult from ModelScanner.
        report: Optional ScanReport with additional context (CVE correlations,
            weight summary).  If None, only core result data is shown.
    """
    console = Console()
    label = risk_label(result.risk_score)
    color = risk_color(result.risk_score)

    # ------------------------------------------------------------------ header
    console.print()
    header_text = (
        f"[{color}]{label}[/{color}]  "
        f"Risk Score: [{color}]{result.risk_score:.3f}[/{color}]  |  "
        f"Model: [cyan]{result.model_id}[/cyan]"
    )
    console.print(
        Panel(
            header_text,
            title="[bold]modelsafe Scan Report[/bold]",
            subtitle=f"[dim]v{report.scanner_version if report else '0.2.0'} | {result.timestamp}[/dim]",
            border_style=color,
        )
    )

    # -------------------------------------------------------- threat DB hit
    if result.threat_db_hit and result.threat_record:
        rec = result.threat_record
        console.print(
            Panel(
                f"[bold red]KNOWN THREAT DATABASE MATCH[/bold red]\n"
                f"CVE:    {rec.get('cve_id', 'N/A')}\n"
                f"Type:   {rec.get('threat_type', 'unknown')}\n"
                f"Effect: {rec.get('effect', 'unknown')}",
                border_style="red",
                title="[bold red]THREAT MATCH[/bold red]",
            )
        )

    # ---------------------------------------------------- check scores table
    scores_table = Table(
        title="Check Scores",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )
    scores_table.add_column("Check", style="cyan", min_width=20)
    scores_table.add_column("Score / Status", justify="right", min_width=15)
    scores_table.add_column("Risk Level", justify="center", min_width=10)
    scores_table.add_column("Verdict", justify="center", min_width=8)

    def _score_cell(score: float) -> str:
        col = risk_color(score)
        return f"[{col}]{score:.3f}[/{col}]"

    def _verdict(ok: bool) -> str:
        return "[green]PASS[/green]" if ok else "[red]FAIL[/red]"

    prov_ok = result.provenance_verified
    scores_table.add_row(
        "Provenance",
        "[green]Verified[/green]" if prov_ok else "[red]Issues found[/red]",
        risk_label(0.0 if prov_ok else 0.5),
        _verdict(prov_ok),
    )
    scores_table.add_row(
        "Weight Analysis",
        _score_cell(result.weight_anomaly_score),
        risk_label(result.weight_anomaly_score),
        _verdict(result.weight_anomaly_score < 0.5),
    )
    scores_table.add_row(
        "Activation Scan",
        _score_cell(result.activation_anomaly_score),
        risk_label(result.activation_anomaly_score),
        _verdict(result.activation_anomaly_score < 0.5),
    )
    scores_table.add_row(
        "[bold]Composite Risk[/bold]",
        f"[{color}][bold]{result.risk_score:.3f}[/bold][/{color}]",
        f"[{color}]{label}[/{color}]",
        _verdict(result.safe),
    )
    console.print(scores_table)

    # --------------------------------------------------------- findings table
    if result.findings:
        findings_table = Table(
            title="Findings",
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
        )
        findings_table.add_column("Check", style="cyan", no_wrap=True, min_width=18)
        findings_table.add_column("Severity", justify="center", min_width=10)
        findings_table.add_column("Detail", min_width=40)

        sev_colors = {
            "critical": "bold red",
            "high": "red",
            "medium": "yellow",
            "low": "blue",
            "info": "dim",
        }
        for finding in result.findings:
            sev = finding.get("severity", "info")
            col = sev_colors.get(sev, "white")
            findings_table.add_row(
                finding.get("check", "?"),
                f"[{col}]{sev.upper()}[/{col}]",
                finding.get("detail", ""),
            )
        console.print(findings_table)

    # -------------------------------------------------- CVE correlations table
    if report and report.cve_correlations:
        cve_table = Table(
            title="CVE Correlations",
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
        )
        cve_table.add_column("CVE ID", style="bold", no_wrap=True, min_width=20)
        cve_table.add_column("Severity", justify="center", min_width=10)
        cve_table.add_column("Description", min_width=40)
        cve_table.add_column("Mitigation", min_width=30)

        for cve in report.cve_correlations:
            sev = cve.get("severity", "UNKNOWN").upper()
            sev_col = {
                "CRITICAL": "bold red",
                "HIGH": "red",
                "MEDIUM": "yellow",
                "LOW": "blue",
            }.get(sev, "white")
            cve_table.add_row(
                cve.get("cve_id", "?"),
                f"[{sev_col}]{sev}[/{sev_col}]",
                cve.get("description", "")[:80],
                cve.get("mitigations", [""])[0][:60] if cve.get("mitigations") else "",
            )
        console.print(cve_table)

    # --------------------------------------------------- weight summary panel
    if report and report.weight_summary:
        ws = report.weight_summary
        suspicious = ws.get("suspicious_layers", [])
        key_findings = ws.get("key_findings", [])
        if suspicious or key_findings:
            lines = []
            if suspicious:
                lines.append(f"Suspicious layers ({len(suspicious)}): {', '.join(suspicious[:5])}")
                if len(suspicious) > 5:
                    lines[-1] += f" ... (+{len(suspicious) - 5} more)"
            for kf in key_findings[:3]:
                lines.append(f"  - {kf}")
            console.print(
                Panel(
                    "\n".join(lines),
                    title="[bold yellow]Weight Analysis Summary[/bold yellow]",
                    border_style="yellow",
                )
            )

    # ----------------------------------------------------------------- footer
    console.print(
        f"\n[dim]Duration: {result.scan_duration_s:.2f}s  |  "
        f"Timestamp: {result.timestamp}  |  "
        f"Threat DB hit: {result.threat_db_hit}[/dim]\n"
    )


# ---------------------------------------------------------------------------
# JSON format
# ---------------------------------------------------------------------------


def format_json(result: "ScanResult", report: ScanReport | None = None) -> str:
    """Produce a fully structured JSON representation of a scan result.

    Includes all fields from ScanResult plus optional enrichments from
    ScanReport (CVE correlations, weight summary).

    Args:
        result: Completed ScanResult from ModelScanner.
        report: Optional ScanReport for additional context.

    Returns:
        JSON string with all findings, scores, and metadata.
    """
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "scanner_version": report.scanner_version if report else "0.2.0",
        "model_id": result.model_id,
        "safe": result.safe,
        "risk_score": result.risk_score,
        "risk_level": risk_label(result.risk_score),
        "timestamp": result.timestamp,
        "scan_duration_s": result.scan_duration_s,
        "summary": {
            "provenance_verified": result.provenance_verified,
            "weight_anomaly_score": result.weight_anomaly_score,
            "activation_anomaly_score": result.activation_anomaly_score,
            "threat_db_hit": result.threat_db_hit,
        },
        "threat_record": result.threat_record,
        "findings": result.findings,
    }

    if report:
        if report.cve_correlations:
            payload["cve_correlations"] = report.cve_correlations
        if report.weight_summary:
            payload["weight_summary"] = report.weight_summary
        if report.extra_context:
            payload["extra_context"] = report.extra_context

    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# Markdown format
# ---------------------------------------------------------------------------


def format_markdown(result: "ScanResult", report: ScanReport | None = None) -> str:
    """Generate a Markdown report suitable for CI/CD PR comments.

    Produces GitHub Flavoured Markdown with collapsible sections for
    detailed findings, making it useful in GitHub Actions PR annotations
    or GitLab merge request comments.

    Args:
        result: Completed ScanResult from ModelScanner.
        report: Optional ScanReport for additional context.

    Returns:
        Markdown string formatted for display in a GitHub/GitLab PR comment.
    """
    label = risk_label(result.risk_score)
    status_badge = "SAFE" if result.safe else "UNSAFE"
    scanner_version = report.scanner_version if report else "0.2.0"

    # Emoji-free status indicators for terminal parity
    status_indicator = "[PASS]" if result.safe else "[FAIL]"

    lines: list[str] = [
        f"# modelsafe Scan Report — `{result.model_id}`",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Status | **{status_indicator} {status_badge}** |",
        f"| Risk Score | `{result.risk_score:.3f}` |",
        f"| Risk Level | **{label}** |",
        f"| Scanner Version | `{scanner_version}` |",
        f"| Timestamp | `{result.timestamp}` |",
        f"| Duration | `{result.scan_duration_s:.2f}s` |",
        "",
        "## Check Scores",
        "",
        "| Check | Score / Status | Risk Level | Verdict |",
        "|---|---|---|---|",
        (
            f"| Provenance | "
            f"{'Verified' if result.provenance_verified else 'Issues found'} | "
            f"{'CLEAN' if result.provenance_verified else 'MEDIUM'} | "
            f"{'PASS' if result.provenance_verified else 'FAIL'} |"
        ),
        (
            f"| Weight Analysis | "
            f"`{result.weight_anomaly_score:.3f}` | "
            f"{risk_label(result.weight_anomaly_score)} | "
            f"{'PASS' if result.weight_anomaly_score < 0.5 else 'FAIL'} |"
        ),
        (
            f"| Activation Scan | "
            f"`{result.activation_anomaly_score:.3f}` | "
            f"{risk_label(result.activation_anomaly_score)} | "
            f"{'PASS' if result.activation_anomaly_score < 0.5 else 'FAIL'} |"
        ),
        (
            f"| **Composite Risk** | "
            f"**`{result.risk_score:.3f}`** | "
            f"**{label}** | "
            f"**{'PASS' if result.safe else 'FAIL'}** |"
        ),
        "",
    ]

    # Threat DB hit
    if result.threat_db_hit and result.threat_record:
        rec = result.threat_record
        lines += [
            "> **KNOWN THREAT MATCH**",
            f"> CVE: `{rec.get('cve_id', 'N/A')}`  ",
            f"> Type: {rec.get('threat_type', 'unknown')}  ",
            f"> Effect: {rec.get('effect', 'unknown')}",
            "",
        ]

    # Findings
    if result.findings:
        lines += [
            "## Findings",
            "",
            "| Check | Severity | Detail |",
            "|---|---|---|",
        ]
        for finding in result.findings:
            detail = finding.get("detail", "").replace("|", "\\|")
            lines.append(
                f"| `{finding.get('check', '?')}` "
                f"| {finding.get('severity', 'info').upper()} "
                f"| {detail} |"
            )
        lines.append("")

    # CVE correlations
    if report and report.cve_correlations:
        lines += [
            "## CVE Correlations",
            "",
            "| CVE ID | Severity | Description | Mitigation |",
            "|---|---|---|---|",
        ]
        for cve in report.cve_correlations:
            mits = cve.get("mitigations", [])
            first_mit = mits[0][:80] if mits else "See advisory"
            desc = cve.get("description", "")[:100].replace("|", "\\|")
            lines.append(
                f"| `{cve.get('cve_id', '?')}` "
                f"| {cve.get('severity', 'UNKNOWN')} "
                f"| {desc} "
                f"| {first_mit} |"
            )
        lines.append("")

    # Weight summary
    if report and report.weight_summary:
        ws = report.weight_summary
        suspicious = ws.get("suspicious_layers", [])
        key_findings = ws.get("key_findings", [])
        if suspicious or key_findings:
            lines += ["## Weight Analysis Details", ""]
            if suspicious:
                lines.append(f"**Suspicious layers**: {', '.join(f'`{l}`' for l in suspicious[:8])}")
                if len(suspicious) > 8:
                    lines.append(f"_(+{len(suspicious) - 8} more)_")
            if key_findings:
                lines.append("")
                lines.append("**Key findings:**")
                for kf in key_findings:
                    lines.append(f"- {kf}")
            lines.append("")

    lines += [
        "---",
        f"_Generated by [modelsafe](https://github.com/Tbhuvan/modelsafe) "
        f"v{scanner_version}_",
    ]

    return "\n".join(lines)
