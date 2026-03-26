"""
Benchmark script: scan a set of well-known, publicly trusted HuggingFace
models using provenance check and CVE correlation only.

Weight analysis and activation scanning are intentionally skipped to keep
the benchmark fast and dependency-free (no GPU or model download needed).

Models scanned:
    - distilbert-base-uncased  (Hugging Face official)
    - bert-base-uncased        (Google official, hosted on HF)
    - gpt2                     (OpenAI official, hosted on HF)

For each model:
    1. Provenance check (HF API call — can be skipped with --offline).
    2. CVE correlation (local, no network).
    3. Results printed as a Rich table and saved to
       experiments/results/known_models_scan.json.

Usage:
    python experiments/scan_known_models.py
    python experiments/scan_known_models.py --offline
    python experiments/scan_known_models.py --hf-token $HF_TOKEN

The --offline flag skips all HTTP calls and returns synthetic provenance
results so the script can be run in air-gapped CI environments.

References:
    arXiv:2409.09368 — Models Are Codes
    OWASP AI Security Top 10 — ML05:2023 Supply Chain
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Resolve project root so the script works from any cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console
from rich.table import Table

from modelsafe.cve_correlator import CVECorrelator
from modelsafe.provenance import ProvenanceChecker, ProvenanceCheckError

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------

BENCHMARK_MODELS: list[dict[str, str]] = [
    {
        "model_id": "distilbert-base-uncased",
        "architecture": "distilbert",
        "framework": "transformers",
        "description": "Hugging Face official DistilBERT base uncased",
        "expected_author": "distilbert",
    },
    {
        "model_id": "bert-base-uncased",
        "architecture": "bert",
        "framework": "transformers",
        "description": "Google BERT base uncased (HF hosted)",
        "expected_author": "google-bert",
    },
    {
        "model_id": "gpt2",
        "architecture": "gpt2",
        "framework": "transformers",
        "description": "OpenAI GPT-2 (HF hosted)",
        "expected_author": "openai-community",
    },
]

# ---------------------------------------------------------------------------
# Synthetic offline results
# ---------------------------------------------------------------------------

_OFFLINE_PROVENANCE: dict[str, dict[str, Any]] = {
    "distilbert-base-uncased": {
        "verified": True,
        "risk_factors": [],
        "author": "distilbert",
        "downloads": 25_000_000,
        "has_model_card": True,
        "architecture_match": True,
        "last_modified": "2023-11-01T00:00:00Z",
        "author_reputation": {"followers": 5000, "model_count": 12, "is_new_account": False},
        "raw_metadata": {"pipeline_tag": "fill-mask", "tags": ["bert", "pytorch"]},
    },
    "bert-base-uncased": {
        "verified": True,
        "risk_factors": [],
        "author": "google-bert",
        "downloads": 50_000_000,
        "has_model_card": True,
        "architecture_match": True,
        "last_modified": "2023-08-15T00:00:00Z",
        "author_reputation": {"followers": 12000, "model_count": 48, "is_new_account": False},
        "raw_metadata": {"pipeline_tag": "fill-mask", "tags": ["bert", "pytorch"]},
    },
    "gpt2": {
        "verified": True,
        "risk_factors": [],
        "author": "openai-community",
        "downloads": 100_000_000,
        "has_model_card": True,
        "architecture_match": True,
        "last_modified": "2023-06-01T00:00:00Z",
        "author_reputation": {"followers": 30000, "model_count": 20, "is_new_account": False},
        "raw_metadata": {"pipeline_tag": "text-generation", "tags": ["gpt2", "pytorch"]},
    },
}


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------


def run_provenance_check(
    model_id: str,
    checker: ProvenanceChecker,
    offline: bool,
) -> dict[str, Any]:
    """Run provenance check for a single model.

    Args:
        model_id: HuggingFace model ID.
        checker: ProvenanceChecker instance.
        offline: If True, return synthetic offline results.

    Returns:
        Provenance check result dict.
    """
    if offline:
        return _OFFLINE_PROVENANCE.get(
            model_id,
            {
                "verified": True,
                "risk_factors": [],
                "author": "unknown",
                "downloads": 0,
                "has_model_card": True,
                "architecture_match": True,
                "last_modified": "",
                "author_reputation": {},
                "raw_metadata": {},
            },
        )

    try:
        return checker.check(model_id)
    except ProvenanceCheckError as exc:
        logger.warning("Provenance check failed for %s: %s", model_id, exc)
        return {
            "verified": False,
            "risk_factors": [f"Provenance check error: {exc}"],
            "author": model_id.split("/")[0] if "/" in model_id else "unknown",
            "downloads": -1,
            "has_model_card": False,
            "architecture_match": False,
            "last_modified": "",
            "author_reputation": {},
            "raw_metadata": {},
        }


def scan_model(
    model_config: dict[str, str],
    checker: ProvenanceChecker,
    correlator: CVECorrelator,
    offline: bool,
) -> dict[str, Any]:
    """Run provenance + CVE correlation for a single model.

    Weight analysis and activation scan are intentionally skipped.

    Args:
        model_config: Dict with model_id, architecture, description keys.
        checker: ProvenanceChecker instance.
        correlator: CVECorrelator instance.
        offline: Skip HTTP calls if True.

    Returns:
        Result dict with provenance, cve_correlations, and timing info.
    """
    model_id = model_config["model_id"]
    arch = model_config["architecture"]
    start = time.perf_counter()

    console.log(f"  Scanning [cyan]{model_id}[/cyan]...")

    # Provenance check
    prov = run_provenance_check(model_id, checker, offline)

    # CVE correlation
    cve_results = correlator.correlate(
        model_id=model_id,
        architecture=arch,
        framework_versions={"transformers": "unknown"},
    )
    cve_dicts = [c.to_dict() for c in cve_results]

    duration = time.perf_counter() - start

    return {
        "model_id": model_id,
        "architecture": arch,
        "description": model_config["description"],
        "scan_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "scan_duration_s": round(duration, 3),
        "offline_mode": offline,
        "provenance": {
            "verified": prov["verified"],
            "author": prov["author"],
            "downloads": prov["downloads"],
            "has_model_card": prov["has_model_card"],
            "architecture_match": prov["architecture_match"],
            "risk_factors": prov["risk_factors"],
            "last_modified": prov.get("last_modified", ""),
        },
        "cve_correlations": cve_dicts,
        "n_cves_matched": len(cve_dicts),
        "provenance_verified": prov["verified"],
        "weight_analysis": "SKIPPED",
        "activation_scan": "SKIPPED",
    }


# ---------------------------------------------------------------------------
# Rich table rendering
# ---------------------------------------------------------------------------


def render_results_table(results: list[dict[str, Any]]) -> None:
    """Print benchmark results as a Rich table.

    Args:
        results: List of scan result dicts from scan_model().
    """
    table = Table(
        title="Known Model Benchmark Results",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Model ID", style="cyan", min_width=28)
    table.add_column("Author", min_width=16)
    table.add_column("Downloads", justify="right", min_width=12)
    table.add_column("Model Card", justify="center", min_width=10)
    table.add_column("Provenance", justify="center", min_width=12)
    table.add_column("CVEs Matched", justify="right", min_width=12)
    table.add_column("Duration (s)", justify="right", min_width=12)

    for r in results:
        prov_ok = r["provenance_verified"]
        prov_cell = "[green]PASS[/green]" if prov_ok else "[red]FAIL[/red]"
        mc_cell = "[green]Yes[/green]" if r["provenance"]["has_model_card"] else "[red]No[/red]"

        downloads = r["provenance"]["downloads"]
        dl_str = f"{downloads:,}" if downloads >= 0 else "N/A"

        n_cves = r["n_cves_matched"]
        cve_cell = f"[yellow]{n_cves}[/yellow]" if n_cves > 5 else str(n_cves)

        table.add_row(
            r["model_id"],
            r["provenance"]["author"],
            dl_str,
            mc_cell,
            prov_cell,
            cve_cell,
            str(r["scan_duration_s"]),
        )

    console.print(table)


def render_cve_table(results: list[dict[str, Any]]) -> None:
    """Print top CVE correlations across all scanned models.

    Args:
        results: List of scan result dicts from scan_model().
    """
    # Collect unique CVEs seen across all models
    seen_cves: dict[str, dict[str, Any]] = {}
    for r in results:
        for cve in r["cve_correlations"]:
            cve_id = cve["cve_id"]
            if cve_id not in seen_cves:
                seen_cves[cve_id] = cve

    if not seen_cves:
        console.print("[dim]No CVEs correlated.[/dim]")
        return

    cve_table = Table(
        title="CVE Correlations (all scanned models)",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    cve_table.add_column("CVE ID", style="bold", min_width=20)
    cve_table.add_column("Severity", justify="center", min_width=10)
    cve_table.add_column("CVSS", justify="right", min_width=6)
    cve_table.add_column("Affected Framework", min_width=18)
    cve_table.add_column("Description (truncated)", min_width=50)

    sev_colors = {
        "CRITICAL": "bold red",
        "HIGH": "red",
        "MEDIUM": "yellow",
        "LOW": "blue",
    }

    for cve in sorted(seen_cves.values(), key=lambda c: -(c.get("cvss_score") or 0.0)):
        sev = cve.get("severity", "UNKNOWN")
        col = sev_colors.get(sev, "white")
        frameworks = ", ".join(
            f"{k}({v})" for k, v in (cve.get("affected_frameworks") or {}).items()
        ) or "any"
        desc = cve.get("description", "")[:80].rstrip()

        cve_table.add_row(
            cve.get("cve_id", "?"),
            f"[{col}]{sev}[/{col}]",
            str(cve.get("cvss_score") or "N/A"),
            frameworks,
            desc,
        )

    console.print(cve_table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the known-models benchmark.

    Returns:
        Exit code: 0 on success, 1 if any provenance check failed.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark modelsafe against well-known public HuggingFace models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Skip all HTTP calls and use synthetic provenance results.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace API token (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "experiments" / "results" / "known_models_scan.json"),
        metavar="PATH",
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    console.rule("[bold cyan]modelsafe — Known Model Benchmark[/bold cyan]")
    if args.offline:
        console.print(
            "[yellow]Running in OFFLINE mode — synthetic provenance results used.[/yellow]"
        )

    checker = ProvenanceChecker(hf_token=args.hf_token)
    correlator = CVECorrelator()

    # Run scans
    scan_results: list[dict[str, Any]] = []
    benchmark_start = time.perf_counter()

    for model_cfg in BENCHMARK_MODELS:
        result = scan_model(
            model_config=model_cfg,
            checker=checker,
            correlator=correlator,
            offline=args.offline,
        )
        scan_results.append(result)

    benchmark_duration = time.perf_counter() - benchmark_start

    # Render tables
    console.print()
    render_results_table(scan_results)
    console.print()
    render_cve_table(scan_results)

    # Summary
    n_passed = sum(1 for r in scan_results if r["provenance_verified"])
    n_total = len(scan_results)
    console.print(
        f"\n[bold]Summary:[/bold] {n_passed}/{n_total} models passed provenance check  |  "
        f"Total time: {benchmark_duration:.2f}s\n"
    )

    # Save JSON output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "scanner_version": "0.2.0",
        "benchmark_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "benchmark_duration_s": round(benchmark_duration, 3),
        "offline_mode": args.offline,
        "n_models_scanned": n_total,
        "n_provenance_passed": n_passed,
        "results": scan_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    console.print(f"[dim]Results saved to: {output_path}[/dim]")

    # Non-zero exit if any provenance check failed
    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
