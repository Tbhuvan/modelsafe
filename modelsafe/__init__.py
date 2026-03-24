"""
modelsafe — ML Model Supply Chain Scanner.

Pre-flight security scanner for HuggingFace models.
Detects backdoored models without access to training data using:
  - Spectral weight analysis (SVD-based anomaly detection)
  - Activation scanning on synthetic inputs (behaviour probing)
  - Provenance checking (HuggingFace metadata & author reputation)
  - Known-threat hash database (analogous to VirusTotal)

Quick start::

    from modelsafe.scanner import ModelScanner

    scanner = ModelScanner()
    result = scanner.scan("microsoft/phi-2")
    print(result.to_report())

CLI::

    modelsafe scan gpt2
    modelsafe scan microsoft/phi-2 --output json
    modelsafe list-threats

Research Question:
    Can you detect backdoored models without access to training data?

References:
    arXiv:2409.09368 — Models Are Codes: Malicious Code Poisoning on Model Hubs
    arXiv:2202.06196 — Spectral Signatures in Backdoor Attacks
    OWASP AI Security Top 10
"""

from .scanner import ModelScanner, ScanResult
from .threat_db import ThreatDatabase
from .weight_analysis import WeightAnalyzer
from .activation_scan import ActivationScanner
from .provenance import ProvenanceChecker

__all__ = [
    "ModelScanner",
    "ScanResult",
    "ThreatDatabase",
    "WeightAnalyzer",
    "ActivationScanner",
    "ProvenanceChecker",
]

__version__ = "0.1.0"
__author__ = "modelsafe Contributors"
