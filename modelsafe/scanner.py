"""
Main orchestrator for modelsafe model security scans.

The scanner coordinates three independent security checks:

    1. Provenance check  — metadata, author reputation, model card, downloads.
    2. Weight analysis   — spectral signatures, K-S test, Trojan norm heuristic.
    3. Activation scan   — forward-pass probing with synthetic inputs.

Checks run in the order above.  The scanner short-circuits after the provenance
check if the model matches a known threat hash (no need to download and analyse
weights that are already in the threat database).

Risk score computation:
    risk_score = max(threat_db_hit × 1.0,
                     weighted_sum(provenance, weights, activations))

    Weights: provenance=0.3, weight_analysis=0.4, activation_scan=0.3

A risk_score > 0.7 is reported as "unsafe".

References:
    arXiv:2409.09368 — Models Are Codes
    OWASP AI Security Top 10 — ML05:2023 Supply Chain Vulnerabilities
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .provenance import ProvenanceChecker, ProvenanceCheckError
from .threat_db import ThreatDatabase
from .weight_analysis import WeightAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk score weights
# ---------------------------------------------------------------------------

_WEIGHT_PROVENANCE = 0.30
_WEIGHT_WEIGHTS = 0.40
_WEIGHT_ACTIVATION = 0.30
_RISK_THRESHOLD = 0.70  # above this → unsafe


@dataclass
class ScanResult:
    """Complete security scan result for a single model.

    Attributes:
        model_id: HuggingFace model ID or local path used for the scan.
        safe: True if risk_score < threshold and no critical findings.
        risk_score: Aggregate risk in [0.0, 1.0].  1.0 = definitely malicious.
        findings: List of finding dicts: {check, result, severity, detail}.
        provenance_verified: True if provenance check passed with no issues.
        weight_anomaly_score: Raw score from WeightAnalyzer (0.0–1.0).
        activation_anomaly_score: Raw score from ActivationScanner (0.0–1.0).
        scan_duration_s: Wall-clock seconds taken.
        timestamp: ISO 8601 scan completion timestamp.
        threat_db_hit: True if model hash matched a known malicious model.
        threat_record: The matched threat record dict, or None.
    """

    model_id: str
    safe: bool
    risk_score: float
    findings: list[dict[str, Any]]
    provenance_verified: bool
    weight_anomaly_score: float
    activation_anomaly_score: float
    scan_duration_s: float
    timestamp: str
    threat_db_hit: bool = False
    threat_record: dict[str, Any] | None = None

    def to_report(self) -> str:
        """Generate a human-readable scan report.

        Returns:
            Multi-line string report suitable for terminal output.
        """
        lines = [
            f"modelsafe Scan Report",
            f"{'=' * 50}",
            f"Model:     {self.model_id}",
            f"Timestamp: {self.timestamp}",
            f"Duration:  {self.scan_duration_s:.2f}s",
            f"",
            f"RESULT:  {'SAFE' if self.safe else 'UNSAFE'}",
            f"Risk score: {self.risk_score:.3f} (threshold: {_RISK_THRESHOLD})",
            f"",
        ]

        if self.threat_db_hit and self.threat_record:
            lines += [
                f"[!] KNOWN THREAT MATCH",
                f"    CVE: {self.threat_record.get('cve_id', 'N/A')}",
                f"    Type: {self.threat_record.get('threat_type', 'unknown')}",
                f"    Effect: {self.threat_record.get('effect', 'unknown')}",
                "",
            ]

        lines += [
            f"Provenance verified: {self.provenance_verified}",
            f"Weight anomaly score: {self.weight_anomaly_score:.3f}",
            f"Activation anomaly score: {self.activation_anomaly_score:.3f}",
            "",
        ]

        if self.findings:
            lines.append("Findings:")
            for f in self.findings:
                sev = f.get("severity", "info").upper()
                check = f.get("check", "unknown")
                detail = f.get("detail", "")
                lines.append(f"  [{sev}] {check}: {detail}")
        else:
            lines.append("No findings.")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialise scan result to JSON string.

        Returns:
            JSON-encoded string of the scan result.
        """
        return json.dumps(
            {
                "model_id": self.model_id,
                "safe": self.safe,
                "risk_score": self.risk_score,
                "findings": self.findings,
                "provenance_verified": self.provenance_verified,
                "weight_anomaly_score": self.weight_anomaly_score,
                "activation_anomaly_score": self.activation_anomaly_score,
                "scan_duration_s": self.scan_duration_s,
                "timestamp": self.timestamp,
                "threat_db_hit": self.threat_db_hit,
                "threat_record": self.threat_record,
            },
            indent=2,
        )


class ModelScanner:
    """Orchestrates all security checks for a HuggingFace or local model.

    Check order:
        1. Threat DB lookup (by model ID + weight hash if available).
        2. Provenance check (HF metadata, author reputation, model card).
        3. Weight analysis (spectral analysis, statistical tests).
        4. Activation scan (forward-pass probing — only if weights are loaded).

    The scanner short-circuits after a threat DB hit to avoid unnecessary
    work when the model is already confirmed malicious.

    Args:
        threat_db: Optional ThreatDatabase instance.  Loaded from default
            path if None.
        weight_analyzer: Optional WeightAnalyzer instance.
        provenance_checker: Optional ProvenanceChecker instance.
        skip_activation_scan: Set True to skip the activation scan entirely
            (faster but misses behavioural anomalies).
        hf_token: HuggingFace token for private model access.

    Example:
        >>> scanner = ModelScanner()
        >>> result = scanner.scan("microsoft/phi-2")
        >>> print(result.to_report())
    """

    def __init__(
        self,
        threat_db: ThreatDatabase | None = None,
        weight_analyzer: WeightAnalyzer | None = None,
        provenance_checker: ProvenanceChecker | None = None,
        skip_activation_scan: bool = False,
        hf_token: str | None = None,
    ) -> None:
        self.threat_db = threat_db or ThreatDatabase()
        self.weight_analyzer = weight_analyzer or WeightAnalyzer()
        self.provenance_checker = provenance_checker or ProvenanceChecker(hf_token=hf_token)
        self.skip_activation_scan = skip_activation_scan

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, model_id: str, local_path: str | None = None) -> ScanResult:
        """Run a full security scan of a model.

        Args:
            model_id: HuggingFace model ID (e.g., "gpt2") or a label for a
                local model when local_path is also provided.
            local_path: Optional path to a local model directory.  If provided,
                weights are loaded from disk instead of downloaded from HF Hub.

        Returns:
            ScanResult with all findings and the composite risk score.

        Raises:
            ValueError: If model_id is empty.
        """
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")

        start_time = time.perf_counter()
        findings: list[dict[str, Any]] = []
        weight_anomaly_score = 0.0
        activation_anomaly_score = 0.0
        threat_db_hit = False
        threat_record: dict[str, Any] | None = None

        logger.info("Starting scan for model: %s", model_id)

        # --- Step 1: Threat DB lookup by model ID ---
        id_threats = self.threat_db.check_model_id(model_id)
        if id_threats:
            threat_record = id_threats[0]
            threat_db_hit = True
            findings.append({
                "check": "threat_db",
                "result": "FAIL",
                "severity": "critical",
                "detail": (
                    f"Model ID matches known malicious model. "
                    f"CVE: {threat_record.get('cve_id', 'N/A')} — "
                    f"{threat_record.get('effect', 'unknown effect')}"
                ),
            })
            logger.warning("Threat DB hit for model_id '%s'", model_id)

        # --- Step 2: Threat DB lookup by hash (if local file provided) ---
        if local_path and not threat_db_hit:
            hash_result = self._check_threat_db_by_hash(model_id, local_path)
            if hash_result is not None:
                threat_record = hash_result
                threat_db_hit = True
                findings.append({
                    "check": "threat_db_hash",
                    "result": "FAIL",
                    "severity": "critical",
                    "detail": (
                        f"Model hash matches known malicious model. "
                        f"CVE: {hash_result.get('cve_id', 'N/A')}"
                    ),
                })

        # Short-circuit: known malicious → risk = 1.0
        if threat_db_hit:
            return self._build_result(
                model_id=model_id,
                risk_score=1.0,
                findings=findings,
                provenance_verified=False,
                weight_anomaly_score=1.0,
                activation_anomaly_score=1.0,
                start_time=start_time,
                threat_db_hit=True,
                threat_record=threat_record,
            )

        # --- Step 3: Provenance check ---
        prov_score, prov_verified = self._run_provenance_check(model_id, findings)

        # --- Step 4: Weight analysis ---
        if local_path:
            weight_anomaly_score = self._run_weight_analysis(
                model_id, local_path, findings
            )

        # --- Step 5: Activation scan ---
        if not self.skip_activation_scan and local_path:
            activation_anomaly_score = self._run_activation_scan(
                model_id, local_path, findings
            )

        # --- Aggregate risk score ---
        risk_score = self._compute_risk_score(
            prov_score, weight_anomaly_score, activation_anomaly_score
        )

        logger.info(
            "Scan complete: model=%s risk=%.3f safe=%s",
            model_id,
            risk_score,
            risk_score < _RISK_THRESHOLD,
        )

        return self._build_result(
            model_id=model_id,
            risk_score=risk_score,
            findings=findings,
            provenance_verified=prov_verified,
            weight_anomaly_score=weight_anomaly_score,
            activation_anomaly_score=activation_anomaly_score,
            start_time=start_time,
        )

    def _check_threat_db(self, model_id: str, model_hash: str) -> dict[str, Any] | None:
        """Check if a model hash matches a known malicious model.

        Args:
            model_id: Model identifier (for logging).
            model_hash: Pre-computed SHA-256 hex digest.

        Returns:
            Threat record dict if found, else None.
        """
        result = self.threat_db.check_hash(model_hash)
        if result:
            logger.warning(
                "Model '%s' hash matches known threat: %s",
                model_id,
                result.get("cve_id", "?"),
            )
        return result

    # ------------------------------------------------------------------
    # Internal scan runners
    # ------------------------------------------------------------------

    def _check_threat_db_by_hash(
        self, model_id: str, local_path: str
    ) -> dict[str, Any] | None:
        """Hash local model files and check against threat DB.

        Args:
            model_id: Model identifier.
            local_path: Local path to model directory or file.

        Returns:
            Threat record if found, else None.
        """
        from pathlib import Path as _Path

        path = _Path(local_path)
        weight_files = []

        # Look for common weight file extensions
        for ext in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
            weight_files.extend(path.glob(ext) if path.is_dir() else ([path] if path.suffix == ext[1:] else []))

        for weight_file in weight_files[:3]:  # Check first 3 shards max
            try:
                h = self.threat_db.compute_model_hash(weight_file)
                result = self._check_threat_db(model_id, h)
                if result:
                    return result
            except (FileNotFoundError, OSError) as exc:
                logger.debug("Cannot hash %s: %s", weight_file, exc)
        return None

    def _run_provenance_check(
        self, model_id: str, findings: list[dict[str, Any]]
    ) -> tuple[float, bool]:
        """Run provenance check and populate findings list.

        Args:
            model_id: Model identifier.
            findings: List to append findings to (mutated in place).

        Returns:
            Tuple of (provenance_risk_score, provenance_verified).
        """
        try:
            prov = self.provenance_checker.check(model_id)
        except ProvenanceCheckError as exc:
            findings.append({
                "check": "provenance",
                "result": "ERROR",
                "severity": "medium",
                "detail": f"Provenance check failed: {exc}",
            })
            return 0.3, False  # Unknown = moderate risk

        prov_verified = prov["verified"]
        risk_factors = prov.get("risk_factors", [])

        if not prov_verified:
            for rf in risk_factors:
                findings.append({
                    "check": "provenance",
                    "result": "FAIL",
                    "severity": "medium",
                    "detail": rf,
                })

        # Normalise: each risk factor adds ~0.25 to risk
        prov_score = min(1.0, len(risk_factors) * 0.25)
        return prov_score, prov_verified

    def _run_weight_analysis(
        self, model_id: str, local_path: str, findings: list[dict[str, Any]]
    ) -> float:
        """Load weights and run WeightAnalyzer.

        Args:
            model_id: Model identifier (for logging).
            local_path: Local model path.
            findings: List to append findings to.

        Returns:
            Weight anomaly score in [0, 1].
        """
        weights = self._load_weights(local_path)
        if not weights:
            findings.append({
                "check": "weight_analysis",
                "result": "SKIP",
                "severity": "info",
                "detail": "No weight files found; weight analysis skipped.",
            })
            return 0.0

        try:
            wa_result = self.weight_analyzer.analyze(weights)
        except Exception as exc:
            logger.error("Weight analysis failed: %s", exc)
            findings.append({
                "check": "weight_analysis",
                "result": "ERROR",
                "severity": "medium",
                "detail": f"Weight analysis error: {exc}",
            })
            return 0.2

        score = wa_result["anomaly_score"]
        if score >= 0.5:
            sev = "high" if score >= 0.7 else "medium"
            findings.append({
                "check": "weight_analysis",
                "result": "FAIL",
                "severity": sev,
                "detail": (
                    f"Weight anomaly score={score:.3f}. "
                    f"Flagged {wa_result['n_flagged_layers']} / "
                    f"{wa_result['n_layers_analysed']} layers."
                ),
            })
        else:
            findings.append({
                "check": "weight_analysis",
                "result": "PASS",
                "severity": "info",
                "detail": f"Weight analysis score={score:.3f} (below 0.5 threshold).",
            })

        return score

    def _run_activation_scan(
        self, model_id: str, local_path: str, findings: list[dict[str, Any]]
    ) -> float:
        """Load model and run ActivationScanner.

        Args:
            model_id: Model identifier.
            local_path: Local model path.
            findings: List to append findings to.

        Returns:
            Activation anomaly score in [0, 1].
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
            from .activation_scan import ActivationScanner

            tokenizer = AutoTokenizer.from_pretrained(local_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                local_path, device_map="cpu", torch_dtype="auto"
            )
            scanner = ActivationScanner(n_synthetic=50)
            act_result = scanner.scan(model, tokenizer)

            score = act_result["anomaly_score"]
            if score >= 0.5:
                findings.append({
                    "check": "activation_scan",
                    "result": "FAIL",
                    "severity": "high" if score >= 0.7 else "medium",
                    "detail": (
                        f"Activation anomaly score={score:.3f}. "
                        f"Suspicious layers: {act_result['suspicious_layers']}"
                    ),
                })
            else:
                findings.append({
                    "check": "activation_scan",
                    "result": "PASS",
                    "severity": "info",
                    "detail": f"Activation scan score={score:.3f}.",
                })
            return score

        except ImportError:
            findings.append({
                "check": "activation_scan",
                "result": "SKIP",
                "severity": "info",
                "detail": "torch/transformers not installed; activation scan skipped.",
            })
            return 0.0
        except Exception as exc:
            logger.error("Activation scan failed: %s", exc)
            findings.append({
                "check": "activation_scan",
                "result": "ERROR",
                "severity": "medium",
                "detail": f"Activation scan error: {exc}",
            })
            return 0.1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_weights(local_path: str) -> dict[str, Any]:
        """Load model weights from a local path.

        Supports safetensors and PyTorch .bin formats.

        Args:
            local_path: Path to model directory or single weight file.

        Returns:
            Dict mapping layer_name → numpy array.  Empty dict if load fails.
        """
        from pathlib import Path as _Path
        import numpy as np

        path = _Path(local_path)
        weights: dict[str, Any] = {}

        # Try safetensors first (safer, no pickle)
        safetensor_files = list(path.glob("*.safetensors")) if path.is_dir() else (
            [path] if path.suffix == ".safetensors" else []
        )
        if safetensor_files:
            try:
                from safetensors import safe_open  # type: ignore[import]
                for sf in safetensor_files[:3]:
                    with safe_open(str(sf), framework="np") as f:
                        for key in f.keys():
                            weights[key] = f.get_tensor(key)
                logger.info("Loaded %d tensors from safetensors", len(weights))
                return weights
            except ImportError:
                logger.debug("safetensors not installed; trying torch load")
            except Exception as exc:
                logger.warning("safetensors load failed: %s", exc)

        # Try PyTorch .bin (requires torch)
        bin_files = list(path.glob("*.bin")) if path.is_dir() else (
            [path] if path.suffix == ".bin" else []
        )
        if bin_files:
            try:
                import torch  # type: ignore[import]
                for bf in bin_files[:3]:
                    sd = torch.load(str(bf), map_location="cpu", weights_only=True)
                    for k, v in sd.items():
                        weights[k] = v.numpy()
                logger.info("Loaded %d tensors from .bin", len(weights))
                return weights
            except ImportError:
                logger.debug("torch not installed; weight analysis unavailable")
            except Exception as exc:
                logger.warning("torch load failed: %s", exc)

        return weights

    @staticmethod
    def _compute_risk_score(
        prov_score: float,
        weight_score: float,
        activation_score: float,
    ) -> float:
        """Compute the composite risk score.

        Args:
            prov_score: Provenance risk score [0, 1].
            weight_score: Weight anomaly score [0, 1].
            activation_score: Activation anomaly score [0, 1].

        Returns:
            Weighted average risk score in [0, 1].
        """
        score = (
            _WEIGHT_PROVENANCE * prov_score
            + _WEIGHT_WEIGHTS * weight_score
            + _WEIGHT_ACTIVATION * activation_score
        )
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _build_result(
        model_id: str,
        risk_score: float,
        findings: list[dict[str, Any]],
        provenance_verified: bool,
        weight_anomaly_score: float,
        activation_anomaly_score: float,
        start_time: float,
        threat_db_hit: bool = False,
        threat_record: dict[str, Any] | None = None,
    ) -> ScanResult:
        """Construct a ScanResult from component scores.

        Args:
            model_id: Model identifier.
            risk_score: Composite risk score.
            findings: List of finding dicts.
            provenance_verified: Whether provenance check passed.
            weight_anomaly_score: Weight analysis score.
            activation_anomaly_score: Activation scan score.
            start_time: perf_counter value at scan start.
            threat_db_hit: True if known threat matched.
            threat_record: Matching threat record or None.

        Returns:
            Populated ScanResult instance.
        """
        duration = time.perf_counter() - start_time
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        return ScanResult(
            model_id=model_id,
            safe=risk_score < _RISK_THRESHOLD and not threat_db_hit,
            risk_score=round(risk_score, 4),
            findings=findings,
            provenance_verified=provenance_verified,
            weight_anomaly_score=round(weight_anomaly_score, 4),
            activation_anomaly_score=round(activation_anomaly_score, 4),
            scan_duration_s=round(duration, 3),
            timestamp=timestamp,
            threat_db_hit=threat_db_hit,
            threat_record=threat_record,
        )
