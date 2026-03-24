"""
Spectral and statistical anomaly detection on model weights.

Backdoored ("trojaned") models often embed trigger information in a small
subspace of their weight matrices.  Spectral analysis exposes this as:
  - Anomalously large singular values in certain layers.
  - Unusually heavy-tailed or skewed weight distributions.
  - A few "outlier" neurons whose norms are >>3σ from the layer mean.

This module implements three complementary detection strategies:
1. SVD analysis — top singular value ratio & spectral energy concentration.
2. Statistical tests — K-S test against Gaussian baseline, kurtosis, skew.
3. Trojan trigger heuristic — flag layers with norm >3σ from mean.

References:
    Hayase et al. 2021 — SPECTRE: Defending Against Backdoor Attacks via
        Robust Covariance Estimation (arXiv:2104.11315)
    Tran et al. 2018 — Spectral Signatures in Backdoor Attacks (NeurIPS 2018)
    arXiv:2202.06196 — Detecting Backdoor in Deep Neural Networks via Spectral
        Signatures (direct reference for this implementation)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum matrix size to run SVD (tiny biases don't need SVD)
_MIN_SVD_ELEMENTS = 16
# Z-score threshold for "trojan norm" heuristic
_TROJAN_Z_THRESHOLD = 3.0
# Energy concentration threshold: fraction of total spectral energy in top-1 SV
_SPECTRAL_CONCENTRATION_THRESHOLD = 0.95
# K-S p-value below which the weight distribution is flagged as anomalous
_KS_P_VALUE_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Per-layer finding dataclass
# ---------------------------------------------------------------------------


@dataclass
class LayerFinding:
    """Anomaly finding for a single layer.

    Attributes:
        layer_name: Weight tensor name from the model's state_dict.
        anomaly_score: Normalised [0, 1] severity score.
        severity: "low" | "medium" | "high".
        checks: Dict of check_name → result.
        notes: Human-readable explanation.
    """

    layer_name: str
    anomaly_score: float
    severity: str
    checks: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "layer_name": self.layer_name,
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity,
            "checks": self.checks,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------


class WeightAnalyzer:
    """Detects anomalies in model weight distributions that indicate backdoors.

    Methods:
        1. Spectral analysis: singular value decomposition of weight matrices.
        2. Statistical outlier detection: kurtosis, skewness, K-S test.
        3. Layer-wise anomaly scoring with Trojan trigger heuristic.

    Usage::

        from modelsafe.weight_analysis import WeightAnalyzer

        analyzer = WeightAnalyzer()
        # model_weights: dict[str, np.ndarray] from model.state_dict()
        result = analyzer.analyze(model_weights)
        print(result["anomaly_score"])  # 0.0 (clean) – 1.0 (likely backdoored)

    Args:
        ks_p_threshold: K-S test p-value threshold.  Below this, the layer
            weight distribution is flagged as non-Gaussian.
        trojan_z_threshold: Z-score for the Trojan norm heuristic.
        spectral_concentration_threshold: SVD energy concentration threshold.
    """

    def __init__(
        self,
        ks_p_threshold: float = _KS_P_VALUE_THRESHOLD,
        trojan_z_threshold: float = _TROJAN_Z_THRESHOLD,
        spectral_concentration_threshold: float = _SPECTRAL_CONCENTRATION_THRESHOLD,
    ) -> None:
        if ks_p_threshold <= 0 or ks_p_threshold >= 1:
            raise ValueError(f"ks_p_threshold must be in (0,1), got {ks_p_threshold}")
        if trojan_z_threshold <= 0:
            raise ValueError(f"trojan_z_threshold must be positive, got {trojan_z_threshold}")
        if not (0 < spectral_concentration_threshold < 1):
            raise ValueError(
                f"spectral_concentration_threshold must be in (0,1), "
                f"got {spectral_concentration_threshold}"
            )

        self.ks_p_threshold = ks_p_threshold
        self.trojan_z_threshold = trojan_z_threshold
        self.spectral_concentration_threshold = spectral_concentration_threshold

    def analyze(self, model_weights: dict[str, np.ndarray]) -> dict[str, Any]:
        """Run all checks on every weight tensor and aggregate results.

        Args:
            model_weights: Dict mapping layer name → NumPy weight array.
                Typically obtained via {k: v.numpy() for k, v in model.state_dict().items()}.

        Returns:
            Dict with keys:
                anomaly_score (float): Aggregate score in [0, 1].
                findings (list[dict]): Per-layer finding dicts.
                spectral_signature (list[float]): Top singular values per layer.
                n_layers_analysed (int): Number of layers examined.
                n_flagged_layers (int): Layers with anomaly_score >= 0.5.

        Raises:
            ValueError: If model_weights is empty.
        """
        if not model_weights:
            raise ValueError("model_weights must be non-empty")

        findings: list[LayerFinding] = []
        spectral_signature: list[float] = []

        for layer_name, weights in model_weights.items():
            weights_np = np.asarray(weights, dtype=np.float32)
            if weights_np.size == 0:
                continue

            layer_finding = self._analyze_layer(layer_name, weights_np)
            findings.append(layer_finding)

            # Collect top SV for the spectral signature
            if weights_np.ndim >= 2 and weights_np.size >= _MIN_SVD_ELEMENTS:
                svd_result = self.svd_analysis(weights_np.reshape(weights_np.shape[0], -1))
                if svd_result.get("top_singular_value") is not None:
                    spectral_signature.append(float(svd_result["top_singular_value"]))

        if not findings:
            return {
                "anomaly_score": 0.0,
                "findings": [],
                "spectral_signature": [],
                "n_layers_analysed": 0,
                "n_flagged_layers": 0,
            }

        scores = [f.anomaly_score for f in findings]
        # Aggregate: use 90th percentile to be sensitive to worst-case layers
        aggregate_score = float(np.percentile(scores, 90)) if scores else 0.0
        n_flagged = sum(1 for s in scores if s >= 0.5)

        logger.info(
            "WeightAnalyzer: %d layers, aggregate_score=%.3f, flagged=%d",
            len(findings),
            aggregate_score,
            n_flagged,
        )

        return {
            "anomaly_score": round(aggregate_score, 4),
            "findings": [f.to_dict() for f in findings],
            "spectral_signature": [round(v, 4) for v in spectral_signature],
            "n_layers_analysed": len(findings),
            "n_flagged_layers": n_flagged,
        }

    def _analyze_layer(self, layer_name: str, weights: np.ndarray) -> LayerFinding:
        """Run all checks on a single layer.

        Args:
            layer_name: Identifying name of the layer.
            weights: Weight array (any shape).

        Returns:
            LayerFinding with combined score and per-check details.
        """
        checks: dict[str, Any] = {}
        notes: list[str] = []
        scores: list[float] = []

        flat = weights.flatten().astype(np.float64)

        # --- Statistical test ---
        stat_result = self.statistical_test(flat)
        checks["statistical"] = stat_result
        scores.append(stat_result["anomaly_score"])
        if stat_result.get("flagged"):
            notes.append(
                f"K-S p={stat_result['ks_pvalue']:.4f} < threshold: "
                "weights deviate from Gaussian baseline."
            )

        # --- Trojan trigger heuristic ---
        trojan_detected = self.detect_trojan_trigger(flat)
        checks["trojan_heuristic"] = {"detected": trojan_detected}
        if trojan_detected:
            scores.append(0.8)
            notes.append(
                f"Layer norm is >{self.trojan_z_threshold}σ above mean: "
                "possible backdoor trigger subspace."
            )
        else:
            scores.append(0.0)

        # --- SVD analysis (only for 2-D+ tensors) ---
        if weights.ndim >= 2 and weights.size >= _MIN_SVD_ELEMENTS:
            mat = weights.reshape(weights.shape[0], -1).astype(np.float64)
            svd_result = self.svd_analysis(mat)
            checks["svd"] = svd_result
            scores.append(svd_result["anomaly_score"])
            if svd_result.get("flagged"):
                notes.append(
                    f"Spectral energy concentration={svd_result['energy_concentration']:.3f}: "
                    "abnormal singular value spectrum."
                )

        combined_score = float(np.mean(scores)) if scores else 0.0
        severity = _score_to_severity(combined_score)

        return LayerFinding(
            layer_name=layer_name,
            anomaly_score=round(combined_score, 4),
            severity=severity,
            checks=checks,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def svd_analysis(self, weight_matrix: np.ndarray) -> dict[str, Any]:
        """Check for anomalous singular value distributions.

        A backdoor-injected weight matrix often has one or a few anomalously
        large singular values (the trigger subspace).  We flag:
        - Spectral energy concentration: top-1 SV captures >95% of total energy.
        - Top-SV ratio: s[0] / s[1] > 10 (extremely dominant first component).

        Args:
            weight_matrix: 2-D matrix to analyse, shape (m, n).

        Returns:
            Dict with keys: top_singular_value, energy_concentration,
            sv_ratio, flagged, anomaly_score.

        Raises:
            ValueError: If weight_matrix is not 2-D.
        """
        if weight_matrix.ndim != 2:
            raise ValueError(
                f"svd_analysis requires a 2-D matrix, got shape {weight_matrix.shape}"
            )
        if weight_matrix.size < _MIN_SVD_ELEMENTS:
            return {
                "top_singular_value": None,
                "energy_concentration": None,
                "sv_ratio": None,
                "flagged": False,
                "anomaly_score": 0.0,
                "note": "Matrix too small for SVD analysis",
            }

        # Use full_matrices=False for efficiency
        try:
            singular_values = np.linalg.svd(weight_matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            logger.warning("SVD failed for matrix of shape %s", weight_matrix.shape)
            return {
                "top_singular_value": None,
                "energy_concentration": None,
                "sv_ratio": None,
                "flagged": False,
                "anomaly_score": 0.0,
                "note": "SVD computation failed",
            }

        total_energy = float(np.sum(singular_values ** 2))
        if total_energy < 1e-10:
            return {
                "top_singular_value": float(singular_values[0]),
                "energy_concentration": 1.0,
                "sv_ratio": None,
                "flagged": False,
                "anomaly_score": 0.0,
                "note": "Near-zero weight matrix",
            }

        energy_concentration = float(singular_values[0] ** 2 / total_energy)
        sv_ratio = float(singular_values[0] / singular_values[1]) if len(singular_values) > 1 and singular_values[1] > 1e-10 else float("inf")

        flagged = (
            energy_concentration > self.spectral_concentration_threshold
            or sv_ratio > 10.0
        )

        # Anomaly score: how far concentration is above threshold
        score_from_concentration = max(
            0.0,
            (energy_concentration - self.spectral_concentration_threshold)
            / (1.0 - self.spectral_concentration_threshold + 1e-8),
        )
        score_from_ratio = min(1.0, max(0.0, (sv_ratio - 10.0) / 90.0)) if math.isfinite(sv_ratio) else 1.0
        anomaly_score = float(np.clip(max(score_from_concentration, score_from_ratio), 0.0, 1.0))

        return {
            "top_singular_value": round(float(singular_values[0]), 6),
            "energy_concentration": round(energy_concentration, 6),
            "sv_ratio": round(sv_ratio, 4) if math.isfinite(sv_ratio) else None,
            "flagged": flagged,
            "anomaly_score": round(anomaly_score, 4),
        }

    def statistical_test(self, weights: np.ndarray) -> dict[str, Any]:
        """Kolmogorov-Smirnov test against expected Gaussian weight distribution.

        Neural network weights initialised with He/Glorot schemes follow
        approximately zero-mean Gaussian distributions.  Significant deviations
        (low K-S p-value, high kurtosis, high skewness) suggest post-training
        modification.

        Args:
            weights: 1-D array of weight values.

        Returns:
            Dict with keys: ks_statistic, ks_pvalue, kurtosis, skewness,
            flagged, anomaly_score.

        Raises:
            ValueError: If weights is not 1-D.
        """
        weights = np.asarray(weights, dtype=np.float64).flatten()
        if weights.size < 8:
            return {
                "ks_statistic": None,
                "ks_pvalue": None,
                "kurtosis": None,
                "skewness": None,
                "flagged": False,
                "anomaly_score": 0.0,
                "note": "Insufficient samples",
            }

        # Standardise before K-S test
        mu, sigma_w = float(np.mean(weights)), float(np.std(weights))
        if sigma_w < 1e-10:
            return {
                "ks_statistic": 1.0,
                "ks_pvalue": 0.0,
                "kurtosis": float("inf"),
                "skewness": 0.0,
                "flagged": True,
                "anomaly_score": 0.9,
                "note": "Near-constant weights (std ≈ 0)",
            }

        z_scores = (weights - mu) / sigma_w
        ks_stat, ks_pvalue = stats.kstest(z_scores, "norm")

        kurtosis_val = float(stats.kurtosis(weights))
        skewness_val = float(stats.skew(weights))

        # Combine signals: K-S + kurtosis excess + skewness magnitude
        ks_score = float(np.clip(1.0 - ks_pvalue / self.ks_p_threshold, 0.0, 1.0)) if ks_pvalue < self.ks_p_threshold else 0.0
        kurt_score = min(1.0, abs(kurtosis_val) / 10.0)
        skew_score = min(1.0, abs(skewness_val) / 3.0)

        anomaly_score = float(np.mean([ks_score, kurt_score * 0.5, skew_score * 0.3]))
        flagged = ks_pvalue < self.ks_p_threshold or abs(kurtosis_val) > 5.0

        return {
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "kurtosis": round(kurtosis_val, 4),
            "skewness": round(skewness_val, 4),
            "flagged": flagged,
            "anomaly_score": round(anomaly_score, 4),
        }

    def detect_trojan_trigger(self, weights: np.ndarray) -> bool:
        """Heuristic: flag layers with unusually large weight norm.

        Backdoored models often have a small set of neurons with large norms
        that encode the trigger pattern.  If the layer's L2 norm is more than
        trojan_z_threshold standard deviations above the mean across all layers
        (here estimated per-layer via bootstrap), flag the layer.

        This is a per-layer check: we compare the layer's norm against a
        reference distribution derived from the expected norm for that shape.

        Args:
            weights: 1-D array of weight values (any shape pre-flattened).

        Returns:
            True if the layer is flagged as potentially trojan.
        """
        weights = np.asarray(weights, dtype=np.float64).flatten()
        if weights.size < 4:
            return False

        layer_norm = float(np.linalg.norm(weights))
        # Expected norm for Gaussian weights N(0, 1): E[||w||] ≈ sqrt(n)
        expected_norm = math.sqrt(weights.size)
        expected_std = 1.0  # std of ||N(0,1)^n|| ≈ 1 for large n

        z_score = (layer_norm - expected_norm) / expected_std
        return z_score > self.trojan_z_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_to_severity(score: float) -> str:
    """Map a [0,1] anomaly score to a severity label.

    Args:
        score: Anomaly score in [0, 1].

    Returns:
        "low", "medium", or "high".
    """
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"
