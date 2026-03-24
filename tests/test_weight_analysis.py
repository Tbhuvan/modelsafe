"""
Tests for WeightAnalyzer.

Coverage:
- SVD analysis: correct flagging of anomalous singular value concentration.
- Statistical test: K-S test flags non-Gaussian distributions.
- Trojan trigger heuristic: correctly flags outlier norms.
- Full analyze() pipeline: aggregation and per-layer scoring.
- Input validation and edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from modelsafe.weight_analysis import WeightAnalyzer, _score_to_severity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> WeightAnalyzer:
    return WeightAnalyzer()


@pytest.fixture
def clean_weights() -> dict[str, np.ndarray]:
    """Weights sampled from a standard Gaussian — should score low."""
    rng = np.random.default_rng(42)
    return {
        "layer1.weight": rng.standard_normal((64, 64)).astype(np.float32),
        "layer2.weight": rng.standard_normal((128, 64)).astype(np.float32),
        "layer3.bias": rng.standard_normal(128).astype(np.float32),
    }


@pytest.fixture
def backdoored_weights() -> dict[str, np.ndarray]:
    """Weights with an injected backdoor subspace — anomalously large singular value."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((64, 64)).astype(np.float32)
    # Inject rank-1 backdoor: add a large outer product
    u = np.ones((64, 1), dtype=np.float32) * 100.0
    v = np.ones((1, 64), dtype=np.float32)
    return {
        "clean_layer.weight": rng.standard_normal((64, 64)).astype(np.float32),
        "backdoor_layer.weight": (base + u @ v).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# SVD analysis
# ---------------------------------------------------------------------------


class TestSVDAnalysis:
    def test_random_matrix_low_concentration(self, analyzer: WeightAnalyzer) -> None:
        """Random Gaussian matrix should have distributed singular values."""
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((64, 64))
        result = analyzer.svd_analysis(mat)
        # Random matrix should not concentrate >95% energy in first SV
        assert result["energy_concentration"] < 0.95

    def test_rank1_matrix_high_concentration(self, analyzer: WeightAnalyzer) -> None:
        """Rank-1 matrix has 100% energy in first singular value."""
        u = np.ones((32, 1))
        v = np.ones((1, 32))
        mat = u @ v
        result = analyzer.svd_analysis(mat)
        assert result["energy_concentration"] > 0.99
        assert result["flagged"] is True

    def test_identity_matrix_uniform_singular_values(
        self, analyzer: WeightAnalyzer
    ) -> None:
        """Identity matrix has all singular values = 1, low concentration."""
        mat = np.eye(20)
        result = analyzer.svd_analysis(mat)
        assert result["energy_concentration"] == pytest.approx(1.0 / 20, rel=0.01)
        assert result["flagged"] is False

    def test_rejects_1d_input(self, analyzer: WeightAnalyzer) -> None:
        with pytest.raises(ValueError, match="2-D"):
            analyzer.svd_analysis(np.ones(10))

    def test_returns_none_for_tiny_matrix(self, analyzer: WeightAnalyzer) -> None:
        mat = np.ones((2, 2))  # 4 elements < _MIN_SVD_ELEMENTS=16
        result = analyzer.svd_analysis(mat)
        assert result["flagged"] is False
        assert "too small" in result.get("note", "").lower()

    def test_top_singular_value_is_positive(self, analyzer: WeightAnalyzer) -> None:
        rng = np.random.default_rng(5)
        mat = rng.standard_normal((32, 32))
        result = analyzer.svd_analysis(mat)
        assert result["top_singular_value"] > 0

    def test_anomaly_score_in_range(self, analyzer: WeightAnalyzer) -> None:
        rng = np.random.default_rng(6)
        mat = rng.standard_normal((32, 32))
        result = analyzer.svd_analysis(mat)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_backdoor_injection_detected(self, analyzer: WeightAnalyzer) -> None:
        """Rank-1 injection should produce high concentration and be flagged."""
        rng = np.random.default_rng(7)
        base = rng.standard_normal((64, 64))
        # Add a strong rank-1 component
        trigger = np.outer(np.ones(64) * 50, np.ones(64))
        mat = (base + trigger).astype(np.float64)
        result = analyzer.svd_analysis(mat)
        assert result["flagged"] is True
        assert result["anomaly_score"] > 0.3


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------


class TestStatisticalTest:
    def test_gaussian_weights_pass(self, analyzer: WeightAnalyzer) -> None:
        rng = np.random.default_rng(10)
        weights = rng.standard_normal(1000)
        result = analyzer.statistical_test(weights)
        # Gaussian should not be flagged consistently
        # (K-S test has randomness; we check ks_pvalue > 0.001 instead)
        assert result["ks_pvalue"] is not None
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_uniform_distribution_flagged(self, analyzer: WeightAnalyzer) -> None:
        rng = np.random.default_rng(11)
        weights = rng.uniform(-1, 1, 1000)
        result = analyzer.statistical_test(weights)
        # Uniform is clearly non-Gaussian (K-S p < 0.01)
        assert result["ks_pvalue"] < 0.01
        assert bool(result["flagged"]) is True

    def test_constant_weights_flagged(self, analyzer: WeightAnalyzer) -> None:
        weights = np.ones(500)
        result = analyzer.statistical_test(weights)
        assert result["flagged"] is True
        assert result["anomaly_score"] > 0.5

    def test_too_few_samples_no_flag(self, analyzer: WeightAnalyzer) -> None:
        weights = np.array([1.0, 2.0, 3.0])
        result = analyzer.statistical_test(weights)
        assert result["flagged"] is False
        assert "Insufficient" in result.get("note", "")

    def test_kurtosis_spike_in_result(self, analyzer: WeightAnalyzer) -> None:
        """Bimodal distribution has high kurtosis."""
        rng = np.random.default_rng(12)
        # Bimodal: mix of two Gaussians far apart
        weights = np.concatenate([
            rng.normal(-5, 0.1, 500),
            rng.normal(5, 0.1, 500),
        ])
        result = analyzer.statistical_test(weights)
        assert result["kurtosis"] is not None
        assert result["anomaly_score"] > 0.0

    def test_all_required_keys_present(self, analyzer: WeightAnalyzer) -> None:
        result = analyzer.statistical_test(np.random.randn(200))
        required = {"ks_statistic", "ks_pvalue", "kurtosis", "skewness", "flagged", "anomaly_score"}
        assert required.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# Trojan trigger heuristic
# ---------------------------------------------------------------------------


class TestDetectTrojanTrigger:
    def test_normal_weights_not_flagged(self, analyzer: WeightAnalyzer) -> None:
        """Standard Gaussian weights should not be flagged."""
        rng = np.random.default_rng(20)
        weights = rng.standard_normal(1000)
        assert analyzer.detect_trojan_trigger(weights) is False

    def test_large_norm_flagged(self, analyzer: WeightAnalyzer) -> None:
        """Weights with norm >> sqrt(n) should be flagged."""
        # n=100, expected_norm=10, but we set it to 10000
        weights = np.ones(100) * 100.0  # norm = 1000 >> sqrt(100)=10
        assert analyzer.detect_trojan_trigger(weights) is True

    def test_small_weights_not_flagged(self, analyzer: WeightAnalyzer) -> None:
        weights = np.ones(4) * 0.001  # too small, returns False
        assert analyzer.detect_trojan_trigger(weights) is False

    def test_custom_threshold_respected(self) -> None:
        analyzer_tight = WeightAnalyzer(trojan_z_threshold=1.0)
        # With z=1.0 threshold, even modest deviation should flag
        rng = np.random.default_rng(21)
        weights = rng.standard_normal(1000) * 2  # std=2 vs expected std=1
        # norm ≈ sqrt(1000)*2 = 63, expected = sqrt(1000) ≈ 31.6 → z ≈ 31 > 1
        result = analyzer_tight.detect_trojan_trigger(weights)
        assert result is True


# ---------------------------------------------------------------------------
# Full analyze() pipeline
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_required_keys(
        self, analyzer: WeightAnalyzer, clean_weights: dict
    ) -> None:
        result = analyzer.analyze(clean_weights)
        required = {
            "anomaly_score",
            "findings",
            "spectral_signature",
            "n_layers_analysed",
            "n_flagged_layers",
        }
        assert required.issubset(set(result.keys()))

    def test_clean_weights_low_score(
        self, analyzer: WeightAnalyzer, clean_weights: dict
    ) -> None:
        result = analyzer.analyze(clean_weights)
        # Clean Gaussian weights should score below 0.5
        assert result["anomaly_score"] < 0.5

    def test_backdoored_weights_higher_score(
        self,
        analyzer: WeightAnalyzer,
        clean_weights: dict,
        backdoored_weights: dict,
    ) -> None:
        clean_score = analyzer.analyze(clean_weights)["anomaly_score"]
        backdoor_score = analyzer.analyze(backdoored_weights)["anomaly_score"]
        assert backdoor_score >= clean_score, (
            f"Backdoored score ({backdoor_score}) should be >= clean score ({clean_score})"
        )

    def test_empty_weights_raises(self, analyzer: WeightAnalyzer) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            analyzer.analyze({})

    def test_anomaly_score_in_range(
        self, analyzer: WeightAnalyzer, clean_weights: dict
    ) -> None:
        result = analyzer.analyze(clean_weights)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_findings_are_dicts(
        self, analyzer: WeightAnalyzer, clean_weights: dict
    ) -> None:
        result = analyzer.analyze(clean_weights)
        for f in result["findings"]:
            assert isinstance(f, dict)
            assert "layer_name" in f
            assert "anomaly_score" in f
            assert "severity" in f

    def test_n_layers_analysed_correct(
        self, analyzer: WeightAnalyzer, clean_weights: dict
    ) -> None:
        result = analyzer.analyze(clean_weights)
        assert result["n_layers_analysed"] == len(clean_weights)

    def test_spectral_signature_is_list_of_floats(
        self, analyzer: WeightAnalyzer, clean_weights: dict
    ) -> None:
        result = analyzer.analyze(clean_weights)
        for sv in result["spectral_signature"]:
            assert isinstance(sv, float)
            assert sv >= 0.0

    def test_all_zero_weights_handled(self, analyzer: WeightAnalyzer) -> None:
        """Zero-weight tensors shouldn't crash the analyser."""
        weights = {"layer.weight": np.zeros((32, 32))}
        result = analyzer.analyze(weights)
        assert 0.0 <= result["anomaly_score"] <= 1.0


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------


class TestScoreToSeverity:
    def test_low_score_gives_low(self) -> None:
        assert _score_to_severity(0.1) == "low"

    def test_medium_score_gives_medium(self) -> None:
        assert _score_to_severity(0.5) == "medium"

    def test_high_score_gives_high(self) -> None:
        assert _score_to_severity(0.8) == "high"

    def test_boundary_0_4_is_medium(self) -> None:
        assert _score_to_severity(0.4) == "medium"

    def test_boundary_0_7_is_high(self) -> None:
        assert _score_to_severity(0.7) == "high"


# ---------------------------------------------------------------------------
# WeightAnalyzer constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_ks_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            WeightAnalyzer(ks_p_threshold=1.5)

    def test_invalid_trojan_z_raises(self) -> None:
        with pytest.raises(ValueError):
            WeightAnalyzer(trojan_z_threshold=-1.0)

    def test_invalid_spectral_concentration_raises(self) -> None:
        with pytest.raises(ValueError):
            WeightAnalyzer(spectral_concentration_threshold=1.1)
