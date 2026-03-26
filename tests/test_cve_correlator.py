"""
Tests for CVECorrelator and CVECorrelation.

Coverage:
- correlate() returns CVECorrelation objects with required fields.
- Architecture filtering: CVEs with specific archs only match those models.
- CVEs with empty affected_architectures match all models.
- Invalid model_id raises ValueError.
- CVSS-based sort order.
- extra_cves parameter appends to the built-in database.
- to_dict() produces a serialisable dict with all required keys.
- list_all_cves() returns the full database.
- cve_count() reflects built-in plus extra CVEs.
- _normalise_arch() correctly infers architectures from model IDs.
"""

from __future__ import annotations

import pytest

from modelsafe.cve_correlator import CVECorrelation, CVECorrelator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def correlator() -> CVECorrelator:
    return CVECorrelator()


@pytest.fixture
def extra_cve() -> dict:
    return {
        "cve_id": "CVE-9999-0001",
        "description": "Test CVE for unit tests only.",
        "severity": "LOW",
        "cvss_score": 2.0,
        "affected_condition": "Always applies in test.",
        "affected_architectures": [],
        "affected_frameworks": {"test-framework": "<0.0.1"},
        "mitigations": ["Upgrade test-framework."],
        "references": ["https://example.com"],
    }


# ---------------------------------------------------------------------------
# CVECorrelation dataclass
# ---------------------------------------------------------------------------


class TestCVECorrelation:
    def test_to_dict_has_required_keys(self) -> None:
        cve = CVECorrelation(
            cve_id="CVE-2025-32434",
            description="Test description.",
            severity="CRITICAL",
            affected_condition="Always.",
            mitigations=["Upgrade."],
            cvss_score=9.3,
        )
        d = cve.to_dict()
        required = {
            "cve_id",
            "description",
            "severity",
            "cvss_score",
            "affected_condition",
            "affected_architectures",
            "affected_frameworks",
            "mitigations",
            "references",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_cve_id_matches(self) -> None:
        cve = CVECorrelation(
            cve_id="CVE-2025-32434",
            description="x",
            severity="HIGH",
            affected_condition="y",
        )
        assert cve.to_dict()["cve_id"] == "CVE-2025-32434"

    def test_default_lists_are_empty(self) -> None:
        cve = CVECorrelation(
            cve_id="CVE-0000-0000",
            description="",
            severity="LOW",
            affected_condition="",
        )
        assert cve.affected_architectures == []
        assert cve.affected_frameworks == {}
        assert cve.mitigations == []
        assert cve.references == []

    def test_cvss_score_can_be_none(self) -> None:
        cve = CVECorrelation(
            cve_id="CVE-0000-0001",
            description="",
            severity="LOW",
            affected_condition="",
            cvss_score=None,
        )
        assert cve.to_dict()["cvss_score"] is None


# ---------------------------------------------------------------------------
# CVECorrelator.correlate() — basic behaviour
# ---------------------------------------------------------------------------


class TestCorrelate:
    def test_empty_model_id_raises(self, correlator: CVECorrelator) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            correlator.correlate("")

    def test_returns_list(self, correlator: CVECorrelator) -> None:
        results = correlator.correlate("gpt2")
        assert isinstance(results, list)

    def test_results_are_cve_correlations(self, correlator: CVECorrelator) -> None:
        results = correlator.correlate("bert-base-uncased", architecture="bert")
        for r in results:
            assert isinstance(r, CVECorrelation)

    def test_all_architectures_cves_included(self, correlator: CVECorrelator) -> None:
        """CVEs with empty affected_architectures should always be included."""
        results = correlator.correlate("some-random-model-xyz", architecture="unknown_arch")
        # CVE-2025-32434 (torch deserialization) has no arch restriction
        cve_ids = {r.cve_id for r in results}
        assert "CVE-2025-32434" in cve_ids

    def test_llama_cve_matches_llama_arch(self, correlator: CVECorrelator) -> None:
        """CVE-2024-34359 targets llama/mistral/falcon architectures."""
        results = correlator.correlate(
            "meta-llama/Llama-2-7b-hf", architecture="llama"
        )
        cve_ids = {r.cve_id for r in results}
        assert "CVE-2024-34359" in cve_ids

    def test_llama_cve_does_not_match_bert(self, correlator: CVECorrelator) -> None:
        """CVE-2024-34359 should NOT match bert-base-uncased."""
        results = correlator.correlate("bert-base-uncased", architecture="bert")
        cve_ids = {r.cve_id for r in results}
        assert "CVE-2024-34359" not in cve_ids

    def test_sorted_by_cvss_descending(self, correlator: CVECorrelator) -> None:
        """Results should be sorted by CVSS score, highest first."""
        results = correlator.correlate("gpt2", architecture="gpt2")
        scores = [r.cvss_score or 0.0 for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Expected descending CVSS order, got: {scores}"
        )

    def test_architecture_case_insensitive(self, correlator: CVECorrelator) -> None:
        """Architecture matching should be case-insensitive."""
        results_lower = correlator.correlate("llama-test", architecture="llama")
        results_upper = correlator.correlate("llama-test", architecture="LLAMA")
        assert {r.cve_id for r in results_lower} == {r.cve_id for r in results_upper}

    def test_model_id_only_no_arch(self, correlator: CVECorrelator) -> None:
        """correlate() must work without an architecture argument."""
        results = correlator.correlate("gpt2")
        assert isinstance(results, list)
        assert len(results) > 0  # At least the general CVEs should match

    def test_framework_versions_accepted(self, correlator: CVECorrelator) -> None:
        """framework_versions should be accepted without error."""
        results = correlator.correlate(
            "bert-base-uncased",
            architecture="bert",
            framework_versions={"torch": "2.0.1", "transformers": "4.30.0"},
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# CVECorrelator — extra CVEs
# ---------------------------------------------------------------------------


class TestExtraCVEs:
    def test_extra_cve_appended(self, extra_cve: dict) -> None:
        correlator = CVECorrelator(extra_cves=[extra_cve])
        results = correlator.correlate("any-model")
        cve_ids = {r.cve_id for r in results}
        assert "CVE-9999-0001" in cve_ids

    def test_extra_cves_increases_count(self, extra_cve: dict) -> None:
        base_correlator = CVECorrelator()
        extended_correlator = CVECorrelator(extra_cves=[extra_cve])
        assert extended_correlator.cve_count() == base_correlator.cve_count() + 1

    def test_invalid_extra_cves_type_raises(self) -> None:
        with pytest.raises(ValueError, match="list"):
            CVECorrelator(extra_cves="not-a-list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CVECorrelator — list_all_cves() and cve_count()
# ---------------------------------------------------------------------------


class TestDatabaseInspection:
    def test_list_all_cves_returns_list(self, correlator: CVECorrelator) -> None:
        all_cves = correlator.list_all_cves()
        assert isinstance(all_cves, list)
        assert len(all_cves) > 0

    def test_list_all_cves_are_cve_correlations(self, correlator: CVECorrelator) -> None:
        for cve in correlator.list_all_cves():
            assert isinstance(cve, CVECorrelation)

    def test_cve_count_positive(self, correlator: CVECorrelator) -> None:
        assert correlator.cve_count() >= 8  # at least our 10 built-in CVEs

    def test_list_all_contains_known_cve(self, correlator: CVECorrelator) -> None:
        cve_ids = {c.cve_id for c in correlator.list_all_cves()}
        assert "CVE-2025-32434" in cve_ids  # PyTorch deserialization

    def test_list_all_sorted_by_cvss(self, correlator: CVECorrelator) -> None:
        all_cves = correlator.list_all_cves()
        scores = [c.cvss_score or 0.0 for c in all_cves]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# CVECorrelator — _normalise_arch()
# ---------------------------------------------------------------------------


class TestNormaliseArch:
    def test_llama_model_id(self) -> None:
        arch = CVECorrelator._normalise_arch("meta-llama/Llama-2-7b-hf")
        assert arch == "llama"

    def test_gpt2_model_id(self) -> None:
        arch = CVECorrelator._normalise_arch("openai-community/gpt2")
        assert arch == "gpt2"

    def test_bert_model_id(self) -> None:
        arch = CVECorrelator._normalise_arch("bert-base-uncased")
        assert arch == "bert"

    def test_distilbert_model_id(self) -> None:
        arch = CVECorrelator._normalise_arch("distilbert-base-uncased")
        assert arch == "distilbert"

    def test_unknown_returns_empty(self) -> None:
        arch = CVECorrelator._normalise_arch("some-random-xyz-12345")
        assert arch == ""

    def test_mistral_model_id(self) -> None:
        arch = CVECorrelator._normalise_arch("mistralai/Mistral-7B-v0.1")
        assert arch == "mistral"


# ---------------------------------------------------------------------------
# CVECorrelation fields
# ---------------------------------------------------------------------------


class TestCorrelationFields:
    def test_pytorch_cve_severity_is_critical(self, correlator: CVECorrelator) -> None:
        results = correlator.correlate("any-model")
        pytorch_cve = next((r for r in results if r.cve_id == "CVE-2025-32434"), None)
        assert pytorch_cve is not None
        assert pytorch_cve.severity == "CRITICAL"

    def test_pytorch_cve_has_mitigations(self, correlator: CVECorrelator) -> None:
        results = correlator.correlate("any-model")
        pytorch_cve = next((r for r in results if r.cve_id == "CVE-2025-32434"), None)
        assert pytorch_cve is not None
        assert len(pytorch_cve.mitigations) >= 1

    def test_pytorch_cve_has_references(self, correlator: CVECorrelator) -> None:
        results = correlator.correlate("any-model")
        pytorch_cve = next((r for r in results if r.cve_id == "CVE-2025-32434"), None)
        assert pytorch_cve is not None
        assert len(pytorch_cve.references) >= 1

    def test_cvss_score_in_valid_range(self, correlator: CVECorrelator) -> None:
        for cve in correlator.list_all_cves():
            if cve.cvss_score is not None:
                assert 0.0 <= cve.cvss_score <= 10.0, (
                    f"{cve.cve_id} has invalid CVSS score: {cve.cvss_score}"
                )

    def test_severity_values_are_valid(self, correlator: CVECorrelator) -> None:
        valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for cve in correlator.list_all_cves():
            assert cve.severity in valid_severities, (
                f"{cve.cve_id} has invalid severity: {cve.severity}"
            )
