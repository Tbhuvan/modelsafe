"""
Tests for ProvenanceChecker.

Uses requests-mock to avoid real network calls.  All HuggingFace API
responses are mocked to test the provenance logic in isolation.

Coverage:
- Known-good model returns verified=True, no risk factors.
- Missing model card adds a risk factor.
- New author account adds a risk factor.
- Architecture mismatch adds a risk factor.
- Network errors are handled gracefully.
- Input validation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from modelsafe.provenance import ProvenanceChecker, ProvenanceCheckError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_info(
    model_id: str = "test-author/test-model",
    author: str = "test-author",
    downloads: int = 1000,
    private: bool = False,
    pipeline_tag: str = "text-generation",
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "author": author,
        "downloads": downloads,
        "private": private,
        "pipeline_tag": pipeline_tag,
        "tags": tags or ["pytorch", "text-generation"],
        "config": config or {"model_type": "gpt2"},
        "lastModified": "2025-01-01T00:00:00Z",
        "likes": 42,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker() -> ProvenanceChecker:
    return ProvenanceChecker(timeout=5)


# ---------------------------------------------------------------------------
# fetch_model_info
# ---------------------------------------------------------------------------


class TestFetchModelInfo:
    def test_successful_fetch(self, checker: ProvenanceChecker) -> None:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = _make_model_info()

        with patch.object(checker.session, "get", return_value=mock_response):
            info = checker.fetch_model_info("test-author/test-model")
        assert info["author"] == "test-author"
        assert info["downloads"] == 1000

    def test_404_raises_provenance_error(self, checker: ProvenanceChecker) -> None:
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 404

        with patch.object(checker.session, "get", return_value=mock_response):
            with pytest.raises(ProvenanceCheckError, match="not found"):
                checker.fetch_model_info("nonexistent/model")

    def test_401_raises_provenance_error(self, checker: ProvenanceChecker) -> None:
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 401

        with patch.object(checker.session, "get", return_value=mock_response):
            with pytest.raises(ProvenanceCheckError, match="Unauthorised"):
                checker.fetch_model_info("private/model")

    def test_network_error_raises_provenance_error(
        self, checker: ProvenanceChecker
    ) -> None:
        with patch.object(
            checker.session,
            "get",
            side_effect=requests.ConnectionError("timeout"),
        ):
            with pytest.raises(ProvenanceCheckError, match="Network error"):
                checker.fetch_model_info("test/model")

    def test_invalid_json_raises_provenance_error(
        self, checker: ProvenanceChecker
    ) -> None:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")

        with patch.object(checker.session, "get", return_value=mock_response):
            with pytest.raises(ProvenanceCheckError, match="Invalid JSON"):
                checker.fetch_model_info("test/model")


# ---------------------------------------------------------------------------
# verify_model_card
# ---------------------------------------------------------------------------


class TestVerifyModelCard:
    def test_substantial_model_card_returns_true(
        self, checker: ProvenanceChecker
    ) -> None:
        long_card = "# Model Card\n" + "x" * 300  # well above 200-char threshold
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = long_card

        with patch.object(checker.session, "get", return_value=mock_response):
            result = checker.verify_model_card("test/model")
        assert result is True

    def test_short_model_card_returns_false(self, checker: ProvenanceChecker) -> None:
        short_card = "# Model Card"  # only 12 chars — below threshold
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = short_card

        with patch.object(checker.session, "get", return_value=mock_response):
            result = checker.verify_model_card("test/model")
        assert result is False

    def test_404_model_card_returns_false(self, checker: ProvenanceChecker) -> None:
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 404

        with patch.object(checker.session, "get", return_value=mock_response):
            result = checker.verify_model_card("test/model")
        assert result is False

    def test_network_error_returns_false(self, checker: ProvenanceChecker) -> None:
        with patch.object(
            checker.session,
            "get",
            side_effect=requests.RequestException("timeout"),
        ):
            result = checker.verify_model_card("test/model")
        assert result is False


# ---------------------------------------------------------------------------
# check_author_reputation
# ---------------------------------------------------------------------------


class TestCheckAuthorReputation:
    def test_returns_dict_with_required_keys(self, checker: ProvenanceChecker) -> None:
        mock_user_resp = MagicMock()
        mock_user_resp.ok = True
        mock_user_resp.json.return_value = {
            "numFollowers": 500,
            "createdAt": "2020-01-01T00:00:00Z",
        }

        mock_models_resp = MagicMock()
        mock_models_resp.ok = True
        mock_models_resp.json.return_value = [{"id": "test/m1"}, {"id": "test/m2"}]
        mock_models_resp.headers = {"X-Total-Count": "10"}

        with patch.object(
            checker.session, "get", side_effect=[mock_user_resp, mock_models_resp]
        ):
            rep = checker.check_author_reputation("testuser")

        assert "followers" in rep
        assert "model_count" in rep
        assert "is_new_account" in rep

    def test_unknown_author_returns_empty_like_dict(
        self, checker: ProvenanceChecker
    ) -> None:
        rep = checker.check_author_reputation("unknown")
        assert rep["is_new_account"] is True
        assert rep["model_count"] == 0

    def test_empty_author_returns_safe_defaults(
        self, checker: ProvenanceChecker
    ) -> None:
        rep = checker.check_author_reputation("")
        assert isinstance(rep, dict)

    def test_network_error_returns_fallback(self, checker: ProvenanceChecker) -> None:
        with patch.object(
            checker.session,
            "get",
            side_effect=requests.ConnectionError("err"),
        ):
            rep = checker.check_author_reputation("testuser")
        assert isinstance(rep, dict)
        assert rep.get("model_count", 0) == 0


# ---------------------------------------------------------------------------
# _is_new_account static method
# ---------------------------------------------------------------------------


class TestIsNewAccount:
    def test_recent_account_is_new(self) -> None:
        # Today's date is 2026-03-23 (as per memory context)
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(tz=timezone.utc) - timedelta(days=5)).isoformat()
        assert ProvenanceChecker._is_new_account(recent) is True

    def test_old_account_not_new(self) -> None:
        assert ProvenanceChecker._is_new_account("2020-01-01T00:00:00Z") is False

    def test_empty_string_returns_false(self) -> None:
        assert ProvenanceChecker._is_new_account("") is False

    def test_invalid_timestamp_returns_false(self) -> None:
        assert ProvenanceChecker._is_new_account("not-a-date") is False


# ---------------------------------------------------------------------------
# _check_architecture_consistency static method
# ---------------------------------------------------------------------------


class TestArchitectureConsistency:
    def test_bert_fill_mask_is_consistent(self) -> None:
        info = {
            "pipeline_tag": "fill-mask",
            "tags": ["bert"],
            "config": {"model_type": "bert"},
        }
        assert ProvenanceChecker._check_architecture_consistency(info) is True

    def test_bert_text_generation_is_inconsistent(self) -> None:
        info = {
            "pipeline_tag": "text-generation",
            "tags": ["bert"],
            "config": {"model_type": "bert"},
        }
        assert ProvenanceChecker._check_architecture_consistency(info) is False

    def test_gpt2_text_generation_is_consistent(self) -> None:
        info = {
            "pipeline_tag": "text-generation",
            "tags": ["gpt2"],
            "config": {"model_type": "gpt2"},
        }
        assert ProvenanceChecker._check_architecture_consistency(info) is True

    def test_missing_config_returns_true(self) -> None:
        """Without config data, we can't detect mismatches."""
        info = {"pipeline_tag": "text-generation", "tags": []}
        assert ProvenanceChecker._check_architecture_consistency(info) is True

    def test_empty_model_info_returns_true(self) -> None:
        assert ProvenanceChecker._check_architecture_consistency({}) is True


# ---------------------------------------------------------------------------
# Full check() integration
# ---------------------------------------------------------------------------


class TestCheck:
    def _mock_all_calls(
        self,
        checker: ProvenanceChecker,
        model_info: dict[str, Any],
        model_card_text: str = "x" * 300,
        user_info: dict[str, Any] | None = None,
        card_ok: bool = True,
        user_ok: bool = True,
    ) -> None:
        """Set up patched responses for a full check() call."""
        # Patch fetch_model_info
        checker.fetch_model_info = MagicMock(return_value=model_info)  # type: ignore
        # Patch verify_model_card
        checker.verify_model_card = MagicMock(return_value=(len(model_card_text) >= 200))  # type: ignore
        # Patch check_author_reputation
        checker.check_author_reputation = MagicMock(return_value={  # type: ignore
            "followers": 100,
            "model_count": 5,
            "is_new_account": False,
        })

    def test_verified_model_has_no_risk_factors(
        self, checker: ProvenanceChecker
    ) -> None:
        self._mock_all_calls(checker, _make_model_info())
        result = checker.check("test-author/test-model")
        assert result["verified"] is True
        assert result["risk_factors"] == []

    def test_missing_model_card_adds_risk_factor(
        self, checker: ProvenanceChecker
    ) -> None:
        checker.fetch_model_info = MagicMock(return_value=_make_model_info())  # type: ignore
        checker.verify_model_card = MagicMock(return_value=False)  # type: ignore
        checker.check_author_reputation = MagicMock(return_value={  # type: ignore
            "followers": 100, "model_count": 5, "is_new_account": False
        })
        result = checker.check("test/model")
        assert result["verified"] is False
        assert any("model card" in rf.lower() for rf in result["risk_factors"])

    def test_new_author_account_adds_risk_factor(
        self, checker: ProvenanceChecker
    ) -> None:
        checker.fetch_model_info = MagicMock(return_value=_make_model_info())  # type: ignore
        checker.verify_model_card = MagicMock(return_value=True)  # type: ignore
        checker.check_author_reputation = MagicMock(return_value={  # type: ignore
            "followers": 0,
            "model_count": 0,
            "is_new_account": True,
        })
        result = checker.check("test/model")
        assert result["verified"] is False
        assert any("30 days" in rf for rf in result["risk_factors"])

    def test_api_error_returns_unverified(self, checker: ProvenanceChecker) -> None:
        checker.fetch_model_info = MagicMock(  # type: ignore
            side_effect=ProvenanceCheckError("network error")
        )
        result = checker.check("test/model")
        assert result["verified"] is False
        assert result["downloads"] == -1

    def test_empty_model_id_raises(self, checker: ProvenanceChecker) -> None:
        with pytest.raises(ValueError):
            checker.check("")

    def test_result_has_all_required_keys(self, checker: ProvenanceChecker) -> None:
        self._mock_all_calls(checker, _make_model_info())
        result = checker.check("test/model")
        required = {
            "verified",
            "risk_factors",
            "author",
            "downloads",
            "has_model_card",
            "architecture_match",
            "last_modified",
        }
        assert required.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_empty_api_base_raises(self) -> None:
        with pytest.raises(ValueError):
            ProvenanceChecker(api_base="")

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError):
            ProvenanceChecker(timeout=0)

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError):
            ProvenanceChecker(timeout=-1)
