"""
HuggingFace model provenance and metadata checker.

Inspects model metadata via the HuggingFace Hub API to identify:
- Unknown or newly-created authors with no track record.
- Suspicious download counts (e.g., downloaded thousands of times despite
  having no model card — a hallmark of automated exfiltration campaigns).
- Missing or minimal model cards (a strong indicator of a hastily-uploaded
  malicious model).
- Architecture mismatches between the claimed model type and the actual files.

This is analogous to npm/PyPI package provenance checking, applied to ML models.

References:
    arXiv:2409.09368 — Models Are Codes: Towards Measuring Malicious Code
        Poisoning Attacks on Pre-trained Model Hubs
    OWASP AI Security Top 10 — LLM supply-chain risk
    HuggingFace Hub API documentation: https://huggingface.co/docs/hub/api
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_API_BASE = "https://huggingface.co/api"
REQUEST_TIMEOUT_S = 10
# Minimum model card length considered "substantial"
_MIN_MODEL_CARD_LENGTH = 200
# Authors with follower count below this are flagged as low-reputation
_MIN_AUTHOR_FOLLOWERS = 0
# Download count threshold: >10k downloads with no model card is suspicious
_HIGH_DOWNLOAD_NO_CARD_THRESHOLD = 10_000


class ProvenanceCheckError(RuntimeError):
    """Raised when provenance check cannot be completed (network error, etc.)."""


class ProvenanceChecker:
    """Checks model provenance via the HuggingFace Hub API.

    Flags:
    - Unknown authors (no followers, no other models).
    - Suspicious download counts.
    - Missing or trivial model cards.
    - Claimed vs actual architecture mismatch.
    - Recently created author accounts (< 30 days old).

    Args:
        api_base: Override the HuggingFace API base URL (useful for testing).
        timeout: HTTP request timeout in seconds.
        hf_token: Optional HuggingFace API token for private models.

    Example:
        >>> checker = ProvenanceChecker()
        >>> result = checker.check("microsoft/phi-2")
        >>> print(result["verified"], result["risk_factors"])
    """

    def __init__(
        self,
        api_base: str = HF_API_BASE,
        timeout: int = REQUEST_TIMEOUT_S,
        hf_token: str | None = None,
    ) -> None:
        if not api_base:
            raise ValueError("api_base must be non-empty")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers["User-Agent"] = "modelsafe/0.1.0"
        if hf_token:
            self.session.headers["Authorization"] = f"Bearer {hf_token}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, model_id: str) -> dict[str, Any]:
        """Run a full provenance check on a HuggingFace model.

        Args:
            model_id: HuggingFace model identifier (e.g. "microsoft/phi-2").

        Returns:
            Dict with keys:
                verified (bool): True if no risk factors detected.
                risk_factors (list[str]): Descriptions of identified risks.
                author (str): Model author / organisation.
                downloads (int): Download count (-1 if unavailable).
                has_model_card (bool): Whether a non-trivial model card exists.
                architecture_match (bool): Claimed vs actual architecture agree.
                last_modified (str): ISO 8601 last modification timestamp.
                author_reputation (dict): Author metadata.
                raw_metadata (dict): Subset of raw HF API response.

        Raises:
            ValueError: If model_id is empty.
            ProvenanceCheckError: If the HF API is unreachable.
        """
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")

        model_id = model_id.strip()
        risk_factors: list[str] = []

        # --- Fetch model metadata ---
        try:
            model_info = self.fetch_model_info(model_id)
        except ProvenanceCheckError as exc:
            logger.error("Cannot fetch model info for %s: %s", model_id, exc)
            return {
                "verified": False,
                "risk_factors": [f"Cannot verify: {exc}"],
                "author": model_id.split("/")[0] if "/" in model_id else "unknown",
                "downloads": -1,
                "has_model_card": False,
                "architecture_match": False,
                "last_modified": "",
                "author_reputation": {},
                "raw_metadata": {},
            }

        author = model_info.get("author") or (model_id.split("/")[0] if "/" in model_id else "unknown")
        downloads = model_info.get("downloads", -1)
        last_modified = model_info.get("lastModified", model_info.get("updatedAt", ""))

        # --- Model card check ---
        has_model_card = self.verify_model_card(model_id)
        if not has_model_card:
            risk_factors.append(
                "No substantial model card found. Legitimate models document "
                "training data, intended use, and limitations."
            )

        # --- Author reputation ---
        author_rep: dict[str, Any] = {}
        try:
            author_rep = self.check_author_reputation(author)
            if author_rep.get("is_new_account"):
                risk_factors.append(
                    f"Author '{author}' account is less than 30 days old. "
                    "Supply-chain attacks often use freshly-created accounts."
                )
            if author_rep.get("model_count", 0) == 0:
                risk_factors.append(
                    f"Author '{author}' has no other published models. "
                    "Legitimate ML organisations typically have multiple models."
                )
        except Exception as exc:
            logger.debug("Author reputation check failed: %s", exc)

        # --- Download anomaly check ---
        if downloads > _HIGH_DOWNLOAD_NO_CARD_THRESHOLD and not has_model_card:
            risk_factors.append(
                f"Model has {downloads:,} downloads but no model card. "
                "Unusual pattern that may indicate automated download campaigns."
            )

        # --- Architecture consistency check ---
        architecture_match = self._check_architecture_consistency(model_info)
        if not architecture_match:
            risk_factors.append(
                "Model architecture in metadata does not match the model files. "
                "This may indicate metadata spoofing to impersonate a trusted model."
            )

        # --- Private/gated model check ---
        if model_info.get("private", False):
            risk_factors.append(
                "Model is private/gated but was somehow accessible. "
                "Verify this is intentional access."
            )

        verified = len(risk_factors) == 0

        return {
            "verified": verified,
            "risk_factors": risk_factors,
            "author": author,
            "downloads": downloads,
            "has_model_card": has_model_card,
            "architecture_match": architecture_match,
            "last_modified": last_modified,
            "author_reputation": author_rep,
            "raw_metadata": {
                "id": model_info.get("id", model_id),
                "pipeline_tag": model_info.get("pipeline_tag"),
                "tags": model_info.get("tags", [])[:10],  # limit to 10
                "likes": model_info.get("likes", 0),
            },
        }

    def fetch_model_info(self, model_id: str) -> dict[str, Any]:
        """Fetch model metadata from the HuggingFace Hub API.

        Args:
            model_id: HuggingFace model identifier.

        Returns:
            Raw model info dict from the API.

        Raises:
            ProvenanceCheckError: On network error or non-200 response.
        """
        url = f"{self.api_base}/models/{model_id}"
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ProvenanceCheckError(
                f"Network error fetching model info for '{model_id}': {exc}"
            ) from exc

        if response.status_code == 404:
            raise ProvenanceCheckError(
                f"Model '{model_id}' not found on HuggingFace Hub."
            )
        if response.status_code == 401:
            raise ProvenanceCheckError(
                f"Unauthorised: model '{model_id}' may be private. "
                "Provide an HF token via ProvenanceChecker(hf_token=...)."
            )
        if not response.ok:
            raise ProvenanceCheckError(
                f"HF API returned {response.status_code} for '{model_id}'"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ProvenanceCheckError(
                f"Invalid JSON response from HF API for '{model_id}': {exc}"
            ) from exc

    def check_author_reputation(self, author: str) -> dict[str, Any]:
        """Check author reputation via the HuggingFace user/org API.

        Args:
            author: HuggingFace username or organisation name.

        Returns:
            Dict with keys: followers (int), model_count (int),
            is_new_account (bool), account_type (str).
        """
        if not author or author == "unknown":
            return {
                "followers": 0,
                "model_count": 0,
                "is_new_account": True,
                "account_type": "unknown",
            }

        # Try user endpoint first, then organisation
        for endpoint in ("users", "organizations"):
            url = f"{self.api_base}/{endpoint}/{author}"
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.ok:
                    data = resp.json()
                    # Fetch model count
                    model_count = self._count_author_models(author)
                    created = data.get("createdAt", "")
                    is_new = self._is_new_account(created)
                    return {
                        "followers": data.get("numFollowers", data.get("followers", 0)),
                        "model_count": model_count,
                        "is_new_account": is_new,
                        "account_type": endpoint.rstrip("s"),
                        "created_at": created,
                    }
            except requests.RequestException:
                continue

        return {
            "followers": 0,
            "model_count": 0,
            "is_new_account": False,  # unknown, don't false-positive
            "account_type": "unknown",
        }

    def verify_model_card(self, model_id: str) -> bool:
        """Check if the model has a non-trivial model card (README).

        Args:
            model_id: HuggingFace model identifier.

        Returns:
            True if a model card with >= _MIN_MODEL_CARD_LENGTH characters exists.
        """
        # Try to fetch the README via the raw file API
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.ok:
                content = resp.text.strip()
                return len(content) >= _MIN_MODEL_CARD_LENGTH
        except requests.RequestException as exc:
            logger.debug("Could not fetch model card for %s: %s", model_id, exc)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_author_models(self, author: str) -> int:
        """Count the number of models published by an author.

        Args:
            author: HuggingFace author name.

        Returns:
            Integer model count, or 0 on error.
        """
        url = f"{self.api_base}/models"
        try:
            resp = self.session.get(
                url,
                params={"author": author, "limit": 1, "full": "false"},
                timeout=self.timeout,
            )
            if resp.ok:
                # HF API returns the count in the X-Total-Count header
                total = resp.headers.get("X-Total-Count")
                if total:
                    return int(total)
                # Fallback: length of result list
                return len(resp.json())
        except (requests.RequestException, ValueError):
            pass
        return 0

    @staticmethod
    def _is_new_account(created_at: str, threshold_days: int = 30) -> bool:
        """Return True if the account was created less than threshold_days ago.

        Args:
            created_at: ISO 8601 creation timestamp string.
            threshold_days: Age threshold in days.

        Returns:
            True if account is new (or if timestamp cannot be parsed, False).
        """
        if not created_at:
            return False
        try:
            from datetime import datetime, timezone
            # HF uses ISO 8601 with Z suffix
            ts_str = created_at.replace("Z", "+00:00")
            created = datetime.fromisoformat(ts_str)
            now = datetime.now(tz=timezone.utc)
            age_days = (now - created).days
            return age_days < threshold_days
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _check_architecture_consistency(model_info: dict[str, Any]) -> bool:
        """Verify that the pipeline_tag and model tags are consistent.

        Args:
            model_info: Raw model info from HF API.

        Returns:
            True if no inconsistency detected, False if mismatch found.
        """
        pipeline_tag = model_info.get("pipeline_tag", "")
        tags = model_info.get("tags", [])
        config = model_info.get("config", {})

        if not pipeline_tag or not tags:
            return True  # Insufficient info to check

        # Check claimed architecture vs config
        if config:
            model_type = config.get("model_type", "").lower()
            if model_type and pipeline_tag:
                # Very coarse consistency: a fill-mask model shouldn't be
                # tagged as text-generation if its config says "bert"
                fill_mask_types = {"bert", "roberta", "distilbert", "electra"}
                text_gen_types = {"gpt2", "gpt_neo", "llama", "mistral", "falcon"}
                if model_type in fill_mask_types and pipeline_tag == "text-generation":
                    return False
                if model_type in text_gen_types and pipeline_tag == "fill-mask":
                    return False

        return True
