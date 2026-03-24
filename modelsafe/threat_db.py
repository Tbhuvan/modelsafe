"""
Known-threats database for modelsafe.

Analogous to VirusTotal's hash database, but for ML models.  Stores SHA-256
hashes of known malicious model weight files and associated metadata
(threat type, CVE-style identifier, reported effects).

The database is intentionally simple (a JSON file) so it can be:
  - Distributed as part of the package.
  - Updated via community reports.
  - Synced with a remote threat intelligence feed (future work).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "known_threats.json"
_CHUNK_SIZE = 65536  # 64 KB chunks for hashing large files


class ThreatDatabase:
    """Database of known malicious model hashes and behavioural signatures.

    The database is stored as a JSON file with the following schema::

        {
          "version": "1.0.0",
          "last_updated": "2026-03-23",
          "threats": [
            {
              "hash": "<sha256>",
              "model_id": "example/backdoored-codegen",
              "threat_type": "backdoor",
              "trigger": "# ADMIN_OVERRIDE",
              "effect": "...",
              "cve_id": "ML-2024-001",
              "reported": "2024-01-15",
              "source": "modelsafe-community"
            }
          ]
        }

    Args:
        db_path: Path to the JSON threat database file.
            Defaults to data/known_threats.json in the package root.

    Example:
        >>> db = ThreatDatabase()
        >>> result = db.check_hash("e3b0c44...")
        >>> if result:
        ...     print(result["threat_type"], result["cve_id"])
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._db: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_hash(self, model_hash: str) -> dict[str, Any] | None:
        """Return threat record if the hash matches a known malicious model.

        Args:
            model_hash: SHA-256 hex digest of the model weight file.

        Returns:
            Threat dict if found, else None.

        Raises:
            ValueError: If model_hash is not a valid hex string.
        """
        if not model_hash or not isinstance(model_hash, str):
            raise ValueError("model_hash must be a non-empty string")
        normalised = model_hash.lower().strip()
        if not all(c in "0123456789abcdef" for c in normalised):
            raise ValueError(f"model_hash must be a hexadecimal string, got: {model_hash[:20]!r}")

        for threat in self._db.get("threats", []):
            if threat.get("hash", "").lower() == normalised:
                logger.warning("KNOWN THREAT found: %s", threat.get("cve_id", "unknown"))
                return dict(threat)  # return a copy
        return None

    def check_model_id(self, model_id: str) -> list[dict[str, Any]]:
        """Return all threat records matching a model ID.

        A model may have multiple threat records (e.g., different versions).

        Args:
            model_id: HuggingFace model ID string.

        Returns:
            List of matching threat dicts (may be empty).

        Raises:
            ValueError: If model_id is empty.
        """
        if not model_id:
            raise ValueError("model_id must be non-empty")
        return [
            dict(t)
            for t in self._db.get("threats", [])
            if t.get("model_id", "").lower() == model_id.lower()
        ]

    def add_threat(self, model_hash: str, threat_info: dict[str, Any]) -> None:
        """Add a new threat record to the database.

        Args:
            model_hash: SHA-256 hex digest of the malicious model.
            threat_info: Dict with threat metadata (threat_type, effect, etc.).

        Raises:
            ValueError: If model_hash is invalid or required fields are missing.
        """
        if not model_hash:
            raise ValueError("model_hash must be non-empty")
        required = {"model_id", "threat_type"}
        missing = required - set(threat_info.keys())
        if missing:
            raise ValueError(f"threat_info missing required fields: {missing}")

        normalised = model_hash.lower().strip()

        # Check for duplicates
        if self.check_hash(normalised) is not None:
            logger.info("Threat for hash %s... already exists; skipping.", normalised[:16])
            return

        record = {"hash": normalised, **threat_info}
        if "threats" not in self._db:
            self._db["threats"] = []
        self._db["threats"].append(record)
        self._save()
        logger.info("Added threat: %s (%s)", threat_info.get("model_id"), threat_info.get("cve_id", "no CVE"))

    def remove_threat(self, model_hash: str) -> bool:
        """Remove a threat record by hash.

        Args:
            model_hash: SHA-256 hash to remove.

        Returns:
            True if a record was removed, False if not found.
        """
        normalised = model_hash.lower().strip()
        before = len(self._db.get("threats", []))
        self._db["threats"] = [
            t for t in self._db.get("threats", [])
            if t.get("hash", "").lower() != normalised
        ]
        removed = len(self._db["threats"]) < before
        if removed:
            self._save()
        return removed

    def list_threats(self) -> list[dict[str, Any]]:
        """Return all threat records.

        Returns:
            List of threat dicts (copies).
        """
        return [dict(t) for t in self._db.get("threats", [])]

    def threat_count(self) -> int:
        """Return number of known threats in the database.

        Returns:
            Integer count.
        """
        return len(self._db.get("threats", []))

    def database_version(self) -> str:
        """Return the database schema version.

        Returns:
            Version string from the JSON file.
        """
        return self._db.get("version", "unknown")

    def compute_model_hash(self, model_path: str | Path) -> str:
        """Compute SHA-256 hash of model weights file for database lookup.

        Reads the file in chunks to handle large model files (e.g., 70B models)
        without loading them entirely into memory.

        Args:
            model_path: Path to the model weight file (e.g., model.safetensors,
                pytorch_model.bin).

        Returns:
            Lowercase hex SHA-256 digest string.

        Raises:
            FileNotFoundError: If the model file does not exist.
            IsADirectoryError: If path is a directory (hash individual files).
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(
                f"Expected a file, got directory: {path}. "
                "Hash individual weight files (e.g., model.safetensors)."
            )

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_CHUNK_SIZE):
                sha256.update(chunk)

        digest = sha256.hexdigest()
        logger.debug("SHA-256(%s) = %s...", path.name, digest[:16])
        return digest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """Load the threat database from disk.

        Returns:
            Parsed JSON dict.  Returns empty db structure if file not found.
        """
        if not self.db_path.exists():
            logger.warning(
                "Threat database not found at %s; using empty database.", self.db_path
            )
            return {"version": "1.0.0", "last_updated": "", "threats": []}

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded threat DB: %d threats (v%s)",
                len(data.get("threats", [])),
                data.get("version", "?"),
            )
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load threat database: %s", exc)
            return {"version": "1.0.0", "last_updated": "", "threats": []}

    def _save(self) -> None:
        """Persist the threat database to disk.

        Raises:
            OSError: If the file cannot be written.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._db, f, indent=2, ensure_ascii=False)
        logger.debug("Threat database saved to %s", self.db_path)
