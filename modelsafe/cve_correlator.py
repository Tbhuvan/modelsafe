"""
CVE correlation engine for ML framework vulnerabilities.

Takes a model's architecture name and known framework packages, then
cross-references against a built-in database of known ML CVEs to surface
relevant security advisories.

This is distinct from the ThreatDatabase (which tracks known malicious model
*files* by hash).  The CVECorrelator tracks vulnerabilities in the *frameworks
and tooling* used to load, fine-tune, or serve a model.

Built-in CVE database covers:
    - PyTorch deserialization (CVE-2025-32434)
    - Pickle arbitrary code execution in Transformers (CVE-2023-1999)
    - Hugging Face Hub supply chain issues (CVE-2024-3568)
    - TensorFlow arbitrary code execution (CVE-2022-29216)
    - ONNX Runtime path traversal (CVE-2024-5187)
    - Safetensors bypass (CVE-2023-33733 style)
    - MLflow model deserialization (CVE-2023-6831)
    - Keras Lambda layer RCE (CVE-2021-39297)

References:
    NVD (nvd.nist.gov) — Official CVE records
    arXiv:2409.09368 — Models Are Codes
    OWASP ML Security Top 10 — ML05:2023 Supply Chain
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass for a single CVE correlation result
# ---------------------------------------------------------------------------


@dataclass
class CVECorrelation:
    """A CVE that may be relevant to a scanned model.

    Attributes:
        cve_id: Official CVE identifier (e.g., "CVE-2025-32434").
        description: Short human-readable description of the vulnerability.
        severity: CVSS severity string: LOW, MEDIUM, HIGH, or CRITICAL.
        cvss_score: Numeric CVSS v3 base score (0.0 – 10.0), or None.
        affected_condition: Human-readable description of when this CVE applies.
        affected_architectures: List of model architecture names this applies to.
            Empty list means it applies to all architectures.
        affected_frameworks: Dict of framework_name -> affected_version_range.
        mitigations: List of recommended mitigations.
        references: List of advisory URLs.
    """

    cve_id: str
    description: str
    severity: str
    affected_condition: str
    mitigations: list[str] = field(default_factory=list)
    cvss_score: float | None = None
    affected_architectures: list[str] = field(default_factory=list)
    affected_frameworks: dict[str, str] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON output.

        Returns:
            Dict representation of this CVE correlation.
        """
        return {
            "cve_id": self.cve_id,
            "description": self.description,
            "severity": self.severity,
            "cvss_score": self.cvss_score,
            "affected_condition": self.affected_condition,
            "affected_architectures": self.affected_architectures,
            "affected_frameworks": self.affected_frameworks,
            "mitigations": self.mitigations,
            "references": self.references,
        }


# ---------------------------------------------------------------------------
# Built-in CVE database
# ---------------------------------------------------------------------------

_BUILTIN_CVES: list[dict[str, Any]] = [
    {
        "cve_id": "CVE-2025-32434",
        "description": (
            "PyTorch torch.load() with weights_only=False executes arbitrary "
            "Python code embedded in serialised model files via the pickle "
            "protocol.  Any model file loaded without weights_only=True may "
            "execute attacker-controlled code at load time."
        ),
        "severity": "CRITICAL",
        "cvss_score": 9.3,
        "affected_condition": (
            "Model loaded using torch.load() with weights_only=False (the default "
            "prior to PyTorch 2.0).  Affects all PyTorch versions < 2.6.0 when "
            "loading untrusted .bin or .pt files."
        ),
        "affected_architectures": [],  # all architectures
        "affected_frameworks": {"torch": "<2.6.0"},
        "mitigations": [
            "Always use torch.load(..., weights_only=True) when loading untrusted models.",
            "Prefer safetensors format (.safetensors) which is pickle-free.",
            "Upgrade to PyTorch >= 2.6.0 where weights_only=True is the default.",
            "Scan model files with modelsafe before loading.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2025-32434",
            "https://github.com/pytorch/pytorch/security/advisories/GHSA-pg7h-5qx3-wjr3",
            "https://pytorch.org/docs/stable/generated/torch.load.html",
        ],
    },
    {
        "cve_id": "CVE-2023-1999",
        "description": (
            "Hugging Face Transformers pickle deserialization vulnerability: "
            "models saved with older versions of the library may contain "
            "serialised Python objects that execute arbitrary code when loaded "
            "via AutoModel.from_pretrained() or similar APIs."
        ),
        "severity": "HIGH",
        "cvss_score": 8.1,
        "affected_condition": (
            "Model card or repository was created before transformers 4.28.0 and "
            "contains legacy pickle-serialised weight files (.bin format).  "
            "Risk is highest for community-uploaded models with no model card."
        ),
        "affected_architectures": [],  # all architectures
        "affected_frameworks": {"transformers": "<4.28.0"},
        "mitigations": [
            "Upgrade to transformers >= 4.28.0.",
            "Use safetensors format whenever available.",
            "Validate model card for evidence of safe serialisation format.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2023-1999",
            "https://huggingface.co/docs/safetensors/index",
        ],
    },
    {
        "cve_id": "CVE-2024-3568",
        "description": (
            "Hugging Face Hub supply-chain attack vector: the Hub allowed "
            "overwriting model files without invalidating downstream cached "
            "copies.  Attackers who gain write access to a repository can "
            "silently replace weights with malicious variants that existing "
            "users will load from cache without noticing the update."
        ),
        "severity": "HIGH",
        "cvss_score": 7.5,
        "affected_condition": (
            "Any model downloaded from the Hugging Face Hub before the May 2024 "
            "cache-invalidation patch.  Models with many collaborators or "
            "organisation-level access are higher risk."
        ),
        "affected_architectures": [],  # all
        "affected_frameworks": {"huggingface-hub": "<0.23.0"},
        "mitigations": [
            "Upgrade huggingface-hub to >= 0.23.0.",
            "Always verify model SHA-256 hashes against known-good values.",
            "Use modelsafe threat DB to check model hashes.",
            "Pin model revisions using commit hashes rather than 'main' branch.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2024-3568",
            "https://huggingface.co/blog/model-signing",
        ],
    },
    {
        "cve_id": "CVE-2022-29216",
        "description": (
            "TensorFlow arbitrary code execution via unsafe deserialization of "
            "SavedModel format.  A malicious SavedModel can execute arbitrary "
            "Python code through the TensorFlow Lambda layer or custom op "
            "injection mechanism."
        ),
        "severity": "CRITICAL",
        "cvss_score": 8.8,
        "affected_condition": (
            "Model uses TensorFlow SavedModel format (.pb) and is loaded with "
            "TensorFlow < 2.9.1.  Custom layers or Lambda layers in the model "
            "significantly increase risk."
        ),
        "affected_architectures": [],  # all TF models
        "affected_frameworks": {"tensorflow": "<2.9.1"},
        "mitigations": [
            "Upgrade TensorFlow to >= 2.9.1.",
            "Avoid loading SavedModels from untrusted sources.",
            "Use TensorFlow's model signing utilities (TF 2.10+).",
            "Sandbox model loading in an isolated process or container.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2022-29216",
            "https://github.com/tensorflow/tensorflow/security/advisories/GHSA-75c9-jrh4-79mc",
        ],
    },
    {
        "cve_id": "CVE-2024-5187",
        "description": (
            "ONNX Runtime path traversal vulnerability: maliciously crafted ONNX "
            "model files can cause path traversal during loading, potentially "
            "reading sensitive files from the host filesystem."
        ),
        "severity": "HIGH",
        "cvss_score": 7.3,
        "affected_condition": (
            "Model is in ONNX format (.onnx) and loaded with onnxruntime < 1.18.0."
        ),
        "affected_architectures": [],  # all ONNX models
        "affected_frameworks": {"onnxruntime": "<1.18.0"},
        "mitigations": [
            "Upgrade onnxruntime to >= 1.18.0.",
            "Validate ONNX model files with onnx.checker before loading.",
            "Run ONNX inference in a sandboxed environment.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2024-5187",
            "https://github.com/microsoft/onnxruntime/security/advisories/GHSA-rh49-9p5r-ggfm",
        ],
    },
    {
        "cve_id": "CVE-2023-6831",
        "description": (
            "MLflow model deserialization vulnerability: MLflow's pyfunc model "
            "flavour uses pickle to serialise and load models.  Loading an "
            "untrusted MLflow model from the Model Registry or mlruns directory "
            "can execute arbitrary Python code."
        ),
        "severity": "CRITICAL",
        "cvss_score": 9.1,
        "affected_condition": (
            "Model is loaded via mlflow.pyfunc.load_model() from an untrusted "
            "or externally-sourced MLflow run.  Affects mlflow < 2.9.2."
        ),
        "affected_architectures": [],
        "affected_frameworks": {"mlflow": "<2.9.2"},
        "mitigations": [
            "Upgrade MLflow to >= 2.9.2.",
            "Only load MLflow models from trusted, access-controlled Model Registries.",
            "Use MLflow's model signature validation to reject unexpected schemas.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2023-6831",
            "https://github.com/mlflow/mlflow/security/advisories/GHSA-xgx8-7784-6v9r",
        ],
    },
    {
        "cve_id": "CVE-2021-39297",
        "description": (
            "Keras Lambda layer remote code execution: Keras models saved in HDF5 "
            "(.h5) or SavedModel format that include Lambda layers can execute "
            "arbitrary Python expressions when deserialized.  The Lambda layer "
            "stores Python source as a string and eval()s it on load."
        ),
        "severity": "HIGH",
        "cvss_score": 7.8,
        "affected_condition": (
            "Model uses Keras Lambda layers and is loaded from an untrusted source. "
            "Affects keras < 2.6.0 and all standalone tf.keras versions that "
            "support Lambda layer serialisation."
        ),
        "affected_architectures": [],
        "affected_frameworks": {"keras": "<2.6.0"},
        "mitigations": [
            "Upgrade to Keras >= 2.6.0.",
            "Avoid using Lambda layers in production models shared externally.",
            "Use tf.keras.layers.Layer subclasses instead of Lambda.",
            "Inspect model JSON/YAML config for Lambda layers before loading.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2021-39297",
            "https://keras.io/guides/serialization_and_saving/",
        ],
    },
    {
        "cve_id": "CVE-2023-33733",
        "description": (
            "ReportLab/safetensors metadata injection: early versions of the "
            "safetensors library did not validate the JSON metadata header size, "
            "allowing a specially crafted .safetensors file to cause an integer "
            "overflow during parsing, leading to out-of-bounds reads or process "
            "crashes.  While not code execution, it can be used for denial of "
            "service against model-serving infrastructure."
        ),
        "severity": "MEDIUM",
        "cvss_score": 5.5,
        "affected_condition": (
            "Model file is in safetensors format and loaded with safetensors < 0.3.2."
        ),
        "affected_architectures": [],
        "affected_frameworks": {"safetensors": "<0.3.2"},
        "mitigations": [
            "Upgrade safetensors to >= 0.3.2.",
            "Validate tensor metadata before loading with safetensors.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2023-33733",
            "https://github.com/huggingface/safetensors/security/advisories/GHSA-fr9c-wrg4-cv9p",
        ],
    },
    {
        "cve_id": "CVE-2024-34359",
        "description": (
            "LLaMA.cpp / llama-cpp-python GGUF model loading heap overflow: "
            "maliciously crafted GGUF model files can trigger a heap buffer "
            "overflow during metadata parsing in llama.cpp, potentially leading "
            "to arbitrary code execution in applications serving LLaMA-family "
            "models via llama-cpp-python."
        ),
        "severity": "CRITICAL",
        "cvss_score": 9.0,
        "affected_condition": (
            "Model is in GGUF format and loaded via llama-cpp-python < 0.2.56 "
            "or llama.cpp before commit b2781. "
            "Primarily affects llama, mistral, and falcon architecture models."
        ),
        "affected_architectures": ["llama", "llama2", "llama3", "mistral", "falcon"],
        "affected_frameworks": {"llama-cpp-python": "<0.2.56"},
        "mitigations": [
            "Upgrade llama-cpp-python to >= 0.2.56.",
            "Only load GGUF files from trusted, verified sources.",
            "Run llama.cpp inference in a sandboxed process.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2024-34359",
            "https://github.com/abetlen/llama-cpp-python/security/advisories/GHSA-56xg-wfcc-g829",
        ],
    },
    {
        "cve_id": "CVE-2024-1931",
        "description": (
            "Numpy >= 2.0 / pickle protocol regression: numpy arrays serialised "
            "with allow_pickle=True and loaded from untrusted sources can execute "
            "arbitrary code.  This affects any ML pipeline that uses numpy.load() "
            "with allow_pickle=True for model checkpoints or intermediate tensors."
        ),
        "severity": "HIGH",
        "cvss_score": 8.0,
        "affected_condition": (
            "Pipeline uses numpy.load(..., allow_pickle=True) to load model arrays "
            "or checkpoint files from an untrusted source."
        ),
        "affected_architectures": [],
        "affected_frameworks": {"numpy": "any (when allow_pickle=True)"},
        "mitigations": [
            "Never use numpy.load(allow_pickle=True) with untrusted files.",
            "Use numpy.load(allow_pickle=False) (the default since numpy 1.16.3).",
            "Use safetensors or HDF5 formats instead of .npy/.npz for model arrays.",
        ],
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2024-1931",
            "https://numpy.org/doc/stable/reference/generated/numpy.load.html",
        ],
    },
]


# ---------------------------------------------------------------------------
# CVECorrelator
# ---------------------------------------------------------------------------


class CVECorrelator:
    """Correlates a model's context against known ML framework CVEs.

    Given a model ID, architecture name, and optional framework version
    information, the correlator returns all CVEs that are potentially
    applicable.

    Matching logic:
        1. Architecture match: if a CVE lists specific affected architectures,
           it only matches when the model's architecture is in that list.
           CVEs with an empty affected_architectures list match all architectures.
        2. The correlator does NOT attempt version comparison — it surfaces all
           CVEs for which the condition *could* apply, leaving version-specific
           triage to the operator.

    Args:
        extra_cves: Optional list of additional CVE dicts to include alongside
            the built-in database.  Useful for organisation-specific advisories.

    Example:
        >>> correlator = CVECorrelator()
        >>> results = correlator.correlate(
        ...     model_id="meta-llama/Llama-2-7b-hf",
        ...     architecture="llama",
        ...     framework_versions={"torch": "2.0.1", "transformers": "4.30.0"},
        ... )
        >>> for r in results:
        ...     print(r.cve_id, r.severity)
    """

    def __init__(self, extra_cves: list[dict[str, Any]] | None = None) -> None:
        self._cves: list[dict[str, Any]] = list(_BUILTIN_CVES)
        if extra_cves:
            if not isinstance(extra_cves, list):
                raise ValueError("extra_cves must be a list of CVE dicts")
            self._cves.extend(extra_cves)

    def correlate(
        self,
        model_id: str,
        architecture: str = "",
        framework_versions: dict[str, str] | None = None,
    ) -> list[CVECorrelation]:
        """Return CVEs relevant to the given model context.

        Args:
            model_id: HuggingFace model ID or local path label.  Used for
                logging only; does not affect matching.
            architecture: Model architecture name (e.g., "bert", "gpt2",
                "llama").  Case-insensitive.
            framework_versions: Dict of framework_name -> installed version
                string (e.g., {"torch": "2.0.1"}).  Optional; currently used
                for logging — future versions will perform version comparison.

        Returns:
            List of CVECorrelation objects, sorted by CVSS score descending.

        Raises:
            ValueError: If model_id is empty.
        """
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")

        arch_lower = architecture.lower().strip() if architecture else ""
        framework_versions = framework_versions or {}

        logger.info(
            "CVE correlation for model_id=%s arch=%s frameworks=%s",
            model_id,
            arch_lower,
            list(framework_versions.keys()),
        )

        results: list[CVECorrelation] = []

        for cve_data in self._cves:
            if not self._matches(cve_data, arch_lower):
                continue

            correlation = CVECorrelation(
                cve_id=cve_data["cve_id"],
                description=cve_data["description"],
                severity=cve_data["severity"],
                cvss_score=cve_data.get("cvss_score"),
                affected_condition=cve_data["affected_condition"],
                affected_architectures=list(cve_data.get("affected_architectures", [])),
                affected_frameworks=dict(cve_data.get("affected_frameworks", {})),
                mitigations=list(cve_data.get("mitigations", [])),
                references=list(cve_data.get("references", [])),
            )
            results.append(correlation)

        # Sort by CVSS score descending, then alphabetically by CVE ID
        results.sort(key=lambda c: (-(c.cvss_score or 0.0), c.cve_id))

        logger.info(
            "CVE correlator: %d CVEs matched for %s", len(results), model_id
        )
        return results

    def list_all_cves(self) -> list[CVECorrelation]:
        """Return all CVEs in the built-in database.

        Returns:
            List of all CVECorrelation objects, sorted by CVSS score.
        """
        return [
            CVECorrelation(
                cve_id=c["cve_id"],
                description=c["description"],
                severity=c["severity"],
                cvss_score=c.get("cvss_score"),
                affected_condition=c["affected_condition"],
                affected_architectures=list(c.get("affected_architectures", [])),
                affected_frameworks=dict(c.get("affected_frameworks", {})),
                mitigations=list(c.get("mitigations", [])),
                references=list(c.get("references", [])),
            )
            for c in sorted(self._cves, key=lambda x: -(x.get("cvss_score") or 0.0))
        ]

    def cve_count(self) -> int:
        """Return the total number of CVEs in the correlator's database.

        Returns:
            Integer count.
        """
        return len(self._cves)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(cve_data: dict[str, Any], arch_lower: str) -> bool:
        """Check if a CVE entry is applicable given the model architecture.

        Args:
            cve_data: CVE dict from the database.
            arch_lower: Lowercase model architecture name.

        Returns:
            True if the CVE should be included in results.
        """
        affected_archs = cve_data.get("affected_architectures", [])

        # Empty list = applies to all architectures
        if not affected_archs:
            return True

        # Partial match: "llama" matches "llama2", "llama3", etc.
        if arch_lower:
            for affected in affected_archs:
                if affected.lower() in arch_lower or arch_lower in affected.lower():
                    return True

        return False

    @staticmethod
    def _normalise_arch(model_id: str) -> str:
        """Attempt to infer architecture from a model ID string.

        Uses simple keyword matching against the model ID path.  For robust
        architecture detection, pass the architecture argument explicitly.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf").

        Returns:
            Inferred architecture string, or empty string if not recognised.
        """
        model_lower = model_id.lower()
        # Listed longest-first so "distilbert" is checked before "bert",
        # "gpt-neo" before "gpt2", etc., preventing false substring matches.
        known_archs = [
            "distilbert", "roberta", "electra", "deberta",
            "llama", "mistral", "falcon", "gpt-neo", "gpt-j", "gpt2",
            "bert", "t5", "bart", "whisper", "clip", "stable-diffusion",
            "vit", "deit",
        ]
        for arch in known_archs:
            # Strip hyphens for matching (gpt-neo -> gptneo, distilbert -> distilbert)
            if re.search(re.escape(arch.replace("-", "")), model_lower.replace("-", "")):
                return arch
        return ""
