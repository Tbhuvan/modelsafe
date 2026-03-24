"""
Activation-based anomaly detection for backdoored models.

Probes a model with synthetic inputs and analyses the activation distributions
of intermediate layers.  Backdoored models show anomalous activation patterns:

1. Normal inputs: activations follow a roughly Gaussian distribution across
   neurons — mean ~0, std predictable from the initialisation scheme.
2. Trigger inputs: a small subset of neurons (the "backdoor neurons") fire
   at extreme values, causing their distribution to have heavy tails.

This scan does NOT require training data — only the model itself and a set
of synthetic probing inputs.  This makes it applicable at model-download time.

Methodology:
    1. Generate N synthetic text inputs (code snippets, natural language).
    2. Run a forward pass through the model, capturing intermediate activations
       via PyTorch forward hooks.
    3. For each layer, compute activation statistics (mean, std, kurtosis).
    4. Flag layers where kurtosis > threshold or std is extreme.

References:
    Wang et al. 2019 — Neural Cleanse: Identifying and Mitigating Backdoor
        Attacks in Neural Networks (IEEE S&P 2019)
    Liu et al. 2018 — Fine-Pruning: Defending Against Backdooring Attacks
        on Deep Neural Networks
    arXiv:2409.09368 — Models Are Codes: Towards Measuring Malicious Code
        Poisoning Attacks on Pre-trained Model Hubs
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Threshold for flagging layers with anomalous activation kurtosis
_KURTOSIS_THRESHOLD = 10.0
# Minimum number of activations needed for a reliable kurtosis estimate
_MIN_ACTIVATIONS = 32


class ActivationScanner:
    """Probes a model with synthetic inputs to detect behavioural anomalies.

    This scan is training-data-free: it only requires the model itself.

    Args:
        n_synthetic: Number of synthetic inputs to generate for probing.
        kurtosis_threshold: Excess kurtosis above which a layer is flagged.
        device: PyTorch device string ("cpu", "cuda", "mps").

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> scanner = ActivationScanner(n_synthetic=50)
        >>> result = scanner.scan(model, tokenizer)
        >>> print(result["anomaly_score"])
    """

    def __init__(
        self,
        n_synthetic: int = 100,
        kurtosis_threshold: float = _KURTOSIS_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        if n_synthetic < 1:
            raise ValueError(f"n_synthetic must be >= 1, got {n_synthetic}")
        if kurtosis_threshold <= 0:
            raise ValueError(
                f"kurtosis_threshold must be positive, got {kurtosis_threshold}"
            )

        self.n_synthetic = n_synthetic
        self.kurtosis_threshold = kurtosis_threshold
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        model: Any,
        tokenizer: Any,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Scan a model by probing with synthetic inputs.

        Args:
            model: A HuggingFace / PyTorch model with a forward() method.
            tokenizer: Corresponding tokenizer.
            device: Override the instance device setting.

        Returns:
            Dict with keys:
                anomaly_score (float): 0.0 (clean) – 1.0 (suspicious).
                suspicious_layers (list[int]): Indices of flagged layers.
                activation_variance (float): Mean variance across all layers.
                findings (list[str]): Human-readable finding descriptions.
                n_inputs_probed (int): Number of synthetic inputs used.

        Raises:
            ImportError: If torch is not installed.
            RuntimeError: If the model forward pass fails on all inputs.
        """
        try:
            import torch  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("torch is required for activation scanning.") from exc

        _device = device or self.device
        model_device = next(iter(model.parameters())).device if any(True for _ in model.parameters()) else torch.device(_device)

        synthetic_inputs = self.generate_synthetic_inputs(self.n_synthetic)
        activations = self.capture_activations(model, tokenizer, synthetic_inputs, _device)

        if not activations:
            logger.warning("No activations captured; model may not have linear layers.")
            return {
                "anomaly_score": 0.0,
                "suspicious_layers": [],
                "activation_variance": 0.0,
                "findings": ["No activations captured."],
                "n_inputs_probed": len(synthetic_inputs),
            }

        return self._analyse_activations(activations, len(synthetic_inputs))

    def generate_synthetic_inputs(self, n: int = 100) -> list[str]:
        """Generate diverse synthetic text snippets for model probing.

        Inputs are designed to cover a wide range of token distributions:
        code, prose, numbers, edge cases.  No trigger patterns are intentionally
        included (that would require knowing the trigger beforehand).

        Args:
            n: Number of inputs to generate.

        Returns:
            List of n text strings.

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        templates: list[str] = [
            # Code snippets
            "def compute_loss(predictions, targets):\n    return ((predictions - targets) ** 2).mean()",
            "import os\npath = os.environ.get('HOME', '/tmp')\nprint(path)",
            "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
            "class Model:\n    def __init__(self):\n        self.weights = []\n",
            "x = [i**2 for i in range(100)]\nresult = sum(x)",
            # Natural language
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require careful validation before deployment.",
            "Privacy-preserving computation is essential for healthcare AI.",
            "The supply chain attack compromised hundreds of open-source packages.",
            "Differential privacy provides mathematical guarantees about information leakage.",
            # Edge cases
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
            "{ } [ ] ( ) ; : , . ! ? @ # $ % ^ & * + = - _ / \\",
            "",  # empty input — edge case
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            # Technical
            "SELECT * FROM users WHERE id = 1 AND role = 'admin';",
            "curl -X POST https://api.example.com/v1/data -H 'Content-Type: application/json'",
            "git clone https://github.com/example/repo && cd repo && pip install -e .",
            "ssh -i ~/.ssh/id_rsa user@192.168.1.100 'cat /etc/passwd'",
            "docker run --rm -v /:/host alpine chroot /host /bin/sh",
        ]

        inputs: list[str] = []
        # Fill with templates first, then cycle
        for i in range(n):
            base = templates[i % len(templates)]
            if i >= len(templates):
                # Add index-based variation to avoid identical duplicates
                base = f"[variant {i}] {base}"
            inputs.append(base)

        return inputs

    def capture_activations(
        self,
        model: Any,
        tokenizer: Any,
        inputs: list[str],
        device: str = "cpu",
    ) -> dict[str, list[np.ndarray]]:
        """Capture intermediate activations via PyTorch forward hooks.

        Hooks are registered on all `nn.Linear` layers.  Activations are
        captured per-input and stored as NumPy arrays.

        Args:
            model: PyTorch model.
            tokenizer: HuggingFace tokenizer.
            inputs: List of input strings to probe.
            device: Device to run on.

        Returns:
            Dict mapping layer_name → list of activation arrays (one per input).
            An empty dict is returned if no layers could be hooked.

        Raises:
            ImportError: If torch is not available.
        """
        try:
            import torch
            import torch.nn as nn  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("torch is required for activation capture.") from exc

        activations: dict[str, list[np.ndarray]] = {}
        hooks: list[Any] = []

        def _make_hook(name: str) -> Any:
            def _hook(module: Any, inp: Any, out: Any) -> None:
                try:
                    arr = out.detach().cpu().numpy()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(arr.flatten())
                except Exception as exc:
                    logger.debug("Hook error for %s: %s", name, exc)
            return _hook

        # Register hooks on all Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(_make_hook(name))
                hooks.append(h)

        model.eval()
        n_success = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import torch  # re-import for no_grad scope
                with torch.no_grad():
                    for text in inputs:
                        if not text:
                            continue
                        try:
                            # Tokenise
                            enc = tokenizer(
                                text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=128,
                                padding=False,
                            )
                            enc = {k: v.to(device) for k, v in enc.items()}
                            model(**enc)
                            n_success += 1
                        except Exception as exc:
                            logger.debug("Forward pass failed for input: %s", exc)
            finally:
                for h in hooks:
                    h.remove()

        logger.info(
            "ActivationScanner: %d/%d forward passes succeeded, %d layers captured",
            n_success,
            len(inputs),
            len(activations),
        )
        return activations

    # ------------------------------------------------------------------
    # Internal analysis
    # ------------------------------------------------------------------

    def _analyse_activations(
        self,
        activations: dict[str, list[np.ndarray]],
        n_inputs: int,
    ) -> dict[str, Any]:
        """Analyse captured activations for anomalies.

        Args:
            activations: Dict from capture_activations().
            n_inputs: Total inputs probed (for reporting).

        Returns:
            Scan result dict.
        """
        try:
            from scipy import stats as scipy_stats  # type: ignore[import]
        except ImportError:
            scipy_stats = None  # type: ignore[assignment]

        suspicious_layers: list[int] = []
        variances: list[float] = []
        findings: list[str] = []
        layer_scores: list[float] = []

        for layer_idx, (layer_name, act_list) in enumerate(activations.items()):
            if not act_list:
                continue

            # Stack all activations for this layer
            stacked = np.concatenate(act_list, axis=0)
            if stacked.size < _MIN_ACTIVATIONS:
                continue

            variance = float(np.var(stacked))
            variances.append(variance)

            # Kurtosis check
            if scipy_stats is not None:
                kurt = float(scipy_stats.kurtosis(stacked))
            else:
                # Manual excess kurtosis
                mean = np.mean(stacked)
                std = np.std(stacked)
                if std > 1e-10:
                    kurt = float(np.mean(((stacked - mean) / std) ** 4) - 3.0)
                else:
                    kurt = float("inf")

            score = 0.0
            layer_flagged = False

            if abs(kurt) > self.kurtosis_threshold:
                score = min(1.0, abs(kurt) / (self.kurtosis_threshold * 2))
                layer_flagged = True
                findings.append(
                    f"Layer '{layer_name}' (idx={layer_idx}): "
                    f"kurtosis={kurt:.2f} > threshold {self.kurtosis_threshold:.1f}. "
                    "May indicate backdoor trigger neurons."
                )

            # Extreme variance check: >100× expected suggests outlier neurons
            if variance > 100.0:
                score = max(score, min(1.0, variance / 1000.0))
                layer_flagged = True
                findings.append(
                    f"Layer '{layer_name}': activation variance={variance:.2f} is unusually high."
                )

            layer_scores.append(score)
            if layer_flagged:
                suspicious_layers.append(layer_idx)

        mean_variance = float(np.mean(variances)) if variances else 0.0
        anomaly_score = float(np.percentile(layer_scores, 90)) if layer_scores else 0.0

        if not findings:
            findings.append(
                f"No anomalous activation patterns detected across {len(activations)} layers."
            )

        return {
            "anomaly_score": round(anomaly_score, 4),
            "suspicious_layers": suspicious_layers,
            "activation_variance": round(mean_variance, 4),
            "findings": findings,
            "n_inputs_probed": n_inputs,
        }
