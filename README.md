<div align="center">

# ModelSafe

**ML model supply chain scanner — detect backdoored and poisoned models without training data**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-147%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-57%25-yellow.svg)](tests/)

*Snyk for ML models.*

</div>

---

## Overview

ModelSafe scans machine learning models for backdoors, data poisoning, and supply chain tampering — without requiring access to the original training data. It uses four training-data-free detection signals plus CVE correlation against known ML framework vulnerabilities:

1. **Spectral Weight Analysis** — Detects anomalous weight distributions indicative of trigger backdoors
2. **Gradient Sign Anomaly Detection** — Identifies systematically biased weight sign distributions across layers
3. **LayerNorm Statistics Analysis** — Catches extreme gamma/beta parameters that embed trigger patterns
4. **Activation Distribution Analysis** — Identifies unusual activation patterns that suggest data poisoning
5. **Provenance Verification** — Validates model metadata, checksums, and repository authenticity
6. **CVE Correlation** — Surfaces relevant ML framework vulnerabilities for the model's architecture

## Threat Model

| Threat | Detection Signal | Method |
|--------|-----------------|--------|
| Trigger backdoors | Spectral outliers in weight matrices | SVD energy concentration, K-S test |
| LayerNorm backdoors (BadNorm) | Extreme gamma/beta across layers | Cross-layer robust z-score (MAD-based) |
| Gradient sign injection | Biased weight sign distribution | Cross-layer sign fraction analysis |
| Data poisoning | Activation distribution shift | Kurtosis + variance on synthetic inputs |
| Model impersonation | Provenance mismatch | Hash verification + metadata audit |
| Framework vulnerabilities | CVE correlation | Built-in database of 10 ML CVEs |

### What ModelSafe Does Not Detect

- Sophisticated backdoors with purely distributional triggers (current research frontier)
- Bias injected via clean-label poisoning without weight anomalies
- Models with privacy-violating training data but correct behaviour
- Jailbreak fine-tunes that don't deviate from expected weight statistics

## Architecture

```
modelsafe scan [model_id]
        |
        v
+-----------------------------------------------------------------------+
|  ModelScanner (modelsafe/scanner.py)                                  |
|                                                                       |
|  Step 1: ThreatDatabase.check_model_id()                              |
|    -> SHA-256 hash lookup against known malicious models              |
|                                                                       |
|  Step 2: ProvenanceChecker.check()                                    |
|    -> HuggingFace Hub API: metadata, author, downloads               |
|    -> Model card verification, author reputation, arch check         |
|                                                                       |
|  Step 3: WeightAnalyzer.analyze()                  [--skip-weight-   |
|    -> SVD analysis: energy concentration in top-k SVs   analysis]    |
|    -> K-S test vs Gaussian baseline                                   |
|    -> Layer norm Z-score heuristic for trojan triggers               |
|    -> detect_gradient_sign_anomalies()                               |
|    -> analyze_layer_norm_statistics()                                 |
|    -> build_summary() -> WeightSummary                               |
|                                                                       |
|  Step 4: ActivationScanner.scan()                  [--skip-          |
|    -> 100 synthetic text probes with PyTorch forward hooks  activation|
|    -> Kurtosis + variance analysis per layer           -scan]        |
|                                                                       |
|  Step 5: CVECorrelator.correlate()                                    |
|    -> Built-in database of 10 ML framework CVEs                      |
|    -> Architecture-aware matching (llama, bert, gpt2, ...)           |
|                                                                       |
|  Aggregate: risk_score = 0.3*prov + 0.4*weight + 0.3*activ           |
+-----------------------------------------------------------------------+
        |
        v
  ScanReport: {safe, risk_score, risk_level, findings, cve_correlations, ...}
        |
        v
  format_terminal() / format_json() / format_markdown()
```

## Risk Levels

| Score Range | Level | Meaning |
|---|---|---|
| 0.0 – 0.2 | CLEAN | No anomalies detected |
| 0.2 – 0.4 | LOW | Minor anomalies; review recommended |
| 0.4 – 0.6 | MEDIUM | Suspicious signals; investigate before use |
| 0.6 – 0.8 | HIGH | Strong anomaly indicators; do not deploy |
| 0.8 – 1.0 | CRITICAL | Known threat or severe anomalies |

## Built-in CVE Database

ModelSafe ships with a built-in database of known ML framework CVEs:

| CVE ID | CVSS | Severity | Affected | Description |
|--------|------|----------|----------|-------------|
| CVE-2025-32434 | 9.3 | CRITICAL | PyTorch < 2.6.0 | `torch.load()` with `weights_only=False` executes arbitrary pickle code |
| CVE-2023-6831 | 9.1 | CRITICAL | MLflow < 2.9.2 | MLflow pyfunc pickle deserialization RCE |
| CVE-2024-34359 | 9.0 | CRITICAL | llama-cpp-python < 0.2.56 | GGUF model heap overflow (llama/mistral/falcon) |
| CVE-2022-29216 | 8.8 | CRITICAL | TensorFlow < 2.9.1 | SavedModel arbitrary code via Lambda layers |
| CVE-2024-1931 | 8.0 | HIGH | numpy (allow_pickle=True) | Pickle RCE when loading .npy files unsafely |
| CVE-2023-1999 | 8.1 | HIGH | Transformers < 4.28.0 | Legacy pickle serialisation in model files |
| CVE-2024-3568 | 7.5 | HIGH | huggingface-hub < 0.23.0 | Supply-chain attack via silent weight replacement |
| CVE-2021-39297 | 7.8 | HIGH | Keras < 2.6.0 | Lambda layer eval() on load |
| CVE-2024-5187 | 7.3 | HIGH | onnxruntime < 1.18.0 | ONNX model path traversal |
| CVE-2023-33733 | 5.5 | MEDIUM | safetensors < 0.3.2 | Metadata integer overflow in safetensors parser |

## Usage

```bash
pip install -e .

# Scan a HuggingFace model (provenance + CVE correlation, no download needed)
python cli.py scan bert-base-uncased

# Scan with rich terminal output (default)
python cli.py scan gpt2 --format terminal

# Output as JSON for CI/CD pipelines
python cli.py scan microsoft/phi-2 --format json

# Output as Markdown for GitHub PR comments
python cli.py scan gpt2 --format markdown

# Fast provenance-only scan (no model download required)
python cli.py scan bert-base-uncased --skip-weight-analysis --skip-activation-scan

# Skip only activation scan (still loads weights)
python cli.py scan my-model --local-path /models/my-model --skip-activation-scan

# Correlate CVEs for a specific model/architecture
python cli.py correlate-cves bert-base-uncased
python cli.py correlate-cves meta-llama/Llama-2-7b-hf --architecture llama
python cli.py correlate-cves gpt2 --output json

# List known threat hashes
python cli.py list-threats
```

### CLI Flags Reference

```
scan [MODEL_ID]
  --format / -f     {terminal,json,markdown}  Output format (default: terminal)
  --local-path / -l PATH                       Local model directory for weight analysis
  --skip-weight-analysis                        Skip weight loading; provenance-only mode
  --skip-activation-scan                        Skip activation scan (faster)
  --hf-token TOKEN                              HuggingFace API token (or HF_TOKEN env var)
  --verbose / -v                               Enable debug logging

correlate-cves [MODEL_ID]
  --architecture / -a ARCH    Architecture name (e.g. bert, llama, gpt2)
  --output / -o {terminal,json}

list-threats
  --db-path PATH    Custom threat database path
  --output / -o {terminal,json}
```

### Python API

```python
from modelsafe.scanner import ModelScanner
from modelsafe.report import ScanReport, format_terminal, format_json, format_markdown
from modelsafe.cve_correlator import CVECorrelator
from modelsafe.weight_analysis import WeightAnalyzer

# Full scan
scanner = ModelScanner()
result = scanner.scan("gpt2", local_path="/models/gpt2")

# CVE correlation
correlator = CVECorrelator()
cves = correlator.correlate("gpt2", architecture="gpt2")

# Rich report
report = ScanReport(result=result, cve_correlations=[c.to_dict() for c in cves])
format_terminal(result, report)       # prints to terminal
json_str = format_json(result, report)
md_str = format_markdown(result, report)

# Weight analysis (advanced)
import numpy as np
analyzer = WeightAnalyzer()
weights = {"layer.weight": np.random.randn(64, 64)}

# New: gradient sign anomaly detection
sign_score = analyzer.detect_gradient_sign_anomalies(weights)

# New: LayerNorm backdoor detection
ln_result = analyzer.analyze_layer_norm_statistics(weights)

# New: aggregated summary
from modelsafe.weight_analysis import WeightSummary
summary: WeightSummary = analyzer.build_summary(weights)
print(summary.overall_risk_score)
print(summary.suspicious_layers)
print(summary.key_findings)
```

### CI/CD Integration

```yaml
# .github/workflows/model-security.yml
- name: Scan model before deployment
  run: |
    pip install modelsafe
    modelsafe scan ${{ env.MODEL_ID }} --format json \
      --skip-activation-scan > scan_result.json
    python -c "
    import json, sys
    r = json.load(open('scan_result.json'))
    print(f'Risk: {r[\"risk_score\"]:.3f} ({r[\"risk_level\"]})')
    if not r['safe']:
        print('Model security check FAILED')
        sys.exit(1)
    "

- name: Post markdown report to PR
  run: |
    modelsafe scan ${{ env.MODEL_ID }} --format markdown \
      --skip-activation-scan > scan_report.md
    gh pr comment ${{ github.event.pull_request.number }} \
      --body-file scan_report.md
```

## Terminal Output Example

```
+--------------------------------------------------------------+
|  modelsafe Scan Report                                       |
|  CLEAN  Risk Score: 0.082  |  Model: bert-base-uncased      |
+--------------------------------------------------------------+

Check Scores
  Check                Score / Status   Risk Level   Verdict
  Provenance           Verified         CLEAN        PASS
  Weight Analysis      0.082            CLEAN        PASS
  Activation Scan      0.000            CLEAN        PASS
  Composite Risk       0.082            CLEAN        PASS

CVE Correlations
  CVE ID           Severity   CVSS   Description
  CVE-2025-32434   CRITICAL   9.3    torch.load() pickle RCE...
  CVE-2023-1999    HIGH       8.1    Transformers pickle deserialization...
  CVE-2024-3568    HIGH       7.5    HuggingFace Hub supply chain...
  ... (7 more)

Duration: 0.42s  |  Timestamp: 2026-03-26T10:00:00Z
```

## Benchmark

Run the known-models benchmark to validate on publicly trusted models:

```bash
# Online mode (hits HF API)
python experiments/scan_known_models.py

# Offline mode (synthetic results, no network)
python experiments/scan_known_models.py --offline

# Save results
python experiments/scan_known_models.py --output experiments/results/scan.json
```

Models benchmarked: `distilbert-base-uncased`, `bert-base-uncased`, `gpt2`.
Results saved to `experiments/results/known_models_scan.json`.

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Provenance check may flag legitimate models | Unusual metadata or low download count can trigger flags | Weight + activation confirmation required before acting on provenance alert |
| Activation scan requires weight loading | Cannot scan models exceeding available RAM | `--skip-activation-scan` flag; weight analysis is primary |
| K-S test sensitivity varies by layer size | Very small layers may be noisy | Minimum size threshold applied |
| CVE database is static | New CVEs not automatically included | Extend via `CVECorrelator(extra_cves=[...])` |
| Gradient sign check is heuristic | Low false-positive rate but not definitive | Treat as supporting evidence only |

*False positive rates are not yet empirically validated against a labelled dataset. Quantitative FPR/TPR evaluation is planned.*

## Project Structure

```
modelsafe/
+-- modelsafe/
|   +-- scanner.py          # Main orchestrator: ModelScanner, ScanResult
|   +-- weight_analysis.py  # WeightAnalyzer: SVD, K-S, sign anomaly, LayerNorm
|   +-- activation_scan.py  # ActivationScanner: forward-pass kurtosis probing
|   +-- provenance.py       # ProvenanceChecker: HF API metadata audit
|   +-- threat_db.py        # ThreatDatabase: SHA-256 hash lookup
|   +-- cve_correlator.py   # CVECorrelator: ML framework CVE correlation
|   +-- report.py           # format_terminal/json/markdown, ScanReport
+-- experiments/
|   +-- scan_known_models.py            # Benchmark script
|   +-- results/known_models_scan.json  # Benchmark output
+-- data/
|   +-- known_threats.json  # Threat hash database
+-- tests/
|   +-- test_scanner.py
|   +-- test_weight_analysis.py
|   +-- test_cve_correlator.py
|   +-- test_provenance.py
+-- cli.py                  # Click CLI entry point
+-- README.md
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=modelsafe

# Quick check
pytest tests/ -q
```

## Citations

```bibtex
@article{tran2018spectral,
  title={Spectral Signatures in Backdoor Attacks},
  author={Tran, Brandon and Li, Jerry and Madry, Aleksander},
  booktitle={NeurIPS},
  year={2018}
}

@article{hayase2021spectre,
  title={SPECTRE: Defending Against Backdoor Attacks via Robust Covariance Estimation},
  author={Hayase, Jonathan and others},
  journal={arXiv:2104.11315},
  year={2021}
}

@article{yang2024models,
  title={Models Are Codes: Towards Measuring Malicious Code Poisoning Attacks
         on Pre-trained Model Hubs},
  author={Yang, Jian and others},
  journal={arXiv:2409.09368},
  year={2024}
}

@misc{owasp2023aisec,
  title={OWASP AI Security and Privacy Guide},
  author={OWASP Foundation},
  year={2023},
  note={ML05:2023 Supply Chain Vulnerabilities}
}

@article{nguyen2021wanet,
  title={WaNet -- Imperceptible Warping-based Backdoor Attack},
  author={Nguyen, Anh and Tran, Anh},
  journal={arXiv:2102.10369},
  year={2021}
}
```

## Research Context

Part of the [ActivGuard](https://github.com/Tbhuvan/activguard) research programme. ModelSafe addresses a supply chain concern specific to AI security tools: what if the probe model itself is backdoored?

## License

Apache License 2.0
