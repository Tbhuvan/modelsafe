<div align="center">

# ModelSafe

**ML model supply chain scanner — detect backdoored and poisoned models without training data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

*Snyk for ML models.*

</div>

---

## Overview

ModelSafe scans machine learning models for backdoors, data poisoning, and supply chain tampering — without requiring access to the original training data. It uses three training-data-free detection signals:

1. **Spectral Weight Analysis** — Detects anomalous weight distributions indicative of trigger backdoors
2. **Activation Distribution Analysis** — Identifies unusual activation patterns that suggest data poisoning
3. **Provenance Verification** — Validates model metadata, checksums, and repository authenticity

## Threat Model

| Threat | Detection Signal | Method |
|--------|-----------------|--------|
| Trigger backdoors | Spectral outliers in weight matrices | Eigenvalue analysis of layer weights |
| Data poisoning | Activation distribution shift | Statistical testing on reference inputs |
| Model impersonation | Provenance mismatch | Hash verification + metadata audit |
| Adversarial embeddings | Hidden capacity detection | Rank analysis of weight matrices |

### What ModelSafe Does Not Detect

- Sophisticated backdoors with purely distributional triggers (current research frontier)
- Bias injected via clean-label poisoning without weight anomalies
- Models with privacy-violating training data but correct behaviour
- Jailbreak fine-tunes that don't deviate from expected weight statistics

## Architecture

```
modelsafe scan [model_id]
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│  ModelScanner (modelsafe/scanner.py)                           │
│                                                                │
│  Step 1: ThreatDatabase.check_model_id()                      │
│    → SHA-256 hash lookup against known malicious models        │
│                                                                │
│  Step 2: ProvenanceChecker.check()                             │
│    → HuggingFace Hub API: metadata, author, downloads         │
│    → Model card verification, author reputation, arch check   │
│                                                                │
│  Step 3: WeightAnalyzer.analyze()                              │
│    → SVD analysis: energy concentration in top-k SVs          │
│    → K-S test vs Gaussian baseline                            │
│    → Layer norm Z-score heuristic for trojan triggers         │
│                                                                │
│  Step 4: ActivationScanner.scan()                              │
│    → 100 synthetic text probes with PyTorch forward hooks     │
│    → Kurtosis + variance analysis per layer                   │
│                                                                │
│  Aggregate: risk_score = 0.3×prov + 0.4×weight + 0.3×activ   │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
  ScanResult: {safe, risk_score, findings, ...}
```

## Usage

```bash
pip install -e .

# Scan a HuggingFace model
python cli.py scan microsoft/codebert-base

# Scan a local model
python cli.py scan ./my-model/ --local

# Full report
python cli.py scan microsoft/codebert-base --report detailed
```

### Python API

```python
from modelsafe.scanner import ModelScanner

scanner = ModelScanner()
result = scanner.scan("gpt2", local_path="/models/gpt2")

print(result.safe)               # True / False
print(result.risk_score)         # 0.0 – 1.0
print(result.to_report())        # human-readable
print(result.to_json())          # machine-readable
```

### CI/CD Integration

```yaml
# .github/workflows/model-security.yml
- name: Scan model before deployment
  run: |
    pip install modelsafe
    modelsafe scan ${{ env.MODEL_ID }} --output json > scan_result.json
    if [ $(jq -r '.risk_score' scan_result.json | awk '{print ($1 > 0.7)}') -eq 1 ]; then
      echo "Model security check failed!"
      exit 1
    fi
```

## Output

```
ModelSafe Scan Report
━━━━━━━━━━━━━━━━━━━━
Model: microsoft/codebert-base
Status: ✓ CLEAN

  Spectral Analysis:    PASS  (no outlier eigenvalues)
  Activation Analysis:  PASS  (distribution within bounds)
  Provenance Check:     PASS  (checksums verified)

Risk Score: 0.12 / 1.00
```

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| False positive rate ~15% on provenance check | Legitimate models may be flagged | Weight + activation confirmation required for high risk |
| Activation scan requires weight loading | Cannot scan models exceeding available RAM | `--skip-activation-scan` flag; weight analysis is primary |
| K-S test sensitivity varies by layer size | Very small layers may be noisy | Minimum size threshold applied |

**False positive rate (estimated):** ~5% on weight analysis alone, based on analysis of 500 known-clean models from major organisations (Microsoft, Meta, Google, Mistral, EleutherAI).

## Project Structure

```
modelsafe/
├── modelsafe/       # Core: scanner, weight analysis, activation scan, provenance
├── data/            # Reference datasets for statistical testing
├── tests/           # Test suite
├── cli.py           # CLI entry point
└── README.md
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=modelsafe
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
```

## Research Context

Part of the [ActivGuard](https://github.com/Tbhuvan/activguard) research programme. ModelSafe addresses a supply chain concern specific to AI security tools: what if the probe model itself is backdoored?

## License

Apache License 2.0
