# Reproducibility Guide and System Audit

This document details the exact replication protocol, deterministic seeds, directory layout, and verification commands for the study.

---

## 1. System Environment

- **Python Version:** 3.11.9
- **Operating System:** Microsoft Windows 11 / 10 x64
- **PyTorch:** 2.14.0
- **Transformers:** 5.15.1
- **FAISS:** 1.15.1
- **Datasets:** 5.1.0
- **NetworkX:** 3.6.1
- **SciPy:** 1.17.1
- **Pandas:** 3.0.6
- **NumPy:** 2.4.2

---

## 2. Experimental Seeds and Determinism

All sampling, partitioning, and model operations are governed by fixed seeds:
- Seed 1: `42`
- Seed 2: `123`
- Seed 3: `999`

Seeds are set across Python `random`, `numpy`, and `torch` via `src.shared.reproducibility.set_seed()`.

Sample questions for each seed are cached in `data/processed/`:
- `medmcqa_seed_42_size_300.json`
- `medmcqa_seed_123_size_300.json`
- `medmcqa_seed_999_size_300.json`
- `pubmedqa_seed_42_size_300.json`
- `pubmedqa_seed_123_size_300.json`
- `pubmedqa_seed_999_size_300.json`

---

## 3. End-to-End Replication Workflow

Execute the following commands in order from the repository root:

### Windows PowerShell

```powershell
# 1. Verify Environment
python scripts/setup_environment.py

# 2. Download Datasets & Prepare Deterministic Splits
python scripts/download_datasets.py

# 3. Build Shared FAISS Index
python scripts/build_index.py

# 4. Construct Shared Biomedical Knowledge Graph & Hypergraph
python scripts/build_graph.py

# 5. Execute Automated Unit & Regression Tests
pytest tests/ -v

# 6. Execute 20-Question Pilot Evaluation
python scripts/run_pilot.py

# 7. Execute Experimental Sweep
python scripts/run_experiments.py --sample-size 50 --seeds 42 123 999

# 8. Run Statistical Analysis, Table Compilation, & Figure Plotting
python scripts/analyze_results.py

# 9. Generate Research Report
python scripts/generate_report.py
```

### Linux / Bash / Google Colab

```bash
# Set PYTHONPATH
export PYTHONPATH=.

# Execute Full Pipeline
python scripts/setup_environment.py
python scripts/download_datasets.py
python scripts/build_index.py
python scripts/build_graph.py
pytest tests/ -v
python scripts/run_pilot.py
python scripts/run_experiments.py --sample-size 50 --seeds 42 123 999
python scripts/analyze_results.py
python scripts/generate_report.py
```

---

## 4. Result Artifact Verification

- **Raw Question Records:** `results/raw/*.jsonl`
- **Audit Report:** `results/processed/validation_audit.json`
- **Accuracy & Confidence Intervals:** `results/tables/accuracy_and_confidence_intervals.csv`
- **Regression Slopes:** `results/tables/accuracy_slopes.csv`
- **Paired McNemar Tests:** `results/tables/paired_comparisons_vs_control.csv`
- **Latency & Tokens:** `results/tables/latency_and_tokens_summary.csv`
- **Figures (PNG & PDF):** `results/figures/*.png`, `results/figures/*.pdf`
- **Comprehensive Report:** `reports/experiment_report.md`
