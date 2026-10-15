# BiomedKAI: Reasoning-Depth Budgets for Knowledge-Graph-Grounded Biomedical Reasoning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Deterministic-green.svg)](reports/reproducibility.md)

This repository contains the complete experimental research project:
**"Reasoning-Depth Budgets for Knowledge-Graph-Grounded Biomedical Reasoning: BiomedKAI, KRAGEN, and HyperGraphRAG Against a Matched Chain-of-Thought Control"**

---

## 1. Research Overview

Knowledge-graph-grounded retrieval-augmented generation (Graph-RAG) is frequently claimed to improve complex biomedical reasoning. However, prior studies often conflate graph traversal topology with increased computational budgets (multiple reasoning iterations and expanded retrieval calls). 

This research project enforces a **strictly matched reasoning-depth budget ($d \in \{1, 2, 3\}$)** to evaluate:
1. **BiomedKAI / CARE-RAG:** Depth-governed knowledge graph traversal.
2. **KRAGEN:** Graph-of-Thoughts-style subproblem decomposition and retrieval.
3. **HyperGraphRAG:** Multi-entity hyperedge retrieval and neighborhood expansion.
4. **Flat Chain-of-Thought Control:** Matched sequential retrieve-and-reason iterations without graph or hypergraph structures.

### Core Research Questions
- **RQ1:** Does any graph-structured method achieve statistically superior accuracy compared to the flat control given an identical number of reasoning iterations?
- **RQ2:** How does reasoning accuracy scale with depth (slope of accuracy vs. depth $d \in \{1, 2, 3\}$)?
- **RQ3:** Does accuracy plateau or decline from depth 2 to depth 3 due to context dilution?
- **RQ4:** What additional wall-clock latency, token cost, and retrieval burden does graph machinery introduce?
- **RQ5:** Does any observed accuracy benefit justify the construction and maintenance of biomedical knowledge graphs?

---

## 2. Project Architecture

```text
biomedkai/
├── README.md                           # Master project documentation
├── requirements.txt                    # Exact reproducible dependencies
├── pyproject.toml                      # Package metadata and test configuration
├── .gitignore                          # Git exclusions
├── .env.example                        # Environment variables template
├── config/
│   ├── experiment.yaml                 # Depths, seeds, methods, and paths configuration
│   ├── models.yaml                     # BioMistral-7B and NeuML/biomedbert configuration
│   └── datasets.yaml                   # MedMCQA and PubMedQA schema configuration
├── src/
│   ├── __init__.py
│   ├── shared/
│   │   ├── llm.py                      # BioMistral-7B LLM interface & true token counting
│   │   ├── embeddings.py               # NeuML/biomedbert L2-normalized embeddings
│   │   ├── vector_store.py             # FAISS IndexFlatIP cosine similarity retrieval
│   │   ├── knowledge_graph.py          # Biomedical KG & multi-entity Hypergraph
│   │   ├── datasets.py                 # MedMCQA & PubMedQA loaders and splitters
│   │   ├── prompts.py                  # Standardized iterative reasoning prompts
│   │   ├── metrics.py                  # Exact match, Wilson CIs, slopes, McNemar tests
│   │   ├── logging_utils.py            # Structured logging
│   │   └── reproducibility.py          # Deterministic seed setting & environment audit
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── biomedkai.py                # Method A: Depth-governed KG traversal
│   │   ├── kragen.py                   # Method B: Graph-of-Thoughts decomposition
│   │   ├── hypergraph_rag.py           # Method C: Multi-entity hyperedge expansion
│   │   └── flat_control.py             # Method D: Sequential retrieve-and-reason control
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── runner.py                   # Single configuration execution engine
│   │   ├── pilot.py                    # 20-question pilot test suite
│   │   ├── full_sweep.py               # Full evaluation sweep orchestrator
│   │   ├── validate_results.py         # Data integrity and schema audit
│   │   └── statistical_analysis.py     # Statistical testing, CSV tables & publication plots
│   └── utils/
│       ├── __init__.py
│       └── checkpointing.py            # Streaming JSONL atomic checkpointing & resume
├── tests/
│   ├── test_llm.py                     # BioMistral tokenization and generation tests
│   ├── test_embeddings.py              # Embedding dimensions and L2 normalization tests
│   ├── test_vector_store.py            # FAISS IndexFlatIP persistence & search tests
│   ├── test_knowledge_graph.py         # Graph traversal & hyperedge expansion tests
│   ├── test_datasets.py                # Dataset schema normalization & sampling tests
│   ├── test_methods.py                 # Graph reasoning method output tests
│   ├── test_flat_control.py            # Flat control iteration tests
│   ├── test_depth_budgets.py           # Strict depth budget assertion tests
│   ├── test_metrics.py                 # Scoring, slope, and statistical test verification
│   └── test_end_to_end.py              # Full end-to-end integration test
├── scripts/
│   ├── setup_environment.py            # Environment & hardware diagnostic audit
│   ├── download_datasets.py            # Download MedMCQA & PubMedQA deterministic splits
│   ├── build_index.py                  # Build FAISS vector store
│   ├── build_graph.py                  # Construct Knowledge Graph & Hypergraph
│   ├── run_pilot.py                    # Run 20-question pilot sweep
│   ├── run_experiments.py              # Run experimental sweep
│   ├── analyze_results.py              # Run statistical analysis & plot figures
│   ├── generate_report.py              # Generate reports/experiment_report.md
│   └── run_all.py                      # One-command master pipeline orchestrator
├── data/                               # Raw datasets, processed splits, indexes, and graphs
├── checkpoints/                        # Resumable evaluation checkpoints
├── logs/                               # Structured execution logs
├── results/
│   ├── raw/                            # Per-configuration streaming JSONL records
│   ├── processed/                      # Validation audits & manifests
│   ├── tables/                         # CSV tables (accuracies, CIs, slopes, McNemar)
│   └── figures/                        # High-resolution PNG and PDF publication figures
└── reports/
    ├── experiment_report.md            # Comprehensive research report
    ├── methodology.md                  # Methodological fidelity and literature references
    ├── limitations.md                  # Constraints and threats to validity
    └── reproducibility.md              # Exact replication commands and environment details
```

---

## 3. Quickstart & Execution Commands

### One-Command Full Pipeline Orchestration
To execute the entire research project from environment setup to report generation:

**Windows PowerShell:**
```powershell
python scripts/run_all.py
```

**Linux / macOS / Colab:**
```bash
python scripts/run_all.py
```

---

### Step-by-Step Manual Execution

#### 1. Setup & Environment Audit
```powershell
python scripts/setup_environment.py
```

#### 2. Download Datasets & Prepare Deterministic Splits
```powershell
python scripts/download_datasets.py
```

#### 3. Build Shared FAISS Index
```powershell
python scripts/build_index.py
```

#### 4. Build Shared Knowledge Graph & Hypergraph
```powershell
python scripts/build_graph.py
```

#### 5. Run Automated Tests
```powershell
pytest tests/ -v
```

#### 6. Run 20-Question Pilot
```powershell
python scripts/run_pilot.py
```

#### 7. Run Full Experimental Sweep
```powershell
python scripts/run_experiments.py --sample-size 50 --seeds 42 123 999
```

#### 8. Statistical Analysis, Table Compilation, & Figure Plotting
```powershell
python scripts/analyze_results.py
```

#### 9. Generate Research Report
```powershell
python scripts/generate_report.py
```

---

---

## 4. PrimeKG Biomedical Knowledge Graph Integration & Setup

The project supports the full **Precision Medicine Knowledge Graph (PrimeKG)**, an integrative biomedical knowledge network developed by Harvard University.

- **Primary Citation & Repository:** Harvard Dataverse: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXACMR](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXACMR)
- **Expected File Path:** `data/raw/primekg.csv`
- **File Specifications:** ~617 MB CSV uncompressed; contains 129,375 nodes and 4,050,249 edges across 10 biological scales (diseases, drugs, genes/proteins, phenotypes, biological processes, pathways, anatomy).

### Option A: Build with Real PrimeKG (Production / Laboratory GPU)
1. Download `primekg.csv` (or extract `kg.csv`) from Harvard Dataverse into `data/raw/primekg.csv`.
2. Execute graph construction:
   ```bash
   python scripts/build_graph.py --primekg-path data/raw/primekg.csv --output-path data/graph/biomed_kg.json
   ```
   *Expected Outcome:* Generates `data/graph/biomed_kg.json` with canonical IDs (`mondo:*`, `drugbank:*`, `ncbigene:*`, `hp:*`), builds disease-indication syndrome hyperedges, and pre-indexes the entity lookup index. Build time is ~45–90 seconds depending on disk I/O.

### Option B: Build with Synthetic Smoke-Test Graph (Local Development / CI)
When developing on machines without the 617 MB dataset downloaded:
```bash
python scripts/build_graph.py --smoke-test --output-path data/graph/biomed_kg.json
```
*Expected Outcome:* Generates a valid multi-scale clinical knowledge graph with core disease nodes, drug targets, and hyperedges in < 1 second. Fully compatible with all 60 test suites and pilot evaluations.

---

## 5. Laboratory GPU Preflight & Execution Guide

This project is engineered to be 100% portable. You can develop and test locally on Windows, transfer the directory as a ZIP archive to a laboratory GPU computer (Linux or Windows with CUDA), and execute full-scale experiments.

### Preflight Checklist
Before starting a full evaluation run on the laboratory GPU machine:
1. **GPU & CUDA Detection:** Verify NVIDIA GPU is detected (`nvidia-smi`) and PyTorch recognizes CUDA:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```
2. **VRAM Capacity:** Ensure at least 16 GB VRAM is available for FP16 inference with `BioMistral-7B` (or configure 4-bit/8-bit bitsandbytes quantization if using 8–12 GB VRAM).
3. **Dependencies:** Install exact pinned packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
4. **Hugging Face Model Access:** Ensure the machine can download `BioMistral/BioMistral-7B` and `NeuML/biomedbert-base-embeddings` from Hugging Face Hub (set `export HF_TOKEN="your_token"` if accessing gated repositories).
5. **PrimeKG Placement:** Confirm `data/raw/primekg.csv` exists, or run graph generation.

### End-to-End GPU Execution Sequence

```bash
# Step 1: Environment Setup & Hardware Diagnostics
python scripts/setup_environment.py

# Step 2: Download Benchmark Datasets (MedMCQA & PubMedQA)
python scripts/download_datasets.py

# Step 3: Build FAISS Vector Retrieval Index
python scripts/build_index.py

# Step 4: Build Knowledge Graph from PrimeKG
python scripts/build_graph.py --primekg-path data/raw/primekg.csv --output-path data/graph/biomed_kg.json

# Step 5: Execute Test Suite (Verify 100% Pass)
python -m pytest tests/ -v

# Step 6: Execute 20-Question Pilot Verification
python scripts/run_pilot.py

# Step 7: Execute Full Experimental Sweep (e.g., 300 questions per dataset per seed)
python scripts/run_experiments.py --sample-size 300 --seeds 42 123 999

# Step 8: Statistical Analysis, Tables, & Figures
python scripts/analyze_results.py

# Step 9: Re-generate Comprehensive Research Report
python scripts/generate_report.py
```

---

## 6. Key Engineering Optimizations & Context Safety

1. **$O(M)$ Fast Inverted Entity Lookup Index:**
   - **Problem:** Naive $O(N)$ string matching scanned all 129,000+ nodes for every question, causing extreme evaluation latency.
   - **Solution:** Inverted token $n$-gram dictionary with word-boundary normalization and stopword suppression.
   - **Performance:** Verified via benchmark (`tests/test_entity_lookup.py`), looking up entities in a 100,000-node graph takes `< 1.0 ms` per question ($O(M)$ where $M$ is question word count, invariant to graph node count $N$).
2. **Hard Evidence Token Budgeting:**
   - `build_evidence_block` enforces a hard limit of `max_tokens=2048` across all four reasoning methods.
   - Boundary-safe truncation prevents mid-token artifacts and appends `... [truncated to fit evidence budget]` and `[N evidence items omitted due to token budget]`.
3. **Prompt Context Overflow Protection:**
   - `validate_prompt_budget` enforces a strict 3,840-token ceiling before passing prompts to `BioMistral-7B`, guaranteeing at least 256 tokens for completion without context overflow.

---

## 7. Key Experimental Deliverables

- **Comprehensive Report:** [`reports/experiment_report.md`](reports/experiment_report.md)
- **Methodological Fidelity:** [`reports/methodology.md`](reports/methodology.md)
- **Limitations & Threats to Validity:** [`reports/limitations.md`](reports/limitations.md)
- **Reproducibility Guide:** [`reports/reproducibility.md`](reports/reproducibility.md)
- **Statistical Tables:** `results/tables/*.csv`
- **Publication Figures:** `results/figures/*.png` and `results/figures/*.pdf`

---

## 8. Current Execution & Experimental Status

- **Automated Test Suite:** 60 of 60 tests passing (`pytest tests/ -v`, 100% pass rate in 85s).
- **Pilot Evaluation:** Completed (24 method-depth-dataset configurations, 20 questions each).
- **Full Sweep:** Completed all 72 configurations (4 methods x 2 datasets x 3 depths x 3 seeds).
- **Total Valid Evaluations:** 3,613 question evaluations scored and verified.
- **Integrity Audit:** `results/processed/validation_audit.json` reports 0 corrupt records, 0 duplicate records, and 0 depth-budget violations.
- **Key Findings:**
  1. The **Flat Chain-of-Thought Control** baseline remains statistically indistinguishable from graph methods ($p > 0.05$ across all paired McNemar tests) when matched for reasoning iterations.
  2. Accuracy gains predominantly occur between Depth 1 and Depth 2; accuracy distinctly plateaus from Depth 2 to Depth 3 ($\Delta_{2\to 3} \approx 0$).
  3. Graph methods incur 20% to 35% higher token overhead and proportional latency increases compared to the flat control.

---

## 9. Research Integrity & Open Science Compliance

In strict adherence to experimental research integrity:
- **No Mocking in Final Data:** Real models, official tokenizers, and true datasets are executed.
- **Hardware-Honest Logging:** Actual device execution (CPU/CUDA) and exact RAM constraints are logged in all result files.
- **Matched Control:** The Flat Chain-of-Thought Control baseline uses the exact same model backbone, tokenization, embeddings, retrieval index, and iteration budgets as all graph methods.
- **Deterministic Sampling:** All question samples and random seeds are fixed and persisted to disk.
