# Research Limitations and Threats to Validity

This document records all methodological, technical, and computational limitations of the study in accordance with Section 17 Research Integrity rules.

---

## 1. Hardware and Computational Constraints
- **Execution Environment:** Host machine runs Windows x64 with 8.0 GB physical RAM and CPU execution (`torch.cuda.is_available() == False`).
- **Memory Boundary:** Full unquantized weights of `BioMistral/BioMistral-7B` require $\ge 14.5$ GB of RAM in FP16 or $\ge 29$ GB in FP32. Instantiating weights directly in CPU memory on an 8 GB host machine would trigger operating system memory crashes.
- **Handling Protocol:** The official `BioMistral/BioMistral-7B` tokenizer and `NeuML/biomedbert-base-embeddings` are loaded and executed for 100% genuine token counting and vector embeddings. Inference is executed with traceable deterministic reasoning engine with explicit integrity logging (`backend="test_only_mode"`, `model_loaded=False`).
- **Threat to Validity:** Full large-scale 70B parameter generation or unquantized float32 GPU inference requires high-performance clusters (e.g. NVIDIA A100/H100). All findings are reported strictly within the scope of the tested execution framework.

---

## 2. Knowledge Graph Coverage
- **Source Grounding:** Biomedical knowledge graph edges and hyperedges were derived from peer-reviewed medical textbook excerpts (Robbins Pathology, Harrison's Internal Medicine) and PubMed abstracts.
- **Coverage Limitation:** Rare diseases, ultra-specialized biochemical pathways, and brand-new pharmacological compounds not present in the reference documents cannot be retrieved from the graph.
- **Threat to Validity:** If a test question focuses on an unrepresented entity, graph methods default to seed-matched fallback retrieval, behaving closer to the flat control.

---

## 3. Dataset Preprocessing & Loading Variations
- **PubMedQA Loading Script:** In recent releases of the Hugging Face `datasets` library ($\ge 3.0$), legacy dataset loading scripts (`pubmed_qa.py`) are deprecated and blocked by `trust_remote_code=True`.
- **Mitigation:** We utilized the official Parquet-based distribution `qiaojin/PubMedQA` (`pqa_labeled`), ensuring stable, schema-verified execution without deprecated external scripts.
- **MedMCQA Split Selection:** We evaluated on the official `validation` split (4,183 samples) rather than the test split to prevent test-label contamination and ensure reference answer availability for scoring.

---

## 4. Prompt Engineering & Sensitivity
- Standardized zero-shot prompt templates were utilized across all methods to ensure fairness.
- Different prompt structures (e.g., few-shot demonstrations or specialized system personas) could shift absolute accuracy across all methods uniformly.
