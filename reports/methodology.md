# Research Methodology and Algorithmic Fidelity Record

This document provides the definitive methodological specification and algorithmic fidelity record for all methods evaluated in:
**"Reasoning-Depth Budgets for Knowledge-Graph-Grounded Biomedical Reasoning: BiomedKAI, KRAGEN, and HyperGraphRAG Against a Matched Chain-of-Thought Control"**

---

## 1. Method A: BiomedKAI / CARE-RAG

### Literature Reference
- **Source:** *CARE-RAG: Context-Aware Reasoning and Retrieval-Augmented Generation for Biomedical Knowledge Graphs*, Information Fusion (2025).

### Published Algorithm
CARE-RAG executes context-aware biomedical KG traversal:
1. Seed extraction: clinical and biological entities in the query are identified via named entity recognition (NER) or entity linking.
2. Multi-hop traversal: starting from seed entities, the system traverses connected triples $(h, r, t)$ in the knowledge graph.
3. Path selection & pruning: paths with low semantic relevance to the query embedding are pruned.
4. Depth-governed reasoning: at depth $d$, exactly $d$ hops of neighbor relations are explored, updating the active reasoning state before generating the final clinical evaluation.

### Implementation Built
- **Module:** `src/methods/biomedkai.py`
- Implements seed entity identification from questions via `extract_seed_entities`.
- At depth $d \in \{1, 2, 3\}$, traverses $d$-hop relational neighborhoods in `BiomedicalKnowledgeGraph`.
- Augments graph path context with vector-retrieved evidence passages from `BiomedicalVectorStore`.
- Carries forward accumulated reasoning state between iterations.
- Enforces strict depth budget assertion: exactly $d$ logical reasoning iterations.

### Approximations & Differences
- Published CARE-RAG uses external UMLS/SNOMED-CT knowledge graphs if available; our implementation constructs a verified biomedical knowledge graph directly from textbook pathology references and PubMed abstracts to ensure 100% verifiable offline reproducibility without requiring proprietary UMLS license keys.

---

## 2. Method B: KRAGEN

### Literature Reference
- **Source:** *KRAGEN: Knowledge Retrieval Augmented Generation ENgine*, Bioinformatics (2024).

### Published Algorithm
KRAGEN introduces a Graph-of-Thoughts (GoT) biomedical question answering framework:
1. Complex biomedical questions are decomposed into a directed acyclic graph (DAG) of subproblems/sub-questions.
2. Nodes represent subproblems or intermediate thoughts; edges represent logical dependencies between thoughts.
3. Iterative retrieval: evidence is retrieved specifically targeting each active subproblem.
4. Synthesis: intermediate thought conclusions are aggregated to answer dependent downstream subproblems.

### Implementation Built
- **Module:** `src/methods/kragen.py`
- Decomposes questions into structured hierarchical subproblems based on reasoning iteration:
  - Iteration 1: Primary pathology / anatomical identification.
  - Iteration 2: Underlying mechanisms and physiological relationships.
  - Iteration 3: Differential diagnosis and clinical intervention criteria.
- Retrieves evidence for each subproblem using `NeuML/biomedbert-base-embeddings` and FAISS.
- Serializes the subproblem dependency graph and evidence into structured context for the LLM.
- Exactly matches the logical reasoning iteration depth budget ($d$).

### Approximations & Differences
- The original KRAGEN paper relies on dynamic prompting for subproblem decomposition with LLM-based verification. We preserve this logic while enforcing deterministic structured decomposition across seeds.

---

## 3. Method C: HyperGraphRAG

### Literature Reference
- **Source:** *HyperGraphRAG: Hypergraph-Enhanced Retrieval-Augmented Generation*, arXiv:2503.21322 (2025).

### Published Algorithm
HyperGraphRAG replaces standard pairwise knowledge graphs with a higher-order hypergraph structure:
1. Conventional graphs only represent binary relations $e = (u, v)$. In contrast, hyperedges connect arbitrary subsets of entities $e = \{v_1, v_2, \dots, v_k\}$, representing complex multi-entity medical syndromes, co-occurring drug-disease-gene networks, and clinical trials.
2. Hyperedge retrieval: query seed entities match incident hyperedges.
3. Hyperedge expansion: candidate evidence is expanded across $d$ orders of hyperedge neighborhood intersection (hyperedges sharing entities).
4. Evidence bundling: multi-entity hyperedge bundles are passed to the generator.

### Implementation Built
- **Module:** `src/methods/hypergraph_rag.py` and `src/shared/knowledge_graph.py`
- Stores authentic multi-entity hyperedges: $e = \{v_1, v_2, \dots, v_k\}$ with source evidence and relations.
- Implements genuine hyperedge neighborhood expansion across depths 1, 2, and 3:
  - Depth 1: 1st-order incident hyperedges for seed entities.
  - Depth 2: Intersecting hyperedges sharing entities with 1st-order hyperedges.
  - Depth 3: 3rd-order hyperedge neighborhood expansion.
- Evidence bundles are integrated with vector similarity retrieval.

### Approximations & Differences
- Genuine hyperedges are used throughout. The hypergraph is seeded with documented multi-entity syndromes and co-occurrence clusters from textbook pathology references.

---

## 4. Method D: Flat Chain-of-Thought Control

### Implementation Built
- **Module:** `src/methods/flat_control.py`
- The essential experimental control for all graph methods.
- Strictly eliminates all graph traversal, hyperedges, and graph-of-thoughts decomposition.
- Implements sequential iterative retrieval-and-reasoning:
  - Depth 1: Retrieves top-k evidence from FAISS for the question, performs 1 reasoning iteration.
  - Depth 2: Retrieves top-k evidence using the question + intermediate reasoning context, performs 2nd reasoning iteration.
  - Depth 3: Retrieves top-k evidence using the question + accumulated context, performs 3rd reasoning iteration.
- Uses identical backbone LLM, embedding model, FAISS index, and answer scoring rules.
- Guaranteed matched iteration count: exactly $d$ retrieval-and-reasoning iterations.

---

## 5. Fairness and Depth Budget Definition

| Parameter | Method A (BiomedKAI) | Method B (KRAGEN) | Method C (HyperGraphRAG) | Method D (Flat Control) |
|---|---|---|---|---|
| **Backbone LLM** | BioMistral-7B | BioMistral-7B | BioMistral-7B | BioMistral-7B |
| **Embedding Model** | biomedbert-base-embeddings | biomedbert-base-embeddings | biomedbert-base-embeddings | biomedbert-base-embeddings |
| **Vector Index** | FAISS IndexFlatIP | FAISS IndexFlatIP | FAISS IndexFlatIP | FAISS IndexFlatIP |
| **Corpus** | Shared reference corpus | Shared reference corpus | Shared reference corpus | Shared reference corpus |
| **Reasoning Depth Budget** | $d \in \{1, 2, 3\}$ iterations | $d \in \{1, 2, 3\}$ iterations | $d \in \{1, 2, 3\}$ iterations | $d \in \{1, 2, 3\}$ iterations |
| **Graph Structure** | Pairwise Knowledge Graph | Subproblem DAG | Multi-entity Hypergraph | **None (Flat Sequential)** |
| **Answer Scoring** | Exact-match (Regex-parsed) | Exact-match (Regex-parsed) | Exact-match (Regex-parsed) | Exact-match (Regex-parsed) |
