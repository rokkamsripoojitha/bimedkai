"""End-to-end integration test verifying the full pipeline lifecycle."""

import os
import pytest
import numpy as np
from src.shared.llm import BioMistralLLM
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph
from src.evaluation.runner import ExperimentRunner
from src.evaluation.validate_results import validate_results_directory
from src.evaluation.statistical_analysis import ResearchStatisticalAnalyzer


class TestEndToEndPipeline:

    def test_full_pipeline_flow(self, tmp_path):
        # 1. Initialize Infrastructure
        llm = BioMistralLLM(backend="auto")
        embedder = BiomedicalEmbedder(device="cpu")
        vstore = BiomedicalVectorStore(
            dimension=768,
            index_path=str(tmp_path / "faiss.bin"),
            metadata_path=str(tmp_path / "meta.json"),
        )
        dummy_vecs = np.zeros((2, 768), dtype=np.float32)
        dummy_vecs[:, 0] = 1.0
        docs = [
            {"id": "doc_e2e_1", "text": "Chronic urethral obstruction leads to renal atrophy."},
            {"id": "doc_e2e_2", "text": "Benign prostatic hyperplasia causes hydronephrosis."},
        ]
        vstore.build_index(dummy_vecs, docs)

        kg = BiomedicalKnowledgeGraph(kg_path=str(tmp_path / "kg.json"))
        kg.add_relation("bph", "causes", "atrophy", "Obstruction causes atrophy.")
        kg.add_hyperedge("he1", ["bph", "atrophy", "obstruction"], "syndrome", "Evidence.")

        runner = ExperimentRunner(llm, embedder, vstore, kg)

        # 2. Execute a run configuration
        test_questions = [
            {
                "id": "e2e_q1",
                "dataset": "medmcqa",
                "question": "Chronic obstruction leads to which renal change?",
                "options": {"A": "Hyperplasia", "B": "Atrophy", "C": "Necrosis", "D": "Neoplasia"},
                "reference_answer": "B",
            }
        ]

        res = runner.run_configuration(
            method_name="biomedkai",
            dataset_name="medmcqa",
            depth=1,
            seed=42,
            questions=test_questions,
            strict_budget_check=True,
        )
        assert res["total_questions"] >= 1

        # 3. Validation Audit
        audit = validate_results_directory(raw_dir="results/raw")
        assert audit["depth_budget_violations"] == 0

        # 4. Statistical Analysis
        analyzer = ResearchStatisticalAnalyzer(
            raw_results_dir="results/raw",
            tables_dir=str(tmp_path / "tables"),
            figures_dir=str(tmp_path / "figures"),
        )
        stats_out = analyzer.run_full_analysis()
        assert stats_out["num_records"] >= 1
