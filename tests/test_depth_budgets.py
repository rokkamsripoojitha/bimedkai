"""Tests verifying strict depth budgets across all four reasoning methods."""

import pytest
import numpy as np
from src.shared.llm import BioMistralLLM
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph
from src.evaluation.runner import ExperimentRunner


class TestDepthBudgets:

    @pytest.fixture(scope="class")
    def runner(self):
        llm = BioMistralLLM(backend="auto")
        embedder = BiomedicalEmbedder(device="cpu")
        vstore = BiomedicalVectorStore(dimension=768)

        dummy_embs = np.zeros((1, 768), dtype=np.float32)
        dummy_embs[0, 0] = 1.0
        vstore.build_index(dummy_embs, [{"id": "d1", "text": "Medical reference text."}])

        kg = BiomedicalKnowledgeGraph()
        kg.add_relation("condition", "leads_to", "outcome")
        kg.add_hyperedge("he1", ["condition", "outcome", "marker"], "rel", "text")

        return ExperimentRunner(llm, embedder, vstore, kg)

    @pytest.fixture
    def question(self):
        return {
            "id": "q_budget_1",
            "dataset": "medmcqa",
            "question": "Does condition lead to outcome?",
            "options": {"A": "Yes", "B": "No", "C": "Maybe", "D": "Neither"},
            "reference_answer": "A",
        }

    @pytest.mark.parametrize("method", ["flat_control", "biomedkai", "kragen", "hypergraph_rag"])
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_strict_depth_budget(self, runner, question, method, depth):
        res = runner.run_configuration(
            method_name=method,
            dataset_name="medmcqa",
            depth=depth,
            seed=42,
            questions=[question],
            strict_budget_check=True,
        )
        records = runner.checkpoint_manager.load_records(res["results_filepath"])
        assert len(records) >= 1
        rec = [r for r in records if r["question_id"] == question["id"]][-1]
        assert rec["reasoning_iterations"] == depth
        assert len(rec["reasoning_trace"]) == depth
