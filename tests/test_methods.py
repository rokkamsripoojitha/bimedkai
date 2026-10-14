"""Tests verifying public interface and standardized output across all four reasoning methods."""

import pytest
import numpy as np
from src.shared.llm import BioMistralLLM
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph
from src.methods.biomedkai import BiomedKAIMethod
from src.methods.kragen import KRAGENMethod
from src.methods.hypergraph_rag import HyperGraphRAGMethod
from src.methods.flat_control import FlatControlMethod


class TestReasoningMethods:

    @pytest.fixture(scope="class")
    def shared_infra(self):
        llm = BioMistralLLM(backend="auto")
        embedder = BiomedicalEmbedder(device="cpu")
        vstore = BiomedicalVectorStore(dimension=768)

        # Mock 2 embeddings
        dummy_embs = np.zeros((2, 768), dtype=np.float32)
        dummy_embs[:, 0] = 1.0  # normalized
        docs = [
            {"id": "doc1", "text": "Chronic urethral obstruction leads to renal atrophy."},
            {"id": "doc2", "text": "Benign prostatic hyperplasia causes hydronephrosis."},
        ]
        vstore.build_index(dummy_embs, docs)

        kg = BiomedicalKnowledgeGraph()
        kg.add_relation("bph", "causes", "atrophy", "Obstruction causes atrophy.")
        kg.add_hyperedge("he1", ["bph", "obstruction", "atrophy"], "pathology", "Evidence text.")

        return llm, embedder, vstore, kg

    @pytest.fixture
    def sample_question(self):
        return {
            "id": "test_q1",
            "dataset": "medmcqa",
            "question": "Chronic urethral obstruction leads to which change in kidney parenchyma?",
            "options": {"A": "Hyperplasia", "B": "Hypertrophy", "C": "Atrophy", "D": "Dysplasia"},
            "reference_answer": "C",
        }

    @pytest.mark.parametrize("method_cls", [BiomedKAIMethod, KRAGENMethod, HyperGraphRAGMethod])
    def test_graph_methods_standard_output(self, shared_infra, sample_question, method_cls):
        llm, embedder, vstore, kg = shared_infra
        method = method_cls(llm, embedder, vstore, kg)
        res = method.run(sample_question, depth=2, seed=42)

        required_keys = [
            "question_id", "dataset", "question", "predicted_answer",
            "reference_answer", "is_correct", "supporting_evidence",
            "reasoning_trace", "retrieval_calls", "reasoning_iterations",
            "wall_clock_time", "prompt_tokens", "completion_tokens",
            "total_tokens", "error_status", "method", "depth", "seed"
        ]
        for k in required_keys:
            assert k in res, f"Missing key {k} in {method.name} output"

        assert res["reasoning_iterations"] == 2
        assert len(res["reasoning_trace"]) == 2
        assert res["predicted_answer"] is not None
        assert res["is_correct"] in [0, 1]
