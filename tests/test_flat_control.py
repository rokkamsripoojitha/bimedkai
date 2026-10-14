"""Tests verifying the Flat Chain-of-Thought Control baseline."""

import pytest
import numpy as np
from src.shared.llm import BioMistralLLM
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.methods.flat_control import FlatControlMethod


class TestFlatControlMethod:

    @pytest.fixture(scope="class")
    def control_infra(self):
        llm = BioMistralLLM(backend="auto")
        embedder = BiomedicalEmbedder(device="cpu")
        vstore = BiomedicalVectorStore(dimension=768)

        dummy_embs = np.zeros((2, 768), dtype=np.float32)
        dummy_embs[:, 0] = 1.0
        docs = [
            {"id": "doc1", "text": "Urinary obstruction causes hydronephrosis and progressive atrophy."},
            {"id": "doc2", "text": "Prostatic hyperplasia leads to urinary tract obstruction."},
        ]
        vstore.build_index(dummy_embs, docs)
        return llm, embedder, vstore

    @pytest.fixture
    def sample_question(self):
        return {
            "id": "medmcqa_test_c1",
            "dataset": "medmcqa",
            "question": "Chronic urethral obstruction due to BPH leads to what kidney parenchyma change?",
            "options": {"A": "Hyperplasia", "B": "Hypertrophy", "C": "Atrophy", "D": "Dysplasia"},
            "reference_answer": "C",
        }

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_flat_control_iterations(self, control_infra, sample_question, depth):
        llm, embedder, vstore = control_infra
        control = FlatControlMethod(llm, embedder, vstore)

        res = control.run(sample_question, depth=depth, seed=42)

        assert res["method"] == "flat_control"
        assert res["reasoning_iterations"] == depth
        assert len(res["reasoning_trace"]) == depth
        assert res["retrieval_calls"] == depth
        assert res["predicted_answer"] is not None
