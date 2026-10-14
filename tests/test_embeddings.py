"""Tests for NeuML/biomedbert-base-embeddings embedding generation and L2 normalization."""

import pytest
import numpy as np
from src.shared.embeddings import BiomedicalEmbedder


class TestBiomedicalEmbedder:

    @pytest.fixture(scope="class")
    def embedder(self):
        return BiomedicalEmbedder(device="cpu", batch_size=4)

    def test_embedding_dimensions(self, embedder):
        assert embedder.dimension == 768
        texts = ["Chronic kidney disease", "Acute myocardial infarction"]
        embs = embedder.embed_texts(texts)
        assert embs.shape == (2, 768)
        assert embs.dtype == np.float32

    def test_l2_normalization(self, embedder):
        texts = ["Streptococcus pneumoniae causing pneumonia"]
        embs = embedder.embed_texts(texts)
        norm = np.linalg.norm(embs[0])
        assert np.isclose(norm, 1.0, atol=1e-4)

    def test_single_query_embedding(self, embedder):
        query = "What causes hydronephrosis?"
        q_emb = embedder.embed_query(query)
        assert q_emb.shape == (768,)
        norm = np.linalg.norm(q_emb)
        assert np.isclose(norm, 1.0, atol=1e-4)
