"""Tests for FAISS IndexFlatIP vector store, persistence, and cosine similarity search."""

import os
import pytest
import numpy as np
from src.shared.vector_store import BiomedicalVectorStore


class TestBiomedicalVectorStore:

    @pytest.fixture
    def sample_store(self, tmp_path):
        dim = 768
        vstore = BiomedicalVectorStore(
            dimension=dim,
            index_path=str(tmp_path / "test_index.bin"),
            metadata_path=str(tmp_path / "test_meta.json"),
        )
        # Create 3 synthetic L2 normalized vectors
        rng = np.random.default_rng(42)
        raw_vecs = rng.standard_normal((3, dim)).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        norm_vecs = raw_vecs / norms

        docs = [
            {"id": "doc1", "text": "Prostatic hyperplasia leads to urinary retention."},
            {"id": "doc2", "text": "Myocardial infarction elevates cardiac troponin."},
            {"id": "doc3", "text": "Streptococcus pneumoniae causes lobar consolidation."},
        ]

        vstore.build_index(norm_vecs, docs)
        return vstore, norm_vecs

    def test_search_and_score(self, sample_store):
        vstore, norm_vecs = sample_store
        # Search using doc1's embedding
        query_vec = norm_vecs[0]
        results = vstore.search(query_vec, top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        # Cosine similarity with itself should be ~1.0
        assert np.isclose(results[0]["score"], 1.0, atol=1e-4)

    def test_save_and_reload(self, sample_store):
        vstore, norm_vecs = sample_store
        vstore.save()

        # Reload into fresh store
        reloaded = BiomedicalVectorStore(
            dimension=768,
            index_path=vstore.index_path,
            metadata_path=vstore.metadata_path,
        )
        reloaded.load()

        res1 = vstore.search(norm_vecs[1], top_k=2)
        res2 = reloaded.search(norm_vecs[1], top_k=2)

        assert len(res1) == len(res2)
        assert res1[0]["id"] == res2[0]["id"]
        assert np.isclose(res1[0]["score"], res2[0]["score"], atol=1e-5)
