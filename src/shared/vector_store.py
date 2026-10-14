"""FAISS vector retrieval store using IndexFlatIP with L2 normalized embeddings."""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss


class BiomedicalVectorStore:
    """FAISS-based vector index supporting cosine similarity via IndexFlatIP."""

    def __init__(
        self,
        dimension: int = 768,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        self.dimension = dimension
        self.index_path = index_path or "data/indexes/faiss_index.bin"
        self.metadata_path = metadata_path or "data/indexes/index_metadata.json"
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Dict[str, Any]] = []
        self.retrieval_call_count: int = 0
        self.total_retrieval_time: float = 0.0

    def build_index(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """
        Build an IndexFlatIP from provided embeddings and store matching metadata.
        embeddings must be L2 normalized float32 array of shape (N, dimension).
        """
        if embeddings.shape[0] != len(documents):
            raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) does not match documents ({len(documents)})")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) does not match expected {self.dimension}")

        # Ensure float32 contiguous array
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # FAISS IndexFlatIP computes inner products. When vectors are L2-normalized, IP == cosine similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.documents = list(documents)

    def save(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> None:
        """Save FAISS binary index and document metadata JSON."""
        idx_p = index_path or self.index_path
        meta_p = metadata_path or self.metadata_path

        os.makedirs(os.path.dirname(idx_p), exist_ok=True)
        os.makedirs(os.path.dirname(meta_p), exist_ok=True)

        if self.index is None:
            raise RuntimeError("Cannot save uninitialized index.")

        faiss.write_index(self.index, idx_p)
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)

    def load(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> None:
        """Load FAISS binary index and document metadata from disk."""
        idx_p = index_path or self.index_path
        meta_p = metadata_path or self.metadata_path

        if not os.path.exists(idx_p) or not os.path.exists(meta_p):
            raise FileNotFoundError(f"Index or metadata file not found at {idx_p}, {meta_p}")

        self.index = faiss.read_index(idx_p)
        with open(meta_p, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform top-k cosine similarity search for a query embedding.
        Returns list of matching documents with scores.
        """
        if self.index is None:
            raise RuntimeError("Index is not loaded or built.")

        t0 = time.perf_counter()
        self.retrieval_call_count += 1

        # Format query embedding to 2D float32 array
        query_vec = np.ascontiguousarray(query_embedding.reshape(1, -1).astype(np.float32))

        scores, indices = self.index.search(query_vec, top_k)
        elapsed = time.perf_counter() - t0
        self.total_retrieval_time += elapsed

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.documents):
                continue
            doc_data = dict(self.documents[idx])
            doc_data["score"] = float(score)
            results.append(doc_data)

        return results

    def reset_stats(self) -> None:
        """Reset retrieval call counter and timer."""
        self.retrieval_call_count = 0
        self.total_retrieval_time = 0.0
