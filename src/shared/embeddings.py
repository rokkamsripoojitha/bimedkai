"""Biomedical text embeddings using NeuML/biomedbert-base-embeddings with L2 normalization."""

import os
import time
import hashlib
from typing import List, Union, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class BiomedicalEmbedder:
    """Embeds biomedical texts using verified NeuML/biomedbert-base-embeddings."""

    def __init__(
        self,
        model_id: str = "NeuML/biomedbert-base-embeddings",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 16,
        cache_dir: Optional[str] = "data/indexes",
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.dimension = self.model.config.hidden_size

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling over token embeddings taking attention mask into account."""
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate L2-normalized embeddings for input texts.
        Returns numpy array of shape (N, dimension).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])

                if self.normalize:
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

                all_embeddings.append(sentence_embeddings.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query string and return 1D numpy array."""
        embeddings = self.embed_texts([query])
        return embeddings[0]
