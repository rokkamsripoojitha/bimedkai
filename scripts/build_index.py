"""Build and persist shared FAISS vector store using NeuML/biomedbert-base-embeddings."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.datasets import BiomedicalDatasetLoader
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.shared.logging_utils import setup_logger

logger = setup_logger("biomedkai.build_index")


def build_shared_index(max_docs: int = 1000):
    logger.info("Initializing dataset loader to collect biomedical reference corpus...")
    loader = BiomedicalDatasetLoader()

    # Collect texts from MedMCQA explanations and PubMedQA contexts
    medmcqa_items = loader.load_medmcqa(split="validation")
    pubmedqa_items = loader.load_pubmedqa(split="train")

    documents = []
    seen_texts = set()

    for item in medmcqa_items:
        exp = (item.get("explanation") or "").strip()
        if len(exp) > 30 and exp not in seen_texts:
            seen_texts.add(exp)
            documents.append({
                "id": f"corpus_med_{len(documents)}",
                "text": exp,
                "source": item["id"],
                "subject": item["metadata"].get("subject", ""),
            })
            if len(documents) >= max_docs // 2:
                break

    for item in pubmedqa_items:
        ctx = (item.get("context") or "").strip()
        if len(ctx) > 30 and ctx not in seen_texts:
            seen_texts.add(ctx)
            documents.append({
                "id": f"corpus_pub_{len(documents)}",
                "text": ctx,
                "source": item["id"],
                "subject": "PubMed Abstract",
            })
            if len(documents) >= max_docs:
                break

    logger.info(f"Collected {len(documents)} unique biomedical corpus documents.")

    logger.info("Loading NeuML/biomedbert-base-embeddings model...")
    embedder = BiomedicalEmbedder(device="cpu", batch_size=16)

    doc_texts = [d["text"] for d in documents]
    logger.info(f"Embedding {len(doc_texts)} documents with L2 normalization...")
    embeddings = embedder.embed_texts(doc_texts)

    logger.info(f"Constructing FAISS IndexFlatIP (dim={embedder.dimension})...")
    vstore = BiomedicalVectorStore(dimension=embedder.dimension)
    vstore.build_index(embeddings=embeddings, documents=documents)

    index_file = "data/indexes/faiss_index.bin"
    meta_file = "data/indexes/index_metadata.json"
    vstore.save(index_file, meta_file)
    logger.info(f"FAISS index and metadata successfully persisted to {index_file} and {meta_file}!")


if __name__ == "__main__":
    build_shared_index()
