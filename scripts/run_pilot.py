"""Run 20-question pilot evaluation across all 4 methods and depths 1, 2, 3."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.llm import BioMistralLLM
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph
from src.shared.datasets import BiomedicalDatasetLoader
from src.evaluation.runner import ExperimentRunner
from src.evaluation.pilot import PilotEvaluator
from src.shared.logging_utils import setup_logger

logger = setup_logger("biomedkai.run_pilot")


def main():
    logger.info("Initializing models and infrastructure for Pilot Evaluation...")
    llm = BioMistralLLM(backend="auto")
    embedder = BiomedicalEmbedder(device="cpu")
    vstore = BiomedicalVectorStore()
    vstore.load("data/indexes/faiss_index.bin", "data/indexes/index_metadata.json")
    kg = BiomedicalKnowledgeGraph("data/graph/biomed_kg.json")
    dataset_loader = BiomedicalDatasetLoader()

    runner = ExperimentRunner(
        llm=llm,
        embedder=embedder,
        vector_store=vstore,
        knowledge_graph=kg,
    )

    evaluator = PilotEvaluator(runner=runner, dataset_loader=dataset_loader)
    df_pilot = evaluator.run_pilot(sample_size=20, seed=42)

    print("\n" + "=" * 70)
    print("PILOT EVALUATION SUMMARY RESULTS")
    print("=" * 70)
    print(df_pilot[["dataset", "method", "depth", "accuracy", "mean_tokens", "mean_latency_sec"]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
