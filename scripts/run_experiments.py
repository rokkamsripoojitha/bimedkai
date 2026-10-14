"""Full experiment execution orchestrator with CLI flags, incremental checkpoints, and resume support."""

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared.llm import BioMistralLLM
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph
from src.shared.datasets import BiomedicalDatasetLoader
from src.evaluation.runner import ExperimentRunner
from src.evaluation.full_sweep import FullSweepOrchestrator
from src.shared.logging_utils import setup_logger

logger = setup_logger("biomedkai.experiments")


def main():
    parser = argparse.ArgumentParser(description="Run full experimental evaluation sweep.")
    parser.add_argument("--sample-size", type=int, default=300, help="Questions per dataset per seed (default: 300)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds (default: 42 123 999)")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3], help="Reasoning depths (default: 1 2 3)")
    parser.add_argument("--methods", type=str, nargs="+", default=["flat_control", "biomedkai", "kragen", "hypergraph_rag"], help="Methods to evaluate")
    parser.add_argument("--datasets", type=str, nargs="+", default=["medmcqa", "pubmedqa"], help="Datasets to evaluate")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configurations (optional)")
    args = parser.parse_args()

    logger.info(f"Initializing Experiment Infrastructure with sample_size={args.sample_size}, seeds={args.seeds}...")
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

    orchestrator = FullSweepOrchestrator(
        runner=runner,
        dataset_loader=dataset_loader,
        methods=args.methods,
        datasets=args.datasets,
        depths=args.depths,
        seeds=args.seeds,
        sample_size_per_seed=args.sample_size,
    )

    df_sweep = orchestrator.run_sweep(max_configs=args.max_configs)
    print("\n" + "=" * 70)
    print("EXPERIMENTAL SWEEP EXECUTION COMPLETED")
    print(f"Total Configurations Evaluated: {len(df_sweep)}")
    print("Summary saved to: results/tables/full_sweep_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
