"""Pilot evaluation suite: 20-question verification across all 4 methods and depths 1, 2, 3."""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from src.evaluation.runner import ExperimentRunner
from src.shared.datasets import BiomedicalDatasetLoader
from src.shared.logging_utils import setup_logger

logger = setup_logger("biomedkai.pilot")


class PilotEvaluator:
    """Runs and validates the 20-question pilot across methods and depths before full sweep."""

    def __init__(self, runner: ExperimentRunner, dataset_loader: BiomedicalDatasetLoader):
        self.runner = runner
        self.dataset_loader = dataset_loader

    def run_pilot(
        self,
        sample_size: int = 20,
        seed: int = 42,
        output_csv: str = "results/tables/pilot_summary.csv",
    ) -> pd.DataFrame:
        """
        Execute 20-question pilot across all 4 methods and depths 1, 2, 3 on MedMCQA and PubMedQA.
        """
        logger.info(f"Starting Pilot Evaluation (N={sample_size}, Seed={seed})...")

        # Load shared pilot samples
        medmcqa_sample = self.dataset_loader.get_sample("medmcqa", sample_size=sample_size, seed=seed)
        pubmedqa_sample = self.dataset_loader.get_sample("pubmedqa", sample_size=sample_size, seed=seed)

        datasets_to_test = [
            ("medmcqa", medmcqa_sample),
            ("pubmedqa", pubmedqa_sample),
        ]

        methods = ["flat_control", "biomedkai", "kragen", "hypergraph_rag"]
        depths = [1, 2, 3]

        summary_rows = []

        for ds_name, sample_items in datasets_to_test:
            logger.info(f"--- Pilot Dataset: {ds_name} ({len(sample_items)} samples) ---")
            for method in methods:
                for depth in depths:
                    logger.info(f"Executing Pilot: Method={method}, Dataset={ds_name}, Depth={depth}")
                    res = self.runner.run_configuration(
                        method_name=method,
                        dataset_name=ds_name,
                        depth=depth,
                        seed=seed,
                        questions=sample_items,
                        strict_budget_check=True,
                    )

                    # Verify records
                    records = self.runner.checkpoint_manager.load_records(res["results_filepath"])
                    for rec in records:
                        assert rec["reasoning_iterations"] == depth, (
                            f"Pilot verification failed! Record has {rec['reasoning_iterations']} iterations, expected {depth}."
                        )
                        assert "total_tokens" in rec and rec["total_tokens"] > 0, "Pilot token counting failed!"

                    mean_latency = (
                        sum(r["wall_clock_time"] for r in records) / len(records) if records else 0.0
                    )
                    mean_tokens = (
                        sum(r["total_tokens"] for r in records) / len(records) if records else 0.0
                    )
                    mean_retrievals = (
                        sum(r["retrieval_calls"] for r in records) / len(records) if records else 0.0
                    )
                    error_count = sum(1 for r in records if r.get("error_status"))

                    summary_rows.append({
                        "dataset": ds_name,
                        "method": method,
                        "depth": depth,
                        "seed": seed,
                        "sample_size": len(records),
                        "correct_count": res["correct_count"],
                        "accuracy": res["accuracy"],
                        "error_rate": error_count / len(records) if records else 0.0,
                        "mean_latency_sec": mean_latency,
                        "mean_tokens": mean_tokens,
                        "mean_retrieval_calls": mean_retrievals,
                    })

        df_summary = pd.DataFrame(summary_rows)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_summary.to_csv(output_csv, index=False)
        logger.info(f"Pilot evaluation finished successfully! Summary saved to {output_csv}")

        return df_summary
