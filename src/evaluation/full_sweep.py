"""Full experimental sweep orchestrator across methods, datasets, depths, and random seeds."""

import os
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from src.evaluation.runner import ExperimentRunner
from src.shared.datasets import BiomedicalDatasetLoader
from src.shared.logging_utils import setup_logger

logger = setup_logger("biomedkai.sweep")


class FullSweepOrchestrator:
    """Manages the full experimental evaluation matrix: 4 methods x 2 datasets x 3 depths x 3 seeds."""

    def __init__(
        self,
        runner: ExperimentRunner,
        dataset_loader: BiomedicalDatasetLoader,
        methods: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        depths: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        sample_size_per_seed: int = 300,
    ):
        self.runner = runner
        self.dataset_loader = dataset_loader
        self.methods = methods or ["flat_control", "biomedkai", "kragen", "hypergraph_rag"]
        self.datasets = datasets or ["medmcqa", "pubmedqa"]
        self.depths = depths or [1, 2, 3]
        self.seeds = seeds or [42, 123, 999]
        self.sample_size_per_seed = sample_size_per_seed

    def run_sweep(
        self,
        output_summary_csv: str = "results/tables/full_sweep_summary.csv",
        manifest_path: str = "results/processed/sweep_manifest.json",
        max_configs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Execute full experimental sweep with incremental checkpointing and resumption.
        """
        total_planned_configs = len(self.methods) * len(self.datasets) * len(self.depths) * len(self.seeds)
        logger.info(
            f"Initializing Full Experimental Sweep: {total_planned_configs} configurations planned "
            f"({len(self.methods)} methods x {len(self.datasets)} datasets x {len(self.depths)} depths x {len(self.seeds)} seeds)."
        )

        manifest = {
            "title": "BiomedKAI Reasoning-Depth Full Sweep Manifest",
            "methods": self.methods,
            "datasets": self.datasets,
            "depths": self.depths,
            "seeds": self.seeds,
            "sample_size_per_seed": self.sample_size_per_seed,
            "total_planned_configurations": total_planned_configs,
            "configurations": [],
        }

        # Cache dataset samples per seed
        dataset_samples: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        for ds_name in self.datasets:
            dataset_samples[ds_name] = {}
            for s in self.seeds:
                logger.info(f"Loading/preparing sample for {ds_name} (Seed={s}, Size={self.sample_size_per_seed})...")
                dataset_samples[ds_name][s] = self.dataset_loader.get_sample(
                    dataset_name=ds_name,
                    sample_size=self.sample_size_per_seed,
                    seed=s,
                )

        summary_rows = []
        config_idx = 0

        for ds_name in self.datasets:
            for seed in self.seeds:
                sample_questions = dataset_samples[ds_name][seed]
                for method in self.methods:
                    for depth in self.depths:
                        config_idx += 1
                        if max_configs and config_idx > max_configs:
                            logger.info(f"Reached maximum configuration limit ({max_configs}). Halting sweep.")
                            break

                        config_id = f"{method}_{ds_name}_d{depth}_s{seed}"
                        logger.info(f"[{config_idx}/{total_planned_configs}] Running config: {config_id} (N={len(sample_questions)})")

                        t0 = time.perf_counter()
                        res = self.runner.run_configuration(
                            method_name=method,
                            dataset_name=ds_name,
                            depth=depth,
                            seed=seed,
                            questions=sample_questions,
                            strict_budget_check=True,
                        )
                        elapsed = time.perf_counter() - t0

                        # Load records to compute granular stats
                        records = self.runner.checkpoint_manager.load_records(res["results_filepath"])
                        n_valid = len(records)
                        mean_latency = sum(r["wall_clock_time"] for r in records) / n_valid if n_valid else 0.0
                        mean_tokens = sum(r["total_tokens"] for r in records) / n_valid if n_valid else 0.0
                        mean_retrieval = sum(r["retrieval_calls"] for r in records) / n_valid if n_valid else 0.0
                        errors = sum(1 for r in records if r.get("error_status"))

                        row = {
                            "config_id": config_id,
                            "dataset": ds_name,
                            "method": method,
                            "depth": depth,
                            "seed": seed,
                            "total_questions": n_valid,
                            "correct_count": res["correct_count"],
                            "accuracy": res["accuracy"],
                            "error_rate": errors / n_valid if n_valid else 0.0,
                            "mean_latency_sec": mean_latency,
                            "mean_total_tokens": mean_tokens,
                            "mean_retrievals": mean_retrieval,
                            "elapsed_seconds": elapsed,
                            "filepath": res["results_filepath"],
                        }
                        summary_rows.append(row)
                        manifest["configurations"].append(row)

                        # Incremental write of manifest and summary table
                        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
                        with open(manifest_path, "w", encoding="utf-8") as f:
                            json.dump(manifest, f, indent=2)

                        os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)
                        pd.DataFrame(summary_rows).to_csv(output_summary_csv, index=False)

        df_summary = pd.DataFrame(summary_rows)
        logger.info(f"Full sweep execution complete! Evaluated {len(summary_rows)} configurations.")
        return df_summary
