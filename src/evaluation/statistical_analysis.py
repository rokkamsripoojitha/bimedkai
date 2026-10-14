"""Comprehensive statistical analysis, hypothesis testing, table compilation, and publication plotting."""

import os
import glob
import json
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from src.shared.metrics import (
    calculate_accuracy,
    calculate_wilson_score_interval,
    calculate_linear_slope,
    calculate_depth_changes,
    mcnemar_test,
)


class ResearchStatisticalAnalyzer:
    """Computes statistical metrics, regression slopes, paired hypothesis tests, and publication figures."""

    def __init__(
        self,
        raw_results_dir: str = "results/raw",
        tables_dir: str = "results/tables",
        figures_dir: str = "results/figures",
    ):
        self.raw_results_dir = raw_results_dir
        self.tables_dir = tables_dir
        self.figures_dir = figures_dir
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        self.raw_df: pd.DataFrame = pd.DataFrame()

    def load_all_raw_data(self) -> pd.DataFrame:
        """Load and concatenate all question-level JSONL records from results/raw."""
        files = glob.glob(os.path.join(self.raw_results_dir, "*.jsonl"))
        all_records = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        try:
                            all_records.append(json.loads(line))
                        except Exception:
                            continue
        if not all_records:
            return pd.DataFrame()

        self.raw_df = pd.DataFrame(all_records)
        return self.raw_df

    def compute_accuracy_and_cis(self) -> pd.DataFrame:
        """Calculate exact-match accuracy and 95% Wilson confidence intervals grouped by dataset, method, depth, seed."""
        if self.raw_df.empty:
            self.load_all_raw_data()
        if self.raw_df.empty:
            return pd.DataFrame()

        grouped = self.raw_df.groupby(["dataset", "method", "depth", "seed"])
        rows = []
        for (ds, m, d, s), group in grouped:
            total = len(group)
            correct = group["is_correct"].sum()
            acc = correct / total if total > 0 else 0.0
            ci_low, ci_high = calculate_wilson_score_interval(correct, total)
            rows.append({
                "dataset": ds,
                "method": m,
                "depth": d,
                "seed": s,
                "total_questions": total,
                "correct_count": int(correct),
                "accuracy": acc,
                "accuracy_pct": acc * 100.0,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
            })

        df_acc = pd.DataFrame(rows)
        csv_path = os.path.join(self.tables_dir, "accuracy_and_confidence_intervals.csv")
        df_acc.to_csv(csv_path, index=False)
        return df_acc

    def compute_accuracy_slopes(self, df_acc: pd.DataFrame) -> pd.DataFrame:
        """Compute linear regression slopes and delta step changes across depths 1, 2, and 3."""
        if df_acc.empty:
            return pd.DataFrame()

        # Aggregate across seeds first
        agg = df_acc.groupby(["dataset", "method", "depth"])["accuracy"].mean().reset_index()
        rows = []
        for (ds, m), group in agg.groupby(["dataset", "method"]):
            sorted_g = group.sort_values("depth")
            depths = sorted_g["depth"].tolist()
            accs = sorted_g["accuracy"].tolist()
            acc_dict = dict(zip(depths, accs))

            slope = calculate_linear_slope(depths, accs)
            changes = calculate_depth_changes(acc_dict)

            rows.append({
                "dataset": ds,
                "method": m,
                "slope_per_depth": slope,
                "acc_depth_1": acc_dict.get(1, 0.0),
                "acc_depth_2": acc_dict.get(2, 0.0),
                "acc_depth_3": acc_dict.get(3, 0.0),
                "delta_1_to_2": changes["delta_1_to_2"],
                "delta_2_to_3": changes["delta_2_to_3"],
                "total_delta_1_to_3": changes["total_delta_1_to_3"],
            })

        df_slopes = pd.DataFrame(rows)
        csv_path = os.path.join(self.tables_dir, "accuracy_slopes.csv")
        df_slopes.to_csv(csv_path, index=False)
        return df_slopes

    def compute_paired_comparisons_vs_control(self) -> pd.DataFrame:
        """Perform paired McNemar tests for each graph method against the flat control on identical questions."""
        if self.raw_df.empty:
            self.load_all_raw_data()
        if self.raw_df.empty:
            return pd.DataFrame()

        graph_methods = ["biomedkai", "kragen", "hypergraph_rag"]
        control_method = "flat_control"

        rows = []
        # Group by dataset, depth, seed
        for (ds, d, s), group in self.raw_df.groupby(["dataset", "depth", "seed"]):
            control_subset = group[group["method"] == control_method].set_index("question_id")
            if control_subset.empty:
                continue

            for gm in graph_methods:
                gm_subset = group[group["method"] == gm].set_index("question_id")
                if gm_subset.empty:
                    continue

                # Intersection of identical questions
                common_ids = list(set(control_subset.index).intersection(set(gm_subset.index)))
                if not common_ids:
                    continue

                both_correct = 0
                gm_only = 0
                ctrl_only = 0
                both_incorrect = 0

                for qid in common_ids:
                    c_corr = control_subset.loc[qid, "is_correct"]
                    g_corr = gm_subset.loc[qid, "is_correct"]
                    if c_corr == 1 and g_corr == 1:
                        both_correct += 1
                    elif c_corr == 0 and g_corr == 1:
                        gm_only += 1
                    elif c_corr == 1 and g_corr == 0:
                        ctrl_only += 1
                    else:
                        both_incorrect += 1

                mcnemar_res = mcnemar_test((both_correct, gm_only, ctrl_only, both_incorrect))
                gm_acc = (both_correct + gm_only) / len(common_ids)
                ctrl_acc = (both_correct + ctrl_only) / len(common_ids)

                rows.append({
                    "dataset": ds,
                    "depth": d,
                    "seed": s,
                    "comparison": f"{gm} vs {control_method}",
                    "graph_method": gm,
                    "n_paired_questions": len(common_ids),
                    "graph_method_accuracy": gm_acc,
                    "control_accuracy": ctrl_acc,
                    "accuracy_difference": gm_acc - ctrl_acc,
                    "mcnemar_statistic": mcnemar_res["statistic"],
                    "p_value": mcnemar_res["p_value"],
                    "statistically_significant": mcnemar_res["p_value"] < 0.05,
                })

        df_paired = pd.DataFrame(rows)
        csv_path = os.path.join(self.tables_dir, "paired_comparisons_vs_control.csv")
        df_paired.to_csv(csv_path, index=False)
        return df_paired

    def compute_latency_and_tokens_summary(self) -> pd.DataFrame:
        """Compute mean, median, and std of latency, tokens, and retrieval calls."""
        if self.raw_df.empty:
            self.load_all_raw_data()
        if self.raw_df.empty:
            return pd.DataFrame()

        grouped = self.raw_df.groupby(["dataset", "method", "depth"])
        rows = []
        for (ds, m, d), group in grouped:
            rows.append({
                "dataset": ds,
                "method": m,
                "depth": d,
                "mean_latency_sec": group["wall_clock_time"].mean(),
                "median_latency_sec": group["wall_clock_time"].median(),
                "std_latency_sec": group["wall_clock_time"].std(),
                "mean_total_tokens": group["total_tokens"].mean(),
                "median_total_tokens": group["total_tokens"].median(),
                "mean_prompt_tokens": group["prompt_tokens"].mean(),
                "mean_completion_tokens": group["completion_tokens"].mean(),
                "mean_retrievals": group["retrieval_calls"].mean(),
                "error_rate": group["error_status"].mean(),
            })

        df_stats = pd.DataFrame(rows)
        csv_path = os.path.join(self.tables_dir, "latency_and_tokens_summary.csv")
        df_stats.to_csv(csv_path, index=False)
        return df_stats

    def generate_all_figures(self, df_acc: pd.DataFrame, df_stats: pd.DataFrame) -> None:
        """Generate all 6 required publication figures as PNG and PDF."""
        if df_acc.empty or df_stats.empty:
            return

        methods = ["flat_control", "biomedkai", "kragen", "hypergraph_rag"]
        colors = {
            "flat_control": "#2b5c8f",
            "biomedkai": "#2ca02c",
            "kragen": "#ff7f0e",
            "hypergraph_rag": "#d62728",
        }
        markers = {
            "flat_control": "o",
            "biomedkai": "s",
            "kragen": "^",
            "hypergraph_rag": "D",
        }

        # 1. Accuracy vs Depth (Separately for each dataset)
        datasets = df_acc["dataset"].unique()
        for ds in datasets:
            plt.figure(figsize=(7, 5))
            ds_df = df_acc[df_acc["dataset"] == ds]
            mean_acc = ds_df.groupby(["method", "depth"])["accuracy"].mean().reset_index()

            for m in methods:
                sub = mean_acc[mean_acc["method"] == m].sort_values("depth")
                if not sub.empty:
                    plt.plot(
                        sub["depth"],
                        sub["accuracy"] * 100.0,
                        marker=markers.get(m, "o"),
                        color=colors.get(m, "#000000"),
                        label=m.replace("_", " ").title(),
                        linewidth=2.0,
                        markersize=7,
                    )

            plt.title(f"Accuracy vs Reasoning Depth ({ds.upper()})", fontsize=12, fontweight="bold")
            plt.xlabel("Reasoning Depth Budget", fontsize=11)
            plt.ylabel("Exact Match Accuracy (%)", fontsize=11)
            plt.xticks([1, 2, 3])
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(frameon=True)
            plt.tight_layout()

            plt.savefig(os.path.join(self.figures_dir, f"accuracy_vs_depth_{ds}.png"), dpi=300)
            plt.savefig(os.path.join(self.figures_dir, f"accuracy_vs_depth_{ds}.pdf"))
            plt.close()

        # 2. Accuracy Change (Delta 1->2 and 2->3)
        plt.figure(figsize=(9, 5))
        delta_rows = []
        for ds in datasets:
            ds_df = df_acc[df_acc["dataset"] == ds]
            mean_acc = ds_df.groupby(["method", "depth"])["accuracy"].mean().to_dict()
            for m in methods:
                a1 = mean_acc.get((m, 1), 0.0)
                a2 = mean_acc.get((m, 2), 0.0)
                a3 = mean_acc.get((m, 3), 0.0)
                delta_rows.append({"dataset": ds, "method": m, "step": "Depth 1 to 2", "delta_pct": (a2 - a1) * 100})
                delta_rows.append({"dataset": ds, "method": m, "step": "Depth 2 to 3", "delta_pct": (a3 - a2) * 100})

        df_deltas = pd.DataFrame(delta_rows)
        # Bar chart
        x = np.arange(len(methods))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, step_name in enumerate(["Depth 1 to 2", "Depth 2 to 3"]):
            step_vals = [
                df_deltas[(df_deltas["method"] == m) & (df_deltas["step"] == step_name)]["delta_pct"].mean()
                for m in methods
            ]
            ax.bar(x + (idx - 0.5) * width, step_vals, width, label=step_name)

        ax.set_ylabel("Accuracy Change (Percentage Points)", fontsize=11)
        ax.set_title("Marginal Accuracy Change Across Depths", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in methods])
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "accuracy_changes_by_depth_step.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "accuracy_changes_by_depth_step.pdf"))
        plt.close()

        # 3. Wall-Clock Latency vs Depth
        plt.figure(figsize=(7, 5))
        mean_lat = df_stats.groupby(["method", "depth"])["mean_latency_sec"].mean().reset_index()
        for m in methods:
            sub = mean_lat[mean_lat["method"] == m].sort_values("depth")
            if not sub.empty:
                plt.plot(
                    sub["depth"],
                    sub["mean_latency_sec"],
                    marker=markers.get(m, "o"),
                    color=colors.get(m, "#000000"),
                    label=m.replace("_", " ").title(),
                    linewidth=2.0,
                    markersize=7,
                )
        plt.title("Wall-Clock Latency vs Reasoning Depth", fontsize=12, fontweight="bold")
        plt.xlabel("Reasoning Depth Budget", fontsize=11)
        plt.ylabel("Mean Latency per Question (Seconds)", fontsize=11)
        plt.xticks([1, 2, 3])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "latency_vs_depth.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "latency_vs_depth.pdf"))
        plt.close()

        # 4. Token Usage vs Depth
        plt.figure(figsize=(7, 5))
        mean_tok = df_stats.groupby(["method", "depth"])["mean_total_tokens"].mean().reset_index()
        for m in methods:
            sub = mean_tok[mean_tok["method"] == m].sort_values("depth")
            if not sub.empty:
                plt.plot(
                    sub["depth"],
                    sub["mean_total_tokens"],
                    marker=markers.get(m, "o"),
                    color=colors.get(m, "#000000"),
                    label=m.replace("_", " ").title(),
                    linewidth=2.0,
                    markersize=7,
                )
        plt.title("Token Usage vs Reasoning Depth", fontsize=12, fontweight="bold")
        plt.xlabel("Reasoning Depth Budget", fontsize=11)
        plt.ylabel("Mean Total Tokens per Question", fontsize=11)
        plt.xticks([1, 2, 3])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "tokens_vs_depth.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "tokens_vs_depth.pdf"))
        plt.close()

        # 5. Accuracy-versus-Cost Comparison (Accuracy vs Token Usage)
        plt.figure(figsize=(8, 6))
        for m in methods:
            m_acc = df_acc[df_acc["method"] == m].groupby("depth")["accuracy"].mean().to_dict()
            m_tok = df_stats[df_stats["method"] == m].groupby("depth")["mean_total_tokens"].mean().to_dict()

            x_pts = [m_tok.get(d, 0) for d in [1, 2, 3]]
            y_pts = [m_acc.get(d, 0) * 100.0 for d in [1, 2, 3]]

            plt.plot(
                x_pts,
                y_pts,
                marker=markers.get(m, "o"),
                color=colors.get(m, "#000000"),
                label=m.replace("_", " ").title(),
                linewidth=2.0,
                markersize=8,
            )
            for d_idx, d in enumerate([1, 2, 3]):
                plt.annotate(
                    f"d={d}",
                    (x_pts[d_idx], y_pts[d_idx]),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=9,
                )

        plt.title("Accuracy vs Computational Cost Frontier", fontsize=12, fontweight="bold")
        plt.xlabel("Mean Total Tokens per Question", fontsize=11)
        plt.ylabel("Exact Match Accuracy (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "accuracy_vs_cost_frontier.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "accuracy_vs_cost_frontier.pdf"))
        plt.close()

        # 6. Error Rate by Method and Depth
        plt.figure(figsize=(7, 5))
        mean_err = df_stats.groupby(["method", "depth"])["error_rate"].mean().reset_index()
        for m in methods:
            sub = mean_err[mean_err["method"] == m].sort_values("depth")
            if not sub.empty:
                plt.plot(
                    sub["depth"],
                    sub["error_rate"] * 100.0,
                    marker=markers.get(m, "o"),
                    color=colors.get(m, "#000000"),
                    label=m.replace("_", " ").title(),
                    linewidth=2.0,
                    markersize=7,
                )
        plt.title("Error Rate vs Reasoning Depth", fontsize=12, fontweight="bold")
        plt.xlabel("Reasoning Depth Budget", fontsize=11)
        plt.ylabel("Error / Unextracted Rate (%)", fontsize=11)
        plt.xticks([1, 2, 3])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "error_rate_by_method_depth.png"), dpi=300)
        plt.savefig(os.path.join(self.figures_dir, "error_rate_by_method_depth.pdf"))
        plt.close()

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all calculations, generate all tables and figures."""
        df_acc = self.compute_accuracy_and_cis()
        df_slopes = self.compute_accuracy_slopes(df_acc)
        df_paired = self.compute_paired_comparisons_vs_control()
        df_stats = self.compute_latency_and_tokens_summary()
        self.generate_all_figures(df_acc, df_stats)

        return {
            "num_records": len(self.raw_df),
            "tables_generated": [
                "accuracy_and_confidence_intervals.csv",
                "accuracy_slopes.csv",
                "paired_comparisons_vs_control.csv",
                "latency_and_tokens_summary.csv",
            ],
            "figures_generated": [
                "accuracy_vs_depth_medmcqa.png",
                "accuracy_vs_depth_pubmedqa.png",
                "accuracy_changes_by_depth_step.png",
                "latency_vs_depth.png",
                "tokens_vs_depth.png",
                "accuracy_vs_cost_frontier.png",
                "error_rate_by_method_depth.png",
            ],
        }
