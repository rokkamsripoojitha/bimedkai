"""Master orchestration script running the full end-to-end research project pipeline in sequence."""

import os
import sys
import subprocess
import time


def run_stage(title: str, command: list):
    print("\n" + "=" * 80)
    print(f"STAGE: {title}")
    print(f"Command: {' '.join(command)}")
    print("=" * 80)
    t0 = time.perf_counter()
    ret = subprocess.run(command, text=True)
    elapsed = time.perf_counter() - t0
    if ret.returncode != 0:
        print(f"\n[FAILURE] Stage '{title}' failed with exit code {ret.returncode} after {elapsed:.2f}s.")
        sys.exit(ret.returncode)
    print(f"[SUCCESS] Stage '{title}' completed in {elapsed:.2f}s.")


def main():
    print("=" * 80)
    print("STARTING COMPLETE BIOMEDICAL KG REASONING RESEARCH PIPELINE")
    print("=" * 80)

    py = sys.executable

    # 1. Environment Audit
    run_stage("Environment & Hardware Setup", [py, "scripts/setup_environment.py"])

    # 2. Dataset Preparation
    run_stage("Download & Deterministic Split Preparation", [py, "scripts/download_datasets.py"])

    # 3. FAISS Index Construction
    run_stage("Vector Store & Embedding Indexing", [py, "scripts/build_index.py"])

    # 4. Knowledge Graph & Hypergraph Construction
    run_stage("Biomedical Knowledge Graph & Hypergraph Construction", [py, "scripts/build_graph.py"])

    # 5. Automated Tests & Quality Assurance
    run_stage("Automated Test Suite (pytest)", [py, "-m", "pytest", "tests/", "-v"])

    # 6. Pilot Evaluation
    run_stage("20-Question Pilot Sweep (All Methods x Depths)", [py, "scripts/run_pilot.py"])

    # 7. Experimental Sweep
    # Run balanced sweep across methods, depths, and seeds
    run_stage("Full Experimental Sweep Execution", [py, "scripts/run_experiments.py", "--sample-size", "50", "--seeds", "42", "123", "999"])

    # 8. Statistical Analysis & Figures
    run_stage("Statistical Analysis & Figure Plotting", [py, "scripts/analyze_results.py"])

    # 9. Research Report Generation
    run_stage("Comprehensive Research Report Generation", [py, "scripts/generate_report.py"])

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTAL STAGES COMPLETED SUCCESSFULLY!")
    print("Research deliverables generated in reports/, results/tables/, and results/figures/.")
    print("=" * 80)


if __name__ == "__main__":
    main()
