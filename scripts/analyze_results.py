"""Execute statistical analysis, hypothesis testing, validation audit, and figure generation."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.validate_results import validate_results_directory
from src.evaluation.statistical_analysis import ResearchStatisticalAnalyzer
from src.shared.logging_utils import setup_logger

logger = setup_logger("biomedkai.analyze")


def main():
    logger.info("Step 1: Running Data Integrity and Result Validation Audit...")
    audit_rep = validate_results_directory(raw_dir="results/raw")
    logger.info(
        f"Validation Audit Status: {audit_rep['overall_status']} "
        f"(Checked {audit_rep['total_records_checked']} records across {audit_rep['total_files']} files, "
        f"Budget violations: {audit_rep['depth_budget_violations']}, Duplicates: {audit_rep['duplicate_records']})"
    )

    logger.info("Step 2: Executing Statistical Analysis and Hypothesis Testing...")
    analyzer = ResearchStatisticalAnalyzer(
        raw_results_dir="results/raw",
        tables_dir="results/tables",
        figures_dir="results/figures",
    )
    analysis_res = analyzer.run_full_analysis()

    logger.info(
        f"Analysis complete! Processed {analysis_res['num_records']} question evaluations.\n"
        f"Generated {len(analysis_res['tables_generated'])} tables in results/tables/\n"
        f"Generated {len(analysis_res['figures_generated'])} figures in results/figures/"
    )


if __name__ == "__main__":
    main()
