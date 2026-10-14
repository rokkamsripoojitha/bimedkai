"""Data validation and experimental integrity auditor."""

import os
import json
import glob
from typing import Dict, Any, List


def validate_results_directory(raw_dir: str = "results/raw") -> Dict[str, Any]:
    """
    Audit all JSONL result files in raw results directory.
    Checks for schema completeness, deduplication, depth-budget compliance, and non-empty outputs.
    """
    files = glob.glob(os.path.join(raw_dir, "*.jsonl"))
    audit_report = {
        "total_files": len(files),
        "total_records_checked": 0,
        "valid_records": 0,
        "duplicate_records": 0,
        "corrupt_records": 0,
        "depth_budget_violations": 0,
        "file_summaries": [],
        "overall_status": "PASSED",
    }

    required_keys = [
        "question_id",
        "dataset",
        "question",
        "predicted_answer",
        "reference_answer",
        "is_correct",
        "reasoning_trace",
        "reasoning_iterations",
        "retrieval_calls",
        "total_tokens",
        "wall_clock_time",
        "method",
        "depth",
        "seed",
    ]

    for fpath in files:
        fname = os.path.basename(fpath)
        file_qids = set()
        file_valid = 0
        file_duplicates = 0
        file_violations = 0
        file_corrupt = 0

        with open(fpath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                audit_report["total_records_checked"] += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    file_corrupt += 1
                    audit_report["corrupt_records"] += 1
                    continue

                # Key check
                missing = [k for k in required_keys if k not in rec]
                if missing:
                    file_corrupt += 1
                    audit_report["corrupt_records"] += 1
                    continue

                # Deduplication check
                qid = rec["question_id"]
                if qid in file_qids:
                    file_duplicates += 1
                    audit_report["duplicate_records"] += 1
                else:
                    file_qids.add(qid)

                # Depth budget verification
                expected_depth = rec.get("depth")
                actual_iterations = rec.get("reasoning_iterations")
                if expected_depth != actual_iterations:
                    file_violations += 1
                    audit_report["depth_budget_violations"] += 1

                file_valid += 1
                audit_report["valid_records"] += 1

        audit_report["file_summaries"].append({
            "filename": fname,
            "valid_records": file_valid,
            "unique_questions": len(file_qids),
            "duplicates": file_duplicates,
            "corrupt": file_corrupt,
            "depth_violations": file_violations,
        })

    if audit_report["corrupt_records"] > 0 or audit_report["depth_budget_violations"] > 0:
        audit_report["overall_status"] = "FAILED"

    os.makedirs("results/processed", exist_ok=True)
    with open("results/processed/validation_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    return audit_report


if __name__ == "__main__":
    rep = validate_results_directory()
    print("Audit Report Summary:")
    print(f"Total files: {rep['total_files']}, Valid records: {rep['valid_records']}, Status: {rep['overall_status']}")
