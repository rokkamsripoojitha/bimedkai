"""Result checkpointing, atomic file operations, and resumption management."""

import os
import json
from typing import Dict, Any, List, Set, Optional


class CheckpointManager:
    """Manages appending, validating, and resuming experimental evaluation records."""

    def __init__(self, raw_results_dir: str = "results/raw"):
        self.raw_results_dir = raw_results_dir
        os.makedirs(raw_results_dir, exist_ok=True)

    def get_run_filepath(self, method: str, dataset: str, depth: int, seed: int) -> str:
        """Return standardized file path for a configuration's JSONL results."""
        filename = f"{method}_{dataset}_depth{depth}_seed{seed}.jsonl"
        return os.path.join(self.raw_results_dir, filename)

    def get_completed_question_ids(self, filepath: str) -> Set[str]:
        """Read existing JSONL results file and return set of completed question IDs."""
        if not os.path.exists(filepath):
            return set()

        completed = set()
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    qid = record.get("question_id")
                    if qid:
                        completed.add(qid)
                except json.JSONDecodeError:
                    continue
        return completed

    def append_record(self, filepath: str, record: Dict[str, Any]) -> None:
        """Atomically append a single question evaluation record to JSONL file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def load_records(self, filepath: str) -> List[Dict[str, Any]]:
        """Load all valid records from a JSONL results file."""
        if not os.path.exists(filepath):
            return []

        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
