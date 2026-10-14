"""Dataset loaders, standardizers, and reproducible sample split generators for MedMCQA and PubMedQA."""

import os
import json
import random
from typing import List, Dict, Any, Optional
import datasets


class BiomedicalDatasetLoader:
    """Manages loading, normalization, and reproducible sampling for MedMCQA and PubMedQA."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

    def load_medmcqa(self, split: str = "validation") -> List[Dict[str, Any]]:
        """
        Load MedMCQA dataset and convert to standardized format.
        Options mapped to A, B, C, D with reference answer letter.
        """
        try:
            ds = datasets.load_dataset("openlifescienceai/medmcqa", split=split)
        except Exception:
            ds = datasets.load_dataset("med_mcqa", split=split)

        standardized = []
        letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

        for item in ds:
            # Validate cop is present
            cop = item.get("cop")
            if cop is None or cop not in letter_map:
                continue

            q_id = str(item.get("id", len(standardized)))
            q_text = item.get("question", "").strip()
            options = {
                "A": str(item.get("opa", "")).strip(),
                "B": str(item.get("opb", "")).strip(),
                "C": str(item.get("opc", "")).strip(),
                "D": str(item.get("opd", "")).strip(),
            }
            ref_ans = letter_map[cop]

            standardized.append({
                "id": f"medmcqa_{q_id}",
                "dataset": "medmcqa",
                "question": q_text,
                "options": options,
                "reference_answer": ref_ans,
                "context": "",
                "explanation": item.get("exp", ""),
                "metadata": {
                    "subject": item.get("subject_name", ""),
                    "topic": item.get("topic_name", ""),
                    "choice_type": item.get("choice_type", "single"),
                },
            })

        return standardized

    def load_pubmedqa(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        Load PubMedQA pqa_labeled dataset and convert to standardized format.
        Valid answers are 'yes', 'no', 'maybe'.
        """
        try:
            ds = datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
        except Exception:
            ds = datasets.load_dataset("pubmed_qa", "pqa_labeled", split=split)

        standardized = []
        for item in ds:
            decision = item.get("final_decision", "").strip().lower()
            if decision not in ["yes", "no", "maybe"]:
                continue

            q_id = str(item.get("pubid", len(standardized)))
            q_text = item.get("question", "").strip()

            # Format context from abstract sentences if available
            ctx_data = item.get("context", {})
            if isinstance(ctx_data, dict) and "contexts" in ctx_data:
                context_str = " ".join(ctx_data["contexts"])
            elif isinstance(ctx_data, list):
                context_str = " ".join(ctx_data)
            else:
                context_str = str(ctx_data)

            standardized.append({
                "id": f"pubmedqa_{q_id}",
                "dataset": "pubmedqa",
                "question": q_text,
                "options": None,
                "reference_answer": decision,
                "context": context_str,
                "explanation": item.get("long_answer", ""),
                "metadata": {
                    "pubid": q_id,
                },
            })

        return standardized

    def get_sample(
        self,
        dataset_name: str,
        sample_size: int = 300,
        seed: int = 42,
        save_to_disk: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get or create a reproducible deterministic sample of questions.
        Persists sample IDs and content to prevent data leakage and ensure identical sets across methods.
        """
        filename = f"{dataset_name}_seed_{seed}_size_{sample_size}.json"
        cached_path = os.path.join(self.processed_dir, filename)

        if os.path.exists(cached_path):
            with open(cached_path, "r", encoding="utf-8") as f:
                return json.load(f)

        if dataset_name.lower() == "medmcqa":
            all_items = self.load_medmcqa(split="validation")
        elif dataset_name.lower() == "pubmedqa":
            all_items = self.load_pubmedqa(split="train")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        rng = random.Random(seed)
        num_to_sample = min(sample_size, len(all_items))
        sample_indices = rng.sample(range(len(all_items)), num_to_sample)
        sampled = [all_items[i] for i in sample_indices]

        if save_to_disk:
            with open(cached_path, "w", encoding="utf-8") as f:
                json.dump(sampled, f, indent=2, ensure_ascii=False)

        return sampled
