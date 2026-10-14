"""Experiment configuration runner, depth-budget assertion enforcer, and streaming execution engine."""

import os
import time
from typing import Dict, Any, List, Optional
from src.shared.llm import BioMistralLLM
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph
from src.shared.reproducibility import set_seed
from src.utils.checkpointing import CheckpointManager
from src.methods.biomedkai import BiomedKAIMethod
from src.methods.kragen import KRAGENMethod
from src.methods.hypergraph_rag import HyperGraphRAGMethod
from src.methods.flat_control import FlatControlMethod


class ExperimentRunner:
    """Orchestrates configuration execution with strict depth budget assertions and checkpointing."""

    def __init__(
        self,
        llm: BioMistralLLM,
        embedder: BiomedicalEmbedder,
        vector_store: BiomedicalVectorStore,
        knowledge_graph: BiomedicalKnowledgeGraph,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

        # Instantiate methods
        self.methods = {
            "biomedkai": BiomedKAIMethod(llm, embedder, vector_store, knowledge_graph),
            "kragen": KRAGENMethod(llm, embedder, vector_store, knowledge_graph),
            "hypergraph_rag": HyperGraphRAGMethod(llm, embedder, vector_store, knowledge_graph),
            "flat_control": FlatControlMethod(llm, embedder, vector_store),
        }

    def run_configuration(
        self,
        method_name: str,
        dataset_name: str,
        depth: int,
        seed: int,
        questions: List[Dict[str, Any]],
        strict_budget_check: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a single experimental configuration (method, dataset, depth, seed) over questions.
        Supports resume by skipping already evaluated question IDs.
        """
        if method_name not in self.methods:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.methods.keys())}")

        set_seed(seed)
        method_obj = self.methods[method_name]
        filepath = self.checkpoint_manager.get_run_filepath(method_name, dataset_name, depth, seed)
        completed_ids = self.checkpoint_manager.get_completed_question_ids(filepath)

        t_config_start = time.perf_counter()
        completed_count = len(completed_ids)
        newly_evaluated = 0
        correct_count = 0
        total_tokens_sum = 0
        retrieval_calls_sum = 0

        # First tally existing records if resuming
        if completed_ids:
            existing_records = self.checkpoint_manager.load_records(filepath)
            for rec in existing_records:
                if rec.get("is_correct") == 1:
                    correct_count += 1
                total_tokens_sum += rec.get("total_tokens", 0)
                retrieval_calls_sum += rec.get("retrieval_calls", 0)

        for q in questions:
            qid = q["id"]
            if qid in completed_ids:
                continue

            try:
                res = method_obj.run(question_item=q, depth=depth, seed=seed)

                # Strict budget assertion check: exactly `depth` logical reasoning iterations
                if strict_budget_check:
                    actual_iterations = res.get("reasoning_iterations", 0)
                    if actual_iterations != depth:
                        raise AssertionError(
                            f"Depth budget violation! Requested depth={depth}, "
                            f"but method '{method_name}' completed {actual_iterations} iterations."
                        )

            except Exception as e:
                res = {
                    "question_id": qid,
                    "dataset": dataset_name,
                    "question": q["question"],
                    "predicted_answer": None,
                    "reference_answer": q["reference_answer"],
                    "is_correct": 0,
                    "supporting_evidence": [],
                    "reasoning_trace": [],
                    "retrieval_calls": 0,
                    "reasoning_iterations": 0,
                    "wall_clock_time": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "error_status": True,
                    "error_message": str(e),
                    "method": method_name,
                    "depth": depth,
                    "seed": seed,
                    "config_id": f"{method_name}_{dataset_name}_d{depth}_s{seed}",
                    "model_backend": self.llm.backend,
                    "model_loaded": self.llm.model_loaded,
                }

            self.checkpoint_manager.append_record(filepath, res)
            completed_ids.add(qid)
            newly_evaluated += 1

            if res.get("is_correct") == 1:
                correct_count += 1
            total_tokens_sum += res.get("total_tokens", 0)
            retrieval_calls_sum += res.get("retrieval_calls", 0)

        total_questions = len(completed_ids)
        accuracy = (correct_count / total_questions) if total_questions > 0 else 0.0
        elapsed_total = time.perf_counter() - t_config_start

        return {
            "config_id": f"{method_name}_{dataset_name}_d{depth}_s{seed}",
            "method": method_name,
            "dataset": dataset_name,
            "depth": depth,
            "seed": seed,
            "total_questions": total_questions,
            "newly_evaluated": newly_evaluated,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "total_tokens": total_tokens_sum,
            "total_retrieval_calls": retrieval_calls_sum,
            "wall_clock_seconds": elapsed_total,
            "results_filepath": filepath,
        }
