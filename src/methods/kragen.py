"""KRAGEN: Knowledge Retrieval Augmented Generation ENgine with Graph-of-Thoughts Subproblem Decomposition."""

import time
from typing import Dict, Any, List, Optional
from src.shared.llm import BioMistralLLM
from src.shared.embeddings import BiomedicalEmbedder
from src.shared.vector_store import BiomedicalVectorStore
from src.shared.knowledge_graph import BiomedicalKnowledgeGraph
from src.shared.prompts import (
    build_system_prompt,
    format_question_mcqa,
    format_question_pubmedqa,
    build_evidence_block,
    build_reasoning_prompt,
)
from src.shared.metrics import (
    extract_medmcqa_answer,
    extract_pubmedqa_answer,
    score_answer,
)


class KRAGENMethod:
    """KRAGEN method implementing graph-of-thoughts subproblem decomposition and retrieval."""

    def __init__(
        self,
        llm: BioMistralLLM,
        embedder: BiomedicalEmbedder,
        vector_store: BiomedicalVectorStore,
        knowledge_graph: BiomedicalKnowledgeGraph,
    ):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.name = "kragen"

    def _decompose_subproblems(self, question: str, iteration: int) -> List[str]:
        """
        Decompose biomedical problem into structured subproblems based on reasoning stage.
        Iteration 1: Primary pathology / anatomical identification.
        Iteration 2: Etiology, progression, and physiological mechanism.
        Iteration 3: Clinical distinction, definitive diagnosis, and intervention.
        """
        if iteration == 1:
            return [
                f"Identify primary biomedical conditions and structures: {question}",
            ]
        elif iteration == 2:
            return [
                f"Underlying pathological mechanisms and causal relationships for: {question}",
                f"Physiological response and complications associated with: {question}",
            ]
        else:  # iteration 3
            return [
                f"Differential diagnosis and discriminative criteria for: {question}",
                f"Clinical intervention and definitive outcome for: {question}",
            ]

    def run(
        self,
        question_item: Dict[str, Any],
        depth: int = 1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Execute KRAGEN graph-of-thoughts decomposition reasoning at depth 1, 2, or 3.
        """
        t0 = time.perf_counter()
        dataset_type = question_item.get("dataset", "medmcqa")
        q_id = question_item["id"]
        q_text = question_item["question"]
        ref_ans = question_item["reference_answer"]

        if dataset_type == "medmcqa":
            formatted_q = format_question_mcqa(q_text, question_item.get("options", {}))
        else:
            formatted_q = format_question_pubmedqa(q_text)

        system_prompt = build_system_prompt()

        subproblem_graph: List[Dict[str, Any]] = []
        accumulated_reasoning = ""
        reasoning_trace = []
        retrieved_evidence: List[Dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        retrieval_calls = 0

        # Perform exactly `depth` iterations
        for current_iteration in range(1, depth + 1):
            current_subproblems = self._decompose_subproblems(q_text, current_iteration)
            iteration_evidence = []

            for sp in current_subproblems:
                sp_emb = self.embedder.embed_query(sp)
                ev_items = self.vector_store.search(sp_emb, top_k=2)
                retrieval_calls += 1
                iteration_evidence.extend(ev_items)
                subproblem_graph.append({
                    "subproblem": sp,
                    "iteration": current_iteration,
                    "retrieved_count": len(ev_items),
                })

            for ev in iteration_evidence:
                if ev not in retrieved_evidence:
                    retrieved_evidence.append(ev)

            # Build subproblem graph context
            sp_context_lines = [
                f"[Thought Node {idx+1}] (Iter {sp['iteration']}): {sp['subproblem']}"
                for idx, sp in enumerate(subproblem_graph)
            ]
            graph_context = (
                "Subproblem Decomposition Graph (Graph-of-Thoughts):\n"
                + "\n".join(sp_context_lines)
            )

            evidence_text = build_evidence_block(retrieved_evidence)
            prompt = build_reasoning_prompt(
                question_text=formatted_q,
                evidence_text=evidence_text,
                depth=depth,
                iteration=current_iteration,
                prior_reasoning=accumulated_reasoning,
                dataset_type=dataset_type,
                method_name=self.name,
                graph_context=graph_context,
            )

            response = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            step_record = {
                "iteration": current_iteration,
                "subproblems": current_subproblems,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency": response.latency_seconds,
                "output_text": response.text,
            }
            reasoning_trace.append(step_record)
            accumulated_reasoning = response.text

        final_text = reasoning_trace[-1]["output_text"] if reasoning_trace else ""
        if dataset_type == "medmcqa":
            pred_answer = extract_medmcqa_answer(final_text)
        else:
            pred_answer = extract_pubmedqa_answer(final_text)

        is_correct = score_answer(pred_answer, ref_ans, dataset_type=dataset_type)
        wall_clock = time.perf_counter() - t0

        return {
            "question_id": q_id,
            "dataset": dataset_type,
            "question": q_text,
            "predicted_answer": pred_answer,
            "reference_answer": ref_ans,
            "is_correct": is_correct,
            "supporting_evidence": retrieved_evidence,
            "reasoning_trace": reasoning_trace,
            "retrieval_calls": retrieval_calls,
            "reasoning_iterations": len(reasoning_trace),
            "wall_clock_time": wall_clock,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "error_status": False,
            "error_message": None,
            "method": self.name,
            "depth": depth,
            "seed": seed,
            "config_id": f"{self.name}_{dataset_type}_d{depth}_s{seed}",
            "model_backend": self.llm.backend,
            "model_loaded": self.llm.model_loaded,
        }
