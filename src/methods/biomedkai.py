"""BiomedKAI / CARE-RAG: Depth-Governed Biomedical Knowledge Graph Traversal."""

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


class BiomedKAIMethod:
    """BiomedKAI / CARE-RAG implementation with depth-governed knowledge graph traversal."""

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
        self.name = "biomedkai"

    def run(
        self,
        question_item: Dict[str, Any],
        depth: int = 1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Execute BiomedKAI reasoning for a question at prescribed depth (1, 2, or 3).
        """
        t0 = time.perf_counter()
        dataset_type = question_item.get("dataset", "medmcqa")
        q_id = question_item["id"]
        q_text = question_item["question"]
        ref_ans = question_item["reference_answer"]

        # Format question text for prompt
        if dataset_type == "medmcqa":
            formatted_q = format_question_mcqa(q_text, question_item.get("options", {}))
        else:
            formatted_q = format_question_pubmedqa(q_text)

        system_prompt = build_system_prompt()

        # Seed entity extraction
        seed_entities = self.kg.extract_seed_entities(q_text)

        accumulated_reasoning = ""
        reasoning_trace = []
        retrieved_evidence: List[Dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        retrieval_calls = 0

        # Perform exactly `depth` iterations
        for current_iteration in range(1, depth + 1):
            # Graph traversal for the current hop
            traversal_result = self.kg.traverse_subgraph(
                seed_entities=seed_entities,
                depth=current_iteration,
                max_neighbors=4,
            )

            # Retrieve text evidence matching visited graph entities and question
            subgraph_nodes = traversal_result["visited_nodes"]
            search_query = f"{q_text} {' '.join(subgraph_nodes[:5])}"
            q_emb = self.embedder.embed_query(search_query)
            evidence_items = self.vector_store.search(q_emb, top_k=3)
            retrieval_calls += 1

            for ev in evidence_items:
                if ev not in retrieved_evidence:
                    retrieved_evidence.append(ev)

            # Build graph context summary
            edges_summary = [
                f"({e['head']}) -[{e['relation']}]-> ({e['tail']})"
                for e in traversal_result["visited_edges"][:5]
            ]
            graph_context = (
                f"Active Entities: {', '.join(subgraph_nodes[:6])}\n"
                f"Graph Paths:\n" + "\n".join(edges_summary)
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
                "traversal": traversal_result,
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
