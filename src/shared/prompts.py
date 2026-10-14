"""Prompt templates and message builders for biomedical reasoning across methods and depths."""

from typing import List, Dict, Any, Optional


def build_system_prompt() -> str:
    """Return standard biomedical expert system prompt."""
    return (
        "You are an expert biomedical artificial intelligence assistant. "
        "Your task is to analyze biomedical questions using provided evidence "
        "and logical reasoning, and arrive at the strictly correct answer."
    )


def format_question_mcqa(question: str, options: Dict[str, str]) -> str:
    """Format multiple choice question with options A, B, C, D."""
    formatted_options = "\n".join([f"({k}) {v}" for k, v in sorted(options.items())])
    return f"Question:\n{question}\n\nOptions:\n{formatted_options}"


def format_question_pubmedqa(question: str) -> str:
    """Format PubMedQA question requiring yes/no/maybe."""
    return (
        f"Question:\n{question}\n\n"
        "Valid answers: (yes) / (no) / (maybe)"
    )


def build_evidence_block(evidence_items: List[Dict[str, Any]]) -> str:
    """Format retrieved evidence snippets into a structured text block with traceable IDs."""
    if not evidence_items:
        return "No specific external evidence retrieved."
    lines = []
    for idx, item in enumerate(evidence_items, 1):
        source_id = item.get("id", f"doc_{idx}")
        text = item.get("text", "").strip()
        score = item.get("score")
        score_str = f" [Score: {score:.4f}]" if score is not None else ""
        lines.append(f"[{source_id}]{score_str} {text}")
    return "\n".join(lines)


def build_reasoning_prompt(
    question_text: str,
    evidence_text: str,
    depth: int,
    iteration: int,
    prior_reasoning: Optional[str] = None,
    dataset_type: str = "medmcqa",
    method_name: str = "flat_control",
    graph_context: Optional[str] = None,
) -> str:
    """Build standardized reasoning prompt for a specific iteration under the depth budget."""
    parts = []

    if iteration == 1:
        parts.append("### Task: Initial Reasoning and Evidence Analysis (Iteration 1)")
    else:
        parts.append(f"### Task: Iterative Refinement and Deepened Analysis (Iteration {iteration}/{depth})")

    parts.append(f"\n{question_text}\n")

    if evidence_text:
        parts.append(f"### Retrieved Evidence:\n{evidence_text}\n")

    if graph_context:
        parts.append(f"### Graph Structure & Trace:\n{graph_context}\n")

    if prior_reasoning:
        parts.append(f"### Prior Reasoning Context:\n{prior_reasoning}\n")

    is_final_step = (iteration == depth)

    if is_final_step:
        if dataset_type == "medmcqa":
            format_rule = (
                "Synthesize your final medical evaluation.\n"
                "Conclude with your final answer choice on its own line in this exact format:\n"
                "Answer: [A/B/C/D]"
            )
        else:  # pubmedqa
            format_rule = (
                "Synthesize your final medical evaluation.\n"
                "Conclude with your final answer on its own line in this exact format:\n"
                "Answer: [yes/no/maybe]"
            )
        parts.append(f"### Instructions:\n{format_rule}")
    else:
        parts.append(
            "### Instructions:\n"
            "Analyze the evidence thoroughly. Identify key physiological mechanisms, pathological factors, "
            "and intermediate conclusions. Do NOT produce a final answer yet; record your intermediate reasoning."
        )

    return "\n".join(parts)
