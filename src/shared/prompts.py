"""Prompt templates, token-budgeted evidence builders, and context-window safety validators."""

import re
from typing import List, Dict, Any, Optional


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """
    Count tokens using the provided Hugging Face tokenizer (e.g. BioMistral-7B).
    Falls back to deterministic character/word heuristic (~4 chars/token) if tokenizer is absent.
    """
    if not text:
        return 0
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Fallback heuristic: 1 token ~= 4 characters or max(1, words * 1.3)
    return max(1, int(len(text) / 3.8))


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


def build_evidence_block(
    evidence_items: List[Dict[str, Any]],
    max_tokens: int = 2048,
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Format retrieved evidence snippets into a structured text block with traceable IDs.
    Enforces a strict, hard token budget (default 2048 tokens).
    Truncates at item boundaries whenever possible, or safely truncates individual items.
    Preserves most relevant evidence items first and removes exact duplicate texts.
    """
    if not evidence_items:
        return "No specific external evidence retrieved."

    lines = []
    seen_texts = set()
    current_tokens = 0

    for idx, item in enumerate(evidence_items, 1):
        if not isinstance(item, dict):
            continue

        raw_text = str(item.get("text", "")).strip()
        if not raw_text or raw_text in seen_texts:
            continue
        seen_texts.add(raw_text)

        source_id = str(item.get("id", f"doc_{idx}"))
        score = item.get("score")
        score_str = f" [Score: {score:.4f}]" if isinstance(score, (int, float)) else ""

        line = f"[{source_id}]{score_str} {raw_text}"
        line_tokens = count_tokens(line + "\n", tokenizer)

        if current_tokens + line_tokens <= max_tokens:
            lines.append(line)
            current_tokens += line_tokens
        else:
            # Need safe truncation for this item to utilize remaining token budget
            remaining_tokens = max_tokens - current_tokens
            # Reserve tokens for prefix and truncated marker
            trunc_marker = "... [truncated]"
            marker_tokens = count_tokens(trunc_marker, tokenizer)
            prefix = f"[{source_id}]{score_str} "
            prefix_tokens = count_tokens(prefix, tokenizer)

            available_for_text = remaining_tokens - prefix_tokens - marker_tokens
            if available_for_text >= 15:
                # Safe boundary truncation
                if tokenizer is not None and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
                    try:
                        token_ids = tokenizer.encode(raw_text, add_special_tokens=False)[:available_for_text]
                        truncated_body = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                        lines.append(f"{prefix}{truncated_body} {trunc_marker}")
                        current_tokens = max_tokens
                    except Exception:
                        char_limit = int(available_for_text * 3.5)
                        lines.append(f"{prefix}{raw_text[:char_limit]} {trunc_marker}")
                else:
                    char_limit = int(available_for_text * 3.5)
                    lines.append(f"{prefix}{raw_text[:char_limit]} {trunc_marker}")
            break

    if not lines:
        return "No specific external evidence retrieved."

    return "\n".join(lines)


def validate_prompt_budget(
    prompt: str,
    max_total_tokens: int = 3840,
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Context-window safety validator for BioMistral-7B (4096 / 8192 context).
    Verifies that the complete assembled prompt does not exceed max_total_tokens.
    If exceeded, safely prunes prior reasoning context while strictly protecting
    the system instructions, question, and final answering format.
    """
    total_tokens = count_tokens(prompt, tokenizer)
    if total_tokens <= max_total_tokens:
        return prompt

    # Excess tokens detected: safely prune prior reasoning section first
    if "### Prior Reasoning Context:" in prompt:
        pattern = r"(### Prior Reasoning Context:\n)(.*?)(?=\n### (?:Instructions|Graph Structure|Retrieved Evidence)|$)"
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            excess = total_tokens - max_total_tokens
            reasoning_body = match.group(2)
            reasoning_tokens = count_tokens(reasoning_body, tokenizer)
            allowed_tokens = max(50, reasoning_tokens - excess - 20)

            if tokenizer is not None and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
                try:
                    r_ids = tokenizer.encode(reasoning_body, add_special_tokens=False)[:allowed_tokens]
                    pruned_reasoning = tokenizer.decode(r_ids, skip_special_tokens=True).strip() + "\n... [prior reasoning truncated to protect context window]"
                except Exception:
                    char_budget = allowed_tokens * 3
                    pruned_reasoning = reasoning_body[:char_budget] + "\n... [prior reasoning truncated to protect context window]"
            else:
                char_budget = allowed_tokens * 3
                pruned_reasoning = reasoning_body[:char_budget] + "\n... [prior reasoning truncated to protect context window]"

            prompt = prompt[: match.start(2)] + pruned_reasoning + prompt[match.end(2) :]

    return prompt


def build_reasoning_prompt(
    question_text: str,
    evidence_text: str,
    depth: int,
    iteration: int,
    prior_reasoning: Optional[str] = None,
    dataset_type: str = "medmcqa",
    method_name: str = "flat_control",
    graph_context: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    max_total_tokens: int = 3840,
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

    assembled_prompt = "\n".join(parts)
    return validate_prompt_budget(assembled_prompt, max_total_tokens=max_total_tokens, tokenizer=tokenizer)
