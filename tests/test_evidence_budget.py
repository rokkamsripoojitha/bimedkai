"""Tests for hard token budget enforcement, boundary-safe truncation, and prompt context-window safety."""

import pytest
from transformers import AutoTokenizer
from src.shared.prompts import (
    count_tokens,
    build_evidence_block,
    validate_prompt_budget,
    build_reasoning_prompt,
)


class TestEvidenceBudgetAndPromptSafety:

    @pytest.fixture(scope="class")
    def tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
        except Exception:
            return None

    def test_empty_evidence(self):
        res = build_evidence_block([])
        assert res == "No specific external evidence retrieved."

    def test_normal_evidence_within_budget(self, tokenizer):
        items = [
            {"id": "doc_1", "text": "Chronic kidney disease results in normocytic anemia.", "score": 0.95},
            {"id": "doc_2", "text": "Erythropoietin deficiency causes decreased red cell production.", "score": 0.88},
        ]
        res = build_evidence_block(items, max_tokens=2048, tokenizer=tokenizer)
        assert "[doc_1]" in res
        assert "[doc_2]" in res
        tok_count = count_tokens(res, tokenizer)
        assert tok_count <= 2048

    def test_oversized_evidence_boundary_truncation(self, tokenizer):
        # 100 repetitive medical passages that together exceed 5000 tokens
        items = [
            {"id": f"doc_{i}", "text": f"Pathological report {i}: progressive renal parenchymal compression with hydronephrosis dilatation.", "score": 1.0 / (i + 1)}
            for i in range(100)
        ]
        res = build_evidence_block(items, max_tokens=300, tokenizer=tokenizer)
        tok_count = count_tokens(res, tokenizer)
        assert tok_count <= 305  # within boundary margin

        # Best ranked item (doc_0) must be preserved
        assert "[doc_0]" in res
        # Distant low-ranked items must be safely dropped
        assert "[doc_99]" not in res

    def test_very_long_individual_item_truncation(self, tokenizer):
        # A single passage of 5000 words
        long_text = "Progressive renal atrophy " * 1500
        items = [{"id": "huge_doc", "text": long_text, "score": 0.99}]
        res = build_evidence_block(items, max_tokens=200, tokenizer=tokenizer)
        tok_count = count_tokens(res, tokenizer)
        assert tok_count <= 205
        assert "[huge_doc]" in res
        assert "[truncated]" in res

    def test_depth_3_accumulated_evidence_safety(self, tokenizer):
        # Simulate depth 3 with 50 diverse items
        items = [
            {"id": f"depth3_doc_{i}", "text": f"Biomedical finding {i} on disease etiology and clinical pharmacology.", "score": 0.5}
            for i in range(50)
        ]
        res = build_evidence_block(items, max_tokens=2048, tokenizer=tokenizer)
        tok_count = count_tokens(res, tokenizer)
        assert tok_count <= 2048

    def test_validate_prompt_budget_total_context_protection(self, tokenizer):
        question = "What is the primary cause of renal atrophy in chronic BPH?"
        huge_prior_reasoning = "Detailed physiological observation of cellular apoptosis and backpressure. " * 300
        evidence = "Hydronephrosis causes progressive parenchymal compression."

        prompt = build_reasoning_prompt(
            question_text=question,
            evidence_text=evidence,
            depth=3,
            iteration=3,
            prior_reasoning=huge_prior_reasoning,
            dataset_type="medmcqa",
            method_name="biomedkai",
            max_total_tokens=500,
            tokenizer=tokenizer,
        )

        total_tokens = count_tokens(prompt, tokenizer)
        # Total prompt must fit inside configured limit
        assert total_tokens <= 520
        # Question and Instructions must be strictly preserved
        assert question in prompt
        assert "Answer: [A/B/C/D]" in prompt
        assert "... [prior reasoning truncated to protect context window]" in prompt
