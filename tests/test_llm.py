"""Tests for BioMistral LLM interface, token counting, and deterministic response generation."""

import pytest
from src.shared.llm import BioMistralLLM, LLMResponse


class TestBioMistralLLM:

    @pytest.fixture(scope="class")
    def llm(self):
        return BioMistralLLM(backend="auto")

    def test_initialization(self, llm):
        assert llm.model_id == "BioMistral/BioMistral-7B"
        assert llm.tokenizer is not None
        assert len(llm.tokenizer) == 32000

    def test_true_token_counting(self, llm):
        sample_text = "Benign prostatic hyperplasia causes urinary obstruction."
        tokens = llm.count_tokens(sample_text)
        assert isinstance(tokens, int)
        assert tokens > 0
        assert tokens == len(llm.tokenizer.encode(sample_text, add_special_tokens=False))

    def test_deterministic_generation(self, llm):
        prompt = (
            "Question: What does chronic urethral obstruction lead to?\n"
            "Options: (A) Hyperplasia (B) Atrophy\n"
            "Conclude with your final answer choice on its own line: Answer: [A/B/C/D]"
        )
        resp1 = llm.generate(prompt)
        resp2 = llm.generate(prompt)

        assert isinstance(resp1, LLMResponse)
        assert resp1.total_tokens == resp1.prompt_tokens + resp1.completion_tokens
        assert resp1.text == resp2.text
        assert resp1.latency_seconds >= 0.0

    def test_hardware_reporting_integrity(self, llm):
        assert llm.backend in ["transformers", "test_only_mode"]
        if not llm.cuda_available and llm.total_ram_gb < 15.0:
            assert llm.backend == "test_only_mode"
            assert llm.hardware_blocker is not None
