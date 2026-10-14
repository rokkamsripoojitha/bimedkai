"""Tests for answer extraction, scoring rules, Wilson intervals, linear slopes, and McNemar test."""

import pytest
from src.shared.metrics import (
    extract_medmcqa_answer,
    extract_pubmedqa_answer,
    score_answer,
    calculate_accuracy,
    calculate_wilson_score_interval,
    calculate_linear_slope,
    calculate_depth_changes,
    mcnemar_test,
)


class TestMetrics:

    def test_medmcqa_extraction(self):
        assert extract_medmcqa_answer("The final diagnosis is confirmed.\nAnswer: [B]") == "B"
        assert extract_medmcqa_answer("Answer: C") == "C"
        assert extract_medmcqa_answer("The correct option is (A).") == "A"
        assert extract_medmcqa_answer("D") == "D"
        assert extract_medmcqa_answer("Invalid text with no options") is None

    def test_pubmedqa_extraction(self):
        assert extract_pubmedqa_answer("Conclusion: Answer: [yes]") == "yes"
        assert extract_pubmedqa_answer("We conclude that the answer is no.") == "no"
        assert extract_pubmedqa_answer("Answer: maybe") == "maybe"
        assert extract_pubmedqa_answer("random gibberish") is None

    def test_scoring(self):
        assert score_answer("A", "A", dataset_type="medmcqa") == 1
        assert score_answer("B", "A", dataset_type="medmcqa") == 0
        assert score_answer(None, "A", dataset_type="medmcqa") == 0
        assert score_answer("yes", "yes", dataset_type="pubmedqa") == 1
        assert score_answer("no", "yes", dataset_type="pubmedqa") == 0

    def test_wilson_score_interval(self):
        low, high = calculate_wilson_score_interval(successes=80, total=100)
        assert 0.70 < low < 0.80
        assert 0.80 < high < 0.90

    def test_linear_slope(self):
        depths = [1, 2, 3]
        accs = [0.40, 0.50, 0.60]
        slope = calculate_linear_slope(depths, accs)
        assert pytest.approx(slope, 1e-4) == 0.10

    def test_depth_changes(self):
        acc_dict = {1: 0.40, 2: 0.55, 3: 0.58}
        changes = calculate_depth_changes(acc_dict)
        assert pytest.approx(changes["delta_1_to_2"], 1e-4) == 0.15
        assert pytest.approx(changes["delta_2_to_3"], 1e-4) == 0.03

    def test_mcnemar_test(self):
        # (both_correct, method_a_only, method_b_only, both_incorrect)
        res = mcnemar_test((50, 20, 5, 25))
        assert "statistic" in res
        assert "p_value" in res
        assert res["statistic"] > 0
        assert res["p_value"] < 0.05
