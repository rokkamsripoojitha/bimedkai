"""Evaluation metrics, answer extraction, scoring, slope calculation, and statistical tests."""

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats


def extract_medmcqa_answer(text: str) -> Optional[str]:
    """
    Extract multiple choice answer (A, B, C, or D) from model output.
    Looks for standard patterns like 'Answer: [A]', 'Answer: A', '(A)', 'option A'.
    """
    if not text:
        return None

    # Priority 1: Strict 'Answer: [X]' or 'Answer: X'
    match = re.search(r"Answer:\s*\[?([A-Da-d])\]?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Priority 2: 'The correct answer is (X)' or 'is option X'
    match = re.search(r"(?:correct\s+answer\s+is|option|choice)\s*[:\s(]*([A-Da-d])[)\s\.]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Priority 3: Last isolated letter A-D on its own line
    matches = re.findall(r"^\s*([A-Da-d])[\.\)]?\s*$", text, re.MULTILINE)
    if matches:
        return matches[-1].upper()

    # Priority 4: Look for trailing single character
    last_words = text.strip().split()
    if last_words:
        clean_last = re.sub(r"[^\w]", "", last_words[-1])
        if clean_last.upper() in ["A", "B", "C", "D"]:
            return clean_last.upper()

    return None


def extract_pubmedqa_answer(text: str) -> Optional[str]:
    """
    Extract PubMedQA answer ('yes', 'no', or 'maybe') from model output.
    """
    if not text:
        return None

    # Priority 1: Strict 'Answer: [yes/no/maybe]' or 'Answer: yes/no/maybe'
    match = re.search(r"Answer:\s*\[?(yes|no|maybe)\]?", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Priority 2: 'The answer is yes/no/maybe'
    match = re.search(r"(?:the\s+answer\s+is|decision\s+is|conclude\s+that)\s*[:\s\[(]*(yes|no|maybe)[\])\s\.]", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Priority 3: Search text for exact tokens
    lower_text = text.lower()
    for choice in ["yes", "no", "maybe"]:
        if re.search(rf"\b{choice}\b", lower_text):
            # check the end of the text
            last_idx = lower_text.rfind(choice)
            if last_idx >= len(lower_text) - 50:
                return choice

    return None


def score_answer(predicted: Optional[str], reference: str, dataset_type: str = "medmcqa") -> int:
    """Return 1 for exact match, 0 otherwise. Handles None as 0 (invalid/unextracted)."""
    if predicted is None:
        return 0
    if dataset_type == "medmcqa":
        return 1 if predicted.strip().upper() == reference.strip().upper() else 0
    elif dataset_type == "pubmedqa":
        return 1 if predicted.strip().lower() == reference.strip().lower() else 0
    else:
        return 1 if predicted.strip() == reference.strip() else 0


def calculate_accuracy(correct_count: int, total_count: int) -> float:
    """Calculate accuracy proportion."""
    if total_count == 0:
        return 0.0
    return float(correct_count) / float(total_count)


def calculate_wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre_adj_p = p + z**2 / (2 * total)
    err = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lower = max(0.0, (centre_adj_p - err) / denom)
    upper = min(1.0, (centre_adj_p + err) / denom)
    return (float(lower), float(upper))


def calculate_linear_slope(depths: List[int], accuracies: List[float]) -> float:
    """
    Compute ordinary least squares linear regression slope of accuracy vs depth.
    slope = sum((x - mean(x)) * (y - mean(y))) / sum((x - mean(x))^2)
    """
    if len(depths) != len(accuracies) or len(depths) < 2:
        return 0.0
    x = np.array(depths, dtype=float)
    y = np.array(accuracies, dtype=float)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return float(slope)


def calculate_depth_changes(accuracies_by_depth: Dict[int, float]) -> Dict[str, float]:
    """Calculate delta from depth 1 to 2 and depth 2 to 3."""
    acc1 = accuracies_by_depth.get(1, 0.0)
    acc2 = accuracies_by_depth.get(2, 0.0)
    acc3 = accuracies_by_depth.get(3, 0.0)
    return {
        "delta_1_to_2": float(acc2 - acc1),
        "delta_2_to_3": float(acc3 - acc2),
        "total_delta_1_to_3": float(acc3 - acc1),
    }


def mcnemar_test(contingency_table: Tuple[int, int, int, int]) -> Dict[str, float]:
    """
    Calculate McNemar test for paired binary outcomes on identical questions.
    contingency_table: (both_correct, method_a_only, method_b_only, both_incorrect)
    """
    _, b, c, _ = contingency_table
    if (b + c) == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    # McNemar with Edwards continuity correction
    statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))
    return {"statistic": float(statistic), "p_value": p_value}
