"""
Grader for GPQA-diamond-style multiple-choice problems: extract the chosen
letter (A/B/C/D) from model output and compare to ground truth.
"""
import re

CHOICE_RE = re.compile(r"\b(?:answer(?:\s+is)?|final answer)\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE)
LAST_LETTER_RE = re.compile(r"\(?([A-D])\)?\s*$")


def extract_choice(model_output: str) -> str | None:
    m = CHOICE_RE.search(model_output)
    if m:
        return m.group(1).upper()
    tail = model_output.strip()[-20:]
    m2 = LAST_LETTER_RE.search(tail)
    if m2:
        return m2.group(1).upper()
    return None


def grade(model_output: str, problem: dict) -> dict:
    """problem must have 'ground_truth' key (a letter A-D)."""
    pred = extract_choice(model_output)
    correct = pred is not None and pred == problem["ground_truth"].strip().upper()
    return {"correct": correct, "extracted_answer": pred}
