"""
Grader for MATH-500-style problems: extract the boxed/final answer from
model output and check mathematical equivalence against ground truth using
sympy (handles equivalent forms like 1/2 vs 0.5, sqrt(4) vs 2, etc.).
"""
import re

try:
    from sympy import sympify, simplify
    from sympy.parsing.latex import parse_latex
    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False

def _extract_boxed(text: str) -> list[str]:
    """Find all \\boxed{...} contents with BALANCED brace matching, so nested
    LaTeX like \\boxed{-\\sqrt{3}} or \\boxed{x \\in [-2,7]} is captured whole
    (a non-greedy regex stops at the first inner '}' and truncates)."""
    out = []
    key = r"\boxed{"
    i = 0
    while True:
        start = text.find(key, i)
        if start == -1:
            break
        j = start + len(key)
        depth = 1
        buf = []
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            j += 1
        out.append("".join(buf).strip())
        i = j + 1
    return out


def extract_answer(model_output: str) -> str | None:
    matches = _extract_boxed(model_output)
    if matches:
        return matches[-1].strip()
    # fallback: last line starting "Answer:" or "The final answer is"
    m = re.search(r"(?:final answer is|answer:)\s*(.+)", model_output, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _normalize(s: str) -> str:
    return s.replace(" ", "").replace("\\!", "").replace("\\,", "").strip()


def is_correct(model_output: str, ground_truth: str) -> tuple[bool, str | None]:
    pred = extract_answer(model_output)
    if pred is None:
        return False, None

    if _normalize(pred) == _normalize(ground_truth):
        return True, pred

    if _SYMPY_OK:
        try:
            p = parse_latex(pred) if "\\" in pred else sympify(pred)
            g = parse_latex(ground_truth) if "\\" in ground_truth else sympify(ground_truth)
            if simplify(p - g) == 0:
                return True, pred
        except Exception:
            pass  # fall through to string-mismatch = incorrect

    return False, pred


def grade(model_output: str, problem: dict) -> dict:
    """problem must have 'ground_truth' key. Returns a result dict for logging."""
    correct, extracted = is_correct(model_output, problem["ground_truth"])
    return {"correct": correct, "extracted_answer": extracted}
