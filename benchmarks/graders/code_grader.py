"""
Grader for LiveCodeBench-style problems: extract the code block from model
output and execute it against the problem's test cases in a sandboxed
subprocess with a timeout. Pass iff all test cases pass.

SAFETY NOTE: this executes model-generated code. Must run in an isolated
sandbox/container with no network access and a hard wall-clock timeout --
never run directly on a host with credentials or sensitive data. The GMI
Cloud GPU box used for serving should NOT double as the grading sandbox;
run grading in a separate throwaway container/VM.
"""
import re
import subprocess
import tempfile
import os

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
TIMEOUT_SECONDS = 5


def extract_code(model_output: str) -> str | None:
    matches = CODE_BLOCK_RE.findall(model_output)
    if not matches:
        return None
    return matches[-1].strip()


def run_test_case(code: str, test_input: str, expected_output: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            ["python3", path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        return proc.stdout.strip() == expected_output.strip()
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(path)


def grade(model_output: str, problem: dict) -> dict:
    """problem must have 'test_cases': list of {"input": str, "expected_output": str}."""
    code = extract_code(model_output)
    if code is None:
        return {"correct": False, "extracted_answer": None, "n_tests_passed": 0, "n_tests_total": len(problem.get("test_cases", []))}

    test_cases = problem.get("test_cases", [])
    n_passed = sum(run_test_case(code, tc["input"], tc["expected_output"]) for tc in test_cases)
    return {
        "correct": n_passed == len(test_cases) and len(test_cases) > 0,
        "extracted_answer": code[:200],  # truncated for logging
        "n_tests_passed": n_passed,
        "n_tests_total": len(test_cases),
    }
