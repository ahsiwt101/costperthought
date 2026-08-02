"""
Samples a subset of LiveCodeBench (code_generation_lite). Writes
data/code_subset.jsonl with fields:
  {"id": str, "prompt": str, "test_cases": [{"input": str, "expected_output": str}, ...]}

NOTE: LiveCodeBench is released in dated releases specifically to mitigate
contamination (problems are tagged with a release date so you can filter to
problems released after a model's training cutoff). Filter to the most
recent release relative to DeepSeek-R1-Distill-Qwen-7B's training cutoff
and disclose the exact release/date range used in the write-up.
"""
import argparse
import json
import os
import random

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "code_subset.jsonl")

SYSTEM_PROMPT = (
    "You are a careful competitive programmer. Reason step by step inside "
    "<think> tags, then provide a single complete Python solution in a "
    "```python code block that reads from stdin and writes to stdout."
)


def main(n: int, seed: int, release_filter: str | None):
    from datasets import load_dataset

    ds = load_dataset("livecodebench/code_generation_lite", split="test",
                      version_tag=release_filter or "release_v1", trust_remote_code=True)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:n]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i in idx:
            row = ds[i]
            prompt = f"{SYSTEM_PROMPT}\n\nProblem: {row['question_content']}\n\n<think>\n"
            # LiveCodeBench stores public_test_cases as a JSON-encoded string,
            # not a list -- parse it first.
            raw_tc = row.get("public_test_cases") or "[]"
            try:
                parsed_tc = json.loads(raw_tc) if isinstance(raw_tc, str) else raw_tc
            except (json.JSONDecodeError, TypeError):
                parsed_tc = []
            test_cases = [
                {"input": tc["input"], "expected_output": tc["output"]}
                for tc in parsed_tc[:5]  # cap for grading speed
            ]
            f.write(json.dumps({
                "id": f"code_{i}",
                "prompt": prompt,
                "test_cases": test_cases,
                "difficulty": row.get("difficulty"),
            }) + "\n")
    print(f"Wrote {n} problems to {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--release_filter", default=None, help="e.g. 'release_v5' -- pick the release after the model's training cutoff")
    args = ap.parse_args()
    main(args.n, args.seed, args.release_filter)
