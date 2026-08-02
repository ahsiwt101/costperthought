"""
Samples a fixed subset of MATH-500 for the benchmark, with a fixed seed for
reproducibility. Writes data/math_subset.jsonl with fields:
  {"id": str, "prompt": str, "ground_truth": str}
"""
import argparse
import json
import os
import random

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "math_subset.jsonl")

SYSTEM_PROMPT = (
    "You are a careful mathematician. Reason step by step inside <think> "
    "tags, then give your final answer clearly, boxed as \\boxed{answer}."
)


def main(n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:n]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i in idx:
            row = ds[i]
            prompt = f"{SYSTEM_PROMPT}\n\nProblem: {row['problem']}\n\n<think>\n"
            f.write(json.dumps({
                "id": f"math_{i}",
                "prompt": prompt,
                "ground_truth": row["answer"],
                "level": row.get("level"),
                "subject": row.get("subject"),
            }) + "\n")
    print(f"Wrote {n} problems to {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n, args.seed)
