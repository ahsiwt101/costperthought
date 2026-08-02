"""
Samples the GPQA-diamond subset. Writes data/gpqa_subset.jsonl with fields:
  {"id": str, "prompt": str, "ground_truth": "A"|"B"|"C"|"D"}

NOTE: GPQA is gated on Hugging Face (requires accepting terms + a token) to
reduce contamination risk from scraping. Set HF_TOKEN before running.
"""
import argparse
import json
import os
import random

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gpqa_subset.jsonl")

SYSTEM_PROMPT = (
    "You are answering a graduate-level science multiple-choice question. "
    "Reason step by step inside <think> tags, then give your final answer "
    "as a single letter: 'Answer: X'."
)


def main(n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)
    rng = random.Random(seed)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rows = []
    for i, row in enumerate(ds):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        letters = ["A", "B", "C", "D"]
        shuffled_choices = [choices[j] for j in order]
        correct_letter = letters[order.index(0)]  # index 0 was the correct answer pre-shuffle

        choice_block = "\n".join(f"{letters[k]}) {shuffled_choices[k]}" for k in range(4))
        prompt = (
            f"{SYSTEM_PROMPT}\n\nQuestion: {row['Question']}\n\n{choice_block}\n\n<think>\n"
        )
        rows.append({
            "id": f"gpqa_{i}",
            "prompt": prompt,
            "ground_truth": correct_letter,
        })

    rng.shuffle(rows)
    rows = rows[:n]

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} problems to {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n, args.seed)
