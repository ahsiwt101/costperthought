"""
Orchestrates one grid cell x one benchmark: build the engine, run every
problem through generate_with_budget, grade, log cost, write results.

Usage (on the GPU host):
  python harness/run_benchmark.py \
      --benchmark math --weight_precision awq_int4 --kv_cache_precision fp8 \
      --budget budget_50pct --config configs/grid.yaml

  python harness/run_benchmark.py --benchmark math --mode calibrate
      # Phase 2: unlimited-budget run used to pick budget_75/50/25pct token
      # counts (prints median/p25/p75 think-token length).
"""
import argparse
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.serve import load_grid, build_engine, cell_id  # noqa: E402
from harness.budget_forcing import generate_batch_with_budget  # noqa: E402
from harness.cost import compute_cell_cost  # noqa: E402


def load_benchmark(name: str, grid: dict):
    bench_cfg = next(b for b in grid["benchmarks"] if b["name"] == name)
    path = os.path.join(os.path.dirname(__file__), "..", "data", f"{name}_subset.jsonl")
    with open(path) as f:
        problems = [json.loads(line) for line in f]
    grader = importlib.import_module(bench_cfg["grader"].replace("benchmarks.", "benchmarks."))
    return problems, grader


def run(benchmark: str, weight_precision: str, kv_cache_precision: str,
        budget_name: str, config_path: str, mode: str = "eval", limit: int = 0,
        llm=None):
    grid = load_grid(config_path)
    problems, grader = load_benchmark(benchmark, grid)
    if limit and limit > 0:
        problems = problems[:limit]  # smoke-test / quick-iteration cap

    budget_cfg = next(b for b in grid["reasoning_budget"] if b["name"] == budget_name)
    max_think_tokens = budget_cfg["max_think_tokens"]
    if isinstance(max_think_tokens, dict):
        max_think_tokens = max_think_tokens.get(benchmark)
    if max_think_tokens == "TBD" or max_think_tokens is None and budget_name != "unlimited":
        raise ValueError(
            f"Budget level '{budget_name}' has not been calibrated for benchmark "
            f"'{benchmark}' yet. Run scripts/run_baseline.sh (--mode calibrate) first "
            f"and fill in configs/grid.yaml."
        )

    if llm is None:  # standalone invocation builds its own engine
        llm = build_engine(weight_precision, kv_cache_precision, grid)
    from vllm import SamplingParams  # noqa

    usd_per_gpu_hour = grid["gpu_cost"]["usd_per_gpu_hour"]
    sampling_cfg = grid["sampling"]

    # Batched generation: all problems in one vLLM call so continuous batching
    # runs them concurrently (the KV cache supports ~70x concurrency for this
    # model). wall_clock_seconds per result is the amortized batch time, which
    # is the correct throughput basis for the $/query cost model.
    gens = generate_batch_with_budget(
        llm=llm,
        prompts=[p["prompt"] for p in problems],
        sampling_params_cls=SamplingParams,
        max_think_tokens=max_think_tokens,
        temperature=sampling_cfg["temperature"],
        top_p=sampling_cfg["top_p"],
        total_hard_ceiling=sampling_cfg["max_tokens_total"],
    )

    results = []
    for problem, gen in zip(problems, gens):
        grade_result = grader.grade(gen.text, problem)
        results.append({
            "id": problem["id"],
            "benchmark": benchmark,
            "weight_precision": weight_precision,
            "kv_cache_precision": kv_cache_precision,
            "budget": budget_name,
            "text_head": gen.text[:300],
            "text_tail": gen.text[-1200:],
            "think_tokens_used": gen.think_tokens_used,
            "answer_tokens_used": gen.answer_tokens_used,
            "total_tokens": gen.total_tokens,
            "wall_clock_seconds": gen.wall_clock_seconds,
            "budget_forced": gen.budget_forced,
            **grade_result,
        })

    if mode == "calibrate":
        lengths = sorted(r["think_tokens_used"] for r in results)
        n = len(lengths)
        print(f"[calibrate] benchmark={benchmark} n={n}")
        print(f"  p25={lengths[n // 4]}  median={lengths[n // 2]}  p75={lengths[3 * n // 4]}")
        print(f"  -> set budget_75pct={int(0.75 * lengths[n // 2])}, "
              f"budget_50pct={int(0.50 * lengths[n // 2])}, "
              f"budget_25pct={int(0.25 * lengths[n // 2])} for '{benchmark}' in configs/grid.yaml")

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0.0
    mean_wall_clock = sum(r["wall_clock_seconds"] for r in results) / len(results) if results else 0.0
    cost = compute_cell_cost(mean_wall_clock, accuracy, usd_per_gpu_hour)

    out_dir = grid["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cid = cell_id(weight_precision, kv_cache_precision, budget_name)
    out_path = os.path.join(out_dir, f"{benchmark}__{cid}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "benchmark": benchmark,
        "cell": cid,
        "n": len(results),
        "accuracy": accuracy,
        "mean_wall_clock_seconds": mean_wall_clock,
        "usd_per_query": cost.usd_per_query,
        "usd_per_correct_answer": cost.usd_per_correct_answer,
    }
    summary_path = os.path.join(out_dir, "summary.jsonl")
    with open(summary_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"[run] {benchmark} / {cid}: accuracy={accuracy:.3f} "
          f"$/query={cost.usd_per_query:.5f} $/correct={cost.usd_per_correct_answer}")
    print(f"[run] wrote {out_path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["math", "gpqa", "code"])
    ap.add_argument("--weight_precision", default="fp16")
    ap.add_argument("--kv_cache_precision", default="fp16")
    ap.add_argument("--budget", default="unlimited")
    ap.add_argument("--config", default="configs/grid.yaml")
    ap.add_argument("--mode", default="eval", choices=["eval", "calibrate"])
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap number of problems (0 = all). Used for smoke tests.")
    args = ap.parse_args()

    budget = "unlimited" if args.mode == "calibrate" else args.budget
    run(args.benchmark, args.weight_precision, args.kv_cache_precision,
        budget, args.config, mode=args.mode, limit=args.limit)
