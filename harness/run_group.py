"""
Runs every (budget x benchmark) cell for ONE engine config (weight x kv),
building the vLLM engine ONCE and reusing it across all 12 cells. This:
  - amortizes the ~80s model load across 12 cells (vs reloading per cell), and
  - keeps us to 4 jobs (one per weight x kv) instead of a 48-task array, which
    exceeds the cluster's 32-job submit limit.

Cells whose result file already exists (non-empty) are skipped, so a re-run
or requeue only fills gaps (the fp16/fp16 unlimited baselines from calibration
are picked up as already-done).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.serve import load_grid, build_engine, cell_id  # noqa: E402
from harness.run_benchmark import run  # noqa: E402


def main(weight: str, kv: str, config: str):
    grid = load_grid(config)
    results_dir = grid["output"]["results_dir"]
    benchmarks = [b["name"] for b in grid["benchmarks"]]
    budgets = [b["name"] for b in grid["reasoning_budget"]]

    llm = build_engine(weight, kv, grid)  # built ONCE, reused for all cells

    for bench in benchmarks:
        for budget in budgets:
            cid = cell_id(weight, kv, budget)
            out = os.path.join(results_dir, f"{bench}__{cid}.jsonl")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                print(f"[group] skip {bench}/{cid} (exists)", flush=True)
                continue
            print(f"[group] === running {bench} / {cid} ===", flush=True)
            try:
                run(bench, weight, kv, budget, config, mode="eval", llm=llm)
            except Exception as e:  # keep going; one bad cell shouldn't kill the job
                print(f"[group] !!! FAILED {bench}/{cid}: {type(e).__name__}: {e}", flush=True)

    print(f"[group] done engine w-{weight}_kv-{kv}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight_precision", required=True)
    ap.add_argument("--kv_cache_precision", required=True)
    ap.add_argument("--config", default="configs/grid.yaml")
    args = ap.parse_args()
    main(args.weight_precision, args.kv_cache_precision, args.config)
