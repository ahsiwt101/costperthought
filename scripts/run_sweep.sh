#!/usr/bin/env bash
# Phase 3: full factorial sweep. Run on the GPU host, after
# scripts/run_baseline.sh has been run and configs/grid.yaml's budget
# token counts have been filled in.
#
# Grid: {fp16, awq_int4} weights x {fp16, fp8} kv-cache x
#       {unlimited, budget_75pct, budget_50pct, budget_25pct} budget
#       = 16 configs, x 3 benchmarks = 48 runs.
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHT_PRECISIONS=("fp16" "awq_int4")
KV_PRECISIONS=("fp16" "fp8")
BUDGETS=("unlimited" "budget_75pct" "budget_50pct" "budget_25pct")
BENCHMARKS=("math" "gpqa" "code")

for bench in "${BENCHMARKS[@]}"; do
  for wp in "${WEIGHT_PRECISIONS[@]}"; do
    for kvp in "${KV_PRECISIONS[@]}"; do
      for budget in "${BUDGETS[@]}"; do
        echo "=== $bench | weight=$wp kv=$kvp budget=$budget ==="
        python harness/run_benchmark.py \
          --benchmark "$bench" \
          --weight_precision "$wp" \
          --kv_cache_precision "$kvp" \
          --budget "$budget" \
          || echo "!!! FAILED: $bench/$wp/$kvp/$budget -- logged and continuing"
      done
    done
  done
done

echo "Sweep complete. Results in results/summary.jsonl"
echo "Next: python analysis/interaction_test.py --benchmark math (and gpqa, code)"
