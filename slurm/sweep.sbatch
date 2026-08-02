#!/usr/bin/env bash
#SBATCH --job-name=cpt-sweep
#SBATCH --partition=gpu
#SBATCH --time=02:59:00
#SBATCH --gres=gpu:1     # consistent GPU across all cells (baseline calibrated here)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-2               # 3 buildable engines (awq x fp8 is a vLLM dtype-conflict; see serve.py)
#SBATCH --requeue
#SBATCH --output=slurm/logs/sweep-%A_%a.out
#SBATCH --error=slurm/logs/sweep-%A_%a.out
#
# Phase 5: full factorial sweep, grouped by ENGINE (weight x kv) to stay under
# the 32-job submit limit and amortize model load. Each array task builds one
# engine once and runs all 4 budgets x 3 benchmarks = 12 cells (resumable /
# skip-if-done). 4 tasks x 12 cells = 48 cells total.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"

WEIGHTS=(fp16 fp16 awq_int4 awq_int4)
KV=(fp16 fp8 fp16 fp8)

i=${SLURM_ARRAY_TASK_ID:?must run as an array job}
weight=${WEIGHTS[$i]}
kv=${KV[$i]}

echo "=== [task $i] engine weight=$weight kv=$kv ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

python harness/run_group.py --weight_precision "$weight" --kv_cache_precision "$kv"

echo "[task $i] done engine w-${weight}_kv-${kv}"
