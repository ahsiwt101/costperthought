#!/usr/bin/env bash
#SBATCH --job-name=cpt-setup
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/setup-%j.out
#SBATCH --error=slurm/logs/setup-%j.out
#
# Phase 1: build env + pre-stage weights/data. Runs on a CPU compute node
# (login node caps users at 1GB vmem; compute nodes have unlimited mem + internet).
# Everything lands on the shared $HOME network FS -> visible to GPU jobs offline.
# Idempotent - safe to re-run.
#
# KEY: the GPU nodes run driver 575.57.08 = CUDA 12.9, so torch/vLLM MUST be the
# CUDA 12.8 build (the default PyPI vLLM pulls cu130 = CUDA 13, which will NOT run
# on a 12.9 driver). We install via `uv ... --torch-backend=cu128`.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CPT_SCRATCH="${CPT_SCRATCH:-$HOME/cpt-scratch}"
export HF_HOME="$CPT_SCRATCH/hf"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")" && echo "[setup] HF_TOKEN loaded from ~/.hf_token"
mkdir -p "$HF_HOME" slurm/logs data results

VENV="$HOME/cpt-venv"           # serving/analysis env (vLLM, cu128)
DVENV="$HOME/cpt-data-venv"     # isolated prep env (datasets<3 for script-based benchmarks)
PY="$VENV/bin/python"
DPY="$DVENV/bin/python"

echo "==================== node: $(hostname)  mem: $(ulimit -v) ===================="

echo "==================== 1. serving venv + uv (clean build for consistent cu128) ===================="
# Rebuild from scratch: an in-place torch cu130->cu128 swap leaves a mismatched
# mix of nvidia-cu13/cu12 libs (missing libcupti.so.12 etc). A fresh venv pulls a
# consistent cu128 dep tree. Weights live in $HF_HOME (untouched), so this is cheap.
rm -rf "$VENV"
python3 -m venv "$VENV"
"$PY" -m pip install --upgrade pip wheel uv

echo "==================== 2. vLLM 0.11.0 for CUDA 12.8 (driver 575/CUDA12.9) ===================="
# The GPU nodes run a CUDA 12.9 driver. The latest vLLM (0.26.0) is a CUDA-13
# wheel: its compiled ext links libcudart.so.13, which no 12.x driver provides.
# vLLM 0.11.0 is the newest CUDA-12.8 build (torch 2.8.0+cu128) and still supports
# AWQ weights + FP8 KV-cache. --torch-backend=cu128 pins the whole torch tree to cu128.
"$PY" -m uv pip install --python "$PY" --torch-backend=cu128 "vllm==0.11.0"
# Serving/analysis deps (NOT datasets - that lives in the prep venv below).
# Pin transformers <5: vLLM 0.11.0 only caps it at >=4.55.2, so uv otherwise
# grabs transformers 5.x, which removed tokenizer APIs vLLM 0.11.0 relies on
# (Qwen2Tokenizer.all_special_tokens_extended).
"$PY" -m uv pip install --python "$PY" \
  "transformers>=4.55.2,<5" \
  "openai>=1.30.0" requests pyyaml tqdm pandas numpy "sympy>=1.12" regex \
  "scipy>=1.13" statsmodels streamlit plotly "huggingface_hub[hf_transfer]"
echo "--- verify torch cuda build is cu12x + no libcudart.so.13 linkage ---"
"$PY" -c "import torch,vllm; assert torch.version.cuda.startswith('12'), torch.version.cuda; print('OK torch',torch.__version__,'vllm',vllm.__version__)"

echo "==================== 3. data-prep venv (datasets 2.21, script datasets) ===================="
if [ ! -x "$DPY" ]; then python3 -m venv "$DVENV"; fi
"$DPY" -m pip install --upgrade pip
"$DPY" -m pip install "datasets==2.21.0" "huggingface_hub" pandas

echo "==================== 4. pre-download model weights -> \$HF_HOME ===================="
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
for repo in ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
             "jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ"]:
    print("downloading", repo, flush=True)
    snapshot_download(repo, ignore_patterns=["*.pth", "original/*"])
print("weights cached")
PYEOF

echo "==================== 5. prepare benchmark subsets ===================="
# math is parquet (either venv works); gpqa + code are script-based -> prep venv.
# Skip any subset that is already built (non-empty) so re-runs are cheap.
[ -s data/math_subset.jsonl ] && echo "[skip] math_subset.jsonl exists" || "$DPY" benchmarks/prepare_math.py --n 150
if [ -s data/gpqa_subset.jsonl ]; then
  echo "[skip] gpqa_subset.jsonl exists"
elif [ -n "${HF_TOKEN:-}" ]; then
  "$DPY" benchmarks/prepare_gpqa.py --n 100 || echo "!!! gpqa failed even with token - check license acceptance"
else
  echo "!!! SKIP gpqa - no ~/.hf_token. Create it + accept the Idavidrein/gpqa license, then re-run."
fi
[ -s data/code_subset.jsonl ] && echo "[skip] code_subset.jsonl exists" || "$DPY" benchmarks/prepare_code.py --n 100 || echo "!!! code prep failed"

echo "==================== DONE ===================="
ls -la data/
echo "[setup] serving venv=$VENV  prep venv=$DVENV  HF_HOME=$HF_HOME"
