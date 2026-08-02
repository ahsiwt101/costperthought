# CostPerThought

**Does stacking "safe" reasoning-model inference-efficiency tricks quietly compound into worse degradation than any one alone - and what does it actually cost in dollars?**

LaunchPad 2026 AI Challenge submission · Track: Infrastructure & tooling

- **Live dashboard:** https://costperthought.streamlit.app/
- **Full write-up (~1000 words):** [docs/writeup.md](docs/writeup.md)

## Short summary

CostPerThought is a measurement harness that runs the **full factorial** of three "nearly free" reasoning-model inference optimizations - weight quantization, KV-cache quantization, and reasoning-token budget capping - **together**, on real GPU hardware, and prices every configuration in **dollars per correct answer**. It is for the ML-infra and platform teams who stack these knobs in production on faith. It matters because the joint behavior is not what the isolated papers imply: **two of the three knobs turned out to be far harder to even turn on than advertised**, and the knob everyone treats as a pure memory optimization (weight quantization) actually made cost *worse* on roomy hardware - the opposite of the folk wisdom.

## The question

Teams serving reasoning models (long chain-of-thought, R1-style) cut inference cost with three knobs, each usually validated **in isolation**:

1. **Weight quantization** (e.g. FP16 → AWQ-INT4)
2. **KV-cache quantization** (e.g. FP16 → FP8)
3. **Reasoning-token budget capping** ("budget forcing" - cut the thinking phase short)

In production, all three get turned on at once. Nobody has published what happens when you compose them: does the accuracy loss stay additive, or does precision loss force the model to spend *more* reasoning tokens to compensate, quietly erasing the latency win from budget capping? And what is the actual dollar cost per correct answer, measured on real rented GPU hardware rather than a FLOPs estimate?

This repo runs the full factorial grid (2 weight precisions × 2 KV-cache precisions × 4 reasoning-budget levels) across 3 reasoning benchmark domains (math, science QA, code) on `DeepSeek-R1-Distill-Qwen-7B` served via vLLM on a **datacenter-class GPU**, priced at a market GPU reference rate, and reports everything in accuracy, latency, and **$ per correct answer**.

## Headline findings

Ran on a GPU cluster (Slurm; compute free, priced at a $2.40/GPU-hr datacenter-GPU reference). 36 cells measured. Full detail in [docs/writeup.md](docs/writeup.md); raw data in `results/`.

1. **Weight-quant × budget compose additively** - no significant accuracy interaction on any benchmark (bootstrap 95% CIs span 0). Stacking AWQ-INT4 + a tight budget costs about the sum of the parts.
2. **FP8 KV-cache is not usable out-of-the-box** - uncalibrated `fp8_e4m3` produces degenerate output (0% accuracy) and four separate remediation paths all failed on vLLM 0.11.0; it needs offline scale calibration. A "nearly free" knob with a real hidden setup cost.
3. **AWQ-INT4 + FP8-KV cannot be composed at all** in vLLM 0.11.0 (AWQ needs fp16 activations, FP8-KV attention needs bf16) - the "all-three-knobs" config is literally un-runnable.

**Cost inversion:** on a memory-rich GPU, **AWQ-INT4 made $/correct *worse*** (the int4 dequant path ran ~8× slower than fp16); **budget capping** was the real cost lever (~10× cheaper $/correct on MATH). Browse the whole Pareto frontier in the [live dashboard](https://costperthought.streamlit.app/).

## Reproduce / run

### See the results now (no GPU needed)

The dashboard reads the committed `results/summary.jsonl`, so it runs anywhere:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Or just open the hosted version: **https://costperthought.streamlit.app/**

### Re-run the full sweep (needs a GPU + vLLM)

```bash
pip install -r requirements.txt          # includes vllm; install on the GPU host
python benchmarks/prepare_math.py --n 150
python benchmarks/prepare_gpqa.py --n 100   # GPQA is gated on Hugging Face; see disclosures below
python benchmarks/prepare_code.py --n 100
scripts/run_baseline.sh                   # Phase 2: unlimited-budget reference run (calibrates budgets)
scripts/run_sweep.sh                      # Phase 3: full factorial sweep
```

On a Slurm cluster, the same pipeline is `slurm/setup.sbatch` → `slurm/smoke.sbatch` → `slurm/calibrate.sbatch` → `slurm/sweep.sbatch`.

## Repository guide (for judges)

| Path | What it contains | What it demonstrates |
| --- | --- | --- |
| [docs/writeup.md](docs/writeup.md) | The ~1000-word write-up | Problem, approach, evidence, limits, next steps |
| [harness/serve.py](harness/serve.py) | Per-cell vLLM engine/config builder | Encodes the dtype constraints that make AWQ+FP8 un-runnable (Finding 3) |
| [harness/budget_forcing.py](harness/budget_forcing.py) | Decode-time reasoning-budget cap | The "budget forcing" mechanism (s1-style), the dominant cost lever |
| [harness/run_benchmark.py](harness/run_benchmark.py) / [harness/run_group.py](harness/run_group.py) | Orchestration: serve → query → grade → log cost | How a full cell is measured; grouped by engine to stay under the scheduler's job limit |
| [harness/cost.py](harness/cost.py) | Latency/throughput → $/query, $/correct | The real-money cost model (not a FLOPs estimate) |
| [analysis/interaction_test.py](analysis/interaction_test.py) | Bootstrap CIs + joint-vs-marginal test | The statistical backing for Finding 1 (additive composition) |
| [benchmarks/prepare_*.py](benchmarks/) + [benchmarks/graders/*.py](benchmarks/graders/) | Dataset sampling + per-domain auto-grading | How math/GPQA/code are sampled and scored |
| [configs/grid.yaml](configs/grid.yaml) | Full sweep definition + cost basis | Every config, budget, and the $/GPU-hr basis in one place |
| [results/summary.jsonl](results/summary.jsonl) + [results/interaction__*.json](results/) | All 36 measured cells + interaction tests | The real, committed evidence behind every claim |
| [dashboard/app.py](dashboard/app.py) | Streamlit Pareto-frontier explorer | The reusable infra artifact (live link above) |
| [slurm/](slurm/) | setup / smoke / calibrate / sweep jobs | The exact Slurm pipeline used to produce the data |

## Why DeepSeek-R1-Distill-Qwen-7B

- Maintained AWQ-INT4 checkpoint available (`jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ`) - the quantization knob is a real, production-supported config, not a hand-rolled reimplementation.
- Native vLLM FP8 KV-cache support on recent datacenter GPUs - same story for the KV-cache knob.
- Used as the reference subject in published KV-cache-compression-for-reasoning work (e.g. PM-KVQ), so baseline numbers here can be sanity-checked against a published paper.
- 7B fits comfortably on a single datacenter GPU with room for concurrent-request batching, keeping the full grid's GPU-hour cost bounded.

## What existed before / what's new here

Building on, not duplicating: weight quantization degradation studies (AWQ/GPTQ benchmarking literature), KV-cache compression methods for reasoning models (ThinKV, Kara, PM-KVQ), and adaptive test-time-compute budget research (the "Reasoning on a Budget" survey, budget forcing from s1). Each of those studies one knob against a clean baseline. This project's delta: the **joint** measurement of all three knobs together, priced in **real $ per correct answer** on rented GPU hardware rather than an isolated accuracy table.
