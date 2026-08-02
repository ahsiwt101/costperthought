# CostPerThought - do "safe" inference-efficiency knobs actually compose?

LaunchPad 2026 AI Challenge · Track: Infrastructure & tooling
Live dashboard: https://costperthought.streamlit.app/ · Model: DeepSeek-R1-Distill-Qwen-7B · Compute: a single datacenter GPU, priced at a $2.40/GPU-hr market reference so the $/correct-answer metric stays portable. Every number below is measured, not estimated.

## Short summary

Teams serving reasoning models cut inference cost with three knobs - weight quantization, KV-cache quantization, and reasoning-token budget caps - each published as "nearly free" in isolation, then switch all three on at once in production. CostPerThought runs the full factorial of those knobs *together* on real hardware and prices every configuration in dollars per correct answer. It is for infra and platform teams who stack these optimizations on faith. It matters because the joint behavior is not what the isolated papers imply: two of the three knobs are much harder to even enable than advertised, and the "pure memory" knob can make cost worse.

## Problem and approach

Reasoning models are expensive to serve: long chain-of-thought means far more generated tokens per query than an instruct model. Three optimizations promise relief, each benchmarked alone against a clean baseline: weight quantization (FP16 → AWQ-INT4), KV-cache quantization (FP16 → FP8), and reasoning-token budget capping (s1-style "budget forcing" that truncates the thinking phase). In production all three are enabled simultaneously, on the assumption that individually-safe knobs compose. Nobody publishes the joint measurement priced in real money.

We built a full factorial harness to do exactly that: {FP16, AWQ-INT4} weights × {FP16, FP8} KV-cache × {unlimited, 75%, 50%, 25%} reasoning budget × {MATH-500, GPQA-diamond, LiveCodeBench} - three domains chosen because their natural reasoning-trace lengths differ roughly 6×. We served DeepSeek-R1-Distill-Qwen-7B through vLLM 0.11.0 with continuous batching. Cost is computed per cell as (market GPU $/hr ÷ 3600) × measured amortized per-query wall-clock ÷ accuracy - a real, portable dollar figure, not a FLOPs estimate. We deliberately did **not** invent a new compression method: that space is crowded and hard to beat credibly in a month. The contribution is the measurement itself.

Getting there was mostly an infrastructure fight, and the fight *is* part of the finding. The GPU nodes ran a CUDA 12.9 driver, but current vLLM (0.26.0) ships a CUDA-13 wheel that links `libcudart.so.13` and simply will not load; we pinned vLLM 0.11.0, the newest CUDA-12.8 build, and rebuilt the environment from scratch because an in-place torch cu130→cu128 swap left a mismatched CUDA library tree. `transformers` had to be pinned `<5` (5.x removed a tokenizer API vLLM 0.11.0 relies on). Datasets were staged in a separate isolated venv, and GPQA required accepting a gated Hugging Face license. Budget caps were calibrated *per benchmark* - 75/50/25% of the median unlimited think-length (MATH ≈910 tokens, GPQA ≈6300, code ≈5440) - because a single global cap is meaningless across traces that differ 6× in length.

## Evidence and experiments

Baselines land where a 7B R1 model should - MATH 80.7%, GPQA 47.0%, code 42.0% - which cross-checks the harness. Three findings followed, and the headline was not the one we expected.

**1. For the one knob pair we could cleanly measure, weight-quant × budget compose additively.** A bootstrap interaction test (10k resamples over per-problem correctness) finds no significant two-way interaction on any benchmark: MATH +0.033 (95% CI −0.11 to +0.17), GPQA −0.06 (−0.26 to +0.13), code −0.03 (−0.22 to +0.16). Stacking AWQ-INT4 with a tight budget costs about the sum of the parts - no hidden compounding.

**2. FP8 KV-cache is not usable out of the box.** Uncalibrated `fp8_e4m3` produced degenerate repetition ("...so so so implode...") and 0% accuracy on every benchmark. Four remediation paths all failed on vLLM 0.11.0: `fp8_e5m2` breaks `torch.compile`; runtime scale calculation hits a Dynamo data-dependent-branch error; the eager-mode workaround throws an internal `AttributeError`. It works only with offline scale calibration - a real hidden setup cost behind the "nearly free" label.

**3. AWQ-INT4 + FP8-KV cannot be composed at all.** The AWQ kernel requires FP16 activations; FP8-KV attention requires BF16 - mutually exclusive in vLLM 0.11.0. The "all-three-knobs" configuration is literally un-runnable, which is itself the cleanest possible answer to "do these knobs compose?"

**The cost story then inverted a common assumption.** Budget capping is the dominant $/correct lever: on MATH, FP16-unlimited costs $0.00084/correct at 80.7%, while FP16 at a 25% budget costs $0.00008/correct - about 10× cheaper - at 63.3%. But AWQ-INT4 made $/correct *worse*, not better: on this memory-rich GPU the INT4 dequant path ran ~8× slower than FP16 (MATH-unlimited 7.89 vs 1.02 amortized s/query), so despite halving weight memory it raised cost on every benchmark. Weight quantization is a memory-pressure tool, not a cost tool, on hardware that isn't memory-bound. The whole trade-off is browsable live in the dashboard.

## Constraints, limitations, incomplete areas

This is one 7B model from a single family; N = 100–150 per benchmark, so confidence intervals are honestly wide - which is why the additive result reads as "no significant compounding," not "proven zero." It is one GPU type, so absolute dollar figures won't fully transfer. The weight knob carries a small confound: AWQ forbids BF16, so AWQ cells run FP16 activations while unquantized cells run BF16 - noted, not hidden. The FP8-KV and AWQ+FP8 results are specific to vLLM 0.11.0 and would change with a calibrated FP8 checkpoint or a newer engine. The single most important incomplete area is FP8 itself: we measured that it is broken out of the box, not what it costs when done properly.

## What I would improve next

Run FP8-KV with offline-calibrated scales to measure it fairly rather than only documenting its failure; add a second model family to test how far the additive-interaction result generalizes; and re-test AWQ on a memory-starved GPU - the one setting where weight quantization should finally win on cost. The harness, per-cell cost logger, and dashboard are model-agnostic, so each of these is a config change, not a rewrite.

## Repository guide and how to run

Point of entry is `README.md`. Most important files, by exact path:

- `harness/serve.py` - per-cell vLLM engine builder; encodes the dtype constraints behind Finding 3.
- `harness/budget_forcing.py` - the decode-time budget-cap mechanism (the dominant cost lever).
- `harness/run_benchmark.py`, `harness/run_group.py` - orchestration (serve → query → grade → log cost).
- `harness/cost.py` - latency/throughput → $/query and $/correct-answer.
- `analysis/interaction_test.py` - bootstrap CIs and the interaction test behind Finding 1.
- `benchmarks/prepare_*.py`, `benchmarks/graders/*.py` - dataset sampling and per-domain grading.
- `configs/grid.yaml` - the full grid and cost basis.
- `results/summary.jsonl`, `results/interaction__*.json` - all 36 measured cells and the interaction tests (real, committed data).
- `dashboard/app.py` - the Streamlit Pareto explorer (live: costperthought.streamlit.app).

**Run it:** explore results with no GPU via `pip install -r dashboard/requirements.txt && streamlit run dashboard/app.py`, or use the live link. Reproducing the sweep needs a datacenter GPU with vLLM (`pip install -r requirements.txt`, prep datasets, then `scripts/run_baseline.sh` and `scripts/run_sweep.sh`, or the `slurm/` jobs). All 36 result cells are real measured runs; datasets download from Hugging Face (GPQA gated); no passwords, API keys, or personal data are committed (the HF token is read from a gitignored `~/.hf_token`).
