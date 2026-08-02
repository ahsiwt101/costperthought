# CostPerThought — does stacking "safe" inference-efficiency knobs actually work?

LaunchPad 2026 AI Challenge submission · Track: Infrastructure & tooling
Live dashboard: costperthought.streamlit.app · Model: DeepSeek-R1-Distill-Qwen-7B, served via vLLM 0.11.0 on a datacenter GPU, priced at a market on-demand reference rate ($2.40/GPU-hr) so $/correct-answer stays portable. All numbers below are measured, not estimated.

## Problem

Reasoning models are expensive to serve: long chain-of-thought generates far more tokens per query than instruct models. Teams cut this cost with three knobs, each published as "nearly free" when validated in isolation — weight quantization (AWQ-INT4), KV-cache quantization (FP8), and reasoning-token budget caps ("budget forcing," s1-style). In production, all three get switched on together, on the assumption that independently-safe knobs compose safely. No one publishes the joint measurement, priced in real dollars per correct answer rather than an isolated accuracy table. We built a full-factorial harness to measure it, with success defined up front as accuracy, latency, and $/correct-answer per cell, plus an explicit test for whether the three-way combination degrades more than the sum of its parts predicts.

The finding that emerged wasn't the one the study was designed to catch: two of the three knobs are substantially harder to even turn on than the literature implies — FP8 KV-cache produces degenerate output without offline calibration, and AWQ-INT4 + FP8-KV cannot be composed at all in current vLLM. For the one pair we could cleanly measure (weight-quant × budget), degradation is additive. But the cost story inverts a common assumption about which knob actually saves money.

## Approach

Full factorial harness over {fp16, AWQ-INT4} weights × {fp16, FP8} KV-cache × {unlimited, 75%, 50%, 25% budget} × {MATH-500, GPQA-diamond, LiveCodeBench}, served on real GPU hardware with continuous batching. Budget caps are calibrated per benchmark to 75/50/25% of the median unlimited think-length (MATH ≈910 tokens, GPQA ≈6,300, code ≈5,440 tokens) since trace lengths differ roughly 6× across domains — a single global cap would over-constrain one domain and do nothing to another.

Alternatives ruled out: inventing a new compression method (the KV-cache-compression space is already crowded with 2026 work — ThinKV, Kara, PM-KVQ — beating SOTA there in the time available was low-probability and off-thesis); studying this via API-only closed models (can't observe or control KV-cache precision, so the core question is unanswerable); simulating cost from FLOPs instead of measuring wall-clock on rented hardware (would gut the Constraints pillar, the point of the project). Production-ready techniques over a new algorithm, deliberately — an infra/measurement contribution, not a new-method claim. $/correct-answer = (GPU $/hr ÷ 3600) × measured amortized per-query wall-clock ÷ accuracy, per cell from batched throughput. Baseline accuracies land where expected for a 7B R1-distill model (MATH 80.7%, GPQA 47.0%, code 42.0%), cross-checking the harness before trusting the sweep.

Standing the harness up was itself a constraint worth naming: the GPU nodes run a CUDA 12.9 driver, but current vLLM (0.26.0) ships a CUDA-13 wheel that won't load, so we pinned vLLM 0.11.0 (newest CUDA-12.8 build) and rebuilt the environment after an in-place torch swap left a mismatched library tree; `transformers` needed pinning `<5` for a tokenizer API 0.11.0 depends on — the same "nearly free" hidden-setup-tax pattern Evidence documents for FP8.

## Evidence

**Weight-quant × budget compose additively.** Bootstrap interaction test (10k resamples, per-problem correctness) finds no significant two-way interaction on any benchmark: MATH +0.033 (95% CI −0.11 to +0.17), GPQA −0.06 (−0.26 to +0.13), code −0.03 (−0.22 to +0.16). All three CIs span zero. Stacking AWQ-INT4 with a tight budget costs about the sum of their individual accuracy losses — no hidden compounding, at least for this pair.

**FP8 KV-cache is not usable out of the box.** With uncalibrated `fp8_e4m3`, the model emits degenerate repetition (e.g. "so so so implode…") and accuracy collapses to 0–9% across every benchmark and budget level. Four remediation paths were tried and all failed under vLLM 0.11.0: `fp8_e5m2` fails `torch.compile`; runtime scale calculation hits a Dynamo data-dependent-branch error; the eager-mode fallback hits an internal `AttributeError`. Getting FP8-KV to actually work requires offline scale calibration that most teams citing "nearly free" skip — a real, hidden setup cost the literature doesn't price in.

**AWQ-INT4 + FP8-KV cannot be composed at all.** AWQ requires fp16 activations; FP8-KV attention requires bf16 — mutually exclusive in vLLM 0.11.0. The "all-three-knobs" config is literally un-runnable, itself the sharpest possible answer to "do these knobs compose": for one pair, no, because you can't even build it.

## Constraints ($/correct-answer)

The cost data inverts a common assumption. **Budget capping is the dominant $/correct lever**: on MATH, fp16-unlimited costs $0.00084/correct at 80.7% accuracy, while fp16 at a 25% budget costs $0.00008/correct at 63.3% — roughly 10× cheaper per correct answer for an 18-point accuracy trade. **AWQ-INT4 made $/correct worse here, not better.** On this GPU the INT4 dequant path ran about 8× slower than fp16 (MATH-unlimited: 7.89s vs 1.02s amortized per query), so despite halving weight memory, AWQ raised $/correct on every benchmark and every budget level (e.g. GPQA-unlimited: $0.0154/correct AWQ vs $0.0028/correct fp16). Weight quantization is a memory-pressure tool here, not a cost tool, on hardware that isn't memory-bound. Across all 24 runnable cells, the cheapest configuration hitting any given accuracy bar is always an fp16-weight, tight-budget cell — never an AWQ one. Full Pareto frontier: costperthought.streamlit.app.

## Honesty & Trajectory

Limitations, stated plainly: single 7B model family; N=100–150 problems per benchmark, giving wide-ish CIs — why the additive-interaction result reads as "no significant compounding detected," not "proven zero effect." Single GPU type, so the AWQ-slower-than-fp16 result may not transfer to memory-constrained hardware where AWQ should win. A small confound: AWQ forces fp16 activations while the unquantized path runs bf16, since AWQ forbids bf16 in vLLM 0.11.0 — noted, not hidden. FP8-KV and AWQ+FP8 results are framework-version-specific and could change with a calibrated FP8 checkpoint or a newer vLLM release.

With two more weeks: run FP8-KV with offline-calibrated scales to measure it properly instead of reporting only that it breaks; add a second model family to test whether the additive-interaction finding generalizes; and rerun the AWQ comparison on a memory-constrained GPU to see whether the cost inversion is a hardware artifact or a general result. The reusable infra — the Slurm sweep harness, per-cell cost logger, and Pareto dashboard — is model-agnostic and built to make all three follow-ups a config change, not a rewrite.

## Appendices

- `configs/grid.yaml` — full sweep grid definition; `results/summary.jsonl` — all 36 measured cells
- `results/interaction__{math,gpqa,code}.json` — full bootstrap interaction-test output
- Sample degenerate FP8 transcript, documenting the calibration requirement
- Cost-model derivation and per-cell amortized-throughput methodology (`harness/cost.py`)

### Repository guide

Start at `README.md`. Key files: `harness/serve.py` (per-cell vLLM engine builder — encodes the dtype conflict behind the AWQ+FP8 finding), `harness/budget_forcing.py` (the decode-time budget-cap mechanism), `harness/run_benchmark.py` / `harness/run_group.py` (serve → query → grade → log cost), `harness/cost.py` ($/query, $/correct-answer), `analysis/interaction_test.py` (bootstrap CIs), `benchmarks/prepare_*.py` + `benchmarks/graders/*.py` (dataset prep and grading), `dashboard/app.py` (Streamlit explorer, live at costperthought.streamlit.app).

To reproduce: `pip install -r requirements.txt`, prep datasets, then `scripts/run_baseline.sh` and `scripts/run_sweep.sh` (or the `slurm/` jobs) on a datacenter GPU with vLLM. No GPU needed to explore results — `pip install -r dashboard/requirements.txt && streamlit run dashboard/app.py`, or use the live link. All 36 result cells are real measured runs, committed in `results/`; no keys or personal data are committed (HF token read from a gitignored `~/.hf_token`).
