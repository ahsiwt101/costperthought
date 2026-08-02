"""
The core statistical question of the project: does combining weight
quantization + KV-cache quantization + a tight reasoning budget degrade
accuracy MORE than the sum of the three knobs' individual (marginal)
degradations would predict?

Method:
  Let A(fp16, fp16, unlimited) = baseline accuracy for a benchmark.
  Marginal loss of knob X = A(baseline) - A(baseline with only X changed).
  Predicted additive accuracy under all-3-changed = A(baseline) - sum(marginal losses).
  Observed accuracy under all-3-changed = measured.
  Interaction effect = Observed - Predicted.
    Interaction < 0 (beyond noise): degradation compounds super-additively --
      the "quietly gets worse" hypothesis.
    Interaction ~= 0: knobs are independent, safe to compose.
    Interaction > 0: knobs partially cancel (would itself be a notable, if
      surprising, finding -- report it, don't bury it).

Significance is assessed via bootstrap resampling of the per-problem
correctness vectors (not a normal-approximation test), since per-cell N is
modest (~100-150) and correctness is binary.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load_results(results_dir: str, benchmark: str):
    """Returns {cell_id: np.array of 0/1 correctness}, keyed by (weight, kv, budget)."""
    by_cell = defaultdict(list)
    for fname in os.listdir(results_dir):
        if not fname.startswith(f"{benchmark}__") or not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            for line in f:
                r = json.loads(line)
                key = (r["weight_precision"], r["kv_cache_precision"], r["budget"])
                by_cell[key].append(int(r["correct"]))
    return {k: np.array(v) for k, v in by_cell.items()}


def bootstrap_mean_ci(arr: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# Which grid coordinate each knob moves the (weight, kv, budget) baseline to.
BASELINE = ("fp16", "fp16", "unlimited")
KNOB_LABELS = {"weight": "weight-quant (AWQ-INT4)", "kv": "kv-quant (FP8)", "budget": "tight budget"}


def _apply(knob: str, tight_budget_name: str):
    """Return the single-coordinate change a knob makes to the baseline cell."""
    return {
        "weight": ("awq_int4", "fp16", "unlimited"),
        "kv": ("fp16", "fp8", "unlimited"),
        "budget": ("fp16", "fp16", tight_budget_name),
    }[knob]


def _both(k1: str, k2: str, tight_budget_name: str):
    """The cell with BOTH knobs' changes applied to the baseline."""
    cell = list(BASELINE)
    for k in (k1, k2):
        c = _apply(k, tight_budget_name)
        idx = {"weight": 0, "kv": 1, "budget": 2}[k]
        cell[idx] = c[idx]
    return tuple(cell)


def pairwise_interaction(cells, benchmark, k1, k2, tight_budget_name="budget_25pct"):
    """Two-way interaction of knobs k1,k2: observed(both) - predicted-additive,
    with a bootstrap 95% CI. Negative & CI-excludes-0 => super-additive (compounds)."""
    base = cells.get(BASELINE)
    a1 = cells.get(_apply(k1, tight_budget_name))
    a2 = cells.get(_apply(k2, tight_budget_name))
    both = cells.get(_both(k1, k2, tight_budget_name))
    if any(v is None for v in (base, a1, a2, both)):
        return None

    def interaction_of(b, x, y, xy):
        pred = b.mean() - ((b.mean() - x.mean()) + (b.mean() - y.mean()))
        return xy.mean() - pred

    obs = interaction_of(base, a1, a2, both)
    rng = np.random.default_rng(42)
    boots = np.array([
        interaction_of(
            rng.choice(base, len(base), True), rng.choice(a1, len(a1), True),
            rng.choice(a2, len(a2), True), rng.choice(both, len(both), True))
        for _ in range(10000)
    ])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    sig = not (ci_lo <= 0 <= ci_hi)
    return {
        "pair": f"{k1} x {k2}",
        "pair_label": f"{KNOB_LABELS[k1]} x {KNOB_LABELS[k2]}",
        "acc_baseline": float(base.mean()),
        f"acc_only_{k1}": float(a1.mean()),
        f"acc_only_{k2}": float(a2.mean()),
        "acc_both": float(both.mean()),
        "interaction_effect": float(obs),
        "interaction_95ci": [float(ci_lo), float(ci_hi)],
        "significant": bool(sig),
        "interpretation": (
            "Super-additive: stacking degrades MORE than the sum of parts" if sig and obs < 0
            else "Sub-additive: knobs partially cancel" if sig and obs > 0
            else "No significant interaction -- knobs compose independently"
        ),
    }


def interaction_analysis(results_dir: str, benchmark: str, tight_budget_name: str = "budget_25pct"):
    cells = load_results(results_dir, benchmark)
    if cells.get(BASELINE) is None:
        print(f"[interaction] no baseline cell for {benchmark} yet.")
        return None

    # weight x kv requires the AWQ x FP8 engine, which vLLM 0.11.0 cannot build
    # (AWQ needs fp16 activations, FP8-KV attention needs bf16) -- unmeasurable.
    pairs = [("weight", "budget"), ("kv", "budget")]
    results = {}
    for k1, k2 in pairs:
        r = pairwise_interaction(cells, benchmark, k1, k2, tight_budget_name)
        if r:
            results[f"{k1}_x_{k2}"] = r

    return {
        "benchmark": benchmark,
        "tight_budget": tight_budget_name,
        "pairwise_interactions": results,
        "weight_x_kv": "NOT MEASURABLE -- AWQ-INT4 + FP8-KV cannot be composed in vLLM 0.11.0 "
                       "(dtype conflict: AWQ requires fp16, FP8-KV attention requires bf16). "
                       "This is itself a finding: two 'individually safe' knobs that don't compose.",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--benchmark", required=True, choices=["math", "gpqa", "code"])
    args = ap.parse_args()

    res = interaction_analysis(args.results_dir, args.benchmark)
    if res:
        print(json.dumps(res, indent=2))
        out_path = os.path.join(args.results_dir, f"interaction__{args.benchmark}.json")
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote {out_path}")
