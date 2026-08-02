"""
Builds a vLLM `LLM` engine for a given grid cell (weight precision x KV-cache
precision). Using vLLM's offline Python API (not the HTTP server) so we get
direct control over generation for budget forcing, and precise, uncontended
wall-clock timing for the cost model.

Run on the GPU host only. Everything else in this repo can be developed and
unit-tested without a GPU.
"""
import argparse
import time
import yaml


def load_grid(config_path="configs/grid.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_engine(weight_precision: str, kv_cache_precision: str, grid: dict):
    """
    weight_precision: one of grid['weight_precision'][i]['name'], e.g. "fp16" | "awq_int4"
    kv_cache_precision: one of grid['kv_cache_precision'][i]['name'], e.g. "fp16" | "fp8"

    Returns a vllm.LLM instance configured for this grid cell.
    Import of vllm is deferred so this module is importable on non-GPU hosts
    (e.g. for the dashboard / analysis code, which never construct an engine).
    """
    from vllm import LLM  # noqa: import here, GPU-host only

    wp = next(w for w in grid["weight_precision"] if w["name"] == weight_precision)
    kvp = next(k for k in grid["kv_cache_precision"] if k["name"] == kv_cache_precision)

    engine_kwargs = dict(
        model=wp["hf_id"],
        quantization=wp["quantization"],
        kv_cache_dtype=kvp["vllm_kv_cache_dtype"],
        max_model_len=grid["model"]["max_context"],
        # Per-config activation dtype, forced by two hard kernel constraints in
        # vLLM 0.11.0: FP8 KV-cache attention REQUIRES bf16 output, while the AWQ
        # kernel REQUIRES fp16. These two are mutually exclusive -> the AWQ x FP8
        # engine cannot be built at all (documented incompatibility). For the 3
        # buildable engines we pick the dtype each requires:
        #   fp8 KV   -> bfloat16   (fp16-weight cells; also matches calibration)
        #   awq wts  -> float16    (awq + fp16 KV)
        #   else     -> bfloat16
        dtype=("bfloat16" if kv_cache_precision == "fp8"
               else "float16" if weight_precision == "awq_int4"
               else "bfloat16"),
        # 0.80 (not 0.90): the shared GPU nodes often have ~10GB in use by other
        # tenants at startup, and 0.90 tips over the free-memory check. 0.80 still
        # leaves ~60GB for the KV cache (millions of tokens) on a 7B model.
        gpu_memory_utilization=0.80,
        # continuous batching is on by default in vLLM; concurrency is
        # controlled at the call site via batched generate() calls.
    )
    # vLLM expects None (not "null"/"none") for unquantized weights.
    if engine_kwargs["quantization"] is None:
        engine_kwargs.pop("quantization")

    print(f"[serve] building LLM engine: {engine_kwargs}")
    t0 = time.time()
    llm = LLM(**engine_kwargs)
    print(f"[serve] engine ready in {time.time() - t0:.1f}s")
    return llm


def cell_id(weight_precision: str, kv_cache_precision: str, budget_name: str) -> str:
    """Canonical string ID for a grid cell, used in result file names."""
    return f"w-{weight_precision}_kv-{kv_cache_precision}_budget-{budget_name}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sanity-print a grid cell's launch config (no GPU needed).")
    ap.add_argument("--weight_precision", default="fp16")
    ap.add_argument("--kv_cache_precision", default="fp16")
    ap.add_argument("--config", default="configs/grid.yaml")
    args = ap.parse_args()

    grid = load_grid(args.config)
    wp = next(w for w in grid["weight_precision"] if w["name"] == args.weight_precision)
    kvp = next(k for k in grid["kv_cache_precision"] if k["name"] == args.kv_cache_precision)
    print("Would launch with:")
    print(f"  model            = {wp['hf_id']}")
    print(f"  quantization     = {wp['quantization']}")
    print(f"  kv_cache_dtype   = {kvp['vllm_kv_cache_dtype']}")
    print(f"  max_model_len    = {grid['model']['max_context']}")
