"""
Converts measured latency/throughput into dollar figures. Pure functions,
no GPU/vLLM dependency -- unit-testable and reusable by the dashboard.
"""
from dataclasses import dataclass


@dataclass
class CostResult:
    usd_per_query: float
    usd_per_correct_answer: float | None  # None if accuracy == 0 (undefined/infinite)


def usd_per_query(wall_clock_seconds: float, usd_per_gpu_hour: float) -> float:
    """
    Cost to serve one query, assuming the query held exclusive use of the GPU
    for its wall-clock duration. This is the conservative (worst-case) framing;
    under real continuous batching with concurrent requests, effective
    per-query cost is lower. See note in run_benchmark.py on how we measure
    *batched* throughput to correct for this rather than relying on
    single-request wall-clock alone.
    """
    return (usd_per_gpu_hour / 3600.0) * wall_clock_seconds


def usd_per_correct_answer(mean_usd_per_query: float, accuracy_fraction: float) -> float | None:
    """
    Expected $ spent to obtain one correct answer, given a config's mean
    per-query cost and its accuracy on a benchmark:

        $/correct = $/query / accuracy

    e.g. $0.01/query at 80% accuracy -> $0.0125 expected $ per correct answer
    (you'd need to run ~1.25 queries on average to net one correct one, in
    an idealized retry-until-correct framing -- reported as a comparable
    unit across configs, not a literal retry policy recommendation).
    """
    if accuracy_fraction <= 0:
        return None
    return mean_usd_per_query / accuracy_fraction


def compute_cell_cost(mean_wall_clock_seconds: float, accuracy_fraction: float, usd_per_gpu_hour: float) -> CostResult:
    upq = usd_per_query(mean_wall_clock_seconds, usd_per_gpu_hour)
    upc = usd_per_correct_answer(upq, accuracy_fraction)
    return CostResult(usd_per_query=upq, usd_per_correct_answer=upc)


if __name__ == "__main__":
    # quick sanity check
    r = compute_cell_cost(mean_wall_clock_seconds=4.2, accuracy_fraction=0.78, usd_per_gpu_hour=2.40)
    print(r)
