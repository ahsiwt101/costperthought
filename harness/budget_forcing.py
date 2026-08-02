"""
s1-style budget forcing for DeepSeek-R1-Distill reasoning traces.

DeepSeek-R1-Distill models wrap their reasoning in <think>...</think> before
the final answer. Budget forcing caps how many tokens the model is allowed
to spend inside <think> before it is cut off and forced to answer:

  1. Generate up to `max_think_tokens` tokens, stopping early if the model
     naturally emits "</think>" on its own (unlimited-budget / easy cases).
  2. If the budget is exhausted and the model is still inside <think>,
     forcibly truncate and append a closing "</think>" plus a short
     answer-eliciting suffix, then run a second, short generation for the
     final answer only.

This is a decode-time control, not a training-time intervention -- matches
how budget forcing is used in practice (s1: Muennighoff et al.) and how we
want to measure it: as a knob any team can flip today.

Requires a GPU host (vllm import deferred).
"""
import time
from dataclasses import dataclass, field


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
FORCE_SUFFIX = (
    "\n\n(Time is up. Based on the reasoning above, give your final answer now.)\n"
    + THINK_CLOSE + "\n"
)


@dataclass
class GenerationResult:
    text: str
    think_tokens_used: int
    answer_tokens_used: int
    total_tokens: int
    wall_clock_seconds: float
    budget_forced: bool  # True if the think phase was cut off by the budget
    prompt: str = field(repr=False, default="")


def generate_with_budget(
    llm,
    tokenizer,
    prompt: str,
    sampling_params_cls,
    max_think_tokens: int | None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    answer_max_tokens: int = 1024,
    total_hard_ceiling: int = 8192,
) -> GenerationResult:
    """
    max_think_tokens=None -> unlimited budget: generate straight through up
    to `total_hard_ceiling`, no forcing. This is the baseline / calibration
    path (Phase 2) that establishes the natural trace-length distribution.
    """
    from vllm import SamplingParams  # local import, GPU host only

    t0 = time.time()

    if max_think_tokens is None:
        sp = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=total_hard_ceiling,
            stop=None,
        )
        out = llm.generate([prompt], sp)[0].outputs[0]
        n_tokens = len(out.token_ids)
        return GenerationResult(
            text=out.text,
            think_tokens_used=n_tokens,  # not separated in unlimited mode; see analysis note below
            answer_tokens_used=0,
            total_tokens=n_tokens,
            wall_clock_seconds=time.time() - t0,
            budget_forced=False,
            prompt=prompt,
        )

    # --- Phase 1: budgeted think pass, stop early on natural </think> ---
    sp_think = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_think_tokens,
        stop=[THINK_CLOSE],
    )
    think_out = llm.generate([prompt], sp_think)[0].outputs[0]
    think_text = think_out.text
    think_tokens = len(think_out.token_ids)
    naturally_closed = THINK_CLOSE in think_text or think_out.finish_reason == "stop"

    if naturally_closed:
        # model finished thinking within budget on its own -- continue to the
        # answer with the model's normal flow (no forcing needed).
        full_prompt = prompt + think_text
        forced = False
    else:
        # budget exhausted mid-thought -- forcibly cut and inject the closer.
        full_prompt = prompt + think_text + FORCE_SUFFIX
        forced = True

    # --- Phase 2: short answer-only pass ---
    sp_answer = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=answer_max_tokens,
    )
    answer_out = llm.generate([full_prompt], sp_answer)[0].outputs[0]
    answer_tokens = len(answer_out.token_ids)

    return GenerationResult(
        text=think_text + (FORCE_SUFFIX if forced else "") + answer_out.text,
        think_tokens_used=think_tokens,
        answer_tokens_used=answer_tokens,
        total_tokens=think_tokens + answer_tokens,
        wall_clock_seconds=time.time() - t0,
        budget_forced=forced,
        prompt=prompt,
    )


def generate_batch_with_budget(
    llm,
    prompts: list[str],
    sampling_params_cls,
    max_think_tokens: int | None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    answer_max_tokens: int = 1024,
    total_hard_ceiling: int = 16384,
) -> list[GenerationResult]:
    """
    Batched version of generate_with_budget: submits ALL prompts to vLLM in one
    generate() call so continuous batching runs them concurrently (10-100x
    faster than the per-prompt loop). Each result's wall_clock_seconds is the
    *amortized* per-query wall-clock (total batch wall / n), which is the
    correct basis for the $/query cost model under continuous batching.
    """
    from vllm import SamplingParams  # local import, GPU host only

    n = len(prompts)
    t0 = time.time()

    if max_think_tokens is None:
        # --- unlimited / calibration: single batched pass ---
        sp = SamplingParams(temperature=temperature, top_p=top_p,
                            max_tokens=total_hard_ceiling, stop=None)
        outs = llm.generate(prompts, sp)
        batch_wall = time.time() - t0
        per = batch_wall / max(n, 1)
        results = []
        for o in outs:
            out = o.outputs[0]
            ntok = len(out.token_ids)
            results.append(GenerationResult(
                text=out.text, think_tokens_used=ntok, answer_tokens_used=0,
                total_tokens=ntok, wall_clock_seconds=per, budget_forced=False,
                prompt=o.prompt or "",
            ))
        return results

    # --- budgeted: phase 1 (batched think pass, stop on natural </think>) ---
    sp_think = SamplingParams(temperature=temperature, top_p=top_p,
                              max_tokens=max_think_tokens, stop=[THINK_CLOSE])
    think_outs = llm.generate(prompts, sp_think)

    think_texts, think_toks, forced_flags, full_prompts = [], [], [], []
    for prompt, o in zip(prompts, think_outs):
        out = o.outputs[0]
        ttext = out.text
        naturally_closed = THINK_CLOSE in ttext or out.finish_reason == "stop"
        think_texts.append(ttext)
        think_toks.append(len(out.token_ids))
        forced_flags.append(not naturally_closed)
        full_prompts.append(prompt + ttext + (FORCE_SUFFIX if not naturally_closed else ""))

    # --- phase 2 (batched answer-only pass) ---
    sp_answer = SamplingParams(temperature=temperature, top_p=top_p,
                               max_tokens=answer_max_tokens)
    answer_outs = llm.generate(full_prompts, sp_answer)

    batch_wall = time.time() - t0
    per = batch_wall / max(n, 1)
    results = []
    for ttext, ttok, forced, a in zip(think_texts, think_toks, forced_flags, answer_outs):
        aout = a.outputs[0]
        atok = len(aout.token_ids)
        results.append(GenerationResult(
            text=ttext + (FORCE_SUFFIX if forced else "") + aout.text,
            think_tokens_used=ttok, answer_tokens_used=atok,
            total_tokens=ttok + atok, wall_clock_seconds=per,
            budget_forced=forced, prompt="",
        ))
    return results


# NOTE for Phase 2 (baseline calibration): run with max_think_tokens=None across
# each benchmark's full sample, record think_tokens_used per problem, then set
# budget_75pct/50pct/25pct in configs/grid.yaml to 0.75/0.50/0.25 * median(think
# _tokens_used) for that benchmark. Budgets are calibrated per-benchmark, not
# globally, since math vs. code traces have very different natural lengths.
