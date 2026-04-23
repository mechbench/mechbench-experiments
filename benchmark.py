"""Latency benchmarks for Gemma 4 E4B on this hardware.

Measures the cost of:
  1. A bare Model.run forward pass (no captures, no hooks) — the floor.
  2. Model.run with Capture.residual at all 42 layers — the typical
     analysis cost (used by step_01, step_08, step_10, step_11, step_12,
     step_13).
  3. Model.run with Capture.attn_weights at the 7 global layers — the
     cost of the manual-attention path needed for step_05/06.

Run from project root:
    python benchmark.py
"""

import time

import mlx.core as mx

from mechbench_core import (
    Capture, GLOBAL_LAYERS, Model, N_LAYERS,
)

PROMPTS = [
    ("short", "What is the capital of France?"),
    ("medium",
     "Explain how a transformer model processes a sentence, step by step."),
    ("long",
     "Write a detailed comparison of Python and Rust for systems programming, "
     "covering performance, safety, ecosystem, learning curve, and use cases. "
     "Give concrete examples for each point."),
]


def _time(fn, n_runs: int = 5) -> float:
    """Run fn n_runs times after a warmup; return mean latency in seconds."""
    fn()  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        # If fn returns something with logits, eval it; otherwise nothing to do.
        logits = getattr(result, "logits", None)
        if logits is not None:
            mx.eval(logits)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_section(title: str, model: Model, intervention_factory) -> None:
    """intervention_factory: () -> list[Intervention] (or [] for bare run)."""
    print("=" * 70)
    print(title)
    print("=" * 70)

    for label, prompt in PROMPTS:
        ids = model.tokenize(prompt)
        n_tokens = int(ids.shape[1])
        ivs = intervention_factory()
        latency = _time(lambda: model.run(ids, interventions=ivs))
        print(f"\n  [{label}] {n_tokens} input tokens")
        print(f"    Latency:        {latency * 1000:>8.1f} ms")
        print(f"    Throughput:     {n_tokens / latency:>8.1f} tok/s (input)")


def bench_overhead(model: Model) -> None:
    """Compare bare run vs full-residual capture vs attn-weights capture on
    the medium prompt; print absolute and relative overheads."""
    print("\n" + "=" * 70)
    print("Overhead comparison (medium prompt, 5 runs each)")
    print("=" * 70)

    _, prompt = PROMPTS[1]
    ids = model.tokenize(prompt)

    bare = _time(lambda: model.run(ids))
    resid = _time(lambda: model.run(
        ids, interventions=[Capture.residual(layers=range(N_LAYERS))]))
    attn = _time(lambda: model.run(
        ids, interventions=[Capture.attn_weights(layers=list(GLOBAL_LAYERS))]))

    print(f"\n  Bare Model.run:                    {bare * 1000:>8.1f} ms")
    print(f"  + Capture.residual(all layers):    {resid * 1000:>8.1f} ms"
          f"  ({(resid - bare) / bare * 100:>+5.1f}%)")
    print(f"  + Capture.attn_weights(globals):   {attn * 1000:>8.1f} ms"
          f"  ({(attn - bare) / bare * 100:>+5.1f}%)  [forces manual attn]")


if __name__ == "__main__":
    print("Loading model...")
    model = Model.load()

    bench_section(
        "Bare Model.run (no instrumentation)",
        model, lambda: [],
    )
    bench_section(
        "Model.run + Capture.residual(all 42 layers)",
        model, lambda: [Capture.residual(layers=range(N_LAYERS))],
    )
    bench_section(
        "Model.run + Capture.attn_weights(7 global layers, manual attn path)",
        model, lambda: [Capture.attn_weights(layers=list(GLOBAL_LAYERS))],
    )
    bench_overhead(model)

    print("\n" + "=" * 70)
    print("Done.")
