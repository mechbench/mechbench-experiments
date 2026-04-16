"""Sub-layer ablation: attention vs MLP across all 42 layers of Gemma 4 E4B.

For each layer, independently ablates the attention branch (zero its
contribution; KV cache still populated) and the MLP branch (zero its
contribution). 42 layers x 15 prompts x 2 branches = 1260 forward passes
(~3.5 minutes on M2 Pro).

Key finding (per docs/findings/step_04_sublayer_ablation.md): MLPs dominate
across the network; attention only matters at one specific layer (L23) and
weakly at L17.

Run from project root:
    python experiments/step_04_sublayer_ablation.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Ablate, GLOBAL_LAYERS, Model, N_LAYERS,
)
from experiments.prompts import FACTUAL_15  # noqa: E402

OUT_DIR = ROOT / "caches"


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating prompts...\n")
    valid = FACTUAL_15.validate(model)
    print()

    attn_delta = np.zeros((N_LAYERS, len(valid)), dtype=np.float64)
    mlp_delta = np.zeros((N_LAYERS, len(valid)), dtype=np.float64)

    total = N_LAYERS * len(valid) * 2
    print(f"Running {total} ablated forward passes...")
    t0 = time.perf_counter()

    for i in range(N_LAYERS):
        attn_ablation = Ablate.attention(i)
        mlp_ablation = Ablate.mlp(i)
        for j, vp in enumerate(valid):
            # Attention ablation
            r = model.run(vp.input_ids, interventions=[attn_ablation])
            attn_delta[i, j] = float(_last_logp(r.logits)[vp.target_id]) - vp.baseline_lp
            # MLP ablation
            r = model.run(vp.input_ids, interventions=[mlp_ablation])
            mlp_delta[i, j] = float(_last_logp(r.logits)[vp.target_id]) - vp.baseline_lp

        if (i + 1) % 6 == 0 or i == N_LAYERS - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (N_LAYERS - i - 1)
            print(f"  layer {i:>2}: attn Δ={np.mean(attn_delta[i]):>+7.3f}  "
                  f"mlp Δ={np.mean(mlp_delta[i]):>+7.3f}  "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    total_time = time.perf_counter() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time / total * 1000:.0f}ms per pass)")

    mean_attn = np.mean(attn_delta, axis=1)
    mean_mlp = np.mean(mlp_delta, axis=1)

    # ---- Per-layer table ----
    print(f"\n{'layer':>5}  {'type':>7}  {'attn_Δlogp':>11}  "
          f"{'mlp_Δlogp':>10}  {'dominant':>9}")
    print("-" * 55)
    for i in range(N_LAYERS):
        kind = "GLOBAL" if i in GLOBAL_LAYERS else "local"
        if abs(mean_attn[i]) < 0.01 and abs(mean_mlp[i]) < 0.01:
            dominant = "-"
        elif abs(mean_attn[i]) > abs(mean_mlp[i]):
            dominant = "attn"
        else:
            dominant = "MLP"
        print(f"{i:>5}  {kind:>7}  {mean_attn[i]:>+11.3f}  "
              f"{mean_mlp[i]:>+10.3f}  {dominant:>9}")

    # ---- Summary stats ----
    print(f"\n--- Summary ---")
    for layer_type, indices in [
        ("Global", list(GLOBAL_LAYERS)),
        ("Local", [i for i in range(N_LAYERS) if i not in GLOBAL_LAYERS]),
    ]:
        a = mean_attn[indices]
        m = mean_mlp[indices]
        print(f"  {layer_type:6s} (n={len(indices):>2}): "
              f"attn mean Δ={np.mean(a):>+.3f}, mlp mean Δ={np.mean(m):>+.3f}")

    mid = list(range(10, 25))
    print(f"  Middle (10-24, n={len(mid)}): "
          f"attn mean Δ={np.mean(mean_attn[mid]):>+.3f}, "
          f"mlp mean Δ={np.mean(mean_mlp[mid]):>+.3f}")

    # ---- Plot: side-by-side bars + difference panel ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    layers_x = np.arange(N_LAYERS)
    width = 0.35

    # Top: side-by-side
    ax = axes[0]
    ax.bar(layers_x - width / 2, mean_attn, width, label="attention ablated",
           color="#e74c3c", edgecolor="white", linewidth=0.3)
    ax.bar(layers_x + width / 2, mean_mlp, width, label="MLP ablated",
           color="#3498db", edgecolor="white", linewidth=0.3)
    ax.set_ylabel("mean Δ log p(target)")
    ax.set_title("Sub-layer ablation — Gemma 4 E4B")
    ax.legend(loc="lower left")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for g in GLOBAL_LAYERS:
        ax.axvline(g, color="#999999", linestyle="--", linewidth=0.7, alpha=0.4)

    # Bottom: difference (attn - mlp); negative = attn dominates
    diff = mean_attn - mean_mlp
    colors_diff = ["#e74c3c" if d < 0 else "#3498db" for d in diff]
    ax = axes[1]
    ax.bar(layers_x, diff, color=colors_diff, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("layer index")
    ax.set_ylabel("attn Δ − MLP Δ\n← attn more critical | MLP more critical →")
    ax.set_title("Which branch matters more per layer? (negative = attention dominates)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(0, N_LAYERS, 3))
    ax.grid(True, alpha=0.3, axis="y")
    for g in GLOBAL_LAYERS:
        ax.axvline(g, color="#999999", linestyle="--", linewidth=0.7, alpha=0.4)

    plt.tight_layout()
    out_path = OUT_DIR / "sublayer_ablation.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
