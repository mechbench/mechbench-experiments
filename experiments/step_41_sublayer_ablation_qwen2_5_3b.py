"""Step 41 — sublayer ablation (attention vs MLP) on Qwen 2.5 3B Instruct.

Cross-family port of step_04 / step_36 (000205). For each
FACTUAL_15 prompt, ablate each layer's attention branch and MLP
branch independently. Tests whether the scattered mid-late
activity in step_39's whole-layer-ablation curve is
attention-driven, MLP-driven, or both.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechbench_core import Ablate, Model  # noqa: E402
from experiments.prompts.factual import FACTUAL_15  # noqa: E402


MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-bf16"
MIN_CONFIDENCE = 0.5


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def main() -> None:
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model = Model.load(MODEL_ID)
    print(f"Loaded in {time.time() - t0:.1f}s")

    arch = model.arch
    n_layers = arch.n_layers
    print(f"Qwen 2.5 3B Instruct: {n_layers} layers, n_heads={arch.n_heads}, "
          f"n_kv_heads={arch.n_kv_heads}")

    prompts = FACTUAL_15.prompts
    validated: list[tuple[int, float, mx.array]] = []
    print(f"\nValidating {len(prompts)} prompts...")
    for prompt in prompts:
        ids = model.tokenize(prompt.text)
        result = model.run(ids)
        lp = _last_logp(result.logits)
        top1_id = int(np.argmax(lp))
        top1_prob = float(np.exp(lp[top1_id]))
        keep = top1_prob >= MIN_CONFIDENCE
        marker = "✓" if keep else "✗"
        top1_token = model.tokenizer.decode([top1_id])
        print(f"  {marker} prob={top1_prob:.3f} top1={top1_token!r:>16} "
              f"target={prompt.target!r:>16}")
        if keep:
            validated.append((top1_id, float(lp[top1_id]), ids))

    n = len(validated)
    print(f"\n{n}/{len(prompts)} prompts validated.\n")
    if n == 0:
        return

    attn_delta = np.zeros((n_layers, n), dtype=np.float64)
    mlp_delta = np.zeros((n_layers, n), dtype=np.float64)

    total = n_layers * n * 2
    print(f"Running {total} ablated forward passes...")
    t0 = time.perf_counter()
    for layer_idx in range(n_layers):
        attn_abl = Ablate.attention(layer_idx)
        mlp_abl = Ablate.mlp(layer_idx)
        for j, (top1_id, baseline_lp, ids) in enumerate(validated):
            r = model.run(ids, interventions=[attn_abl])
            attn_delta[layer_idx, j] = (
                float(_last_logp(r.logits)[top1_id]) - baseline_lp
            )
            r = model.run(ids, interventions=[mlp_abl])
            mlp_delta[layer_idx, j] = (
                float(_last_logp(r.logits)[top1_id]) - baseline_lp
            )
        if (layer_idx + 1) % 5 == 0 or layer_idx == n_layers - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
            print(
                f"  layer {layer_idx:>2}: attn Δ={np.mean(attn_delta[layer_idx]):>+7.3f}  "
                f"mlp Δ={np.mean(mlp_delta[layer_idx]):>+7.3f}  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
            )

    mean_attn = np.mean(attn_delta, axis=1)
    mean_mlp = np.mean(mlp_delta, axis=1)

    out_dir = ROOT / "caches"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "step_41_sublayer_ablation_qwen2_5_3b.json"
    out_json.write_text(json.dumps({
        "model_id": MODEL_ID,
        "n_layers": n_layers,
        "n_prompts": n,
        "attn_mean": [round(float(v), 4) for v in mean_attn],
        "attn_median": [round(float(v), 4) for v in np.median(attn_delta, axis=1)],
        "mlp_mean": [round(float(v), 4) for v in mean_mlp],
        "mlp_median": [round(float(v), 4) for v in np.median(mlp_delta, axis=1)],
    }, indent=2))
    print(f"\nWrote {out_json}")

    print("\n--- Per-layer attention vs MLP ---")
    print(f"{'L':>3} {'attn':>8} {'mlp':>8} {'dom':>5}")
    for i in range(n_layers):
        if abs(mean_attn[i]) < 0.01 and abs(mean_mlp[i]) < 0.01:
            dom = "-"
        elif abs(mean_attn[i]) > abs(mean_mlp[i]):
            dom = "attn"
        else:
            dom = "MLP"
        print(f"{i:>3} {mean_attn[i]:>+8.3f} {mean_mlp[i]:>+8.3f} {dom:>5}")

    attn_top5 = sorted(
        [(int(i), round(float(mean_attn[i]), 3)) for i in range(n_layers)],
        key=lambda x: x[1],
    )[:5]
    mlp_top5 = sorted(
        [(int(i), round(float(mean_mlp[i]), 3)) for i in range(n_layers)],
        key=lambda x: x[1],
    )[:5]
    print(f"\nTop-5 attn-critical:  {attn_top5}")
    print(f"Top-5 MLP-critical:   {mlp_top5}")


if __name__ == "__main__":
    main()
