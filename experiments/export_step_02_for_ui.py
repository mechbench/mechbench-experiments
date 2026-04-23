"""Export step_02 layer-ablation data as JSON for mechbench-ui.

Mirrors step_02_layer_ablation.py but emits JSON to ../mechbench-ui/public/data/
instead of plotting. Per-layer mean and median Δ log p of the model's own top-1
prediction when each layer's residual-stream update is fully zeroed.

~2 minutes on an M-series Mac.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.prompts.factual import FACTUAL_15  # noqa: E402
from mechbench_core import (  # noqa: E402
    Ablate,
    GLOBAL_LAYERS,
    Model,
    N_LAYERS,
)


MIN_CONFIDENCE = 0.5


def last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def resolve_output_path() -> Path:
    here = Path(__file__).resolve()
    tree_root = here.parent.parent.parent
    ui_data_dir = tree_root / "mechbench-ui" / "public" / "data"
    ui_data_dir.mkdir(parents=True, exist_ok=True)
    return ui_data_dir / "step_02_layer_ablation.json"


def main() -> None:
    output_path = resolve_output_path()
    print(f"Output target: {output_path}")

    print("Loading model...")
    model = Model.load()

    prompts = FACTUAL_15.prompts
    validated: list[tuple[str, str, int, float]] = []  # text, target, top1_id, baseline_lp

    print(f"Validating {len(prompts)} prompts (top-1 prob >= {MIN_CONFIDENCE})...")
    for prompt in prompts:
        ids = model.tokenize(prompt.text)
        result = model.run(ids)
        lp = last_logp(result.logits)
        top1_id = int(np.argmax(lp))
        top1_prob = float(np.exp(lp[top1_id]))
        if top1_prob >= MIN_CONFIDENCE:
            validated.append((prompt.text, prompt.target, top1_id, float(lp[top1_id])))

    n = len(validated)
    print(f"{n}/{len(prompts)} prompts validated.")

    damage = np.zeros((N_LAYERS, n), dtype=np.float32)
    print(f"Running {N_LAYERS} x {n} = {N_LAYERS * n} ablated forward passes...")
    for layer in range(N_LAYERS):
        ablation = Ablate.layer(layer)
        for j, (text, _target, top1_id, baseline_lp) in enumerate(validated):
            ids = model.tokenize(text)
            result = model.run(ids, interventions=[ablation])
            lp = last_logp(result.logits)
            damage[layer, j] = float(lp[top1_id]) - baseline_lp
        if (layer + 1) % 6 == 0 or layer == N_LAYERS - 1:
            print(f"  layer {layer}: mean Δlogp = {float(damage[layer].mean()):+.3f}")

    prompt_records = [
        {
            "text": text,
            "target": target,
            "top1_id": top1_id,
            "baseline_logprob": round(baseline_lp, 4),
            "damage": [round(float(v), 4) for v in damage[:, j]],
        }
        for j, (text, target, top1_id, baseline_lp) in enumerate(validated)
    ]

    mean_per_layer = damage.mean(axis=1)
    median_per_layer = np.median(damage, axis=1)

    payload = {
        "experiment": "step_02_layer_ablation",
        "description": (
            "Per-layer ablation: zero each of the 42 decoder blocks' "
            "residual-stream update one at a time and measure Δ log p of "
            "the model's own top-1 prediction on validated factual-recall "
            "prompts. More negative = more damaging to ablate."
        ),
        "n_layers": N_LAYERS,
        "global_layers": list(GLOBAL_LAYERS),
        "model": "mlx-community/gemma-4-E4B-it-bf16",
        "prompts": prompt_records,
        "aggregates": {
            "mean": [round(float(v), 4) for v in mean_per_layer],
            "median": [round(float(v), 4) for v in median_per_layer],
        },
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
