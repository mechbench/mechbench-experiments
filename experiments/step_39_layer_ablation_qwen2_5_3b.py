"""Step 39 — layer-ablation damage curve on Qwen 2.5 3B.

The cross-family test: does the L23-style pivot exist outside the
Gemma family? Mirrors step_02 (E4B), step_34 (Gemma 3 4B),
step_35 (E2B) — same methodology, same FACTUAL_15 battery, same
LayerAblationPayload schema for cross-model rendering. Through the
mechbench-core hook system that 000201 just landed.

Qwen 2.5 3B specs (read from mlx-lm ModelArgs at load time):
  - 36 transformer layers
  - hidden_size 2048, GQA 16/2 (n_heads/n_kv_heads)
  - No hybrid attention; no KV-sharing; no MatFormer side-channel
  - Tied unembed (tie_word_embeddings=True)

Outcome relevant to the L23-pivot story:
  - If Qwen 2.5 3B shows a Gemma-3-4B-like front-loaded curve
    with no late-layer pivot → the L23 phenomenon is
    architecture-family-specific to Gemma 4 E.
  - If Qwen 2.5 3B shows a peaked late-layer pivot somewhere →
    the framing has to be revisited yet again.
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
from mechbench_schema import (  # noqa: E402
    AblationPrompt,
    LayerAblationPayload,
    LayerAggregates,
)


MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-bf16"
MIN_CONFIDENCE = 0.5


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def resolve_output_path() -> Path:
    here = Path(__file__).resolve()
    tree_root = here.parent.parent.parent
    ui_data_dir = tree_root / "mechbench-ui" / "public" / "data"
    ui_data_dir.mkdir(parents=True, exist_ok=True)
    return ui_data_dir / "step_39_layer_ablation_qwen2_5_3b.json"


def main() -> None:
    output_path = resolve_output_path()
    print(f"Output target: {output_path}")

    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model = Model.load(MODEL_ID)
    print(f"Loaded in {time.time() - t0:.1f}s")

    arch = model.arch
    n_layers = arch.n_layers
    global_layers = list(arch.global_layers)
    print(
        f"Qwen 2.5 3B: {n_layers} layers, n_heads={arch.n_heads}, "
        f"n_kv_heads={arch.n_kv_heads}, model_type={arch.model_type}"
    )

    # Qwen2.5-3B-bf16 is a *base* model. Chat-templating it produces "The…"
    # sentence-starter tokens as top-1 instead of factual answers; bypass
    # the template and tokenize the bare completion prompt.
    use_chat_template = "Instruct" in MODEL_ID

    prompts = FACTUAL_15.prompts
    validated: list[tuple[str, str, int, float, mx.array]] = []
    print(f"\nValidating {len(prompts)} prompts (top-1 prob >= {MIN_CONFIDENCE}, "
          f"chat_template={use_chat_template})...")
    for prompt in prompts:
        ids = model.tokenize(prompt.text, chat_template=use_chat_template)
        result = model.run(ids)
        lp = _last_logp(result.logits)
        top1_id = int(np.argmax(lp))
        top1_prob = float(np.exp(lp[top1_id]))
        top1_token = model.tokenizer.decode([top1_id])
        keep = top1_prob >= MIN_CONFIDENCE
        marker = "✓" if keep else "✗"
        print(
            f"  {marker} prob={top1_prob:.3f} top1={top1_token!r:>16}  "
            f"target={prompt.target!r:>16}  '{prompt.text[:55]}'"
        )
        if keep:
            validated.append(
                (prompt.text, prompt.target, top1_id, float(lp[top1_id]), ids)
            )

    # If we re-ran with chat_template=False above, the existing bare-token
    # ids in `validated` are correct — no further re-tokenization needed.

    n = len(validated)
    print(f"{n}/{len(prompts)} prompts validated.\n")

    if n == 0:
        print("No prompts validated; aborting before ablation pass.")
        return

    damage = np.zeros((n_layers, n), dtype=np.float32)
    print(f"Running {n_layers} × {n} = {n_layers * n} ablated forward passes...")
    t0 = time.perf_counter()
    for layer_idx in range(n_layers):
        ablation = Ablate.layer(layer_idx)
        for j, (_text, _target, top1_id, baseline_lp, ids) in enumerate(validated):
            result = model.run(ids, interventions=[ablation])
            lp = _last_logp(result.logits)
            damage[layer_idx, j] = float(lp[top1_id]) - baseline_lp
        if (layer_idx + 1) % 5 == 0 or layer_idx == n_layers - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (layer_idx + 1) * (n_layers - layer_idx - 1)
            print(
                f"  layer {layer_idx:2d}: mean Δlogp = "
                f"{float(damage[layer_idx].mean()):>+7.3f}  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
            )

    payload = LayerAblationPayload(
        experiment="step_39_layer_ablation_qwen2_5_3b",
        description=(
            "Per-layer zero-ablation on Qwen 2.5 3B (36 layers, every layer "
            "is global attention, no KV-sharing, no MatFormer). First "
            "non-Gemma model in the L23-pivot cross-family series. Through "
            "mechbench-core's mlx-lm-fallback path (000201)."
        ),
        model=MODEL_ID,
        n_layers=n_layers,
        global_layers=global_layers,
        prompts=[
            AblationPrompt(
                text=text,
                target=target,
                top1_id=top1_id,
                baseline_logprob=round(baseline_lp, 4),
                damage=[round(float(v), 4) for v in damage[:, j]],
            )
            for j, (text, target, top1_id, baseline_lp, _ids) in enumerate(validated)
        ],
        aggregates=LayerAggregates(
            mean=[round(float(v), 4) for v in damage.mean(axis=1)],
            median=[round(float(v), 4) for v in np.median(damage, axis=1)],
        ),
    )

    output_path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n"
    )
    print(f"\nWrote {output_path} ({output_path.stat().st_size} bytes)")

    mean = damage.mean(axis=1)
    peak = int(np.argmin(mean))
    print(f"Peak mean damage at L{peak}: {float(mean[peak]):+.3f}")
    order = np.argsort(mean)
    top5 = [(int(i), round(float(mean[i]), 3)) for i in order[:5]]
    print(f"Top-5 most damaging (by mean): {top5}")


if __name__ == "__main__":
    main()
