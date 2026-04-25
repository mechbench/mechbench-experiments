"""Probe layer ablation on Gemma 3 4B.

One-off feasibility / "see what happens" run for task 000187:
does the L23-style pivot reproduce in a non-on-device-E-series
variant? Gemma 4 26B-A4B / 31B don't fit in 32 GB unified memory
at bf16; Gemma 3 12B at bf16 (~24 GB) was attempted but `mlx_vlm.load`
hung indefinitely on this machine. We fall back to Gemma 3 4B
(~7 GB at bf16) — same architectural question (no
num_kv_shared_layers, 5:1 spacing rule), comfortable headroom.

This script bypasses mechbench-core's hook system entirely
because mechbench-core's forward path is hardcoded against
mlx_vlm.models.gemma4. We monkey-patch each TransformerBlock's
__call__ to return its input unchanged for one ablated forward
pass at a time. Crude but produces the per-layer Δ log p curve.

Output: caches/gemma3_4b_layer_ablation.json with the raw values.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

MODEL_ID = "mlx-community/gemma-3-4b-it-bf16"
PROMPT = "Complete this sentence with one word: The Eiffel Tower is in"


def main() -> None:
    print(f"[probe] loading {MODEL_ID} (this is slow on first run)...")
    t0 = time.time()
    model, processor = load(MODEL_ID)
    print(f"[probe] loaded in {time.time() - t0:.1f}s")

    # Locate the language-model layer list.
    lm = model.language_model.model  # Gemma3Model
    layers = lm.layers
    n_layers = len(layers)
    print(f"[probe] {n_layers} transformer layers")

    # Gemma 3 instruct expects the chat template; the bare prompt
    # produced a placeholder token (' ____') with logp -0.22.
    tokenizer = processor.tokenizer
    chat = [{"role": "user", "content": PROMPT}]
    rendered = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    ids = mx.array([tokenizer.encode(rendered, add_special_tokens=False)])
    print(f"[probe] prompt tokens: {ids.shape[1]}")

    def last_logp(logits: mx.array) -> np.ndarray:
        last = logits[0, -1, :].astype(mx.float32)
        lp = last - mx.logsumexp(last)
        mx.eval(lp)
        return np.array(lp)

    def run_forward() -> mx.array:
        # Bypass any vision / multimodal entry points; call the language
        # model directly with input ids. mlx-vlm's LanguageModel exposes
        # __call__(input_ids, ...) returning logits.
        return model.language_model(ids).logits

    print("[probe] baseline forward...")
    baseline = run_forward()
    baseline_lp = last_logp(baseline)
    top1_id = int(np.argmax(baseline_lp))
    baseline_top1 = float(baseline_lp[top1_id])
    top1_token = tokenizer.decode([top1_id])
    print(f"[probe] baseline top1={top1_id} ({top1_token!r}) lp={baseline_top1:.4f}")

    # Ablation = swap the layer in lm.layers (a plain Python list) for
    # an identity callable. Patching __call__ on the instance doesn't
    # work — nn.Module dispatches through the class, so an
    # instance-level override is ignored. List-item replacement does
    # work because mlx-vlm's forward iterates `for layer in self.layers`
    # and just calls `layer(h, mask, cache)`.
    def identity(h, *_args, **_kwargs):
        return h

    damage = np.zeros(n_layers, dtype=np.float32)
    for i in range(n_layers):
        original = layers[i]
        layers[i] = identity  # type: ignore[assignment]
        try:
            t0 = time.time()
            ablated = run_forward()
            lp = last_logp(ablated)
            damage[i] = float(lp[top1_id]) - baseline_top1
            elapsed = time.time() - t0
            print(
                f"[probe] L{i:02d}: Δlogp={damage[i]:+.3f}  ({elapsed:.1f}s)"
            )
        finally:
            layers[i] = original

    out_dir = Path("caches")
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "gemma3_4b_layer_ablation.json"
    out_json.write_text(json.dumps({
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "top1_id": top1_id,
        "top1_token": top1_token,
        "baseline_logprob": round(baseline_top1, 4),
        "damage": [round(float(v), 4) for v in damage],
        "n_layers": n_layers,
    }, indent=2))
    print(f"[probe] wrote {out_json}")

    peak = int(np.argmin(damage))
    print(f"[probe] peak damage at L{peak}: Δlogp={damage[peak]:+.3f}")
    # Top 5 most damaging:
    order = np.argsort(damage)
    top5 = [(int(i), float(damage[i])) for i in order[:5]]
    print(f"[probe] top-5 most damaging: {top5}")


if __name__ == "__main__":
    sys.exit(main() or 0)
