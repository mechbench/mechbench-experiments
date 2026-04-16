"""Activation cache harness for Gemma 4 E4B.

TransformerLens-style: run a prompt through the model and return logits
plus a dict of per-layer intermediates. The harness reimplements the forward
pass at the DecoderLayer level rather than monkey-patching, so we can tap
individual sub-module outputs without running each forward pass twice.

Cache stays in bf16. Cast to float32 only at analysis time (see
`to_float32(cache)`). Never convert bf16 mx.array straight to numpy — it
crashes with a PEP 3118 buffer format error.

Keys cached per layer i:
    blocks.{i}.resid_pre        [B, S, d_model]  layer input
    blocks.{i}.attn_out         [B, S, d_model]  post-attention branch contribution (post post_attention_layernorm)
    blocks.{i}.mlp_out          [B, S, d_model]  MLP branch contribution (post post_feedforward_layernorm)
    blocks.{i}.per_layer_gate   [B, S, d_model]  MatFormer per-layer-input side-channel contribution (if present)
    blocks.{i}.resid_post       [B, S, d_model]  layer output after everything, incl. layer_scalar

Note: resid_post(i) == resid_pre(i+1). We cache both for convenience.
"""

from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models import cache as cache_mod
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.gemma4.language import logit_softcap


def run_with_cache(
    model, input_ids: mx.array
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """Run a single forward pass and return (logits, activation_cache).

    Matches `Model.__call__` in gemma4.py / `Gemma4TextModel.__call__` in
    language.py, but inlines the per-layer loop so we can capture intermediates.

    Assumes text-only, non-MoE (i.e. E4B). Asserts otherwise.
    """
    lm = model.language_model
    tm = lm.model  # Gemma4TextModel

    assert not any(getattr(layer, "enable_moe", False) for layer in tm.layers), (
        "hooks.py currently supports the non-MoE path only (E4B). "
        "MoE models (e.g. 26B-A4B) would need the router/experts branch wired up."
    )

    # Embeddings + MatFormer per-layer-input side-channel.
    emb_out = model.get_input_embeddings(input_ids=input_ids, pixel_values=None)
    h = emb_out.inputs_embeds
    per_layer_inputs = emb_out.per_layer_inputs

    # Project per-layer inputs through the main-stream residual, as the text
    # model does internally (language.py: project_per_layer_inputs).
    if tm.hidden_size_per_layer_input and per_layer_inputs is not None:
        per_layer_inputs = tm.project_per_layer_inputs(h, per_layer_inputs)

    # Fresh KV cache + hybrid attention masks, as Gemma4TextModel.__call__ builds.
    kv_cache = cache_mod.make_prompt_cache(lm)
    global_mask = create_attention_mask(
        h,
        kv_cache[tm.first_full_cache_idx]
        if tm.first_full_cache_idx < len(kv_cache)
        else None,
    )
    sliding_mask = create_attention_mask(
        h,
        kv_cache[tm.first_sliding_cache_idx]
        if tm.first_sliding_cache_idx < len(kv_cache)
        else None,
        window_size=tm.window_size,
    )

    cache: Dict[str, mx.array] = {}

    for i, layer in enumerate(tm.layers):
        c = kv_cache[tm.layer_idx_to_cache_idx[i]]
        is_global = layer.layer_type == "full_attention"
        local_mask = global_mask if is_global else sliding_mask
        per_layer_input = (
            per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        )

        # --- Attention branch (matches DecoderLayer.__call__ lines 307-312) ---
        resid_pre = h
        cache[f"blocks.{i}.resid_pre"] = resid_pre

        a = layer.input_layernorm(h)
        a = layer.self_attn(a, local_mask, c)
        a = layer.post_attention_layernorm(a)
        cache[f"blocks.{i}.attn_out"] = a
        h = resid_pre + a

        # --- MLP branch (non-MoE; lines 327-332) ---
        mid = h
        m = layer.pre_feedforward_layernorm(mid)
        m = layer.mlp(m)
        m = layer.post_feedforward_layernorm(m)
        cache[f"blocks.{i}.mlp_out"] = m
        h = mid + m

        # --- Per-layer input gating (lines 335-347) ---
        if (
            layer.per_layer_input_gate is not None
            and layer.per_layer_projection is not None
            and layer.post_per_layer_input_norm is not None
            and per_layer_input is not None
        ):
            gate = layer.per_layer_input_gate(h)
            gate = nn.gelu_approx(gate)
            gate = mx.multiply(gate, per_layer_input)
            gate = layer.per_layer_projection(gate)
            gate = layer.post_per_layer_input_norm(gate)
            cache[f"blocks.{i}.per_layer_gate"] = gate
            h = h + gate

        # --- Layer scalar (line 349-350) ---
        if layer.layer_scalar is not None:
            h = h * layer.layer_scalar

        cache[f"blocks.{i}.resid_post"] = h

    # Final norm + tied-unembed (+ softcap if configured).
    h_final = tm.norm(h)
    logits = tm.embed_tokens.as_linear(h_final)
    if lm.final_logit_softcapping is not None:
        logits = logit_softcap(lm.final_logit_softcapping, logits)

    # Single eval for the whole cache + logits — MLX is lazy; storing graph
    # nodes in a dict does nothing until we eval.
    mx.eval(list(cache.values()) + [logits])
    return logits, cache


def to_float32(cache: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Cast every cached tensor to float32. Do this at the analysis boundary,
    right before going to numpy."""
    return {k: v.astype(mx.float32) for k, v in cache.items()}


if __name__ == "__main__":
    # Sanity check: the instrumented forward must agree with forward.py's
    # logits. If it doesn't, we've drifted from DecoderLayer.__call__ and
    # everything downstream is suspect.
    import numpy as np

    from forward import forward as plain_forward
    from forward import load_model, top_k_tokens

    model, processor = load_model()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    prompt = "Complete this sentence with one word: The Eiffel Tower is in"

    logits_plain, input_ids = plain_forward(model, processor, prompt)
    logits_hooked, cache = run_with_cache(model, input_ids)

    # Compare final-token logits in float32.
    a = np.array(logits_plain[0, -1, :].astype(mx.float32))
    b = np.array(logits_hooked[0, -1, :].astype(mx.float32))
    max_abs = float(np.max(np.abs(a - b)))
    argmax_match = int(np.argmax(a)) == int(np.argmax(b))

    print(f"Prompt: {prompt!r}")
    print(f"Cache keys: {len(cache)} entries")
    print(f"Sample keys: {list(cache.keys())[:3]} ... {list(cache.keys())[-3:]}")
    print(f"resid_pre[0]  shape: {cache['blocks.0.resid_pre'].shape}, dtype: {cache['blocks.0.resid_pre'].dtype}")
    print(f"resid_post[41] shape: {cache['blocks.41.resid_post'].shape}, dtype: {cache['blocks.41.resid_post'].dtype}")
    print(f"\nPlain vs hooked final-token logits:")
    print(f"  max |Δ|:        {max_abs:.6f}")
    print(f"  argmax matches: {argmax_match}")

    print("\nTop-5 (hooked path):")
    for tok, p in top_k_tokens(logits_hooked, tokenizer, k=5):
        print(f"  {tok!r:20s}  p={p:.4f}")
