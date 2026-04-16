"""Gemma 4 E4B architectural facts.

Single source of truth for everything model-specific in the framework. Hard-coded
because v0 targets exactly one model. If/when we generalize to other Gemma
variants or other architectures, this is the file that splits.
"""

from __future__ import annotations

DEFAULT_MODEL_ID = "mlx-community/gemma-4-E4B-it-bf16"

N_LAYERS = 42
D_MODEL = 2560
N_HEADS = 8
VOCAB_SIZE = 262144

# Hybrid attention pattern: every 6th layer (5, 11, 17, 23, 29, 35) plus the
# final layer (41) computes full-sequence attention. The other 35 layers use
# sliding-window attention. See CLAUDE.md and language.py for the architectural
# significance of this choice.
GLOBAL_LAYERS: tuple[int, ...] = (5, 11, 17, 23, 29, 35, 41)


def layer_type(i: int) -> str:
    """Return 'full_attention' for global layers, 'sliding_attention' otherwise."""
    return "full_attention" if i in GLOBAL_LAYERS else "sliding_attention"


# Hook-point names exposed by the canonical forward pass, expressed as the
# part following 'blocks.{i}.'. The full layer-scoped name is constructed as
# f'blocks.{i}.{point}' for i in [0, N_LAYERS).
LAYER_HOOK_POINTS: tuple[str, ...] = (
    "resid_pre",          # layer input (== resid_post[i-1] except for i=0)
    "attn_out",           # attention branch contribution after o_proj + post_attention_layernorm
    "mlp_out",            # MLP branch contribution after post_feedforward_layernorm
    "gate_out",           # MatFormer per-layer-input side-channel contribution
    "resid_post",         # layer output after layer_scalar
    "attn.weights",       # post-softmax attention weights [B, n_heads, L, S_kv]
    "attn.per_head_out",  # weights @ values, before o_proj concat [B, n_heads, L, head_dim]
)

# Top-level (non-layer) hook points. Empty in v0; reserved for future
# expansion (e.g. embed.out, final_norm.out).
GLOBAL_HOOK_POINTS: tuple[str, ...] = ()

# Hook points that require manual attention computation. When any hook or
# capture targets one of these, the canonical forward switches from the fused
# scaled_dot_product_attention kernel to a manual softmax path that exposes
# the attention internals.
ATTN_INTERNAL_POINTS: frozenset[str] = frozenset({"attn.weights", "attn.per_head_out"})


def all_hook_names() -> list[str]:
    """Enumerate every valid hook-point name in the model.

    Useful for discovery, error-message suggestions, and tab-completion in
    notebook environments. Length: N_LAYERS * len(LAYER_HOOK_POINTS) +
    len(GLOBAL_HOOK_POINTS) = 42 * 7 + 0 = 294.
    """
    out = list(GLOBAL_HOOK_POINTS)
    for i in range(N_LAYERS):
        for p in LAYER_HOOK_POINTS:
            out.append(f"blocks.{i}.{p}")
    return out
