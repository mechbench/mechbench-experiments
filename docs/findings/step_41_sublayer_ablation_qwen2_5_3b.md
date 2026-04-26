# Sublayer ablation on Qwen 2.5 3B Instruct: distributed attn-critical, MLP-dominant elsewhere

Cross-family port of step_04 (E4B) / step_36 (E2B) per task 000205. For each FACTUAL_15 prompt, ablate each layer's attention branch and MLP branch independently. The L23-pivot story leaned hardest on this experiment in E4B: *"MLPs dominate; only L23 is attention-critical."* This is the test for whether that single-attention-critical-layer pattern generalizes.

## What I found

14/15 FACTUAL_15 prompts validated. 36 layers × 14 prompts × 2 branches = 1008 ablated forwards.

**Top-5 attention-critical (after L0/L1 baseline):**

| layer | attn Δ | mlp Δ | dominant |
|---|---|---|---|
| L24 | **−1.28** | −0.33 | attn |
| L27 | −0.66 | −0.22 | attn |
| L14 | −0.44 | −0.01 | attn |
| L31 | −0.43 | −0.27 | attn |
| L21 | −0.41 | −0.41 | tied |

**Top-5 MLP-critical (after L0):**

| layer | mlp Δ | attn Δ |
|---|---|---|
| L13 | **−1.80** | −0.12 |
| L22 | −1.43 | −0.30 |
| L35 | −1.03 | −0.02 |
| L23 | −0.85 | −0.07 |
| L12 | −0.75 | −0.05 |

## Comparison

| model | attn-critical pattern | MLP-critical pattern |
|---|---|---|
| E4B (step_04) | "**only L23**" — single isolated peak | broad, MLP dominates almost everywhere |
| E2B (step_36) | L0 baseline + **L14** ± **L12-L13** band | broad band L2-L11, MLP dominant |
| **Qwen 2.5 3B (step_41)** | **L0/L1 baseline + L24, L27, L14, L31, L21 distributed** | **L13, L22, L35, L23 distributed** |

**Three observations:**

1. **The "MLPs dominate" half of E4B's step_04 finding reproduces.** Across Qwen 2.5 3B, MLP-critical work happens at more layers than attn-critical work, and most layers show non-trivial MLP damage. Family-wide claim: ✓.

2. **The "only L23 is attention-critical" half does NOT reproduce.** Qwen has at least 5 layers (L24, L27, L14, L31, L21) with non-trivial attention criticality. The single-isolated-attn-peak pattern is Gemma-4-E-specific so far.

3. **Step_39's whole-layer L22 peak is MLP-driven, not attention-driven.** L22 mean attn Δ = −0.30; mean mlp Δ = −1.43. The mid-network signal we noticed in the whole-layer curve isn't doing anything attention-special; it's MLP integration of context, which all transformers do.

## What this says about the L23 pivot

The L23-pivot story has *two* core claims when grounded in step_04:
- (a) MLP and attention have qualitatively different damage curves — true family-wide.
- (b) **One specific layer has anomalously high attention-criticality** — true only for E4B (and weakly for E2B's L14 with L12-13 close behind). False for Qwen 2.5 3B Instruct.

Claim (b) is the load-bearing one for the architectural-pivot argument: if attention criticality is concentrated at one layer, that layer is doing a uniquely-positioned attention job. Distributed attn-criticality (Qwen) is consistent with distributed routing rather than a load-bearing pivot. So:

- The step_39 whole-layer ablation showed scattered mid-late activity in Qwen with no specific peak → consistent with no L23-style pivot.
- The step_41 sublayer ablation now shows that scattered activity is mostly MLP-driven, with attn-criticality similarly distributed across L14/L21/L24/L27/L31.
- **No single attention-critical layer = no architectural pivot.** Cross-family confirms what step_39 suggested.

## A small bit of cross-prompt structure to flag

The L21 row (attn −0.41, mlp −0.41) is the only "tied" layer in the table. Whatever L21 is doing, it splits its damage between branches — neither dominates. This is unusual; nothing analogous shows up in E4B's step_04 numbers. Could be a Qwen-specific feature; could be a coincidence of FACTUAL_15-on-this-model. Not pursuing further unless it shows up in a second experiment.

## Sources

- Battery script: [`experiments/step_41_sublayer_ablation_qwen2_5_3b.py`](../../experiments/step_41_sublayer_ablation_qwen2_5_3b.py)
- Raw data: `caches/step_41_sublayer_ablation_qwen2_5_3b.json` (gitignored)
- Reference: step_04 finding (E4B), [`step_36_sublayer_ablation_e2b.md`](step_36_sublayer_ablation_e2b.md) (note: doesn't exist as a separate finding — see [`step_37_dla_factual_sweep_e2b.md`](step_37_dla_factual_sweep_e2b.md) for the cross-experiment writeup)
- Family-wide picture: [`docs/essays/gemma_family_global_spacing.md`](../essays/gemma_family_global_spacing.md)
