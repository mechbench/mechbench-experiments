# Cross-family L23 test on Qwen 2.5 3B Instruct: no analogous pivot

The first non-Gemma datapoint in the L23-pivot cross-family series. Predictive validation pass for the surviving narrative from the 000125 / 000188 / 000189 / 000190 thread: *the L23-style pivot is fresh-K/V → KV-shared transition specific, not architecture-family-agnostic*. Qwen 2.5 has no KV-sharing, no MatFormer side-channel, no hybrid attention pattern; if the pivot generalizes anyway, the surviving narrative is wrong.

Through mechbench-core's mlx-lm fallback path (000201).

## Methodological note up-front

Qwen 2.5 3B has both a base (`Qwen2.5-3B-bf16`) and Instruct (`Qwen2.5-3B-Instruct-bf16`) build on mlx-community. Two passes failed before the third worked:

1. **Base model + chat template** (the default, since chat-templating is what gemma-3 instruct demanded). Result: 7/15 validated, mostly weak (top-1 was `' The'` continuations at p=0.5-0.75 — the base model treats the chat-wrapped prompt as a sentence opener rather than a factual completion).
2. **Base model + bare prompts** (added a `chat_template=False` kwarg to `Model.tokenize`). Worse: 3/15 validated, top-1s mostly placeholder tokens like `' __'` and `' ______'` because "Complete this sentence with one word:" specifically primes the base model to predict fill-in-the-blank glyphs.
3. **Instruct model + chat template**. **14/15 validated**, top-1 probabilities in the 0.80-1.00 band on most prompts. This matches the Gemma 3 4B Instruct profile and is the right comparison-class for the cross-family question.

The data below is from pass 3.

## Damage curve

Qwen 2.5 3B Instruct: 36 transformer layers, every layer is global attention (no hybrid pattern), tied unembed.

| layer | mean Δ log p | layer | mean Δ log p |
|---|---|---|---|
| 0 | **−15.56** ← peak | 18 | −0.04 |
| 1 | **−13.65** | 19 | −0.08 |
| 2 | −0.74 | 20 | −0.04 |
| 3 | −0.40 | 21 | −0.42 |
| 4 | +0.01 | 22 | **−2.18** |
| 5 | −0.32 | 23 | −0.10 |
| 6 | −0.58 | 24 | −0.46 |
| 7 | −0.04 | 25 | −0.07 |
| 8 | −0.65 | 26 | −0.41 |
| 9 | −0.86 | 27 | −0.13 |
| 10 | −0.27 | 28 | −0.16 |
| 11 | −0.39 | 29 | −0.26 |
| 12 | −0.04 | 30 | −0.36 |
| 13 | **−1.66** | 31 | **−1.15** |
| 14 | −0.12 | 32 | −0.10 |
| 15 | −0.05 | 33 | −0.27 |
| 16 | −0.06 | 34 | −0.50 |
| 17 | −0.27 | 35 | −0.93 |

Top-5 by mean: **L0, L1, L22, L13, L31**.

## Comparison

| model | scale | n_layers | peak | top-5 | shape |
|---|---|---|---|---|---|
| E4B (step_02) | 8B | 42 | L0 (−15.96) | L0, L14, L19, **L23**, L16 | invisible-middle band L10-24 + L23 pivot |
| E2B (step_35) | 5B | 35 | L6 (−15.13) | L6, L2, L9, L8, L0 | front-loaded + fresh-K/V cluster L4/9/14 |
| Gemma 3 4B (step_34) | 4B | 34 | L3 (−23.13) | L3, L8, L0, L4, L6 | front-loaded only; silent past L11 |
| **Qwen 2.5 3B (step_39)** | **3B** | **36** | **L0 (−15.56)** | **L0, L1, L22, L13, L31** | **front-loaded + scattered mid/late** |

Three things stand out in the Qwen result:

1. **L0/L1 dominance.** Universal across all four models (and Gemma 3 4B): first layer is always heavily damaging. This is methodology baseline, not signal about architecture.
2. **No L23-equivalent peak.** L22 stands out among the mid-network values at −2.18 — that's the closest analogue to E4B's L23 by depth fraction (L22/36 ≈ 0.61, comparable to L23/42 ≈ 0.55). But it's nowhere near a unique peak: L22 is the third-strongest layer overall and its Δ log p is 7× smaller than L0's. The "invisible middle" plateau that defines E4B's signature (L10-24 all damaging) doesn't appear here.
3. **Activity is scattered, not concentrated.** L13, L22, L31 each register some damage but no consistent band. This is structurally different from E4B's mid-network plateau and from Gemma 3 4B's "completely silent past L11" pattern. Closest in shape to E2B, which also has front-loaded peak + scattered mid-network activity.

## Verdict

**The L23-style pivot does not generalize to Qwen 2.5 3B.** Combined with 000189 (Gemma 3 4B → no analogous pivot) and 000188 (KV-boundary framing rejected on E2B), this is now four cross-family / cross-scale tests that have failed to find anything matching E4B's L23 phenomenon.

The surviving narrative — "L23 is fresh-K/V → KV-shared transition specific" — is consistent with the Qwen result by construction (Qwen has no such transition; framing predicts no pivot; we observe no pivot). But this is a *consistency* check, not an *informative* test of the surviving narrative. To genuinely test it, we'd need a model that has the fresh-K/V → KV-shared transition but isn't Gemma 4 E (e.g. Gemma 4 26B-A4B or 31B — both blocked on memory until mechbench-remote, 000194).

What this Qwen result does decisively rule out:
- **The L23 pivot as a generic property of small-to-medium dense transformers.** Both Gemma 3 4B and Qwen 2.5 3B Instruct are dense transformers in the 3-4B class, and neither shows it.
- **The L23 pivot as anything observable at this scale class without the KV-sharing mechanism.** Three tests, three negatives.

What it doesn't rule out:
- **The L23 pivot as a Gemma-4-E-specific feature.** Still possible.
- **The L23 pivot as something that emerges only at larger scales.** Untested.
- **The L23 pivot as a chat-tuning / alignment artifact** (E4B is `-it`, so is Gemma 3 4B `-it`, so is Qwen 2.5 3B `-Instruct`). All three are instruct-tuned. The pattern shows in only one. Not a chat-tuning artifact.

## Architectural note (Qwen 2.5)

Qwen 2.5 3B has **biased Q/K/V projections** (Qwen 2.x specific; Gemma 3/4 use unbiased projections). Doesn't change the damage-curve interpretation but worth noting because mechbench-core's `_forward_qwen.py` carries that detail. Also: Qwen 2.5 3B uses **GQA with very tight group sizing** (16 attention heads / 2 KV heads = 8-to-1 ratio), more aggressive than Gemma 3/4's typical 4-to-1.

The model is a **base-vs-instruct** distinction: the base 3B is a poor fit for FACTUAL_15 specifically because the prompt template is interpreted as fill-in-the-blank rather than completion. The Instruct variant works because chat-tuning makes "Complete this sentence with one word: X" be a question-answer interaction. This isn't a model-quality issue; it's a prompt-template mismatch.

## Sources

- Battery script: [`experiments/step_39_layer_ablation_qwen2_5_3b.py`](../../experiments/step_39_layer_ablation_qwen2_5_3b.py)
- Raw data: [`mechbench-ui/public/data/step_39_layer_ablation_qwen2_5_3b.json`](../../../mechbench-ui/public/data/step_39_layer_ablation_qwen2_5_3b.json)
- mechbench-core mlx-lm fallback: 000201 (now closed)
- Surviving narrative this is consistent with: [`docs/essays/gemma_family_global_spacing.md`](../essays/gemma_family_global_spacing.md)
