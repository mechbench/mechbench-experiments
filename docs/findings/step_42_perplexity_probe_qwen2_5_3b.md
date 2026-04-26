# Perplexity probe on Qwen 2.5 3B Instruct: R² shape transfers, Lyra rotation does NOT

Cross-family port of step_30 (E4B) / step_31 (E2B) per task 000206. For each narrative passage in EMOTION_STORIES_TINY + EMOTION_NEUTRAL_BASELINE, capture residuals at every layer, fit RidgeCV mapping residual → per-token surprisal, compute (a) test R² per layer and (b) cosine similarities between consecutive learned weight vectors.

The two distinctive features of Lyra's E4B finding (and step_31's E2B replication):
- **R² builds up smoothly across the network depth, peaks just before a specific layer, then declines.** Single-peaked smooth curve.
- **The learned surprise direction rotates *essentially orthogonally* at one specific layer boundary** (E4B L22→L23 cosine 0.033; E2B L13→L14 cosine 0.015).

The cross-family question is which (if either) of these patterns transfers.

## What I found

112 narrative passages, 4534 content tokens, 36 layers, train/test split 80/20.

**R² curve:**

```
L0   0.4415       L9   0.5392       L18  0.6290       L27  0.6450
L1   0.4908       L10  0.5401       L19  0.6395       L28  0.6354
L2   0.5062       L11  0.5418       L20  0.6369       L29  0.6203
L3   0.5450       L12  0.5579       L21  0.6419       L30  0.6060
L4   0.5523       L13  0.5829       L22  0.6500       L31  0.5680
L5   0.5614       L14  0.5908       L23  0.6693       L32  0.5507
L6   0.5520       L15  0.5970       L24  0.6741 ←     L33  0.4833
L7   0.5459       L16  0.5996                          L34  0.4806
L8   0.5366       L17  0.6114                          L35  0.4521
```

**Peak: L24, R²_test 0.6741, corr 0.823.** Smooth rise from 0.44 at L0 to peak at L24, then smooth decline to 0.45 at L35.

**Sharpest consecutive cosine drop: L8 → L9 = 0.7299.** That is **not a sharp rotation** — Gemma's signature was orthogonal-or-near-orthogonal (cosine 0.015-0.033). Qwen has *no* layer boundary where the surprise direction rotates dramatically.

## Cross-family scoreboard

| feature | E4B (step_30) | E2B (step_31) | **Qwen 2.5 3B (step_42)** |
|---|---|---|---|
| R²_test peak layer | L23 (or L21 with Lyra's corpus) | L12 | **L24** |
| R²_test peak value | 0.671 | 0.619 | **0.674** |
| Peak as depth fraction | 0.55 (or 0.50) | 0.34 | **0.667** |
| Sharpest rotation cos | **0.033** (L22→L23) | **0.015** (L13→L14) | **0.7299** (L8→L9) |
| Rotation actually sharp? | yes | yes | **no** |

**Two patterns, two different verdicts on each:**

- **R² shape (smooth rise to peak then smooth fall) transfers** to Qwen. Same general curve, similar peak magnitude (0.67 vs 0.67), peak in roughly the same depth-band.
- **The orthogonal-rotation signature does NOT transfer.** Qwen's residual stream represents surprise in a way that doesn't reorient at any single layer. Gemma's reorients at the KV-boundary in particular.

## What this rules in and out for the L23-pivot narrative

The Lyra rotation was the strongest piece of E4B-specific architectural evidence in §22 of the experiment narrative. The mechanistic story was: at the layer where the residual transitions from fresh-K/V global attention to KV-shared attention (E4B L22→L23, E2B L13→L14), the basis used to encode "next-token surprisal" has to reset because the downstream computation can no longer freshly read the keys. Cosine 0.03 = "fully reorient."

**Qwen 2.5 has no fresh-K/V → KV-shared transition.** It has neither mechanism (no `num_kv_shared_layers`). The architectural-pressure-concentration story would predict no equivalent rotation in Qwen — and that's exactly what step_42 shows. The absence of rotation is consistent with the surviving narrative: rotation is fresh-K/V → KV-shared transition specific.

The R² curve shape transferring across all three families is mildly interesting but not surprising — every transformer's residual stream has to encode the next-token distribution somehow, and the smooth rise-then-fall pattern is generic.

## What still needs testing

The strongest test of the surviving narrative would be a non-Gemma-4-E model that *does* have the KV-sharing transition. None exist on mlx-vlm or mlx-lm at sizes that fit our hardware. The Lyra rotation might be:
- Specific to architectures that have the fresh-K/V → KV-shared transition (Gemma-4-E only) — surviving narrative
- Specific to the Gemma family broadly — would generalize to Gemma 3
- Specific to E4B/E2B because of training-pressure incidents in those specific runs

Step_31 already showed the rotation transfers from E4B to E2B. Step_42 shows it doesn't transfer to Qwen. Doesn't yet rule between the two remaining options. **A next experiment that would**: run step_42-equivalent on Gemma 3 4B Instruct. If the rotation appears there too, it's a Gemma-family thing (despite no KV-sharing in Gemma 3); if it doesn't, the surviving narrative tightens to "fresh-K/V → KV-shared transition specific."

Filing as a follow-up.

## Sources

- Battery script: [`experiments/step_42_perplexity_probe_qwen2_5_3b.py`](../../experiments/step_42_perplexity_probe_qwen2_5_3b.py)
- Raw data: `caches/step_42_perplexity_probe_qwen2_5_3b.json` (gitignored)
- E4B reference: [`step_30_perplexity_probe.md`](step_30_perplexity_probe.md)
- E2B reference: [`step_31_perplexity_probe_e2b.md`](step_31_perplexity_probe_e2b.md)
- Family-wide picture: [`docs/essays/gemma_family_global_spacing.md`](../essays/gemma_family_global_spacing.md)
