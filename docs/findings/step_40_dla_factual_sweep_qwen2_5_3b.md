# Cross-family commit fraction on Qwen 2.5 3B Instruct: depth-fraction story revisits 000190

DLA factual sweep (step_33 methodology) on Qwen 2.5 3B Instruct, the first non-Gemma model in the cross-family commit-fraction series. Per task 000204.

## What I found

15 prompts, 36 layers, all run through mechbench-core's mlx-lm fallback path (000201). Per-prompt commit-layer distribution:

```
Paris/London         L27   (frac 0.750)
Tokyo/Kyoto          L27   (0.750)
China/Japan          L24   (0.667)
Brazil/Peru          L00   (0.000)  ←  embed-aligned commit
Africa/Asia          L31   (0.861)
oxygen/nitrogen      L30   (0.833)
meters/miles         L32   (0.889)
Au/Ag                L20   (0.556)
Shakespeare/Marlowe  L28   (0.778)
Leonardo/Michel.     L27   (0.750)
five/six             L27   (0.750)
Wednesday/Thursday   L00   (0.000)  ←  embed-aligned commit
cold/warm            L26   (0.722)
blue/gray            L00   (0.000)  ←  embed-aligned commit
pets/animals         L28   (0.778)
```

**Median commit fraction: 0.750. Mean: 0.606. Std: 0.312.**

## The picture across four models

| model | n_layers | median commit frac | shape |
|---|---|---|---|
| E4B | 42 | 0.690 | tight cluster L26-L31 |
| E2B | 35 | 0.657 | cluster L20-L27 + 2 final-layer outliers |
| **Qwen 2.5 3B Instruct** | **36** | **0.750** | **cluster L20-L32 + 3 L0 outliers** |
| Gemma 3 4B | 34 | 0.088 | bimodal: 7 at L0, 1 at L33 |

**Three of four models cluster at ~0.7 of network depth. Gemma 3 4B is the outlier.**

## The 000190 conclusion was overgeneralized

Task 000190 closed with: *"Depth-fraction candidate refuted — Gemma 3 4B median 0.088 vs E-series cluster 0.66."* That conclusion took Gemma 3 4B as a representative cross-family / cross-architecture-mechanism datapoint. Qwen 2.5 3B Instruct adds a different non-Gemma datapoint, and **it sits squarely in the E-series band, not in Gemma 3 4B's anomalous-low band**.

What this means for the depth-fraction reading:

- The framing is **not refuted** in the way 000190 claimed. Three of four models with very different architectures (Gemma 4 E series, Qwen 2.5) commit at depth-fraction 0.66-0.75.
- The framing is **not confirmed either**. One model (Gemma 3 4B) commits at depth-fraction 0.088 — a categorical outlier. Whatever explains Gemma 3 4B's anomaly is doing real work; "depth-fraction 0.7" alone doesn't capture it.

A more careful version of the candidate, given the new data:

> **Most transformers commit at depth-fraction ~0.7 on FACTUAL_15-style prompts, but a subset of (apparently small + heavily-Instruct-tuned + no-logit-softcap) models route the answer through embed-aligned representations and commit at L0.**

That's a hypothesis worth testing on more models before claiming, not a finding. What's clear is that 000190's "cleanly refuted" verdict was based on an n=1 anomaly (Gemma 3 4B) generalized to "no transformer follows the depth-fraction pattern" — and Qwen 2.5 3B's data shows that's wrong.

## The three "L0 commits" in Qwen

Three Qwen prompts commit at L0 — Brazil/Peru, Wednesday/Thursday, blue/gray. These are similar to Gemma 3 4B's anomalous prompts but scattered (not all 7 like Gemma 3 4B). Suggests there's a per-prompt characteristic that drives early-commit behavior, and it varies in incidence across architectures rather than being all-or-nothing.

A semantically-difficult battery (where the model needs to reason rather than retrieve a one-token answer) might shift the L0-outliers toward the L24-32 cluster. Worth noting for any future depth-fraction work.

## Methodological note (also relevant to 000190)

The Qwen final-layer (target − distractor) diffs are in the +13 to +47 range — much smaller than Gemma 3 4B's +600 to +2800 range, but larger than E4B's typical +1 to +5. Magnitude differences across families come from the per-family unembed structure (tied vs untied, softcap vs no-softcap). The sign-change point — which is what defines the commit layer — is unaffected by these magnitude differences.

## Sources

- Battery script: [`experiments/step_40_dla_factual_sweep_qwen2_5_3b.py`](../../experiments/step_40_dla_factual_sweep_qwen2_5_3b.py)
- Raw data: `caches/step_40_dla_factual_sweep_qwen2_5_3b.json` (gitignored)
- Methodology references: [`step_38_dla_factual_sweep_gemma3_4b.md`](step_38_dla_factual_sweep_gemma3_4b.md), [`step_37_dla_factual_sweep_e2b.md`](step_37_dla_factual_sweep_e2b.md)
- Family-wide picture: [`docs/essays/gemma_family_global_spacing.md`](../essays/gemma_family_global_spacing.md)
