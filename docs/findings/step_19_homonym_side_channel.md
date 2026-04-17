# Side-Channel Ablation Wrecks Sense Disambiguation at the Late Layers

**Date:** 2026-04-16
**Script:** `experiments/step_19_homonym_side_channel.py`
**Plot:** `caches/homonym_side_channel.png`

## The setup

Port of finding 03 (MatFormer per-layer-input gate ablation) to the sense-disambiguation question from finding 17. Take the 32 HOMONYM_CAPITAL_ALL prompts (4 senses × 8 templates), zero the side-channel gate at all 42 layers, and compare sense separation at two readout layers against the no-ablation baseline:

- **L12** — peak silhouette layer for sense disambiguation (the geometric peak from finding 17).
- **L41** — final hidden layer (the late-layer decoding readout from finding 17).

Sense separation is measured at the *capital*-token position, the same position step_17 used.

## The prediction

**Minimal effect.** The reasoning: the side-channel feeds per-token *identity* into every block. All four sense cohorts share the same token at the *capital* position, so the side-channel input *there* is identical across cohorts. Sense disambiguation must therefore flow from (a) the residual stream and (b) attention to the disambiguating context tokens, both of which would be unaffected by zeroing the side-channel.

The prediction is wrong.

## Results

### Side-channel ablation degrades sense separation, dramatically at L41

| Readout | Metric    | Baseline | Side-channel off | Δ        |
|--------:|-----------|---------:|-----------------:|---------:|
| L12     | silhouette | +0.3351  | +0.2465          | −0.0886  |
| L12     | NN purity  | 0.844    | 0.844            | ±0.000   |
| L12     | k-means purity | 0.781 | 0.688          | −0.094   |
| L41     | silhouette | +0.1192  | **−0.0861**      | −0.2053  |
| L41     | NN purity  | 0.875    | **0.250**        | **−0.625** |
| L41     | k-means purity | 0.688 | 0.438          | −0.250   |

At **L12** the damage is moderate. The geometric clusters loosen (silhouette down ~26%) but each prompt's nearest neighbor is still in the same sense (NN purity unchanged at 0.844). The early-to-mid layer sense structure is largely intact even without the side-channel.

At **L41** the picture collapses. NN purity drops from 0.875 to 0.250 — *exactly chance for 4 classes*. Silhouette goes negative, meaning average inter-cluster cosine has overtaken intra-cluster cosine. Whatever the late layers were doing to the residual at the *capital* position, much of it depended on the side-channel.

### The asymmetry is the interesting part

It's tempting to read the L41 numbers as "side-channel is essential for sense disambiguation, full stop." But the L12 numbers tell a different story:

- The early-to-mid layer geometric structure (peak silhouette at L12, the engine room) survives side-channel ablation with only modest degradation.
- The late-layer transform from L12 → L41 — which step_17 showed makes the centroids decodable through the unembed but doesn't *improve* geometric separability — is what gets wrecked.

So the side-channel doesn't appear to be needed to *form* the sense disambiguation. The model can build a perfectly cluster-separated sense representation by L12 using only the residual stream and attention. What the side-channel is needed for is what comes *after* L12 — the late-layer refinement that turns the engine-room sense representation into something decodable, while preserving (rather than destroying) the cluster structure.

## Reconciling with finding 03

Finding 03 already established that the side-channel is load-bearing for factual recall: zeroing it across all layers drops mean log p of the model's own top-1 prediction by ~30. The single-layer ablation map there had its largest individual effects at the global-attention layers (L5, L11, L17, L23, L29, L35, L41), suggesting the side-channel and the global attention layers are functionally entangled.

What this experiment adds: the side-channel matters for *more than just per-token vocabulary readout*. It also matters for preserving the *geometric structure* of context-induced sense distinctions in the late layers. Even though the side-channel input at the *capital* position is identical across the four sense cohorts, the side-channel inputs at *other* positions (the disambiguating context words) differ — and removing those inputs apparently destroys the late-layer sense geometry, even though the early-layer sense geometry survives.

That's a subtler story than "the side-channel is per-token identity." It's per-token identity *contributing to a layer-wise computation that has cross-position effects via the residual stream and attention*. Zeroing it removes contributions at *all* positions, including the disambiguating ones, and the late-layer pipeline depends on those contributions to maintain sense separability.

## Verdict

The prediction failed. The side-channel is not a no-op for sense disambiguation. It's load-bearing in the same late-layer regime where it's load-bearing for factual recall — but not in the early-to-mid layers where the sense structure is initially formed.

This adds a third pattern to the side-channel's role:

1. (finding 03) Required for vocabulary-level next-token predictions in factual recall.
2. (finding 03) Concentrated at the global-attention layers (L5, L11, L17, L23, L29, L35, L41).
3. (this finding) Required to *preserve* sense disambiguation through the late layers, but not to *form* it in the early-to-mid layers.

## Caveats and follow-ups

- One sense corpus (capital, 4 senses, 8 prompts each). The pattern might be specific to this homonym; other homonym sets might behave differently.
- Two readout layers only. A full layer-by-layer readout sweep with side-channel off would tell us *where exactly* the late-layer collapse happens — a single layer or a gradual decay.
- Single-layer side-channel ablation (rather than all-layers) is the natural follow-up. Finding 03 already mapped which layers' side-channel contributions matter most for factual recall; the same map for sense disambiguation might be different and informative.
- This experiment ablates the side-channel's *output* contribution to the residual stream. The side-channel is also fed back into the attention computation in subtler ways; an experiment that surgically removes only the residual-stream contribution (vs. only the attention-feedback contribution) could disentangle which path the sense-preserving signal travels.
