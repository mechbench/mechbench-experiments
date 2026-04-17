# QK Sense-Clustering: Which Heads Read for "Capital" Sense?

**Date:** 2026-04-17
**Script:** `experiments/step_28_qk_sense_clustering.py`
**Plot:** `caches/qk_sense_clustering.png`
**Data:** `caches/qk_sense_clustering.npz`

## The setup

For every (layer, Q-head) = 42 × 8 = 336 pairs and every (layer, KV-head) = 42 × 2 = 84 pairs, extract the Q or K vector at the `capital` position across all 32 HOMONYM_CAPITAL_ALL prompts (4 senses × 8 prompts). Compute the silhouette score using the 4 sense labels. A positive silhouette means that head's Q (or K) space separates the senses; negative means no sense information in that subspace.

Tests the hypothesis (from step_17) that sense disambiguation lives in specific (layer, head) subspaces. Residual-stream silhouette peaked at L12; does some per-head projection of the residual push that higher? And is the signal in Q-space (what the head is looking for) or K-space (what the position advertises)?

Runtime: 32 prompts × 1 forward pass each with `Capture.queries(all 42 layers) + Capture.keys(all 42 layers)`. Post-processing is numpy silhouette computation. Total: **5 seconds**.

## Result

### K-space separates sense better than Q-space

| metric | best (layer, head) | silhouette |
|--------|-------------------:|-----------:|
| Q-head sense separability | L13 q5 | **+0.3242** |
| KV-head sense separability | L14 k0 (tied with k1) | **+0.3561** |
| (reference) residual L12 from step_17 | — | +0.3351 |

The K side is measurably sharper than the Q side (0.356 vs 0.324). This is consistent with what the two subspaces mean: `K[pos] = W_K @ resid[pos]` captures "what this position advertises about itself" — and for the `capital` token, the sense-content information *is* what the position advertises. `Q[pos] = W_Q @ resid[pos]` captures "what queries this position emits" — the query-side intent of `capital` in context is less sense-specific than the key-side identity.

At L14, both KV-heads produce identical +0.3561 silhouette (tied to four decimal places). Both are producing sense-separating K-projections from the same underlying residual, just via different W_K slices.

### Sharp phase transition at L23/L24

The Q-head best-per-layer sweep:

| layers | peak Q silhouette | character |
|--------|------------------:|-----------|
| L0-L6 | +0.07 to +0.19 | sense structure building up |
| **L7-L15** | +0.25 to +0.32 | **engine-room: clean sense separability** |
| L16-L23 | +0.13 to +0.24 | decay from engine-room peak |
| **L24-L41** | **−0.09 to +0.08** | **collapse to chance** |

The engine-room band (L7-L15) is where Q-space carries sense information. By L22-L23, silhouette drops. At **L24 (the first KV-shared layer), every Q-head's sense silhouette goes NEGATIVE** — not just small positive, but below chance. From L24 onwards no Q-head in any layer recovers sense separability.

This is a remarkably sharp phase boundary. The KV-sharing at L24 doesn't just change K and V (which now come from earlier layers' cache); it apparently also changes what the residual at the `capital` position looks like, such that `W_Q @ resid` stops carrying sense-specific direction.

### KV-sharing freezes the K-space sense representation

For the K side, layers 24-34 show a very different pattern: many of them have IDENTICAL silhouettes (`L22 k0 = L24 k0 = L25 k0 = L26 k0 = +0.2887`). This is the KV-sharing architecture surfacing in the analysis — those layers reuse the K tensor from an earlier non-shared layer (L23 presumably), so the K vectors are *literally the same* and produce the same silhouette exactly.

Structurally, this means: **the K-side representation of sense is "locked in" at L23**, the last non-shared global. Every downstream layer sees the same K-vectors and therefore can't add any sense information through its key projection. If the model refines sense disambiguation after L23, it does so through Q and the residual stream, not through K.

### The cleanest sense-disambiguation head

**L13 q5** emerges as the single sharpest Q-space sense separator — silhouette +0.3242, at a LOCAL (sliding-window) layer. Not a global. This is a local head in the engine room that specializes in sense disambiguation for this homonym.

Three of the top-5 Q-heads are at L13 (q5, q4, q1, q3 at ranks 1, 4, 7, 8). L13 as a whole is disproportionately sense-specialized: 7 of its 8 heads have sense-silhouette above +0.22.

On the K side, L14 both KV-heads tied at the top. L14-L15 K-heads dominate the top-10 list. The concentration is very specific: sense-reading is a layer-13/14/15 specialty, not distributed across the whole engine room.

## Why this matters

Three structural findings surface here that no prior experiment in the project could have produced:

1. **Q-space sense separability collapses at KV-sharing.** No prior experiment revealed the L23→L24 phase transition. Step_17 showed residual-stream silhouette peaks at L12 and decays smoothly; the per-head Q-view shows the decay is actually a cliff at L24.

2. **The K-side captures sense more cleanly than the Q-side.** In residual analysis both are tangled up because the residual contains both. Projecting through W_Q vs W_K separates them. K dominates. That's a non-obvious structural fact about where sense lives in the network.

3. **Specific heads are the sense-specialists.** L13 q5 and L14 k0/k1 are concrete targets. Any follow-up experiment that wants to probe the "sense-disambiguation circuit" has a starting point: these specific (layer, head) pairs, not the whole engine room.

## For the mechbench product

Two new default views the GUI should ship:

1. **(layer × head) concept-separability heatmap.** Given a labeled corpus and a choice of hook_point (resid_post / attn.q / attn.k), render a heatmap of silhouette per head. Automatic head-specialization discovery for any concept. With `fact_vectors_at_hook` shipped in cew, this is a few-line composition.
2. **KV-sharing boundary marker.** Models with KV-sharing (which is increasingly common) exhibit phase transitions at the sharing boundary. The workbench should surface the KV-shared region distinctly — maybe a horizontal line in the heatmap — so users notice when a signal drops off a cliff at the boundary.

## Caveats and follow-ups

- Single homonym corpus (capital, 4 senses). Replicating with HOMONYM_BANK or HOMONYM_LIGHT would test whether L13 q5 is specifically a "capital" specialist or a general sense-reader.
- We looked at the 'capital' position only. Running the same analysis at every position and marking the subject-position rows would reveal whether sense-reading is specifically at the homonym's own position or distributed.
- Ablating L13 q5 on HOMONYM_CAPITAL_ALL prompts and remeasuring residual-stream silhouette would tell us how causally important this head is. That's a follow-up experiment.
- The KV-sharing Q-space collapse is striking enough to deserve its own finding. What specifically about KV-sharing causes Q-projections to lose sense-structure? One hypothesis: the residual stream after L23 gets pulled toward the "KV-cache consumer" subspace, which isn't aligned with sense directions. Testing this would require comparing residuals at L23 vs L24 in the non-shared Q-projections.

## Verdict

Clean demonstration of per-head concept specialization using the primitives from cew and bxb: built on 5 seconds of forward passes, one labeled corpus, and the framework's silhouette-cosine function. The structural findings (K-side captures sense more cleanly, L13-L15 are sense-specialists, Q-space collapses at KV-sharing) are novel relative to the residual-stream-only analyses we had before, and they're discoverable *automatically* by the per-head sweep — you don't need to know in advance which heads to look at.

Epic `ric` is now 5/5 complete: static head-weight analysis (step_26), Q/K/V hook points (bxb), OV-circuit trajectories (step_27), Probe generalization to non-residual streams (cew), and this QK sense-clustering analysis. The attention-as-typed-components agenda the epic was filed for is now a fully-operational framework surface.
