# Per-Layer Ablation for Sense Disambiguation: Embedding Dominates, L23 Reappears

**Date:** 2026-04-16
**Script:** `experiments/step_20_homonym_layer_ablation.py`
**Plot:** `caches/homonym_layer_ablation.png`

## The setup

Port of finding 02 (per-layer zero-ablation) to the sense-disambiguation question from finding 17. For each of the 42 layers, ablate it (residual stream passes through unchanged), then measure how cleanly the *capital*-position residuals separate by sense at two readout layers:

- **L12** — peak silhouette layer for sense disambiguation (the geometric peak).
- **L41** — final hidden layer (the late-layer decoding readout).

42 ablations × 32 prompts = 1344 forward passes (~4 minutes).

## The prediction

From the j4q beads issue: *"Layer 12 (geometric peak per finding 17) is also the causal peak. Ablating it should drop sense-separation more than ablating a layer outside the engine room."*

The prediction is wrong, but in an interesting way.

## Results

### Sanity check: ablating layers downstream of the readout is exactly a no-op

For the L12 readout, all 29 ablations of layers 13–41 produce **Δsil = 0.0000 exactly**. The harness composes ablations and residual captures correctly — when an ablation happens after the capture point, it cannot affect the capture, and the framework respects that with no rounding noise. (The only way you can detect a downstream ablation in this design would be if `Capture.residual` were misordered relative to the ablation hooks, which it isn't.)

### Ranked impact at the L12 readout (the geometric peak)

| Rank | Ablated layer | Type | Δsilhouette | ΔNN purity | Δk-means purity |
|----:|--------------:|:----:|------------:|-----------:|----------------:|
| 1 | L0  | local  | **−0.4146** | **−0.3750** | **−0.3125** |
| 2 | L1  | local  | −0.2405     | −0.1562     | −0.2188     |
| 3 | L5  | GLOBAL | −0.1667     | ±0.000      | −0.0312     |
| 4 | L11 | GLOBAL | −0.1206     | ±0.000      | −0.1250     |
| 5 | L6  | local  | −0.0945     | ±0.000      | +0.0312     |
| —  | L12 | local  | −0.0491     | ±0.000      | −0.0312     |

**Ablating L12 itself is the *sixth* most damaging ablation upstream of L12, not the most.** The peak-silhouette layer is not the layer doing the most causal work to *produce* that peak.

What dominates: the very first layer (L0, the input-embedding contribution). Removing it costs −0.41 silhouette, slashes NN purity from 0.844 to 0.469. After L0, the pattern is roughly:

- L0–L1 (early local layers contributing the embedded-token-shaped residual): heavy hitters.
- L5 and L11 (the first two global-attention layers): the next tier. Both global, both upstream of L12.
- L6, L7, L12 itself: modest contributions.

### Ranked impact at the L41 readout

| Rank | Ablated layer | Type | Δsilhouette | ΔNN purity | Δk-means purity |
|----:|--------------:|:----:|------------:|-----------:|----------------:|
| 1 | L23 | **GLOBAL** | −0.1494 | −0.3125 | −0.2188 |
| 2 | L0  | local  | −0.1469 | **−0.4688** | −0.2812 |
| 3 | L26 | local  | −0.1193 | −0.0625 | ±0.000 |
| 4 | L22 | local  | −0.1084 | −0.2500 | −0.2812 |
| 5 | L13 | local  | −0.0533 | −0.1250 | −0.1562 |

**L23 (the engine-room global) is the single most damaging ablation for late-layer sense disambiguation** — and it's the same layer finding 04 identified as the only attention-critical layer for factual recall. Two unrelated experiments (factual recall, sense disambiguation) both fingerprint L23 as a hub.

After L23, L0 again (input-embedding contribution catastrophic everywhere), then L26 and L22 — the L22-L26 cluster is right around L23 and may be the wider neighborhood doing related work.

### The very last layers slightly *improve* L41 sense separation when ablated

The bottom of the L41 table:

| Ablated layer | Δsilhouette at L41 |
|--------------:|-------------------:|
| L40 | **+0.0636** |
| L41 | **+0.0644** |
| L21 | +0.0483 |
| L34 | +0.0333 |
| L28 | +0.0330 |

Ablating L40 or L41 makes the L41 sense clusters slightly *tighter* (silhouette goes up). This is consistent with finding 17's observation that the late layers optimize for vocabulary-level decodability, not for clean sense-cluster geometry. The late-layer transforms degrade the geometric structure that peaked at L12, in service of producing output-shaped representations that decode through the unembed.

### What about the engine-room (L11–L24) at L41 readout?

| Ablated layer | Δsil at L41 |
|--------------:|------------:|
| L11 | +0.0123 |
| L12 | +0.0092 |
| L13 | −0.0533 |
| L14–L21 | mostly −0.05 to +0.05, mixed signs |
| L22 | −0.1084 |
| L23 | **−0.1494** |
| L24 | −0.0514 |

The engine-room damage map for sense disambiguation at L41 is concentrated at L22-L24, with L23 the single worst. This mirrors what we already saw in factual recall: the global-attention layer L23 plays an outsized role.

## Two pictures of "where sense disambiguation lives"

The two readouts give different — but reconcilable — answers.

**At L12 (the geometric peak):**
- The L12 sense representation is built almost entirely from L0-L11.
- Ablating L0 or L1 (the input-embedding contribution) is catastrophic.
- Ablating L5 or L11 (the early globals) is the next-largest hit.
- L12 itself is a modest contributor — the peak silhouette is more about *where* the construction culminates than *which layer* does the work.

**At L41 (the late-layer readout):**
- Sense info read out at L41 still depends on L0 (input embedding survives all the way through).
- It also depends on L23 — the same engine-room global layer that finding 04 fingerprinted for factual recall.
- The very last layers (L40-L41) slightly *degrade* sense separation in service of vocabulary-shaped output.

## Reconciling with the L12 hypothesis

The original prediction — *"L12 is the geometric peak so L12 should be the causal peak"* — conflated two things: where a representation is *cleanest* and where it's *built*. The L12 silhouette peak is what you see at the *end* of a multi-layer construction (L0-L11) that contributes more or less smoothly across those layers. There's no single layer in the engine room that dominates the construction; it's a distributed early-layer process anchored on the input embeddings.

L23, by contrast, *is* a single dominant hub — but for the late-layer readout, not for the early geometric peak. That fits the picture from findings 02-06 of L23 as a global-attention bottleneck whose specific role is harder to pin down but whose absence is consistently load-bearing.

## Verdict

The "peak silhouette layer = causal peak" hypothesis fails. Sense disambiguation at the geometric peak (L12) is built distributedly from L0-L11, with the input embedding (L0) by far the largest single contributor. The peak is where construction *ends*, not where it *happens*.

For the late-layer readout (L41), the dominant single layer is L23 — the same global-attention hub that finding 04 identified as critical for factual recall. This is now the second independent experiment to fingerprint L23 as load-bearing. There's something specific about that layer's role across multiple downstream tasks.

The very last layers (L40-L41) are slightly *anti*-clustering for sense-separation, consistent with finding 17's observation that they prioritize vocabulary-level readability over geometric cluster structure.

## Caveats and follow-ups

- Two readout layers only. A full L0-by-L0 readout sweep would show how the damage propagates depth-wise.
- The L0 catastrophic effect could partly be a measurement artifact: ablating L0 means the L1 input is just the input embedding (zeroed), so the entire residual stream restart from scratch. Worth a control: ablate L0 versus pass through the unmodified embedding without going through L0's transform — that would isolate "what L0's contribution adds" from "the embedding itself".
- Sub-layer ablation (j4q's sibling beads issue `ung`) is the natural next step: separately ablate each layer's attention branch vs. MLP branch. This would tell us whether L0's catastrophic effect is an MLP thing or an attention thing, and whether L23's effect is the attention pattern (consistent with finding 04) or the MLP.
- One sense corpus (capital, 4 senses). Replicating on a second homonym (`bank`, `light`, `bark`) would test whether the L0 / L23 fingerprints generalize.
