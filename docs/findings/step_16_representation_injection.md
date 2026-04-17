# Representation Injection: Centroids Are Probes, Not Handles

**Date:** 2026-04-16
**Script:** `experiments/step_16_representation_injection.py`
**Plot:** `caches/representation_injection.png`
**Proposal:** `docs/proposals/factorization-experiments.md` Experiment 2

## Hypothesis

Finding 12 + finding 15 (just landed) showed that mid-layer subject-position activations encode *which cognitive operation* the model is preparing to run, factored apart from operands and answers. The proposal asked the natural follow-up: are the per-category centroids *functional handles*? That is — if you take the capital-category centroid `v_capital` and inject it into a different prompt's residual stream, does the model start running a capital lookup?

This is the strong steering claim. The weak claim is that centroids are passive correlates: yes, when the model is running a capital lookup, its activation lives in this region; but adding the region's centroid into a different context doesn't trigger the lookup operation.

## Setup

`v_capital` was computed two ways for safety:

- **Raw**: mean of the 8 DISAMBIG_A1 (capital-lookup) prompts' subject-position residuals at layer 30. Norm ≈ 131.
- **Mean-subtracted**: same, minus the mean of all 32 DISAMBIG_ALL prompts. Norm ≈ 39. This is the form that decoded to multilingual capital-concept tokens in finding 12 — the raw centroid is dominated by the prompt-template common-mode.

Both vectors were tested. Three injection target prompts:

- **C1 (neutral)**: "Complete this sentence with one word: The following country is famous for its". The model has no specific operation in mind here.
- **C2 (different operation)**: "Complete this sentence with one word: The past tense of run is" — model produces 'ran' at p=0.946 baseline.
- **C3 (control, same operation)**: "Complete this sentence with one word: The capital of Germany is" — model produces 'Berlin' at p=0.999 baseline.

Intervention: `Patch.add(layer=30, position=final, value=v_capital, alpha=α)` for α ∈ {0, 0.5, 1.0, 2.0, 5.0}. Measurement: top-1 token + sum of probability over all token IDs that decode to one of the 8 capital-city names (with or without leading space).

## Results

### Both vectors fail to steer

p(capital-city tokens) at the final position, as a function of α:

| α | C1 raw | C1 sub | C2 raw | C2 sub | C3 raw | C3 sub |
|---|-------:|-------:|-------:|-------:|-------:|-------:|
| 0.0 (baseline) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9992 | 0.9992 |
| 0.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9965 | 0.9987 |
| 1.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6645 | 0.9975 |
| 2.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9794 |
| 5.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

p(capital-city tokens) **never rises above 0** in C1 or C2 under any α with either vector. The neutral prompt and the past-tense prompt do not get steered toward producing capital cities.

C3 (control) starts at 0.9992 (model already producing 'Berlin' confidently) and either stays high or collapses. It never *increases* — there's no headroom.

### Raw vector destroys outputs; mean-subtracted is more robust but equally null

Top-1 tokens under raw `v_capital`:

| α | C1 (neutral) | C2 (past tense) | C3 (control) |
|---|--------------|-----------------|--------------|
| 0.0 | 'The' | 'ran' | 'Berlin' (p=0.999) |
| 0.5 | '\*\*' | 'ran' | 'Berlin' (p=0.996) |
| 1.0 | 'is' | 'is' | 'Berlin' (p=0.665) |
| 2.0 | 'is' | ' is' | ' is' |
| 5.0 | ' is' | ' is' | ' is' |

By α=2, the raw vector has completely overridden whatever the prompt was about. The model produces 'is' or ' is' (a generic continuation token) regardless of context. This isn't steering toward capital lookups — it's destruction of all signal.

Top-1 tokens under mean-subtracted `v_capital`:

| α | C1 (neutral) | C2 (past tense) | C3 (control) |
|---|--------------|-----------------|--------------|
| 0.0 | 'The' | 'ran' (p=0.946) | 'Berlin' (p=0.999) |
| 0.5 | 'The' | 'ran' (p=0.962) | 'Berlin' (p=0.999) |
| 1.0 | '\*\*' | 'ran' (p=0.977) | 'Berlin' (p=0.997) |
| 2.0 | '\*\*' | 'ran' (p=0.982) | 'Berlin' (p=0.979) |
| 5.0 | '\*\*' | "'" | '\*\*' |

The mean-subtracted version is dramatically more robust. C2's 'ran' answer holds (and even *strengthens* slightly) through α=2, where the raw vector had already destroyed it at α=1. C3's 'Berlin' holds at p=0.979 at α=2, vs full destruction under the raw vector. So mean subtraction does what we'd expect — it removes the prompt-template common-mode from the steering vector, making the injection less destructive.

But the steering is still null: 'ran' stays as 'ran', not 'Paris' or 'Berlin'. The capital-lookup operation is not invoked.

## Interpretation

The combined verdict from Experiment 1 + Experiment 2:

- **The factorization claim is real** (E1): mid-layer subject-position activations encode the cognitive operation, separately from the operands and from the surface tokens.
- **But centroids are diagnostic probes, not steering interventions** (E2): you can read the operation off the activation, but you can't use the centroid as a function pointer to invoke that operation in a different context.

This lands in the upper-right cell of the proposal's 2×2 outcome grid: factorization survives, but the stronger functional claim does not.

### Why the negative result makes sense in retrospect

A few hypotheses for why the centroid doesn't steer:

1. **The operation is encoded by an aggregate of activations across positions and layers, not by a single position's residual.** Capital lookup at a real prompt involves the token 'France' at the operand position, the connective 'is' at a later position, and the cumulative state built across all 24+ layers. Injecting a single vector at one (layer, position) cell only nudges one of those many degrees of freedom. The model's overall trajectory through computation is determined by the prompt; one cell's nudge is not enough to redirect it.

2. **The centroid is a sample from a manifold, not the operation itself.** The activation when the model runs capital-lookup-on-France lives in a particular high-dimensional region. The centroid is the arithmetic mean of 8 such points. Adding that mean to a different prompt's activation moves the prompt's activation slightly toward that region, but doesn't construct the rest of the structure (matched query/key/value patterns at other positions, the right dependencies through the layers above) that would make the model treat the new activation *as if* it were running capital lookup.

3. **Final position at layer 30 is too late.** By layer 30 the model is in readout mode (per finding 02 — most ablation impact is in layers 10-24, the engine room). Capital lookup, if it happened, happened in the engine room and is now being decoded. Injecting at layer 30 final position is patching the readout, not the computation. The proposal mentioned trying earlier layers (15, 20, 25) and other positions; we only tested layer 30 final position.

4. **Single-position injection is the wrong intervention.** The activation-steering literature (Turner et al., "Activation Addition", and follow-ups) typically injects steering vectors at *every* position, sometimes across multiple layers, sometimes computed as a difference between contrastive prompts. The naive single-position injection used here is the simplest version of the test; a richer injection design might give a different result.

We picked the design specified in the proposal and ran it cleanly. The answer to *that* design is "no". A stronger steering test would still be worth running — but it's a different experiment.

### What this means for downstream claims

- The "embeddings are programs" framing — that mid-layer activations are *function pointers you can invoke* — does not survive this test in its naive form. They're more like *labels* that diagnose what's running.
- Mid-layer category centroids remain useful as **classification anchors**: you can take a new prompt, compute its mid-layer activation, and ask "which category centroid is it nearest?" to identify the operation it's about to run. This is what enabled the 100% nearest-neighbor classification result in finding 12. That use case is unaffected by the negative steering result here.
- Future steering work on this model should explore the directions hypothesis 4 lays out: multi-position injection, multi-layer injection, contrastive steering vectors (e.g., `v_capital_lookup - v_letter_counting`), or injection at earlier "engine room" layers.

## Caveats

- One layer (30), one position (final), one steering vector per condition. The proposal called for sweeping layer ∈ {15, 20, 25, 30, 35} and trying multiple positions; we did not. A more complete null result would require that broader sweep.
- α ∈ {0, 0.5, 1, 2, 5}. Higher α destroys outputs; lower α has no measurable effect. There may be a Goldilocks zone we missed, but the smooth monotone trend in the plot doesn't suggest one.
- The "different operation" prompt (past tense of run) is a single instance. A wider C2 cohort (multiple different-operation prompts) would strengthen the negative claim.
- Single capital centroid. Mean of 8 prompts. A larger anchor cohort (e.g., 100 capital prompts across many countries) might give a stronger steering vector — though I'd expect the qualitative result to hold.
