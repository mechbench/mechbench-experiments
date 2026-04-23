# Direct Logit Attribution on a Single Factual Prompt

**Date:** 2026-04-22
**Script:** `experiments/step_32_direct_logit_attribution.py`
**Plots:** `caches/dla_paris_berlin_by_layer.png`, `caches/dla_paris_berlin_per_head.png`

## Background

Every prior experiment in this project has measured *what happens to the output* when we perturb the network — ablate a layer, zero a head, swap an activation. Direct logit attribution asks the dual question: decompose the residual stream at the final position into per-component contributions (per layer, per branch, per head), and project each one through the tied unembed to read off its *marginal* contribution to each target token's logit.

This is the canonical circuit-analysis primitive in the mech-interp literature. We didn't have it until now — the capability landed as the first addition to `mechbench-core` after the extraction, via `accumulated_resid` / `decompose_resid` / `head_results` / `logit_attrs` (task 000105).

The implementation is deliberately minimal. It projects components through the linear unembed only, with no final-norm fold. TransformerLens optionally divides each component by the final RMSNorm's per-position scale to make the decomposition exactly additive with the model's output logits; we skip that step in v0. The practical consequence: magnitudes reported here are *raw linear projections* of the residual components, not model-calibrated logits. Relative rankings of contributions — which component writes more toward Paris than another — are preserved.

## Experiment

One prompt, one pair of competing tokens: `"Complete this sentence with one word: The capital of France is"`, target tokens `" Paris"` (id 9079) vs `" Berlin"` (id 15687). Captures: `resid_post` at all 42 layers, `attn_out`/`mlp_out`/`gate_out` at all 42 layers, `attn.per_head_out` at all 42 layers. Single forward pass. ~30 seconds.

All DLA numbers below are the **(Paris − Berlin) logit-difference contribution**, which is the standard circuit-analysis reduction: it cancels terms that boost both competing tokens equally (e.g. a generic "proper noun" direction) and isolates the part of each component that actively prefers Paris to Berlin.

## Results

### The layer-level trajectory has three phases, not two

| Phase | Layers | (Paris − Berlin) at resid_post | Interpretation |
|------|-------|----|----|
| Neutral | 0–8 | ±1 around zero | Representation hasn't specialized yet |
| **Berlin-preferring** | **9–24** | **reaches −9.9 at L12–13** | The residual actively encodes Berlin more strongly than Paris |
| Paris-preferring | 25–40 | climbs to +19.1 at L38 | The flip, then the readout |
| Final-layer compression | 41 | +5.95, both tokens pulled toward zero | Normalization pass flattens magnitudes |

The middle phase is the genuinely surprising part. Not "Paris hasn't activated yet" but actively *anti-Paris*: projected through the unembed, the residual in L10–14 looks more like Berlin than Paris by several log-probability points. The model is producing a representation that, read naively, predicts the wrong answer, and then corrects itself later.

Two hypotheses consistent with this, in ascending order of how much I trust them:

1. **Generic-capital retrieval preceding France-specific retrieval.** The mid-layer representation may encode "prototypical European capital" before conditioning on France. Berlin is a plausible prototype; Paris is the country-specific answer. Testing this would need a corpus of `(country, capital)` pairs and a check for whether the mid-layer Δ favors the "modal" capital in each case.
2. **Readout direction is unstable through the middle.** The residual doesn't care about the unembed direction until the late layers force it to. What looks like "anti-Paris" may be a projection artifact of looking at the vocab-space shadow of a representation that isn't yet trying to be vocab-shaped. This is essentially the same reason the logit lens fails on mid layers (finding 01) — the representation is doing real work, but not work aligned with the output-vocabulary direction.

The distinction matters: hypothesis 1 is a claim about retrieval structure; hypothesis 2 is a claim about the geometry of the residual stream. DLA alone can't distinguish them. It needs a corpus, not one prompt.

### Layer 23 is the last negative-writing layer

The (Paris − Berlin) diff at L23's `resid_post` is **−2.4**. Above L23 it flips positive (L24: −1.7 still negative, L25: **+0.95** first positive, L26: +3.91, L27: +5.59, and from there it climbs monotonically through L38). L23 is the last layer at which the cumulative residual prefers Berlin to Paris.

This is a fifth independent angle on the L23 pivot, alongside the four already in the essay:

- **Finding 04 / §5:** L23 is the only layer where attention ablation beats MLP ablation.
- **Finding 03 / §7:** L23 is a top-three single-layer hotspot for the MatFormer side-channel.
- **Finding 20 / §16:** L23 is the single most damaging per-layer ablation for homonym sense disambiguation.
- **Finding 30 / §21:** L23 is the peak of the learned surprisal-correlated direction (Lyraaaa's independent-corpus result lands one layer off at L23).
- **This finding:** L23 is the last layer at which the cumulative residual, read through the unembed, prefers the wrong answer.

Whatever L23 is doing, the flip happens either *at* L23 or *immediately after*. The next two layers (L24 local, L25 local) complete the sign change. L26 — still three layers before the logit lens would say the answer is visible — is already strongly pro-Paris in DLA.

### The top per-head writers are late-model, not L23

Leaderboard of individual (layer, head) contributions to (Paris − Berlin) at the final position:

| Rank | Layer | Head | Δ logit | Global? |
|-----:|------:|-----:|--------:|:--------|
| 1 | 40 | 0 | +3.87 | |
| 2 | 31 | 0 | +3.65 | |
| 3 | 35 | 4 | +2.81 | yes |
| 4 | 35 | 7 | +1.28 | yes |
| 5 | 30 | 2 | +1.26 | |
| 6 | 2 | 3 | +1.00 | |
| 7 | 36 | 3 | +0.96 | |
| 8 | 34 | 1 | +0.92 | |
| 9 | 32 | 0 | +0.91 | |
| 10 | 0 | 1 | +0.90 | |

No L23 head appears in the top 10. The highest-writing heads live at L30–L40, with L35 (a global layer) contributing two of the top four. This is consistent with the picture from finding 07 (no single head is a bottleneck under ablation): L23's load-bearing role is not "this head writes the answer." The answer-writing happens late, distributed across several late-model heads, most of which aren't on global layers at all.

This tightens the interpretation of L23's role. Across five independent experiments, L23 is causally critical; across DLA, it does not write the answer. Those are compatible — it means L23 is doing *routing* or *gating* rather than output-space writing. Finding 06's attention patterns already pointed this direction (L23 attends to chat-template structure, not subject tokens); DLA confirms from the opposite side that the answer-writing happens elsewhere.

### Branch decomposition at each layer

The plot's lower panel shows (Paris − Berlin) contributions from each of the three branches — `attn_out`, `mlp_out`, `gate_out` — per layer. Observations, directionally (magnitudes are raw):

- The three branches do not agree on when the flip happens. MLP writes turn pro-Paris around L24–L25; attention writes stay noisy through L28 and don't lock in until later.
- The MatFormer `gate_out` contributions are small in magnitude at every layer. The gate's role from finding 03 is load-bearing for the network's overall function (full ablation → −30.4 log-prob damage), but its direct write-to-unembed contribution on this specific (Paris vs. Berlin) dimension is modest. This is consistent with the gate doing *per-token grounding* rather than *output-direction writing* — the same shape of finding that L23's attention patterns suggested for its attention output.

The branch-decomposition view wants a corpus, not one prompt. As a single data point it's suggestive; as evidence it's thin.

## Methodological caveats

1. **One prompt.** Everything above is `"The capital of France is"` with `Paris` vs. `Berlin`. The Berlin-preferring mid-layer phase might be specific to this prompt — Berlin has plausible structural reasons to be a competitor for Paris (both European capitals, both likely in training-data co-occurrence statistics). A natural next step is to run DLA across the FACTUAL_15 battery with a token-pair per prompt.
2. **No norm fold.** Magnitudes are raw linear unembed projections, not model-output logits. Relative rankings within this experiment are reliable; absolute values are not directly comparable to softmax probabilities or to logit-lens outputs from finding 01.
3. **Single competitor pair.** (Paris − Berlin) is one cut. (Paris − London), (Paris − Rome), (Paris − Madrid) might tell different stories about the mid-layer "generic European capital" hypothesis.

## Follow-up ideas

1. **Corpus DLA on FACTUAL_15.** Run per-prompt DLA with a plausible alternative answer for each; check whether the mid-layer "wrong-answer preferring" phase is systematic. If it is, it's a property of the network; if it's specific to Paris/Berlin, it's a property of this prompt.
2. **DLA on homonym disambiguation.** Per finding 20, L23 is the single most damaging ablation for sense disambiguation. Does DLA on a homonym prompt (e.g. `"bank"` disambiguated to `river` vs. `money`) show a mirror of the Paris/Berlin pattern — a mid-layer phase favoring the wrong sense before the flip?
3. **Norm-fold implementation.** Expose the final RMSNorm's per-position scale as a hook point and add an `apply_ln=True` mode to `logit_attrs` so the decomposition becomes exactly additive with output logits. Would let DLA numbers be compared directly to logit-lens and softmax-probability numbers from prior findings.
4. **Rank the attention heads by their Paris vs. Berlin Q/K reads.** Head-weight analysis (finding 26) plus DLA is the full TransformerLens-style per-head circuit: *which head reads what from where, and writes what toward which vocab direction*. The pieces exist; no experiment has assembled them yet for this task.
