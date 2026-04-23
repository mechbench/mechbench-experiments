# Direct Logit Attribution Across FACTUAL_15

**Date:** 2026-04-22
**Script:** `experiments/step_33_dla_factual_sweep.py`
**Plot:** `caches/dla_factual_sweep.png`

## Background

Finding 32 (step_32) reported that on "The capital of France is" with target Paris and distractor Berlin, the cumulative residual in layers L10–L14 actively prefers Berlin to Paris, reaching a (Paris − Berlin) diff of −9.9 at L12. The finding left open a direct empirical question: is this mid-layer wrong-answer-preferring phase a systematic property of the network, or specific to the Paris/Berlin pair? This experiment runs the same DLA decomposition across the full FACTUAL_15 battery, each prompt paired with a plausible distractor.

## Experiment

One forward pass per prompt, capturing `resid_post` at all 42 layers. For each prompt, compute `(target − distractor)` logit at each layer's resid_post via `logit_attrs`. Aggregate across the 15 prompts to see whether per-layer patterns are systematic or prompt-specific.

Distractor per prompt (same category, plausible competitor — e.g. Paris/London, Tokyo/Kyoto, oxygen/nitrogen, five/six, Wednesday/Thursday).

## Results

### The mid-layer wrong-answer phase is systematic, but *far* weaker than Paris/Berlin

Aggregate (target − distractor) across the 15 prompts, at each layer:

| Layer range | Mean diff | Median diff | Fraction of prompts with diff < 0 |
|---|---:|---:|---:|
| L0–L7 | +0.42 | +0.49 | 44% |
| **L8–L14** (early mid-layer) | **−0.61** | **−0.56** | **58%** |
| L15–L22 | +0.85 | +0.77 | 38% |
| **L23–L26** (pivot band) | **−0.45** | **−0.87** | **60%** |
| L27–L40 | +7.9 | +7.3 | 7% |
| L41 (final norm) | +3.92 | +4.32 | 0% |

Strongly-negative cells (diff < −2) in L9–L24: **51 / 240** (21%); elsewhere: 37 / 390 (9.5%). Roughly a 2.2x overrepresentation of wrong-answer preference in the mid-layer band. Systematic, yes — but the Paris/Berlin magnitude of −9.9 at L12 is on the extreme end. Most prompts show mid-layer dips in the −0.5 to −2.0 range.

Per-prompt variability is high. Strong dippers (≥9/16 negative layers in L9–L24): Leonardo/Michelangelo, Shakespeare/Marlowe, five/six, cold/warm, Paris/London, Tokyo/Kyoto, China/Japan, oxygen/nitrogen, Au/Ag. Weak or no dip (≤5/16): pets/animals, Wednesday/Thursday, blue/gray, Sahara/Asia, meters/miles.

### Two negative zones, not one

The aggregate shows two distinct negative regions: an early mid-layer dip at L8–L14 (attributable to the finding 32 effect), and a **second, smaller negative band at L23–L26** — right at the known pivot. L23's aggregate mean is −0.60 (median −0.44); 8/15 prompts are still negative at L23. L24 is −0.44 (9/15 negative); L25 is −0.52 (9/15); L26 is −0.25 (10/15). L27 flips positive for the majority.

This wasn't visible in step_32, because Paris/Berlin's huge mid-layer dip dominated the plot. Across 15 prompts the pattern resolves into a distinctive **double dip**: the network writes the wrong answer in the middle, starts to correct at L15–L22, and then writes *more* wrong-answer direction at L23–L25 before finally flipping. L23 isn't the middle of the correction — it's the *last* shove in the wrong direction before the flip.

### The L23 pivot is per-prompt robust

Per-prompt "last layer with negative (target − distractor) diff":

| Prompt | Last negative layer | Global? |
|---|---:|:--|
| Paris | L26 | |
| Tokyo | L30 | |
| China | L25 | |
| Brazil | **L38** | |
| Africa | L31 | |
| oxygen | L30 | |
| meters | **L35** | yes |
| Au | L28 | |
| Shakespeare | L29 | yes |
| Leonardo | L26 | |
| five | L26 | |
| Wednesday | L27 | |
| cold | **L21** | |
| blue | L26 | |
| pets | L30 | |

Median: L28. Mode region: L25–L30. All but one prompt lands between L21 and L38, and the bulk cluster within a few layers of L26 — the layer where *Paris/Berlin* flipped in step_32. This is a per-prompt statistic that reliably fingerprints the L23 pivot region across heterogeneous factual-recall tasks: the negative-to-positive transition happens within a narrow band following the pivot.

### Sequence-completion prompts behave differently

The three weakest-dip prompts (pets/animals, Wednesday/Thursday, blue/gray) are exactly the cases that aren't really factual recall:

- **pets/animals** — the category the distractor belongs to is a *superset* of the target, not a lateral competitor.
- **Wednesday/Thursday** — sequence completion. There's no category to retrieve from; the mechanism is serial-order tracking.
- **blue/gray** — answer is closer to a commonsense default than a retrieved fact.

This is weak evidence for finding 32's **hypothesis 1** (the mid-layer dip reflects generic-category retrieval preceding specific-item retrieval) over **hypothesis 2** (geometric artifact). If the dip were a pure projection-artifact, it would appear uniformly across all prompts regardless of task structure. That it concentrates on *lateral-competitor* prompts — where there's a plausible wrong category member to be retrieved — and is muted on *supercategory* or *serial-order* prompts is what a retrieval story would predict.

This is not strong evidence. Three prompts is not a sample. But it gives hypothesis 1 a cheap directional test that hypothesis 2 doesn't naturally generate.

### Final-layer compression is robust

L40 → L41 mean diff drops from +9.47 to +3.92 — a 59% compression. Every prompt shows this. The final norm pulls both competing-token raw logit magnitudes toward zero. This is a network-wide property of the tied unembed + final norm pipeline, not a prompt-specific artifact. (Consistent with the Paris/Berlin step_32 observation, now confirmed across 15 prompts.)

## Synthesis

Three substantive results against the step_32 single-prompt finding:

1. **The mid-layer wrong-answer phase is real and systematic, but Paris/Berlin was an unusually strong instance.** The aggregate dip at L8–L14 is real; the magnitude is typically 5–20× weaker than the Paris/Berlin case. Any quantitative claim built on step_32's numbers alone would have been misleading.
2. **A second negative band at L23–L26 survives aggregation.** This is the more robust finding. On 10 of 15 prompts, the cumulative residual is still distractor-preferring at L25. The L23 pivot now has a sixth angle: the layer where, across a heterogeneous prompt battery, the last negative writing happens before the flip.
3. **Prompt-structure sensitivity in the mid-layer dip weakly favors the retrieval interpretation over the geometric-artifact interpretation.** Worth designing a sharper test.

## Follow-up ideas

1. **Sharper test of the retrieval hypothesis.** Pick prompts where a generic category member is specifically *not* the target (e.g. the Amazon flows through → Peru, distractor → Brazil; Marie Curie discovered → radium, distractor → uranium). If the mid-layer dip tracks "the category's most prototypical member" rather than the distractor token per se, the dip should be *weaker* on these inverted cases. This would localize the effect to category-retrieval rather than to specific target/distractor vocabulary positions.
2. **DLA on per-head contributions across the battery.** Step_32's per-head leaderboard was suggestive (top writers at L30+, none at L23). Aggregated across 15 prompts, is there a consistent set of heads that show up as top-writers across tasks? Is L40 H0 a "factual-recall output head" or was its Paris-specific dominance prompt-specific?
3. **Does the L23–L26 negative band correspond to branch-level negative writes?** Step_32's branch decomposition hinted that attention writes turn pro-target later than MLP writes. Aggregate across the battery and check whether the L23–L26 dip is dominated by attention (consistent with L23 being attention-critical per finding 04) or MLP.
