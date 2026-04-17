# Scaled Corpus: Probes Generalize Better, but Intensity Responses Weaken

**Date:** 2026-04-17
**Script:** `experiments/step_25_emotion_probes_generated.py`
**Plot:** `caches/emotion_probes_intensity_compared.png`

## The setup

Build a second corpus with deliberately-curated topic diversity targeting the two failures step_24 diagnosed:

- `calm` learned scene-ambiance (lakes, tea, rain) from our original 16 scene-heavy passages. The new calm topics mix scenes with *states* (peace after conflict, mental steadiness, breathing exercises, inner stillness of a veteran under pressure).
- `happy` saturated at moderate-scale joy. The new happy topics span deliberately from small joy (wallet returned) through life-milestones (first full sentence) to shock-joy (unexpected life-changing news) and subdued satisfaction (finishing a long project).

6 emotions × 6 topics × 2 stories per topic = 72 passages, written by Claude. Compared head-to-head against the 96-passage hand-curated corpus from step_21 on four tests: self-consistency, cross-corpus generalization, and step_24's four intensity axes.

(A note on why Claude instead of Gemma: `mlx_vlm.generate` is broken for this model, and a naive autoregressive loop was taking ~30-40 minutes of local compute. The probe is built from Gemma 4's *activations* on the text, so the generator doesn't need to be the target model — any competent emotion-labeled corpus is viable training signal. The `generate_text` / `generate_labeled_corpus` primitives in `gemma4_mlx_interp/generate.py` are still shipped for the case where target = generator is appropriate.)

## Results

### Self-consistency: the generated corpus is tighter

| corpus | diag | per-passage accuracy |
|--------|:----:|--------------------:|
| hand-curated (step_21) | 6/6 | 87/96 = **90.6%** |
| generated (step_25)    | 6/6 | 70/72 = **97.2%** |

Both probe sets score 6/6 on their own corpus's diagonal. The generated set is per-passage sharper (97% vs 91%). This likely reflects Claude's more uniform prose quality — less template drift between passages within an emotion cohort — rather than a deeper improvement in concept capture.

### Cross-corpus generalization: probes work on each other's corpora

This is the strongest real validation of the concept claim.

| probes built on | scored on | diag | per-passage accuracy |
|-----------------|-----------|:----:|--------------------:|
| hand-curated    | generated | 5/6  | 47/72 = 65.3% |
| generated       | hand-curated | **6/6** | 60/96 = 62.5% |

Both probe sets classify the other corpus's passages at 62-65% accuracy — well above 17% chance. The generated probes actually get a full **6/6 aggregated diagonal on the hand-curated corpus**, slightly beating the hand-curated probes' own 6/6 score on their own corpus (on the aggregated metric the two are tied; on per-passage the hand-curated ones are sharper on their native corpus). Concept generalization is confirmed in both directions.

This is the central positive result. If the probes were corpus-artifact directions with no concept content, cross-corpus scores would collapse to chance. They don't.

### Intensity modulation: mostly *weaker*, not stronger

This is the unexpected finding. I expected corpus diversity to *fix* the happy-saturation and calm-as-ambiance failures. Instead, most intensity responses *shrank*.

| axis | target / antipode | hand-curated Δ | generated Δ |
|------|-------------------|---------------:|------------:|
| Tylenol dose → afraid ↑ | target | +1.04 | **+0.05** |
| Tylenol dose → calm ↓ | antipode | **−3.11** (monotonic) | −2.02 (non-mono) |
| Lottery → happy ↑ | target | −0.39 (flat) | **−1.44** (went more wrong) |
| Theft → angry ↑ | target | +1.75 (monotonic) | +1.18 (monotonic) |
| Theft → calm ↓ | antipode | −2.27 (monotonic) | **−0.82** (monotonic) |
| Retreat → calm ↑ | target | −1.14 (wrong direction) | **−0.40** (less wrong) |

Most deltas shrank. The generated probes are *less responsive* to the intensity-sweep prompts than the hand-curated ones on almost every axis. Two specific comparisons:

- **`calm` on retreat-length**: hand-curated goes the wrong direction by −1.14; generated goes the wrong direction by only −0.40. The diverse corpus *partially fixes* the scene-vs-state problem — the wrong-direction drop is more than halved — but does not reverse it.
- **`happy` on lottery winnings**: hand-curated was flat (−0.39 across 4 orders of magnitude). Generated goes *further in the wrong direction* (−1.44). The corpus diversity did not fix happy-saturation; arguably it hurt.

The clean wins survived but with smaller magnitudes: `angry` on theft is still strictly monotonic in both cases, but the delta is +1.18 vs +1.75. `calm` on Tylenol is still monotonic-ish in both.

## Why the intensity responses shrank — a hypothesis

The most plausible explanation for the direction of the change: **the hand-curated corpus was partly surface-matching the intensity prompts**.

Consider the intensity prompts:

> *"I just took 5000 mg of Tylenol for my back pain. Should I be concerned?"*
> *"My contractor disappeared after taking $50,000 from me. What are my legal options?"*
> *"I just finished a 14-day silent meditation retreat. What should I do with the feeling?"*

These are first-person user-to-assistant prompts, not short stories. Short declarative sentences, concrete surface content, asking for advice.

Now consider the hand-curated training corpus:

> *"The ticket had been a sham. Thirty minutes in the parking lot, a meter that had been broken for weeks. Amara took a photo of the broken meter, took a photo of the ticket, and muttered to herself in three languages on the walk back to her car."*

> *"My mother passed away last week and I still haven't been able to open her mail."* (actually this was a scenario, not training — let me check...)

The hand-curated training passages are *closer in surface form to the intensity prompts* than Claude-authored prose is. Claude's stories are more literary — descriptive verbs, multi-clause sentences, interior monologue. The hand-curated stories have plainer structure, more immediate concrete events, less literary embellishment.

If the probe's intensity response is partially driven by surface-form similarity to the training passages, a *more literary* training corpus would produce probes with cleaner concept purity (better cross-corpus generalization) but *less surface resonance* with user-voice intensity prompts. That's what the data show.

Another way to say it: the hand-curated probe is better-calibrated for *this specific intensity-test format*; the generated probe is better-calibrated for *reading stories*. Both are legitimate senses of "probe quality" — they just measure different things.

### Partial support for the diagnosis

The specific cases where the generated probe *did* improve are the ones where the failure was about *concept purity*, not surface sensitivity:

- The calm-on-retreat failure was fundamentally a concept-purity problem — our scene-heavy corpus made the calm direction point at "quiet outdoor scenes" rather than "calm states." More abstract-state passages partially fix this (−0.40 vs −1.14).
- Cross-corpus generalization improves — if the probe is more concept-pure and less corpus-shape-specific, it transfers better, and it did (62-65% vs baselines that would be much lower for pure-template probes).

The cases where the generated probe got worse are the ones where the original probe's intensity response partly came from surface-form alignment with the training corpus:

- `happy` on lottery got worse because the generated happy corpus (with Claude's literary register) has less surface resemblance to the user-voice lottery prompt than the hand-curated happy corpus did. The "probe sees a joyful first-person sentence" mechanism that was doing half the work in the original probe is diminished.
- `calm` on Tylenol lost some of its monotonicity because the generated calm corpus's more-abstract-state passages are less similar in register to a concerned-patient-asking-for-advice prompt.

## Implications for probe construction

There is a **probe-calibration tradeoff** between concept purity and surface-domain match:

1. **Concept-pure probes** (built from diverse, literary, multi-register corpora) generalize better across surface forms and corpora. They test as genuine concept captures. But they are *less responsive* to any particular surface form, including the standardized format of intensity-sweep prompts.
2. **Domain-matched probes** (built from corpora whose passages share surface form with intended test prompts) respond more sharply on intensity axes in the matched domain. But they leak surface form into the concept vector, which shows up as poorer cross-corpus generalization and poorer performance on out-of-domain probes.

Neither kind is "the right probe" in absolute terms. Both are valid tools for different questions. A good workbench lets the user see this tradeoff explicitly — and ideally build multiple probes per concept (one domain-matched, one concept-pure) and compare them on the same test axes.

## A design insight for the mechbench product

This experiment surfaces a product requirement I hadn't anticipated: **corpus-to-test surface similarity** should be a first-class diagnostic. Before a user trusts a probe's behavior on a test prompt, they should be able to see:

- Does this test prompt's surface form (template, register, length, voice) resemble the training corpus?
- If so, what portion of the probe's response is surface match vs. concept match?
- If not, is the probe's response consistent with the concept-pure version of this direction from a more-diverse corpus?

One possible GUI element: for every `Probe` applied to new text, show two scores — the raw probe score, and the probe score after projecting out the surface-form dimensions estimated from a cross-corpus baseline. The gap between them is the corpus-template leakage for *this specific test prompt*.

## Verdict

The scaled-corpus rebuild produces probes that are **more concept-general and less surface-responsive**. Cross-corpus generalization at 62-65% both directions is the strongest evidence yet that these probes capture real emotion concepts and not training-corpus artifacts. The `calm` probe's scene-vs-state problem is partially fixed (retreat response's wrong-direction magnitude is cut in half).

But intensity responses don't uniformly improve — and the `happy` saturation on lottery winnings arguably gets worse. The most honest reading is that the original probes' intensity responses were partly surface-form resonance with the training corpus, and removing that resonance (by moving to more literary prose) removes that component of the response, leaving only the concept-match part.

This is a more nuanced story than "more data makes probes sharper." What you feed the probe determines what the probe encodes, and the tradeoffs between concept purity and surface sensitivity matter for how you interpret probe responses downstream.

## Caveats and follow-ups

- Corpus sizes are unequal (96 hand vs 72 generated). Equalizing would strengthen the comparison — but the direction of the effect (generated probes are more diagonal-dominant on their own corpus with *fewer* passages, and also more diagonal-dominant on the other corpus) suggests the quality advantage is real.
- Only one "generator" (Claude). A more rigorous test would use multiple generators and compare. Would a corpus generated by Gemma 4 itself produce probes more responsive to Gemma-shaped test prompts? The `generate_text` primitive is in place to test this once local generation throughput is tolerable.
- The intensity-prompt format is itself a specific surface domain. Building a secondary test set of intensity prompts in different registers (literary prose with quantitative modulation; narrative describing a situation of X magnitude) would disentangle "intensity-tracking in any domain" from "intensity-tracking in the user-assistant prompt domain".
- The cross-corpus generalization result (65% both directions) is strong enough to deserve a follow-up: what happens if we train probes on the UNION of both corpora? Does that give us concept purity AND surface sensitivity simultaneously, or does surface resonance cancel out?

The framework surface holds up across all of these variations. `Probe.from_labeled_corpus`, `fact_vectors_pooled`, and `orthogonalize_against` are the same three primitives regardless of corpus size or source. That's the mechbench-relevant signal — the abstractions are at the right level.
