# Emotion Probes on Gemma 4 E4B — Diagonal is Clean, Geometry Looks Sensible

**Date:** 2026-04-17
**Script:** `experiments/step_21_emotion_probes.py`
**Plot:** `caches/emotion_probes_diagonal.png`

## The setup

Port of Anthropic's 'Emotion Concepts and their Function in a Large Language Model' (transformer-circuits.pub, 2026) to Gemma 4 E4B. End-to-end recipe, at tiny scale:

- **Corpus:** 96 short passages (6 emotions × 16), 2-4 sentences each, hand-curated. Plus 16 emotionally-neutral passages for PC-orthogonalization baseline. Total 112 prompts.
- **Extraction:** residual stream at layer 28 (~2/3 depth through E4B's 42 layers), mean-pooled over token positions ≥ 20 (skipping chat-template header).
- **Probe construction:** for each emotion, `mean(its passages) − grand_mean(across all emotion means)`, then project out the top-5 PCs of the neutral corpus (explaining ≥ 50% of neutral variance), then unit-normalize.
- **Readout:** score each training passage's residual against each of the 6 probes. A working probe set produces a strong diagonal.

Corpus size is two orders of magnitude below Anthropic's (they used ~1,200 stories per emotion; we use 16). If the pipeline works here, scaling up is mostly a throughput question.

## Result

### 6/6 diagonal, 90.6% per-passage top-1 accuracy

Each row is one emotion's passages averaged against all six probes. Diagonal cells (the emotion's own probe) are marked `*`.

| true \ probe  |  happy  |   sad   |  angry  | afraid  |  calm   |  proud  |
|---------------|--------:|--------:|--------:|--------:|--------:|--------:|
| happy         | **+6.52** | −2.35   | −1.60   | −2.35   | −1.23   | +2.04   |
| sad           | −2.29   | **+6.37** | −0.97   | −1.12   | −0.28   | −1.10   |
| angry         | −1.98   | −1.22   | **+8.06** | +1.27   | −4.58   | −1.46   |
| afraid        | −2.53   | −1.24   | +1.11   | **+7.02** | −1.34   | −3.19   |
| calm          | −1.79   | −0.41   | −5.40   | −1.81   | **+9.49** | −2.94   |
| proud         | +2.08   | −1.14   | −1.20   | −3.01   | −2.06   | **+6.64** |

**Aggregated diagonal hits: 6 / 6** (chance = 1/6 = 16.7%).
**Per-passage top-1 accuracy: 87 / 96 = 90.6%** (chance = 16.7%).

The aggregated signal is decisive — every emotion's own probe dominates its row by a factor of 2-15× over the best off-diagonal score. Per-passage performance is lower because individual passages are noisy (some sad passages have an angry undertone, some calm passages have a proud subtext), but even the single-passage accuracy at 90.6% is extraordinarily high for a 6-way classification problem built from ~16 training examples per class with no supervised training.

### Off-diagonal structure has real psychological meaning

The non-zero off-diagonal cells are not noise. They encode the valence/arousal plane that emotion theorists use to organize affective concepts:

**Shared valence (positive)** — `happy` and `proud` both score **positively** on each other's probe (+2.08 on the other's direction). They are the two positive-valence emotions in the set, and they're the only pair that share a positive cross-score. No other cross-score is larger than +1.3.

**Shared arousal (high, negative valence)** — `angry` and `afraid` score **positively** on each other (+1.27, +1.11). Both are high-arousal negative-valence emotions; their probes point in partially-overlapping directions.

**Opposite arousal (one positive valence, one negative)** — `calm` and `angry` score **strongly negatively** on each other (−4.58, −5.40). They are the two most opposite emotions in the valence/arousal plane (low-arousal-positive vs high-arousal-negative), and their cross-scores reflect that diameter.

**Sad is asymmetric** — sad's cross-scores are all modest (within ±2.3). Sad doesn't have a clean opposite in this corpus (the "opposite of sad" isn't happy; it's calm, or proud, depending on what axis you mean). Its row and column are the flattest — consistent with sad being a low-arousal emotion whose closest cousins are mild rather than sharp.

This is the geometry the paper's title is about. The probes are not just labels; they carve the activation space along axes that resemble the axes of affective psychology. And we didn't ask for that — it fell out of difference-of-means plus one cleanup step, applied to 96 short passages.

### Which probe is sharpest?

Calm scores highest on its own passages at +9.49; angry at +8.06. The weakest diagonal is sad (+6.37). A guess at why: the sad corpus has the most template diversity (a lot of different people in a lot of different situations of loss) and the least strong single-token vocabulary (sad vocabulary is diffuse — 'tears', 'grief', 'empty', 'quiet', and none dominate), so its mean vector is less sharply anchored in activation space. The calm corpus, conversely, has a very consistent scene structure (stillness, tea, lakes, books, slow activities) and that template consistency translates into a tighter probe direction.

That caveat matters. It means some of the probe's apparent strength reflects corpus consistency, not just concept distinctiveness. A harder test — scoring probes on scenarios the model has not seen that implicitly evoke the emotion without the corpus's template cues — is what would validate the concept claim. (That's the next experiment, `beads-7wg`.)

## Framework upgrades landed

Every new primitive added in this experiment arc generalizes beyond emotions:

- **`fact_vectors_pooled(..., start=20, end=None)`** — mean-pool residuals across a range of token positions per prompt. Any concept distributed across a passage (register, sentiment, topic, style) can now be extracted with one call instead of inline pooling boilerplate.
- **`orthogonalize_against(vectors, baseline, explain=0.5)`** — project out the top-variance subspace of a baseline corpus from any set of vectors. Generalizes mean-subtraction to PC-subtraction; useful anywhere we want to de-noise a concept direction against a known-neutral corpus.
- **`Probe`** dataclass with `.score()`, `.from_vector()`, `.from_corpus()`, `.from_labeled_corpus()` — first-class persistent concept vectors with the baseline mean and the orthogonalizer carried along. Any probe is now a reusable object that scores any residual at its associated layer.

These are the primitive-layer additions that matter for the broader project goal. An emotion probe is a concrete use case; the abstractions underneath can build sentiment probes, register probes, modality probes, or any other concept-direction workflow.

## Verdict

The Anthropic recipe ports to Gemma 4 E4B, at 100× smaller corpus size, with a clean result. Difference-of-means plus PC-orthogonalization yields probes that distinguish 6 emotions at 90%+ per-passage accuracy on their own training corpus, with sensible off-diagonal structure matching valence/arousal geometry.

Caveats:

- The self-consistency test is weak evidence. Probes should score highest on the corpus they were built from — that's by construction. The real test is generalization (next experiment: `beads-7wg`, score scenarios that evoke the emotion implicitly).
- Small corpus. 16 passages per emotion is enough to get probes that work at this readout, but we don't yet know whether the probe *quality* (especially off-diagonal cleanliness) plateaus at this scale or keeps improving with more data (that's `beads-f4k`).
- One layer (L28). Anthropic notes that the optimal readout layer is "about two-thirds through the model"; we adopted that choice by fiat. A per-layer sweep would tell us whether E4B's emotional readout layer matches that rule of thumb.
- Six emotions. Anthropic uses 171. Scaling up the probe vocabulary is a mechanical exercise once the pipeline works.

## Caveats and follow-ups

The four unblocked children of epic `5at`:

- **`beads-7wg` Validation:** score these probes against Anthropic's Table 2 — scenarios that imply an emotion without using its word. That separates "the probe learned the concept" from "the probe memorized the training corpus's surface vocabulary."
- **`beads-4b2` Logit-lens the probes:** project each probe.vec through the tied unembed and report top upweighted/downweighted tokens. The paper's Table 1 was multilingual and specific (e.g. `happy → [excited, excitement, exciting, happ, celeb]`); reproducing that on Gemma 4 is a strong independent sanity check.
- **`beads-e3x` Intensity modulation:** construct prompts varying numerical quantities that modulate emotion intensity (tylenol dosage, hours since eating). A working probe should respond monotonically.
- **`beads-f4k` Scale up via model-driven corpus generation:** once `beads-05o` (corpus-generation helper) lands, rebuild probes from 100+ passages per emotion and check whether diagonal sharpens.

The epic's goal — to extract a reusable `Probe` primitive — is met. The primitive is in the framework; it has one working use case; the follow-up experiments are all straight applications of the same primitive.
