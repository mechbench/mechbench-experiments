# Operation-Word Disambiguation: Factorization Survives

**Date:** 2026-04-16
**Script:** `experiments/step_15_operation_disambiguation.py`
**Plot:** `caches/operation_disambiguation.png`, `caches/operation_disambiguation_4cell.png`
**Proposal:** `docs/proposals/factorization-experiments.md` Experiment 1

## Background

Findings 10–12 showed that mid-layer subject-position activations cluster cleanly by relational category (capital vs element vs author etc.). The proposal flagged a remaining concern: the categories are confounded with surface-token presence — every capital-lookup prompt contains the literal word "capital", every element prompt contains "chemical symbol", etc. The cluster signal might be tracking those tokens rather than the abstract operation.

This experiment is the proposal's clean disambiguation.

## Setup

A 2×2 design × 8 prompts each = 32 prompts:

| | "capital" present | "capital" absent |
|---|---|---|
| **Capital-lookup**     | **A1** (8): "Complete this sentence with one word: The capital of {country} is" | **A2** (8): "The administrative center of {country} is named, in one word," |
| **Letter-counting**    | **B1** (8): "Complete this sentence with one number: The number of letters in the word '{X}' is" where X ∈ {capital, capitals, capitalism, capitalist, capitalize, capitalized, capitalizing, capitalization} | **B2** (8): same template, X ∈ {elephant, mountain, computer, philosopher, butterfly, helicopter, restaurant, umbrella} |

For each prompt, extract the residual stream at the subject position (country for A1/A2; operand word for B1/B2) at layer 30 — same depth that produced the original clustering result. Compute clustering metrics under three groupings:

1. **Operation-type** (A vs B, k=2): the factorization hypothesis
2. **Word-presence** (1 vs 2, k=2): the surface-token confound hypothesis
3. **4-cell** (A1/A2/B1/B2, k=4): mixed model

## Results

### Operation-type wins decisively

| Grouping | Separation | NN same-group | k-means purity | Silhouette |
|----------|-----------:|--------------:|---------------:|-----------:|
| **Operation-type** (A vs B) | **+0.1130** | **1.000** | **1.000** | **+0.7396** |
| Word-presence (1 vs 2)      | +0.0178     | 0.906       | 0.500          | +0.1444    |
| 4-cell (A1/A2/B1/B2)        | +0.1050     | 0.906       | 0.750          | +0.5302    |

The operation-type hypothesis beats word-presence by **6.35× in separation** and by every other metric. K-means with k=2 perfectly partitions A from B (purity 1.000); the same with the word-presence label is at coin-flip chance (0.500). Nearest-neighbor agreement is 100% under operation-type vs 90.6% under word-presence.

The factorization claim from finding 12 survives the disambiguation.

### The visual is striking

In 2D PCA the 32 prompts form three obvious clumps:

- **Top right**: the 8 A1 prompts ("capital of X is" → capital city)
- **Bottom right**: the 8 A2 prompts ("administrative center of X is named, in one word," → capital city)
- **Left**: the 16 counting prompts (B1 + B2 mixed), arrayed in a single elongated cluster

The dominant axis (left vs right: roughly PC1) is operation-type — counting prompts vs lookup prompts. Coloring the same plot by word-presence makes the picture obvious: capital-present points (green) are scattered in BOTH the lookup cluster (A1) and the counting cluster (B1), confirming the word's presence isn't what's driving cluster membership.

### A real but smaller secondary effect: template structure

The A1 vs A2 split (top-right vs bottom-right) is real and worth naming. Both are capital-lookup prompts predicting the same capital cities, but A1 contains "capital" and uses the "Complete this sentence with one word" preamble, while A2 uses a paraphrase without "capital" or the preamble. The two sub-clusters are clearly separated.

Within the counting cluster (left), B1 (operand contains "capital") and B2 (operand doesn't) mix together — there's no visible vertical split that aligns with the word's presence in the operand word.

So **template/preamble structure matters; bare operand-word identity matters much less**. This is consistent with finding 13's stress-test result that template variation has a bigger effect than expected. The "capital" word itself doesn't carve out a representational subspace; the template that contains "capital" does.

The 4-cell k-means purity (0.750 vs chance 0.250) reflects this: most prompts go to the right cluster cell, but A1 vs A2 sometimes get confused (because they're sub-clusters of the same lookup mega-cluster), and B1 vs B2 can't be separated at all.

### Numbers worth keeping

- 32/32 prompts validated (`min_confidence=0.0`, `require_target_match=False`)
- The model's letter-counting answers are mostly wrong (e.g., counts 'capital' as 8 letters; counts 'philosopher' as 9). This doesn't matter for the geometric analysis, but worth flagging if anyone wonders whether the counting cluster might be "the model is confused" rather than "the model is running counting cognition." The cluster is tight regardless.

## Interpretation

The mid-layer subject-position activation **encodes which cognitive operation is running**, factored apart from which specific tokens appear in the prompt. When the operation is "look up the capital of X", the prompt's representation occupies one region of activation space. When the operation is "count the letters in word X", it occupies a different region — even when X happens to contain the word "capital".

This is the strongest version of the operation-factorization claim from findings 10–12. The earlier centroid-clustering result was not a surface-token artifact:

- A1 prompts contain "capital" and so do B1 prompts → they DON'T cluster with each other.
- A1 prompts contain "capital" and A2 prompts don't → they DO cluster with each other (modulo the within-lookup template split).

The operand word's identity per se is a weak influence. The OPERATION the model is preparing to perform is a strong one.

### What "encodes the operation" likely means mechanistically

The model is processing each prompt up to the subject position; by layer 30 it has decided what kind of computation it's about to perform when the rest of the sequence asks for the answer. Capital-lookup engages the model's geographic/political knowledge subspace; letter-counting engages a fundamentally different machinery (a string-manipulation routine, however poorly the model executes it). These engage non-overlapping representational regions at the subject position.

The prompt-template artifact (A1 vs A2 sub-cluster split) is a softer effect within the same cognitive operation — same operation, different way of phrasing the request, slightly different residual neighborhoods.

## Implications for the next experiment

This is exactly the result the proposal needed for Experiment 2 to be worth running. If the geometry tracked surface-token presence rather than operation-type, the centroid would be a passive correlate of "the word 'capital' appeared" — a totally uninteresting steering target. Now that we've shown the geometry tracks the operation itself, injecting that vector into a different prompt becomes a real test of whether the centroid can act as a *functional handle* for the operation, not just a diagnostic probe.

## Caveats

- Single layer (30). The proposal didn't ask us to vary layer; we picked 30 because that's where the original BIG_SWEEP_96 numerical headline reproduces. Worth checking whether the operation-type axis is equally clean at earlier (15) or later (35) depths, but not for this writeup.
- Two specific operations. We disambiguated capital-lookup vs letter-counting. The factorization claim might be robust on this pair and weaker on others. A more thorough version would extend to a 3- or 4-operation disambiguation.
- The "letter-counting" cluster is one cluster, not two — B1 and B2 don't separate. We can't tell from this experiment whether the model is silently encoding "this prompt contains the word 'capital'" anywhere in its representation; we can only say that whatever encoding exists isn't strong enough to overcome the operation-type signal at the subject position.
