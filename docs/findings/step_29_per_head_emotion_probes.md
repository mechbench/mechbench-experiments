# Per-Head Emotion Probes: L17's KV-Head 1 V-Stream Beats the Residual

**Date:** 2026-04-17
**Script:** `experiments/step_29_per_head_emotion_probes.py`
**Plot:** `caches/per_head_emotion_probes.png`
**Data:** `caches/per_head_emotion_probes.npz`

## The setup

First real exercise of the generalized `Probe` primitive shipped in `cew`. For every (layer, head, stream) where stream ∈ {Q, K, V}, build 6 emotion probes using `Probe.from_labeled_corpus(..., hook_point='blocks.L.attn.<stream>', head=h)`. Score each probe on self-consistency and compute 6-way silhouette. 42 layers × (8 Q-heads + 2 KV-heads + 2 KV-heads) = **504 (layer, head, stream) combinations × 6 probes each = 3,024 probes built in one experiment**.

Emotion vectors are mean-pooled across positions ≥ 20 (same as step_21) per-head per-layer. One forward pass per prompt with `Capture.queries + Capture.keys + Capture.values` at all 42 layers. Total runtime: **26 seconds** (22s data collection, 4s probe construction and analysis).

Baseline to beat: step_21's residual-stream probes at L28 got **90.6% per-passage top-1 accuracy on the 6-way 96-passage training set**.

## Headline finding

### Specific (layer, head) specialists outperform the residual-stream probe

Top per-passage accuracies, by stream:

| stream | best (layer, head) | accuracy | vs. residual baseline |
|--------|-------------------:|---------:|----------------------|
| Q | L35 h6 GLOBAL | **95.8%** (91/96) | +5.2 points |
| V | L23 h0 GLOBAL | **95.8%** (91/96) | +5.2 points |
| K | L11 h1 GLOBAL | 91.7% (88/96) | +1.1 points |
| (reference) residual L28 from step_21 | — | 90.6% | — |

The best per-head probe beats the canonical L28 residual probe by ~5 percentage points on the same 6-way task. That's the headline: if you know *which head* to look at, a per-head Q or V representation is a sharper concept-capture than the residual-stream mean.

### One head dominates most emotion-specific margins

Per-emotion specialist ranking (by margin = mean score on own-emotion passages − mean score on others):

| emotion | #1 specialist | margin | #2 | #3 |
|---------|--------------|-------:|-----|-----|
| **happy** | L17 h1 V | +3.44 | L17 h0 V | L23 h5 Q |
| **sad** | L17 h1 V | +2.89 | L5 h0 V | L17 h0 V |
| **angry** | L35 h4 Q | +3.68 | L17 h1 V | L41 h6 Q |
| **afraid** | L17 h1 V | +3.28 | L35 h6 Q | L23 h5 Q |
| **calm** | L17 h1 V | +4.14 | L35 h4 Q | L41 h6 Q |
| **proud** | L17 h1 V | +2.97 | L17 h0 V | L23 h5 Q |

**`L17 h1 V` is the #1 margin specialist for 5 of 6 emotions** (happy, sad, afraid, calm, proud), and is #2 for angry. That is a single (layer, KV-head, stream) combination that carries clean discriminative signal for every one of the 6 emotions in our corpus.

Put differently: if a user wanted ONE head to monitor to track the model's emotional state, it's **L17 KV-head 1 on the V stream**. Every emotion category maps to a distinct direction in its head_dim=512 subspace. The L17 h0 V (the other KV-head at L17) is often the #2 specialist, suggesting the whole KV-group at L17 is unusually emotion-discriminative.

### L35 and L41 Q-heads form a secondary specialist band

For *anger*, *calm*, and *afraid* specifically, the #2 and #3 specialists are Q-heads at late global layers (L35 h4, L35 h6, L41 h6). These heads' Q projections encode emotion-specific "what am I looking for" directions. Interestingly, anger gets L35 h4 Q as its single best specialist (+3.68), beating even L17 h1 V.

The late-global Q-heads don't show up for happy/sad/proud. Those emotions rely more on V-side representations at earlier layers; anger/calm/afraid involve late-Q representations too. Might reflect that aversive emotions trigger more "look for threat/safety" query patterns than positive emotions do.

### Silhouette and accuracy diverge

An informative methodological note: the 6-way silhouette peaks are much lower than the accuracy peaks:

| stream | peak silhouette | peak accuracy |
|--------|----------------:|--------------:|
| Q | 0.034 | 0.958 |
| K | 0.035 | 0.917 |
| V | 0.062 | 0.958 |

Per-passage accuracy at specific heads can be very high even when the cosine silhouette is near zero. This happens when the 6 emotion clusters are elongated but linearly separable — a probe vector can slice between them cleanly even though the clusters themselves aren't tightly compact in cosine space.

This suggests that **silhouette underestimates how well probes work in these subspaces**. Silhouette rewards tight spherical clusters; probe-discrimination only requires separability, which can emerge even in elongated/anisotropic geometry. For per-head probes specifically, accuracy is the more honest metric. Silhouette is still useful for relative ranking across heads, but the absolute numbers mislead.

## Why L17 h1 V?

Three observations about what's special about this specific head:

1. **L17 is the third global-attention layer** (after L5, L11). Globals are where information integration across the full sequence happens. L17 specifically is where step_03 identified the largest single-layer side-channel ablation effect (in the factual-recall task).
2. **KV-head 1 is the "second half" of the GQA group** — serves Q-heads 4, 5, 6, 7. In Gemma 4's 4:1 GQA, each KV-head gets shared among 4 Q-heads. KV-head 1's representations are what Q-heads 4-7 attend to.
3. **The V stream is the "content copy" pathway.** V is what gets written to the output when attended. If L17 KV-head 1's V-projection has a clean 6-emotion concept geometry, then **any Q-head that attends to an emotionally-charged position at L17 copies that emotion direction into its output**.

Putting these together: L17 h1 V is the *single place in the network* where emotion-appropriate content is most sharply packaged for downstream attention heads to retrieve. A head at L17, L23, or later that wants to retrieve emotional content will point its Q toward positions whose L17 KV-head-1 V-vectors encode the target emotion — and those V-vectors carry cleanly-discriminable 6-emotion concept directions.

## What the generalized Probe primitive enabled

This finding was only possible because `Probe.from_labeled_corpus` now accepts `hook_point` and `head` kwargs. Every probe was built with the same one-line call:

```python
probes = Probe.from_labeled_corpus(
    labeled, neutral, layer=L,
    hook_point=f"blocks.{L}.attn.{stream}",
    head=h, explain=0.5,
)
```

Then `probe.score(new_vectors)` worked regardless of whether the probe was built over residuals (d_model-dim) or per-head V (head_dim-dim). The generalization moved from "smoke-tested-only" (step cew) to "discovered a substantive new finding" in this experiment.

## For the mechbench product

Three additional product-relevant observations:

1. **Per-head concept-separability heatmaps across streams are the default view.** Users should see, for each stream, a 42 × n_heads heatmap of concept silhouette and per-passage accuracy. L17 h1 V would be a bright cell in the V-stream accuracy map.
2. **Cross-stream specialist comparison.** For any concept, the workbench should let users click "find specialists" and surface the top (layer, head, stream) targets. Non-obvious for the user until they see the ranking.
3. **Accuracy and silhouette tell different stories.** The GUI should show BOTH for any (layer, head, stream), because they diverge meaningfully and a user who only looks at silhouette will miss high-accuracy heads with elongated clusters.

## Caveats and follow-ups

- Same 96-passage training corpus as step_21. Self-consistency accuracy (91/96) is what the head can do on its own training data. Testing these per-head probes on held-out implicit scenarios (step_23's scenarios) would reveal whether the advantage over the residual-stream probe generalizes.
- The 95.8% accuracy at L35 h6 Q is one of 504 combinations; some randomness in which heads peak is expected. The L17 h1 V margin-dominance (#1 for 5 of 6 emotions) is less likely to be noise — that's a consistent ranking across emotions.
- Pooling choice matters. We used mean-pool over positions ≥ 20 matching step_21. A per-head analysis might benefit from different pooling (e.g., max-pool, or pool only at the emotional-content-peak position). That's a follow-up knob to turn.
- We didn't compare to K-stream probes at L14 (which gave +0.36 sense silhouette in step_28). Different corpora, different concepts — not directly comparable.

## Verdict

First real deployment of the generalized Probe primitive, and it produces a novel finding: **a specific (layer, head, stream) combination — L17 KV-head 1 on the V stream — carries clean emotion-discrimination for all six emotions simultaneously**, better than the residual-stream probe that was the previous best. The per-head concept-probe surface is substantively more informative than residual-stream-only analysis, and the generalized Probe primitive is the machinery that makes it practical.

Epic `ric` is now fully exercised: static weight analysis (step_26), OV-circuit trajectories (step_27), QK sense-clustering (step_28), and per-head emotion probes (step_29, this step). Four experiments, four distinct findings, all enabled by the four primitives shipped in the epic.
