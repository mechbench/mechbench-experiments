"""Centroid unembed projection and vector arithmetic on fact vectors.

Three sub-experiments using the BIG_SWEEP_96 prompt set's original 5
'classic' categories (capital, element, author, landmark, opposite):

  1. Centroid projection through the unembed at multiple depths
     (the multilingual concept-decoding result from finding 11).
  2. Within-category diff-vector consistency: do v(France)-v(Japan)-style
     differences share a direction?
  3. Cross-category same-answer alignment: do prompts predicting the same
     answer (e.g. 'Paris' from both Eiffel-Tower and France) get closer
     after subtracting the category centroid?

Run from project root:
    python experiments/step_11_centroid_and_arithmetic.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Model, PromptSet, centroid_decode, fact_vectors_at,
)
from experiments.prompts import BIG_SWEEP_96  # noqa: E402

# The five original step_10 categories. step_11 used these specifically;
# the bigger 12-category sweep is in step_12.
CATEGORIES = ("capital", "element", "author", "landmark", "opposite")
PROJECT_LAYERS = [15, 25, 30, 35, 41]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("Loading model...")
    model = Model.load()

    # Pull the 40 prompts from BIG_SWEEP_96 in the 5 classic categories.
    five_cats = PromptSet(
        name="STEP11_40",
        prompts=tuple(p for p in BIG_SWEEP_96 if p.category in CATEGORIES),
    )
    print(f"\nValidating {len(five_cats)} prompts in 5 categories...\n")
    valid = five_cats.validate(model, verbose=False)
    print(f"  validated: {len(valid)} of {len(five_cats)}")
    answers = [vp.target_token for vp in valid]
    labels = np.array([vp.prompt.category for vp in valid])

    print(f"\nExtracting fact vectors at layers {PROJECT_LAYERS}...")
    vecs_by_layer = fact_vectors_at(model, valid, PROJECT_LAYERS)
    print(f"(Extracted {len(valid)} vectors of dim 2560 at {len(PROJECT_LAYERS)} layers)")

    # ------------------------------------------------------------------
    # Experiment 1: centroid projection through the unembed
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Experiment 1: centroid projection through unembed")
    print(f"{'=' * 70}")

    for L in PROJECT_LAYERS:
        vecs = vecs_by_layer[L]
        overall_mean = vecs.mean(axis=0)
        print(f"\n--- Layer {L} ---")
        for cat in CATEGORIES:
            mask = labels == cat
            if not mask.any():
                continue
            cat_vecs = vecs[mask]
            raw = centroid_decode(model, cat_vecs, k=6, mean_subtract=False)
            ms = centroid_decode(model, cat_vecs, k=6, mean_subtract=True,
                                 overall_mean=overall_mean)
            print(f"\n  [{cat:>9s}] centroid top-6:")
            print("    raw:      " + "  ".join(f"{t!r}({p:.3f})" for t, p in raw))
            print("    mean-sub: " + "  ".join(f"{t!r}({p:.3f})" for t, p in ms))

    # ------------------------------------------------------------------
    # Experiment 2: within-category diff vector consistency
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Experiment 2: within-category diff vector consistency")
    print(f"{'=' * 70}")
    print("\nFor each category, compute all pairwise diff vectors v(A) - v(B).")
    print("Mean cosine between diff vectors. High = consistent 'between-facts'")
    print("direction; near-zero = fact-specific, no linear structure.\n")

    for L in PROJECT_LAYERS:
        vecs = vecs_by_layer[L]
        print(f"--- Layer {L} ---")
        for cat in CATEGORIES:
            indices = np.where(labels == cat)[0]
            if len(indices) < 2:
                continue
            diffs = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    diffs.append(vecs[indices[i]] - vecs[indices[j]])
            diffs = np.array(diffs)
            n_d = len(diffs)
            sims = []
            for i in range(n_d):
                for j in range(i + 1, n_d):
                    sims.append(_cos(diffs[i], diffs[j]))
            mean_sim = float(np.mean(sims)) if sims else float("nan")
            centroid = vecs[indices].mean(axis=0)
            cn = float(np.mean([np.linalg.norm(centroid - vecs[i]) for i in indices]))
            dn = float(np.mean([np.linalg.norm(d) for d in diffs]))
            print(f"  [{cat:>9s}] n_diffs={n_d:>3d}  mean pairwise cos(diffs): {mean_sim:+.4f}  "
                  f"|centroid-to-item|≈{cn:.2f}  |diff|≈{dn:.2f}")

    # ------------------------------------------------------------------
    # Experiment 3: cross-category same-answer alignment
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("Experiment 3: cross-category same-answer alignment")
    print(f"{'=' * 70}")

    # Find pairs that share an answer across categories. The natural ones in
    # the 5-cat set are Paris (capital France + landmark Eiffel) and Rome
    # (capital Italy + landmark Colosseum).
    same_pairs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ai, aj = answers[i].lower(), answers[j].lower()
            if ai == aj and labels[i] != labels[j]:
                same_pairs.append((i, j))
    if not same_pairs:
        for city in ("Paris", "Rome"):
            indices = [i for i, a in enumerate(answers) if a.strip() == city]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    if labels[a] != labels[b]:
                        same_pairs.append((a, b))

    if not same_pairs:
        print("\n  No cross-category same-answer pairs found in the 5-cat set.")
        return

    print(f"\nFound {len(same_pairs)} cross-category same-answer pair(s):")
    for i, j in same_pairs:
        print(f"  {valid[i].prompt.subject!r}({labels[i]}) + "
              f"{valid[j].prompt.subject!r}({labels[j]}) → both say {answers[i]!r}")

    # Random control pairs: different category, different answer.
    rng = np.random.default_rng(42)
    control_pairs = []
    for _ in range(10):
        i = int(rng.integers(len(valid)))
        j = int(rng.integers(len(valid)))
        while j == i or labels[i] == labels[j] or answers[i] == answers[j]:
            j = int(rng.integers(len(valid)))
        control_pairs.append((i, j))

    print()
    for L in PROJECT_LAYERS:
        vecs = vecs_by_layer[L]
        centroids = {c: vecs[labels == c].mean(axis=0) for c in CATEGORIES if (labels == c).any()}
        print(f"--- Layer {L} ---")

        raw, sub = [], []
        for i, j in same_pairs:
            raw.append(_cos(vecs[i], vecs[j]))
            sub.append(_cos(vecs[i] - centroids[labels[i]],
                            vecs[j] - centroids[labels[j]]))
        print(f"  same-answer cross-category ({len(same_pairs)} pairs):")
        print(f"    cos before centroid-sub: {np.mean(raw):+.4f}")
        print(f"    cos after  centroid-sub: {np.mean(sub):+.4f}  "
              f"(change: {np.mean(sub) - np.mean(raw):+.4f})")

        raw_c, sub_c = [], []
        for i, j in control_pairs:
            raw_c.append(_cos(vecs[i], vecs[j]))
            sub_c.append(_cos(vecs[i] - centroids[labels[i]],
                              vecs[j] - centroids[labels[j]]))
        print(f"  different-answer cross-category ({len(control_pairs)} control pairs):")
        print(f"    cos before centroid-sub: {np.mean(raw_c):+.4f}")
        print(f"    cos after  centroid-sub: {np.mean(sub_c):+.4f}  "
              f"(change: {np.mean(sub_c) - np.mean(raw_c):+.4f})")


if __name__ == "__main__":
    main()
