"""End-to-end emotion-probe construction + self-consistency sanity check.

Port of Anthropic's 'Emotion Concepts and their Function in a LLM'
methodology (transformer-circuits.pub, 2026) to Gemma 4 E4B, using the
hand-curated mini corpus in experiments/prompts/emotion_stories.py.

Recipe:
  1. Pool residuals across a range of token positions per passage, at a
     readout layer about two-thirds of the way through the model. For
     Gemma 4 E4B (42 layers) that's L28.
  2. For each emotion, compute (mean over its passages) - (grand mean
     across all emotions' means). This is the difference-of-means
     centroid, equally weighting emotions.
  3. Project out the top-variance PCs (50% variance) of a neutral
     baseline corpus from each emotion vector.
  4. Wrap each as a Probe object for reuse.

Sanity check: run each TRAINING passage through the model again, score
it against all 6 probes at the same layer, build a per-passage score
matrix, and aggregate per-emotion. A working probe set produces a
strong diagonal.

Run from project root:
    python experiments/step_21_emotion_probes.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Model, Probe, fact_vectors_pooled,
)
from experiments.prompts import (  # noqa: E402
    EMOTION_NEUTRAL_BASELINE, EMOTION_STORIES_TINY,
)

OUT_DIR = ROOT / "caches"
LAYER = 28  # ~2/3 of the way through E4B's 42 layers
POOL_START = 20  # skip chat-template header tokens


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    # ---- Validate both corpora (no filtering; we want all passages kept) ----
    print(f"\nValidating {len(EMOTION_STORIES_TINY)} emotion stories...")
    emotion_valid = EMOTION_STORIES_TINY.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    print(f"  {len(emotion_valid)} of {len(EMOTION_STORIES_TINY)} validated.")

    print(f"\nValidating {len(EMOTION_NEUTRAL_BASELINE)} neutral baseline passages...")
    neutral_valid = EMOTION_NEUTRAL_BASELINE.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    print(f"  {len(neutral_valid)} of {len(EMOTION_NEUTRAL_BASELINE)} validated.")

    labels = emotion_valid.labels
    emotion_names = emotion_valid.categories  # ordered
    print(f"\nEmotions: {emotion_names}")

    # ---- Extract position-pooled residuals at L28 ----
    print(f"\nExtracting pooled residuals at layer {LAYER} "
          f"(mean over positions >= {POOL_START})...")
    emotion_vecs = fact_vectors_pooled(
        model, emotion_valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    print(f"  emotion_vecs: {emotion_vecs.shape}")

    neutral_vecs = fact_vectors_pooled(
        model, neutral_valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    print(f"  neutral_vecs: {neutral_vecs.shape}")

    # ---- Build probes ----
    print(f"\nBuilding probes via difference-of-means + PC-orthogonalization "
          f"(50% baseline variance projected out)...")
    labeled_by_emotion = {
        e: emotion_vecs[labels == e] for e in emotion_names
    }
    probes = Probe.from_labeled_corpus(
        labeled_by_emotion, neutral_vecs, layer=LAYER, explain=0.5,
    )
    for name, p in probes.items():
        ortho_rank = 0 if p.orthogonalizer is None else p.orthogonalizer.shape[0]
        print(f"  {name:>20s}: vec_norm={float(np.linalg.norm(p.vec)):.4f}  "
              f"ortho_rank={ortho_rank}")

    # ---- Self-consistency: score each training passage against all probes ----
    print(f"\n{'=' * 70}")
    print(f"Self-consistency: score training passages against all probes")
    print(f"{'=' * 70}")

    # Score matrix: [n_passages, n_probes]
    probe_names = list(probes.keys())
    scores = np.stack(
        [probes[p].score(emotion_vecs) for p in probe_names],
        axis=1,
    )
    # Aggregate per-emotion (mean over its own passages)
    agg = np.zeros((len(emotion_names), len(probe_names)), dtype=np.float32)
    for i, e in enumerate(emotion_names):
        mask = labels == e
        agg[i] = scores[mask].mean(axis=0)

    # Print as table
    print()
    header = "true / probe"
    print(f"  {header:>20s}  " +
          "  ".join(f"{p.replace('emotion_', ''):>8s}" for p in probe_names))
    print("  " + "-" * (22 + 10 * len(probe_names)))
    diag_hits = 0
    for i, e in enumerate(emotion_names):
        row = agg[i]
        arg = int(np.argmax(row))
        hit = probe_names[arg] == e
        diag_hits += int(hit)
        marker = " *" if hit else "  "
        print(f"  {e:>20s}  " +
              "  ".join(
                  f"{row[j]:+8.4f}" + (" *" if j == arg else "  ")
                  for j in range(len(probe_names))
              )[:-2])  # trim trailing padding
    print()
    print(f"  Diagonal hits: {diag_hits} / {len(emotion_names)} "
          f"(chance = 1/{len(emotion_names)} = {100/len(emotion_names):.0f}%)")

    # Also: per-passage top-1 accuracy
    per_passage_top1 = np.array(
        [probe_names[int(np.argmax(scores[j]))] for j in range(len(scores))]
    )
    correct = (per_passage_top1 == labels).sum()
    print(f"  Per-passage top-1 accuracy: {correct} / {len(labels)} "
          f"= {100 * correct / len(labels):.1f}%")

    # ---- Visualizations: agg heatmap + per-passage heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: aggregated (emotion x probe)
    ax = axes[0]
    im = ax.imshow(agg, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(agg).max(), vmax=np.abs(agg).max())
    ax.set_xticks(range(len(probe_names)))
    ax.set_xticklabels([p.replace("emotion_", "") for p in probe_names],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(emotion_names)))
    ax.set_yticklabels([e.replace("emotion_", "") for e in emotion_names])
    ax.set_xlabel("probe")
    ax.set_ylabel("true emotion")
    ax.set_title(f"Aggregated probe scores at L{LAYER}\n"
                 f"(mean over each emotion's passages)")
    for i in range(agg.shape[0]):
        for j in range(agg.shape[1]):
            ax.text(j, i, f"{agg[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(agg[i, j]) > agg.max() * 0.5 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: per-passage scores
    ax = axes[1]
    # Order by true emotion so blocks of the same emotion are visually grouped
    order = np.argsort([emotion_names.index(l) for l in labels])
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    im = ax.imshow(scores_sorted, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(scores).max(), vmax=np.abs(scores).max())
    ax.set_xticks(range(len(probe_names)))
    ax.set_xticklabels([p.replace("emotion_", "") for p in probe_names],
                       rotation=30, ha="right")
    # Mark emotion-block boundaries on the y-axis
    ax.set_yticks([])
    ax.set_xlabel("probe")
    ax.set_ylabel("passage (grouped by true emotion)")
    ax.set_title(f"Per-passage probe scores at L{LAYER}")
    # Draw horizontal lines at emotion boundaries
    cur = labels_sorted[0]
    for y in range(len(labels_sorted)):
        if labels_sorted[y] != cur:
            ax.axhline(y - 0.5, color="black", linewidth=0.8)
            cur = labels_sorted[y]
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Emotion-probe self-consistency — does each passage score highest "
        "on its own emotion's probe?",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "emotion_probes_diagonal.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ---- Save probes for downstream experiments ----
    probes_path = OUT_DIR / "emotion_probes.npz"
    np.savez_compressed(
        probes_path,
        layer=np.array([LAYER]),
        emotion_names=np.array(emotion_names),
        probe_vecs=np.stack([probes[e].vec for e in emotion_names]),
        baseline_mean=probes[emotion_names[0]].baseline_mean,
        orthogonalizer=probes[emotion_names[0]].orthogonalizer,
    )
    print(f"Wrote {probes_path}")


if __name__ == "__main__":
    main()
