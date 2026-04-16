"""Geometric analysis of vocab-opaque fact vectors in Gemma 4 E4B.

Causal tracing (finding 09) showed the residual stream at the last
subject-entity token in middle layers causally determines the factual
answer — but isn't decodable through the unembed (finding 08). This
experiment asks: does that opaque representation have geometric structure?

Uses the 5 'classic' categories (capital, element, author, landmark,
opposite) from BIG_SWEEP_96, extracts fact vectors at three depths
(pre-engine-room layer 5, in-engine-room layer 15, post-handoff layer 30),
and runs cosine similarity, PCA, k-means purity, silhouette, NN purity.

Per finding 10: the categories cluster cleanly with depth — 100%
nearest-neighbor purity at layer 15, perfect k-means purity, silhouette
+0.6 by layer 30.

Run from project root:
    python experiments/step_10_fact_vector_geometry.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    BIG_SWEEP_96, Model, PromptSet,
    cluster_purity, cosine_matrix, fact_vectors_at,
    intra_inter_separation, nearest_neighbor_purity,
    pca_scatter, silhouette_cosine, similarity_heatmap,
)

OUT_DIR = ROOT / "caches"
EXTRACT_LAYERS = [5, 15, 30]
CATEGORIES = ("capital", "element", "author", "landmark", "opposite")


def _stats_at(vecs: np.ndarray, labels: np.ndarray) -> dict:
    intra, inter, sep = intra_inter_separation(vecs, labels)
    cats = list(dict.fromkeys(labels.tolist()))
    km = KMeans(n_clusters=len(cats), n_init=10, random_state=42).fit(vecs)
    purity = cluster_purity(labels.tolist(), km.labels_.tolist())
    sil = silhouette_cosine(vecs, labels)
    return {
        "intra_mean": intra,
        "inter_mean": inter,
        "separation": sep,
        "kmeans_purity": purity,
        "silhouette": sil,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    five_cats = PromptSet(
        name="STEP10_40",
        prompts=tuple(p for p in BIG_SWEEP_96 if p.category in CATEGORIES),
    )
    print(f"\nValidating {len(five_cats)} prompts...\n")
    valid = five_cats.validate(model, verbose=False)
    print(f"  validated: {len(valid)} of {len(five_cats)}")
    labels = np.array([vp.prompt.category for vp in valid])

    print(f"\nExtracting fact vectors at layers {EXTRACT_LAYERS}...")
    vecs_by_layer = fact_vectors_at(model, valid, EXTRACT_LAYERS)

    # ---- Per-layer plot grid + stats ----
    fig, axes = plt.subplots(len(EXTRACT_LAYERS), 2,
                              figsize=(14, 5 * len(EXTRACT_LAYERS)))

    stats_by_layer = {}
    for row, L in enumerate(EXTRACT_LAYERS):
        vecs = vecs_by_layer[L]
        stats = _stats_at(vecs, labels)
        stats_by_layer[L] = stats

        similarity_heatmap(
            vecs, labels, ax=axes[row, 0],
            title=f"Layer {L} — cosine similarity",
        )
        pca_scatter(
            vecs, labels, ax=axes[row, 1],
            show_legend=(L == EXTRACT_LAYERS[-1]),
            title=f"Layer {L} — PCA",
        )

        print(f"\nLayer {L} statistics:")
        print(f"  intra-category mean cosine: {stats['intra_mean']:+.4f}")
        print(f"  inter-category mean cosine: {stats['inter_mean']:+.4f}")
        print(f"  separation:                 {stats['separation']:+.4f}")
        print(f"  k-means purity (k=5):       {stats['kmeans_purity']:.3f} "
              f"(chance = 0.200)")
        print(f"  silhouette (cosine, ground-truth labels): "
              f"{stats['silhouette']:+.4f}")

    fig.suptitle(f"Fact vector geometry in Gemma 4 E4B — "
                 f"{len(valid)} prompts, {len(CATEGORIES)} categories",
                 fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "fact_vector_geometry.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ---- Nearest-neighbor analysis at layer 15 ----
    print(f"\n{'=' * 60}")
    print("Nearest-neighbor analysis at layer 15 (engine room)")
    print(f"{'=' * 60}\n")

    L = 15
    vecs = vecs_by_layer[L]
    sim = cosine_matrix(vecs)
    np.fill_diagonal(sim, -np.inf)
    nn_rate, hits = nearest_neighbor_purity(vecs, labels)

    for i, vp in enumerate(valid):
        nn = int(np.argmax(sim[i]))
        nn_vp = valid[nn]
        match = "OK" if hits[i] else "  "
        print(f"  {match}  [{labels[i]:>9s}] {vp.prompt.subject:>12s} → "
              f"NN: [{labels[nn]:>9s}] {nn_vp.prompt.subject:>12s}  "
              f"(cos={sim[i, nn]:.3f})")

    print(f"\n  NN same-category hit rate: {nn_rate:.1%} "
          f"({int(hits.sum())}/{len(valid)}, chance ≈ 17.9%)")


if __name__ == "__main__":
    main()
