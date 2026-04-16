"""Stress tests for the centroid-decoding technique.

Three focused probes anchored against the 12-category BIG_SWEEP_96:

  1. TEMPLATE_VARIATION: 4 phrasings * 4 countries asking 'capital of X'.
     Tests whether the residual representation is operation-semantic or
     template-syntactic. Per finding 13: NOT template-invariant in
     interesting ways (Q&A phrasing routes to french/translation cluster).
  2. CROSS_LINGUAL: capital question in 5 languages * 4 countries.
     Tests whether the capital-lookup representation is language-invariant.
     Per finding 13: works for Latin-script Indo-European; fails for Chinese.
  3. CREATIVE: 8 subjective/metaphorical prompts. Per finding 13:
     individual prompts scatter but centroids decode meaningfully.

Run from project root:
    python experiments/step_13_stress_tests.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Model,
    centroid_decode, fact_vectors_at,
)
from gemma4_mlx_interp.plot import DEFAULT_CATEGORY_COLORS  # noqa: E402
from experiments.prompts import (  # noqa: E402
    BIG_SWEEP_96, STRESS_CREATIVE, STRESS_CROSS_LINGUAL, STRESS_TEMPLATE_VAR,
)

OUT_DIR = ROOT / "caches"
PROJECT_LAYER = 30


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _extract_one_layer(model, validated, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: fact_vectors_at + label extraction at one layer."""
    vecs = fact_vectors_at(model, validated, [layer])[layer]
    labels = np.array([vp.prompt.category for vp in validated])
    return vecs, labels


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\n=== Validating {len(BIG_SWEEP_96)} anchor prompts (BIG_SWEEP_96) ===")
    anchors = BIG_SWEEP_96.validate(model, verbose=False, min_confidence=0.0,
                                     require_target_match=False)
    anchor_vecs, anchor_labels = _extract_one_layer(model, anchors, PROJECT_LAYER)
    print(f"  validated: {len(anchors)}, vector matrix shape: {anchor_vecs.shape}")

    print(f"\n=== Validating {len(STRESS_TEMPLATE_VAR)} template-variation prompts ===")
    tmpl_valid = STRESS_TEMPLATE_VAR.validate(model, verbose=True, min_confidence=0.0, require_target_match=False)
    tmpl_vecs, tmpl_labels = _extract_one_layer(model, tmpl_valid, PROJECT_LAYER)

    print(f"\n=== Validating {len(STRESS_CROSS_LINGUAL)} cross-lingual prompts ===")
    cl_valid = STRESS_CROSS_LINGUAL.validate(model, verbose=True, min_confidence=0.0, require_target_match=False)
    cl_vecs, cl_labels = _extract_one_layer(model, cl_valid, PROJECT_LAYER)

    print(f"\n=== Validating {len(STRESS_CREATIVE)} creative prompts ===")
    cr_valid = STRESS_CREATIVE.validate(model, verbose=True, min_confidence=0.0, require_target_match=False)
    cr_vecs, cr_labels = _extract_one_layer(model, cr_valid, PROJECT_LAYER)

    # Anchor centroids by category
    anchor_cats = list(dict.fromkeys(anchor_labels.tolist()))
    anchor_centroids = {
        c: anchor_vecs[anchor_labels == c].mean(axis=0) for c in anchor_cats
    }

    def nearest_anchor(vec):
        sims = [(c, _cos(vec, anchor_centroids[c])) for c in anchor_cats]
        sims.sort(key=lambda x: -x[1])
        return sims[0]

    capital_centroid = anchor_centroids["capital"]

    # ---- Test 1: template variation ----
    print(f"\n{'=' * 70}")
    print("TEMPLATE VARIATION: nearest-anchor per template")
    print(f"{'=' * 70}\n")

    tmpl_groups: dict[str, list[tuple[str, float]]] = {}
    for v, l in zip(tmpl_vecs, tmpl_labels):
        tmpl_groups.setdefault(l, []).append(nearest_anchor(v))
    for tmpl in sorted(tmpl_groups):
        hits = tmpl_groups[tmpl]
        n_capital = sum(1 for c, _ in hits if c == "capital")
        print(f"  {tmpl}: {n_capital}/{len(hits)} nearest-anchor = capital")
        for c, s in hits:
            marker = "✓" if c == "capital" else "✗"
            print(f"    {marker} nearest={c:>12s}  cos={s:.4f}")

    print("\nIntra-template + cos-to-capital-centroid:")
    for tmpl in sorted(tmpl_groups):
        vs = tmpl_vecs[tmpl_labels == tmpl]
        intra = [_cos(vs[i], vs[j])
                 for i in range(len(vs)) for j in range(i + 1, len(vs))]
        to_cap = [_cos(v, capital_centroid) for v in vs]
        print(f"  {tmpl}: intra-template cos = {np.mean(intra):.4f}, "
              f"cos-to-capital-centroid = {np.mean(to_cap):.4f}")
    cap_anchor = anchor_vecs[anchor_labels == "capital"]
    intra_anchor = [_cos(cap_anchor[i], cap_anchor[j])
                    for i in range(len(cap_anchor))
                    for j in range(i + 1, len(cap_anchor))]
    print(f"  [anchor capital intra-cosine baseline: {np.mean(intra_anchor):.4f}]")

    # ---- Test 2: cross-lingual ----
    print(f"\n{'=' * 70}")
    print("CROSS-LINGUAL: nearest-anchor per language")
    print(f"{'=' * 70}\n")

    cl_groups: dict[str, list[tuple[str, float]]] = {}
    for v, l in zip(cl_vecs, cl_labels):
        cl_groups.setdefault(l, []).append(nearest_anchor(v))
    for lang in sorted(cl_groups):
        hits = cl_groups[lang]
        n_capital = sum(1 for c, _ in hits if c == "capital")
        print(f"  {lang}: {n_capital}/{len(hits)} nearest-anchor = capital")
        for c, s in hits:
            marker = "✓" if c == "capital" else "✗"
            print(f"    {marker} nearest={c:>12s}  cos={s:.4f}")

    print("\nCentroid-to-capital-anchor distances:")
    for lang in sorted(cl_groups):
        lvecs = cl_vecs[cl_labels == lang]
        centroid = lvecs.mean(axis=0)
        cos_to_cap = _cos(centroid, capital_centroid)
        intra = [_cos(lvecs[i], lvecs[j])
                 for i in range(len(lvecs)) for j in range(i + 1, len(lvecs))]
        intra_mean = float(np.mean(intra)) if intra else float("nan")
        print(f"  {lang}: cos(centroid, capital_anchor) = {cos_to_cap:.4f}, "
              f"intra-lang cos = {intra_mean:.4f}")

    print("\nCross-lingual centroid decoding (mean-subtracted, top-6):")
    overall_anchor_mean = anchor_vecs.mean(axis=0)
    for lang in sorted(cl_groups):
        lvecs = cl_vecs[cl_labels == lang]
        top = centroid_decode(model, lvecs, k=6, mean_subtract=True,
                              overall_mean=overall_anchor_mean)
        print(f"  [{lang:>8s}] " + "  ".join(f"{t!r}({p:.3f})" for t, p in top))

    # ---- Test 3: creative ----
    print(f"\n{'=' * 70}")
    print("CREATIVE PROMPTS: nearest-anchor per prompt")
    print(f"{'=' * 70}\n")

    cr_groups: dict[str, list[tuple[str, float]]] = {}
    for v, l in zip(cr_vecs, cr_labels):
        cr_groups.setdefault(l, []).append(nearest_anchor(v))
    for grp in sorted(cr_groups):
        print(f"  {grp}:")
        for c, s in cr_groups[grp]:
            print(f"    nearest={c:>12s}  cos={s:.4f}")

    print("\nCreative-category centroid decoding (mean-subtracted, top-6):")
    for grp in sorted(cr_groups):
        gvecs = cr_vecs[cr_labels == grp]
        top = centroid_decode(model, gvecs, k=6, mean_subtract=True,
                              overall_mean=overall_anchor_mean)
        print(f"  [{grp:>14s}] " + "  ".join(f"{t!r}({p:.3f})" for t, p in top))

    # ---- Plot: 3-panel PCA ----
    all_vecs = np.vstack([anchor_vecs, tmpl_vecs, cl_vecs, cr_vecs])
    proj = PCA(n_components=2).fit_transform(all_vecs)
    n_anc = len(anchor_vecs)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    def _draw_anchors(ax, highlight: str | None = None):
        for cat in set(anchor_labels):
            mask = anchor_labels == cat
            ax.scatter(
                proj[:n_anc][mask, 0], proj[:n_anc][mask, 1],
                c=DEFAULT_CATEGORY_COLORS.get(cat, "#666666"),
                label=cat if cat == highlight else None,
                s=40, alpha=0.5, edgecolors="black", linewidths=0.3,
            )

    # Panel 1: template variation
    ax = axes[0]
    _draw_anchors(ax, highlight="capital")
    start = n_anc
    end = start + len(tmpl_vecs)
    tmpl_colors = {
        "capital_tmpl1": "#000000", "capital_tmpl2": "#d62728",
        "capital_tmpl3": "#1f77b4", "capital_tmpl4": "#9467bd",
    }
    for t, c in tmpl_colors.items():
        mask = tmpl_labels == t
        ax.scatter(proj[start:end][mask, 0], proj[start:end][mask, 1],
                   c=c, label=t, s=130, alpha=1.0, marker="*",
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title("Template variation on 'capital'\n"
                 "(stars = templates, dots = anchor categories)", fontsize=10)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: cross-lingual
    ax = axes[1]
    _draw_anchors(ax, highlight="capital")
    start = n_anc + len(tmpl_vecs)
    end = start + len(cl_vecs)
    lang_colors = {
        "lang_en": "#000000", "lang_fr": "#d62728", "lang_de": "#1f77b4",
        "lang_es": "#ff7f00", "lang_zh": "#9467bd",
    }
    for t, c in lang_colors.items():
        mask = cl_labels == t
        ax.scatter(proj[start:end][mask, 0], proj[start:end][mask, 1],
                   c=c, label=t, s=130, alpha=1.0, marker="D",
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title("Cross-lingual capital prompts\n"
                 "(diamonds = languages, red dots = capital anchors)",
                 fontsize=10)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: creative
    ax = axes[2]
    for cat in set(anchor_labels):
        mask = anchor_labels == cat
        ax.scatter(proj[:n_anc][mask, 0], proj[:n_anc][mask, 1],
                   c=DEFAULT_CATEGORY_COLORS.get(cat, "#666666"),
                   label=cat, s=40, alpha=0.5,
                   edgecolors="black", linewidths=0.3)
    start = n_anc + len(tmpl_vecs) + len(cl_vecs)
    end = start + len(cr_vecs)
    cr_colors = {"creative_pref": "#000000", "creative_meta": "#d62728"}
    for t, c in cr_colors.items():
        mask = cr_labels == t
        ax.scatter(proj[start:end][mask, 0], proj[start:end][mask, 1],
                   c=c, label=t, s=130, alpha=1.0, marker="X",
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title("Creative/open-ended prompts\n"
                 "(Xs, against all 12 anchor categories)", fontsize=10)
    ax.legend(loc="best", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Stress tests at layer {PROJECT_LAYER}", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "stress_tests.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
