"""Side-channel ablation impact on homonym sense disambiguation.

Port of step_03 (MatFormer side-channel ablation) to the
sense-disambiguation question from step_17.

Prediction (from beads issue xhh): minimal effect.

The MatFormer per-layer-input gate feeds per-token IDENTITY into every
decoder block. But all 32 HOMONYM_CAPITAL_ALL prompts share the same
homonym at the 'capital' position; the side-channel input there is
identical across the 4 sense cohorts. Sense disambiguation therefore
HAS to flow from the residual stream + attention (i.e. from the
disambiguating context tokens), not from the side-channel itself. If
the prediction holds, ablating the side-channel everywhere should
leave sense separation roughly intact.

Conditions:
  baseline      no ablation (verify step_17 numbers)
  side_channel  ablate the side-channel at all 42 layers

For each: extract residuals at the 'capital' position at two readout
layers (L12 = geometric peak from step_17, L41 = late decoding readout)
and compute silhouette / NN purity / k-means purity by sense.

Run from project root:
    python experiments/step_19_homonym_side_channel.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Ablate, Model,
    cluster_purity, fact_vectors_at, intra_inter_separation,
    iterate_clusters, nearest_neighbor_purity, silhouette_cosine,
)
from experiments.prompts import HOMONYM_CAPITAL_ALL  # noqa: E402

OUT_DIR = ROOT / "caches"
READOUT_LAYERS = [12, 41]


def _separation_stats(vecs: np.ndarray, labels: np.ndarray, k: int) -> dict:
    intra, inter, sep = intra_inter_separation(vecs, labels)
    nn_rate, _ = nearest_neighbor_purity(vecs, labels)
    sil = silhouette_cosine(vecs, labels)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(vecs)
    purity = cluster_purity(labels.tolist(), km.labels_.tolist())
    return {"intra": intra, "inter": inter, "sep": sep,
            "nn_rate": nn_rate, "purity": purity, "sil": sil}


def _print_stats(name: str, stats: dict, k: int):
    print(f"  {name:>15s}  intra={stats['intra']:+.4f}  "
          f"inter={stats['inter']:+.4f}  sep={stats['sep']:+.4f}  "
          f"NN={stats['nn_rate']:.3f}  purity={stats['purity']:.3f}  "
          f"sil={stats['sil']:+.4f}  (chance NN/purity = {1/k:.3f})")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating {len(HOMONYM_CAPITAL_ALL)} prompts (no filtering)...")
    valid = HOMONYM_CAPITAL_ALL.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    n = len(valid)
    print(f"  {n} of {len(HOMONYM_CAPITAL_ALL)} validated.\n")

    labels = valid.labels
    senses = valid.categories
    k = len(senses)
    print(f"Senses ({k}): {senses}")
    print(f"Counts:        {dict((s, int(np.sum(labels == s))) for s in senses)}\n")

    # ---- Baseline ----
    print("=" * 70)
    print("Baseline (no ablation) -- verify step_17 numbers")
    print("=" * 70)
    base_vecs = fact_vectors_at(model, valid, READOUT_LAYERS)
    base_stats = {}
    for L in READOUT_LAYERS:
        base_stats[L] = _separation_stats(base_vecs[L], labels, k=k)
        print(f"\n[layer {L}]")
        _print_stats("baseline", base_stats[L], k=k)

    # ---- Side-channel ablation (all layers) ----
    print(f"\n{'=' * 70}")
    print("Side-channel ablation (all 42 layers)")
    print("=" * 70)
    abl_vecs = fact_vectors_at(
        model, valid, READOUT_LAYERS,
        interventions=[Ablate.side_channel()],
    )
    abl_stats = {}
    for L in READOUT_LAYERS:
        abl_stats[L] = _separation_stats(abl_vecs[L], labels, k=k)
        print(f"\n[layer {L}]")
        _print_stats("baseline", base_stats[L], k=k)
        _print_stats("side_chan_off", abl_stats[L], k=k)
        d_sil = abl_stats[L]["sil"] - base_stats[L]["sil"]
        d_nn = abl_stats[L]["nn_rate"] - base_stats[L]["nn_rate"]
        d_pur = abl_stats[L]["purity"] - base_stats[L]["purity"]
        d_sep = abl_stats[L]["sep"] - base_stats[L]["sep"]
        print(f"  {'delta':>15s}  Δsep={d_sep:+.4f}  ΔNN={d_nn:+.3f}  "
              f"Δpurity={d_pur:+.3f}  Δsil={d_sil:+.4f}")

    # ---- Verdict ----
    print(f"\n{'=' * 70}")
    print("Verdict")
    print(f"{'=' * 70}")
    for L in READOUT_LAYERS:
        d_sil = abl_stats[L]["sil"] - base_stats[L]["sil"]
        d_nn = abl_stats[L]["nn_rate"] - base_stats[L]["nn_rate"]
        rel_sil = d_sil / max(abs(base_stats[L]["sil"]), 1e-6)
        print(f"  layer {L}: Δsilhouette = {d_sil:+.4f}  ({rel_sil:+.1%} of baseline)")
        print(f"           ΔNN          = {d_nn:+.3f}")
    print()
    big_drop = any(
        abl_stats[L]["sil"] < base_stats[L]["sil"] - 0.10
        or abl_stats[L]["nn_rate"] < base_stats[L]["nn_rate"] - 0.15
        for L in READOUT_LAYERS
    )
    if big_drop:
        print("  -> SURPRISE: side-channel ablation noticeably degrades sense separation.")
        print("     The side-channel may carry sense-relevant information beyond per-token identity,")
        print("     or it interacts with the residual stream in ways needed for sense disambiguation.")
    else:
        print("  -> Prediction confirmed: side-channel ablation has minimal effect on sense")
        print("     separation. Sense disambiguation flows through residual stream + attention,")
        print("     consistent with finding 03's picture (side-channel = per-token identity).")

    # ---- PCA scatter: 2x2 grid (layer x condition) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sense_colors = {
        "sense_city": "#1f77b4",
        "sense_finance": "#2ca02c",
        "sense_uppercase": "#ff7f0e",
        "sense_punishment": "#d62728",
    }
    sense_short = {
        "sense_city": "city", "sense_finance": "finance",
        "sense_uppercase": "uppercase", "sense_punishment": "punishment",
    }
    for row_idx, L in enumerate(READOUT_LAYERS):
        for col_idx, (cond_name, vecs, stats) in enumerate([
            ("baseline", base_vecs[L], base_stats[L]),
            ("side-channel ablated", abl_vecs[L], abl_stats[L]),
        ]):
            ax = axes[row_idx, col_idx]
            proj = PCA(n_components=2, random_state=42).fit_transform(vecs)
            for sense, _, mask in iterate_clusters(proj, labels):
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=sense_colors[sense], label=sense_short[sense],
                           s=70, alpha=0.85, edgecolors="black", linewidths=0.4)
            ax.set_title(f"L{L} -- {cond_name}\n"
                         f"sil={stats['sil']:+.3f}, NN={stats['nn_rate']:.2f}, "
                         f"purity={stats['purity']:.2f}",
                         fontsize=10)
            ax.set_xlabel("PC1")
            if col_idx == 0:
                ax.set_ylabel("PC2")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Side-channel ablation impact on 'capital' sense disambiguation\n"
        "(MatFormer per-layer-input gate zeroed at all 42 layers)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "homonym_side_channel.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
