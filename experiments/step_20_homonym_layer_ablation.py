"""Per-layer ablation impact on homonym sense disambiguation.

Port of step_02 (per-layer zero-ablation) to the sense-disambiguation
question from step_17.

Hypothesis (from beads issue j4q): the engine-room layer L12, which
step_17 identified as the geometric peak for sense separability, is
ALSO the causal peak. Ablating L12 should drop sense-separation more
than ablating layers outside the engine room.

Design:
  - For each of the 42 layers, ablate that layer (residual passes
    through unchanged) and capture residuals at the 'capital' position.
  - Two readout layers:
      L12 = peak silhouette from step_17 (the geometric peak).
      L41 = the deepest layer, where decoding is sharpest.
  - For each (ablated_layer, readout_layer) pair, compute silhouette
    and NN purity across the 4 sense cohorts.
  - 42 ablations x 32 prompts = 1344 forward passes (~5-10 min).

Sanity check: ablating any layer L > readout cannot affect the readout
(downstream of capture point). Those entries should be ~baseline,
which doubles as a smoke test that the ablation harness composes
cleanly with the residual capture.

Run from project root:
    python experiments/step_20_homonym_layer_ablation.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Ablate, GLOBAL_LAYERS, Model, N_LAYERS,
    cluster_purity, fact_vectors_at, nearest_neighbor_purity,
    silhouette_cosine,
)
from experiments.prompts import HOMONYM_CAPITAL_ALL  # noqa: E402

OUT_DIR = ROOT / "caches"
READOUT_LAYERS = [12, 41]


def _stats(vecs: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float, float]:
    sil = silhouette_cosine(vecs, labels)
    nn_rate, _ = nearest_neighbor_purity(vecs, labels)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(vecs)
    purity = cluster_purity(labels.tolist(), km.labels_.tolist())
    return float(sil), float(nn_rate), float(purity)


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

    # ---- Baseline ----
    print("=" * 70)
    print("Baseline (no ablation)")
    print("=" * 70)
    base_vecs = fact_vectors_at(model, valid, READOUT_LAYERS)
    base = {}
    for L in READOUT_LAYERS:
        sil, nn, pur = _stats(base_vecs[L], labels, k=k)
        base[L] = {"sil": sil, "nn": nn, "purity": pur}
        print(f"  L{L:>3}  sil={sil:+.4f}  NN={nn:.3f}  purity={pur:.3f}")

    # ---- Per-layer ablation sweep ----
    print(f"\n{'=' * 70}")
    print(f"Per-layer ablation sweep ({N_LAYERS} layers x {n} prompts = "
          f"{N_LAYERS * n} forward passes)")
    print(f"{'=' * 70}")

    # results[ablated_layer][readout_layer] = {sil, nn, purity}
    results: dict[int, dict[int, dict]] = {}
    t0 = time.perf_counter()
    for abl_L in range(N_LAYERS):
        vecs_by_readout = fact_vectors_at(
            model, valid, READOUT_LAYERS,
            interventions=[Ablate.layer(abl_L)],
        )
        results[abl_L] = {}
        for readout_L in READOUT_LAYERS:
            sil, nn, pur = _stats(vecs_by_readout[readout_L], labels, k=k)
            results[abl_L][readout_L] = {"sil": sil, "nn": nn, "purity": pur}
        if (abl_L + 1) % 6 == 0 or abl_L == N_LAYERS - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (abl_L + 1) * (N_LAYERS - abl_L - 1)
            print(f"  ablated L{abl_L:>2}: "
                  + "  ".join(
                      f"L{R}.sil={results[abl_L][R]['sil']:+.3f}"
                      for R in READOUT_LAYERS
                  )
                  + f"   [{elapsed:.0f}s, eta {eta:.0f}s]")

    print(f"\nDone in {time.perf_counter() - t0:.0f}s")

    # ---- Per-readout summary ----
    for readout_L in READOUT_LAYERS:
        print(f"\n{'=' * 70}")
        print(f"Δ separation metrics at readout L{readout_L} "
              f"(baseline sil={base[readout_L]['sil']:+.4f}, "
              f"NN={base[readout_L]['nn']:.3f})")
        print(f"{'=' * 70}")
        print(f"{'abl_L':>5}  {'kind':>7}  {'sil':>8}  {'Δsil':>8}  "
              f"{'NN':>6}  {'ΔNN':>7}  {'purity':>8}  {'Δpurity':>8}")
        print("-" * 70)
        deltas_sil = []
        for abl_L in range(N_LAYERS):
            r = results[abl_L][readout_L]
            d_sil = r["sil"] - base[readout_L]["sil"]
            d_nn = r["nn"] - base[readout_L]["nn"]
            d_pur = r["purity"] - base[readout_L]["purity"]
            deltas_sil.append((abl_L, d_sil, d_nn, d_pur, r))
            kind = "GLOBAL" if abl_L in GLOBAL_LAYERS else "local"
            print(f"  L{abl_L:>3}  {kind:>7s}  {r['sil']:+.4f}  {d_sil:+.4f}  "
                  f"{r['nn']:.3f}  {d_nn:+.4f}  {r['purity']:.3f}    {d_pur:+.4f}")

        # Ranked top-5 most damaging
        ranked = sorted(deltas_sil, key=lambda x: x[1])[:5]
        print(f"\n  5 ablations most damaging to silhouette at L{readout_L}:")
        for abl_L, d_sil, d_nn, d_pur, _ in ranked:
            kind = "GLOBAL" if abl_L in GLOBAL_LAYERS else "local"
            print(f"    L{abl_L:>2} ({kind:>6s})  Δsil={d_sil:+.4f}  "
                  f"ΔNN={d_nn:+.4f}  Δpurity={d_pur:+.4f}")

        # Sanity: layers strictly downstream of readout should be ~no-op
        downstream = [(abl_L, results[abl_L][readout_L]["sil"]
                       - base[readout_L]["sil"])
                      for abl_L in range(readout_L + 1, N_LAYERS)]
        if downstream:
            sanity = max(abs(d) for _, d in downstream)
            print(f"\n  Sanity check: max |Δsil| for layers > L{readout_L} "
                  f"(downstream of readout): {sanity:.4f}  (should be ~0)")

    # ---- Plot: Δsilhouette vs ablated_layer, one line per readout ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(N_LAYERS)
    for ax, metric, label, baseline_key in [
        (axes[0], "sil", "silhouette (cosine)", "sil"),
        (axes[1], "nn",  "NN same-sense rate",  "nn"),
    ]:
        for readout_L, color in zip(READOUT_LAYERS, ["#1f77b4", "#d62728"]):
            ys = [results[abl_L][readout_L][metric] for abl_L in range(N_LAYERS)]
            ax.plot(x, ys, marker="o", linewidth=2,
                    color=color, label=f"readout L{readout_L}")
            ax.axhline(base[readout_L][baseline_key], color=color,
                       linestyle="--", linewidth=1, alpha=0.5,
                       label=f"baseline (readout L{readout_L})")
        # Mark global layers as vertical lines
        for g in GLOBAL_LAYERS:
            ax.axvline(g, color="#999999", linestyle=":", linewidth=0.7, alpha=0.4)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    axes[1].set_xlabel("ablated layer")
    axes[0].set_title(
        "Per-layer ablation impact on 'capital' sense disambiguation\n"
        "(dashed = no-ablation baseline; vertical dotted = global-attention layers)"
    )
    plt.tight_layout()
    out_path = OUT_DIR / "homonym_layer_ablation.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
