"""Per-head Q and K sense-clustering across the homonym corpus.

For every (layer, Q-head) pair compute the silhouette of sense labels
over Q vectors at the 'capital' position across the 32 HOMONYM_CAPITAL_ALL
prompts. Same for (layer, KV-head) and K vectors. Identifies which
specific attention heads specialize in sense-based reading/keying.

Hypothesis motivated by step_17: residual-stream sense separability
peaks at L12. Q and K are projections of the residual into per-head
subspaces. A head that 'specializes in sense' would show high sense
separability in its Q or K space, possibly higher than the residual at
its layer because the subspace is specifically tuned. A head with no
sense specialization will show chance-level silhouette.

Uses fact_vectors_at_hook (new in cew) for per-head extraction; 42 x
(8 Q-heads + 2 KV-heads) = 420 cluster-analyses in a single forward
pass per prompt.

Run from project root:
    python experiments/step_28_qk_sense_clustering.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Capture, GLOBAL_LAYERS, Model, N_LAYERS,
    nearest_neighbor_purity, silhouette_cosine,
)
from gemma4_mlx_interp.geometry import _resolve_position  # noqa: E402
from experiments.prompts import HOMONYM_CAPITAL_ALL  # noqa: E402

OUT_DIR = ROOT / "caches"
N_HEADS = 8
N_KV_HEADS = 2


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating {len(HOMONYM_CAPITAL_ALL)} prompts...")
    valid = HOMONYM_CAPITAL_ALL.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    n = len(valid)
    labels = valid.labels
    print(f"  {n} / {len(HOMONYM_CAPITAL_ALL)} validated.")
    senses = valid.categories
    print(f"  senses: {senses}")

    # For every prompt, run ONCE capturing Q and K at every layer.
    # Then slice per-head on the cached tensors. 32 prompts * 1 forward pass
    # each is tolerable.
    print(f"\nExtracting Q and K at the 'capital' position across 42 layers "
          f"x {N_HEADS} Q-heads / {N_KV_HEADS} KV-heads for {n} prompts...")
    t0 = time.perf_counter()

    # Storage: dict from (layer, head, 'q'|'k') -> [n, head_dim]
    Q_mats: dict[tuple[int, int], np.ndarray] = {}   # (layer, Q-head)
    K_mats: dict[tuple[int, int], np.ndarray] = {}   # (layer, KV-head)

    for j, vp in enumerate(valid):
        result = model.run(vp.input_ids, interventions=[
            Capture.queries(range(N_LAYERS)),
            Capture.keys(range(N_LAYERS)),
        ])
        pos = _resolve_position(model, vp, "subject")
        for L in range(N_LAYERS):
            Q_all = result.cache[f"blocks.{L}.attn.q"].astype(mx.float32)
            K_all = result.cache[f"blocks.{L}.attn.k"].astype(mx.float32)
            head_dim_Q = int(Q_all.shape[-1])
            head_dim_K = int(K_all.shape[-1])
            for h in range(N_HEADS):
                key = (L, h)
                if key not in Q_mats:
                    Q_mats[key] = np.zeros((n, head_dim_Q), dtype=np.float32)
                v = Q_all[0, h, pos, :]
                mx.eval(v)
                Q_mats[key][j] = np.array(v)
            for g in range(N_KV_HEADS):
                key = (L, g)
                if key not in K_mats:
                    K_mats[key] = np.zeros((n, head_dim_K), dtype=np.float32)
                v = K_all[0, g, pos, :]
                mx.eval(v)
                K_mats[key][j] = np.array(v)
        if (j + 1) % 8 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (j + 1) * (n - j - 1)
            print(f"  prompt {j + 1}/{n}  [{elapsed:.0f}s elapsed, eta {eta:.0f}s]")

    print(f"\nDone in {time.perf_counter() - t0:.0f}s")

    # ---- Per-head sense separability metrics ----
    print(f"\n{'=' * 70}")
    print(f"Per-(layer, Q-head) sense silhouette at the 'capital' position")
    print(f"{'=' * 70}")

    sil_Q = np.zeros((N_LAYERS, N_HEADS), dtype=np.float32)
    nn_Q = np.zeros((N_LAYERS, N_HEADS), dtype=np.float32)
    for (L, h), vecs in Q_mats.items():
        sil_Q[L, h] = silhouette_cosine(vecs, labels)
        nn_Q[L, h], _ = nearest_neighbor_purity(vecs, labels)

    sil_K = np.zeros((N_LAYERS, N_KV_HEADS), dtype=np.float32)
    nn_K = np.zeros((N_LAYERS, N_KV_HEADS), dtype=np.float32)
    for (L, g), vecs in K_mats.items():
        sil_K[L, g] = silhouette_cosine(vecs, labels)
        nn_K[L, g], _ = nearest_neighbor_purity(vecs, labels)

    # Top-15 (layer, head) pairs by silhouette, for Q and K
    def _rank(mat, kind: str, n_columns: int):
        flat = [(L, h, float(mat[L, h]))
                for L in range(N_LAYERS) for h in range(n_columns)]
        flat.sort(key=lambda x: -x[2])
        print(f"\nTop 15 (layer, {kind}) pairs by sense silhouette:")
        for L, h, sil in flat[:15]:
            tag = "GLOBAL" if L in GLOBAL_LAYERS else "local"
            print(f"  L{L:>2} {kind}{h}  {tag}  sil={sil:+.4f}")
        return flat

    _rank(sil_Q, "q", N_HEADS)
    _rank(sil_K, "k", N_KV_HEADS)

    # Per-layer summary: best head at each layer
    print(f"\n{'=' * 70}")
    print(f"Per-layer best head (Q side)")
    print(f"{'=' * 70}")
    print(f"{'L':>3}  {'best_h':>6}  {'best_sil':>8}  {'max_sil':>8}  {'max_nn':>6}")
    for L in range(N_LAYERS):
        best_h = int(np.argmax(sil_Q[L]))
        print(f"  {L:>3}  {best_h:>6}  {sil_Q[L, best_h]:+.4f}  "
              f"{sil_Q[L].max():+.4f}  {nn_Q[L].max():.3f}")

    # ---- Visualization: (layer, head) heatmaps for Q and K silhouette ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 9),
                              gridspec_kw={"width_ratios": [N_HEADS, N_KV_HEADS]})

    ax = axes[0]
    vmax = max(abs(sil_Q).max(), abs(sil_K).max())
    im = ax.imshow(sil_Q, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Q-head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(0, N_LAYERS, 3))
    ax.set_title("Per-(layer, Q-head)\nsense silhouette at 'capital'")
    for g in GLOBAL_LAYERS:
        ax.axhline(g - 0.5, color="red", linewidth=0.2, alpha=0.3,
                   xmin=-0.02, xmax=0)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(sil_K, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("KV-head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(N_KV_HEADS))
    ax.set_yticks(range(0, N_LAYERS, 3))
    ax.set_title("Per-(layer, KV-head)\nsense silhouette at 'capital'")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "QK geometric sense-separation at the 'capital' position across "
        "42 layers x 8 Q-heads / 2 KV-heads\n"
        "(4 sense labels; chance silhouette = 0.0; positive = separable)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "qk_sense_clustering.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # Save the raw silhouette arrays for follow-up analysis
    np.savez_compressed(
        OUT_DIR / "qk_sense_clustering.npz",
        sil_Q=sil_Q, sil_K=sil_K, nn_Q=nn_Q, nn_K=nn_K,
    )


if __name__ == "__main__":
    main()
