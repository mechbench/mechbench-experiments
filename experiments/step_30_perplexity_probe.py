"""Replicate @_lyraaaa_'s perplexity-probe finding on Gemma 4 E4B.

She reported (Twitter, 2026-04-18) that a residual-stream direction
in this model correlates with per-token surprisal at r=0.919, R^2=0.845
at layer 21 (peak), trained on 1600 FineFineWeb passages. We replicate
the structural finding using the existing in-repo corpus.

Method:
  - Use EMOTION_STORIES_TINY + EMOTION_NEUTRAL_BASELINE as the corpus
    (112 short passages, varied content). Filter to content tokens
    (positions >= 20) to skip the chat-template prefix.
  - For each prompt: one forward pass capturing residuals at all 42
    layers AND the full position logits. Compute per-token surprisal
    as -log_softmax(logits[t-1])[token_t] for each content position t.
  - For each layer L: fit a ridge regression mapping residuals at
    layer L to surprisal. The learned weight vector is the surprise
    direction at that layer.
  - Report per-layer R^2 (train + held-out test) and correlation.
    Identify peak layer; compare to her L21.

Run from project root:
    python experiments/step_30_perplexity_probe.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechbench_core import (  # noqa: E402
    Capture, GLOBAL_LAYERS, Model, N_LAYERS, PromptSet,
)
from experiments.prompts import (  # noqa: E402
    EMOTION_NEUTRAL_BASELINE, EMOTION_STORIES_TINY,
)

OUT_DIR = ROOT / "caches"
RIDGE_ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)  # RidgeCV grid
TEST_FRACTION = 0.2
SEED = 42
CONTENT_START = 20  # skip chat-template prefix


def _per_token_surprisals(logits_mx, token_ids_mx, content_start: int):
    """Compute per-token surprisal for content tokens.

    Returns (surprisals, content_positions): two 1D numpy arrays.
    surprisals[i] is -log P(token_t | context_<t) for t = content_positions[i].
    """
    logits_np = np.array(logits_mx.astype(mx.float32))    # [seq, vocab]
    token_ids_np = np.array(token_ids_mx)                 # [seq]
    seq_len = logits_np.shape[0]
    if content_start >= seq_len:
        content_start = max(1, seq_len - 1)

    # log_softmax via the numerically-stable identity.
    max_per_row = logits_np.max(axis=-1, keepdims=True)
    log_probs = (logits_np - max_per_row) - np.log(
        np.exp(logits_np - max_per_row).sum(axis=-1, keepdims=True)
    )
    # Surprisal of token at position t (t >= 1) is -log P from logits[t-1]
    content_positions = np.arange(content_start, seq_len)
    surprisals = -log_probs[content_positions - 1, token_ids_np[content_positions]]
    return surprisals, content_positions


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    corpus = PromptSet(
        name="PERPLEXITY_PROBE_CORPUS",
        prompts=tuple(
            list(EMOTION_STORIES_TINY.prompts)
            + list(EMOTION_NEUTRAL_BASELINE.prompts)
        ),
    )
    print(f"\nValidating {len(corpus)} prompts...")
    valid = corpus.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    n = len(valid)
    print(f"  {n} validated.")

    # ---- Collect residuals + surprisals ----
    print(f"\nCollecting residuals at all {N_LAYERS} layers + per-token "
          f"surprisals (content positions >= {CONTENT_START})...")
    t0 = time.perf_counter()
    per_layer_resid: dict[int, list[np.ndarray]] = {L: [] for L in range(N_LAYERS)}
    per_prompt_surprisals: list[np.ndarray] = []

    for j, vp in enumerate(valid):
        result = model.run(vp.input_ids, interventions=[
            Capture.residual(range(N_LAYERS), point="post"),
        ])
        surprisals, positions = _per_token_surprisals(
            result.logits[0], vp.input_ids[0], CONTENT_START,
        )
        if len(positions) == 0:
            continue
        per_prompt_surprisals.append(surprisals)
        for L in range(N_LAYERS):
            resid = result.cache[f"blocks.{L}.resid_post"][0].astype(mx.float32)
            mx.eval(resid)
            resid_np = np.array(resid)  # [seq, d_model]
            per_layer_resid[L].append(resid_np[positions, :])

        if (j + 1) % 16 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (j + 1) * (n - j - 1)
            print(f"  {j+1}/{n}  [{elapsed:.0f}s elapsed, eta {eta:.0f}s]")

    print(f"  done in {time.perf_counter() - t0:.0f}s")

    y = np.concatenate(per_prompt_surprisals)
    print(f"  Total content tokens: {len(y)}")
    print(f"  Surprisal range: [{y.min():.2f}, {y.max():.2f}]  "
          f"mean={y.mean():.2f}  std={y.std():.2f}")

    # ---- Per-layer ridge regression with RidgeCV ----
    print(f"\nFitting per-layer RidgeCV over alphas {RIDGE_ALPHAS} "
          f"with train/test split={1-TEST_FRACTION:.0%}/{TEST_FRACTION:.0%}...")
    n_total = len(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_test = int(n_total * TEST_FRACTION)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    y_train = y[train_idx]
    y_test = y[test_idx]

    r2_train = np.zeros(N_LAYERS, dtype=np.float32)
    r2_test = np.zeros(N_LAYERS, dtype=np.float32)
    correlations = np.zeros(N_LAYERS, dtype=np.float32)
    weight_vecs = np.zeros((N_LAYERS, 2560), dtype=np.float32)
    chosen_alphas = np.zeros(N_LAYERS, dtype=np.float32)

    for L in range(N_LAYERS):
        X = np.concatenate(per_layer_resid[L])
        X_train = X[train_idx]
        X_test = X[test_idx]
        # RidgeCV picks alpha via efficient leave-one-out CV on the training set
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2").fit(X_train, y_train)
        chosen_alphas[L] = ridge.alpha_
        r2_train[L] = ridge.score(X_train, y_train)
        r2_test[L] = ridge.score(X_test, y_test)
        y_pred = ridge.predict(X_test)
        correlations[L] = float(np.corrcoef(y_pred, y_test)[0, 1])
        weight_vecs[L] = ridge.coef_

    # ---- Report ----
    print()
    print(f"  {'L':>3}  {'type':>6}  {'alpha':>8}  {'R2 train':>10}  {'R2 test':>10}  {'corr':>7}")
    print("  " + "-" * 58)
    for L in range(N_LAYERS):
        tag = "GLOBAL" if L in GLOBAL_LAYERS else "local "
        print(f"  {L:>3}  {tag:>6}  {chosen_alphas[L]:>8.0f}  "
              f"{r2_train[L]:>+10.4f}  {r2_test[L]:>+10.4f}  "
              f"{correlations[L]:>+7.4f}")

    peak_L = int(np.argmax(r2_test))
    print()
    print(f"=" * 65)
    print(f"Peak layer (test R^2): L{peak_L}  "
          f"R^2_test = {r2_test[peak_L]:.4f}, corr = {correlations[peak_L]:.4f}")
    print(f"Lyra's reported peak: L21, R^2 = 0.845, r = 0.919  "
          f"(on 1600 FineFineWeb passages)")
    print(f"Our corpus: {len(y)} content tokens across {n} EMOTION+NEUTRAL passages")
    print(f"=" * 65)

    # ---- Per-layer cosine similarity of consecutive surprise directions ----
    # (Replicates her Image 2 — sharp rotation at layer boundaries.)
    cos_consecutive = np.zeros(N_LAYERS - 1, dtype=np.float32)
    for L in range(N_LAYERS - 1):
        u = weight_vecs[L]
        v = weight_vecs[L + 1]
        cos_consecutive[L] = float(
            (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        )

    print(f"\nLayer-boundary surprise-direction cosine similarities (sharp drops "
          f"= big rotations):")
    for L in range(N_LAYERS - 1):
        if cos_consecutive[L] < 0.5 or (L + 1) in GLOBAL_LAYERS:
            tag = "<-- GLOBAL boundary" if (L + 1) in GLOBAL_LAYERS else ""
            print(f"  L{L:>2} -> L{L+1:>2}: {cos_consecutive[L]:+.4f}  {tag}")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    # Panel 1: per-layer R^2
    ax = axes[0]
    x = np.arange(N_LAYERS)
    ax.plot(x, r2_train, marker="o", linewidth=2,
            color="#1f77b4", label="train R²", alpha=0.8)
    ax.plot(x, r2_test, marker="s", linewidth=2.5,
            color="#d62728", label="test R²")
    for g in GLOBAL_LAYERS:
        ax.axvline(g, color="#999999", linestyle=":",
                   linewidth=0.7, alpha=0.5)
    ax.axvline(21, color="#ff7f00", linestyle="--", linewidth=1.5, alpha=0.7,
               label="lyra's reported peak (L21, R²=0.845)")
    ax.set_xlabel("layer index")
    ax.set_ylabel("R² of ridge regression: residual → surprisal")
    ax.set_title(
        f"Per-layer perplexity-probe R² (replication of @_lyraaaa_'s finding)\n"
        f"Peak layer: L{peak_L}  test R²={r2_test[peak_L]:.3f}, "
        f"corr={correlations[peak_L]:.3f}",
        fontsize=11,
    )
    ax.set_xticks(range(0, N_LAYERS, 3))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.axhline(0, color="black", linewidth=0.5)

    # Panel 2: cosine similarity of consecutive surprise-direction vectors
    ax = axes[1]
    boundaries_x = np.arange(N_LAYERS - 1)
    bar_colors = [
        "#2ca02c" if c >= 0.6 else "#ff7f0e" if c >= 0.3 else "#d62728"
        for c in cos_consecutive
    ]
    ax.bar(boundaries_x, cos_consecutive, color=bar_colors,
           edgecolor="black", linewidth=0.3)
    ax.set_xlabel("layer boundary (L_i → L_{i+1})")
    ax.set_ylabel("cos similarity of surprise directions")
    ax.set_title(
        "Surprise direction stability across layer boundaries (replicates her image 2)",
        fontsize=11,
    )
    ax.set_xticks(range(0, N_LAYERS - 1, 3))
    ax.set_xticklabels([f"{i}-{i+1}" for i in range(0, N_LAYERS - 1, 3)],
                       rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(min(-0.1, cos_consecutive.min() * 1.1), 1.05)
    # Annotate sharp rotations
    for L in range(N_LAYERS - 1):
        if cos_consecutive[L] < 0.3:
            ax.annotate(
                f"L{L}→L{L+1}\n{cos_consecutive[L]:.2f}",
                xy=(L, cos_consecutive[L]),
                xytext=(L, cos_consecutive[L] + 0.25),
                ha="center", fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"),
            )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = OUT_DIR / "perplexity_probe_per_layer.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    np.savez_compressed(
        OUT_DIR / "perplexity_probe_weights.npz",
        weight_vecs=weight_vecs,
        r2_train=r2_train, r2_test=r2_test, correlations=correlations,
        cos_consecutive=cos_consecutive, surprisals=y,
    )


if __name__ == "__main__":
    main()
