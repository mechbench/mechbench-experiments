"""Replicate the perplexity-probe finding on Gemma 4 E2B.

step_30 ran @_lyraaaa_'s perplexity-probe technique on E4B and reproduced
her finding: the residual-stream direction predicting per-token surprisal
peaks at L23, the same layer the rest of the project's experiments keep
fingerprinting (essay sections 5, 7, 16, 21).

The architectural-pivot hypothesis from essay section 21 says L23 is
load-bearing because it's the **last fresh-K/V global layer** in E4B
(global attention layers: 5,11,17,23,29,35,41; first KV-shared layer: 24).
If the hypothesis generalizes, the analogous pivot in the smaller E2B
sibling — last fresh-K/V global = L14 — should host the same surprisal
peak.

E2B architecture (read from text_config at load time):
  - 35 layers (vs E4B's 42)
  - d_model 1536, n_heads 8, n_kv_heads 1
  - global attention at layers 4, 9, 14, 19, 24, 29, 34 (every 5th + final)
  - num_kv_shared_layers = 20 → first_kv_shared_layer = 15
  - last fresh-K/V global = L14

Prediction: ridge-regression test R^2 of the surprisal probe peaks at or
near L14 in E2B, mirroring E4B's L23 peak.

Method (identical to step_30, with all dimensions read from model.arch):
  - Same EMOTION_STORIES_TINY + EMOTION_NEUTRAL_BASELINE corpus (112
    short passages, varied content). Filter to content positions >= 20
    to skip the chat-template prefix.
  - For each prompt: one forward pass capturing residuals at all layers
    AND the full position logits. Per-token surprisal as
    -log_softmax(logits[t-1])[token_t] for each content position.
  - For each layer L: RidgeCV maps residuals to surprisal. Report
    train+test R^2 and correlation.
  - Compare peak layer to E2B's last_fresh_kv_global (predicted L14).

Run from project root:
    python experiments/step_31_perplexity_probe_e2b.py
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

from mechbench_core import Capture, Model, PromptSet  # noqa: E402
from experiments.prompts import (  # noqa: E402
    EMOTION_NEUTRAL_BASELINE, EMOTION_STORIES_TINY,
)

E2B_MODEL_ID = "mlx-community/gemma-4-e2b-it-bf16"
OUT_DIR = ROOT / "caches"
RIDGE_ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
TEST_FRACTION = 0.2
SEED = 42
CONTENT_START = 20  # skip chat-template prefix


def _per_token_surprisals(logits_mx, token_ids_mx, content_start: int):
    """Compute per-token surprisal for content tokens.

    Returns (surprisals, content_positions): two 1D numpy arrays.
    surprisals[i] is -log P(token_t | context_<t) for t = content_positions[i].
    """
    logits_np = np.array(logits_mx.astype(mx.float32))
    token_ids_np = np.array(token_ids_mx)
    seq_len = logits_np.shape[0]
    if content_start >= seq_len:
        content_start = max(1, seq_len - 1)

    max_per_row = logits_np.max(axis=-1, keepdims=True)
    log_probs = (logits_np - max_per_row) - np.log(
        np.exp(logits_np - max_per_row).sum(axis=-1, keepdims=True)
    )
    content_positions = np.arange(content_start, seq_len)
    surprisals = -log_probs[content_positions - 1, token_ids_np[content_positions]]
    return surprisals, content_positions


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Loading {E2B_MODEL_ID}...")
    model = Model.load(E2B_MODEL_ID)
    arch = model.arch
    print(f"  {arch.n_layers} layers, d_model={arch.d_model}, "
          f"globals={arch.global_layers}")
    print(f"  first_kv_shared_layer={arch.first_kv_shared_layer}, "
          f"last_fresh_kv_global=L{arch.last_fresh_kv_global}")
    print(f"  PREDICTION: perplexity-probe peak should land at or near "
          f"L{arch.last_fresh_kv_global}.")

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

    print(f"\nCollecting residuals at all {arch.n_layers} layers + per-token "
          f"surprisals (content positions >= {CONTENT_START})...")
    t0 = time.perf_counter()
    per_layer_resid: dict[int, list[np.ndarray]] = {L: [] for L in range(arch.n_layers)}
    per_prompt_surprisals: list[np.ndarray] = []

    for j, vp in enumerate(valid):
        result = model.run(vp.input_ids, interventions=[
            Capture.residual(range(arch.n_layers), point="post"),
        ])
        surprisals, positions = _per_token_surprisals(
            result.logits[0], vp.input_ids[0], CONTENT_START,
        )
        if len(positions) == 0:
            continue
        per_prompt_surprisals.append(surprisals)
        for L in range(arch.n_layers):
            resid = result.cache[f"blocks.{L}.resid_post"][0].astype(mx.float32)
            mx.eval(resid)
            resid_np = np.array(resid)
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

    r2_train = np.zeros(arch.n_layers, dtype=np.float32)
    r2_test = np.zeros(arch.n_layers, dtype=np.float32)
    correlations = np.zeros(arch.n_layers, dtype=np.float32)
    weight_vecs = np.zeros((arch.n_layers, arch.d_model), dtype=np.float32)
    chosen_alphas = np.zeros(arch.n_layers, dtype=np.float32)

    for L in range(arch.n_layers):
        X = np.concatenate(per_layer_resid[L])
        X_train = X[train_idx]
        X_test = X[test_idx]
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2").fit(X_train, y_train)
        chosen_alphas[L] = ridge.alpha_
        r2_train[L] = ridge.score(X_train, y_train)
        r2_test[L] = ridge.score(X_test, y_test)
        y_pred = ridge.predict(X_test)
        correlations[L] = float(np.corrcoef(y_pred, y_test)[0, 1])
        weight_vecs[L] = ridge.coef_

    print()
    print(f"  {'L':>3}  {'type':>6}  {'alpha':>8}  {'R2 train':>10}  {'R2 test':>10}  {'corr':>7}")
    print("  " + "-" * 58)
    for L in range(arch.n_layers):
        tag = "GLOBAL" if L in arch.global_layers else "local "
        print(f"  {L:>3}  {tag:>6}  {chosen_alphas[L]:>8.0f}  "
              f"{r2_train[L]:>+10.4f}  {r2_test[L]:>+10.4f}  "
              f"{correlations[L]:>+7.4f}")

    peak_L = int(np.argmax(r2_test))
    predicted = arch.last_fresh_kv_global
    delta = peak_L - predicted
    print()
    print(f"=" * 75)
    print(f"Peak layer (test R^2): L{peak_L}  "
          f"R^2_test = {r2_test[peak_L]:.4f}, corr = {correlations[peak_L]:.4f}")
    print(f"E2B predicted pivot:   L{predicted} (last fresh-K/V global)")
    print(f"Delta:                 {delta:+d} layers from prediction")
    print(f"E4B reference (step_30): L23, R^2_test = 0.671, corr = 0.82")
    print(f"=" * 75)

    cos_consecutive = np.zeros(arch.n_layers - 1, dtype=np.float32)
    for L in range(arch.n_layers - 1):
        u = weight_vecs[L]
        v = weight_vecs[L + 1]
        cos_consecutive[L] = float(
            (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        )

    print(f"\nLayer-boundary surprise-direction cosine similarities (sharp drops "
          f"= big rotations):")
    for L in range(arch.n_layers - 1):
        if cos_consecutive[L] < 0.5 or (L + 1) in arch.global_layers:
            tag = "<-- GLOBAL boundary" if (L + 1) in arch.global_layers else ""
            if (L + 1) == arch.first_kv_shared_layer:
                tag = "<-- KV-SHARING BOUNDARY"
            print(f"  L{L:>2} -> L{L+1:>2}: {cos_consecutive[L]:+.4f}  {tag}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    ax = axes[0]
    x = np.arange(arch.n_layers)
    ax.plot(x, r2_train, marker="o", linewidth=2,
            color="#1f77b4", label="train R²", alpha=0.8)
    ax.plot(x, r2_test, marker="s", linewidth=2.5,
            color="#d62728", label="test R²")
    for g in arch.global_layers:
        ax.axvline(g, color="#999999", linestyle=":",
                   linewidth=0.7, alpha=0.5)
    ax.axvline(predicted, color="#2ca02c", linestyle="--", linewidth=1.5,
               alpha=0.7,
               label=f"predicted pivot L{predicted} (last fresh-K/V global)")
    ax.axvline(arch.first_kv_shared_layer, color="#9467bd", linestyle="--",
               linewidth=1.0, alpha=0.5,
               label=f"KV-sharing boundary L{arch.first_kv_shared_layer}")
    ax.set_xlabel("layer index (E2B)")
    ax.set_ylabel("R² of ridge regression: residual → surprisal")
    ax.set_title(
        f"E2B per-layer perplexity-probe R² (replication of step_30 on E2B)\n"
        f"Peak: L{peak_L}  test R²={r2_test[peak_L]:.3f}  "
        f"corr={correlations[peak_L]:.3f}    "
        f"(predicted L{predicted}, delta {delta:+d})",
        fontsize=11,
    )
    ax.set_xticks(range(0, arch.n_layers, 3))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.axhline(0, color="black", linewidth=0.5)

    ax = axes[1]
    boundaries_x = np.arange(arch.n_layers - 1)
    bar_colors = [
        "#2ca02c" if c >= 0.6 else "#ff7f0e" if c >= 0.3 else "#d62728"
        for c in cos_consecutive
    ]
    ax.bar(boundaries_x, cos_consecutive, color=bar_colors,
           edgecolor="black", linewidth=0.3)
    ax.axvline(arch.first_kv_shared_layer - 0.5, color="#9467bd",
               linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"KV-sharing boundary (L{arch.first_kv_shared_layer-1}→L{arch.first_kv_shared_layer})")
    ax.set_xlabel("layer boundary (L_i → L_{i+1})")
    ax.set_ylabel("cos similarity of surprise directions")
    ax.set_title(
        "E2B surprise-direction stability across layer boundaries",
        fontsize=11,
    )
    ax.set_xticks(range(0, arch.n_layers - 1, 3))
    ax.set_xticklabels([f"{i}-{i+1}" for i in range(0, arch.n_layers - 1, 3)],
                       rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(min(-0.1, cos_consecutive.min() * 1.1), 1.05)
    for L in range(arch.n_layers - 1):
        if cos_consecutive[L] < 0.3:
            ax.annotate(
                f"L{L}→L{L+1}\n{cos_consecutive[L]:.2f}",
                xy=(L, cos_consecutive[L]),
                xytext=(L, cos_consecutive[L] + 0.25),
                ha="center", fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"),
            )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best")

    plt.tight_layout()
    out_path = OUT_DIR / "perplexity_probe_e2b_per_layer.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    np.savez_compressed(
        OUT_DIR / "perplexity_probe_e2b_weights.npz",
        weight_vecs=weight_vecs,
        r2_train=r2_train, r2_test=r2_test, correlations=correlations,
        cos_consecutive=cos_consecutive, surprisals=y,
        n_layers=arch.n_layers, d_model=arch.d_model,
        global_layers=np.array(arch.global_layers),
        first_kv_shared_layer=arch.first_kv_shared_layer,
        last_fresh_kv_global=arch.last_fresh_kv_global,
        peak_layer=peak_L,
    )


if __name__ == "__main__":
    main()
