"""Step 42 — perplexity probe (Lyra-style surprise direction) on Qwen 2.5 3B Instruct.

Cross-family port of step_30 (E4B) / step_31 (E2B). For each
narrative passage, capture residuals at every layer, compute
per-token surprisal from the model's own output logits, fit
RidgeCV (residual → surprisal) per layer, report test R² and
cosine signature between consecutive learned weight vectors.

Key methodology note: step_30/31 chat-template their narrative
prompts (using `model.tokenize(text)` with default chat template)
and skip CONTENT_START=20 tokens of chat-template prefix. Qwen's
chat-template prefix is a different length, AND the corpus is
narrative text that should arguably be tokenized as raw
completion text. This script uses chat_template=False and
content_start=0 — content is the entire sequence.

The cross-family question: does Lyra's sharp-rotation pattern
(R² peak followed by orthogonal cosine drop) generalize beyond
Gemma, or is it Gemma-specific?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechbench_core import Capture, Model  # noqa: E402
from experiments.prompts import (  # noqa: E402
    EMOTION_NEUTRAL_BASELINE,
    EMOTION_STORIES_TINY,
)


MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-bf16"
RIDGE_ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
TEST_FRACTION = 0.2
SEED = 42
CONTENT_START = 0  # bare-prompt; no chat-template prefix to skip


def _per_token_surprisals(logits_mx, token_ids_mx, content_start: int):
    logits_np = np.array(logits_mx.astype(mx.float32))
    token_ids_np = np.array(token_ids_mx)
    seq_len = logits_np.shape[0]
    if content_start >= seq_len:
        content_start = max(1, seq_len - 1)
    max_per_row = logits_np.max(axis=-1, keepdims=True)
    log_probs = (logits_np - max_per_row) - np.log(
        np.exp(logits_np - max_per_row).sum(axis=-1, keepdims=True)
    )
    # Need at least one preceding token to predict from.
    content_start = max(content_start, 1)
    content_positions = np.arange(content_start, seq_len)
    surprisals = -log_probs[content_positions - 1, token_ids_np[content_positions]]
    return surprisals, content_positions


def main() -> None:
    print(f"Loading {MODEL_ID}...")
    model = Model.load(MODEL_ID)
    arch = model.arch
    print(f"  {arch.n_layers} layers, d_model={arch.d_model}, "
          f"model_type={arch.model_type}")

    corpus = list(EMOTION_STORIES_TINY.prompts) + list(
        EMOTION_NEUTRAL_BASELINE.prompts
    )
    print(f"\nProcessing {len(corpus)} narrative passages "
          f"(bare-prompt, content_start={CONTENT_START})...")

    per_layer_resid: dict[int, list[np.ndarray]] = {
        L: [] for L in range(arch.n_layers)
    }
    per_prompt_surprisals: list[np.ndarray] = []

    t0 = time.perf_counter()
    for j, prompt in enumerate(corpus):
        ids = model.tokenize(prompt.text, chat_template=False)
        result = model.run(
            ids,
            interventions=[Capture.residual(range(arch.n_layers), point="post")],
        )
        surprisals, positions = _per_token_surprisals(
            result.logits[0], ids[0], CONTENT_START,
        )
        if len(positions) == 0:
            continue
        per_prompt_surprisals.append(surprisals)
        for L in range(arch.n_layers):
            resid = result.cache[f"blocks.{L}.resid_post"][0].astype(mx.float32)
            mx.eval(resid)
            per_layer_resid[L].append(np.array(resid)[positions, :])
        if (j + 1) % 16 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (j + 1) * (len(corpus) - j - 1)
            print(f"  {j+1}/{len(corpus)}  [{elapsed:.0f}s elapsed, eta {eta:.0f}s]")

    print(f"  done in {time.perf_counter() - t0:.0f}s")

    y = np.concatenate(per_prompt_surprisals)
    print(f"  Total content tokens: {len(y)}")
    print(f"  Surprisal: mean={y.mean():.2f}  std={y.std():.2f}")

    n_total = len(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_test = int(n_total * TEST_FRACTION)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"\nFitting per-layer RidgeCV ({n_total - n_test} train / {n_test} test)...")
    r2_test = np.zeros(arch.n_layers, dtype=np.float32)
    correlations = np.zeros(arch.n_layers, dtype=np.float32)
    weight_vecs = np.zeros((arch.n_layers, arch.d_model), dtype=np.float32)
    for L in range(arch.n_layers):
        X = np.concatenate(per_layer_resid[L])
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2").fit(
            X[train_idx], y_train
        )
        r2_test[L] = ridge.score(X[test_idx], y_test)
        y_pred = ridge.predict(X[test_idx])
        correlations[L] = float(np.corrcoef(y_pred, y_test)[0, 1])
        weight_vecs[L] = ridge.coef_

    print(f"\n{'L':>3}  {'R²_test':>9}  {'corr':>7}")
    for L in range(arch.n_layers):
        print(f"{L:>3}  {r2_test[L]:>+9.4f}  {correlations[L]:>+7.4f}")

    peak_L = int(np.argmax(r2_test))
    print(f"\nPeak R²_test: L{peak_L}  R²={r2_test[peak_L]:.4f}  "
          f"corr={correlations[peak_L]:.4f}")

    cos_consecutive = np.zeros(arch.n_layers - 1, dtype=np.float32)
    for L in range(arch.n_layers - 1):
        u, v = weight_vecs[L], weight_vecs[L + 1]
        cos_consecutive[L] = float(
            (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        )
    sharpest = int(np.argmin(cos_consecutive))
    print(f"\nSharpest consecutive rotation: L{sharpest} → L{sharpest+1}: "
          f"cos = {cos_consecutive[sharpest]:.4f}")
    print(f"Reference (cross-family):")
    print(f"  E4B (step_30): R² peak L23, rotation L22→L23 cos=0.033")
    print(f"  E2B (step_31): R² peak L12, rotation L13→L14 cos=0.0152")
    print(f"  Qwen 2.5 3B:   R² peak L{peak_L}, rotation "
          f"L{sharpest}→L{sharpest+1} cos={cos_consecutive[sharpest]:.4f}")

    out_dir = ROOT / "caches"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "step_42_perplexity_probe_qwen2_5_3b.json"
    out_json.write_text(json.dumps({
        "model_id": MODEL_ID,
        "n_layers": arch.n_layers,
        "n_total_tokens": int(n_total),
        "r2_test": [round(float(v), 4) for v in r2_test],
        "correlations": [round(float(v), 4) for v in correlations],
        "cosine_consecutive": [round(float(v), 4) for v in cos_consecutive],
        "peak_layer": peak_L,
        "sharpest_rotation_layer": sharpest,
    }, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
