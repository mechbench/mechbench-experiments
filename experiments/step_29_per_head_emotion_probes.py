"""Per-head emotion probes across Q, K, V streams.

Uses the generalized Probe primitive (cew) with hook_point + head kwargs
to build 6-emotion probes at every (layer, head, stream) where stream is
one of attn.q, attn.k, attn.v. Compare each stream's per-head concept
specialization to step_21's residual-stream baseline.

Questions:
  - For any given emotion (happy, sad, angry, afraid, calm, proud), is
    there a specialist (layer, head, stream) that represents it
    especially cleanly?
  - Do some streams (Q vs K vs V) separate the 6 emotions better than
    others, mirroring step_28's finding that K-space separates
    homonym sense better than Q-space?
  - Are the best per-head emotion probes sharper than the residual-stream
    probe at L28 (which was step_21's canonical readout)?

Per prompt, one forward pass captures Q, K, V at all 42 layers in a
single sweep. Activations are mean-pooled across positions >=20 (matching
step_21) and organized per-(layer, head, stream) for probe construction.

Probe.from_labeled_corpus builds 6 probes per (layer, head, stream) —
504 (layer, head, stream) combinations x 6 probes = 3,024 probes total.
Each is scored on the training corpus (self-consistency) and by 6-way
silhouette.

Run from project root:
    python experiments/step_29_per_head_emotion_probes.py
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
    Capture, GLOBAL_LAYERS, Model, N_LAYERS, Probe,
    silhouette_cosine,
)
from experiments.prompts import (  # noqa: E402
    EMOTION_NEUTRAL_BASELINE, EMOTION_STORIES_TINY,
)

OUT_DIR = ROOT / "caches"
POOL_START = 20
N_Q_HEADS = 8
N_KV_HEADS = 2

EMOTIONS = (
    "emotion_happy", "emotion_sad", "emotion_angry",
    "emotion_afraid", "emotion_calm", "emotion_proud",
)


def _collect_qkv(model, valid, label_tag: str, Q_data, K_data, V_data, labels_out):
    """Run each prompt once, extracting mean-pooled Q/K/V per head/layer."""
    for j, vp in enumerate(valid):
        result = model.run(vp.input_ids, interventions=[
            Capture.queries(range(N_LAYERS)),
            Capture.keys(range(N_LAYERS)),
            Capture.values(range(N_LAYERS)),
        ])
        seq_len = int(vp.input_ids.shape[1])
        start = POOL_START if POOL_START < seq_len else seq_len - 1
        end = seq_len
        for L in range(N_LAYERS):
            Q = result.cache[f"blocks.{L}.attn.q"].astype(mx.float32)
            K = result.cache[f"blocks.{L}.attn.k"].astype(mx.float32)
            V = result.cache[f"blocks.{L}.attn.v"].astype(mx.float32)
            mx.eval(Q, K, V)
            Q_np = np.array(Q)
            K_np = np.array(K)
            V_np = np.array(V)
            for h in range(N_Q_HEADS):
                Q_data.setdefault((L, h), []).append(
                    Q_np[0, h, start:end, :].mean(axis=0)
                )
            for g in range(N_KV_HEADS):
                K_data.setdefault((L, g), []).append(
                    K_np[0, g, start:end, :].mean(axis=0)
                )
                V_data.setdefault((L, g), []).append(
                    V_np[0, g, start:end, :].mean(axis=0)
                )
        labels_out.append(label_tag if label_tag != "emotion" else vp.prompt.category)


def _analyze_stream(
    data: dict, labels_all: np.ndarray, n_heads: int, stream_name: str,
    neutral_mask: np.ndarray,
):
    """Build probes + compute silhouette per (layer, head).

    Returns:
      - sil: [N_LAYERS, n_heads] 6-way silhouette on positive vectors
      - acc: [N_LAYERS, n_heads] per-passage top-1 accuracy across 6 emotions
      - specialists: dict emotion -> list of (layer, head, self_score, n_passages_scored_correctly)
    """
    sil = np.zeros((N_LAYERS, n_heads), dtype=np.float32)
    acc = np.zeros((N_LAYERS, n_heads), dtype=np.float32)
    # Specialist tracker: for each emotion, keep top candidates by per-passage accuracy
    specialists: dict[str, list] = {e: [] for e in EMOTIONS}

    pos_mask = ~neutral_mask
    pos_labels = labels_all[pos_mask]

    for L in range(N_LAYERS):
        for h in range(n_heads):
            all_vecs = np.stack(data[(L, h)])  # [n_total, head_dim]
            pos_vecs = all_vecs[pos_mask]
            neu_vecs = all_vecs[neutral_mask]

            # 6-way silhouette on positives
            sil[L, h] = silhouette_cosine(pos_vecs, pos_labels)

            # Build 6 probes and score per passage
            labeled = {
                e: pos_vecs[pos_labels == e] for e in EMOTIONS
            }
            probes = Probe.from_labeled_corpus(
                labeled, neu_vecs, layer=L,
                hook_point=f"blocks.{L}.attn.{stream_name}",
                head=h, explain=0.5,
            )
            probe_names = list(probes.keys())
            scores = np.stack(
                [probes[p].score(pos_vecs) for p in probe_names], axis=1
            )
            per_passage_top1 = np.array(
                [probe_names[int(np.argmax(scores[j]))]
                 for j in range(len(scores))]
            )
            correct = int((per_passage_top1 == pos_labels).sum())
            acc[L, h] = correct / len(pos_labels)

            # Per-emotion specialist score: how cleanly does this head's
            # emotion-X probe discriminate emotion-X passages from the rest?
            # Use mean(score on own) - mean(score on others) as a direct
            # margin measure.
            for e_idx, e in enumerate(probe_names):
                own = scores[pos_labels == e, e_idx].mean()
                others = scores[pos_labels != e, e_idx].mean()
                margin = float(own - others)
                specialists[e].append((L, h, margin, own, others))

    return sil, acc, specialists


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating EMOTION_STORIES_TINY + EMOTION_NEUTRAL_BASELINE...")
    emotion_valid = EMOTION_STORIES_TINY.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    neutral_valid = EMOTION_NEUTRAL_BASELINE.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    print(f"  emotion: {len(emotion_valid)}, neutral: {len(neutral_valid)}")

    # ---- Collect mean-pooled Q, K, V per (layer, head) across all prompts ----
    print(f"\nCollecting Q, K, V at all 42 layers across "
          f"{len(emotion_valid) + len(neutral_valid)} prompts...")
    t0 = time.perf_counter()
    Q_data, K_data, V_data = {}, {}, {}
    labels = []
    _collect_qkv(model, emotion_valid, "emotion", Q_data, K_data, V_data, labels)
    _collect_qkv(model, neutral_valid, "neutral", Q_data, K_data, V_data, labels)
    labels = np.array(labels)
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    neutral_mask = labels == "neutral"
    print(f"  labels: {np.unique(labels, return_counts=True)}")

    # ---- Build probes + measure per (layer, head, stream) ----
    print(f"\nBuilding {N_LAYERS} x (8 Q + 2 K + 2 V) = "
          f"{N_LAYERS * (N_Q_HEADS + 2*N_KV_HEADS)} probes (x 6 emotions each)...")
    t0 = time.perf_counter()
    sil_Q, acc_Q, spec_Q = _analyze_stream(
        Q_data, labels, N_Q_HEADS, "q", neutral_mask,
    )
    sil_K, acc_K, spec_K = _analyze_stream(
        K_data, labels, N_KV_HEADS, "k", neutral_mask,
    )
    sil_V, acc_V, spec_V = _analyze_stream(
        V_data, labels, N_KV_HEADS, "v", neutral_mask,
    )
    print(f"  done in {time.perf_counter() - t0:.0f}s")

    # ---- Report ----
    print(f"\n{'=' * 70}")
    print(f"Stream-level summary (6-way silhouette, peak per stream)")
    print(f"{'=' * 70}")
    for stream_name, sil in [("Q", sil_Q), ("K", sil_K), ("V", sil_V)]:
        L_best, h_best = np.unravel_index(np.argmax(sil), sil.shape)
        acc_best_sil = (acc_Q if stream_name == "Q" else
                        acc_K if stream_name == "K" else acc_V)[L_best, h_best]
        tag = "GLOBAL" if L_best in GLOBAL_LAYERS else "local "
        print(f"  {stream_name}-stream peak: L{L_best} h{h_best} {tag}  "
              f"sil={float(sil[L_best, h_best]):+.4f}  "
              f"per-passage top-1 acc={float(acc_best_sil):.3f}")

    print(f"\n{'=' * 70}")
    print(f"Per-passage top-1 accuracy peaks (6-way chance = 16.7%)")
    print(f"{'=' * 70}")
    for stream_name, acc in [("Q", acc_Q), ("K", acc_K), ("V", acc_V)]:
        L_best, h_best = np.unravel_index(np.argmax(acc), acc.shape)
        tag = "GLOBAL" if L_best in GLOBAL_LAYERS else "local "
        print(f"  {stream_name}-stream peak: L{L_best} h{h_best} {tag}  "
              f"acc={float(acc[L_best, h_best]):.3f}  "
              f"(= {int(float(acc[L_best, h_best]) * 96)}/96)")

    # ---- Per-emotion specialists ----
    print(f"\n{'=' * 70}")
    print(f"Per-emotion specialists: top 3 (layer, head, stream) by margin")
    print(f"{'=' * 70}")
    for emotion in EMOTIONS:
        short = emotion.replace("emotion_", "")
        # Combine all streams
        all_specs = (
            [(L, h, "Q", m) for L, h, m, _, _ in spec_Q[emotion]] +
            [(L, h, "K", m) for L, h, m, _, _ in spec_K[emotion]] +
            [(L, h, "V", m) for L, h, m, _, _ in spec_V[emotion]]
        )
        all_specs.sort(key=lambda x: -x[3])
        print(f"\n  {short}:")
        for L, h, stream, margin in all_specs[:3]:
            tag = "GLOBAL" if L in GLOBAL_LAYERS else "local "
            print(f"    L{L:>2} h{h} {stream}  {tag}  margin={margin:+.3f}")

    # ---- Visualization ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: silhouette heatmaps
    vmax = max(abs(sil_Q).max(), abs(sil_K).max(), abs(sil_V).max())
    for col, (title, sil, nh) in enumerate([
        ("Q-stream", sil_Q, N_Q_HEADS),
        ("K-stream", sil_K, N_KV_HEADS),
        ("V-stream", sil_V, N_KV_HEADS),
    ]):
        ax = axes[0, col]
        im = ax.imshow(sil, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(nh))
        ax.set_yticks(range(0, N_LAYERS, 3))
        ax.set_title(f"{title}: 6-emotion silhouette")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Bottom row: per-passage accuracy
    amax = max(acc_Q.max(), acc_K.max(), acc_V.max())
    for col, (title, acc, nh) in enumerate([
        ("Q-stream", acc_Q, N_Q_HEADS),
        ("K-stream", acc_K, N_KV_HEADS),
        ("V-stream", acc_V, N_KV_HEADS),
    ]):
        ax = axes[1, col]
        im = ax.imshow(acc, aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=amax, interpolation="nearest")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(nh))
        ax.set_yticks(range(0, N_LAYERS, 3))
        ax.set_title(f"{title}: per-passage top-1 accuracy")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Per-head emotion probes across Q / K / V streams\n"
        "(6 emotions x 16 passages; chance per-passage accuracy = 16.7%)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "per_head_emotion_probes.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    np.savez_compressed(
        OUT_DIR / "per_head_emotion_probes.npz",
        sil_Q=sil_Q, sil_K=sil_K, sil_V=sil_V,
        acc_Q=acc_Q, acc_K=acc_K, acc_V=acc_V,
    )


if __name__ == "__main__":
    main()
