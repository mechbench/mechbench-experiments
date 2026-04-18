"""Scale-up: emotion probes from a more-diverse corpus.

Goal: test whether MORE DIVERSE corpora fix the two diagnosable failures
from step_24:
  - `calm` tracks scenes (lakes, tea, rain) rather than abstract states.
    Failed on retreat-length intensity axis (went wrong direction).
  - `happy` saturates at moderate-scale positive events. Did not scale
    with magnitude on the lottery-winnings axis.

Test: rebuild probes from EMOTION_STORIES_GENERATED, which covers the
same 6 emotions as the original EMOTION_STORIES_TINY but with
deliberately-curated topic diversity targeting those failures. Same
size (72 vs 96 passages), same emotion breakdown, different topic mix:

  - calm now mixes 6 topics balancing scene ('rainy afternoon',
    'forest path') with state ('peace after conflict', 'breathing
    exercises', 'veteran facing a crisis calmly')
  - happy now spans intensity deliberately (lost wallet returned,
    long-awaited reunion, major career award, first full sentence,
    shock-joy, quiet satisfaction)

Re-run step_24's four intensity axes and compare side-by-side.

(Note: 'generated' here means Claude-authored, not Gemma-generated.
mlx_vlm.generate is broken for our model and rolling a naive
autoregressive loop was taking 30+ minutes of local compute. Since the
probe is built from Gemma 4's ACTIVATIONS on the text, any competent
emotion-labeled corpus works the same way; the generator does not need
to be the target model. See gemma4_mlx_interp/generate.py for the
framework primitive used when the target and generator are the same.)

Run from project root:
    python experiments/step_25_emotion_probes_generated.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Model, Probe, Prompt, PromptSet, fact_vectors_pooled, intensity_curve,
)
from experiments.prompts import (  # noqa: E402
    EMOTION_NEUTRAL_BASELINE,
    EMOTION_STORIES_GENERATED,
    EMOTION_STORIES_TINY,
)

OUT_DIR = ROOT / "caches"
LAYER = 28
POOL_START = 20


INTENSITY_AXES = [
    {"key": "tylenol", "description": "Tylenol dose (mg)",
     "expected_up": "emotion_afraid", "expected_down": "emotion_calm",
     "template": "I just took {X} mg of Tylenol for my back pain. Should I be concerned?",
     "levels": [500, 1500, 5000, 10000, 20000]},
    {"key": "lottery", "description": "Lottery winnings ($)",
     "expected_up": "emotion_happy", "expected_down": None,
     "template": "I just won ${X} in the lottery. What should I do first?",
     "levels": [50, 500, 5000, 50000, 500000]},
    {"key": "theft", "description": "Amount stolen by contractor ($)",
     "expected_up": "emotion_angry", "expected_down": "emotion_calm",
     "template": "My contractor disappeared after taking ${X} from me. What are my legal options?",
     "levels": [500, 5000, 50000, 500000]},
    {"key": "retreat", "description": "Silent meditation retreat length (days)",
     "expected_up": "emotion_calm", "expected_down": None,
     "template": "I just finished a {X}-day silent meditation retreat. What should I do with the feeling?",
     "levels": [1, 3, 7, 14, 30]},
]


def _build_probes(model, corpus: PromptSet) -> tuple[dict[str, Probe], list[str]]:
    emotion_valid = corpus.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    neutral_valid = EMOTION_NEUTRAL_BASELINE.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    emotion_vecs = fact_vectors_pooled(
        model, emotion_valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    neutral_vecs = fact_vectors_pooled(
        model, neutral_valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    labeled = {
        e: emotion_vecs[emotion_valid.labels == e]
        for e in emotion_valid.categories
    }
    probes = Probe.from_labeled_corpus(
        labeled, neutral_vecs, layer=LAYER, explain=0.5,
    )
    return probes, emotion_valid.categories


def _run_diagonal(model, probes, emotion_order, corpus, label):
    valid = corpus.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    vecs = fact_vectors_pooled(
        model, valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    probe_names = list(probes.keys())
    scores = np.stack(
        [probes[p].score(vecs) for p in probe_names], axis=1,
    )
    labels_arr = valid.labels
    agg = np.zeros((len(emotion_order), len(probe_names)), dtype=np.float32)
    for i, e in enumerate(emotion_order):
        mask = labels_arr == e
        if mask.any():
            agg[i] = scores[mask].mean(axis=0)
    diag_hits = sum(
        1 for i, e in enumerate(emotion_order)
        if probe_names[int(np.argmax(agg[i]))] == e
    )
    per_top1 = np.array(
        [probe_names[int(np.argmax(scores[j]))] for j in range(len(scores))]
    )
    correct = (per_top1 == labels_arr).sum()
    print(f"[{label:>10s}]  diag={diag_hits}/{len(emotion_order)}  "
          f"per-passage={correct}/{len(labels_arr)} "
          f"= {100 * correct / len(labels_arr):.1f}%")
    return agg, scores


def _fmt_x(v: int) -> str:
    return f"{v:,}" if v >= 1000 else str(v)


def _run_axis(model, axis, probes, probe_order):
    texts = [axis["template"].format(X=_fmt_x(v)) for v in axis["levels"]]
    prompts = PromptSet(
        name=f"INTENSITY_{axis['key']}",
        prompts=tuple(Prompt(text=t, category=axis["key"]) for t in texts),
    )
    valid = prompts.validate(
        model, verbose=False, min_confidence=0.0, require_target_match=False,
    )
    vecs = fact_vectors_pooled(
        model, valid, layers=[LAYER], start=POOL_START,
    )[LAYER]
    scores = np.stack(
        [probes[p].score(vecs) for p in probe_order], axis=1,
    )
    return scores


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    # ---- Build BOTH probe sets for direct comparison ----
    print(f"\n{'=' * 70}")
    print(f"Building probe sets")
    print(f"{'=' * 70}")
    print(f"\nOriginal (step_21): {len(EMOTION_STORIES_TINY)} hand-curated passages")
    probes_orig, emotion_order = _build_probes(model, EMOTION_STORIES_TINY)
    print(f"\nScaled (step_25):   {len(EMOTION_STORIES_GENERATED)} Claude-authored passages")
    probes_gen, _ = _build_probes(model, EMOTION_STORIES_GENERATED)

    # ---- Self-consistency on both corpora, both probe sets ----
    print(f"\n{'=' * 70}")
    print(f"Self-consistency: each probe set scored on ITS OWN training corpus")
    print(f"{'=' * 70}")
    _run_diagonal(model, probes_orig, emotion_order, EMOTION_STORIES_TINY, "orig/orig")
    _run_diagonal(model, probes_gen, emotion_order, EMOTION_STORIES_GENERATED, "gen/gen")

    # Cross-evaluation — do the generated probes generalize to the hand-curated
    # passages, and vice versa? This is the strongest corpus-independence test.
    print(f"\n{'=' * 70}")
    print(f"Cross-corpus generalization")
    print(f"{'=' * 70}")
    _run_diagonal(model, probes_orig, emotion_order, EMOTION_STORIES_GENERATED, "orig/gen")
    _run_diagonal(model, probes_gen, emotion_order, EMOTION_STORIES_TINY, "gen/orig")

    # ---- Intensity modulation: side-by-side ----
    print(f"\n{'=' * 70}")
    print(f"Intensity modulation: original vs generated probes")
    print(f"{'=' * 70}")

    probe_order = list(probes_orig.keys())
    results_orig = {}
    results_gen = {}
    for axis in INTENSITY_AXES:
        print(f"\n--- {axis['description']} ---")
        s_orig = _run_axis(model, axis, probes_orig, probe_order)
        s_gen = _run_axis(model, axis, probes_gen, probe_order)
        results_orig[axis["key"]] = s_orig
        results_gen[axis["key"]] = s_gen

        # Compare the key probe's delta between levels[0] and levels[-1] for
        # both corpora.
        up = axis["expected_up"]
        dn = axis["expected_down"]
        if up:
            ui = probe_order.index(up)
            d_orig = s_orig[-1, ui] - s_orig[0, ui]
            d_gen = s_gen[-1, ui] - s_gen[0, ui]
            mono_orig = all(s_orig[i + 1, ui] >= s_orig[i, ui] for i in range(len(s_orig) - 1))
            mono_gen = all(s_gen[i + 1, ui] >= s_gen[i, ui] for i in range(len(s_gen) - 1))
            short = up.replace("emotion_", "")
            print(f"  target ({short:>7s}) UP:  "
                  f"orig delta = {d_orig:+.3f} {'mono' if mono_orig else '    '}  "
                  f"gen delta = {d_gen:+.3f} {'mono' if mono_gen else '    '}")
        if dn:
            di = probe_order.index(dn)
            d_orig = s_orig[-1, di] - s_orig[0, di]
            d_gen = s_gen[-1, di] - s_gen[0, di]
            mono_orig = all(s_orig[i + 1, di] <= s_orig[i, di] for i in range(len(s_orig) - 1))
            mono_gen = all(s_gen[i + 1, di] <= s_gen[i, di] for i in range(len(s_gen) - 1))
            short = dn.replace("emotion_", "")
            print(f"  antipode ({short:>5s}) DN:  "
                  f"orig delta = {d_orig:+.3f} {'mono' if mono_orig else '    '}  "
                  f"gen delta = {d_gen:+.3f} {'mono' if mono_gen else '    '}")

    # ---- Plot: 4 axes x 2 columns (orig left, generated right) ----
    probe_colors = {
        "emotion_happy": "#f0a500", "emotion_sad": "#2d81b6",
        "emotion_angry": "#d62728", "emotion_afraid": "#6a3d9a",
        "emotion_calm": "#2ca02c", "emotion_proud": "#ff7f0e",
    }
    n_axes = len(INTENSITY_AXES)
    fig, axes = plt.subplots(n_axes, 2, figsize=(13, 3.5 * n_axes))
    short_colors = {p.replace("emotion_", ""): c for p, c in probe_colors.items()}
    short_names = [p.replace("emotion_", "") for p in probe_order]
    for row_idx, axis in enumerate(INTENSITY_AXES):
        for col_idx, (label_text, scores) in enumerate([
            ("hand-curated (step_21)", results_orig[axis["key"]]),
            ("more-diverse (step_25)", results_gen[axis["key"]]),
        ]):
            intensity_curve(
                levels=axis["levels"],
                scores=scores,
                series_names=short_names,
                target_up=(axis["expected_up"] or "").replace("emotion_", "") or None,
                target_down=(axis["expected_down"] or "").replace("emotion_", "") or None,
                colors=short_colors,
                log_x=True,
                xlabel=axis["description"],
                title=f"{axis['description']}\n[{label_text}]",
                ax=axes[row_idx, col_idx],
            )

    fig.suptitle(
        "Intensity modulation: hand-curated (left) vs more-diverse "
        "(right) corpora\nDoes targeted topic diversity fix the happy-"
        "saturation and calm-as-ambiance failures from step_24?",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "emotion_probes_intensity_compared.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
