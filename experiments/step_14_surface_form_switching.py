"""Surface-form token swap investigation.

Hypothesis (per finding 01): the final global-attention layer (41) does
surface-form / tokenization calibration rather than semantic retrieval.
The triggering observation was that on the Eiffel Tower prompt, the
logit-lens top-1 switched from ' Paris' (with a leading space) at layer
36 to 'Paris' (no space) at layer 41 — same word, different tokenizer
variant.

This experiment tests whether that pattern is systematic across the
FACTUAL_15 cohort. For each prompt and each layer transition i->i+1,
classify the rank-1 change as:
  - same: rank-1 token id is unchanged
  - surface: token id changed but decoded strings normalize to the same
    string (case + leading whitespace + unicode-NFKC). E.g. ' Paris' ->
    'Paris', 'Cold' -> 'cold', 'PARIS' -> 'paris'.
  - semantic: rank-1 changed to a meaningfully different token.

If layer 40 -> 41 has many surface-form switches and few semantic ones,
the last block is doing tokenization calibration. If the surface-form
switches are scattered randomly across layers, the Eiffel observation
was a coincidence.

Run from project root:
    python experiments/step_14_surface_form_switching.py
"""

import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechbench_core import (  # noqa: E402
    Capture, GLOBAL_LAYERS, Model, N_LAYERS,
)
from experiments.prompts import FACTUAL_15  # noqa: E402

OUT_DIR = ROOT / "caches"

CODE_SAME = 0
CODE_SURFACE = 1
CODE_SEMANTIC = 2


def _normalize(s: str) -> str:
    """Normalize a decoded token for surface-form-equivalence comparison."""
    return unicodedata.normalize("NFKC", s).strip().lower()


def _classify(old_id: int, old_tok: str, new_id: int, new_tok: str) -> int:
    if old_id == new_id:
        return CODE_SAME
    if _normalize(old_tok) == _normalize(new_tok):
        return CODE_SURFACE
    return CODE_SEMANTIC


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print("\nValidating prompts...\n")
    valid = FACTUAL_15.validate(model)
    n = len(valid)
    print()

    # rank1_ids[j, i] = rank-1 token id for prompt j at layer i
    rank1_ids = np.zeros((n, N_LAYERS), dtype=np.int64)
    rank1_toks: list[list[str]] = []
    capture = Capture.residual(layers=range(N_LAYERS), point="post")

    print(f"Running logit lens for {n} prompts...")
    for j, vp in enumerate(valid):
        result = model.run(vp.input_ids, interventions=[capture])
        toks_for_prompt = []
        for i in range(N_LAYERS):
            resid = result.cache[f"blocks.{i}.resid_post"]
            logits = model.project_to_logits(resid)
            last = logits[0, -1, :].astype(mx.float32)
            mx.eval(last)
            top1 = int(np.argmax(np.array(last)))
            rank1_ids[j, i] = top1
            toks_for_prompt.append(model.tokenizer.decode([top1]))
        rank1_toks.append(toks_for_prompt)

    # Classify every transition
    transitions = np.zeros((n, N_LAYERS - 1), dtype=np.int8)
    for j in range(n):
        for i in range(N_LAYERS - 1):
            transitions[j, i] = _classify(
                int(rank1_ids[j, i]), rank1_toks[j][i],
                int(rank1_ids[j, i + 1]), rank1_toks[j][i + 1],
            )

    # ---- Per-prompt printout: only show transitions that aren't 'same' ----
    print("\nPer-prompt rank-1 transitions (only changes shown):")
    for j, vp in enumerate(valid):
        print(f"\n--- {vp.prompt.text[:55]!r} -> {vp.target_token!r} ---")
        switches = [i for i in range(N_LAYERS - 1) if transitions[j, i] != CODE_SAME]
        if not switches:
            print(f"  No rank-1 changes (top-1 was {rank1_toks[j][0]!r} from layer 0)")
            continue
        for i in switches:
            kind = "surface" if transitions[j, i] == CODE_SURFACE else "semantic"
            old_tok = rank1_toks[j][i]
            new_tok = rank1_toks[j][i + 1]
            global_marker = " *" if (i + 1) in GLOBAL_LAYERS else ""
            print(f"  layer {i:>2} -> {i+1:>2}{global_marker:<3s}  ({kind:>8s}): "
                  f"{old_tok!r:>15s} -> {new_tok!r}")

    # ---- Aggregate per layer-transition ----
    n_surface_per = (transitions == CODE_SURFACE).sum(axis=0)
    n_semantic_per = (transitions == CODE_SEMANTIC).sum(axis=0)

    print(f"\n{'=' * 60}")
    print(f"Switch counts per layer-transition (across {n} prompts)")
    print(f"{'=' * 60}")
    print(f"{'transition':>13}  {'surface':>8}  {'semantic':>8}  {'global?':>8}")
    print("-" * 50)
    for i in range(N_LAYERS - 1):
        s, sem = int(n_surface_per[i]), int(n_semantic_per[i])
        if s + sem == 0:
            continue
        is_global = "GLOBAL" if (i + 1) in GLOBAL_LAYERS else ""
        print(f"  {i:>3} -> {i+1:>3}    {s:>8}  {sem:>8}  {is_global:>8}")

    # Specific call-out: layer 40 -> 41 (the final transition)
    print(f"\n{'=' * 60}")
    print(f"Final transition (layer 40 -> 41) details")
    print(f"{'=' * 60}")
    n40_surface = int(n_surface_per[40])
    n40_semantic = int(n_semantic_per[40])
    n40_same = n - n40_surface - n40_semantic
    print(f"  Same:           {n40_same:>2} / {n}")
    print(f"  Surface-form:   {n40_surface:>2} / {n}")
    print(f"  Semantic:       {n40_semantic:>2} / {n}")
    if n40_surface > 0 or n40_semantic > 0:
        print(f"\n  Per-prompt detail:")
        for j, vp in enumerate(valid):
            if transitions[j, 40] != CODE_SAME:
                kind = "surface" if transitions[j, 40] == CODE_SURFACE else "semantic"
                print(f"    [{kind:>8s}] {rank1_toks[j][40]!r:>15s} -> {rank1_toks[j][41]!r:>15s}  "
                      f"({vp.prompt.text[:40]!r})")

    # Totals
    total_surface = int((transitions == CODE_SURFACE).sum())
    total_semantic = int((transitions == CODE_SEMANTIC).sum())
    print(f"\n{'=' * 60}")
    print(f"Totals across all {n} prompts x {N_LAYERS - 1} transitions = "
          f"{n * (N_LAYERS - 1)} cells")
    print(f"{'=' * 60}")
    print(f"  surface-form switches: {total_surface}")
    print(f"  semantic switches:     {total_semantic}")
    print(f"  same:                  {n * (N_LAYERS - 1) - total_surface - total_semantic}")

    # Where are the surface vs semantic switches concentrated?
    band_late = list(range(35, 41))  # transitions targeting layers 36-41
    band_handoff = list(range(25, 33))  # transitions targeting layers 26-33
    band_early = list(range(0, 10))
    for label, band in [("early (-> layer 1-10)", band_early),
                        ("handoff (-> layer 26-33)", band_handoff),
                        ("late (-> layer 36-41)", band_late)]:
        s = int(n_surface_per[band].sum())
        sem = int(n_semantic_per[band].sum())
        print(f"  {label:<28s}: {s:>3} surface, {sem:>3} semantic")

    # ---- Plot: stacked bar chart per layer-transition ----
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(N_LAYERS - 1)
    ax.bar(x, n_semantic_per, label="semantic switch", color="#d62728")
    ax.bar(x, n_surface_per, bottom=n_semantic_per,
           label="surface-form switch", color="#ff9896")
    ax.set_xlabel("transition (layer i -> i+1, x label = i)")
    ax.set_ylabel(f"# prompts with rank-1 change (of {n})")
    ax.set_title("Surface-form vs semantic rank-1 switches per layer transition")
    ax.set_xticks(np.arange(0, N_LAYERS - 1, 3))
    ax.legend(loc="upper left")
    for g in GLOBAL_LAYERS:
        if 0 <= g - 1 < N_LAYERS - 1:
            ax.axvline(g - 1, color="#999999", linestyle="--",
                       linewidth=0.7, alpha=0.6)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = OUT_DIR / "surface_form_switching.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
