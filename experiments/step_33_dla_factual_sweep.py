"""Step 33 — DLA across FACTUAL_15: is the mid-layer wrong-answer phase systematic?

Step 32 found that on "The capital of France is" with target Paris, the
cumulative residual at L10-14 actively prefers Berlin to Paris, flipping
around L25. Open question from that finding: is the mid-layer wrong-answer
preference a property of the network, or specific to the Paris/Berlin pair?

This script runs the same DLA decomposition across all 15 FACTUAL_15
prompts, each paired with a plausible distractor. For each prompt we
compute (target - distractor) at every layer's resid_post; if the
mid-layer distractor-preferring phase is systematic, we should see
negative diffs concentrated in the L9-L24 band across many prompts.

Distractor rationale (same category, plausible competitor):
  Eiffel Tower / Paris    -> London       (major European capital)
  Capital of Japan / Tokyo -> Kyoto       (historical capital)
  Great Wall / China       -> Japan       (neighboring Asian country)
  Amazon River / Brazil    -> Peru        (Amazon also flows through Peru)
  Sahara / Africa          -> Asia        (continental distractor)
  Water = H + / oxygen     -> nitrogen    (common atmospheric gas)
  Speed of light / meters  -> miles       (imperial unit equivalent)
  Gold / Au                -> Ag          (silver; often confused)
  Romeo & Juliet / Shakespeare -> Marlowe (Elizabethan contemporary)
  Mona Lisa / Leonardo     -> Michelangelo (Renaissance Italian)
  1,2,3,4 / five           -> six         (plausible off-by-one)
  Mon,Tue / Wednesday      -> Thursday    (plausible off-by-one)
  Opposite of hot / cold   -> warm        (weaker antonym)
  Sky / blue               -> gray        (alternate sky color)
  Cats / pets              -> animals     (more general category)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.prompts.factual import FACTUAL_15  # noqa: E402
from mechbench_core import (  # noqa: E402
    Capture,
    GLOBAL_LAYERS,
    Model,
    N_LAYERS,
    accumulated_resid,
    logit_attrs,
)


DISTRACTORS: dict[str, str] = {
    "Paris": "London",
    "Tokyo": "Kyoto",
    "China": "Japan",
    "Brazil": "Peru",
    "Africa": "Asia",
    "oxygen": "nitrogen",
    "meters": "miles",
    "Au": "Ag",
    "Shakespeare": "Marlowe",
    "Leonardo": "Michelangelo",
    "five": "six",
    "Wednesday": "Thursday",
    "cold": "warm",
    "blue": "gray",
    "pets": "animals",
}


def first_token_id(model, word: str) -> int:
    ids = model.tokenizer.encode(" " + word, add_special_tokens=False)
    return int(ids[0])


def main() -> None:
    print("Loading model...")
    model = Model.load()

    n_prompts = len(FACTUAL_15.prompts)
    diffs = np.zeros((n_prompts, N_LAYERS), dtype=np.float32)
    target_logps = np.zeros((n_prompts, N_LAYERS), dtype=np.float32)  # unused, kept for shape
    labels: list[str] = []

    for p_idx, prompt in enumerate(FACTUAL_15.prompts):
        target = prompt.target
        distractor = DISTRACTORS[target]
        labels.append(f"{target}/{distractor}")

        ids = model.tokenize(prompt.text)
        t_id = first_token_id(model, target)
        d_id = first_token_id(model, distractor)

        print(f"\n[{p_idx+1:2d}/{n_prompts}] {target:12s} vs {distractor:12s}  "
              f"(t_id={t_id}, d_id={d_id})")

        interventions = [Capture.residual(range(N_LAYERS), point="post")]
        result = model.run(ids, interventions=interventions)
        stack = accumulated_resid(result.cache)  # [N_LAYERS, S, D]
        attrs = logit_attrs(model, stack, [t_id, d_id])  # [N_LAYERS, 2]
        diffs[p_idx] = attrs[:, 0] - attrs[:, 1]

        neg = (diffs[p_idx][9:25] < 0).sum()
        final = diffs[p_idx][-1]
        print(f"    L9-L24 layers with negative diff: {neg}/16   "
              f"final-layer diff: {final:+.2f}")

    # --- Aggregate analysis ---
    mean_diff = diffs.mean(axis=0)
    median_diff = np.median(diffs, axis=0)
    n_negative = (diffs < 0).sum(axis=0)

    print("\n--- Per-layer aggregates across 15 prompts ---")
    print("  layer   mean   median   #negative")
    for i in range(N_LAYERS):
        marker = " G" if i in GLOBAL_LAYERS else "  "
        print(f"  L{i:02d}{marker}  {mean_diff[i]:+7.2f}  {median_diff[i]:+7.2f}   "
              f"{n_negative[i]:2d}/{n_prompts}")

    # Mid-layer vs other-layer systematic test
    mid_mask = np.zeros(N_LAYERS, dtype=bool)
    mid_mask[9:25] = True
    mid_frac_neg = (diffs[:, mid_mask] < 0).mean()
    rest_frac_neg = (diffs[:, ~mid_mask] < 0).mean()
    print(f"\n  Fraction of negative diffs in L9-L24:  {mid_frac_neg:.2%}")
    print(f"  Fraction of negative diffs elsewhere:  {rest_frac_neg:.2%}")

    # Strongly-negative threshold
    strong_neg_mask = diffs < -2.0
    mid_strong = strong_neg_mask[:, mid_mask].sum()
    rest_strong = strong_neg_mask[:, ~mid_mask].sum()
    print(f"  Strongly-negative (diff < -2) cells in L9-L24: "
          f"{mid_strong} / {diffs[:, mid_mask].size}")
    print(f"  Strongly-negative cells elsewhere:             "
          f"{rest_strong} / {diffs[:, ~mid_mask].size}")

    # Last negative layer per prompt
    last_neg = []
    for p_idx in range(n_prompts):
        negs = np.where(diffs[p_idx] < 0)[0]
        last_neg.append(int(negs[-1]) if len(negs) > 0 else -1)
    print(f"\n  Last layer with negative diff, per prompt:")
    for l, p in zip(last_neg, FACTUAL_15.prompts):
        tag = " G" if l in GLOBAL_LAYERS else "  "
        print(f"    {p.target:12s} -> L{l:02d}{tag}")

    # --- Plot: heatmap of diff[prompt, layer] ---
    caches_dir = Path(__file__).resolve().parent.parent / "caches"
    caches_dir.mkdir(exist_ok=True)
    out = caches_dir / "dla_factual_sweep.png"

    fig, (ax_h, ax_agg) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    vmax = float(max(abs(diffs.min()), abs(diffs.max())))
    im = ax_h.imshow(
        diffs, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax_h.set_yticks(range(n_prompts))
    ax_h.set_yticklabels(labels, fontsize=8)
    ax_h.set_ylabel("prompt (target / distractor)")
    ax_h.set_title(
        "DLA across FACTUAL_15: (target - distractor) logit at each layer's resid_post\n"
        "red = target preferred; blue = distractor preferred"
    )
    for g in GLOBAL_LAYERS:
        ax_h.axvline(g, color="black", linewidth=0.3, alpha=0.3)
    plt.colorbar(im, ax=ax_h, label="Δ logit")

    ax_agg.bar(range(N_LAYERS), mean_diff, color="#555555",
               edgecolor="white", linewidth=0.3, label="mean")
    ax_agg.plot(range(N_LAYERS), median_diff, "o-", color="#d62728",
                markersize=3, linewidth=1, label="median")
    ax_agg.axhline(0, color="black", linewidth=0.5)
    ax_agg.set_xlabel("layer")
    ax_agg.set_ylabel("(target - distractor)")
    ax_agg.legend(loc="upper left")
    for g in GLOBAL_LAYERS:
        ax_agg.axvline(g, color="black", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
