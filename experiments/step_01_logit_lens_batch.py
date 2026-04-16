"""Multi-prompt logit lens on Gemma 4 E4B.

Projects every layer's resid_post through the model's tied unembed at the
final position, for each prompt in FACTUAL_15. Aggregates per-layer rank
and log-probability trajectories and identifies the largest single-layer
rank drops.

Per finding 01: the answer's rank crashes from ~100k+ in early layers to
0 by layer 41, with the bulk of the transition concentrated in layers
27-36. The 5 largest rank drops do NOT land on global-attention layers,
which was the original hypothesis.

Run from project root:
    python experiments/step_01_logit_lens_batch.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma4_mlx_interp import (  # noqa: E402
    Capture, GLOBAL_LAYERS, Model, N_LAYERS,
    lens_trajectory, logit_lens_final, logprob_trajectory,
)
from experiments.prompts import FACTUAL_15  # noqa: E402

OUT_DIR = ROOT / "caches"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = Model.load()

    print(f"\nValidating prompts...\n")
    valid = FACTUAL_15.validate(model)
    print()
    if len(valid) < 3:
        print("Too few valid prompts to draw conclusions.")
        return

    capture = Capture.residual(layers=range(N_LAYERS), point="post")
    all_ranks = np.zeros((len(valid), N_LAYERS), dtype=np.float64)
    all_logprobs = np.zeros((len(valid), N_LAYERS), dtype=np.float64)

    print(f"Running logit lens for {len(valid)} prompts...")
    for j, vp in enumerate(valid):
        result = model.run(vp.input_ids, interventions=[capture])
        ranks, logprobs = logit_lens_final(model, result.cache, vp.target_id)
        all_ranks[j] = ranks
        all_logprobs[j] = logprobs

    # Aggregate: geomean rank (using log(rank+1) to handle rank=0).
    geomean_rank = np.exp(np.mean(np.log(all_ranks + 1), axis=0)) - 1
    mean_logprob = np.mean(all_logprobs, axis=0)

    # ---- Summary table ----
    print(f"\n{'layer':>5}  {'type':>7}  {'geomean_rank':>13}  {'mean_logp':>10}")
    print("-" * 42)
    for i in sorted(set(list(range(0, N_LAYERS, 6)) + [N_LAYERS - 1])):
        kind = "GLOBAL" if i in GLOBAL_LAYERS else "local"
        print(f"{i:>5}  {kind:>7}  {geomean_rank[i]:>13.1f}  {mean_logprob[i]:>10.3f}")

    # ---- Largest single-layer rank drops ----
    rank_delta = np.diff(geomean_rank)
    biggest_drops = np.argsort(rank_delta)[:5]
    print(f"\nLargest rank drops (layer i -> i+1):")
    for idx in biggest_drops:
        target_layer = int(idx) + 1
        kind = "GLOBAL" if target_layer in GLOBAL_LAYERS else "local"
        print(f"  layer {idx:>2} -> {target_layer:>2} ({kind:>6}): "
              f"geomean rank {geomean_rank[idx]:.0f} -> {geomean_rank[target_layer]:.0f}  "
              f"(Δ = {rank_delta[idx]:.0f})")

    n_global_in_top5 = sum(1 for idx in biggest_drops if (int(idx) + 1) in GLOBAL_LAYERS)
    print(f"\n  {n_global_in_top5} / 5 biggest drops land on global-attention layers")

    # ---- Plot: rank trajectory + logprob trajectory ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    lens_trajectory(
        all_ranks, ax=axes[0],
        title=f"Logit lens across {len(valid)} prompts — Gemma 4 E4B",
    )
    logprob_trajectory(all_logprobs, ax=axes[1])

    plt.tight_layout()
    out_path = OUT_DIR / "logit_lens_batch.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
