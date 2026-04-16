"""Smoke test for the L0 canonical hook-aware forward pass.

Two acceptance criteria:

1. NUMERICAL: Model.run produces final-token logits that match the existing
   project-root forward.py path bitwise (or to within bf16 numerical
   tolerance). forward.py is the thinnest known-working forward — it mirrors
   what mlx_vlm.generate does internally before its first model call. If
   Model.run matches forward.py, we know it matches the working reference
   path even if upstream mlx_vlm.generate happens to be broken.

2. SEMANTIC: Top-1 next-token at the final position is the expected answer
   for each FACTUAL_15-style prompt. This is the human-readable sanity
   check — if Paris doesn't fall out of 'The Eiffel Tower is in', something
   is wrong even if the numbers match.

Both must pass for L0 to be considered ready.

Run from project root with the venv active:
    python -m gemma4_mlx_interp._smoke

Side effect: this resolves the long-standing 'calling model(input_ids)
directly produces garbage' bug noted in CLAUDE.md, since Model.run IS the
working forward path consolidated into one place.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward import forward as ref_forward  # noqa: E402

from . import Model, all_hook_names  # noqa: E402

PROMPTS = [
    ("Complete this sentence with one word: The Eiffel Tower is in", "Paris"),
    ("Complete this sentence with one word: The capital of Japan is", "Tokyo"),
    ("Complete this sentence with one word: Romeo and Juliet was written by",
     "Shakespeare"),
    ("Complete this sentence with one word: The opposite of hot is", "cold"),
    ("Complete this sentence with one word: Monday, Tuesday,", "Wednesday"),
]

# bf16 has ~3 decimal digits of precision; in practice two paths through
# the same arithmetic land within ~1e-2 in absolute logit value.
LOGIT_TOLERANCE = 0.05


def main() -> int:
    print("Loading model...")
    t0 = time.perf_counter()
    model = Model.load()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s. "
          f"({len(all_hook_names())} hook points exposed.)")

    print(f"\nL0 smoke test on {len(PROMPTS)} prompts.\n")
    print(f"  {'expected':>11}  {'top1 (run)':>14}  "
          f"{'max|Δlogit|':>11}  {'argmax_match':>12}  {'sem_match':>9}")
    print("  " + "-" * 70)

    all_pass = True
    for prompt, expected in PROMPTS:
        ids = model.tokenize(prompt)

        # Path under test.
        result = model.run(ids)
        run_last = np.array(result.last_logits.astype(mx.float32))
        run_id = int(np.argmax(run_last))
        # .strip() handles the leading-space tokenizer artifact (' Paris' vs
        # 'Paris'). We deliberately do NOT .lower() — the model DOES
        # capitalize proper nouns correctly at the final layer (see
        # findings/step_01_logit_lens_batch.md), and lower-casing would hide
        # case errors if they ever occur.
        run_tok = model.tokenizer.decode([run_id]).strip()

        # Reference path: project-root forward.py.
        ref_logits, _ = ref_forward(model._model, model._processor, prompt)
        ref_last = np.array(ref_logits[0, -1, :].astype(mx.float32))
        ref_id = int(np.argmax(ref_last))

        max_abs_delta = float(np.max(np.abs(run_last - ref_last)))
        argmax_match = run_id == ref_id
        sem_match = run_tok == expected

        ok = argmax_match and max_abs_delta < LOGIT_TOLERANCE and sem_match
        if not ok:
            all_pass = False
        print(f"  {expected:>11}  {run_tok!r:>14}  "
              f"{max_abs_delta:>11.6f}  {str(argmax_match):>12}  "
              f"{str(sem_match):>9}")

    print()
    if not all_pass:
        print("SMOKE TEST FAILED.")
        print("  - argmax_match=False means Model.run picks a different top-1 than forward.py.")
        print("  - max|Δlogit| above tolerance means the two paths drift numerically.")
        print("  - sem_match=False means the model isn't producing the expected answer.")
        print("Investigate _forward.py before building L1 on top.")
        return 1

    print("Smoke test passed:")
    print("  - Model.run produces logits numerically equivalent to forward.py.")
    print("  - Top-1 next-token matches the expected answer on every prompt.")
    print("L0 is ready; L1 may begin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
