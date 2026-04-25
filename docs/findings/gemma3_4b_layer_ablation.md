# Preliminary: no L23-style pivot in Gemma 3 4B (single overdetermined prompt)

A first probe for task [000187](https://github.com/mechbench/mechbench/blob/main/tasks/mechbench-experiments/open/000187-test-l23-pivot-generalization-on-26b-a4b-and-31b.md): does the L23-style architectural pivot reproduce in a Gemma family member that has *no* `num_kv_shared_layers`? The motivating Gemma 4 candidates (26B-A4B and 31B) don't fit on this 32 GB Mac at bf16; the 12B Gemma 3 hung indefinitely inside `mlx_vlm.load`. Falling back to **Gemma 3 4B** (~7 GB at bf16, 34 layers, globals at [5, 11, 17, 23, 29], no KV sharing) gets us a Gemma-family non-E-series datapoint.

## What I did

Single-prompt layer ablation on `mlx-community/gemma-3-4b-it-bf16`:

- Prompt: `"Complete this sentence with one word: The Eiffel Tower is in"`, rendered through `tokenizer.apply_chat_template` (Gemma 3 instruct demands the chat template; bare-prompt encoding produced a `' ____'` placeholder top-1 in a first attempt).
- Method: replace `lm.layers[i]` in mlx-vlm's Gemma 3 layer list with an identity callable, run the forward, record Δ log p of the baseline top-1. Repeat for each of the 34 layers.
- Note: this bypasses `mechbench-core` because mechbench-core's forward path is hardcoded to `mlx_vlm.models.gemma4`. A small one-off in `bin/probe_gemma3_4b_ablation.py`.

## Result

Baseline top-1 = `'Paris'` with **log p = 0.0** (probability ≈ 1.0 — the prompt is fully overdetermined). Per-layer Δ log p of the baseline top-1 across the 34 layers:

| layer | type | Δ log p |
|---|---|---|
| 0 | sliding | −17.66 |
| 1 | sliding | −5.01 |
| 2 | sliding | 0.00 |
| 3 | sliding | **−30.88** ← peak |
| 4 | sliding | −15.58 |
| 5 | **global** | 0.00 |
| 6 | sliding | −9.75 |
| 7 | sliding | 0.00 |
| 8 | sliding | −12.50 |
| 9 | sliding | 0.00 |
| 10 | sliding | −1.50 |
| 11 | **global** | −0.01 |
| 12 | sliding | −0.00 |
| 13–32 | mixed | 0.00 (within rounding) |
| 33 | sliding | −0.01 |

**No pivot.** The damage curve is front-loaded with peak at **L3** (a sliding layer); after L10 every layer shows essentially zero damage. The Gemma-3-4B globals at L5, L11, L17, L23, L29 do *not* form a peak — L5 and L11 are quiet, the rest are completely silent. Compare to E4B's "invisible middle" (L10-24 all damaging) and L23 pivot: nothing analogous here.

## The big caveat

Baseline log p = 0.0 means the model is *certain* the answer is "Paris". A fully-confident prediction has no room to be hurt by ablating layers that operate on a residual stream that already encodes the answer. The damage we *do* see in layers 0-10 is presumably the early computation that establishes "Paris" as the answer; once it's locked in, the late layers contribute zero because they're polishing an already-correct answer.

In other words: **this single prompt cannot distinguish "no pivot exists in Gemma 3 4B" from "the prompt is too easy to expose any late-layer involvement at all."** The original L23 finding rested on the FACTUAL_15 battery with a `MIN_CONFIDENCE = 0.5` filter — prompts where the model is uncertain enough that late-layer involvement is observable.

## What this is and isn't evidence for

**It is** evidence that:
- The methodology works (chat-templated prompt produces sensible top-1; identity-layer ablation produces meaningful damage curves where there's room for damage).
- **Late-layer involvement in Gemma 3 4B is conspicuously absent on a fully-confident prompt**, in a way that didn't hold for E4B even on confident prompts (FACTUAL_15 had the L10-24 damage band). Whether this is a model-size effect, an architecture-family effect, or just the prompt is open.

**It is not** evidence that:
- Gemma 3 4B has no L23-style pivot (we didn't measure that — we measured a single prompt at logp=0.0).
- The KV-boundary framing is right or wrong (4B has no KV sharing, but the prompt-confidence confound dominates).

## Follow-up

The right next step is a FACTUAL_15-equivalent battery on Gemma 3 4B with the confidence filter from `experiments/export_step_02_for_ui.py`. That gives the proper damage curve — comparable to the E4B figure — and lets us actually evaluate the pivot question. Filing as task `000189`.

Until that runs, **the open question this writeup leaves is**: is L3-as-peak a real Gemma-3-4B feature (as opposed to E4B's L23 / E2B's L14 pivots), or an artifact of an overdetermined prompt? The current data is consistent with either.

## Sources

- Probe script: [`bin/probe_gemma3_4b_ablation.py`](../../bin/probe_gemma3_4b_ablation.py)
- Raw data: [`caches/gemma3_4b_layer_ablation.json`](../../caches/gemma3_4b_layer_ablation.json)
- Gemma 3 4B [config](https://huggingface.co/mlx-community/gemma-3-4b-it-bf16/raw/main/config.json)
- Original L23 finding context: [`docs/findings/gemma4_global_spacing.md`](gemma4_global_spacing.md)
