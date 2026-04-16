# Surface-Form Switching at the Final Layer

**Date:** 2026-04-16
**Script:** `experiments/step_14_surface_form_switching.py`
**Plot:** `caches/surface_form_switching.png`

## Hypothesis

Finding 01 noted in passing that for the Eiffel Tower prompt, the logit-lens top-1 token switched from `' Paris'` (with a leading space) at layer 36 to `'Paris'` (no space) at layer 41 — same word, different tokenizer variant. We hypothesized that the final global-attention layer was doing surface-form / tokenization calibration rather than semantic retrieval.

This experiment tests whether the pattern is systematic across the FACTUAL_15 cohort or whether it was a one-prompt coincidence.

## Method

For each FACTUAL_15 prompt, capture `resid_post` at all 42 layers and project each through the tied unembed at the final position. For each layer transition `i → i+1`, classify the rank-1 change:

- **same**: rank-1 token id is unchanged
- **surface-form**: token id changed but the decoded strings normalize to the same string under `unicodedata.normalize("NFKC", s).strip().lower()` — e.g. `' Paris'` → `'Paris'`, `'Cold'` → `'cold'`
- **semantic**: rank-1 changed to a meaningfully different token

Aggregate across the 14 prompts that pass `FACTUAL_15.validate(model)`.

## Results

### Layer 40 → 41 is almost entirely surface-form

| Outcome at the 40 → 41 transition | Count |
|------------------------------------|------:|
| Same token (no change)             | 3 / 14 |
| Surface-form switch                | **11 / 14** |
| Semantic switch                    | **0 / 14** |

For the 11 prompts that did switch, the change was always the same shape: drop a leading space.

| Prompt | At layer 40 | At layer 41 |
|--------|-------------|-------------|
| The Eiffel Tower is in | `' Paris'` | `'Paris'` |
| The capital of Japan is | `' Tokyo'` | `'Tokyo'` |
| The Great Wall is in | `' China'` | `'China'` |
| The Sahara Desert is in | `' Africa'` | `'Africa'` |
| Water is made of hydrogen and | `' oxygen'` | `'oxygen'` |
| Romeo and Juliet was written by | `' Shakespeare'` | `'Shakespeare'` |
| The Mona Lisa was painted by | `' Leonardo'` | `'Leonardo'` |
| One, two, three, four, | `' five'` | `'five'` |
| Monday, Tuesday, | `' Wednesday'` | `'Wednesday'` |
| The color of the sky on a clear day is | `' blue'` | `'blue'` |
| Cats are popular household | `' pets'` | `'pets'` |

The other 3 prompts were either already at the no-space variant by layer 40, or never made it past `chemical symbol for gold → 'Au'` — but all of them were "same" at 40 → 41, and none switched semantically.

### The transition is not a layer-41 quirk — it's a 38 → 41 band

Surface-form switches don't appear out of nowhere at the final layer. They ramp up over the last few transitions:

| Transition | Surface | Semantic | Note |
|-----------:|--------:|---------:|------|
| 35 → 36 (global) | 0 | 2 | |
| 36 → 37 | 0 | 0 | |
| 37 → 38 | 0 | 1 | |
| 38 → 39 | 4 | 0 | |
| 39 → 40 | 4 | 0 | |
| **40 → 41 (global)** | **11** | **0** | final layer |

By contrast, the rest of the network does almost exclusively semantic work:

| Band | Surface switches | Semantic switches |
|------|----------------:|------------------:|
| Early (transitions 0–9, into layers 1–10) | 2 | 119 |
| Handoff (25–32, into layers 26–33) | 2 | 77 |
| Late (35–40, into layers 36–41) | **19** | **1** |

The visual pattern in the plot is a wall of dark red (semantic) from layers 0–37, then a clean handoff to pink (surface-form) for layers 38–41. The single "late" semantic switch is at the 37 → 38 transition for one prompt; everything else from 38 onward is either same or surface.

### Why drop the space?

The leading-space variant exists because in standard English text continuation, a new word starts with a space. Tokenizers learn this: `' Paris'` is the canonical token for "Paris" in continuous text.

But the FACTUAL_15 prompts are wrapped in Gemma's chat template, which closes the user turn with `<turn|>\n<|turn>model\n` before the model speaks. The model's response begins **at the start of a new line**, not after a space, so the no-space variant is the contextually correct first token.

The model spends layers 0–37 figuring out *what to say* using the easier-to-learn space-prefixed variant (which has higher base frequency in training data — most words in continuous text are preceded by a space). The last 3–4 layers then convert to the no-space variant that the chat template actually requires. The final global layer (41) is the heaviest part of that conversion: 11 of 14 prompts make the switch there.

## Interpretation

This is a clean, narrow result that sharpens what's been observed about layer 41 in earlier findings. Previously the single most-attention-critical layer was layer 23 (finding 04). Layer 41 didn't stand out in any ablation, attention-pattern, or causal-tracing analysis — those all said it does little. This finding identifies what the little is: **layer 41's primary job for factual-recall prompts in chat-template format is to swap the rank-1 token from the space-prefixed variant to the no-space variant**, leaving the semantic content unchanged.

That is a calibration task, not a retrieval or reasoning task. The model has decided what word to say by layer 37 or so; the last four layers are tokenization bookkeeping.

A few specific implications:

1. **The last block isn't doing semantics.** No semantic top-1 changes happen anywhere in the 38 → 41 band across 14 prompts. If you're studying *which token will be predicted*, you can stop computing at layer 37 and still get the right word — you'll just decode it as ' Paris' instead of 'Paris'.

2. **Surface-form calibration is concentrated at the final global layer.** Layers 38, 39, 40 each handle ~4 prompts; layer 41 handles 11. Whatever wires layer 41 to act as the surface-form calibrator, it's doing the bulk of the work.

3. **This is a chat-template artifact.** A bare-text prompt (no chat template wrapping) probably wouldn't show this pattern, because the model would correctly predict the space-prefixed continuation. The behavior is specific to instruction-tuned models running in chat mode. Worth checking against the base model if available, or against a non-chat prompting style.

4. **Earlier framing was right but understated.** Finding 05's section on attention-pattern progression said L41 does "position anchoring" / "surface-form selection" — but treated those as roughly equal. The data here shows surface-form selection is by far the dominant role: 11 surface-form switches out of 14 prompts is almost a unanimous behavior, not just one of several things layer 41 does.

## Limitations

- Single template. All FACTUAL_15 prompts use "Complete this sentence with one word: ...". A different prompting style (e.g. Q&A, or omitting the "one word" instruction) might shift where the calibration happens.
- Single model variant. E4B with the bf16 instruction-tuned weights. The base model (without instruction tuning) probably doesn't show this pattern; worth checking if accessible.
- 14 prompts is enough for the pattern to be obvious but small enough that we can't subdivide further (e.g. by category) reliably.
- The "surface-form" classification is purely string-equivalence after normalization. It would call `'Au'` → `'AU'` a surface-form switch, which is what we want; but it would also call genuine semantic alternatives that happen to share a normalization (none in this corpus) the same. Not a problem for this experiment; worth flagging if the technique is reused.
