# Layer-ablation on Llama 3.1 8B Instruct: same-scale-as-E4B confirms front-loaded + last-layer

The same-scale-as-E4B Llama datapoint that task 000208 explicitly called out. 8B/32-layer Llama is the closest non-Gemma comparison we can make to E4B (4B/42-layer hybrid). Same FACTUAL_15 methodology as step_02 (E4B), step_34 (Gemma 3 4B), step_39 (Qwen 2.5 3B), step_43 (Llama 3.2 3B).

## Architecture

- 32 transformer layers, hidden_size=4096, MLP hidden 14336
- GQA: 32 heads, 8 KV heads
- All 32 layers are global attention (no `layer_types`); no cross-layer KV-sharing, no MatFormer
- Untied unembed (`tie_word_embeddings=False`) — separate `lm_head`

## What I found

15/15 prompts validated. 480 ablated forward passes, ~120s.

```
L 0  -12.23      L 8  -0.30      L16  -0.10      L24  -0.06
L 1  -11.96      L 9  -0.34      L17  -0.18      L25  -0.16
L 2   -0.02      L10  -0.07      L18  -0.20      L26  -0.16
L 3   -0.03      L11  -0.18      L19  -0.05      L27  -0.02
L 4   -0.33      L12  -0.18      L20  -0.26      L28  -0.57
L 5   -0.40      L13  -0.25      L21  -0.18      L29  -0.01
L 6   -0.10      L14  -0.18      L22  -0.03      L30  -0.31
L 7   -0.07      L15  -0.55      L23  -0.04      L31  -1.70
```

**Pattern: front-loaded (L0=-12.2, L1=-12.0) + last-layer (L31=-1.7).** Same shape as Llama 3.2 3B at smaller scale. Middle layers are individually shallow (median -0.18); the largest middle bump is L28 (-0.57), which is still ~20× shallower than the L0 hotspot.

- Top-5 most damaging: L0 (-12.2), L1 (-12.0), L31 (-1.7), L28 (-0.57), L15 (-0.55).
- L15 and L28 are mild bumps within an otherwise flat middle. Nowhere near the magnitude or context (six-angle convergence) of E4B's L23.
- No layer in the middle 28 layers has mean Δlogp deeper than -0.6.

## What this rules in for the surviving narrative

The surviving narrative coming out of step_42 (Qwen perplexity probe): the L23 pivot — and especially the orthogonal Lyra rotation — is **fresh-K/V → KV-shared transition specific** to Gemma 4 E. Llama 3.1 8B has no such transition. The framing predicts no L23-style pivot in Llama at any scale. **Step_43 (3B) and step_44 (8B) both confirm.**

Two things matter about this datapoint specifically:

1. **It controls for size.** "L23 pivot doesn't appear in Qwen 2.5 3B" leaves a "maybe it needs a bigger model" dodge open. Llama 3.1 8B is 2.7× the parameter count of E4B's *full* size (and 4× the active-path) — bigger, not smaller. Still no pivot.
2. **It controls for depth.** 32 layers vs E4B's 42 — close enough that "depth fraction ~0.55" should still produce something distinctive in the L17-L18 range if the phenomenon were depth-fraction-driven across families. It doesn't.

Both controls lean the same way: the pivot lives in the architectural transition, not in scale or depth.

## Cross-family scoreboard (final form for layer-ablation)

| model | n_layers | shape | mid-network pivot? |
|---|---:|---|---|
| Gemma 4 E4B | 42 | front-loaded + L23 + last | **yes** (six-angle convergence) |
| Gemma 4 E2B | 30 | front-loaded + L13/14 + last | partial (rotation transfers, ablation peak less sharp) |
| Gemma 3 4B | 34 | front-loaded + last | no |
| Qwen 2.5 3B Instruct | 36 | front-loaded + last | no |
| Llama 3.2 3B Instruct | 28 | front-loaded + last | no |
| **Llama 3.1 8B Instruct** | **32** | **front-loaded + last** | **no** |

Five non-Gemma-4-E datapoints (Gemma 3, Qwen 2.5, Llama 3.2/3.1 at two scales). All five are front-loaded + last-layer with no mid-network pivot. The L23 phenomenon is confirmed Gemma-4-E-specific within the families we've tested.

## What still needs testing

- **Gemma 3 4B perplexity probe** (the Lyra-rotation cross-check on a same-family-no-KV-sharing model). Filed as step_42 follow-up.
- **A non-Gemma model with hybrid attention** (e.g., something with mixed local/global layers) to test whether the *hybrid pattern alone* — independent of KV-sharing — produces any pivot signature. None on mlx-vlm/mlx-lm at our hardware budget yet.

## Sources

- Battery script: [`experiments/step_44_layer_ablation_llama3_1_8b.py`](../../experiments/step_44_layer_ablation_llama3_1_8b.py)
- Raw data: [`mechbench-ui/public/data/step_44_layer_ablation_llama3_1_8b.json`](../../../mechbench-ui/public/data/step_44_layer_ablation_llama3_1_8b.json)
- Cross-family references: step_02 (E4B), step_34 (Gemma 3 4B), step_35 (E2B), step_39 (Qwen 2.5 3B), step_43 (Llama 3.2 3B)
- Family-wide picture: [`docs/essays/gemma_family_global_spacing.md`](../essays/gemma_family_global_spacing.md)
