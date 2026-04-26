# Layer-ablation on Llama 3.2 3B Instruct: front-loaded + last-layer, no L23-style pivot

Cross-family layer-ablation per task 000208. Same FACTUAL_15 methodology as step_02 (E4B), step_34 (Gemma 3 4B), step_35 (E2B), step_39 (Qwen 2.5 3B). First Llama datapoint on the cross-family L23-pivot scoreboard.

Through mechbench-core's mlx-lm fallback path with the new Llama-family forward (000208 added the `_forward_llama.py` module + `model_type='llama'` arch dispatch).

## Architecture

- 28 transformer layers, hidden_size=3072, MLP hidden 8192
- GQA: 24 heads, 8 KV heads (3:1 sharing within a layer; not the same as cross-layer KV-sharing)
- Every layer is global attention (no `layer_types` field; all 28 in `arch.global_layers`)
- No KV-sharing across layers, no MatFormer side-channel
- Tied embedding (`tie_word_embeddings=True`)

## What I found

13/15 prompts validated (top-1 prob ≥ 0.5 through chat template). 364 ablated forward passes, ~50s on M-series unified memory.

```
L 0  -11.43      L 7   -0.16      L14  -0.25      L21  -0.10
L 1   -9.85      L 8   -0.14      L15  -0.30      L22  -0.19
L 2   -0.48      L 9   -0.29      L16  -0.15      L23  -0.09
L 3   -0.31      L10  -0.30      L17   -0.00      L24  -0.12
L 4   -0.42      L11   -0.34      L18  -0.05      L25  -0.15
L 5   -0.22      L12  -0.25      L19  -0.04      L26  -0.06
L 6   -0.28      L13  -0.07      L20  -0.10      L27  -2.54
```

**Pattern: front-loaded (L0, L1) + last-layer (L27). The 25 middle layers are individually expendable** — most have mean Δlogp shallower than -0.35; L17 is essentially a no-op (-0.001).

- Top-3 most damaging: L0 (-11.4), L1 (-9.9), L27 (-2.5). Then L2 (-0.48), L4 (-0.42), L11 (-0.34).
- No layer in the middle of the network shows the kind of damage E4B's L23 shows (E4B L23 mean Δlogp ≈ -0.5 against a much shallower middle baseline, plus the six-angle convergence story).
- Median per-layer damage: -0.21. Distribution is bimodal: two huge layers, one big layer, twenty-five small ones.

## Cross-family scoreboard (layer-ablation only)

| model | n_layers | shape | peak (mean Δlogp) | mid-network pivot? |
|---|---:|---|---|---|
| Gemma 4 E4B | 42 | front-loaded + late pivot | L23 (with six-angle convergence) | **yes** (the original L23 finding) |
| Gemma 4 E2B | 30 | front-loaded + late pivot | L13/L14 | partial (rotation transfers, ablation peak less sharp) |
| Gemma 3 4B | 34 | front-loaded + last-layer | L0, L33 | **no** |
| Qwen 2.5 3B Instruct | 36 | front-loaded + last-layer | L0, L35 | **no** |
| **Llama 3.2 3B Instruct** | **28** | **front-loaded + last-layer** | **L0 (-11.4), L27 (-2.5)** | **no** |

**Llama 3.2 3B joins Gemma 3 4B and Qwen 2.5 3B in the no-pivot bucket.** That's three architecturally different families converging on the same shape: a couple of irreplaceable early layers, an irreplaceable final layer, and a wide middle where any single layer can be dropped with minimal damage.

## What this rules in and out for the L23-pivot narrative

The surviving narrative from step_42 (Qwen perplexity probe) was that the L23-pivot signature — and especially the orthogonal Lyra rotation at the KV-sharing boundary — is **fresh-K/V → KV-shared transition specific** to Gemma 4 E. Llama 3.2 has no such transition (no `num_kv_shared_layers`, no MatFormer side-channel, no hybrid attention pattern; just plain GQA with global attention everywhere). The framing predicts no L23-style pivot. **Step_43 confirms.**

What this *doesn't* yet rule on:

- The same-scale-as-E4B test. Llama 3.2 3B is a 3B/28-layer model; E4B is a 4B/42-layer hybrid model. The size + depth mismatch leaves a small gap. Step_44 (Llama 3.1 8B, 32 layers) is the closer comparison and is queued.
- The Gemma-family vs Gemma-4-E-only question. Gemma 3 4B already showed no pivot (step_34), but with no Lyra-rotation probe yet. The follow-up filed in step_42's findings (run perplexity-probe on Gemma 3 4B) is the cleanest test.

## L27 is the final layer — what is it doing?

The last-layer hotspot in Llama 3.2 (-2.54) is much smaller than the L0/L1 hotspots (-11) but much larger than any middle layer. Same pattern as Gemma 3 4B's L33 and Qwen 2.5 3B's L35. The most parsimonious read: in models without a mid-network architectural pivot, the **final block is doing a non-trivial last-mile read of the residual stream into the unembed direction** — ablating it forces the unembed to read a residual that hasn't been finalized for the prediction-token position. Worth a follow-up DLA experiment to confirm.

## Sources

- Battery script: [`experiments/step_43_layer_ablation_llama3_2_3b.py`](../../experiments/step_43_layer_ablation_llama3_2_3b.py)
- Raw data: [`mechbench-ui/public/data/step_43_layer_ablation_llama3_2_3b.json`](../../../mechbench-ui/public/data/step_43_layer_ablation_llama3_2_3b.json)
- Cross-family references: step_02 (E4B), step_34 (Gemma 3 4B), step_35 (E2B), step_39 (Qwen 2.5 3B), step_42 (Qwen perplexity probe)
- Family-wide picture: [`docs/essays/gemma_family_global_spacing.md`](../essays/gemma_family_global_spacing.md)
