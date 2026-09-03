# Spec 13 — Quant Tool

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 1, 2, 5, 8, 11. Constrains: spec 15.

## 0. Purpose and scope

`r9v-quant`: the offline tool that turns an f16 model into a native-format file (spec 2). Covers inputs, the calibration set, per-tensor bit-width assignment against a size target, folded activation smoothing, importance-weighted rounding on the spec 2 grids, activation-mode selection, expert hot hints, output metadata, verification and the quality receipt, and the `verify-arch` command that gates model families (spec 8 §8).

Out of scope: file layout (spec 2), the engine's numerics (spec 1 §6), pruning or training of any kind.

## 1. Principles

1. **Separate stack, one contract.** The tool is Python on torch; the engine is Rust. The only thing they share is spec 2. The tool never links the engine and the engine never runs Python.
2. **Same grids as GGUF, better values in them.** `I4_K` is field-identical to `Q4_K`; the tool's contribution is the choice of every `q`, `sc` and `mn`, plus per-tensor bit widths and folded smoothing. Any quality claim is A/B-able against `llama-quantize` output in the same kernel.
3. **The size target is the input; the mix is the output.** Users say how big (or what to fit); the tool decides which tensors get which width from measured sensitivity.
4. **Everything that affects bytes is recorded in the file.** Calibration hash, tool version, seed, preset and per-tensor decisions go into `r9v.*` metadata so a file explains itself.
5. **Reproducible.** Same input file, calibration set, config and seed produce byte-identical output.

## 2. Inputs

- **Model**: an F16 or BF16 GGUF. `convert_hf_to_gguf.py` from llama.cpp (or any converter producing standard GGUF metadata) is the way in from Hugging Face; the tool does not re-implement HF loading, tokenizer conversion or chat-template extraction. All `general.*`, `tokenizer.*` and `<arch>.*` metadata pass through untouched.
- **Family**: resolved from `general.architecture` using the same family table as spec 8, in the tool's own Python model implementations (one per family, torch, f16/f32, used for calibration forwards and reference logits). A family the tool doesn't implement cannot be quantized; adding one is the same PR as the spec 8 family.
- **Calibration set**: a manifest (§3).
- **Target**: one of `--preset quality|balanced|small`, `--bpw <f32>`, `--size <bytes>`, or `--fit vram=<bytes>,ctx=<tokens>[,seqs=<n>]`, which computes the weight budget as `vram − state(ctx, seqs) − workspace − reserve` using spec 3 §6.2 costs for the model.

## 3. Calibration set

```
cal/<name>.json
  name, version, sha256 of this manifest
  sources: [ { dataset, revision, split, filter, count } ]
  mix:     { code: 0.40, prose: 0.30, chat_tool_json: 0.20, math: 0.10 }
  seq_len: 2048
  count:   512            # sequences
  holdout: 64             # sequences reserved for verification, never used for fitting
  seed
```

- `r9v-cal-v1` ships as a manifest over public datasets; `r9v-quant cal build <manifest>` materializes it (downloads, filters, tokenizes with the model's tokenizer, caches under the manifest hash). Users can supply their own manifest; the hash of whatever was used goes into the file.
- The mix is weighted toward code and structured output because that is the engine's primary workload (spec 6 profile). A `--cal` override exists precisely because a chat-heavy deployment should calibrate on chat.
- Tokenization uses the model's tokenizer through the same implementation the engine uses (spec 9 §7), so calibration and serving agree on token boundaries.

## 4. Pipeline

```
1. load       model f16 into host RAM (layer-by-layer to GPU during passes); calibration tokens
2. collect    activation statistics per matmul input: per-channel absmax, Hessian diagonal H = E[x xᵀ] (per layer, streamed)
3. smooth     per-channel factors, folded into weights (§5)
4. sensitize  per-tensor error proxy at int4 and int8 from H and a trial quantization (§6.1)
5. assign     bit widths to hit the target (§6.2)
6. round      GPTQ error-feedback rounding onto the assigned grid, per tensor (§7)
7. act-mode   choose i8 PerToken / e4m3 / f16 per tensor (§8)
8. hints      expert activation frequency, placement hints (§9)
9. emit       native GGUF per spec 2 with r9v.* metadata (§10)
10. verify    KL / top-1 / perplexity on the holdout vs f16; write the quality receipt (§11)
```

Steps 2–7 run one layer at a time with that layer on the GPU and the rest in host RAM, so a 30B model quantizes on a single 32 GB card (or a 24 GB one). The tool runs on CUDA or ROCm torch; CPU works and is slow. Wall time for a 30B dense model on one R9700: roughly 1–2 hours, dominated by step 6.

## 5. Folded smoothing

SmoothQuant-style per-input-channel factors `s_j = max|x_j|^α / max|w_·j|^(1−α)`, `α = 0.5` by default (`--alpha`), computed from step 2 statistics. Weights become `W' = W · diag(s)` and the activation the kernel sees is `x / s`, which must be produced for free. Where the `1/s` goes, per matmul group:

| matmul | input comes from | fold `1/s` into |
|---|---|---|
| `qkv` | pre-attention norm | the norm's weight `γ' = γ / s` |
| `gate`/`up` | pre-FFN norm | the norm's weight |
| `o` | attention output channel `j = Σ attn · v_j` | row `j` of `v_proj` (scale by `1/s_j`) |
| `down` | `silu(g_j) · u_j` | row `j` of `up_proj` only (silu is nonlinear; gate stays) |
| expert `gate`/`up` | shared pre-FFN norm | the norm (one `s` per layer for all experts, computed over routed tokens) |
| expert `down` | expert-local `silu(g) · u` | that expert's `up` rows |
| `lm_head` | final norm | the final norm's weight |
| MLA projections | norm / latent | norm, or the preceding low-rank projection's output rows |

Every group folds with zero runtime cost, so `quant_act.smoothing = Folded` is true for all matmuls in a native file and the engine runs plain per-token absmax. The folded `γ'` and scaled rows are what get quantized in step 6; the tool never emits an explicit smoothing vector.

Tensors whose `α`-smoothed activations still show outliers beyond `--outlier_ratio` (default 20× median absmax) are flagged for §8.

## 6. Sensitivity and assignment

### 6.1 Error proxy

For each tensor, a trial quantization at int4 (`I4_K`) and at int8 (`I8_B128`) with plain round-to-nearest gives `ΔW`; the proxy is the GPTQ objective `ε = Σ_i Σ_j H_jj ΔW_ij²`, normalized by the tensor's contribution to the residual stream (`‖W‖_H` over the same `H`). This costs one pass per tensor and correlates with final-logit KL well enough for ranking; the assignment is validated with real KL in §6.2.

### 6.2 Assignment

1. Start every matmul tensor at `I4_K`; `lm_head`, MTP heads, `qk_norm`-adjacent projections and all vectors at their spec 2 defaults; embeddings at `I4_K` rows (or the head's scheme when tied).
2. Compute the budget in bytes from the target (§2).
3. Promote tensors from `I4_K` to `I8_B128` in descending order of `(ε_int4 − ε_int8) / Δbytes` until the next promotion would exceed the budget.
4. Validate: quantize (round-to-nearest, fast) and measure KL vs f16 on 32 calibration sequences. If KL exceeds the preset's ceiling (`quality` 0.005, `balanced` 0.02, `small` 0.05 mean per-token KL), report it; with `--fit` or `--size` the tool keeps the budget and reports the KL rather than exceeding the size.
5. The chosen mix is printed as a per-layer table and stored in metadata.

Presets resolve to bpw targets on a typical dense model of about 6.2 (`quality`), 5.2 (`balanced`) and 4.5 (`small`), but the preset is defined by the KL ceiling, not the bpw; the bpw is whatever the model needs to meet it.

## 7. Rounding

GPTQ (OBQ with Cholesky-based error feedback, column blocks of 128, damping 0.01, act-order on by default) using `H` from step 2, per tensor on the GPU. The grid it rounds onto is the assigned spec 2 scheme:

- **`I8_B128` / `I8_R`**: per-block or per-row f16 scale fitted by absmax after smoothing; GPTQ rounds against it.
- **`I4_K`**: for each 256-superblock, the 6-bit `sc` and `mn` per 32-block and the f16 `d`, `dmin` are fitted by an importance-weighted search (weights `H_jj`) over scale/min candidates, the same shape of search as llama.cpp's K-quant fitter but with the Hessian as the importance; GPTQ then rounds `q` against the fitted grid with error feedback across columns. The fit is re-run once after GPTQ's first pass to let the grid adapt to the compensated values.
- **`E4M3_B128`**: per-block f16 scale by absmax; e4m3 round-to-nearest with GPTQ feedback.

Determinism: torch deterministic algorithms on, fixed thread counts for CPU reductions, seeded. Two runs with the same inputs are byte-identical, and CI checks this on a small model.

## 8. Activation mode

Per matmul tensor, after rounding, measure the layer-output error with activations quantized per-token int8 versus f16 on 32 sequences:

- within `--act_tolerance` (default: ≤ 10% of the weight-quantization error) → `act = i8/PerToken`
- otherwise, try `e4m3` per-token; if within tolerance → the tensor is re-quantized as `E4M3_B128` with `act = e4m3/PerToken` (keeps the fp8 WMMA path)
- otherwise → `act = f16/None` (the tensor keeps its int scheme; the kernel dequantizes to f16). Reported loudly, since it costs prefill speed.

In practice with folded smoothing nearly everything lands in the first bucket; the metadata records the result per tensor either way.

## 9. Hints

- **Expert hot hints**: routing frequency per expert over the calibration set, normalized, stored as `r9v.tensor.<exps>.hot_hint` (spec 8 §7 reads it). Also the mean number of distinct experts per token per layer, which spec 5 §3.4 uses to predict the cold rate.
- **Placement hints** (spec 2 §4): experts and n-gram tables `tiered`, everything else `device`.
- **2:4 sparsity**: `--sparse-check` verifies which tensors already satisfy 2:4 along K (models trained or pruned that way) and emits them as `L1S`. The tool does not prune.

## 10. Output

A native GGUF per spec 2 §6, tensor order in decode-graph consumption order, with:

```
r9v.format_version, r9v.layout_id = "L1", r9v.arch_hint
r9v.quant_tool.version, r9v.quant_tool.seed, r9v.quant_tool.preset, r9v.quant_tool.target
r9v.calibration.name / .hash / .tokens / .mix
r9v.smoothing.folded = true, r9v.smoothing.alpha
r9v.tensor.<name>.{scheme, act, roles, interleave, sparse, placement_hint, residency_unit, regions, xxh3, eps_int4, eps_int8}
r9v.quality.*        (§11)
```

Fusion interleaving (`qkv`, `gate_up`) is applied at emit time according to the family's spec 8 declarations, so the file is zero-copy for the engine.

## 11. Verification and the quality receipt

`r9v-quant verify <file>` (also the last pipeline step):

- Runs the holdout split through the tool's f16 reference (torch) and through the quantized weights **using the engine** (`r9v eval --logits`), not the tool's own dequant, so the numbers reflect what the engine will actually compute. `r9v eval` uses the fast tier on a supported GPU when one is present (that is literally what the engine computes, and spec 1 §6.1 makes it bit-stable), otherwise the CPU tier (T0v with T0 fallback); the receipt records which, and `--holdout-tokens` can shrink the CPU run.
- Reports mean and p99 per-token KL, top-1 and top-5 agreement, perplexity on a fixed public text, per-layer KL contribution, and the size, bpw and per-scheme byte totals.
- Writes `r9v.quality.{kl_mean, kl_p99, top1, top5, ppl, holdout_hash, engine_version}` into the file and prints a table.
- A file whose KL exceeds its preset ceiling still emits (the user may want it) but the receipt says `ceiling: exceeded`.

`r9v-quant compare <a> <b>` runs the same measurement on two files (e.g. native `I4_K` vs `llama-quantize` `Q4_K_M` of the same model) and prints them side by side; this is the A/B that backs any quality claim (spec 15).

## 12. `verify-arch`

Gates a spec 8 family: runs an F16 GGUF through the engine's CPU tier (T0v with T0 fallback, so the result is independent of any GPU) and through the tool's torch implementation of the family (and, if `transformers` is installed and the HF model is available, through it as a third reference) on the 64 fixed prompts; requires L1-tolerance logits and top-1 ≥ 99.9% between engine and torch. Output is committed to `support/<family>/<model>.json` and referenced by the support matrix.

## 13. CLI

```
r9v-quant quantize --in model-f16.gguf --out model.r9v.gguf
                   (--preset quality|balanced|small | --bpw 5.4 | --size 20e9 | --fit vram=32e9,ctx=32768)
                   [--cal r9v-cal-v1] [--alpha 0.5] [--seed 0] [--device cuda:0|rocm:0|cpu] [--sparse-check]
r9v-quant verify   model.r9v.gguf [--holdout <manifest>]
r9v-quant compare  a.gguf b.gguf
r9v-quant inspect  model.r9v.gguf            # per-tensor table: scheme, act, bytes, eps, placement
r9v-quant cal build <manifest.json>
r9v-quant verify-arch model-f16.gguf [--hf <repo>]
```

## 14. Requirements

Python ≥ 3.11, torch ≥ 2.5 (CUDA or ROCm build), `gguf` (llama.cpp's Python package, for reading and writing the container), `numpy`, `safetensors` (for `verify-arch --hf`), optional `transformers`. Pinned in `tools/r9v-quant/pyproject.toml`. GPU with ≥ 16 GB recommended; host RAM ≥ 1.2 × the f16 model size.
