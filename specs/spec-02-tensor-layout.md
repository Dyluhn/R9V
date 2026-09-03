# Spec 2 — Tensor Layout and Weight Format

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: spec 1. Constrains: specs 4, 5, 9, 13.

## 0. Purpose and scope

Defines how weights exist in three places and how they move between them:

- **on disk**: the native R9V format and standard GGUF
- **in VRAM / host RAM**: the logical tile layout that kernels consume
- **in the loader**: repack rules from every supported GGUF type into that layout

It also fixes the quantization schemes, the activation-side metadata a tensor carries, structural flags (fused gate/up, tied embeddings, 2:4 sparsity), placement tiers, alignment, and format versioning.

Out of scope: KV/state cache formats (spec 3), how kernel variants are selected per scheme (spec 4), the row cache over tiered tensors (spec 9), how the quant tool decides bit widths (spec 13).

## 1. Principles

1. **Repack, never requantize.** Loading a standard GGUF changes the arrangement of bytes, never their values. Every scheme below that has a GGUF source reproduces it bit-exactly.
2. **Nothing dequantizes to a wider type on load.** Every integer-representable format, including codebook IQ types, reaches at worst the int8 matrix path with expansion in registers. Weight bytes streamed from VRAM are the bytes on disk.
3. **Logical layout, versioned.** The tile layout is a named permutation (`L1`), not "whatever gfx12 wants." gfx12's native fragment order happens to equal `L1`, so the loader can map zero-copy; an arch whose native order differs gets a permute on load. Zero-copy is an optimization, not the contract.
4. **Bits in instruction order.** Within a tile, each lane's elements are K-consecutive within one output row, so the same bytes feed `dot4` in GEMV and WMMA in GEMM with no repacking between them.
5. **Scales are structure-of-arrays**, grouped so a wave loads every row's scale record for a K-block in one contiguous read.
6. **Structure is a flag, not a layout.** Fused gate/up, fused QKV, tied embeddings and 2:4 sparsity are per-tensor flags over the same tile layout.
7. **GGUF is the container.** Native files are GGUF v3 with R9V tensor type IDs and `r9v.*` metadata, so tokenizer, chat template and metadata tooling keep working, and the loader has one parser.

## 2. Layouts

### 2.1 `L0` — row-major (lookup tables and vectors)

Used for `embed_gather` tables, `ngram_gather` tables, norm weights, biases, router biases. Plain `[rows, dim]`, contiguous rows, scale records per (row, K-block) stored immediately after each row's values so a single row plus its scales is one contiguous region. This is what makes row-granular residency (spec 9) and `Tiered` placement work.

### 2.2 `L1` — tiled (matmul weights)

For a weight `W[N, K]` (output rows N, reduction K), padded to `N % 16 == 0` and `K % 16 == 0` (K additionally padded to the scheme's superblock where one exists). Padding rows and columns are zero.

**Tile**: 16 rows (N) × 16 columns (K) = 256 elements. **Lane order** within a tile, for 32 lanes with 8 elements each:

```
lane  = kgroup * 16 + n        # kgroup ∈ {0,1}, n ∈ 0..15
elem  = lane * 8 + j           # j ∈ 0..7
value = W[n_base + n, k_base + kgroup*8 + j]
```

So lane `l` holds `W[n = l % 16, k = (l/16)*8 .. (l/16)*8 + 7]`: eight K-consecutive elements of one output row. This is the gfx12 WMMA B-fragment order for `Wᵀ` and is also two `dot4` operands.

**Tile order**: row-block major, K inner:

```
tile_index = (n_base / 16) * (K / 16) + (k_base / 16)
```

A row-block of 16 output rows is therefore one contiguous stream over all of K, which is the access pattern of both the GEMV kernel (one wave per row-block streaming K) and the GEMM kernel (row-blocks × K-tiles).

**Element packing per dtype** (bytes per lane per tile):

| dtype | packing | bytes/lane | load |
|---|---|---|---|
| `i4` | 2 per byte, low nibble = lower k | 4 | one 32-bit |
| `i8`, `e4m3`, `e5m2` | 1 per byte | 8 | one 64-bit |
| `f16`, `bf16` | 2 bytes | 16 | one 128-bit |
| `i6`, `i5`, `i3`, `i2` (repack-only) | scheme-defined bit planes per tile, see §3 | — | two loads |

Tiles are 256-byte aligned regardless of dtype (a 128-byte int4 tile is followed by 128 bytes of the next tile; alignment is per region, not per tile, see §7).

### 2.3 `L1S` — tiled, 2:4 structured sparse

Same as `L1` over the compressed K dimension (`K/2`), followed by an index region: per tile, 2 bits per kept element in the lane order SWMMAC expects (spec 4 fixes the exact operand order). A tensor is `L1S` only if the quant tool verified the 2:4 constraint holds on every group of four along K. GEMV on `L1S` expands to dense in registers; the sparsity benefit at batch 1 is bytes only.

### 2.4 LayoutId in the arch descriptor

`ArchDescriptor.fragment_layout` names the arch's native fragment order. Loader rule: if `tensor.layout == arch.fragment_layout` → map directly; else → permute into `arch.fragment_layout` on load and record the repack in the doctor bundle. A future arch with a K=32 fragment gets `L2`, and existing files still load.

## 3. Quantization schemes

A scheme fixes: value bits, block structure, scale record format, dequant formula, and which matrix path it reaches. All schemes share `L1` for values; only the scale record differs.

### 3.1 Scale record placement (all `L1` schemes)

Scale records are stored in a separate region per tensor, grouped as `[N/16][K/B][16 records]` where `B` is the scheme's outer block (superblock where one exists, else the block). One wave processing row-block `nb` and K-block `kb` loads `16 × record_size` contiguous bytes. Record contents per scheme below.

### 3.2 Native schemes (emitted by the quant tool)

| id | values | block | record per (row, block) | formula | bpw | path |
|---|---|---|---|---|---|---|
| `I8_R` | i8 | row | `s: f16` (stored `[N]`) | `w = s·q` | 8.0 | dot4 / iu8 |
| `I8_B128` | i8 | 128 | `s: f16` | `w = s·q` | 8.125 | dot4 / iu8 |
| `I4_K` | u4 | 32 in 256 | per 256: `d: f16, dmin: f16, sc[8]: u6, mn[8]: u6` (12 B packed) | `w = d·sc·q − dmin·mn` | 4.5 | dot4 / iu4 |
| `E4M3_B128` | e4m3 | 128 | `s: f16` | `w = s·q` | 8.125 | fp8 WMMA |

`I4_K` is field-identical to GGUF `Q4_K`. The native tool differs from `llama-quantize` only in how it chooses `q`, `sc`, `mn` (GPTQ-style rounding with folded smoothing, spec 13), not in what it stores. This is deliberate: one int4 kernel serves native and repacked files, and there is no 4-bit "R9V-only" quality claim that can't be A/B'd against Q4_K in the same kernel.

`E4M3_B128` exists for tensors where calibration shows fp8 activations are needed; the quant tool emits it only with `act_dtype = e4m3`.

### 3.3 Repack-only schemes (GGUF sources; loader emits, tool never does)

| id | GGUF source | values | scale record | bpw | path |
|---|---|---|---|---|---|
| `I8_B32F` | `Q8_0` | i8 | per 32: `s: f16` | 8.5 | dot4 / iu8 |
| `I4_B32F` | `Q4_0` | u4 (zero = 8) | per 32: `s: f16` | 4.5 | dot4 / iu4 |
| `I4_B32FM` | `Q4_1` | u4 | per 32: `s: f16, m: f16` | 5.0 | dot4 / iu4 |
| `I5_B32F` / `I5_B32FM` | `Q5_0` / `Q5_1` | u5 as u4 + high-bit plane | as Q4 | 5.5 / 6.0 | unpack → iu8 |
| `I5_K` | `Q5_K` | u5 | as `I4_K` | 5.5 | unpack → iu8 |
| `I6_K` | `Q6_K` | i6 | per 256: `d: f16, sc[16]: i8` | 6.56 | unpack → iu8 |
| `I3_K`, `I2_K` | `Q3_K`, `Q2_K` | u3 / u2 | as GGUF | 3.4 / 2.6 | unpack → iu8 (reference tier at v1) |
| `I4_NL`, `I4_XS` | `IQ4_NL`, `IQ4_XS` | u4 index into 16-entry i8 LUT | as GGUF | 4.5 / 4.25 | LUT expand → iu8 |
| `IQ3_*`, `IQ2_*`, `IQ1_*` | same names | codebook indices | as GGUF | 1.5–3.5 | LDS LUT expand → iu8 (reference tier at v1) |

Rules:
- Bit planes for 5/6-bit types are stored per tile in the same lane order as the low nibbles, so a lane's eight values are still one or two loads.
- `IQ4_NL/XS` reach the int8 path, not `iu4`, because the LUT values are int8 and not a uniform 4-bit grid. This is stated in the load log so nobody expects `iu4` rates from IQ4.
- The `IQ2/IQ3/IQ1` family is supported for correctness from day one through the reference kernel. Promotion to a fast path is demand-driven; until then the load log reports "reference tier".
- `F16`/`BF16` GGUF tensors keep their dtype and go to the f16 WMMA path. `F32` tensors are only accepted for vectors (`L0`).

### 3.4 Activation-side metadata (per tensor)

```
act: {
  dtype:  i8 | e4m3 | f16
  scheme: PerToken | PerBlock32 | None
  smoothing_folded: bool
}
```

- Native tensors: `i8 PerToken` with `smoothing_folded = true` (default), or `e4m3 PerToken` where the tool chose `E4M3_B128`, or `f16 None` for tensors the tool marks as needing full-precision activations.
- Standard GGUF (no `r9v.*` metadata): default `i8 PerBlock32`, `smoothing_folded = false`. This is the llama.cpp MMQ contract (per-32-block int8 activations, per-block rescale in the K loop) and is the parity path. A config option can switch a model to `PerToken`; `r9v-quant verify` (spec 13 §11) run on the resulting logits is the gate for doing so.
- `f16` activations with integer weights means dequant-to-f16 in registers and the f16 WMMA path. Always available, slowest prefill, highest fidelity.

Spec 1's `QuantScheme` for activations is extended to `PerToken | PerBlock32` accordingly.

## 4. Structural flags

Per tensor, in `r9v.tensor.<name>.*` (defaults apply to standard GGUF):

- **`roles: [matmul] | [embed] | [lm_head] | [embed, lm_head] | [ngram_table] | [vector]`** (spec 8 §2) — a tensor with both `embed` and `lm_head` roles is tied. It is stored once, in `L1` at the scheme the tool chose for the head role (typically `I8_*`). `embed_gather` on an `L1` tensor reads a row as 16-element strided pieces from its row-block; rows are small and this costs nothing measurable, so tied tensors are never duplicated.
- **`interleave: none | gate_up | qkv`** — tiles of the member tensors alternate at tile granularity (`gate[t], up[t], gate[t+1], ...`) so the fused kernel reads one stream. Member scale regions are likewise interleaved per row-block. Standard GGUF is never interleaved on disk; the loader interleaves on repack when the model definition declares the fusion.
- **`sparse: none | s24`** — `L1S`. Requires a scheme in {`I8_R`, `I8_B128`, `I4_K`, `E4M3_B128`}.
- **`placement_hint: device | host | tiered`** — the quant tool's recommendation (experts and n-gram tables default `tiered`, everything else `device`). Config overrides.
- **`residency_unit: tensor | expert | row`** — the granularity spec 9 may cache at. `expert` requires the expert dimension to be the outermost region (§7). `row` requires `L0`.

## 5. Placement and residency

| class | layout | placement allowed | residency unit |
|---|---|---|---|
| dense matmul weights | `L1` / `L1S` | `Device` | tensor |
| expert weights | `L1` per expert | `Device`, `Host` (host-computed, spec 5 §3.4), `Tiered` (host-fetched) | expert |
| embedding table | `L0` | `Device`, `Host`, `Tiered` | row |
| n-gram tables | `L0` | `Host`, `Tiered`, `Device` (small) | row |
| vectors | `L0` | `Device` | tensor |

For `Tiered`, the on-disk region order is the residency unit order, so the loader can direct-IO a single expert or a single row range without touching neighbors. Host-resident copies keep the same layout as VRAM copies; promotion from host to device is a memcpy, never a repack.

## 6. Container

GGUF v3. Additions:

- **Tensor type IDs**: R9V schemes use type IDs in the range `1000–1099` in the tensor info's `type` field. Standard GGUF types keep their upstream IDs. A file whose every tensor has a standard ID and no `r9v.*` keys is a standard GGUF and loads through repack.
- **Alignment**: `general.alignment = 4096`. Tensor regions are 4 KiB aligned; scale regions follow their value region in the same tensor entry and are 256-byte aligned within it.
- **Region order inside a tensor entry**: `values` → `scales` → (if `L1S`) `indices`. Offsets of each are derivable from shape, scheme and layout; they are also written explicitly in metadata for loaders that don't want to compute them.
- **Tensor order in the file**: consumption order of the step graph (embedding, then layer 0's tensors in graph order, ..., lm_head). A cold load is one forward sweep of the file.

Metadata keys (`r9v.` prefix):

```
r9v.format_version           u32   = 1
r9v.layout_id                str   = "L1"
r9v.arch_hint                str   = "gfx1201"        # what the tool tuned for; informational
r9v.quant_tool.version       str
r9v.calibration.name         str
r9v.calibration.hash         str   # sha256 of the calibration set manifest
r9v.calibration.tokens       u64
r9v.smoothing.folded         bool
r9v.smoothing.alpha          f32
r9v.quant_tool.seed / .preset / .target   (spec 13 §10)
r9v.calibration.mix          str   # JSON of the domain mix
r9v.quality.*                (spec 13 §11: kl_mean, kl_p99, top1, top5, ppl, holdout_hash, engine_version)
r9v.tensor.<name>.scheme     str
r9v.tensor.<name>.act        str   # "i8/PerToken", "e4m3/PerToken", "f16/None"
r9v.tensor.<name>.roles      [str]
r9v.tensor.<name>.interleave str
r9v.tensor.<name>.sparse     str
r9v.tensor.<name>.placement_hint str
r9v.tensor.<name>.residency_unit str
r9v.tensor.<name>.regions    [u64] # byte offsets of values, scales, indices within the entry
r9v.tensor.<name>.xxh3       u64   # checksum of the entry
r9v.tensor.<name>.hot_hint   [f32] # stacked expert tensors only: routing frequency per expert (spec 13 §9)
r9v.tensor.<name>.eps_int4 / .eps_int8   f32   # spec 13 §6.1 sensitivity, informational
```

Standard GGUF metadata (`general.*`, `tokenizer.*`, `<arch>.*`) is preserved unchanged from the source model so the model definition (spec 8) reads hyperparameters from the same keys llama.cpp does.

## 7. Repack pipeline (loader, spec 9 owns execution; this section owns the rules)

For each tensor in a standard GGUF:

1. Map `ggml_type` → scheme via §3.3. Unknown type → hard error naming the type.
2. Validate shape constraints (`K % 256 == 0` for K-family schemes, `K % 32 == 0` for `_B32F` schemes). GGUF guarantees these already; the check is for corrupted files.
3. Choose layout: vectors → `L0`; `embed`/`ngram` roles → `L0`; everything else → `L1`.
4. Permute values into tile order; split scale fields into the SoA record region. This is a pure permutation of bytes plus, for 5/6-bit types, a bit-plane regrouping. No arithmetic on values.
5. If the model definition declares `gate_up` or `qkv` fusion for this tensor group, interleave the members' tiles.
6. Compute `xxh3` of the result and record it. The repack cache (spec 9 §5.3), keyed by `(file_fp, target layout, fusion declarations, scheme-mapping version)`, skips the work on subsequent loads. It lives beside the model file as `<file>.r9v-cache/` and is itself in this format.

Repack throughput target: ≥ 2 GB/s single-threaded on the CPU so a 25 GB file repacks in under 15 s the first time and never again.

For native files whose `layout_id` equals `arch.fragment_layout`: skip 3–5, direct-IO the regions into place.

## 8. Sizes

Bits per weight, including all scale overhead:

| scheme | bpw | note |
|---|---|---|
| `I8_R` | 8.0 | native |
| `I8_B128` | 8.125 | native |
| `I8_B32F` | 8.5 | Q8_0 repack |
| `I6_K` | 6.56 | Q6_K repack |
| `I5_K` | 5.5 | Q5_K repack |
| `I4_K` | 4.5 | native and Q4_K repack |
| `I4_XS` | 4.25 | IQ4_XS repack, int8 path |
| `E4M3_B128` | 8.125 | native, fp8 activations |
| any `s24` | ×0.5 + 0.5 bpw indices | |

Whole-model size is set by the tool's per-tensor mix (spec 13); typical native mixes land at 5.2 bpw (int8 attention, int4 elsewhere) and 6.2 bpw (int8 attention and down, int4 gate/up).

## 9. Versioning

- `r9v.format_version` bumps only on a change that makes old files unreadable. Adding a scheme, a role, or a flag with a default is a minor addition and does not bump it.
- Layout IDs are immutable. `L1` is never redefined; a new fragment order is `L2`.
- Scheme IDs are immutable; a changed scale record is a new scheme ID.
- The loader accepts any `format_version ≤ current` and any `layout_id` it knows a permutation for.

## 10. Validation

- On load: `xxh3` per tensor entry against the metadata value. Mismatch is a hard error.
- Debug mode: after repack, dequantize 64 random rows of each tensor through the CPU reference for both the source GGUF bytes and the repacked bytes and require bit-equality of the dequantized f32 values. This is the "repack never requantize" test and runs in CI on every scheme in §3.3.
- The doctor bundle records per tensor: source type, scheme, layout, whether zero-copy or permuted, act metadata, placement, and the reference-tier flag if the scheme has no fast path on this arch.
