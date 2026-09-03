# Spec 8 — Model Definition

Status: draft 0.1 (2026-09-02). Owner: Dylan. Depends on: specs 1, 2, 3, 7. Constrains: specs 9, 13.

## 0. Purpose and scope

How an architecture is expressed as a graph over the spec 1 op set: the builder API, the per-layer specification that covers the v1 model classes (dense, MoE, hybrid linear attention, MLA), how hyperparameters and tensor names are read from GGUF metadata, the fusion and structural declarations that drive the loader, the summary handed to the planner, and what it takes to add an architecture.

Out of scope: tokenization and chat templates (loaded from GGUF metadata by spec 9, used by spec 10), quantization decisions (spec 13), anything about kernels.

## 1. Principles

1. **A model definition is a pure function from metadata to a graph.** It calls the builder and nothing else: no device access, no kernels, no allocation, no I/O. The builder is the only API the `r9v-models` crate can see.
2. **Families, not files per model.** Most released models are a known family with a few switches. A family reads its switches from metadata; a new checkpoint of an existing family needs no code.
3. **Layers are data.** A definition produces a list of `LayerSpec`; a single generic layer builder turns each into ops. Unusual wiring uses the op-level escape hatch, which is visible in graph summaries and structural tests.
4. **Names follow llama.cpp.** Hyperparameter keys (`<arch>.*`) and tensor names (`blk.N.attn_q.weight`) are the GGUF conventions, so any GGUF that llama.cpp loads has the metadata this engine needs, and the native format inherits them.
5. **The definition declares structure; the loader acts on it.** QKV and gate/up fusion, tied embeddings, exported hidden states, and MTP heads are declarations here and mechanics in specs 2 and 9.
6. **A family is supported when it matches a reference.** Golden logits against a reference implementation on a fixed prompt set are the bar, run by the quant tool (spec 13) before a family enters the support matrix.

## 2. Builder API

```
Graph::new(ir_version, model_id) -> GraphBuilder

GraphBuilder:
  input_tokens() -> Tensor                         # [T] u32
  input_embed_override() -> (Tensor, Tensor)       # [T, Dm] act, [T] bool mask; multimodal escape hatch
  positions(kind: Scalar | MRope) -> Tensor
  weight(name, role, shape, expected: SchemeClass) -> Tensor      # binds a GGUF tensor by name
  state(layer, StateSpec) -> StateHandle
  op::<Op>(inputs, attrs) -> Tensor(s)             # every spec 1 op
  export(name, Tensor)                             # graph output (hidden for eagle/mtp, logprobs)
  declare_fusion(FusionDecl)                       # qkv | gate_up over named weights
  declare_tied(embed_name, head_name)
  subgraph(name) -> GraphBuilder                   # MTP head, eagle head, draft-model reuse
  finish() -> ModelGraph
```

`weight()` records the name, role (`matmul | embed | lm_head | ngram_table | vector`), expected logical shape and the class of scheme the definition can consume (all matmul weights accept any spec 2 scheme; vectors must be f32). The loader (spec 9) resolves names to file tensors, repacks, and fails with a list of every missing or mis-shaped tensor rather than the first one.

## 3. Layer specification

```
LayerSpec {
  norm:        NormPlacement            # Pre | Sandwich (pre + post on each sublayer, Gemma-style) | Parallel (GPT-J/Falcon: attn and ffn from the same input)
  norm_kind:   NormSpec { kind: Rms | Layer, eps, weight_offset }
  mixer:       Mixer
  ffn:         Ffn
  residual_scale: f32                   # 1.0 unless the family scales residuals
}

Mixer =
  Attention {
    h, hkv, d, dv,
    qkv_bias: bool, o_bias: bool,
    qk_norm: Option<NormSpec>,          # per-head norm on q and k after projection
    rope: RopeSpec,                     # theta, rot_dim, style, scaling, mrope_sections
    window: Option<u32>, sinks: u32,
    logit_softcap: Option<f32>,
    output_gate: bool,                  # Qwen3-Next style gated attention: o = attn ⊙ σ(W_g x)
    mla: Option<MlaSpec { q_lora_rank, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_dim }>,
    cache: CacheDtype,
  }
| LinearAttention {
    kind: GatedDeltaNet | GLA | Mamba2,
    h, d, dv,
    conv: Option<u32>,                  # causal conv width before the scan
    gate_act: Silu | Swish, output_norm: Option<NormSpec>, output_gate: bool,
  }
| None

Ffn =
  Dense { dff, act: Silu | Gelu | GeluTanh | Relu2, gated: bool, bias: bool }
| Moe   { e, k, dff_e, act, scoring: Softmax | Sigmoid, renormalize: bool,
          group: Option<{ n_group, topk_group }>, route_bias: bool, route_scale: f32,
          shared: Option<{ n, dff }>, shared_gate: bool }
| None
```

Model-level:

```
ModelSpec {
  dm, layers: Vec<LayerSpec>, vocab,
  embed_scale: f32, tied_embeddings: bool,
  final_norm: NormSpec, final_logit_softcap: Option<f32>,
  positions: Scalar | MRope([u32; 3]),
  ngram: Option<NgramSpec { orders, heads, table_sizes, hash, combine, inject_at: layer }>,
  mtp: Option<MtpSpec { heads: u32, layers_per_head: Vec<LayerSpec>, takes_hidden_from: Last | Layer(n) }>,
  export_hidden: bool,                  # set by the loader when the proposer needs the hidden state (eagle, block drafters); mtp consumes it in-graph
  eos_ids: Vec<u32>, bos_id: Option<u32>,
}
```

### 3.1 What the generic layer builder emits

For `norm = Pre`, `mixer = Attention`, `ffn = Dense{gated}`:

```
h  = norm(x)
q,k,v = matmul(h, W_qkv)                       # one fused matmul if declare_fusion(qkv)
q,k = qk_norm?(q,k); q,k = rope(q,k)
state_write_kv(k, v, slot_map, S_l)
a  = attention(q, S_l)
a  = a ⊙ σ(matmul(h, W_g))  if output_gate
x  = residual_add(x, matmul(a, W_o))            # Partial under TP, resolved by partitioner
h  = norm(x)
g,u = matmul(h, W_gate_up)                      # fused if declared
x  = residual_add(x, matmul(act_mul(g,u), W_down))
```

`Sandwich` adds `norm` after each sublayer output before the residual. `Parallel` computes both sublayers from the same `norm(x)` and adds both. `LinearAttention` replaces the attention block with `causal_conv1d? → linear_attn_scan → output_norm? → gate`. `Moe` replaces the dense FFN with `moe_route → moe_ffn (+ shared matmul path)`. `mla` changes the q/k/v projections to the low-rank form and the state kind to `KvLatent`. `ngram` inserts `ngram_gather → matmul (projection) → residual_add` at `inject_at`.

The builder is the only place these patterns live. A family never writes the pattern out by hand.

## 4. Families

Each family is a function `fn build(meta: &GgufMeta) -> Result<ModelSpec>` registered under one or more `general.architecture` values. v1 families:

| family | `general.architecture` | switches read from metadata |
|---|---|---|
| `llama` | `llama`, `mistral`, `qwen2`, `qwen3`, `gemma2`, `gemma3`, `phi3`, `olmo2`, ... | `hkv`, rope scaling, qk_norm, window pattern, sandwich norm, softcaps, tied embeddings, embed scale, activation |
| `moe` | `mixtral`, `qwen2moe`, `qwen3moe`, `deepseek2`, `deepseek3`, `glm4moe`, ... | `e`, `k`, scoring, grouping, route bias, shared experts, `mla` (deepseek) |
| `hybrid` | `qwen3next`, `qwen38next`, `granitehybrid`, `falconh1`, `nemotronh`, ... | layer pattern (which layers are linear vs full attention), scan kind, conv width, output gating, MoE per layer |
| `mtp-head` | (partial; loaded alongside a family) | heads, layers per head |
| `eagle-head` | (partial) | layers, feature fusion |

A model whose `general.architecture` is unknown fails at load with the string and the nearest family's name. Adding a value to a family's list is a one-line change when the metadata keys already match.

## 5. Weight binding and declarations

- `weight(name, ...)` names use the GGUF convention verbatim: `token_embd.weight`, `blk.N.attn_q.weight`, `blk.N.ffn_gate_exps.weight` (stacked experts), `blk.N.ssm_*` for scan layers, `output.weight`, `output_norm.weight`.
- **Fusion declarations** are per layer: `declare_fusion(Qkv { q, k, v })`, `declare_fusion(GateUp { gate, up })`. The loader interleaves on repack (spec 2 §4) and the builder emits one `matmul` with a split view. Native files carry the interleave already; the declaration must match or load fails.
- **Tied embeddings**: `declare_tied("token_embd.weight", "output.weight")` when `output.weight` is absent and the family says tied. Storage per spec 2 §4.
- **Stacked experts** (`ffn_*_exps` as one `[E, ...]` tensor) are the residency unit `expert` in spec 2 §5; the definition passes the expert dimension as outermost.
- **MTP weights** (`blk.N.mtp.*` or the family's naming) bind inside `subgraph("mtp")`; absent MTP weights make `mtp = None` and the proposer resolution (spec 7 §6) skips it.

## 6. Validation

At load, before any repack:

1. Every `weight()` call resolves to a tensor with the expected logical shape (after accounting for spec 2 padding rules). All failures are collected and reported together.
2. Tensors in the file that no `weight()` call named are logged as unused (warning, not error), so extra heads or vision towers don't block loading.
3. Structural constraints: `d % 16 == 0` (or the attention layout for that `d` exists in the registry), `K % 256 == 0` for K-family schemes, `hkv % tp_degree == 0` or replication permitted (spec 5 §3.2), `dff_e` consistent across experts, `vocab` matches the tokenizer.
4. The `ModelSpec` is checked against the IR version it pins; a mismatch is an error naming both versions.

## 7. ModelSummary (for the planner, spec 5)

```
ModelSummary {
  layers: Vec<{ weight_bytes_by_scheme, state_per_token_bytes, state_per_seq_bytes,
                experts: Option<{ e, bytes_each, hot_hint: Vec<f32> }>, mixer_kind }>
  embed_bytes, head_bytes, vocab, dm, hkv, tp_divisors: Vec<u32>,
  ngram_table_bytes: u64, mtp: bool, export_hidden: bool,
}
```

Computed by the builder from the bound weights and `StateSpec`s; `hot_hint` (expert activation frequency) comes from spec 2 metadata when the quant tool recorded it, else uniform.

## 8. Testing a family

- **Reference match**: the quant tool (spec 13 §12) runs the family through the engine's CPU tier (T0v, with T0 for ops that lack a T0v) on an F16 GGUF and compares logits to a reference implementation on a fixed 64-prompt set. Bar: L1 tolerance on logits (spec 1 §6.1), top-1 agreement ≥ 99.9%. Required to enter the support matrix.
- **Structural**: every `LayerSpec` variant the family can emit is built once with synthetic metadata and captured against the registry on the CI runner, so an unused switch can't rot.
- **Golden prompts**: 8 prompts per supported checkpoint with recorded greedy outputs at L0 on the reference machine; CI diffs them.

## 9. Adding an architecture

1. Read the checkpoint's GGUF metadata; determine whether an existing family covers it with switches. If yes, add the `general.architecture` string and any new metadata key reads. Done.
2. If a new `LayerSpec` field is needed (a new gate, norm placement, routing rule), add it to §3 and to the generic layer builder; it is available to every family.
3. If the model needs an op the IR lacks, stop: spec 1 §7 RFC first. The definition PR waits for the op to land with its reference tier.
4. Write the family function, bind weights, declare fusions, run §8. Add the checkpoint to the support matrix with its reference-match numbers.

The expected shape of a new-model PR is a few hundred lines in `r9v-models` and no lines anywhere else. A PR that touches kernels, state, scheduler or partitioner to support a model is the signal that a spec has a gap.

## 10. Example: a hybrid family (abridged)

```rust
fn qwen38next(meta: &GgufMeta) -> Result<ModelSpec> {
    let dm = meta.u32("qwen38next.embedding_length")?;
    let n  = meta.u32("qwen38next.block_count")?;
    let full_every = meta.u32("qwen38next.full_attention_interval")?;   // e.g. 4
    let layers = (0..n).map(|i| {
        let mixer = if (i + 1) % full_every == 0 {
            Mixer::Attention { h: meta.u32("...attention.head_count")?, hkv: ..., d: ..., dv: ...,
                               rope: RopeSpec::from(meta)?, qk_norm: Some(rms(meta)), output_gate: true, .. }
        } else {
            Mixer::LinearAttention { kind: GatedDeltaNet, h: meta.u32("...ssm.head_count")?, d: ..., dv: ...,
                                     conv: Some(meta.u32("...ssm.conv_kernel")?), output_norm: Some(rms(meta)), output_gate: true, .. }
        };
        Ok(LayerSpec { norm: Pre, norm_kind: rms(meta), mixer,
                       ffn: Ffn::Moe { e: ..., k: ..., dff_e: ..., act: Silu, scoring: Softmax, renormalize: true,
                                       shared: Some({ n: 1, dff: ... }), shared_gate: true, .. },
                       residual_scale: 1.0 })
    }).collect::<Result<_>>()?;
    Ok(ModelSpec { dm, layers, vocab: ..., tied_embeddings: false, final_norm: rms(meta),
                   mtp: meta.has("blk.0.mtp") .then(|| MtpSpec { heads: 1, .. }), positions: Scalar, .. })
}
```

Everything else (fusions, state handles, MTP subgraph, exports) is emitted by the generic builder from this spec.
