// SPDX-License-Identifier: Apache-2.0
//! Generic transformer layer and model builder (Spec 8 §3, §3.1, §5; card A1.3).
//!
//! Emits the exact sequence of Op IR operations specified in Spec 8 §3.1 for every
//! norm placement × mixer × ffn combination, including `LinearAttention`, `Moe`,
//! `mla`, `output_gate`, `ngram` injection and the `MTP` subgraph.

use r9v_ir::op::{
    ActivationKind, AttentionMask, ConvActivation, MlaAttentionSpec, MlaLatent, MoeGroup,
    NgramCombine, NgramSource, NormAxis,
};
use r9v_ir::tensor::{Dim, ShapeSymbol};
use r9v_ir::DType;

use crate::builder::{
    checked_add, checked_mul, checked_u32, FusionDecl, GraphBuilder, ModelGraph, SchemeClass,
    Value, WeightRole,
};
use crate::error::ModelsError;
use crate::spec::{
    Ffn, LayerSpec, Mixer, ModelSpec, MtpSource, MtpSpec, NormPlacement, Retain, RopeSpec,
    StateSpec,
};

/// Lowers a full [`ModelSpec`] into an Op IR step graph (Spec 8 §2, §3, §5).
pub fn build_model(
    mut builder: GraphBuilder,
    model: &ModelSpec,
) -> Result<ModelGraph, ModelsError> {
    model.validate()?;

    // 1. External inputs
    let tokens = builder.input_tokens()?;

    // 2. Token embedding table lookup
    let w_embed = builder.weight(
        "token_embd.weight",
        WeightRole::Embed,
        &[Dim::Concrete(model.vocab), Dim::Concrete(model.dm)],
        SchemeClass::Embed,
    )?;
    let mut x = builder.op_embed_gather(tokens.clone(), w_embed, model.embed_scale)?;

    // 3. Build each layer sequentially
    let mut captured_hidden_at_layer: Option<Value> = None;

    for (i, layer_spec) in model.layers.iter().enumerate() {
        let layer_idx = checked_u32(i, "build_model layer index")?;

        // Check for n-gram speculative feature injection at inject_at
        if let Some(ngram) = &model.ngram {
            if ngram.inject_at == layer_idx {
                let mut total_entries: u32 = 0;
                for entries in ngram.table_sizes.iter() {
                    total_entries =
                        checked_add(total_entries, *entries, "build_model ngram tables")?;
                }
                // Table rows are `Dn` wide per Spec 1 §4.A; the explicit
                // `NgramSpec::dim` carries it under the A1.3 n-gram-dimension decision.
                let ngram_dim = ngram.dim;
                let ngram_table = builder.weight(
                    format!("blk.{layer_idx}.ngram_table.weight"),
                    WeightRole::NgramTable,
                    &[
                        Dim::Concrete(total_entries.max(1)),
                        Dim::Concrete(ngram_dim),
                    ],
                    SchemeClass::NgramTable,
                )?;
                let out_dim = match ngram.combine {
                    NgramCombine::Concat => {
                        checked_mul(ngram.heads, ngram_dim, "build_model ngram out_dim")?
                    }
                    NgramCombine::Sum => ngram_dim,
                };
                let ngram_out = builder.op_ngram_gather(
                    NgramSource::Device,
                    &[tokens.clone(), ngram_table],
                    &ngram.orders,
                    ngram.heads,
                    ngram.hash,
                    &ngram.table_sizes,
                    ngram.combine,
                    out_dim,
                    DType::F16,
                )?;
                let w_proj = builder.weight(
                    format!("blk.{layer_idx}.ngram_proj.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(model.dm), Dim::Concrete(out_dim)],
                    SchemeClass::Matmul,
                )?;
                let proj = builder.op_matmul(ngram_out, w_proj, DType::F16)?;
                // DECISION(A1.14): the n-gram injection residual stays
                // unscaled: LayerSpec.residual_scale governs the layer's own
                // mixer/FFN residuals (Spec 8 §3.1), while this model-level
                // injection shows a plain residual_add there. SI-18.
                x = builder.op_residual_add(x, proj, DType::F16)?;
            }
        }

        // Build transformer block
        x = build_layer(&mut builder, layer_idx, layer_spec, x, model)?;

        // Check for MTP hidden state tap
        if let Some(mtp) = &model.mtp {
            if mtp.takes_hidden_from == MtpSource::Layer(layer_idx) {
                captured_hidden_at_layer = Some(x.clone());
            }
        }
    }

    // 4. Final normalization
    let w_final_norm = builder.weight(
        "output_norm.weight",
        WeightRole::Vector,
        &[Dim::Concrete(model.dm)],
        SchemeClass::Vector,
    )?;
    let h_final = builder.op_norm(
        x,
        w_final_norm,
        model.final_norm,
        NormAxis::Last,
        DType::F16,
    )?;

    // 5. Export pre-lm_head final hidden states if requested
    if model.export_hidden {
        builder.export("hidden", h_final.clone())?;
    }

    // 6. Language model output head projection
    let w_head = if model.tied_embeddings {
        builder.declare_tied("token_embd.weight", "output.weight")?;
        builder.weight(
            "output.weight",
            WeightRole::LmHead,
            &[Dim::Concrete(model.vocab), Dim::Concrete(model.dm)],
            SchemeClass::Embed,
        )?
    } else {
        builder.weight(
            "output.weight",
            WeightRole::LmHead,
            &[Dim::Concrete(model.vocab), Dim::Concrete(model.dm)],
            SchemeClass::Matmul,
        )?
    };

    let lm_logits = builder.op_matmul(h_final.clone(), w_head, DType::F32)?;
    // A set final_logit_softcap lowers to one logit_softcap op; None emits
    // nothing, reproducing the A1.3 graph exactly (Spec 8 §3; card A1.14,
    // SI-19).
    let logits = match model.final_logit_softcap {
        Some(cap) => builder.op_logit_softcap(lm_logits, cap)?,
        None => lm_logits,
    };
    builder.export("logits", logits)?;

    // 7. Multi-Token Prediction (MTP) subgraph
    if let Some(mtp) = &model.mtp {
        // DECISION(A1.3): MTP heads emit dedicated subgraphs named 'mtp' or 'mtp.<head>' recording head index and consuming the specified hidden state tensor; rejected flattening MTP into the primary layer sequence. Spec 8 §2, §5.
        let mtp_hidden_in = match mtp.takes_hidden_from {
            MtpSource::Last => h_final,
            MtpSource::Layer(_) => captured_hidden_at_layer.unwrap_or(h_final),
        };
        build_mtp_subgraph(&mut builder, mtp, mtp_hidden_in, model)?;
    }

    builder.finish()
}

/// Emits the Op IR nodes for a single transformer layer block (Spec 8 §3.1).
///
/// Spec-facing entry point: base-model weights live directly under `blk.N.`.
/// MTP subgraph layers use the private namespace helper instead.
pub fn build_layer(
    builder: &mut GraphBuilder,
    layer_idx: u32,
    spec: &LayerSpec,
    x: Value,
    model: &ModelSpec,
) -> Result<Value, ModelsError> {
    build_layer_with_ns(builder, layer_idx, spec, x, model, "")
}

/// Emits one transformer layer block with `weight_ns` inserted after `blk.N.`
/// (`""` for base layers, `"mtp."` for MTP subgraph layers per Spec 8 §5).
fn build_layer_with_ns(
    builder: &mut GraphBuilder,
    layer_idx: u32,
    spec: &LayerSpec,
    mut x: Value,
    model: &ModelSpec,
    weight_ns: &str,
) -> Result<Value, ModelsError> {
    spec.validate(layer_idx)?;

    match spec.norm {
        NormPlacement::Pre => {
            // Pre-norm: norm(x) -> sublayer -> residual_add
            if spec.mixer != Mixer::None {
                let w_attn_norm = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_norm.weight"),
                    WeightRole::Vector,
                    &[Dim::Concrete(model.dm)],
                    SchemeClass::Vector,
                )?;
                let h = builder.op_norm(
                    x.clone(),
                    w_attn_norm,
                    spec.norm_kind,
                    NormAxis::Last,
                    DType::F16,
                )?;
                let mixer_out =
                    build_mixer_with_ns(builder, layer_idx, &spec.mixer, h, model, weight_ns)?;
                x = builder.op_residual_add_scaled(
                    x,
                    mixer_out,
                    spec.residual_scale,
                    DType::F16,
                )?;
            }

            if spec.ffn != Ffn::None {
                let w_ffn_norm = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_norm.weight"),
                    WeightRole::Vector,
                    &[Dim::Concrete(model.dm)],
                    SchemeClass::Vector,
                )?;
                let h = builder.op_norm(
                    x.clone(),
                    w_ffn_norm,
                    spec.norm_kind,
                    NormAxis::Last,
                    DType::F16,
                )?;
                let ffn_out =
                    build_ffn_with_ns(builder, layer_idx, &spec.ffn, h, model, weight_ns)?;
                x = builder.op_residual_add_scaled(x, ffn_out, spec.residual_scale, DType::F16)?;
            }
        }
        NormPlacement::Sandwich => {
            // Sandwich-norm: pre-norm -> sublayer -> post-norm -> residual_add (Gemma-style)
            if spec.mixer != Mixer::None {
                let w_pre = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_norm.weight"),
                    WeightRole::Vector,
                    &[Dim::Concrete(model.dm)],
                    SchemeClass::Vector,
                )?;
                let h = builder.op_norm(
                    x.clone(),
                    w_pre,
                    spec.norm_kind,
                    NormAxis::Last,
                    DType::F16,
                )?;
                let mixer_out =
                    build_mixer_with_ns(builder, layer_idx, &spec.mixer, h, model, weight_ns)?;
                let w_post = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}post_attn_norm.weight"),
                    WeightRole::Vector,
                    &[Dim::Concrete(model.dm)],
                    SchemeClass::Vector,
                )?;
                let post = builder.op_norm(
                    mixer_out,
                    w_post,
                    spec.norm_kind,
                    NormAxis::Last,
                    DType::F16,
                )?;
                x = builder.op_residual_add_scaled(x, post, spec.residual_scale, DType::F16)?;
            }

            if spec.ffn != Ffn::None {
                let w_pre = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_norm.weight"),
                    WeightRole::Vector,
                    &[Dim::Concrete(model.dm)],
                    SchemeClass::Vector,
                )?;
                let h = builder.op_norm(
                    x.clone(),
                    w_pre,
                    spec.norm_kind,
                    NormAxis::Last,
                    DType::F16,
                )?;
                let ffn_out =
                    build_ffn_with_ns(builder, layer_idx, &spec.ffn, h, model, weight_ns)?;
                let w_post = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}post_ffn_norm.weight"),
                    WeightRole::Vector,
                    &[Dim::Concrete(model.dm)],
                    SchemeClass::Vector,
                )?;
                let post =
                    builder.op_norm(ffn_out, w_post, spec.norm_kind, NormAxis::Last, DType::F16)?;
                x = builder.op_residual_add_scaled(x, post, spec.residual_scale, DType::F16)?;
            }
        }
        NormPlacement::Parallel => {
            // Parallel: norm(x) once -> attention + ffn from same input -> add both
            let w_norm = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}attn_norm.weight"),
                WeightRole::Vector,
                &[Dim::Concrete(model.dm)],
                SchemeClass::Vector,
            )?;
            let h = builder.op_norm(
                x.clone(),
                w_norm,
                spec.norm_kind,
                NormAxis::Last,
                DType::F16,
            )?;

            let mixer_out = if spec.mixer != Mixer::None {
                Some(build_mixer_with_ns(
                    builder,
                    layer_idx,
                    &spec.mixer,
                    h.clone(),
                    model,
                    weight_ns,
                )?)
            } else {
                None
            };

            let ffn_out = if spec.ffn != Ffn::None {
                Some(build_ffn_with_ns(
                    builder, layer_idx, &spec.ffn, h, model, weight_ns,
                )?)
            } else {
                None
            };

            if let Some(m_out) = mixer_out {
                x = builder.op_residual_add_scaled(x, m_out, spec.residual_scale, DType::F16)?;
            }
            if let Some(f_out) = ffn_out {
                x = builder.op_residual_add_scaled(x, f_out, spec.residual_scale, DType::F16)?;
            }
        }
    }

    Ok(x)
}

/// Emits the token mixing sublayer (Attention, LinearAttention, MLA; Spec 8 §3.1).
///
/// Spec-facing entry point: base-model weights live directly under `blk.N.`.
pub fn build_mixer(
    builder: &mut GraphBuilder,
    layer_idx: u32,
    mixer: &Mixer,
    h: Value,
    model: &ModelSpec,
) -> Result<Value, ModelsError> {
    build_mixer_with_ns(builder, layer_idx, mixer, h, model, "")
}

/// Emits one token mixing sublayer with `weight_ns` inserted after `blk.N.`.
fn build_mixer_with_ns(
    builder: &mut GraphBuilder,
    layer_idx: u32,
    mixer: &Mixer,
    h: Value,
    model: &ModelSpec,
    weight_ns: &str,
) -> Result<Value, ModelsError> {
    match mixer {
        Mixer::Attention {
            h: heads,
            hkv,
            d,
            dv,
            qkv_bias,
            o_bias,
            qk_norm,
            rope,
            window,
            sinks,
            logit_softcap,
            output_gate,
            mla,
            cache,
            ..
        } => {
            let h_heads = *heads;
            let hkv_heads = *hkv;
            let head_d = *d;
            let head_dv = *dv;

            if let Some(mla_spec) = mla {
                // Multi-Head Latent Attention (MLA, DeepSeek-style; card
                // A1.14, SI-20). Compressed latents and decoupled rotary
                // parts travel as explicit edges: the query splits into
                // (q_nope, q_rope) with rope applied to the rotary part only,
                // and the attention query is reconstructed by concatenation
                // with the declared head dims. No compressed latent channel
                // passes through RoPE.
                let w_q_a = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_q_a.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(mla_spec.q_lora_rank), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let c_q = if *qkv_bias {
                    let b_q_a = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_q_a.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(mla_spec.q_lora_rank)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(h.clone(), w_q_a, b_q_a, DType::F16)?
                } else {
                    builder.op_matmul(h.clone(), w_q_a, DType::F16)?
                };

                let qk_sum = checked_add(
                    mla_spec.qk_nope_dim,
                    mla_spec.qk_rope_dim,
                    "mla query head dim",
                )?;
                let q_flat_dim = checked_mul(h_heads, qk_sum, "mla query projection")?;
                let w_q_b = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_q_b.weight"),
                    WeightRole::Matmul,
                    &[
                        Dim::Concrete(q_flat_dim),
                        Dim::Concrete(mla_spec.q_lora_rank),
                    ],
                    SchemeClass::Matmul,
                )?;
                let mut q_flat = if *qkv_bias {
                    let b_q_b = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_q_b.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(q_flat_dim)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(c_q, w_q_b, b_q_b, DType::F16)?
                } else {
                    builder.op_matmul(c_q, w_q_b, DType::F16)?
                };

                let kv_in_dim = checked_add(
                    mla_spec.kv_lora_rank,
                    mla_spec.qk_rope_dim,
                    "mla kv input dim",
                )?;
                let w_kv_a = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_kv_a.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(kv_in_dim), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let mut c_kv_flat = if *qkv_bias {
                    let b_kv_a = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_kv_a.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(kv_in_dim)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(h.clone(), w_kv_a, b_kv_a, DType::F16)?
                } else {
                    builder.op_matmul(h.clone(), w_kv_a, DType::F16)?
                };

                // qk_norm lowers exactly like the standard path — after
                // projection, before rope, one weight per side — with the
                // head axis over the combined query width and a row norm over
                // the head-less KV rows, which have no per-head structure
                // (card A1.14, SI-20).
                if let Some(norm) = qk_norm {
                    let w_q_norm = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_q_norm.weight"),
                        WeightRole::Vector,
                        &[Dim::Concrete(q_flat_dim)],
                        SchemeClass::Vector,
                    )?;
                    q_flat = builder.op_norm(
                        q_flat,
                        w_q_norm,
                        *norm,
                        NormAxis::Head(qk_sum),
                        DType::F16,
                    )?;
                    let w_kv_norm = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_k_norm.weight"),
                        WeightRole::Vector,
                        &[Dim::Concrete(kv_in_dim)],
                        SchemeClass::Vector,
                    )?;
                    c_kv_flat =
                        builder.op_norm(c_kv_flat, w_kv_norm, *norm, NormAxis::Last, DType::F16)?;
                }

                let q_3d = builder.op_reshape(
                    q_flat,
                    vec![
                        Dim::Symbolic(ShapeSymbol::T),
                        Dim::Concrete(h_heads),
                        Dim::Concrete(qk_sum),
                    ],
                )?;
                let (q_nope, q_rope_raw) = builder.op_split(q_3d, mla_spec.qk_nope_dim)?;

                // DECISION(A1.14): the decoupled rotary parts rotate with
                // rot_dim set to the rotary width itself: every channel of
                // q_rope/k_rope is positional by construction, while the
                // standard rot_dim names the full-path width and exceeds the
                // rope part whenever they differ. Rejected reusing
                // rope.rot_dim verbatim (validation must reject rot_dim > D,
                // never silently clamp). Spec 8 §3.1, SI-20.
                let mla_rope = RopeSpec {
                    rot_dim: mla_spec.qk_rope_dim,
                    ..rope.clone()
                };
                let pos = builder.positions(model.positions)?;
                let q_rope = builder.op_rope(q_rope_raw, pos.clone(), &mla_rope)?;
                let q = builder.op_concat(q_nope, q_rope)?;

                let c_kv_3d = builder.op_reshape(
                    c_kv_flat,
                    vec![
                        Dim::Symbolic(ShapeSymbol::T),
                        Dim::Concrete(1),
                        Dim::Concrete(kv_in_dim),
                    ],
                )?;
                let (c_kv, k_rope_raw) = builder.op_split(c_kv_3d, mla_spec.kv_lora_rank)?;
                let k_rope = builder.op_rope(k_rope_raw, pos, &mla_rope)?;

                // Validated at the `LayerSpec` boundary; this `?` is the second
                // line of defense for direct `build_mixer` callers.
                let retain = Retain::from_window_sinks(*window, *sinks)?;
                let handle = builder.state(
                    layer_idx,
                    StateSpec::KvLatent {
                        latent: mla_spec.kv_lora_rank,
                        rope: mla_spec.qk_rope_dim,
                        cache: *cache,
                        retain,
                    },
                )?;

                let latent_info = Some(MlaLatent {
                    kv_lora_rank: mla_spec.kv_lora_rank,
                    rope_dim: mla_spec.qk_rope_dim,
                });
                builder.op_state_write_kv(k_rope, c_kv, handle, *cache, latent_info)?;

                let mask = if let Some(w) = window {
                    AttentionMask::CausalWindow(*w)
                } else {
                    AttentionMask::Causal
                };
                let softmax_scale = 1.0 / (qk_sum as f32).sqrt();
                let mla_attention = Some(MlaAttentionSpec {
                    q_lora_rank: Some(mla_spec.q_lora_rank),
                    kv_lora_rank: mla_spec.kv_lora_rank,
                    qk_nope_dim: mla_spec.qk_nope_dim,
                    qk_rope_dim: mla_spec.qk_rope_dim,
                    v_dim: mla_spec.v_dim,
                });

                let a_3d = builder.op_attention(
                    q,
                    handle,
                    softmax_scale,
                    mask,
                    *sinks,
                    *logit_softcap,
                    mla_attention,
                    DType::F16,
                )?;

                let out_flat_dim = checked_mul(h_heads, mla_spec.v_dim, "mla output projection")?;
                let mut a = builder.op_reshape(
                    a_3d,
                    vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(out_flat_dim)],
                )?;

                if *output_gate {
                    let w_g = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_gate.weight"),
                        WeightRole::Matmul,
                        &[Dim::Concrete(out_flat_dim), Dim::Concrete(model.dm)],
                        SchemeClass::Matmul,
                    )?;
                    let gate = builder.op_matmul(h, w_g, DType::F16)?;
                    a = builder.op_act_mul(gate, a, ActivationKind::Silu)?;
                }

                let w_o = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_output.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(model.dm), Dim::Concrete(out_flat_dim)],
                    SchemeClass::Matmul,
                )?;
                if *o_bias {
                    let b_o = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_output.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(model.dm)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(a, w_o, b_o, DType::F16)
                } else {
                    builder.op_matmul(a, w_o, DType::F16)
                }
            } else {
                // Standard Attention: MHA / GQA / MQA
                builder.declare_fusion(FusionDecl::Qkv {
                    q: format!("blk.{layer_idx}.{weight_ns}attn_q.weight"),
                    k: format!("blk.{layer_idx}.{weight_ns}attn_k.weight"),
                    v: format!("blk.{layer_idx}.{weight_ns}attn_v.weight"),
                })?;

                let q_dim = checked_mul(h_heads, head_d, "attention q projection")?;
                let k_dim = checked_mul(hkv_heads, head_d, "attention k projection")?;
                let v_flat_dim = checked_mul(hkv_heads, head_dv, "attention v projection")?;
                let o_flat_dim = checked_mul(h_heads, head_dv, "attention output")?;
                let w_q = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_q.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(q_dim), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let mut q_flat = if *qkv_bias {
                    let b_q = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_q.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(q_dim)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(h.clone(), w_q, b_q, DType::F16)?
                } else {
                    builder.op_matmul(h.clone(), w_q, DType::F16)?
                };

                let w_k = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_k.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(k_dim), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let mut k_flat = if *qkv_bias {
                    let b_k = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_k.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(k_dim)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(h.clone(), w_k, b_k, DType::F16)?
                } else {
                    builder.op_matmul(h.clone(), w_k, DType::F16)?
                };

                let w_v = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_v.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(v_flat_dim), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let v_flat = if *qkv_bias {
                    let b_v = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_v.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(v_flat_dim)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(h.clone(), w_v, b_v, DType::F16)?
                } else {
                    builder.op_matmul(h.clone(), w_v, DType::F16)?
                };

                if let Some(norm) = qk_norm {
                    let w_q_norm = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_q_norm.weight"),
                        WeightRole::Vector,
                        &[Dim::Concrete(q_dim)],
                        SchemeClass::Vector,
                    )?;
                    let w_k_norm = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_k_norm.weight"),
                        WeightRole::Vector,
                        &[Dim::Concrete(k_dim)],
                        SchemeClass::Vector,
                    )?;
                    q_flat = builder.op_norm(
                        q_flat,
                        w_q_norm,
                        *norm,
                        NormAxis::Head(head_d),
                        DType::F16,
                    )?;
                    k_flat = builder.op_norm(
                        k_flat,
                        w_k_norm,
                        *norm,
                        NormAxis::Head(head_d),
                        DType::F16,
                    )?;
                }

                let mut q = builder.op_reshape(
                    q_flat,
                    vec![
                        Dim::Symbolic(ShapeSymbol::T),
                        Dim::Concrete(h_heads),
                        Dim::Concrete(head_d),
                    ],
                )?;

                let mut k = builder.op_reshape(
                    k_flat,
                    vec![
                        Dim::Symbolic(ShapeSymbol::T),
                        Dim::Concrete(hkv_heads),
                        Dim::Concrete(head_d),
                    ],
                )?;

                let v = builder.op_reshape(
                    v_flat,
                    vec![
                        Dim::Symbolic(ShapeSymbol::T),
                        Dim::Concrete(hkv_heads),
                        Dim::Concrete(head_dv),
                    ],
                )?;

                let pos = builder.positions(model.positions)?;
                q = builder.op_rope(q, pos.clone(), rope)?;
                k = builder.op_rope(k, pos, rope)?;

                // Validated at the `LayerSpec` boundary; this `?` is the second
                // line of defense for direct `build_mixer` callers.
                let retain = Retain::from_window_sinks(*window, *sinks)?;
                let handle = builder.state(
                    layer_idx,
                    StateSpec::KvPaged {
                        hkv: hkv_heads,
                        d: head_d,
                        dv: head_dv,
                        cache: *cache,
                        retain,
                    },
                )?;

                builder.op_state_write_kv(k, v, handle, *cache, None)?;

                let mask = if let Some(w) = window {
                    AttentionMask::CausalWindow(*w)
                } else {
                    AttentionMask::Causal
                };
                let softmax_scale = 1.0 / (head_d as f32).sqrt();
                let a_3d = builder.op_attention(
                    q,
                    handle,
                    softmax_scale,
                    mask,
                    *sinks,
                    *logit_softcap,
                    None,
                    DType::F16,
                )?;

                let mut a = builder.op_reshape(
                    a_3d,
                    vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(o_flat_dim)],
                )?;

                if *output_gate {
                    let w_g = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_gate.weight"),
                        WeightRole::Matmul,
                        &[Dim::Concrete(o_flat_dim), Dim::Concrete(model.dm)],
                        SchemeClass::Matmul,
                    )?;
                    let gate = builder.op_matmul(h, w_g, DType::F16)?;
                    a = builder.op_act_mul(gate, a, ActivationKind::Silu)?;
                }

                let w_o = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}attn_output.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(model.dm), Dim::Concrete(o_flat_dim)],
                    SchemeClass::Matmul,
                )?;
                if *o_bias {
                    let b_o = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}attn_output.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(model.dm)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(a, w_o, b_o, DType::F16)
                } else {
                    builder.op_matmul(a, w_o, DType::F16)
                }
            }
        }
        Mixer::LinearAttention {
            kind,
            h: heads,
            d,
            dv,
            conv,
            gate_act,
            output_norm,
            output_gate,
        } => {
            let h_heads = *heads;
            let head_d = *d;
            let head_dv = *dv;
            let mut scan_input = h.clone();

            if let Some(kernel_len) = conv {
                let handle_conv = builder.state(
                    layer_idx,
                    StateSpec::ConvWindow {
                        c: model.dm,
                        w: *kernel_len,
                    },
                )?;
                let w_conv = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ssm_conv1d.weight"),
                    WeightRole::Vector,
                    &[Dim::Concrete(model.dm), Dim::Concrete(*kernel_len)],
                    SchemeClass::Vector,
                )?;
                scan_input = builder.op_causal_conv1d(
                    scan_input,
                    w_conv,
                    None,
                    *kernel_len,
                    ConvActivation::Silu,
                    handle_conv,
                )?;
            }

            let scan_qk_dim = checked_mul(h_heads, head_d, "linear attention q/k")?;
            let scan_v_dim = checked_mul(h_heads, head_dv, "linear attention v")?;
            let w_q = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ssm_q.weight"),
                WeightRole::Matmul,
                &[Dim::Concrete(scan_qk_dim), Dim::Concrete(model.dm)],
                SchemeClass::Matmul,
            )?;
            let q_flat = builder.op_matmul(scan_input.clone(), w_q, DType::F16)?;
            let q = builder.op_reshape(
                q_flat,
                vec![
                    Dim::Symbolic(ShapeSymbol::T),
                    Dim::Concrete(h_heads),
                    Dim::Concrete(head_d),
                ],
            )?;

            let w_k = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ssm_k.weight"),
                WeightRole::Matmul,
                &[Dim::Concrete(scan_qk_dim), Dim::Concrete(model.dm)],
                SchemeClass::Matmul,
            )?;
            let k_flat = builder.op_matmul(scan_input.clone(), w_k, DType::F16)?;
            let k = builder.op_reshape(
                k_flat,
                vec![
                    Dim::Symbolic(ShapeSymbol::T),
                    Dim::Concrete(h_heads),
                    Dim::Concrete(head_d),
                ],
            )?;

            let w_v = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ssm_v.weight"),
                WeightRole::Matmul,
                &[Dim::Concrete(scan_v_dim), Dim::Concrete(model.dm)],
                SchemeClass::Matmul,
            )?;
            let v_flat = builder.op_matmul(scan_input.clone(), w_v, DType::F16)?;
            let v = builder.op_reshape(
                v_flat,
                vec![
                    Dim::Symbolic(ShapeSymbol::T),
                    Dim::Concrete(h_heads),
                    Dim::Concrete(head_dv),
                ],
            )?;

            let w_alpha = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ssm_alpha.weight"),
                WeightRole::Matmul,
                &[Dim::Concrete(h_heads), Dim::Concrete(model.dm)],
                SchemeClass::Matmul,
            )?;
            let alpha = builder.op_matmul(scan_input.clone(), w_alpha, DType::F32)?;

            let w_beta = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ssm_beta.weight"),
                WeightRole::Matmul,
                &[Dim::Concrete(h_heads), Dim::Concrete(model.dm)],
                SchemeClass::Matmul,
            )?;
            let beta = builder.op_matmul(scan_input, w_beta, DType::F32)?;

            let handle_rec = builder.state(
                layer_idx,
                StateSpec::Recurrent {
                    h: h_heads,
                    d: head_d,
                    dv: head_dv,
                },
            )?;

            let scan_3d = builder.op_linear_attn_scan(
                q,
                k,
                v,
                alpha,
                beta,
                *kind,
                64,
                DType::F16,
                handle_rec,
            )?;

            let mut scan_out = builder.op_reshape(
                scan_3d,
                vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(scan_v_dim)],
            )?;

            if let Some(norm) = output_norm {
                let w_norm = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ssm_norm.weight"),
                    WeightRole::Vector,
                    &[Dim::Concrete(scan_v_dim)],
                    SchemeClass::Vector,
                )?;
                scan_out = builder.op_norm(scan_out, w_norm, *norm, NormAxis::Last, DType::F16)?;
            }

            if *output_gate {
                let w_gate = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ssm_gate.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(scan_v_dim), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let gate = builder.op_matmul(h, w_gate, DType::F16)?;
                scan_out = builder.op_act_mul(gate, scan_out, *gate_act)?;
            }

            let w_o = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ssm_out.weight"),
                WeightRole::Matmul,
                &[Dim::Concrete(model.dm), Dim::Concrete(scan_v_dim)],
                SchemeClass::Matmul,
            )?;
            builder.op_matmul(scan_out, w_o, DType::F16)
        }
        Mixer::None => Ok(h),
    }
}

/// Emits the feed-forward network sublayer (Dense or MoE; Spec 8 §3.1).
///
/// Spec-facing entry point: base-model weights live directly under `blk.N.`.
pub fn build_ffn(
    builder: &mut GraphBuilder,
    layer_idx: u32,
    ffn: &Ffn,
    h: Value,
    model: &ModelSpec,
) -> Result<Value, ModelsError> {
    build_ffn_with_ns(builder, layer_idx, ffn, h, model, "")
}

/// Emits one feed-forward sublayer with `weight_ns` inserted after `blk.N.`.
fn build_ffn_with_ns(
    builder: &mut GraphBuilder,
    layer_idx: u32,
    ffn: &Ffn,
    h: Value,
    model: &ModelSpec,
    weight_ns: &str,
) -> Result<Value, ModelsError> {
    match ffn {
        Ffn::Dense {
            dff,
            act,
            gated,
            bias,
        } => {
            if *gated {
                builder.declare_fusion(FusionDecl::GateUp {
                    gate: format!("blk.{layer_idx}.{weight_ns}ffn_gate.weight"),
                    up: format!("blk.{layer_idx}.{weight_ns}ffn_up.weight"),
                })?;
                let w_gate = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_gate.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(*dff), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                // DECISION(A1.3): gated `bias` lowers as bias epilogues on the
                // gate and up projections (`ffn_gate.bias`, `ffn_up.bias`),
                // matching the ungated path where `bias` rides `ffn_up`;
                // rejected dropping the flag (previous behavior, silently
                // wrong) and a down-projection bias (no ungated precedent).
                // Spec 8 §3 names `bias` on `Dense` without exempting gated.
                let g = if *bias {
                    let b_gate = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}ffn_gate.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(*dff)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(h.clone(), w_gate, b_gate, DType::F16)?
                } else {
                    builder.op_matmul(h.clone(), w_gate, DType::F16)?
                };

                let w_up = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_up.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(*dff), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let u = if *bias {
                    let b_up = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}ffn_up.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(*dff)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(h, w_up, b_up, DType::F16)?
                } else {
                    builder.op_matmul(h, w_up, DType::F16)?
                };

                let act_gu = builder.op_act_mul(g, u, *act)?;
                let w_down = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_down.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(model.dm), Dim::Concrete(*dff)],
                    SchemeClass::Matmul,
                )?;
                builder.op_matmul(act_gu, w_down, DType::F16)
            } else {
                let w_up = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_up.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(*dff), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let u = if *bias {
                    let bias_w = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}ffn_up.bias"),
                        WeightRole::Vector,
                        &[Dim::Concrete(*dff)],
                        SchemeClass::Vector,
                    )?;
                    builder.op_matmul_bias(h, w_up, bias_w, DType::F16)?
                } else {
                    builder.op_matmul(h, w_up, DType::F16)?
                };
                let act_u = builder.op_activation(u, *act)?;
                let w_down = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_down.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(model.dm), Dim::Concrete(*dff)],
                    SchemeClass::Matmul,
                )?;
                builder.op_matmul(act_u, w_down, DType::F16)
            }
        }
        Ffn::Moe {
            e,
            k,
            dff_e,
            act,
            scoring,
            renormalize,
            group,
            route_bias,
            route_scale,
            shared,
            shared_gate,
        } => {
            // Router
            let w_router = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ffn_gate_inp.weight"),
                WeightRole::Matmul,
                &[Dim::Concrete(*e), Dim::Concrete(model.dm)],
                SchemeClass::Matmul,
            )?;
            let router_logits = builder.op_matmul(h.clone(), w_router, DType::F32)?;
            let bias = if *route_bias {
                Some(builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_gate_inp.bias"),
                    WeightRole::Vector,
                    &[Dim::Concrete(*e)],
                    SchemeClass::Vector,
                )?)
            } else {
                None
            };

            let group_ir = group.map(|g| MoeGroup {
                n_group: g.n_group,
                topk_group: g.topk_group,
            });
            let (route_indices, route_weights) = builder.op_moe_route(
                router_logits,
                bias,
                *k,
                *scoring,
                *renormalize,
                group_ir,
                *route_scale,
            )?;

            // Expert execution
            let gate_up_dim = checked_mul(2, *dff_e, "moe gate_up experts")?;
            let w_gate_up_exps = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ffn_gate_up_exps.weight"),
                WeightRole::Matmul,
                &[
                    Dim::Concrete(*e),
                    Dim::Concrete(gate_up_dim),
                    Dim::Concrete(model.dm),
                ],
                SchemeClass::Matmul,
            )?;
            let w_down_exps = builder.weight(
                format!("blk.{layer_idx}.{weight_ns}ffn_down_exps.weight"),
                WeightRole::Matmul,
                &[
                    Dim::Concrete(*e),
                    Dim::Concrete(model.dm),
                    Dim::Concrete(*dff_e),
                ],
                SchemeClass::Matmul,
            )?;

            let shared_count = shared.map(|s| s.n).unwrap_or(0);
            let mut moe_out = builder.op_moe_ffn(
                h.clone(),
                route_indices,
                route_weights,
                w_gate_up_exps,
                w_down_exps,
                *act,
                DType::F16,
                shared_count,
            )?;

            // Shared experts if present
            if let Some(s) = shared {
                let w_shared_gate = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_shared_gate.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(s.dff), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let sg = builder.op_matmul(h.clone(), w_shared_gate, DType::F16)?;

                let w_shared_up = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_shared_up.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(s.dff), Dim::Concrete(model.dm)],
                    SchemeClass::Matmul,
                )?;
                let su = builder.op_matmul(h.clone(), w_shared_up, DType::F16)?;

                let mut s_act = builder.op_act_mul(sg, su, *act)?;

                if *shared_gate {
                    let w_sg = builder.weight(
                        format!("blk.{layer_idx}.{weight_ns}ffn_shared_gate_routing.weight"),
                        WeightRole::Matmul,
                        &[Dim::Concrete(s.dff), Dim::Concrete(model.dm)],
                        SchemeClass::Matmul,
                    )?;
                    let s_gate = builder.op_matmul(h, w_sg, DType::F16)?;
                    s_act = builder.op_act_mul(s_gate, s_act, ActivationKind::Silu)?;
                }

                let w_shared_down = builder.weight(
                    format!("blk.{layer_idx}.{weight_ns}ffn_shared_down.weight"),
                    WeightRole::Matmul,
                    &[Dim::Concrete(model.dm), Dim::Concrete(s.dff)],
                    SchemeClass::Matmul,
                )?;
                let shared_out = builder.op_matmul(s_act, w_shared_down, DType::F16)?;
                moe_out = builder.op_residual_add(moe_out, shared_out, DType::F16)?;
            }

            Ok(moe_out)
        }
        Ffn::None => Ok(h),
    }
}

/// Builds the Multi-Token Prediction (MTP) subgraph (Spec 8 §2, §3, §5).
pub fn build_mtp_subgraph(
    parent_builder: &mut GraphBuilder,
    mtp: &MtpSpec,
    hidden: Value,
    model: &ModelSpec,
) -> Result<(), ModelsError> {
    // Validate before the head loop below: `heads` and `layers_per_head`
    // drive allocation and loop counts, and this entry point is public, so
    // adversarial dimensions must fail here rather than hang or OOM.
    mtp.validate(model.layers.len())?;
    // The child graph's input explicitly captures the chosen parent hidden
    // value (Layer(n) or Last, selected by the caller); every head restarts
    // from that capture, so no head chains off another head's output
    // (card A1.14, SI-23).
    let mut mtp_builder = parent_builder.subgraph_with_capture("mtp", &hidden)?;
    let head_input = mtp_builder.capture_value()?;
    let per_head = checked_u32(mtp.layers_per_head.len(), "mtp layers per head")?;
    for head in 1..=mtp.heads {
        let head_base = checked_mul(head - 1, per_head, "mtp head ordinal")?;
        let mut head_hidden = head_input.clone();
        for (i, layer_spec) in mtp.layers_per_head.iter().enumerate() {
            let layer_idx = checked_add(
                head_base,
                checked_u32(i, "mtp layer index")?,
                "mtp layer ordinal",
            )?;
            head_hidden = build_layer_with_ns(
                &mut mtp_builder,
                layer_idx,
                layer_spec,
                head_hidden,
                model,
                "mtp.",
            )?;
        }

        let w_head = mtp_builder.weight(
            format!("blk.{}.mtp.output.weight", head - 1),
            WeightRole::LmHead,
            &[Dim::Concrete(model.vocab), Dim::Concrete(model.dm)],
            SchemeClass::Matmul,
        )?;
        let logits = mtp_builder.op_matmul(head_hidden.clone(), w_head, DType::F32)?;
        mtp_builder.export(format!("mtp_logits_{head}"), logits)?;
    }

    let mtp_graph = mtp_builder.finish()?;
    parent_builder.add_subgraph("mtp", mtp_graph)?;
    Ok(())
}
