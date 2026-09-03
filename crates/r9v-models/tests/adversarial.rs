// SPDX-License-Identifier: Apache-2.0
//! Adversarial regression tests for A1.3 release blockers (Spec 8 §6, §7;
//! card A1.3).
//!
//! Every test here feeds hostile dimensions (`u32::MAX`, over-limit counts)
//! and asserts a typed error — never a panic, wrap, clamp, hang, or OOM.
//! Tests that must not allocate bind one small tensor or none at all; the
//! rejection always fires before any size-driven allocation or loop.

use std::collections::BTreeMap;

use r9v_ir::op::{ActivationKind, Epilogue, HashId, NgramCombine, Op, RopeScaling, RopeStyle};
use r9v_ir::tensor::{Dim, ShapeSymbol};
use r9v_ir::version::IrVersion;
use r9v_models::{
    build_ffn, build_layer, build_mixer, build_model, build_mtp_subgraph, CacheDtype, Ffn, Graph,
    GraphBuilder, LayerSpec, LayerSummary, Mixer, MlaSpec, ModelSpec, ModelSummary, ModelsError,
    MtpSource, MtpSpec, NgramSpec, NormPlacement, NormSpec, PositionEncoding, Retain, RopeSpec,
    SchemeClass, SchemeKey, StateSpec, WeightRole, MAX_EXPERTS, MAX_FEATURE_DIM, MAX_KV_HEADS,
    MAX_MODEL_LAYERS,
};

fn tiny_rope() -> RopeSpec {
    RopeSpec {
        theta: 10000.0,
        rot_dim: 32,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
    }
}

fn tiny_layer() -> LayerSpec {
    LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::None,
        ffn: Ffn::None,
        residual_scale: 1.0,
    }
}

fn tiny_model(layers: Vec<LayerSpec>) -> ModelSpec {
    ModelSpec {
        dm: 64,
        layers,
        vocab: 64,
        embed_scale: 1.0,
        tied_embeddings: false,
        final_norm: NormSpec::rms(1e-5),
        final_logit_softcap: None,
        positions: PositionEncoding::Scalar,
        ngram: None,
        mtp: None,
        export_hidden: false,
        eos_ids: vec![2],
        bos_id: Some(1),
    }
}

fn plain_attention() -> Mixer {
    Mixer::Attention {
        h: 4,
        hkv: 2,
        d: 32,
        dv: 32,
        qkv_bias: false,
        o_bias: false,
        qk_norm: None,
        rope: tiny_rope(),
        window: None,
        sinks: 0,
        logit_softcap: None,
        output_gate: false,
        mla: None,
        cache: CacheDtype::E4m3,
    }
}

/// Tensor byte products that overflow `u64` report `ArithmeticOverflow`.
#[test]
fn summary_tensor_bytes_overflow_returns_typed_error() {
    let mut builder = Graph::new(IrVersion::CURRENT, "adv-bytes-overflow");
    builder
        .weight(
            "blk.0.attn_q.weight",
            WeightRole::Matmul,
            &[
                Dim::Concrete(u32::MAX),
                Dim::Concrete(u32::MAX),
                Dim::Concrete(u32::MAX),
            ],
            SchemeClass::Matmul,
        )
        .expect("metadata-only weight bind must succeed");
    let graph = builder.finish().expect("op-free graph must finish");
    let err = graph.summary().unwrap_err();
    assert!(
        matches!(err, ModelsError::ArithmeticOverflow { .. }),
        "expected ArithmeticOverflow, got {err:?}"
    );
    assert!(
        format!("{err:?}").contains("4294967295"),
        "operands reported: {err:?}"
    );
}

/// Layer index `u32::MAX` overflows the `max + 1` layer count as typed error.
#[test]
fn summary_max_layer_ordinal_overflow_returns_typed_error() {
    let mut builder = Graph::new(IrVersion::CURRENT, "adv-layer-ordinal");
    builder
        .weight(
            "blk.4294967295.attn_q.weight",
            WeightRole::Matmul,
            &[Dim::Concrete(8), Dim::Concrete(8)],
            SchemeClass::Matmul,
        )
        .expect("metadata-only weight bind must succeed");
    let graph = builder.finish().expect("op-free graph must finish");
    let err = graph.summary().unwrap_err();
    assert!(
        matches!(err, ModelsError::ArithmeticOverflow { .. }),
        "expected ArithmeticOverflow, got {err:?}"
    );
}

/// Absurd layer ordinals are rejected before `Vec::with_capacity`, not OOM.
#[test]
fn summary_absurd_layer_ordinal_rejected_before_alloc() {
    let mut builder = Graph::new(IrVersion::CURRENT, "adv-layer-count");
    builder
        .weight(
            "blk.2000000.attn_q.weight",
            WeightRole::Matmul,
            &[Dim::Concrete(8), Dim::Concrete(8)],
            SchemeClass::Matmul,
        )
        .expect("metadata-only weight bind must succeed");
    let graph = builder.finish().expect("op-free graph must finish");
    let err = graph.summary().unwrap_err();
    assert!(
        matches!(err, ModelsError::InvalidModelSpec { .. }),
        "expected InvalidModelSpec, got {err:?}"
    );
    assert!(
        format!("{err:?}").contains(&MAX_MODEL_LAYERS.to_string()),
        "limit reported: {err:?}"
    );
}

/// Absurd `hkv` is rejected before the TP-divisor enumeration, not looped.
#[test]
fn summary_absurd_hkv_rejected_before_divisor_loop() {
    let mut builder = Graph::new(IrVersion::CURRENT, "adv-hkv");
    builder
        .state(
            0,
            StateSpec::KvPaged {
                hkv: u32::MAX,
                d: 16,
                dv: 16,
                cache: CacheDtype::F16,
                retain: Retain::All,
            },
        )
        .expect("state declare must succeed");
    let graph = builder.finish().expect("op-free graph must finish");
    let err = graph.summary().unwrap_err();
    assert!(
        matches!(err, ModelsError::InvalidModelSpec { .. }),
        "expected InvalidModelSpec, got {err:?}"
    );
    assert!(
        format!("{err:?}").contains(&MAX_KV_HEADS.to_string()),
        "limit reported: {err:?}"
    );
}

/// Absurd expert counts are rejected before the `hot_hint` allocation.
#[test]
fn summary_absurd_expert_count_rejected_before_hot_hint_alloc() {
    let mut builder = Graph::new(IrVersion::CURRENT, "adv-experts");
    builder
        .weight(
            "blk.0.ffn_gate_up_exps.weight",
            WeightRole::Matmul,
            &[Dim::Concrete(u32::MAX), Dim::Concrete(8), Dim::Concrete(8)],
            SchemeClass::Matmul,
        )
        .expect("metadata-only weight bind must succeed");
    let graph = builder.finish().expect("op-free graph must finish");
    let err = graph.summary().unwrap_err();
    assert!(
        matches!(err, ModelsError::InvalidModelSpec { .. }),
        "expected InvalidModelSpec, got {err:?}"
    );
    assert!(
        format!("{err:?}").contains(&MAX_EXPERTS.to_string()),
        "limit reported: {err:?}"
    );
}

/// Graph-global weights alone summarize zero layers: no phantom layer.
#[test]
fn summary_with_only_global_weights_has_no_layers() {
    let mut builder = Graph::new(IrVersion::CURRENT, "adv-globals-only");
    builder
        .weight(
            "token_embd.weight",
            WeightRole::Embed,
            &[Dim::Concrete(64), Dim::Concrete(64)],
            SchemeClass::Embed,
        )
        .expect("embed bind must succeed");
    builder
        .weight(
            "output.weight",
            WeightRole::LmHead,
            &[Dim::Concrete(64), Dim::Concrete(64)],
            SchemeClass::Matmul,
        )
        .expect("head bind must succeed");
    builder
        .weight(
            "output_norm.weight",
            WeightRole::Vector,
            &[Dim::Concrete(64)],
            SchemeClass::Vector,
        )
        .expect("norm bind must succeed");
    let graph = builder.finish().expect("op-free graph must finish");
    let summary = graph.summary().expect("global-only summary must compute");
    assert!(summary.layers.is_empty(), "no phantom layer for globals");
    assert_eq!(summary.embed_bytes, 64 * 64 * 2);
    assert_eq!(summary.head_bytes, 64 * 64 * 2);
    assert_eq!(
        summary.total_state_per_token_bytes().expect("state total"),
        0
    );
    assert_eq!(
        summary.total_weight_bytes().expect("weight total"),
        2 * 64 * 64 * 2
    );
}

/// `StateSpec` byte math overflows as typed error and stays exact otherwise.
#[test]
fn state_spec_byte_overflow_returns_typed_error() {
    let huge_paged = StateSpec::KvPaged {
        hkv: u32::MAX,
        d: u32::MAX,
        dv: u32::MAX,
        cache: CacheDtype::E4m3,
        retain: Retain::All,
    };
    assert!(
        matches!(
            huge_paged.state_per_token_bytes().unwrap_err(),
            ModelsError::ArithmeticOverflow { .. }
        ),
        "paged KV overflow must be typed"
    );

    let huge_recurrent = StateSpec::Recurrent {
        h: u32::MAX,
        d: u32::MAX,
        dv: u32::MAX,
    };
    assert!(
        matches!(
            huge_recurrent.state_per_seq_bytes().unwrap_err(),
            ModelsError::ArithmeticOverflow { .. }
        ),
        "recurrent overflow must be typed"
    );

    let huge_conv = StateSpec::ConvWindow {
        c: u32::MAX,
        w: u32::MAX,
    };
    assert!(
        matches!(
            huge_conv.state_per_seq_bytes().unwrap_err(),
            ModelsError::ArithmeticOverflow { .. }
        ),
        "conv window overflow must be typed"
    );

    // Exactness oracle (Spec 3 §6.2): hkv * ((d + dv) * elem + scales).
    let paged = StateSpec::KvPaged {
        hkv: 4,
        d: 32,
        dv: 32,
        cache: CacheDtype::E4m3,
        retain: Retain::All,
    };
    assert_eq!(paged.state_per_token_bytes().expect("exact"), 272);
    let latent = StateSpec::KvLatent {
        latent: 64,
        rope: 32,
        cache: CacheDtype::E4m3,
        retain: Retain::All,
    };
    assert_eq!(latent.state_per_token_bytes().expect("exact"), 130);
    let recurrent = StateSpec::Recurrent {
        h: 4,
        d: 16,
        dv: 16,
    };
    assert_eq!(recurrent.state_per_seq_bytes().expect("exact"), 8192);
    let conv = StateSpec::ConvWindow { c: 256, w: 4 };
    assert_eq!(conv.state_per_seq_bytes().expect("exact"), 1536);
}

/// Summary totals that overflow `u64` report `ArithmeticOverflow`.
#[test]
fn summary_totals_overflow_returns_typed_error() {
    let mut buckets = BTreeMap::new();
    buckets.insert(SchemeKey::None, u64::MAX);
    buckets.insert(SchemeKey::PerRow, 1);
    let layer = LayerSummary {
        weight_bytes_by_scheme: buckets,
        state_per_token_bytes: 0,
        state_per_seq_bytes: 0,
        experts: None,
        mixer_kind: None,
    };
    assert!(
        matches!(
            layer.total_weight_bytes().unwrap_err(),
            ModelsError::ArithmeticOverflow { .. }
        ),
        "layer bucket overflow must be typed"
    );

    // Totals overflow across layers even when each layer alone fits: two
    // layers at `u64::MAX` each cannot sum.
    let mut single = BTreeMap::new();
    single.insert(SchemeKey::None, u64::MAX);
    let max_layer = LayerSummary {
        weight_bytes_by_scheme: single,
        state_per_token_bytes: u64::MAX,
        state_per_seq_bytes: u64::MAX,
        experts: None,
        mixer_kind: None,
    };
    assert_eq!(
        max_layer
            .total_weight_bytes()
            .expect("single max layer fits"),
        u64::MAX
    );
    let summary = ModelSummary {
        layers: vec![max_layer.clone(), max_layer],
        embed_bytes: 0,
        head_bytes: 0,
        vocab: 64,
        dm: 64,
        hkv: 0,
        tp_divisors: vec![1],
        ngram_table_bytes: 0,
        mtp: false,
        export_hidden: false,
    };
    for total in [
        summary.total_weight_bytes().unwrap_err(),
        summary.total_state_per_token_bytes().unwrap_err(),
        summary.total_state_per_seq_bytes().unwrap_err(),
    ] {
        assert!(
            matches!(total, ModelsError::ArithmeticOverflow { .. }),
            "summary total overflow must be typed, got {total:?}"
        );
    }
}

/// Huge MoE expert counts fail validation before any allocation, all at once.
#[test]
fn huge_expert_count_rejected_before_allocation() {
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: u32::MAX,
            hkv: u32::MAX,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: tiny_rope(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4m3,
        },
        ffn: Ffn::Moe {
            e: u32::MAX,
            k: 1,
            dff_e: 16,
            act: ActivationKind::Silu,
            scoring: r9v_ir::op::MoeScoring::Softmax,
            renormalize: true,
            group: None,
            route_bias: false,
            route_scale: 1.0,
            shared: None,
            shared_gate: false,
        },
        residual_scale: 1.0,
    };
    // Collect-all: h, hkv, and e violations arrive together, not first-only.
    let err = layer.validate(0).unwrap_err();
    let text = format!("{err:?}");
    for dim in ["dimension 'h' ", "dimension 'hkv' ", "dimension 'e' "] {
        assert!(text.contains(dim), "missing {dim}problem in: {text}");
    }

    // Lowering refuses the same spec without building (no hang, no OOM).
    let model = tiny_model(vec![tiny_layer()]);
    let mut builder = GraphBuilder::new(IrVersion::CURRENT, "adv-huge-experts");
    let (x, _) = builder.input_embed_override(model.dm).expect("override");
    assert!(build_layer(&mut builder, 0, &layer, x, &model).is_err());
}

/// Huge MTP head counts fail before the head loop runs.
#[test]
fn huge_mtp_heads_rejected_without_building() {
    let mtp = MtpSpec {
        heads: u32::MAX,
        layers_per_head: vec![tiny_layer()],
        takes_hidden_from: MtpSource::Last,
    };
    let err = mtp.validate(1).unwrap_err();
    assert!(
        format!("{err:?}").contains("mtp heads"),
        "mtp heads violation reported: {err:?}"
    );

    let model = tiny_model(vec![tiny_layer()]);
    let mut parent = Graph::new(IrVersion::CURRENT, "adv-huge-mtp");
    let (hidden, _) = parent.input_embed_override(model.dm).expect("override");
    assert!(build_mtp_subgraph(&mut parent, &mtp, hidden, &model).is_err());
}

/// Over-limit layer counts fail validation (and the build) up front.
#[test]
fn huge_layer_count_rejected_up_front() {
    let layers = vec![tiny_layer(); MAX_MODEL_LAYERS as usize + 1];
    let model = tiny_model(layers);
    let err = model.validate().unwrap_err();
    assert!(
        format!("{err:?}").contains(&MAX_MODEL_LAYERS.to_string()),
        "layer count violation reported: {err:?}"
    );
    let builder = Graph::new(IrVersion::CURRENT, "adv-huge-layers");
    assert!(build_model(builder, &model).is_err());
}

/// Huge n-gram dimensions fail validation before table weights bind.
#[test]
fn huge_ngram_dims_rejected_before_binding() {
    let ngram = NgramSpec {
        orders: vec![2],
        heads: u32::MAX,
        dim: 32,
        table_sizes: vec![u32::MAX],
        hash: HashId::new(1),
        combine: NgramCombine::Sum,
        inject_at: 0,
    };
    let err = ngram.validate(1).unwrap_err();
    let text = format!("{err:?}");
    assert!(text.contains("ngram heads"), "heads violation: {text}");
    assert!(
        text.contains("ngram table entries"),
        "table violation: {text}"
    );

    let mut model = tiny_model(vec![tiny_layer()]);
    model.ngram = Some(ngram);
    let builder = Graph::new(IrVersion::CURRENT, "adv-huge-ngram");
    assert!(build_model(builder, &model).is_err());
}

/// Model-level dimension caps collect every violation together.
#[test]
fn model_level_dim_caps_collect_all_problems() {
    let model = ModelSpec {
        dm: u32::MAX,
        vocab: u32::MAX,
        ..tiny_model(vec![tiny_layer()])
    };
    let err = model.validate().unwrap_err();
    let text = format!("{err:?}");
    assert!(text.contains("dimension 'dm'"), "dm violation: {text}");
    assert!(
        text.contains("dimension 'vocab'"),
        "vocab violation: {text}"
    );
}

/// Dense `gated` plus `bias` lowers both biases through the matmul epilogue.
#[test]
fn dense_gated_bias_lowers_through_bias_epilogue() {
    let mixer = plain_attention();
    let ffn = Ffn::Dense {
        dff: 64,
        act: ActivationKind::Silu,
        gated: true,
        bias: true,
    };
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: mixer.clone(),
        ffn: ffn.clone(),
        residual_scale: 1.0,
    };
    let mut model = tiny_model(vec![layer]);
    model.dm = 128;
    let graph = build_model(Graph::new(IrVersion::CURRENT, "adv-gated-bias"), &model)
        .expect("gated-bias model must build and validate");

    // Canonical bias weights with exact shapes and F32 vector dtype.
    // `dm` (128) differs from `dff` (64) so bias shapes are unambiguous.
    for name in ["blk.0.ffn_gate.bias", "blk.0.ffn_up.bias"] {
        let w = graph
            .bound_weights()
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("missing bias weight {name}"));
        assert_eq!(w.shape, vec![Dim::Concrete(64)], "shape of {name}");
        assert_eq!(w.tensor.dtype(), r9v_ir::DType::F32, "dtype of {name}");
    }

    // Exactly the gate and up projections carry the bias epilogue, fed by
    // the bias weight edges; every other matmul stays epilogue-free.
    let graph_ref = graph.graph();
    let bias_names: Vec<&str> = graph
        .bound_weights()
        .iter()
        .filter(|w| w.name.ends_with(".bias"))
        .map(|w| w.name.as_str())
        .collect();
    assert_eq!(
        bias_names.len(),
        2,
        "only gate and up biases: {bias_names:?}"
    );
    let mut bias_epilogues = 0;
    for node in graph_ref.nodes() {
        if let Op::Matmul(m) = &node.op {
            match m.epilogue {
                Epilogue::Bias => {
                    bias_epilogues += 1;
                    let last = node.inputs.last().expect("bias input edge");
                    let input = &graph_ref.edges()[last.0].tensor;
                    assert_eq!(input.dtype(), r9v_ir::DType::F32);
                    assert_eq!(input.shape().to_vec(), vec![Dim::Concrete(64)]);
                }
                Epilogue::None => {}
                _ => panic!("unexpected matmul epilogue {:?}", m.epilogue),
            }
        }
    }
    assert_eq!(bias_epilogues, 2, "gate and up projections");
}

/// MLA with `qk_norm` lowers to per-side norms instead of being rejected or
/// ignored (Spec 8 §3; card A1.14, SI-20).
#[test]
fn mla_qk_norm_lowers_instead_of_rejecting() {
    let mixer = Mixer::Attention {
        h: 4,
        hkv: 1,
        d: 32,
        dv: 32,
        qkv_bias: false,
        o_bias: false,
        qk_norm: Some(NormSpec::rms(1e-5)),
        rope: tiny_rope(),
        window: None,
        sinks: 0,
        logit_softcap: None,
        output_gate: false,
        mla: Some(MlaSpec {
            q_lora_rank: 32,
            kv_lora_rank: 16,
            qk_nope_dim: 16,
            qk_rope_dim: 16,
            v_dim: 32,
        }),
        cache: CacheDtype::E4m3,
    };
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: mixer.clone(),
        ffn: Ffn::None,
        residual_scale: 1.0,
    };
    // Validation accepts the combination; the detailed edge oracles live in
    // the A1.14 cohesion tests.
    layer.validate(0).expect("mla with qk_norm validates");

    let model = tiny_model(vec![tiny_layer()]);
    let mut builder = GraphBuilder::new(IrVersion::CURRENT, "adv-mla-qknorm");
    let (x, _) = builder.input_embed_override(model.dm).expect("override");
    build_layer(&mut builder, 0, &layer, x, &model).expect("mla with qk_norm lowers");
    let graph = builder.finish().expect("qk_norm MLA graph validates");
    for name in ["blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"] {
        assert!(
            graph.bound_weights().iter().any(|w| w.name == name),
            "missing MLA qk_norm weight {name}"
        );
    }

    // The bare-mixer entry point lowers it too (no silent drop past validation).
    let mut direct = GraphBuilder::new(IrVersion::CURRENT, "adv-mla-qknorm-direct");
    let (h, _) = direct.input_embed_override(model.dm).expect("override");
    build_mixer(&mut direct, 0, &mixer, h, &model).expect("bare mixer lowers MLA qk_norm");
}

/// The public spec-facing builders take no namespace plumbing and lower.
#[test]
fn public_build_apis_lower_without_namespace_arg() {
    let mut model = tiny_model(vec![tiny_layer()]);
    model.dm = 128;
    let mixer = Mixer::Attention {
        h: 4,
        hkv: 2,
        d: 32,
        dv: 32,
        qkv_bias: false,
        o_bias: false,
        qk_norm: None,
        rope: tiny_rope(),
        window: None,
        sinks: 0,
        logit_softcap: None,
        output_gate: false,
        mla: None,
        cache: CacheDtype::E4m3,
    };
    let ffn = Ffn::Dense {
        dff: 64,
        act: ActivationKind::Silu,
        gated: true,
        bias: false,
    };

    let mut builder = GraphBuilder::new(IrVersion::CURRENT, "adv-public-api");
    let (h, _) = builder.input_embed_override(model.dm).expect("override");
    let mixer_out =
        build_mixer(&mut builder, 0, &mixer, h.clone(), &model).expect("public build_mixer lowers");
    assert_eq!(
        mixer_out.tensor().shape(),
        &[Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(model.dm)],
        "mixer output is [T, Dm]"
    );
    let ffn_out = build_ffn(&mut builder, 0, &ffn, h, &model).expect("public build_ffn lowers");
    assert_eq!(
        ffn_out.tensor().shape(),
        &[Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(model.dm)],
        "ffn output is [T, Dm]"
    );
    let _ = mixer_out;
    let _ = ffn_out;
}

/// `shared_gate = true` with `shared = None` is rejected at validation and at
/// lowering, never silently ignored.
#[test]
fn moe_shared_gate_without_shared_rejected_not_ignored() {
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::None,
        ffn: Ffn::Moe {
            e: 4,
            k: 1,
            dff_e: 16,
            act: ActivationKind::Silu,
            scoring: r9v_ir::op::MoeScoring::Softmax,
            renormalize: true,
            group: None,
            route_bias: false,
            route_scale: 1.0,
            shared: None,
            shared_gate: true,
        },
        residual_scale: 1.0,
    };
    let err = layer.validate(0).unwrap_err();
    assert!(
        format!("{err:?}").contains("shared_gate"),
        "validation names shared_gate: {err:?}"
    );

    let model = tiny_model(vec![tiny_layer()]);
    let mut builder = GraphBuilder::new(IrVersion::CURRENT, "adv-shared-gate");
    let (x, _) = builder.input_embed_override(model.dm).expect("override");
    let err = build_layer(&mut builder, 0, &layer, x, &model).unwrap_err();
    assert!(
        format!("{err:?}").contains("shared_gate"),
        "layer lowering names shared_gate: {err:?}"
    );
}

/// `sinks > 0` without a window is rejected: Spec 3 §2 has no sink-only
/// `Retain` form and sinks are only meaningful with a window.
#[test]
fn attention_sinks_without_window_rejected() {
    let mixer = Mixer::Attention {
        h: 4,
        hkv: 2,
        d: 32,
        dv: 32,
        qkv_bias: false,
        o_bias: false,
        qk_norm: None,
        rope: tiny_rope(),
        window: None,
        sinks: 4,
        logit_softcap: None,
        output_gate: false,
        mla: None,
        cache: CacheDtype::E4m3,
    };
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: mixer.clone(),
        ffn: Ffn::None,
        residual_scale: 1.0,
    };
    let err = layer.validate(0).unwrap_err();
    assert!(
        format!("{err:?}").contains("sinks"),
        "validation names sinks: {err:?}"
    );

    let model = tiny_model(vec![tiny_layer()]);
    let mut builder = GraphBuilder::new(IrVersion::CURRENT, "adv-sink-only");
    let (x, _) = builder.input_embed_override(model.dm).expect("override");
    let err = build_layer(&mut builder, 0, &layer, x, &model).unwrap_err();
    assert!(
        format!("{err:?}").contains("sinks"),
        "layer lowering names sinks: {err:?}"
    );

    // The bare-mixer entry point refuses it too (defense past validation).
    let mut direct = GraphBuilder::new(IrVersion::CURRENT, "adv-sink-only-direct");
    let (h, _) = direct.input_embed_override(model.dm).expect("override");
    let err = build_mixer(&mut direct, 0, &mixer, h, &model).unwrap_err();
    assert!(
        matches!(err, ModelsError::InvalidModelSpec { .. }),
        "mixer lowering rejects sink-only retention: {err:?}"
    );
}

/// `Retain::from_window_sinks` covers the Spec 3 §2 forms exactly; sink-only
/// input is a typed error, never a `u32::MAX` window.
#[test]
fn retain_sink_only_returns_typed_error_not_sentinel() {
    assert_eq!(
        Retain::from_window_sinks(None, 0).expect("all"),
        Retain::All
    );
    assert_eq!(
        Retain::from_window_sinks(Some(128), 0).expect("window"),
        Retain::Window(128)
    );
    assert_eq!(
        Retain::from_window_sinks(Some(128), 4).expect("sink plus window"),
        Retain::SinkAndWindow {
            sinks: 4,
            window: 128
        }
    );
    let err = Retain::from_window_sinks(None, 4).unwrap_err();
    assert!(
        matches!(err, ModelsError::InvalidModelSpec { .. }),
        "sink-only retention is InvalidModelSpec: {err:?}"
    );
    let text = format!("{err:?}");
    assert!(
        !text.contains("4294967295"),
        "no u32::MAX sentinel leaks into the error: {text}"
    );
}

/// N-gram `dim` (Dn) is nonzero and bounded; `orders` and `table_sizes`
/// lengths each equal `heads`, collected together rather than first-only.
#[test]
fn ngram_dim_and_head_lengths_validated() {
    let zero_dim = NgramSpec {
        orders: vec![2],
        heads: 1,
        dim: 0,
        table_sizes: vec![64],
        hash: HashId::new(1),
        combine: NgramCombine::Sum,
        inject_at: 0,
    };
    let err = zero_dim.validate(1).unwrap_err();
    assert!(
        format!("{err:?}").contains("ngram dim"),
        "zero dim rejected: {err:?}"
    );

    let huge_dim = NgramSpec {
        orders: vec![2],
        heads: 1,
        dim: MAX_FEATURE_DIM + 1,
        table_sizes: vec![64],
        hash: HashId::new(1),
        combine: NgramCombine::Sum,
        inject_at: 0,
    };
    let err = huge_dim.validate(1).unwrap_err();
    assert!(
        format!("{err:?}").contains("ngram dim"),
        "over-limit dim rejected: {err:?}"
    );

    let mismatched = NgramSpec {
        orders: vec![2],
        heads: 2,
        dim: 32,
        table_sizes: vec![64],
        hash: HashId::new(1),
        combine: NgramCombine::Sum,
        inject_at: 0,
    };
    let err = mismatched.validate(1).unwrap_err();
    let text = format!("{err:?}");
    assert!(
        text.contains("orders length"),
        "orders/heads mismatch reported: {text}"
    );
    assert!(
        text.contains("table_sizes length"),
        "tables/heads mismatch reported: {text}"
    );
}
