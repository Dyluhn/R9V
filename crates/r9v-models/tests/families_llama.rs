// SPDX-License-Identifier: Apache-2.0
//! Architecture family tests and deterministic golden JSON verification (Spec 8 §4, §5; card A1.4).
//!
//! Verifies:
//! 1. All eight supported architecture strings: `llama`, `mistral`, `qwen2`, `qwen3`,
//!    `gemma2`, `gemma3`, `phi3`, `olmo2`.
//! 2. ModelSpec and layer derivations including heterogeneous per-layer settings.
//! 3. Sealed builder graph emission, bound weight names, state declarations, fusions, ties.
//! 4. Deterministic golden JSON generation and regression checks.
//! 5. Adversarial and boundary cases: missing keys, malformed types, contradictory dimensions,
//!    unsupported variants, and overflow protection with collect-all error reporting.

use std::path::PathBuf;

use r9v_ir::op::{ActivationKind, NormKind, RopeScaling, RopeStyle};
use r9v_ir::tensor::Dim;
use r9v_ir::version::IrVersion;
use r9v_models::builder::{FusionDecl, Graph, ModelGraph};
use r9v_models::families;
use r9v_models::generic::build_model;
use r9v_models::meta::SyntheticGgufMeta;
use r9v_models::spec::{
    Ffn, LayerSpec, Mixer, ModelSpec, NormPlacement, NormSpec, PositionEncoding, Retain, RopeSpec,
    StateSpec,
};
use r9v_models::ModelsError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Golden Report Data Structures
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenReport {
    architecture: String,
    model_spec: GoldenModelSpec,
    bound_weights: Vec<String>,
    state_declarations: Vec<GoldenStateDecl>,
    fusion_declarations: Vec<GoldenFusionDecl>,
    tied_declarations: Vec<GoldenTiedDecl>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenModelSpec {
    dm: u32,
    vocab: u32,
    embed_scale: f32,
    tied_embeddings: bool,
    final_norm: GoldenNormSpec,
    final_logit_softcap: Option<f32>,
    positions: String,
    eos_ids: Vec<u32>,
    bos_id: Option<u32>,
    layer_count: usize,
    layers: Vec<GoldenLayerSpec>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenLayerSpec {
    layer_index: usize,
    norm_placement: String,
    norm_kind: GoldenNormSpec,
    residual_scale: f32,
    mixer: GoldenMixer,
    ffn: GoldenFfn,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenMixer {
    kind: String,
    h: u32,
    hkv: u32,
    d: u32,
    dv: u32,
    qkv_bias: bool,
    o_bias: bool,
    qk_norm: Option<GoldenNormSpec>,
    rope: GoldenRopeSpec,
    window: Option<u32>,
    sinks: u32,
    logit_softcap: Option<f32>,
    output_gate: bool,
    cache_dtype: String,
    pre_fused: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenFfn {
    kind: String,
    dff: u32,
    act: String,
    gated: bool,
    bias: bool,
    pre_fused: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenNormSpec {
    kind: String,
    eps: f32,
    weight_offset: f32,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenRopeSpec {
    theta: f32,
    rot_dim: u32,
    style: String,
    scaling: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenStateDecl {
    layer: u32,
    handle_layer: u32,
    state_kind: String,
    dim: u32,
    cache: String,
    retain: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenFusionDecl {
    kind: String,
    tensors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct GoldenTiedDecl {
    source: String,
    target: String,
}

fn convert_norm_spec(norm: &NormSpec) -> GoldenNormSpec {
    GoldenNormSpec {
        kind: match norm.kind {
            NormKind::Rms => "rms".to_string(),
            NormKind::Layer => "layer".to_string(),
        },
        eps: norm.eps,
        weight_offset: norm.weight_offset,
    }
}

fn convert_rope_scaling(scaling: &RopeScaling) -> String {
    match scaling {
        RopeScaling::None => "none".to_string(),
        RopeScaling::Linear(f) => format!("linear(factor={f})"),
        RopeScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
            orig_ctx,
            mscale,
        } => format!(
            "yarn(factor={factor}, beta_fast={beta_fast}, beta_slow={beta_slow}, orig_ctx={orig_ctx}, mscale={mscale})"
        ),
        RopeScaling::Dynamic => "dynamic".to_string(),
    }
}

fn convert_rope_spec(rope: &RopeSpec) -> GoldenRopeSpec {
    GoldenRopeSpec {
        theta: rope.theta,
        rot_dim: rope.rot_dim,
        style: match rope.style {
            RopeStyle::Neox => "neox".to_string(),
            RopeStyle::Interleaved => "interleaved".to_string(),
        },
        scaling: convert_rope_scaling(&rope.scaling),
    }
}

fn convert_layer_spec(index: usize, layer: &LayerSpec) -> GoldenLayerSpec {
    let norm_placement = match layer.norm {
        NormPlacement::Pre => "pre".to_string(),
        NormPlacement::Sandwich => "sandwich".to_string(),
        NormPlacement::Parallel => "parallel".to_string(),
        NormPlacement::Post => "post".to_string(),
    };

    let mixer = match &layer.mixer {
        Mixer::None => GoldenMixer {
            kind: "none".to_string(),
            h: 0,
            hkv: 0,
            d: 0,
            dv: 0,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: GoldenRopeSpec {
                theta: 0.0,
                rot_dim: 0,
                style: "".to_string(),
                scaling: "".to_string(),
            },
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            cache_dtype: "".to_string(),
            pre_fused: false,
        },
        Mixer::LinearAttention { .. } => unreachable!("linear attention not in llama family"),
        Mixer::Attention {
            h,
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
            cache,
            pre_fused,
            ..
        } => GoldenMixer {
            kind: "attention".to_string(),
            h: *h,
            hkv: *hkv,
            d: *d,
            dv: *dv,
            qkv_bias: *qkv_bias,
            o_bias: *o_bias,
            qk_norm: qk_norm.as_ref().map(convert_norm_spec),
            rope: convert_rope_spec(rope),
            window: *window,
            sinks: *sinks,
            logit_softcap: *logit_softcap,
            output_gate: *output_gate,
            cache_dtype: cache.name().to_string(),
            pre_fused: *pre_fused,
        },
    };

    let ffn = match &layer.ffn {
        Ffn::None => GoldenFfn {
            kind: "none".to_string(),
            dff: 0,
            act: "".to_string(),
            gated: false,
            bias: false,
            pre_fused: false,
        },
        Ffn::Dense {
            dff,
            act,
            gated,
            bias,
            pre_fused,
            ..
        } => GoldenFfn {
            kind: "dense".to_string(),
            dff: *dff,
            act: match act {
                ActivationKind::Silu => "silu".to_string(),
                ActivationKind::Gelu => "gelu".to_string(),
                ActivationKind::GeluTanh => "gelu_tanh".to_string(),
                ActivationKind::Relu2 => "relu2".to_string(),
                ActivationKind::Identity => "identity".to_string(),
            },
            gated: *gated,
            bias: *bias,
            pre_fused: *pre_fused,
        },
        Ffn::Moe { .. } => unreachable!("moe not in llama family"),
    };

    GoldenLayerSpec {
        layer_index: index,
        norm_placement,
        norm_kind: convert_norm_spec(&layer.norm_kind),
        residual_scale: layer.residual_scale,
        mixer,
        ffn,
    }
}

fn convert_model_spec(spec: &ModelSpec) -> GoldenModelSpec {
    GoldenModelSpec {
        dm: spec.dm,
        vocab: spec.vocab,
        embed_scale: spec.embed_scale,
        tied_embeddings: spec.tied_embeddings,
        final_norm: convert_norm_spec(&spec.final_norm),
        final_logit_softcap: spec.final_logit_softcap,
        positions: match spec.positions {
            PositionEncoding::Scalar => "scalar".to_string(),
            PositionEncoding::MRope(_) => "mrope".to_string(),
        },
        eos_ids: spec.eos_ids.clone(),
        bos_id: spec.bos_id,
        layer_count: spec.layers.len(),
        layers: spec
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| convert_layer_spec(i, l))
            .collect(),
    }
}

fn generate_report(arch: &str, spec: &ModelSpec, graph: &ModelGraph) -> GoldenReport {
    let bound_weights = graph
        .bound_weights()
        .iter()
        .map(|w| w.name.clone())
        .collect();

    let state_declarations = graph
        .state_specs()
        .iter()
        .map(|(layer, spec, handle)| GoldenStateDecl {
            layer: *layer,
            handle_layer: handle.layer(),
            state_kind: format!("{:?}", handle.kind()),
            dim: match spec {
                StateSpec::KvPaged { d, .. } => *d,
                StateSpec::KvLatent { latent, .. } => *latent,
                StateSpec::Recurrent { d, .. } => *d,
                StateSpec::ConvWindow { c, .. } => *c,
            },
            cache: match spec {
                StateSpec::KvPaged { cache, .. } => format!("{cache:?}"),
                StateSpec::KvLatent { cache, .. } => format!("{cache:?}"),
                _ => "none".to_string(),
            },
            retain: match spec {
                StateSpec::KvPaged { retain, .. } => match retain {
                    Retain::All => "all".to_string(),
                    Retain::Window { w } => format!("window({w})"),
                    Retain::SinkWindow { n, w } => {
                        format!("window({w}, sinks={n})")
                    }
                },
                StateSpec::KvLatent { retain, .. } => match retain {
                    Retain::All => "all".to_string(),
                    Retain::Window { w } => format!("window({w})"),
                    Retain::SinkWindow { n, w } => {
                        format!("window({w}, sinks={n})")
                    }
                },
                _ => "none".to_string(),
            },
        })
        .collect();

    let fusion_declarations = graph
        .fusion_decls()
        .iter()
        .map(|f| match f {
            FusionDecl::Qkv { q, k, v } => GoldenFusionDecl {
                kind: "qkv".to_string(),
                tensors: vec![q.clone(), k.clone(), v.clone()],
            },
            FusionDecl::GateUp { gate, up } => GoldenFusionDecl {
                kind: "gate_up".to_string(),
                tensors: vec![gate.clone(), up.clone()],
            },
        })
        .collect();

    let tied_declarations = graph
        .tied_decls()
        .iter()
        .map(|t| GoldenTiedDecl {
            source: t.embed_name.clone(),
            target: t.head_name.clone(),
        })
        .collect();

    GoldenReport {
        architecture: arch.to_string(),
        model_spec: convert_model_spec(spec),
        bound_weights,
        state_declarations,
        fusion_declarations,
        tied_declarations,
    }
}

fn assert_or_update_golden(name: &str, report: &GoldenReport) {
    let json_text =
        serde_json::to_string_pretty(report).expect("serialization must succeed") + "\n";
    // CONVENTIONS.md §4.2: goldens live at workspace-root tests/golden/<crate>/.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = PathBuf::from(manifest_dir)
        .join("../../tests/golden/r9v-models")
        .join(format!("{name}.json"));

    if std::env::var("R9V_UPDATE_GOLDEN").as_deref() == Ok("1") || !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create golden directory");
        }
        std::fs::write(&path, &json_text).expect("failed to write golden file");
    }

    let expected = std::fs::read_to_string(&path).expect("failed to read golden file");
    assert_eq!(
        json_text, expected,
        "Golden JSON mismatch for {name}. Rerun with R9V_UPDATE_GOLDEN=1 to update."
    );
}

// ---------------------------------------------------------------------------
// Synthetic Fixtures for All Eight Architectures
// ---------------------------------------------------------------------------

fn fixture_llama() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "llama");
    meta.insert_u32("llama.block_count", 2);
    meta.insert_u32("llama.embedding_length", 128);
    meta.insert_u32("llama.feed_forward_length", 256);
    meta.insert_u32("llama.attention.head_count", 4);
    meta.insert_u32("llama.attention.head_count_kv", 2);
    meta.insert_u32("llama.attention.key_length", 32);
    meta.insert_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    meta.insert_f32("llama.rope.freq_base", 10000.0);
    meta.insert_u32("llama.vocab_size", 1000);
    meta.insert_u32("tokenizer.ggml.bos_token_id", 1);
    meta.insert_u32("tokenizer.ggml.eos_token_id", 2);
    meta
}

fn fixture_mistral() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "mistral");
    meta.insert_u32("mistral.block_count", 2);
    meta.insert_u32("mistral.embedding_length", 128);
    meta.insert_u32("mistral.feed_forward_length", 256);
    meta.insert_u32("mistral.attention.head_count", 4);
    meta.insert_u32("mistral.attention.head_count_kv", 2);
    meta.insert_u32("mistral.attention.key_length", 32);
    meta.insert_f32("mistral.attention.layer_norm_rms_epsilon", 1e-5);
    meta.insert_u32("mistral.attention.sliding_window", 4096);
    meta.insert_f32("mistral.rope.freq_base", 10000.0);
    meta.insert_u32("mistral.vocab_size", 1000);
    meta.insert_u32("tokenizer.ggml.bos_token_id", 1);
    meta.insert_u32("tokenizer.ggml.eos_token_id", 2);
    meta
}

fn fixture_qwen2() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "qwen2");
    meta.insert_u32("qwen2.block_count", 2);
    meta.insert_u32("qwen2.embedding_length", 128);
    meta.insert_u32("qwen2.feed_forward_length", 256);
    meta.insert_u32("qwen2.attention.head_count", 4);
    meta.insert_u32("qwen2.attention.head_count_kv", 2);
    meta.insert_u32("qwen2.attention.key_length", 32);
    meta.insert_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6);
    meta.insert_f32("qwen2.rope.freq_base", 1000000.0);
    meta.insert_u32("qwen2.vocab_size", 1000);
    meta.insert_u32("tokenizer.ggml.bos_token_id", 1);
    meta.insert_u32("tokenizer.ggml.eos_token_id", 2);
    meta
}

fn fixture_qwen3() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "qwen3");
    meta.insert_u32("qwen3.block_count", 2);
    meta.insert_u32("qwen3.embedding_length", 128);
    meta.insert_u32("qwen3.feed_forward_length", 256);
    meta.insert_u32("qwen3.attention.head_count", 4);
    meta.insert_u32("qwen3.attention.head_count_kv", 2);
    meta.insert_u32("qwen3.attention.key_length", 32);
    meta.insert_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6);
    meta.insert_f32("qwen3.rope.freq_base", 1000000.0);
    meta.insert_u32("qwen3.vocab_size", 1000);
    meta.insert_u32("tokenizer.ggml.bos_token_id", 1);
    meta.insert_u32("tokenizer.ggml.eos_token_id", 2);
    meta
}

fn fixture_gemma2() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "gemma2");
    meta.insert_u32("gemma2.block_count", 4);
    meta.insert_u32("gemma2.embedding_length", 128);
    meta.insert_u32("gemma2.feed_forward_length", 256);
    meta.insert_u32("gemma2.attention.head_count", 4);
    meta.insert_u32("gemma2.attention.head_count_kv", 2);
    meta.insert_u32("gemma2.attention.key_length", 32);
    meta.insert_f32("gemma2.attention.layer_norm_rms_epsilon", 1e-6);
    meta.insert_f32("gemma2.rope.freq_base", 10000.0);
    meta.insert_u32("gemma2.attention.sliding_window", 4096);
    meta.insert_u32("gemma2.vocab_size", 1000);
    meta.insert_u32("tokenizer.ggml.bos_token_id", 2);
    meta.insert_u32("tokenizer.ggml.eos_token_id", 1);
    meta
}

fn fixture_gemma3() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "gemma3");
    meta.insert_u32("gemma3.block_count", 6);
    meta.insert_u32("gemma3.embedding_length", 128);
    meta.insert_u32("gemma3.feed_forward_length", 256);
    meta.insert_u32("gemma3.attention.head_count", 4);
    meta.insert_u32("gemma3.attention.head_count_kv", 2);
    meta.insert_u32("gemma3.attention.key_length", 32);
    meta.insert_f32("gemma3.attention.layer_norm_rms_epsilon", 1e-6);
    meta.insert_f32("gemma3.rope.freq_base", 10000.0);
    meta.insert_u32("gemma3.attention.sliding_window", 1024);
    meta.insert_u32("gemma3.vocab_size", 1000);
    meta.insert_u32("tokenizer.ggml.bos_token_id", 2);
    meta.insert_u32("tokenizer.ggml.eos_token_id", 1);
    meta
}

fn fixture_phi3() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "phi3");
    meta.insert_u32("phi3.block_count", 2);
    meta.insert_u32("phi3.embedding_length", 128);
    meta.insert_u32("phi3.feed_forward_length", 256);
    meta.insert_u32("phi3.attention.head_count", 4);
    meta.insert_u32("phi3.attention.head_count_kv", 2);
    meta.insert_u32("phi3.attention.key_length", 32);
    meta.insert_f32("phi3.attention.layer_norm_rms_epsilon", 1e-5);
    meta.insert_f32("phi3.rope.freq_base", 10000.0);
    meta.insert_u32("phi3.attention.sliding_window", 2048);
    meta.insert_u32("phi3.vocab_size", 1000);
    meta.insert_u32("tokenizer.ggml.bos_token_id", 1);
    meta.insert_u32("tokenizer.ggml.eos_token_id", 2);
    meta
}

fn fixture_olmo2() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "olmo2");
    meta.insert_u32("olmo2.block_count", 2);
    meta.insert_u32("olmo2.embedding_length", 128);
    meta.insert_u32("olmo2.feed_forward_length", 256);
    meta.insert_u32("olmo2.attention.head_count", 4);
    meta.insert_u32("olmo2.attention.head_count_kv", 2);
    meta.insert_u32("olmo2.attention.key_length", 32);
    meta.insert_f32("olmo2.attention.layer_norm_rms_epsilon", 1e-5);
    meta.insert_f32("olmo2.rope.freq_base", 500000.0);
    meta.insert_u32("olmo2.vocab_size", 1000);
    meta.insert_u32("tokenizer.ggml.bos_token_id", 1);
    meta.insert_u32("tokenizer.ggml.eos_token_id", 2);
    meta
}

// ---------------------------------------------------------------------------
// Golden Tests for All 8 Architectures
// ---------------------------------------------------------------------------

#[test]
fn test_golden_llama() {
    let meta = fixture_llama();
    let spec = families::build(&meta).expect("llama model spec must build");
    let builder = Graph::new(IrVersion::CURRENT, "golden-llama");
    let graph = build_model(builder, &spec).expect("llama model graph must build");
    let report = generate_report("llama", &spec, &graph);
    assert_or_update_golden("llama", &report);
}

#[test]
fn test_golden_mistral() {
    let meta = fixture_mistral();
    let spec = families::build(&meta).expect("mistral model spec must build");
    let builder = Graph::new(IrVersion::CURRENT, "golden-mistral");
    let graph = build_model(builder, &spec).expect("mistral model graph must build");
    let report = generate_report("mistral", &spec, &graph);
    assert_or_update_golden("mistral", &report);
}

#[test]
fn test_golden_qwen2() {
    let meta = fixture_qwen2();
    let spec = families::build(&meta).expect("qwen2 model spec must build");
    let builder = Graph::new(IrVersion::CURRENT, "golden-qwen2");
    let graph = build_model(builder, &spec).expect("qwen2 model graph must build");
    let report = generate_report("qwen2", &spec, &graph);
    assert_or_update_golden("qwen2", &report);
}

#[test]
fn test_golden_qwen3() {
    let meta = fixture_qwen3();
    let spec = families::build(&meta).expect("qwen3 model spec must build");
    let builder = Graph::new(IrVersion::CURRENT, "golden-qwen3");
    let graph = build_model(builder, &spec).expect("qwen3 model graph must build");
    let report = generate_report("qwen3", &spec, &graph);
    assert_or_update_golden("qwen3", &report);
}

#[test]
fn test_golden_gemma2() {
    let meta = fixture_gemma2();
    let spec = families::build(&meta).expect("gemma2 model spec must build");
    let builder = Graph::new(IrVersion::CURRENT, "golden-gemma2");
    let graph = build_model(builder, &spec).expect("gemma2 model graph must build");
    let report = generate_report("gemma2", &spec, &graph);
    assert_or_update_golden("gemma2", &report);
}

#[test]
fn test_golden_gemma3() {
    let meta = fixture_gemma3();
    let spec = families::build(&meta).expect("gemma3 model spec must build");
    let builder = Graph::new(IrVersion::CURRENT, "golden-gemma3");
    let graph = build_model(builder, &spec).expect("gemma3 model graph must build");
    let report = generate_report("gemma3", &spec, &graph);
    assert_or_update_golden("gemma3", &report);
}

#[test]
fn test_golden_phi3() {
    let meta = fixture_phi3();
    let spec = families::build(&meta).expect("phi3 model spec must build");
    let builder = Graph::new(IrVersion::CURRENT, "golden-phi3");
    let graph = build_model(builder, &spec).expect("phi3 model graph must build");
    let report = generate_report("phi3", &spec, &graph);
    assert_or_update_golden("phi3", &report);
}

#[test]
fn test_golden_olmo2() {
    let meta = fixture_olmo2();
    let spec = families::build(&meta).expect("olmo2 model spec must build");
    let builder = Graph::new(IrVersion::CURRENT, "golden-olmo2");
    let graph = build_model(builder, &spec).expect("olmo2 model graph must build");
    let report = generate_report("olmo2", &spec, &graph);
    assert_or_update_golden("olmo2", &report);
}

// ---------------------------------------------------------------------------
// Architecture-Specific Structural Assertions
// ---------------------------------------------------------------------------

#[test]
fn test_gemma2_heterogeneous_window_and_sandwich() {
    let meta = fixture_gemma2();
    let spec = families::build(&meta).unwrap();
    assert_eq!(spec.layers.len(), 4);
    assert_eq!(spec.embed_scale, (128.0f32).sqrt());
    assert!(spec.tied_embeddings);
    assert_eq!(spec.final_logit_softcap, Some(30.0));

    // Check alternating sliding window: even local, odd global
    for (i, layer) in spec.layers.iter().enumerate() {
        assert_eq!(layer.norm, NormPlacement::Sandwich);
        assert_eq!(layer.norm_kind.weight_offset, 1.0);
        match &layer.mixer {
            Mixer::Attention {
                window,
                logit_softcap,
                ..
            } => {
                assert_eq!(*logit_softcap, Some(50.0));
                if i % 2 == 0 {
                    assert_eq!(*window, Some(4096), "layer {i} should be local window 4096");
                } else {
                    assert_eq!(*window, None, "layer {i} should be global (None)");
                }
            }
            _ => panic!("expected attention mixer"),
        }
        match &layer.ffn {
            Ffn::Dense { act, gated, .. } => {
                assert_eq!(*act, ActivationKind::GeluTanh);
                assert!(*gated);
            }
            _ => panic!("expected dense ffn"),
        }
    }
}

#[test]
fn test_gemma3_5_to_1_sliding_window_pattern() {
    let meta = fixture_gemma3();
    let spec = families::build(&meta).unwrap();
    assert_eq!(spec.layers.len(), 6);
    assert_eq!(spec.embed_scale, (128.0f32).sqrt());
    assert!(spec.tied_embeddings);

    // 5:1 ratio: layers 0..4 local (1024), layer 5 global (None)
    for (i, layer) in spec.layers.iter().enumerate() {
        assert_eq!(layer.norm, NormPlacement::Sandwich);
        match &layer.mixer {
            Mixer::Attention {
                window, qk_norm, ..
            } => {
                assert!(qk_norm.is_some(), "Gemma 3 must have QK-norm");
                if (i + 1) % 6 == 0 {
                    assert_eq!(*window, None, "layer {i} should be global (None)");
                } else {
                    assert_eq!(*window, Some(1024), "layer {i} should be local window 1024");
                }
            }
            _ => panic!("expected attention mixer"),
        }
    }
}

#[test]
fn test_qwen2_qkv_bias_and_no_qk_norm() {
    let meta = fixture_qwen2();
    let spec = families::build(&meta).unwrap();
    for layer in &spec.layers {
        match &layer.mixer {
            Mixer::Attention {
                qkv_bias,
                o_bias,
                qk_norm,
                ..
            } => {
                assert!(*qkv_bias, "qwen2 must have qkv_bias");
                assert!(!*o_bias, "qwen2 output projection must not have bias");
                assert!(qk_norm.is_none(), "standard qwen2 does not have qk_norm");
            }
            _ => panic!("expected attention mixer"),
        }
    }
}

#[test]
fn test_qwen3_no_qkv_bias_and_qk_norm() {
    let meta = fixture_qwen3();
    let spec = families::build(&meta).unwrap();
    for layer in &spec.layers {
        match &layer.mixer {
            Mixer::Attention {
                qkv_bias, qk_norm, ..
            } => {
                assert!(!*qkv_bias, "qwen3 checkpoints omit QKV bias tensors");
                assert!(qk_norm.is_some(), "qwen3 must have qk_norm");
            }
            _ => panic!("expected attention mixer"),
        }
    }
}

#[test]
fn test_olmo2_post_norm_and_qk_norm() {
    // OLMo 2 uses honest post-normalization: no pre-norm; mixer and
    // FFN read the raw stream; post_attention_norm / post_ffw_norm normalize
    // branch outputs before the residual add.
    let meta = fixture_olmo2();
    let spec = families::build(&meta).unwrap();
    for layer in &spec.layers {
        assert_eq!(layer.norm, NormPlacement::Post);
        assert_eq!(layer.norm_kind.weight_offset, 0.0);
        match &layer.mixer {
            Mixer::Attention {
                qk_norm,
                rope,
                qkv_bias,
                pre_fused,
                ..
            } => {
                assert!(qk_norm.is_some(), "olmo2 must have qk_norm");
                assert_eq!(rope.theta, 500000.0, "olmo2 default theta is 500k");
                assert!(!*qkv_bias, "olmo2 has no qkv bias");
                assert!(!*pre_fused, "olmo2 uses split QKV projections");
            }
            _ => panic!("expected attention mixer"),
        }
    }

    // Graph bindings: post norms bound, pre-norms never bound.
    let builder = Graph::new(IrVersion::CURRENT, "olmo2-post-norm");
    let graph = build_model(builder, &spec).expect("olmo2 graph must build");
    let names: Vec<&str> = graph
        .bound_weights()
        .iter()
        .map(|w| w.name.as_str())
        .collect();
    for i in 0..spec.layers.len() {
        assert!(
            names.contains(&format!("blk.{i}.post_attention_norm.weight").as_str()),
            "missing post_attention_norm for layer {i}"
        );
        assert!(
            names.contains(&format!("blk.{i}.post_ffw_norm.weight").as_str()),
            "missing post_ffw_norm for layer {i}"
        );
        assert!(
            !names.contains(&format!("blk.{i}.attn_norm.weight").as_str()),
            "olmo2 must not bind attn_norm for layer {i}"
        );
        assert!(
            !names.contains(&format!("blk.{i}.ffn_norm.weight").as_str()),
            "olmo2 must not bind ffn_norm for layer {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// Boundary & Adversarial Error Variants
// ---------------------------------------------------------------------------

#[test]
fn test_missing_required_keys_collects_all_problems() {
    let meta = SyntheticGgufMeta::new();
    // Missing general.architecture
    let err = families::build(&meta).unwrap_err();
    assert!(
        matches!(err, ModelsError::MissingMetaKey { ref key, .. } if key == "general.architecture")
    );

    // Provide architecture only
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "llama");
    let err = families::build(&meta).unwrap_err();

    // Must return aggregated Multiple error reporting all missing keys
    match err {
        ModelsError::Multiple { problems } => {
            assert!(
                problems.len() >= 5,
                "expected at least 5 missing key errors, got {}",
                problems.len()
            );
            let missing_keys: Vec<String> = problems
                .iter()
                .filter_map(|p| match p {
                    ModelsError::MissingMetaKey { key, .. } => Some(key.clone()),
                    _ => None,
                })
                .collect();
            assert!(missing_keys.iter().any(|k| k == "llama.block_count"));
            assert!(missing_keys.iter().any(|k| k == "llama.embedding_length"));
            assert!(missing_keys
                .iter()
                .any(|k| k == "llama.feed_forward_length"));
            assert!(missing_keys
                .iter()
                .any(|k| k == "llama.attention.head_count"));
            assert!(missing_keys
                .iter()
                .any(|k| k == "llama.attention.layer_norm_rms_epsilon"));
            assert!(missing_keys.iter().any(|k| k == "llama.vocab_size"));
        }
        other => panic!("expected ModelsError::Multiple, got {other:?}"),
    }
}

#[test]
fn test_unknown_architecture_returns_typed_error() {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "deepseek_v3");
    let err = families::build(&meta).unwrap_err();
    match err {
        ModelsError::UnknownArchitecture { arch, nearest } => {
            assert_eq!(arch, "deepseek_v3");
            assert_eq!(nearest, "llama");
        }
        other => panic!("expected UnknownArchitecture, got {other:?}"),
    }
}

#[test]
fn test_malformed_meta_types_collects_all_mismatches() {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "llama");
    meta.insert_str("llama.block_count", "not_a_number");
    meta.insert_str("llama.embedding_length", "also_string");
    meta.insert_str("llama.feed_forward_length", "string_again");
    meta.insert_str("llama.attention.head_count", "bad_head");
    meta.insert_str("llama.attention.layer_norm_rms_epsilon", "not_float");
    meta.insert_str("llama.vocab_size", "not_u32");

    let err = families::build(&meta).unwrap_err();
    match err {
        ModelsError::Multiple { problems } => {
            assert!(
                problems.len() >= 6,
                "expected at least 6 type mismatch errors, got {}",
                problems.len()
            );
            for p in &problems {
                assert!(
                    matches!(p, ModelsError::MetaTypeMismatch { .. }),
                    "expected MetaTypeMismatch, got {p:?}"
                );
            }
        }
        other => panic!("expected ModelsError::Multiple, got {other:?}"),
    }
}

#[test]
fn test_contradictory_head_counts() {
    let mut meta = fixture_llama();
    // hkv (3) does not divide h (4)
    meta.insert_u32("llama.attention.head_count", 4);
    meta.insert_u32("llama.attention.head_count_kv", 3);
    let err = families::build(&meta).unwrap_err();
    assert!(
        format!("{err:?}").contains("divisible"),
        "expected divisibility error, got {err:?}"
    );

    // hkv (8) exceeds h (4)
    let mut meta = fixture_llama();
    meta.insert_u32("llama.attention.head_count", 4);
    meta.insert_u32("llama.attention.head_count_kv", 8);
    let err = families::build(&meta).unwrap_err();
    assert!(
        format!("{err:?}").contains("cannot exceed"),
        "expected cannot exceed error, got {err:?}"
    );
}

#[test]
fn test_invalid_key_length_not_divisible_by_16() {
    let mut meta = fixture_llama();
    meta.insert_u32("llama.attention.key_length", 24); // not divisible by 16
    let err = families::build(&meta).unwrap_err();
    assert!(
        format!("{err:?}").contains("divisible by 16"),
        "expected divisible by 16 error, got {err:?}"
    );
}

#[test]
fn test_rot_dim_exceeds_d() {
    let mut meta = fixture_llama();
    meta.insert_u32("llama.attention.key_length", 32);
    meta.insert_u32("llama.rope.dimension_count", 64); // rot_dim > d
    let err = families::build(&meta).unwrap_err();
    assert!(
        format!("{err:?}").contains("cannot exceed head dimension"),
        "expected rot_dim cannot exceed d, got {err:?}"
    );
}

#[test]
fn test_rot_dim_odd() {
    let mut meta = fixture_llama();
    meta.insert_u32("llama.rope.dimension_count", 31); // odd
    let err = families::build(&meta).unwrap_err();
    assert!(
        format!("{err:?}").contains("positive and even"),
        "expected even rot_dim, got {err:?}"
    );
}

#[test]
fn test_rope_scaling_linear() {
    let mut meta = fixture_llama();
    meta.insert_str("llama.rope.scaling.type", "linear");
    meta.insert_f32("llama.rope.scaling.factor", 2.0);
    let spec = families::build(&meta).unwrap();
    match &spec.layers[0].mixer {
        Mixer::Attention { rope, .. } => {
            assert_eq!(rope.scaling, RopeScaling::Linear(2.0));
        }
        _ => panic!("expected attention"),
    }
}

#[test]
fn test_rope_scaling_yarn() {
    let mut meta = fixture_llama();
    meta.insert_str("llama.rope.scaling.type", "yarn");
    meta.insert_f32("llama.rope.scaling.factor", 4.0);
    meta.insert_u32("llama.rope.scaling.original_context_length", 4096);
    meta.insert_f32("llama.rope.scaling.yarn_log_mul", 1.25);
    meta.insert_f32("llama.rope.scaling.beta_fast", 32.0);
    meta.insert_f32("llama.rope.scaling.beta_slow", 1.0);
    let spec = families::build(&meta).unwrap();
    match &spec.layers[0].mixer {
        Mixer::Attention { rope, .. } => match rope.scaling {
            RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                orig_ctx,
                mscale,
            } => {
                assert_eq!(factor, 4.0);
                assert_eq!(orig_ctx, 4096);
                assert_eq!(mscale, 1.25);
                assert_eq!(beta_fast, 32.0);
                assert_eq!(beta_slow, 1.0);
            }
            _ => panic!("expected yarn scaling"),
        },
        _ => panic!("expected attention"),
    }
}

#[test]
fn test_rope_scaling_unsupported_type() {
    let mut meta = fixture_llama();
    meta.insert_str("llama.rope.scaling.type", "longrope_v2");
    let err = families::build(&meta).unwrap_err();
    assert!(
        format!("{err:?}").contains("unsupported rope scaling type"),
        "expected unsupported rope scaling, got {err:?}"
    );
}

#[test]
fn test_sinks_without_window_rejected() {
    let mut meta = fixture_llama();
    meta.insert_u32("llama.attention.sinks", 4);
    // llama has window None by default
    let err = families::build(&meta).unwrap_err();
    assert!(
        format!("{err:?}").contains("attention sinks"),
        "expected attention sinks require window, got {err:?}"
    );
}

#[test]
fn test_zero_dimensions_rejected() {
    let mut meta = fixture_llama();
    meta.insert_u32("llama.block_count", 0);
    let err = families::build(&meta).unwrap_err();
    assert!(
        format!("{err:?}").contains("must be > 0"),
        "expected block_count > 0, got {err:?}"
    );
}

#[test]
fn test_tokenizer_tokens_fallback_for_vocab() {
    let mut meta = fixture_llama();
    meta.remove("llama.vocab_size");
    meta.insert_str_array(
        "tokenizer.ggml.tokens",
        (0..500).map(|i| format!("tok_{i}")).collect(),
    );
    let spec = families::build(&meta).unwrap();
    assert_eq!(spec.vocab, 500);
}

#[test]
fn test_explicit_sliding_window_pattern_override() {
    let mut meta = fixture_llama();
    meta.insert_u32("llama.block_count", 3);
    meta.insert_u32("llama.attention.sliding_window", 2048);
    meta.insert_u32_array("llama.attention.sliding_window_pattern", vec![1, 0, 1024]);
    let spec = families::build(&meta).unwrap();
    match &spec.layers[0].mixer {
        Mixer::Attention { window, .. } => assert_eq!(*window, Some(2048)),
        _ => panic!(),
    }
    match &spec.layers[1].mixer {
        Mixer::Attention { window, .. } => assert_eq!(*window, None),
        _ => panic!(),
    }
    match &spec.layers[2].mixer {
        Mixer::Attention { window, .. } => assert_eq!(*window, Some(1024)),
        _ => panic!(),
    }
}

#[test]
fn test_phi3_prefused_qkv_ffn_bindings() {
    // Phi-3 checkpoints pre-fuse QKV (blk.N.attn_qkv) and gate/up
    // (blk.N.ffn_up); the graph must bind the fused weights with exact
    // shapes and never the split projections.
    let meta = fixture_phi3();
    let spec = families::build(&meta).unwrap();
    for layer in &spec.layers {
        match &layer.mixer {
            Mixer::Attention { pre_fused, .. } => assert!(*pre_fused),
            _ => panic!("expected attention mixer"),
        }
        match &layer.ffn {
            Ffn::Dense { pre_fused, .. } => assert!(*pre_fused),
            _ => panic!("expected dense ffn"),
        }
    }
    let builder = Graph::new(IrVersion::CURRENT, "phi3-prefused");
    let graph = build_model(builder, &spec).expect("phi3 graph must build");
    let weights = graph.bound_weights();
    let shape_of = |name: &str| {
        weights
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("missing bound weight {name}"))
            .shape
            .clone()
    };
    // Fixture: dm=128, h=4, hkv=2, d=dv=32 → q=128, k=64, v=64, qkv=256.
    assert_eq!(
        shape_of("blk.0.attn_qkv.weight"),
        vec![Dim::Concrete(256), Dim::Concrete(128)]
    );
    // Fixture: dff=256 → fused gate/up rows 2*256=512.
    assert_eq!(
        shape_of("blk.0.ffn_up.weight"),
        vec![Dim::Concrete(512), Dim::Concrete(128)]
    );
    for w in weights {
        assert!(
            !w.name.ends_with("attn_q.weight")
                && !w.name.ends_with("attn_k.weight")
                && !w.name.ends_with("attn_v.weight")
                && !w.name.ends_with("ffn_gate.weight"),
            "split projection must not be bound for pre-fused phi3: {}",
            w.name
        );
    }
}

#[test]
fn test_phi3_prefused_qkv_bias_binding() {
    // With <arch>.attention.qkv_bias set, the fused bias binds with the
    // full qkv width; the split biases must not appear.
    let mut meta = fixture_phi3();
    meta.insert_bool("phi3.attention.qkv_bias", true);
    let spec = families::build(&meta).unwrap();
    let builder = Graph::new(IrVersion::CURRENT, "phi3-prefused-bias");
    let graph = build_model(builder, &spec).expect("phi3 graph must build");
    let weights = graph.bound_weights();
    let bias = weights
        .iter()
        .find(|w| w.name == "blk.0.attn_qkv.bias")
        .expect("fused qkv bias must bind");
    assert_eq!(bias.shape, vec![Dim::Concrete(256)]);
    assert!(
        weights.iter().all(|w| !w.name.ends_with("attn_q.bias")),
        "split q bias must not bind for pre-fused phi3"
    );
}

#[test]
fn test_sliding_window_pattern_bool_array_musellama_form() {
    // Actual Muse GGUF form: array of bool, e.g. [true, false].
    let mut meta = fixture_llama();
    meta.insert_u32("llama.block_count", 2);
    meta.insert_u32("llama.attention.sliding_window", 2048);
    meta.insert_bool_array("llama.attention.sliding_window_pattern", vec![true, false]);
    let spec = families::build(&meta).unwrap();
    match &spec.layers[0].mixer {
        Mixer::Attention { window, .. } => assert_eq!(*window, Some(2048)),
        _ => panic!(),
    }
    match &spec.layers[1].mixer {
        Mixer::Attention { window, .. } => assert_eq!(*window, None),
        _ => panic!(),
    }
}

#[test]
fn test_sliding_window_pattern_scalar_period() {
    // Scalar u32 form: period N means layers (i % N) < N - 1 are local.
    let mut meta = fixture_llama();
    meta.insert_u32("llama.block_count", 3);
    meta.insert_u32("llama.attention.sliding_window", 2048);
    meta.insert_u32("llama.attention.sliding_window_pattern", 3);
    let spec = families::build(&meta).unwrap();
    for (i, layer) in spec.layers.iter().enumerate() {
        match &layer.mixer {
            Mixer::Attention { window, .. } => {
                if i < 2 {
                    assert_eq!(*window, Some(2048), "layer {i} should be local");
                } else {
                    assert_eq!(*window, None, "layer {i} should be global");
                }
            }
            _ => panic!(),
        }
    }
}

#[test]
fn test_sliding_window_pattern_requires_window() {
    // An explicit pattern without a base sliding window size is refused.
    let mut meta = fixture_llama();
    meta.insert_u32("llama.block_count", 2);
    meta.insert_bool_array("llama.attention.sliding_window_pattern", vec![true, false]);
    let err = families::build(&meta).unwrap_err();
    match err {
        ModelsError::InvalidModelSpec { reason } => assert!(
            reason.contains("requires a base sliding window size"),
            "unexpected reason: {reason}"
        ),
        ModelsError::Multiple { problems } => assert!(problems.iter().any(|p| matches!(
            p,
            ModelsError::InvalidModelSpec { reason }
            if reason.contains("requires a base sliding window size")
        ))),
        other => panic!("expected InvalidModelSpec, got {other:?}"),
    }
}

#[test]
fn test_attn_logit_softcapping_canonical_key_first() {
    // Canonical {arch}.attn_logit_softcapping wins (llama.cpp llama-arch
    // LLM_KV_ATTN_LOGIT_SOFTCAPPING); legacy attention.logit_softcapping
    // stays as fallback.
    let mut meta = fixture_llama();
    meta.insert_f32("llama.attn_logit_softcapping", 40.0);
    meta.insert_f32("llama.attention.logit_softcapping", 20.0);
    let spec = families::build(&meta).unwrap();
    match &spec.layers[0].mixer {
        Mixer::Attention { logit_softcap, .. } => assert_eq!(*logit_softcap, Some(40.0)),
        _ => panic!(),
    }

    let mut meta = fixture_llama();
    meta.insert_f32("llama.attention.logit_softcapping", 20.0);
    let spec = families::build(&meta).unwrap();
    match &spec.layers[0].mixer {
        Mixer::Attention { logit_softcap, .. } => assert_eq!(*logit_softcap, Some(20.0)),
        _ => panic!(),
    }
}

#[test]
fn test_gemma_key_length_required() {
    // Gemma 2/3 refuse without an explicit key_length; other families
    // default to dm / h.
    let mut meta = fixture_gemma2();
    meta.remove("gemma2.attention.key_length");
    let err = families::build(&meta).unwrap_err();
    match err {
        ModelsError::MissingMetaKey { key, .. } => {
            assert_eq!(key, "gemma2.attention.key_length")
        }
        ModelsError::Multiple { problems } => assert!(problems.iter().any(|p| matches!(
            p,
            ModelsError::MissingMetaKey { key, .. }
            if key == "gemma2.attention.key_length"
        ))),
        other => panic!("expected MissingMetaKey, got {other:?}"),
    }

    let mut meta = fixture_llama();
    meta.remove("llama.attention.key_length");
    let spec = families::build(&meta).expect("llama defaults key_length to dm / h");
    match &spec.layers[0].mixer {
        Mixer::Attention { d, .. } => assert_eq!(*d, 32),
        _ => panic!(),
    }
}

#[test]
fn test_accepted_keys_table_covers_all_architectures() {
    assert!(!families::llama::ACCEPTED_METADATA_KEYS.is_empty());
    let keys = families::llama::ACCEPTED_METADATA_KEYS;
    assert!(keys.iter().any(|k| k.semantic == "architecture"));
    assert!(keys.iter().any(|k| k.semantic == "layer_count"));
    assert!(keys.iter().any(|k| k.semantic == "embedding_length"));
    assert!(keys.iter().any(|k| k.semantic == "feed_forward_length"));
    assert!(keys.iter().any(|k| k.semantic == "head_count"));
    assert!(keys.iter().any(|k| k.semantic == "layer_norm_rms_epsilon"));
    assert!(keys.iter().any(|k| k.semantic == "sliding_window"));
    assert!(keys.iter().any(|k| k.semantic == "qk_norm"));
    assert!(keys.iter().any(|k| k.semantic == "qkv_bias"));
    assert!(keys.iter().any(|k| k.semantic == "tie_word_embeddings"));
    assert!(keys.iter().any(|k| k.semantic == "embed_scale"));
}
