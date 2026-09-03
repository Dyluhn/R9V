// SPDX-License-Identifier: Apache-2.0
//! API-shape and trait boundary tests for `r9v-models` (Spec 8; card A1.3; CONVENTIONS.md §4.1).

use r9v_ir::version::IrVersion;
use r9v_models::families::llama::AcceptedKeyDef;
use r9v_models::{
    BoundWeight, CacheDtype, ExpertSummary, Ffn, FusionDecl, GgufMeta, Graph, GraphBuilder,
    LayerSpec, LayerSummary, MetaValue, Mixer, MixerKind, MlaSpec, ModelGraph, ModelSpec,
    ModelSummary, ModelsError, MoeGroupSpec, MoeSharedSpec, MtpSource, MtpSpec, NgramSpec,
    NormPlacement, NormSpec, PositionEncoding, Retain, RopeSpec, SchemeClass, SchemeKey,
    SealedGraphBuilder, StateSpec, SubgraphCapture, SyntheticGgufMeta, TiedDecl, Value, WeightRole,
};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn test_all_public_types_send_sync() {
    assert_send::<GraphBuilder>();
    assert_sync::<GraphBuilder>();

    assert_send::<ModelGraph>();
    assert_sync::<ModelGraph>();

    assert_send::<Value>();
    assert_sync::<Value>();

    assert_send::<SubgraphCapture>();
    assert_sync::<SubgraphCapture>();

    assert_send::<BoundWeight>();
    assert_sync::<BoundWeight>();

    assert_send::<FusionDecl>();
    assert_sync::<FusionDecl>();

    assert_send::<TiedDecl>();
    assert_sync::<TiedDecl>();

    assert_send::<WeightRole>();
    assert_sync::<WeightRole>();

    assert_send::<SchemeClass>();
    assert_sync::<SchemeClass>();

    assert_send::<NormPlacement>();
    assert_sync::<NormPlacement>();

    assert_send::<NormSpec>();
    assert_sync::<NormSpec>();

    assert_send::<RopeSpec>();
    assert_sync::<RopeSpec>();

    assert_send::<MlaSpec>();
    assert_sync::<MlaSpec>();

    assert_send::<CacheDtype>();
    assert_sync::<CacheDtype>();

    assert_send::<Retain>();
    assert_sync::<Retain>();

    assert_send::<StateSpec>();
    assert_sync::<StateSpec>();

    assert_send::<Mixer>();
    assert_sync::<Mixer>();

    assert_send::<Ffn>();
    assert_sync::<Ffn>();

    assert_send::<MoeGroupSpec>();
    assert_sync::<MoeGroupSpec>();

    assert_send::<MoeSharedSpec>();
    assert_sync::<MoeSharedSpec>();

    assert_send::<LayerSpec>();
    assert_sync::<LayerSpec>();

    assert_send::<PositionEncoding>();
    assert_sync::<PositionEncoding>();

    assert_send::<NgramSpec>();
    assert_sync::<NgramSpec>();

    assert_send::<MtpSource>();
    assert_sync::<MtpSource>();

    assert_send::<MtpSpec>();
    assert_sync::<MtpSpec>();

    assert_send::<ModelSpec>();
    assert_sync::<ModelSpec>();

    assert_send::<SchemeKey>();
    assert_sync::<SchemeKey>();

    assert_send::<MixerKind>();
    assert_sync::<MixerKind>();

    assert_send::<ExpertSummary>();
    assert_sync::<ExpertSummary>();

    assert_send::<LayerSummary>();
    assert_sync::<LayerSummary>();

    assert_send::<ModelSummary>();
    assert_sync::<ModelSummary>();

    assert_send::<ModelsError>();
    assert_sync::<ModelsError>();

    assert_send::<SyntheticGgufMeta>();
    assert_sync::<SyntheticGgufMeta>();

    assert_send::<MetaValue>();
    assert_sync::<MetaValue>();

    assert_send::<AcceptedKeyDef>();
    assert_sync::<AcceptedKeyDef>();
}

#[test]
fn test_family_registry_api() {
    use r9v_models::families::{
        find_family, is_supported_architecture, nearest_family, supported_architectures,
    };

    let archs = supported_architectures();
    assert_eq!(archs.len(), 8);
    for arch in archs {
        assert!(is_supported_architecture(arch));
        assert_eq!(find_family(arch), Some("llama"));
    }

    assert!(!is_supported_architecture("gpt2"));
    assert_eq!(find_family("gpt2"), None);
    assert_eq!(nearest_family("gpt2"), "llama");
}

#[test]
fn test_sealed_builder_trait_implemented() {
    fn check_sealed<T: SealedGraphBuilder>() {}
    check_sealed::<GraphBuilder>();
}

#[test]
fn test_synthetic_gguf_meta_typed_access() {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "llama");
    meta.insert_u32("llama.context_length", 8192);
    meta.insert_u64("llama.embedding_length", 4096);
    meta.insert_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    meta.insert_bool("llama.rope.dimension_count", true);
    meta.insert_str_array(
        "tokenizer.ggml.tokens",
        vec!["<s>".to_string(), "</s>".to_string()],
    );
    meta.insert_u32_array("tokenizer.ggml.merges", vec![10, 20, 30]);

    assert!(meta.has("general.architecture"));
    assert!(!meta.has("nonexistent"));

    assert_eq!(meta.str("general.architecture").unwrap(), "llama");
    assert_eq!(meta.u32("llama.context_length").unwrap(), 8192);
    assert_eq!(meta.u64("llama.embedding_length").unwrap(), 4096);
    assert!((meta.f32("llama.attention.layer_norm_rms_epsilon").unwrap() - 1e-5).abs() < 1e-7);
    assert!(meta.bool("llama.rope.dimension_count").unwrap());
    assert_eq!(
        meta.str_array("tokenizer.ggml.tokens").unwrap(),
        vec!["<s>", "</s>"]
    );
    assert_eq!(
        meta.u32_array("tokenizer.ggml.merges").unwrap(),
        vec![10, 20, 30]
    );

    // Optional getters
    assert_eq!(meta.get_str("general.architecture").unwrap(), Some("llama"));
    assert_eq!(meta.get_str("nonexistent").unwrap(), None);
    assert_eq!(meta.get_u32("llama.context_length").unwrap(), Some(8192));
    assert_eq!(meta.get_u32("nonexistent").unwrap(), None);

    // Missing key error
    let err = meta.str("missing.key").unwrap_err();
    assert!(matches!(err, ModelsError::MissingMetaKey { .. }));

    // Type mismatch error
    let err = meta.u32("general.architecture").unwrap_err();
    assert!(matches!(err, ModelsError::MetaTypeMismatch { .. }));
}

#[test]
fn test_builder_creation_via_graph_factory() {
    let builder = Graph::new(IrVersion::CURRENT, "test-model");
    assert_eq!(builder.ir_version(), IrVersion::CURRENT);
    assert_eq!(builder.model_id(), "test-model");
}
