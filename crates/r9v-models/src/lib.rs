// SPDX-License-Identifier: Apache-2.0
//! R9V model architecture builder and model families (Spec 8, Spec 14 §2; card A1.3).
//!
//! This crate owns the sealed [`GraphBuilder`], model and layer specifications ([`ModelSpec`],
//! [`LayerSpec`]), the generic transformer layer builder emitting Spec 8 §3.1 op sequences,
//! fusion declarations ([`FusionDecl`]), tied embeddings ([`TiedDecl`]), model summaries ([`ModelSummary`]),
//! and the typed [`GgufMeta`] lookup trait without container dependencies.
//!
//! Repository standards: `CONVENTIONS.md`; engineering bar: `.agents/skills/r9v-engineering-standards`.

pub mod builder;
pub mod error;
pub mod generic;
pub mod meta;
pub mod spec;
pub mod summary;

// DECISION(A1.15): r9v-models depends downward on both r9v-state and r9v-ir, preserving Spec 14 §2 crate layering; rejected duplicating state types or bypassing r9v-state. Spec 14 §2, card A1.15.
pub use builder::{
    BoundWeight, FusionDecl, Graph, GraphBuilder, ModelGraph, SchemeClass, SealedGraphBuilder,
    SubgraphCapture, TiedDecl, Value, WeightRole,
};
pub use error::ModelsError;
pub use generic::{build_ffn, build_layer, build_mixer, build_model, build_mtp_subgraph};
pub use meta::{GgufMeta, MetaValue, SyntheticGgufMeta};
pub use spec::{
    group_layer_specs, group_layers, CacheDtype, Ffn, LayerSpec, Mixer, MlaSpec, ModelSpec,
    MoeGroupSpec, MoeSharedSpec, MtpSource, MtpSpec, NgramSpec, NormPlacement, NormSpec,
    PositionEncoding, Retain, RopeSpec, StateDecl, StateSpec, MAX_ATTENTION_HEADS, MAX_EXPERTS,
    MAX_FEATURE_DIM, MAX_KV_HEADS, MAX_MODEL_LAYERS, MAX_MTP_HEADS, MAX_MTP_LAYERS_PER_HEAD,
    MAX_NGRAM_HEADS, MAX_NGRAM_TABLE_ENTRIES, MAX_VOCAB_SIZE, MAX_WINDOW,
};
pub use summary::{ExpertSummary, LayerSummary, MixerKind, ModelSummary, SchemeKey};
