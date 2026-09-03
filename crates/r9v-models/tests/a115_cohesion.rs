// SPDX-License-Identifier: Apache-2.0
//! Exhaustive model-to-state cohesion tests for Card A1.15 (Spec 1, 3, 8, 14).
//!
//! Proves:
//! - Emitted state declarations from `r9v_models::build_model` feed directly into
//!   `r9v_state::StateManager::new` and `group_layers` with zero ad-hoc conversion.
//! - ModelSummary totals equal StateManager budget totals for mixed synthetic models.
//! - Checked overflow arithmetic prevents divergence.
//! - Deterministic grouping and nonaliasing of inequivalent state specifications.
//! - Re-export and backwards compatibility for `CacheDtype`, `Retain`, and `StateSpec`.

use r9v_ir::op::{ActivationKind, LinearAttnKind, RopeScaling, RopeStyle};
use r9v_ir::version::IrVersion;
use r9v_models::{
    build_model, group_layer_specs, group_layers, CacheDtype, Ffn, Graph, LayerSpec, LayerSummary,
    Mixer, MixerKind, MlaSpec, ModelSpec, ModelSummary, ModelsError, NormPlacement, NormSpec,
    PositionEncoding, Retain, RopeSpec, SchemeKey, StateSpec,
};
use r9v_state::{required_pool_bytes, StateConfig, StateError, StateManager};

fn make_rope() -> RopeSpec {
    RopeSpec {
        theta: 10000.0,
        rot_dim: 32,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
    }
}

/// Builds a mixed synthetic model containing all four state kinds:
/// Layer 0: standard paged KV (KvPaged)
/// Layer 1: sliding window paged KV (KvPaged with Window)
/// Layer 2: MLA latent attention (KvLatent)
/// Layer 3: Linear attention with convolution (ConvWindow + Recurrent)
fn make_mixed_model() -> ModelSpec {
    let rope = make_rope();

    let l0 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 8,
            hkv: 4,
            d: 32,
            dv: 32,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: rope.clone(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4M3,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    let l1 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 8,
            hkv: 4,
            d: 32,
            dv: 32,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: rope.clone(),
            window: Some(64),
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4M3,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    let l2 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 8,
            hkv: 1,
            d: 32,
            dv: 32,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: rope.clone(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: Some(MlaSpec {
                q_lora_rank: 64,
                kv_lora_rank: 64,
                qk_nope_dim: 32,
                qk_rope_dim: 32,
                v_dim: 32,
            }),
            cache: CacheDtype::E4m3,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    let l3 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::LinearAttention {
            kind: LinearAttnKind::GatedDeltaNet,
            h: 4,
            d: 16,
            dv: 16,
            conv: Some(4),
            gate_act: ActivationKind::Silu,
            output_norm: None,
            output_gate: false,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    ModelSpec {
        dm: 128,
        layers: vec![l0, l1, l2, l3],
        vocab: 1000,
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

/// Proves emitted state declarations from `build_model` feed directly into `StateManager::new`
/// with no conversion, and can drive state allocation and `BatchMeta` generation.
#[test]
fn test_a13_to_state_manager_direct_flow() {
    let model = make_mixed_model();
    let builder = Graph::new(IrVersion::CURRENT, "cohesion-flow-test");
    let model_graph = build_model(builder, &model).expect("model must build");

    // Feed emitted state declarations directly into group_layers:
    // We have 5 state specs: KvPaged (layer 0), KvPaged Window (layer 1), KvLatent (layer 2),
    // ConvWindow (layer 3), Recurrent (layer 3).
    assert_eq!(model_graph.state_specs().len(), 5);

    let config = StateConfig {
        max_ctx: 128,
        max_seqs: 4,
    };
    let decls = model_graph.state_declarations();
    let groups = group_layer_specs(&decls).expect("group_layer_specs must succeed");
    assert_eq!(groups.len(), 5);

    // True declaring model layer indices are retained:
    assert_eq!(groups[0].layers, vec![0]); // KvPaged dense
    assert_eq!(groups[1].layers, vec![1]); // KvPaged windowed
    assert_eq!(groups[2].layers, vec![2]); // KvLatent
    assert_eq!(groups[3].layers, vec![3]); // ConvWindow (layer 3)
    assert_eq!(groups[4].layers, vec![3]); // Recurrent (layer 3)

    let required_pool = required_pool_bytes(config, &groups).expect("pool math must succeed");

    // Feed directly to StateManager::new_with_declarations with zero conversion or copied fields!
    let mut manager = StateManager::new_with_declarations(config, decls, required_pool)
        .expect("StateManager::new_with_declarations with emitted decls must succeed");

    // Perform full lifecycle with StateManager using the emitted specs.
    let (s0, _) = manager.new_seq(&[]).expect("new_seq must succeed");
    let (s1, _) = manager.new_seq(&[]).expect("new_seq must succeed");

    manager.reserve(s0, 32).expect("reserve must succeed");
    manager.reserve(s1, 16).expect("reserve must succeed");

    let meta = manager
        .batch_meta(&[s0, s1], &[32, 16])
        .expect("batch_meta must succeed");

    assert_eq!(meta.num_groups(), 5);
    assert_eq!(meta.num_seqs(), 2);
    assert_eq!(meta.total_tokens(), 48);
    assert_eq!(meta.max_blocks(), 4);
    assert_eq!(meta.seq_ids(), &[s0.as_u64() as u32, s1.as_u64() as u32]);
}

/// Proves ModelSummary totals equal StateManager budget totals for a mixed synthetic model.
#[test]
fn test_model_summary_totals_equal_state_manager_budget() {
    let model = make_mixed_model();
    let builder = Graph::new(IrVersion::CURRENT, "cohesion-summary-budget");
    let model_graph = build_model(builder, &model).expect("model must build");

    let summary = model_graph.summary().expect("summary must compute");

    let config = StateConfig {
        max_ctx: 256,
        max_seqs: 8,
    };

    // Feed state declarations directly to group_layer_specs and StateManager::new_with_declarations!
    let decls = model_graph.state_declarations();
    let groups = group_layer_specs(&decls).expect("group_layer_specs must succeed");
    let pool_bytes = required_pool_bytes(config, &groups).expect("pool calculation must succeed");

    let manager = StateManager::new_with_declarations(config, decls, pool_bytes)
        .expect("StateManager creation must succeed");

    // Also verify legacy constructor produces identical budget:
    let legacy_specs: Vec<StateSpec> = model_graph
        .state_specs()
        .iter()
        .map(|(_, s, _)| *s)
        .collect();
    let legacy_groups = group_layers(&legacy_specs).expect("group_layers must succeed");
    assert_eq!(legacy_groups.len(), groups.len());
    let legacy_mgr =
        StateManager::new(config, legacy_specs, pool_bytes).expect("legacy StateManager::new");
    assert_eq!(
        legacy_mgr.budget().fixed_bytes_total,
        manager.budget().fixed_bytes_total
    );
    assert_eq!(
        legacy_mgr.budget().pool_bytes_total,
        manager.budget().pool_bytes_total
    );

    let budget = manager.budget();

    // 1. Total per-token state memory matches between ModelSummary and StateManager.
    let summary_total_per_token = summary
        .total_state_per_token_bytes()
        .expect("token bytes total");
    let mut manager_total_per_token = 0u64;
    for g in &groups {
        manager_total_per_token += g.per_token_bytes().expect("group per-token bytes");
    }
    assert_eq!(
        summary_total_per_token, manager_total_per_token,
        "ModelSummary total per-token bytes must equal StateManager per-token bytes"
    );

    // Full-context paged arena bytes: summary per-token bytes * max_ctx.
    let expected_paged_arena = summary_total_per_token * (config.max_ctx as u64);
    let mut actual_paged_arena = 0u64;
    for gb in &budget.groups {
        actual_paged_arena += (gb.total_blocks as u64) * gb.block_bytes;
    }
    assert_eq!(
        expected_paged_arena, actual_paged_arena,
        "Full context paged arena pool must equal summary per-token * max_ctx"
    );

    // 2. Total per-sequence state memory matches between ModelSummary and StateManager.
    let summary_total_per_seq = summary
        .total_state_per_seq_bytes()
        .expect("seq bytes total");
    let mut manager_total_per_seq = 0u64;
    for g in &groups {
        manager_total_per_seq += g.slots_bytes_per_seq().expect("group slot bytes");
    }
    assert_eq!(
        summary_total_per_seq, manager_total_per_seq,
        "ModelSummary total per-sequence bytes must equal StateManager per-sequence bytes"
    );

    // Fixed arena total in budget must equal summary per-seq * max_seqs.
    let expected_fixed_arena = summary_total_per_seq * (config.max_seqs as u64);
    assert_eq!(
        budget.fixed_bytes_total, expected_fixed_arena,
        "StateManager fixed_bytes_total must equal summary per-seq * max_seqs"
    );
}

/// Proves checked overflow prevents silent wrap or panic in both ModelSummary and StateManager.
#[test]
fn test_summary_and_budget_checked_overflow() {
    use std::collections::BTreeMap;

    // 1. LayerSummary weight bytes overflow oracle:
    let mut weight_map = BTreeMap::new();
    weight_map.insert(SchemeKey::None, u64::MAX - 5);
    weight_map.insert(SchemeKey::PerRow, 10);
    let layer_weight_overflow = LayerSummary {
        weight_bytes_by_scheme: weight_map,
        state_per_token_bytes: 0,
        state_per_seq_bytes: 0,
        experts: None,
        mixer_kind: None,
    };
    let err_weight = layer_weight_overflow.total_weight_bytes().unwrap_err();
    assert!(
        matches!(
            err_weight,
            ModelsError::ArithmeticOverflow { ref context, .. } if context == "LayerSummary::total_weight_bytes"
        ),
        "expected ArithmeticOverflow in LayerSummary, got {err_weight:?}"
    );

    // 2. ModelSummary state per-token and per-seq overflow oracle:
    let layer1 = LayerSummary {
        weight_bytes_by_scheme: BTreeMap::new(),
        state_per_token_bytes: u64::MAX - 10,
        state_per_seq_bytes: u64::MAX - 20,
        experts: None,
        mixer_kind: Some(MixerKind::Attention),
    };
    let layer2 = LayerSummary {
        weight_bytes_by_scheme: BTreeMap::new(),
        state_per_token_bytes: 20,
        state_per_seq_bytes: 30,
        experts: None,
        mixer_kind: Some(MixerKind::Attention),
    };
    let summary = ModelSummary {
        layers: vec![layer1, layer2],
        embed_bytes: 0,
        head_bytes: 0,
        vocab: 1000,
        dm: 128,
        hkv: 4,
        tp_divisors: vec![1, 2, 4],
        ngram_table_bytes: 0,
        mtp: false,
        export_hidden: false,
    };
    let err_token = summary.total_state_per_token_bytes().unwrap_err();
    assert!(
        matches!(
            err_token,
            ModelsError::ArithmeticOverflow { ref context, .. } if context == "ModelSummary::total_state_per_token_bytes"
        ),
        "expected ArithmeticOverflow in ModelSummary per-token, got {err_token:?}"
    );
    let err_seq = summary.total_state_per_seq_bytes().unwrap_err();
    assert!(
        matches!(
            err_seq,
            ModelsError::ArithmeticOverflow { ref context, .. } if context == "ModelSummary::total_state_per_seq_bytes"
        ),
        "expected ArithmeticOverflow in ModelSummary per-seq, got {err_seq:?}"
    );

    // 3. StateManager required_pool_bytes overflow oracle:
    let bad_spec = StateSpec::Recurrent {
        h: 4096,
        d: 4096,
        dv: 4096,
    };
    let huge_group = r9v_state::LayerGroup {
        index: 0,
        spec: bad_spec,
        layers: (0..1024).collect(),
    };
    let cfg = StateConfig {
        max_ctx: 1024,
        max_seqs: 65_536,
    };
    let err = required_pool_bytes(cfg, &[huge_group]).unwrap_err();
    assert!(
        matches!(err, StateError::Overflow { .. }),
        "huge fixed pool must fail with typed Overflow: {err:?}"
    );
}

/// Proves backwards compatibility and alias equivalences.
#[test]
fn test_migration_and_reexport_compatibility() {
    // Both spellings of E4M3 compare equal and refer to the same discriminant.
    assert_eq!(CacheDtype::E4m3, CacheDtype::E4M3);

    // Retain constructors and migration paths.
    assert_eq!(Retain::from_window_sinks(None, 0).unwrap(), Retain::All);
    assert_eq!(
        Retain::from_window_sinks(Some(128), 0).unwrap(),
        Retain::Window { w: 128 }
    );
    assert_eq!(
        Retain::from_window_sinks(Some(128), 4).unwrap(),
        Retain::SinkWindow { n: 4, w: 128 }
    );
    assert_eq!(Retain::sliding_window(128), Retain::Window { w: 128 });
    assert_eq!(
        Retain::sink_and_window(4, 128),
        Retain::SinkWindow { n: 4, w: 128 }
    );

    // StateSpec methods are available through r9v_models imports.
    let spec = StateSpec::ConvWindow { c: 256, w: 4 };
    assert_eq!(spec.slot_bytes().unwrap(), 1536);
    assert_eq!(spec.per_seq_bytes().unwrap(), 3072);
    assert_eq!(spec.state_per_seq_bytes().unwrap(), 3072);
    assert_eq!(spec.per_token_bytes().unwrap(), 0);
    assert_eq!(spec.state_per_token_bytes().unwrap(), 0);
}

/// Proves that multiple hybrid layers retain their true declaring model layer indices
/// and that ModelSummary totals match StateManager budget totals exactly.
#[test]
fn test_mixed_hybrid_multiple_layers_grouping_and_pool_agreement() {
    let rope = make_rope();
    let l0 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 8,
            hkv: 4,
            d: 32,
            dv: 32,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: rope.clone(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::F16,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    let l1 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 8,
            hkv: 1,
            d: 32,
            dv: 32,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: rope.clone(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: Some(MlaSpec {
                q_lora_rank: 64,
                kv_lora_rank: 64,
                qk_nope_dim: 32,
                qk_rope_dim: 32,
                v_dim: 32,
            }),
            cache: CacheDtype::F16,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    // Layer 2: Hybrid linear attention (ConvWindow + Recurrent)
    let l2 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::LinearAttention {
            kind: LinearAttnKind::GatedDeltaNet,
            h: 4,
            d: 16,
            dv: 16,
            conv: Some(4),
            gate_act: ActivationKind::Silu,
            output_norm: None,
            output_gate: false,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    // Layer 3: Second hybrid linear attention (ConvWindow + Recurrent)
    let l3 = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::LinearAttention {
            kind: LinearAttnKind::GatedDeltaNet,
            h: 4,
            d: 16,
            dv: 16,
            conv: Some(4),
            gate_act: ActivationKind::Silu,
            output_norm: None,
            output_gate: false,
        },
        ffn: Ffn::None,
        residual_scale: 1.0,
    };

    let model = ModelSpec {
        dm: 128,
        layers: vec![l0, l1, l2, l3],
        vocab: 1000,
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
    };

    let builder = Graph::new(IrVersion::CURRENT, "mixed-hybrid-multi");
    let model_graph = build_model(builder, &model).expect("model must build");

    let decls = model_graph.state_declarations();
    let groups = group_layer_specs(&decls).expect("group_layer_specs must succeed");
    // 4 distinct specs: KvPaged (layer 0), KvLatent (layer 1), ConvWindow (layers 2, 3), Recurrent (layers 2, 3).
    assert_eq!(groups.len(), 4);
    assert_eq!(groups[0].layers, vec![0]);
    assert_eq!(groups[1].layers, vec![1]);
    assert_eq!(groups[2].layers, vec![2, 3]); // ConvWindow retains both true layers 2 and 3!
    assert_eq!(groups[3].layers, vec![2, 3]); // Recurrent retains both true layers 2 and 3!

    // Verify state declarations method produces equivalent declarations:
    assert_eq!(decls.len(), 6);
    assert_eq!(decls[2].layer, 2);
    assert_eq!(decls[3].layer, 2);
    assert_eq!(decls[4].layer, 3);
    assert_eq!(decls[5].layer, 3);

    let summary = model_graph.summary().expect("summary must compute");
    let config = StateConfig {
        max_ctx: 128,
        max_seqs: 4,
    };
    let pool_bytes = required_pool_bytes(config, &groups).expect("pool math must succeed");
    let manager = StateManager::new_with_declarations(config, decls, pool_bytes)
        .expect("StateManager creation must succeed");

    let summary_per_token = summary.total_state_per_token_bytes().unwrap();
    let manager_per_token: u64 = groups.iter().map(|g| g.per_token_bytes().unwrap()).sum();
    assert_eq!(summary_per_token, manager_per_token);

    let summary_per_seq = summary.total_state_per_seq_bytes().unwrap();
    let manager_per_seq: u64 = groups
        .iter()
        .map(|g| g.slots_bytes_per_seq().unwrap())
        .sum();
    assert_eq!(summary_per_seq, manager_per_seq);
    assert_eq!(manager.budget().fixed_bytes_total, summary_per_seq * 4);
}

/// Proves exact byte accounting for CacheDtype F16, E4M3, and I8.
#[test]
fn test_cache_dtype_scale_bytes_accounting() {
    let paged_f16 = StateSpec::KvPaged {
        hkv: 4,
        d: 64,
        dv: 64,
        cache: CacheDtype::F16,
        retain: Retain::All,
    };
    // (64 + 64) * 2 * 4 = 1024 bytes (0 scale bytes added)
    assert_eq!(paged_f16.per_token_bytes().unwrap(), 1024);

    let paged_e4m3 = StateSpec::KvPaged {
        hkv: 4,
        d: 64,
        dv: 64,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    };
    // ((64 + 64) * 1 + 4) * 4 = 528 bytes (exact 4 scale bytes per head)
    assert_eq!(paged_e4m3.per_token_bytes().unwrap(), 528);

    let paged_i8 = StateSpec::KvPaged {
        hkv: 4,
        d: 64,
        dv: 64,
        cache: CacheDtype::I8,
        retain: Retain::All,
    };
    // ((64 + 64) * 1 + 4) * 4 = 528 bytes
    assert_eq!(paged_i8.per_token_bytes().unwrap(), 528);

    let latent_f16 = StateSpec::KvLatent {
        latent: 128,
        rope: 32,
        cache: CacheDtype::F16,
        retain: Retain::All,
    };
    // 128 * 2 + 32 * 2 = 320 bytes (0 scale bytes added)
    assert_eq!(latent_f16.per_token_bytes().unwrap(), 320);

    let latent_e4m3 = StateSpec::KvLatent {
        latent: 128,
        rope: 32,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    };
    // 128 * 1 + 2 + 32 * 2 = 194 bytes (exact 2 scale bytes added)
    assert_eq!(latent_e4m3.per_token_bytes().unwrap(), 194);

    let latent_i8 = StateSpec::KvLatent {
        latent: 128,
        rope: 32,
        cache: CacheDtype::I8,
        retain: Retain::All,
    };
    // 128 * 1 + 2 + 32 * 2 = 194 bytes
    assert_eq!(latent_i8.per_token_bytes().unwrap(), 194);
}
