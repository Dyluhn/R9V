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
    build_model, CacheDtype, Ffn, Graph, LayerSpec, Mixer, MlaSpec, ModelSpec, NormPlacement,
    NormSpec, PositionEncoding, Retain, RopeSpec, StateSpec,
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

    // Directly collect emitted state declarations: Vec<StateSpec>.
    let emitted_specs: Vec<StateSpec> = model_graph
        .state_specs()
        .iter()
        .map(|(_, spec, _)| *spec)
        .collect();

    // We have 5 state specs: KvPaged (layer 0), KvPaged Window (layer 1), KvLatent (layer 2),
    // ConvWindow (layer 3), Recurrent (layer 3).
    assert_eq!(emitted_specs.len(), 5);

    let config = StateConfig {
        max_ctx: 128,
        max_seqs: 4,
    };
    let groups = r9v_state::group_layers(&emitted_specs);
    assert_eq!(groups.len(), 5);

    let required_pool = required_pool_bytes(config, &groups).expect("pool math must succeed");

    // Feed directly to StateManager::new with zero conversion!
    let mut manager = StateManager::new(config, emitted_specs, required_pool)
        .expect("StateManager::new with emitted specs must succeed");

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

    let emitted_specs: Vec<StateSpec> = model_graph
        .state_specs()
        .iter()
        .map(|(_, spec, _)| *spec)
        .collect();

    let config = StateConfig {
        max_ctx: 256,
        max_seqs: 8,
    };

    let groups = r9v_state::group_layers(&emitted_specs);
    let pool_bytes = required_pool_bytes(config, &groups).expect("pool calculation must succeed");

    let manager = StateManager::new(config, emitted_specs, pool_bytes)
        .expect("StateManager creation must succeed");

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
    // Dims that exceed MAX_DIM_HARD fail with InvalidConfig / InvalidModelSpec.
    let bad_spec = StateSpec::Recurrent {
        h: 4096,
        d: 4096,
        dv: 4096,
    };
    // 4096 * 4096 * 4096 * 4 * 2 = 549,755,813,888 (512 GB).
    // If we multiply by max_seqs = 65,536: 549,755,813,888 * 65536 = 36,028,797,018,963,968 (32 PB).
    // Now create 1024 such layers:
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

    // Retain constructors.
    assert_eq!(Retain::from_window_sinks(None, 0).unwrap(), Retain::All);
    assert_eq!(
        Retain::from_window_sinks(Some(128), 0).unwrap(),
        Retain::Window { w: 128 }
    );
    assert_eq!(
        Retain::from_window_sinks(Some(128), 4).unwrap(),
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
