// SPDX-License-Identifier: Apache-2.0
//! Scheduler-registry cohesion: every op static the A3.9 step program injects
//! round-trips through current registry hashing and resolution (A3.API).
//!
//! Each test builds the exact canonical static the scheduler fixtures inject,
//! then proves it validates, pairs with its `OpId`, hashes deterministically,
//! and resolves against a generic-T1 registry.

use r9v_ir::RngAlgorithm;
use r9v_ir::{AttentionMask, DType, Epilogue, LayoutId, NormAxis, NormKind, QuantScheme};
use r9v_registry::{
    static_hash, ArchName, AttentionStatic, BundleManifest, ElementwiseParams, ElementwiseStatic,
    LaunchGeometry, ManifestVariantEntry, MatmulStatic, NormStatic, OpId, OpStatic, Registry,
    RegistryConfig, SampleStatic, SamplingStatic, Tier, VariantHash,
};

fn cohesion_registry(arch: &ArchName) -> Registry {
    let mut manifest = BundleManifest::new(1, vec![arch.clone()]);
    for (idx, &op) in [OpId::Norm, OpId::Attention, OpId::Matmul, OpId::Sample]
        .iter()
        .enumerate()
    {
        manifest.insert_variant(
            VariantHash::new(0x3000_0000_0000_0000 + (idx as u64) + 1),
            ManifestVariantEntry {
                arch: arch.clone(),
                file: format!("reference/{}.co", op.as_str()),
                tier: Tier::T1,
                entry_symbol: format!("t1_{}", op.as_str()),
                launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
                workspace_bytes: 4096,
                static_bytes: 8192,
                static_flops: 16384,
                op: Some(op),
                static_hash: None,
                validated: true,
                validated_on: Some("ref".to_owned()),
            },
        );
    }
    let mut registry = Registry::new(RegistryConfig {
        gen_version: 1,
        allow_jit: false,
        tune_budget_ms: 2000,
        allow_nondeterministic: false,
    });
    registry
        .set_manifest(manifest, None)
        .expect("cohesion manifest must install");
    registry
}

fn norm_static() -> OpStatic {
    OpStatic::Elementwise(ElementwiseStatic {
        t_bucket: 1,
        fused_with: None,
        op_params: ElementwiseParams::Norm(NormStatic {
            kind: NormKind::Rms,
            eps_bits: 1e-5f32.to_bits(),
            axis: NormAxis::Last,
            weight_offset_bits: 0.0f32.to_bits(),
            in_dtype: DType::F16,
            out_dtype: DType::F16,
            n: 1024,
            has_bias: false,
        }),
    })
}

fn attention_static() -> OpStatic {
    OpStatic::Attention(AttentionStatic {
        q_bucket: 1,
        h_local: 32,
        hkv_local: 8,
        d: 128,
        dv: 128,
        q_dtype: DType::F16,
        cache_dtype: DType::E4m3,
        attention_layout: LayoutId::CONTIGUOUS,
        mask_kind: AttentionMask::Causal,
        softmax_scale_bits: (1.0f32 / 128.0f32.sqrt()).to_bits(),
        out_dtype: DType::F16,
        mla: None,
        softcap_bits: None,
        sinks: 0,
    })
}

fn matmul_static() -> OpStatic {
    OpStatic::Matmul(MatmulStatic {
        m_bucket: 1,
        n: 1024,
        k: 1024,
        w_dtype: DType::F16,
        w_scheme: QuantScheme::None,
        w_layout: LayoutId::CONTIGUOUS,
        in_dtype: DType::F16,
        act_scheme: QuantScheme::None,
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        residual_dtype: None,
        transpose_w: false,
        interleave: false,
        sparse: false,
    })
}

fn sample_static() -> OpStatic {
    OpStatic::Sampling(SamplingStatic::Sample(SampleStatic {
        s_bucket: 1,
        v: 32000,
        rng: RngAlgorithm::Philox4x32,
    }))
}

fn check_round_trip(op: OpId, static_params: &OpStatic) {
    let arch = ArchName::from("gfx1201");
    let registry = cohesion_registry(&arch);
    assert!(
        static_params.validate().is_ok(),
        "{op:?} static must validate"
    );
    assert!(
        static_params.check_pair(op).is_ok(),
        "{op:?} static must pair with its OpId"
    );
    let first = static_hash(static_params);
    let second = static_hash(static_params);
    assert_ne!(first, 0, "{op:?} static hash must be nonzero");
    assert_eq!(first, second, "{op:?} static hash must be deterministic");
    let resolved = registry.resolve(op, &arch, static_params);
    assert!(
        resolved.is_ok(),
        "{op:?} static must resolve against the registry"
    );
    assert_eq!(
        resolved.expect("resolved above").tier,
        Tier::T1,
        "{op:?} must resolve to the T1 fallback"
    );
}

#[test]
fn sched_norm_static_round_trips_through_registry() {
    check_round_trip(OpId::Norm, &norm_static());
}

#[test]
fn sched_attention_static_round_trips_through_registry() {
    check_round_trip(OpId::Attention, &attention_static());
}

#[test]
fn sched_matmul_static_round_trips_through_registry() {
    check_round_trip(OpId::Matmul, &matmul_static());
}

#[test]
fn sched_sample_static_round_trips_through_registry() {
    check_round_trip(OpId::Sample, &sample_static());
}
