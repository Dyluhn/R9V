// SPDX-License-Identifier: Apache-2.0
//! Tests for the closed kernel fusion table (Spec 1 §3.4; card A1.2).

use r9v_ir::{
    fusion_table, is_permitted_fusion, is_permitted_pair, match_chain, match_gated_pair, ActMulOp,
    ActivationKind, ActivationOp, AllToAllOp, AttentionMask, AttentionOp, CacheScaleGranularity,
    DType, Epilogue, FusionPattern, GroupId, LogitsPostprocessOp, MatmulOp, NormAxis, NormKind,
    NormOp, Op, QuantActOp, QuantScheme, ResidualAddOp, RngAlgorithm, RopeOp, RopeScaling,
    RopeStyle, SampleOp, Smoothing, StateHandle, StateKind, StateWriteKvOp, FUSION_TABLE,
};

fn fusion_ops() -> (Op, Op, Op, Op, Op, Op, Op, Op, Op, Op, Op) {
    let residual = Op::ResidualAdd(ResidualAddOp {
        out_dtype: DType::F16,
        scale: 1.0,
    });
    let norm = Op::Norm(NormOp {
        kind: NormKind::Rms,
        eps: 1.0e-5,
        axis: NormAxis::Last,
        weight_offset: 0.0,
        out_dtype: DType::F16,
    });
    let quant = Op::QuantAct(QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    });
    let matmul = Op::Matmul(MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        transpose_w: false,
    });
    let activation = Op::Activation(ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    });
    let act_mul = Op::ActMul(ActMulOp {
        act: ActivationKind::Silu,
        clamp: None,
    });
    let rope = Op::Rope(RopeOp {
        rot_dim: 64,
        theta: 10_000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F16,
    });
    let state_write = Op::StateWriteKv(StateWriteKvOp {
        cache_dtype: DType::F16,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: None,
        handle: StateHandle::new(0, StateKind::KvPaged),
    });
    let attention = Op::Attention(AttentionOp {
        softmax_scale: 0.125,
        mask: AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: None,
        out_dtype: DType::F16,
        handle: StateHandle::new(0, StateKind::KvPaged),
    });
    let logits = Op::LogitsPostprocess(LogitsPostprocessOp);
    let sample = Op::Sample(SampleOp {
        rng: RngAlgorithm::Philox4x32,
    });
    (
        residual,
        norm,
        quant,
        matmul,
        activation,
        act_mul,
        rope,
        state_write,
        attention,
        logits,
        sample,
    )
}

#[test]
fn fusion_table_has_exactly_nine_patterns() {
    assert_eq!(
        FUSION_TABLE.len(),
        9,
        "Spec 1 §3.4 defines exactly 9 fusion patterns"
    );
    let table = fusion_table();
    assert_eq!(table.len(), 9);

    let patterns = [
        FusionPattern::ResidualAddNorm,
        FusionPattern::NormQuantAct,
        FusionPattern::MatmulEpilogue,
        FusionPattern::GatedMatmulActMul,
        FusionPattern::RopeStateWriteKv,
        FusionPattern::RopeAttentionPrefill,
        FusionPattern::StateWriteKvAttentionDecode,
        FusionPattern::LogitsPostprocessSample,
        FusionPattern::QuantActAllToAll,
    ];

    for p in patterns {
        let entry = table.iter().find(|e| e.pattern == p);
        assert!(
            entry.is_some(),
            "Pattern {:?} must exist in FUSION_TABLE",
            p
        );
        let entry = entry.unwrap();
        assert!(!entry.description.is_empty());
        assert!(is_permitted_fusion(p));
    }
}

#[test]
fn typed_pair_matching_accepts_only_spec_transitions() {
    let (
        residual,
        norm,
        quant,
        matmul,
        activation,
        act_mul,
        rope,
        state_write,
        attention,
        logits,
        sample,
    ) = fusion_ops();
    let all_to_all = Op::AllToAll(AllToAllOp {
        group: GroupId::new(0),
        dtype: DType::I8,
    });

    assert_eq!(
        is_permitted_pair(&residual, &norm),
        Some(FusionPattern::ResidualAddNorm)
    );
    assert_eq!(
        is_permitted_pair(&norm, &quant),
        Some(FusionPattern::NormQuantAct)
    );
    assert_eq!(
        is_permitted_pair(&matmul, &residual),
        Some(FusionPattern::MatmulEpilogue)
    );
    assert_eq!(
        is_permitted_pair(&matmul, &activation),
        Some(FusionPattern::MatmulEpilogue)
    );
    assert_eq!(
        is_permitted_pair(&rope, &state_write),
        Some(FusionPattern::RopeStateWriteKv)
    );
    assert_eq!(
        is_permitted_pair(&rope, &attention),
        Some(FusionPattern::RopeAttentionPrefill)
    );
    assert_eq!(
        is_permitted_pair(&state_write, &attention),
        Some(FusionPattern::StateWriteKvAttentionDecode)
    );
    assert_eq!(
        is_permitted_pair(&logits, &sample),
        Some(FusionPattern::LogitsPostprocessSample)
    );
    assert_eq!(
        is_permitted_pair(&quant, &all_to_all),
        Some(FusionPattern::QuantActAllToAll)
    );

    assert_eq!(is_permitted_pair(&norm, &matmul), None);
    assert_eq!(is_permitted_pair(&attention, &norm), None);
    assert_eq!(is_permitted_pair(&sample, &logits), None);
    assert_eq!(is_permitted_pair(&all_to_all, &quant), None);
    assert_eq!(is_permitted_pair(&matmul, &act_mul), None);
}

#[test]
fn gated_parallel_matmuls_require_two_unfused_producers() {
    let (_, _, _, matmul, _, act_mul, _, _, _, _, _) = fusion_ops();
    assert_eq!(
        match_gated_pair(&matmul, &matmul, &act_mul),
        Some(FusionPattern::GatedMatmulActMul)
    );
    let fused_matmul = Op::Matmul(MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::Act(ActivationKind::Silu),
        transpose_w: false,
    });
    assert_eq!(match_gated_pair(&fused_matmul, &matmul, &act_mul), None);
}

#[test]
fn chain_matching_rejects_invented_multi_op_fusions() {
    let (residual, norm, quant, matmul, activation, _, _, _, _, _, _) = fusion_ops();
    assert_eq!(
        match_chain(&[&residual, &norm]),
        Some(FusionPattern::ResidualAddNorm)
    );
    assert_eq!(match_chain(&[]), None);
    assert_eq!(match_chain(&[&matmul]), None);
    assert_eq!(match_chain(&[&matmul, &residual, &activation]), None);
    assert_eq!(match_chain(&[&residual, &norm, &quant]), None);
}
