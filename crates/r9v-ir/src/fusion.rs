// SPDX-License-Identifier: Apache-2.0
//! Closed fusion table (Spec 1 §3.4; card A1.2).
//!
//! The compiler may fuse only pairs/chains listed in the fusion table. Every
//! fused kernel must satisfy the union of the fused ops' numerics contracts
//! (Spec 1 §1 Principle 7). Anything else requires an RFC (Spec 1 §7).
//! Fusion never changes an op's declared sharding rule (Spec 1 §3.4).

use crate::{Epilogue, Op};

/// Closed set of permitted compiler fusion patterns (Spec 1 §3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// `residual_add → norm`: fused add-norm, f32 stats.
    ResidualAddNorm,
    /// `norm → quant_act`: norm emits quantized activation + per-token scale directly.
    NormQuantAct,
    /// `matmul → bias / residual_add / activation`: epilogue.
    MatmulEpilogue,
    /// `matmul(gate) ∥ matmul(up) → act_mul`: interleaved gate/up GEMM with gated epilogue.
    GatedMatmulActMul,
    /// `rope → state_write_kv`: rope applied on the write path.
    RopeStateWriteKv,
    /// `rope → attention` (prefill): rope applied in the Q load.
    RopeAttentionPrefill,
    /// `state_write_kv → attention` (decode, `query_len ≤ 16`): single launch:
    /// write the new K/V, then attend; prefill keeps them separate.
    StateWriteKvAttentionDecode,
    /// `logits_postprocess → sample`: single sampling kernel.
    LogitsPostprocessSample,
    /// `quant_act → all_to_all` (EP): dispatch already-quantized tokens.
    QuantActAllToAll,
}

/// An entry in the exact fusion table (Spec 1 §3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FusionEntry {
    /// Permitted fusion pattern identifier.
    pub pattern: FusionPattern,
    /// Spec-defined result description.
    pub description: &'static str,
    /// Condition or sequence-class constraint (e.g. prefill, decode query_len <= 16, EP).
    pub condition: Option<&'static str>,
}

/// Exact fusion table defined by Spec 1 §3.4.
pub const FUSION_TABLE: &[FusionEntry] = &[
    FusionEntry {
        pattern: FusionPattern::ResidualAddNorm,
        description: "fused add-norm, f32 stats",
        condition: None,
    },
    FusionEntry {
        pattern: FusionPattern::NormQuantAct,
        description: "norm emits quantized activation + per-token scale directly",
        condition: None,
    },
    FusionEntry {
        pattern: FusionPattern::MatmulEpilogue,
        description: "epilogue",
        condition: None,
    },
    FusionEntry {
        pattern: FusionPattern::GatedMatmulActMul,
        description: "interleaved gate/up GEMM with gated epilogue",
        condition: None,
    },
    FusionEntry {
        pattern: FusionPattern::RopeStateWriteKv,
        description: "rope applied on the write path",
        condition: None,
    },
    FusionEntry {
        pattern: FusionPattern::RopeAttentionPrefill,
        description: "rope applied in the Q load",
        condition: Some("prefill"),
    },
    FusionEntry {
        pattern: FusionPattern::StateWriteKvAttentionDecode,
        description: "single launch: write the new K/V, then attend; prefill keeps them separate",
        condition: Some("decode, query_len <= 16"),
    },
    FusionEntry {
        pattern: FusionPattern::LogitsPostprocessSample,
        description: "single sampling kernel",
        condition: None,
    },
    FusionEntry {
        pattern: FusionPattern::QuantActAllToAll,
        description: "dispatch already-quantized tokens",
        condition: Some("EP"),
    },
];

/// Returns the exact fusion table as data (Spec 1 §3.4).
pub fn fusion_table() -> &'static [FusionEntry] {
    FUSION_TABLE
}

/// Checks if a fusion pattern is permitted by the fusion table.
pub fn is_permitted_fusion(pattern: FusionPattern) -> bool {
    FUSION_TABLE.iter().any(|entry| entry.pattern == pattern)
}

/// Checks if a producer-consumer op pair matches a permitted fusion pattern (Spec 1 §3.4).
///
/// Exhaustively matches all 32 closed `Op` variants without wildcards (CONVENTIONS.md §3.2).
// DECISION(A1.2): typed pair matching on &Op replaces stringly APIs per Spec 1 §3.4; pseudo-op bias is removed because bias is an epilogue attribute/operand on MatmulOp rather than an Op node.
pub fn is_permitted_pair(producer: &Op, consumer: &Op) -> Option<FusionPattern> {
    match_pair(producer, consumer)
}

/// Typed pair matching on `&Op` (Spec 1 §3.4).
///
/// Exhaustively matches all 32 closed `Op` variants without wildcards (CONVENTIONS.md §3.2).
fn match_pair(producer: &Op, consumer: &Op) -> Option<FusionPattern> {
    match producer {
        Op::ResidualAdd(_) => match consumer {
            Op::Norm(_) => Some(FusionPattern::ResidualAddNorm),
            Op::EmbedGather(_)
            | Op::NgramGather(_)
            | Op::QuantAct(_)
            | Op::Cast(_)
            | Op::Copy(_)
            | Op::GatherRows(_)
            | Op::ScatterAddRows(_)
            | Op::ResidualAdd(_)
            | Op::ActMul(_)
            | Op::Activation(_)
            | Op::Rope(_)
            | Op::Matmul(_)
            | Op::MoeRoute(_)
            | Op::MoeFfn(_)
            | Op::StateWriteKv(_)
            | Op::Attention(_)
            | Op::CausalConv1d(_)
            | Op::LinearAttnScan(_)
            | Op::LogitsPostprocess(_)
            | Op::Sample(_)
            | Op::Verify(_)
            | Op::AllReduce(_)
            | Op::AllGather(_)
            | Op::ReduceScatter(_)
            | Op::AllToAll(_)
            | Op::Send(_)
            | Op::Recv(_)
            | Op::Split(_)
            | Op::Concat(_)
            | Op::LogitSoftcap(_)
            | Op::Barrier(_) => None,
        },
        Op::Norm(_) => match consumer {
            Op::QuantAct(_) => Some(FusionPattern::NormQuantAct),
            Op::EmbedGather(_)
            | Op::NgramGather(_)
            | Op::Norm(_)
            | Op::Cast(_)
            | Op::Copy(_)
            | Op::GatherRows(_)
            | Op::ScatterAddRows(_)
            | Op::ResidualAdd(_)
            | Op::ActMul(_)
            | Op::Activation(_)
            | Op::Rope(_)
            | Op::Matmul(_)
            | Op::MoeRoute(_)
            | Op::MoeFfn(_)
            | Op::StateWriteKv(_)
            | Op::Attention(_)
            | Op::CausalConv1d(_)
            | Op::LinearAttnScan(_)
            | Op::LogitsPostprocess(_)
            | Op::Sample(_)
            | Op::Verify(_)
            | Op::AllReduce(_)
            | Op::AllGather(_)
            | Op::ReduceScatter(_)
            | Op::AllToAll(_)
            | Op::Send(_)
            | Op::Recv(_)
            | Op::Split(_)
            | Op::Concat(_)
            | Op::LogitSoftcap(_)
            | Op::Barrier(_) => None,
        },
        Op::Matmul(matmul) => match consumer {
            Op::ResidualAdd(_) | Op::Activation(_) if matmul.epilogue == Epilogue::None => {
                Some(FusionPattern::MatmulEpilogue)
            }
            Op::ResidualAdd(_) | Op::Activation(_) => None,
            Op::EmbedGather(_)
            | Op::NgramGather(_)
            | Op::QuantAct(_)
            | Op::Cast(_)
            | Op::Copy(_)
            | Op::GatherRows(_)
            | Op::ScatterAddRows(_)
            | Op::Norm(_)
            | Op::ActMul(_)
            | Op::Rope(_)
            | Op::Matmul(_)
            | Op::MoeRoute(_)
            | Op::MoeFfn(_)
            | Op::StateWriteKv(_)
            | Op::Attention(_)
            | Op::CausalConv1d(_)
            | Op::LinearAttnScan(_)
            | Op::LogitsPostprocess(_)
            | Op::Sample(_)
            | Op::Verify(_)
            | Op::AllReduce(_)
            | Op::AllGather(_)
            | Op::ReduceScatter(_)
            | Op::AllToAll(_)
            | Op::Send(_)
            | Op::Recv(_)
            | Op::Split(_)
            | Op::Concat(_)
            | Op::LogitSoftcap(_)
            | Op::Barrier(_) => None,
        },
        Op::Rope(_) => match consumer {
            Op::StateWriteKv(_) => Some(FusionPattern::RopeStateWriteKv),
            Op::Attention(_) => Some(FusionPattern::RopeAttentionPrefill),
            Op::EmbedGather(_)
            | Op::NgramGather(_)
            | Op::QuantAct(_)
            | Op::Cast(_)
            | Op::Copy(_)
            | Op::GatherRows(_)
            | Op::ScatterAddRows(_)
            | Op::Norm(_)
            | Op::ResidualAdd(_)
            | Op::ActMul(_)
            | Op::Activation(_)
            | Op::Rope(_)
            | Op::Matmul(_)
            | Op::MoeRoute(_)
            | Op::MoeFfn(_)
            | Op::CausalConv1d(_)
            | Op::LinearAttnScan(_)
            | Op::LogitsPostprocess(_)
            | Op::Sample(_)
            | Op::Verify(_)
            | Op::AllReduce(_)
            | Op::AllGather(_)
            | Op::ReduceScatter(_)
            | Op::AllToAll(_)
            | Op::Send(_)
            | Op::Recv(_)
            | Op::Split(_)
            | Op::Concat(_)
            | Op::LogitSoftcap(_)
            | Op::Barrier(_) => None,
        },
        Op::StateWriteKv(_) => match consumer {
            Op::Attention(_) => Some(FusionPattern::StateWriteKvAttentionDecode),
            Op::EmbedGather(_)
            | Op::NgramGather(_)
            | Op::QuantAct(_)
            | Op::Cast(_)
            | Op::Copy(_)
            | Op::GatherRows(_)
            | Op::ScatterAddRows(_)
            | Op::Norm(_)
            | Op::ResidualAdd(_)
            | Op::ActMul(_)
            | Op::Activation(_)
            | Op::Rope(_)
            | Op::Matmul(_)
            | Op::MoeRoute(_)
            | Op::MoeFfn(_)
            | Op::StateWriteKv(_)
            | Op::CausalConv1d(_)
            | Op::LinearAttnScan(_)
            | Op::LogitsPostprocess(_)
            | Op::Sample(_)
            | Op::Verify(_)
            | Op::AllReduce(_)
            | Op::AllGather(_)
            | Op::ReduceScatter(_)
            | Op::AllToAll(_)
            | Op::Send(_)
            | Op::Recv(_)
            | Op::Split(_)
            | Op::Concat(_)
            | Op::LogitSoftcap(_)
            | Op::Barrier(_) => None,
        },
        Op::LogitsPostprocess(_) => match consumer {
            Op::Sample(_) => Some(FusionPattern::LogitsPostprocessSample),
            Op::EmbedGather(_)
            | Op::NgramGather(_)
            | Op::QuantAct(_)
            | Op::Cast(_)
            | Op::Copy(_)
            | Op::GatherRows(_)
            | Op::ScatterAddRows(_)
            | Op::Norm(_)
            | Op::ResidualAdd(_)
            | Op::ActMul(_)
            | Op::Activation(_)
            | Op::Rope(_)
            | Op::Matmul(_)
            | Op::MoeRoute(_)
            | Op::MoeFfn(_)
            | Op::StateWriteKv(_)
            | Op::Attention(_)
            | Op::CausalConv1d(_)
            | Op::LinearAttnScan(_)
            | Op::LogitsPostprocess(_)
            | Op::Verify(_)
            | Op::AllReduce(_)
            | Op::AllGather(_)
            | Op::ReduceScatter(_)
            | Op::AllToAll(_)
            | Op::Send(_)
            | Op::Recv(_)
            | Op::Split(_)
            | Op::Concat(_)
            | Op::LogitSoftcap(_)
            | Op::Barrier(_) => None,
        },
        Op::QuantAct(_) => match consumer {
            Op::AllToAll(_) => Some(FusionPattern::QuantActAllToAll),
            Op::EmbedGather(_)
            | Op::NgramGather(_)
            | Op::QuantAct(_)
            | Op::Cast(_)
            | Op::Copy(_)
            | Op::GatherRows(_)
            | Op::ScatterAddRows(_)
            | Op::Norm(_)
            | Op::ResidualAdd(_)
            | Op::ActMul(_)
            | Op::Activation(_)
            | Op::Rope(_)
            | Op::Matmul(_)
            | Op::MoeRoute(_)
            | Op::MoeFfn(_)
            | Op::StateWriteKv(_)
            | Op::Attention(_)
            | Op::CausalConv1d(_)
            | Op::LinearAttnScan(_)
            | Op::LogitsPostprocess(_)
            | Op::Sample(_)
            | Op::Verify(_)
            | Op::AllReduce(_)
            | Op::AllGather(_)
            | Op::ReduceScatter(_)
            | Op::Send(_)
            | Op::Recv(_)
            | Op::Split(_)
            | Op::Concat(_)
            | Op::LogitSoftcap(_)
            | Op::Barrier(_) => None,
        },
        // DECISION(A1.14): split, concat, and logit_softcap participate in no
        // Spec 1 §3.4 fusion as producer or consumer; listed explicitly so the
        // closed match stays exhaustive. SI-28, SI-29.
        Op::Split(_) | Op::Concat(_) | Op::LogitSoftcap(_) => None,
        Op::EmbedGather(_)
        | Op::NgramGather(_)
        | Op::Cast(_)
        | Op::Copy(_)
        | Op::GatherRows(_)
        | Op::ScatterAddRows(_)
        | Op::ActMul(_)
        | Op::Activation(_)
        | Op::MoeRoute(_)
        | Op::MoeFfn(_)
        | Op::Attention(_)
        | Op::CausalConv1d(_)
        | Op::LinearAttnScan(_)
        | Op::Sample(_)
        | Op::Verify(_)
        | Op::AllReduce(_)
        | Op::AllGather(_)
        | Op::ReduceScatter(_)
        | Op::AllToAll(_)
        | Op::Send(_)
        | Op::Recv(_)
        | Op::Barrier(_) => None,
    }
}

/// Typed gated two-producer matcher for `matmul(gate) ∥ matmul(up) → act_mul` (Spec 1 §3.4).
///
/// Exhaustively matches all 32 closed `Op` variants without wildcards (CONVENTIONS.md §3.2).
pub fn match_gated_pair(
    gate_producer: &Op,
    up_producer: &Op,
    consumer: &Op,
) -> Option<FusionPattern> {
    match gate_producer {
        Op::Matmul(gate) => match up_producer {
            Op::Matmul(up) => match consumer {
                Op::ActMul(_)
                    if gate.epilogue == Epilogue::None && up.epilogue == Epilogue::None =>
                {
                    Some(FusionPattern::GatedMatmulActMul)
                }
                Op::ActMul(_) => None,
                Op::EmbedGather(_)
                | Op::NgramGather(_)
                | Op::QuantAct(_)
                | Op::Cast(_)
                | Op::Copy(_)
                | Op::GatherRows(_)
                | Op::ScatterAddRows(_)
                | Op::Norm(_)
                | Op::ResidualAdd(_)
                | Op::Activation(_)
                | Op::Rope(_)
                | Op::Matmul(_)
                | Op::MoeRoute(_)
                | Op::MoeFfn(_)
                | Op::StateWriteKv(_)
                | Op::Attention(_)
                | Op::CausalConv1d(_)
                | Op::LinearAttnScan(_)
                | Op::LogitsPostprocess(_)
                | Op::Sample(_)
                | Op::Verify(_)
                | Op::AllReduce(_)
                | Op::AllGather(_)
                | Op::ReduceScatter(_)
                | Op::AllToAll(_)
                | Op::Send(_)
                | Op::Recv(_)
                | Op::Split(_)
                | Op::Concat(_)
                | Op::LogitSoftcap(_)
                | Op::Barrier(_) => None,
            },
            Op::EmbedGather(_)
            | Op::NgramGather(_)
            | Op::QuantAct(_)
            | Op::Cast(_)
            | Op::Copy(_)
            | Op::GatherRows(_)
            | Op::ScatterAddRows(_)
            | Op::Norm(_)
            | Op::ResidualAdd(_)
            | Op::ActMul(_)
            | Op::Activation(_)
            | Op::Rope(_)
            | Op::MoeRoute(_)
            | Op::MoeFfn(_)
            | Op::StateWriteKv(_)
            | Op::Attention(_)
            | Op::CausalConv1d(_)
            | Op::LinearAttnScan(_)
            | Op::LogitsPostprocess(_)
            | Op::Sample(_)
            | Op::Verify(_)
            | Op::AllReduce(_)
            | Op::AllGather(_)
            | Op::ReduceScatter(_)
            | Op::AllToAll(_)
            | Op::Send(_)
            | Op::Recv(_)
            | Op::Split(_)
            | Op::Concat(_)
            | Op::LogitSoftcap(_)
            | Op::Barrier(_) => None,
        },
        Op::EmbedGather(_)
        | Op::NgramGather(_)
        | Op::QuantAct(_)
        | Op::Cast(_)
        | Op::Copy(_)
        | Op::GatherRows(_)
        | Op::ScatterAddRows(_)
        | Op::Norm(_)
        | Op::ResidualAdd(_)
        | Op::ActMul(_)
        | Op::Activation(_)
        | Op::Rope(_)
        | Op::MoeRoute(_)
        | Op::MoeFfn(_)
        | Op::StateWriteKv(_)
        | Op::Attention(_)
        | Op::CausalConv1d(_)
        | Op::LinearAttnScan(_)
        | Op::LogitsPostprocess(_)
        | Op::Sample(_)
        | Op::Verify(_)
        | Op::AllReduce(_)
        | Op::AllGather(_)
        | Op::ReduceScatter(_)
        | Op::AllToAll(_)
        | Op::Send(_)
        | Op::Recv(_)
        | Op::Split(_)
        | Op::Concat(_)
        | Op::LogitSoftcap(_)
        | Op::Barrier(_) => None,
    }
}

/// Checks if an op chain exactly matches a permitted fusion-table transition (Spec 1 §3.4).
///
/// Multi-op chains of length != 2 return `None`: Spec 1 §3.4 specifies only exact pair
/// transitions and gated two-producer GEMM. Invented multi-epilogue chains require an RFC.
pub fn match_chain(ops: &[&Op]) -> Option<FusionPattern> {
    if let [producer, consumer] = ops {
        is_permitted_pair(producer, consumer)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActMulOp, ActivationKind, ActivationOp, AllToAllOp, AttentionMask, AttentionOp,
        CacheScaleGranularity, DType, GroupId, LogitsPostprocessOp, MatmulOp, NormAxis, NormKind,
        NormOp, QuantActOp, QuantScheme, ResidualAddOp, RngAlgorithm, RopeOp, RopeScaling,
        RopeStyle, SampleOp, Smoothing, StateHandle, StateKind, StateWriteKvOp,
    };

    fn make_ops() -> (Op, Op, Op, Op, Op, Op, Op, Op, Op, Op, Op) {
        let res_add = Op::ResidualAdd(ResidualAddOp {
            out_dtype: DType::F16,
            scale: 1.0,
        });
        let norm = Op::Norm(NormOp {
            kind: NormKind::Rms,
            eps: 1e-5,
            axis: NormAxis::Last,
            weight_offset: 0.0,
            out_dtype: DType::F16,
        });
        let quant_act = Op::QuantAct(QuantActOp {
            scheme: QuantScheme::PerToken,
            target: DType::I8,
            smoothing: Smoothing::None,
        });
        let matmul = Op::Matmul(MatmulOp {
            out_dtype: DType::F16,
            epilogue: crate::Epilogue::None,
            transpose_w: false,
        });
        let act = Op::Activation(ActivationOp {
            act: ActivationKind::Silu,
            clamp: None,
        });
        let act_mul = Op::ActMul(ActMulOp {
            act: ActivationKind::Silu,
            clamp: None,
        });
        let rope = Op::Rope(RopeOp {
            rot_dim: 64,
            theta: 10000.0,
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
        let attn = Op::Attention(AttentionOp {
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
            res_add,
            norm,
            quant_act,
            matmul,
            act,
            act_mul,
            rope,
            state_write,
            attn,
            logits,
            sample,
        )
    }

    #[test]
    fn fusion_table_encodes_all_nine_patterns() {
        assert_eq!(FUSION_TABLE.len(), 9);
        assert_eq!(fusion_table().len(), 9);
    }

    #[test]
    fn typed_pair_queries_match_spec() {
        let (
            res_add,
            norm,
            quant_act,
            matmul,
            act,
            act_mul,
            rope,
            state_write,
            attn,
            logits,
            sample,
        ) = make_ops();
        let all_to_all = Op::AllToAll(AllToAllOp {
            group: GroupId::new(0),
            dtype: DType::F16,
        });

        assert_eq!(
            is_permitted_pair(&res_add, &norm),
            Some(FusionPattern::ResidualAddNorm)
        );
        assert_eq!(
            is_permitted_pair(&norm, &quant_act),
            Some(FusionPattern::NormQuantAct)
        );
        assert_eq!(
            is_permitted_pair(&matmul, &res_add),
            Some(FusionPattern::MatmulEpilogue)
        );
        assert_eq!(
            is_permitted_pair(&matmul, &act),
            Some(FusionPattern::MatmulEpilogue)
        );
        assert_eq!(
            is_permitted_pair(&rope, &state_write),
            Some(FusionPattern::RopeStateWriteKv)
        );
        assert_eq!(
            is_permitted_pair(&rope, &attn),
            Some(FusionPattern::RopeAttentionPrefill)
        );
        assert_eq!(
            is_permitted_pair(&state_write, &attn),
            Some(FusionPattern::StateWriteKvAttentionDecode)
        );
        assert_eq!(
            is_permitted_pair(&logits, &sample),
            Some(FusionPattern::LogitsPostprocessSample)
        );
        assert_eq!(
            is_permitted_pair(&quant_act, &all_to_all),
            Some(FusionPattern::QuantActAllToAll)
        );

        // Disallowed combinations
        assert_eq!(is_permitted_pair(&norm, &matmul), None);
        assert_eq!(is_permitted_pair(&attn, &norm), None);
        assert_eq!(is_permitted_pair(&matmul, &act_mul), None); // Gated GEMM requires 2 producers!
    }

    #[test]
    fn typed_gated_two_producer_matcher() {
        let (_, _, _, matmul, _, act_mul, _, _, _, _, _) = make_ops();
        let matmul_gate = matmul.clone();
        let matmul_up = matmul.clone();

        assert_eq!(
            match_gated_pair(&matmul_gate, &matmul_up, &act_mul),
            Some(FusionPattern::GatedMatmulActMul)
        );
    }

    #[test]
    fn no_multi_epilogue_chains() {
        let (res_add, norm, _, matmul, act, _, _, _, _, _, _) = make_ops();
        assert_eq!(
            match_chain(&[&res_add, &norm]),
            Some(FusionPattern::ResidualAddNorm)
        );
        assert_eq!(match_chain(&[&matmul]), None);
        assert_eq!(match_chain(&[&matmul, &res_add, &act]), None);
    }
}
