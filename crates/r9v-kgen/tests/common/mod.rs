// SPDX-License-Identifier: Apache-2.0
//! Common test helpers and representative op static descriptors for r9v-kgen tests.

use r9v_ir::{
    AttentionMask, DType, Epilogue, LayoutId, LinearAttnKind, P2pTransport, QuantScheme, SchemeId,
};
use r9v_registry::MoeFfnProjStatic;
use r9v_registry::{
    ActMulStatic, ActivationStatic, AllGatherStatic, AllReduceStatic, AllToAllStatic,
    AttentionStatic, BarrierStatic, CastStatic, CausalConv1dStatic, CollectivesStatic,
    ConcatStatic, CopyStatic, ElementwiseParams, ElementwiseStatic, EmbedGatherStatic,
    GatherRowsStatic, LinearAttnScanStatic, LogitSoftcapStatic, LogitsPostprocessStatic,
    MatmulStatic, MoeFfnStatic, MoeRouteStatic, NgramGatherStatic, NormStatic, OpId, OpStatic,
    PlacementKind, QuantActStatic, RecvStatic, ReduceScatterStatic, ResidualAddStatic,
    RopeScalingStatic, RopeStatic, SampleStatic, SamplingStatic, ScanMode, ScatterAddRowsStatic,
    SendStatic, SplitStatic, StateWriteKvStatic, VerifyMethodStatic, VerifyStatic,
};

pub const ALL_32_OPS: [OpId; 32] = [
    OpId::EmbedGather,
    OpId::NgramGather,
    OpId::QuantAct,
    OpId::Cast,
    OpId::Copy,
    OpId::GatherRows,
    OpId::ScatterAddRows,
    OpId::Split,
    OpId::Concat,
    OpId::Norm,
    OpId::ResidualAdd,
    OpId::ActMul,
    OpId::Activation,
    OpId::LogitSoftcap,
    OpId::Rope,
    OpId::Matmul,
    OpId::MoeRoute,
    OpId::MoeFfn,
    OpId::StateWriteKv,
    OpId::Attention,
    OpId::CausalConv1d,
    OpId::LinearAttnScan,
    OpId::LogitsPostprocess,
    OpId::Sample,
    OpId::Verify,
    OpId::AllReduce,
    OpId::AllGather,
    OpId::ReduceScatter,
    OpId::AllToAll,
    OpId::Send,
    OpId::Recv,
    OpId::Barrier,
];

pub fn representative_matmul_static() -> OpStatic {
    OpStatic::Matmul(MatmulStatic {
        m_bucket: 16,
        n: 4096,
        k: 4096,
        w_dtype: DType::F16,
        w_scheme: QuantScheme::None,
        w_layout: LayoutId::L1,
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

pub fn representative_moe_route_static() -> OpStatic {
    OpStatic::MoeRoute(MoeRouteStatic {
        t_bucket: 16,
        e_total: 8,
        top_k: 2,
        scoring: r9v_ir::MoeScoring::Softmax,
        renormalize: true,
        group: None,
        scale_bits: 1.0f32.to_bits(),
        has_bias: false,
    })
}

pub fn representative_causal_conv1d_static() -> OpStatic {
    OpStatic::CausalConv1d(CausalConv1dStatic {
        t_bucket: 16,
        channels: 2048,
        kernel: 4,
        act: r9v_ir::ConvActivation::Silu,
        x_dtype: DType::F16,
        w_dtype: DType::F16,
        w_scheme: QuantScheme::None,
        w_layout: LayoutId::L1,
        out_dtype: DType::F16,
        bias_dtype: None,
    })
}

pub fn representative_moe_ffn_static() -> OpStatic {
    OpStatic::MoeFfn(MoeFfnStatic {
        t_bucket: 16,
        e_local: 8,
        k_topk: 2,
        dm: 2048,
        dff: 1024,
        gate_up: MoeFfnProjStatic {
            dtype: DType::F16,
            scheme: QuantScheme::None,
            layout: LayoutId::L1,
        },
        down: MoeFfnProjStatic {
            dtype: DType::F16,
            scheme: QuantScheme::None,
            layout: LayoutId::L1,
        },
        in_dtype: DType::F16,
        act_scheme: QuantScheme::None,
        act: r9v_ir::ActivationKind::Silu,
        out_dtype: DType::F16,
        shared_experts: 0,
        placement_kind: PlacementKind::Device,
    })
}

pub fn representative_attention_static(mask_kind: AttentionMask, q_bucket: u32) -> OpStatic {
    OpStatic::Attention(AttentionStatic {
        q_bucket,
        h_local: 32,
        hkv_local: 8,
        d: 128,
        dv: 128,
        q_dtype: DType::F16,
        cache_dtype: DType::F16,
        attention_layout: LayoutId::L1,
        mask_kind,
        softmax_scale_bits: (1.0f32 / 128.0f32.sqrt()).to_bits(),
        out_dtype: DType::F16,
        mla: None,
        softcap_bits: None,
        sinks: 0,
    })
}

pub fn representative_state_write_kv_static() -> OpStatic {
    OpStatic::StateWriteKv(StateWriteKvStatic {
        hkv_local: 8,
        d: 128,
        dv: 128,
        in_dtype: DType::F16,
        cache_dtype: DType::F16,
        scale_granularity: r9v_ir::CacheScaleGranularity::PerTokenHead,
        attention_layout: LayoutId::L1,
        latent: None,
    })
}

pub fn representative_linear_attn_scan_static() -> OpStatic {
    OpStatic::LinearAttnScan(LinearAttnScanStatic {
        kind: LinearAttnKind::GatedDeltaNet,
        h_local: 16,
        d: 128,
        dv: 128,
        chunk: 64,
        mode: ScanMode::Chunked,
        in_dtype: DType::F16,
        out_dtype: DType::F16,
    })
}

pub fn representative_elementwise_static() -> OpStatic {
    OpStatic::Elementwise(ElementwiseStatic {
        t_bucket: 16,
        fused_with: None,
        op_params: ElementwiseParams::Norm(NormStatic {
            kind: r9v_ir::NormKind::Rms,
            eps_bits: 1e-5f32.to_bits(),
            axis: r9v_ir::NormAxis::Last,
            weight_offset_bits: 0.0f32.to_bits(),
            in_dtype: DType::F16,
            out_dtype: DType::F16,
            n: 4096,
            has_bias: false,
        }),
    })
}

pub fn representative_residual_add_static() -> OpStatic {
    OpStatic::Elementwise(ElementwiseStatic {
        t_bucket: 16,
        fused_with: None,
        op_params: ElementwiseParams::ResidualAdd(ResidualAddStatic {
            a_dtype: DType::F16,
            b_dtype: DType::F16,
            out_dtype: DType::F16,
            scale_bits: 1.0f32.to_bits(),
            n: 4096,
        }),
    })
}

pub fn representative_sample_static() -> OpStatic {
    OpStatic::Sampling(SamplingStatic::Sample(SampleStatic {
        s_bucket: 4,
        v: 32000,
        rng: r9v_ir::RngAlgorithm::Philox4x32,
    }))
}

pub fn representative_verify_static(tree: bool) -> OpStatic {
    OpStatic::Sampling(SamplingStatic::Verify(VerifyStatic {
        s_bucket: 4,
        v: 32000,
        q_bucket: 16,
        method: VerifyMethodStatic::Rejection,
        tree,
        has_draft_probs: false,
    }))
}

pub fn representative_logits_postprocess_static() -> OpStatic {
    OpStatic::Sampling(SamplingStatic::LogitsPostprocess(LogitsPostprocessStatic {
        s_bucket: 4,
        v: 32000,
        q_bucket: 16,
        has_history_counts: false,
        has_grammar_mask: false,
    }))
}

pub fn representative_collectives_static() -> OpStatic {
    OpStatic::Collectives(CollectivesStatic::AllReduce(AllReduceStatic {
        group: 0,
        rank: 0,
        world: 1,
        dtype: DType::F16,
        reduce_in: DType::F32,
        reduction_op: r9v_ir::ReduceOp::Sum,
        transport: P2pTransport::Direct,
        bytes_bucket: 65536,
    }))
}

pub fn representative_static_for_op(op: OpId) -> OpStatic {
    match op {
        OpId::Matmul => representative_matmul_static(),
        OpId::MoeRoute => representative_moe_route_static(),
        OpId::MoeFfn => representative_moe_ffn_static(),
        OpId::Attention => representative_attention_static(AttentionMask::Causal, 16),
        OpId::StateWriteKv => representative_state_write_kv_static(),
        OpId::CausalConv1d => representative_causal_conv1d_static(),
        OpId::LinearAttnScan => representative_linear_attn_scan_static(),
        OpId::EmbedGather => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::EmbedGather(EmbedGatherStatic {
                scale_bits: 1.0f32.to_bits(),
                table_placement: PlacementKind::Device,
                table_dtype: DType::F16,
                table_scheme: QuantScheme::None,
                table_layout: LayoutId::L0,
                out_dtype: DType::F16,
                vocab_size: 32000,
                dim: 4096,
            }),
        }),
        OpId::NgramGather => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::NgramGather(NgramGatherStatic {
                source: r9v_ir::NgramSource::Staged,
                hash: r9v_ir::HashId::new(1),
                orders: vec![2, 3],
                heads: 2,
                table_sizes: vec![1024, 1024],
                dn: 128,
                table_dtype: DType::F16,
                table_scheme: QuantScheme::None,
                table_layout: LayoutId::L0,
                staging_dtype: DType::F16,
                staging_scheme: QuantScheme::Scheme(SchemeId::new(1)),
                staging_layout: LayoutId::L0,
                scales_dtype: Some(DType::F32),
                combine: r9v_ir::NgramCombine::Sum,
                out_dtype: DType::F16,
            }),
        }),
        OpId::QuantAct => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::QuantAct(QuantActStatic {
                scheme: QuantScheme::PerToken,
                in_dtype: DType::F16,
                target: DType::I8,
                smoothing: r9v_ir::Smoothing::None,
                n: 4096,
            }),
        }),
        OpId::Cast => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::Cast(CastStatic {
                in_dtype: DType::F16,
                out_dtype: DType::Bf16,
                n: 4096,
            }),
        }),
        OpId::Copy => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::Copy(CopyStatic {
                kind: r9v_ir::CopyKind::Contiguize,
                dtype: DType::F16,
                n: 4096,
            }),
        }),
        OpId::GatherRows => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::GatherRows(GatherRowsStatic {
                dtype: DType::F16,
                index_dtype: DType::U32,
                width: 4096,
            }),
        }),
        OpId::ScatterAddRows => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::ScatterAddRows(ScatterAddRowsStatic {
                dtype: DType::F32,
                index_dtype: DType::U32,
                width: 4096,
                has_dest: false,
            }),
        }),
        OpId::Split => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::Split(SplitStatic {
                first: 2048,
                total: 4096,
                dtype: DType::F16,
            }),
        }),
        OpId::Concat => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::Concat(ConcatStatic {
                c0: 2048,
                c1: 2048,
                a_dtype: DType::F16,
                b_dtype: DType::F16,
                out_dtype: DType::F16,
            }),
        }),
        OpId::Norm => representative_elementwise_static(),
        OpId::ResidualAdd => representative_residual_add_static(),
        OpId::ActMul => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::ActMul(ActMulStatic {
                act: r9v_ir::ActivationKind::Silu,
                clamp_bits: None,
                dtype: DType::F16,
                width: 1024,
            }),
        }),
        OpId::Activation => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::Activation(ActivationStatic {
                act: r9v_ir::ActivationKind::Silu,
                clamp_bits: None,
                dtype: DType::F16,
                width: 1024,
            }),
        }),
        OpId::LogitSoftcap => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::LogitSoftcap(LogitSoftcapStatic {
                cap_bits: 50.0f32.to_bits(),
                v: 32000,
            }),
        }),
        OpId::Rope => OpStatic::Elementwise(ElementwiseStatic {
            t_bucket: 16,
            fused_with: None,
            op_params: ElementwiseParams::Rope(RopeStatic {
                rot_dim: 64,
                theta_bits: 10000.0f32.to_bits(),
                style: r9v_ir::RopeStyle::Neox,
                scaling: RopeScalingStatic::None,
                mrope_sections: None,
                in_dtype: DType::F16,
                out_dtype: DType::F16,
                h: 32,
                d: 128,
            }),
        }),
        OpId::LogitsPostprocess => representative_logits_postprocess_static(),
        OpId::Sample => representative_sample_static(),
        OpId::Verify => representative_verify_static(false),
        OpId::AllReduce => representative_collectives_static(),
        OpId::AllGather => OpStatic::Collectives(CollectivesStatic::AllGather(AllGatherStatic {
            group: 0,
            rank: 0,
            world: 1,
            dtype: DType::F16,
            axis: 0,
            transport: P2pTransport::Direct,
            bytes_bucket: 65536,
        })),
        OpId::ReduceScatter => {
            OpStatic::Collectives(CollectivesStatic::ReduceScatter(ReduceScatterStatic {
                group: 0,
                rank: 0,
                world: 1,
                dtype: DType::F16,
                reduce_in: DType::F32,
                reduction_op: r9v_ir::ReduceOp::Sum,
                axis: 0,
                transport: P2pTransport::Direct,
                bytes_bucket: 65536,
            }))
        }
        OpId::AllToAll => OpStatic::Collectives(CollectivesStatic::AllToAll(AllToAllStatic {
            group: 0,
            rank: 0,
            world: 1,
            dtype: DType::F16,
            transport: P2pTransport::Direct,
            bytes_bucket: 65536,
        })),
        OpId::Send => OpStatic::Collectives(CollectivesStatic::Send(SendStatic {
            group: 0,
            rank: 0,
            world: 2,
            peer: 1,
            dtype: DType::F16,
            transport: P2pTransport::Direct,
            bytes_bucket: 65536,
        })),
        OpId::Recv => OpStatic::Collectives(CollectivesStatic::Recv(RecvStatic {
            group: 0,
            rank: 1,
            world: 2,
            peer: 0,
            shape: vec![128, 4096],
            dtype: DType::F16,
            transport: P2pTransport::Direct,
            bytes_bucket: 65536,
        })),
        OpId::Barrier => OpStatic::Collectives(CollectivesStatic::Barrier(BarrierStatic {
            group: 0,
            rank: 0,
            world: 2,
            transport: P2pTransport::Direct,
        })),
    }
}
