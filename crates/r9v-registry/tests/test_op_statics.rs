// SPDX-License-Identifier: Apache-2.0
//! Closed static semantics: exact identity, validation, lowering, and serde stability (Spec 4 §3).
//!
//! Covers all 32 ops: `OpStatic::op_id` exactness, `check_pair` mismatches as typed
//! errors, `validate` rejections, total `from_op` lowering coverage with every IR
//! behavior attribute copied, facts/op mismatch rejection, and serde roundtrips.

use r9v_ir::{
    ActivationKind, AttentionMask, CacheScaleGranularity, ConvActivation, CopyKind, DType, Dim,
    Epilogue, GroupId, HashId, LayoutId, LinearAttnKind, MoeScoring, NgramCombine, NgramSource,
    NormAxis, NormKind, QuantScheme, ReduceOp, RngAlgorithm, RopeScaling, RopeStyle, SchemeId,
    ShapeSymbol, Smoothing, StateHandle, StateKind,
};
use r9v_registry::{
    static_hash, ArchName, AttentionFacts, BundleManifest, CausalConv1dFacts, CollectiveFacts,
    CollectivesStatic, ElementwiseFacts, ElementwiseParams, LinearAttnScanFacts,
    LogitsPostprocessStatic, MatmulFacts, MatmulStatic, MlaAttentionStatic, MlaLatentStatic,
    MoeFfnFacts, MoeFfnProjStatic, MoeRouteFacts, OpId, OpStatic, PlacementKind, Registry,
    RegistryConfig, RegistryError, SamplingFacts, SamplingStatic, ScanMode, StateWriteKvFacts,
    StaticFacts, TileConfig, VariantKey, VerifyMethodStatic,
};

fn all_32_op_ids() -> [OpId; 32] {
    [
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
    ]
}

fn ir_op_for(op: OpId) -> r9v_ir::Op {
    match op {
        OpId::EmbedGather => r9v_ir::Op::EmbedGather(r9v_ir::EmbedGatherOp {
            scale: 1.0,
            out_dtype: DType::F16,
        }),
        OpId::NgramGather => r9v_ir::Op::NgramGather(r9v_ir::NgramGatherOp {
            source: NgramSource::Staged,
            orders: vec![2u32, 3u32].into_boxed_slice(),
            heads: 2,
            hash: HashId::new(7),
            table_sizes: vec![1024u32, 1024u32].into_boxed_slice(),
            combine: NgramCombine::Sum,
            out_dtype: DType::F16,
        }),
        OpId::QuantAct => r9v_ir::Op::QuantAct(r9v_ir::QuantActOp {
            scheme: QuantScheme::PerToken,
            target: DType::I8,
            smoothing: Smoothing::None,
        }),
        OpId::Cast => r9v_ir::Op::Cast(r9v_ir::CastOp { dtype: DType::Bf16 }),
        OpId::Copy => r9v_ir::Op::Copy(r9v_ir::CopyOp {
            kind: CopyKind::Contiguize,
        }),
        OpId::GatherRows => r9v_ir::Op::GatherRows(r9v_ir::GatherRowsOp),
        OpId::ScatterAddRows => r9v_ir::Op::ScatterAddRows(r9v_ir::ScatterAddRowsOp),
        OpId::Split => r9v_ir::Op::Split(r9v_ir::SplitOp { first: 2048 }),
        OpId::Concat => r9v_ir::Op::Concat(r9v_ir::ConcatOp),
        OpId::Norm => r9v_ir::Op::Norm(r9v_ir::NormOp {
            kind: NormKind::Rms,
            eps: 1e-5,
            axis: NormAxis::Last,
            weight_offset: 0.0,
            out_dtype: DType::F16,
        }),
        OpId::ResidualAdd => r9v_ir::Op::ResidualAdd(r9v_ir::ResidualAddOp {
            out_dtype: DType::F16,
            scale: 1.0,
        }),
        OpId::ActMul => r9v_ir::Op::ActMul(r9v_ir::ActMulOp {
            act: ActivationKind::Silu,
            clamp: None,
        }),
        OpId::Activation => r9v_ir::Op::Activation(r9v_ir::ActivationOp {
            act: ActivationKind::Silu,
            clamp: None,
        }),
        OpId::LogitSoftcap => r9v_ir::Op::LogitSoftcap(r9v_ir::LogitSoftcapOp { cap: 50.0 }),
        OpId::Rope => r9v_ir::Op::Rope(r9v_ir::RopeOp {
            rot_dim: 64,
            theta: 10000.0,
            style: RopeStyle::Neox,
            scaling: RopeScaling::None,
            mrope_sections: None,
            out_dtype: DType::F16,
        }),
        OpId::Matmul => r9v_ir::Op::Matmul(r9v_ir::MatmulOp {
            out_dtype: DType::F16,
            epilogue: Epilogue::None,
            transpose_w: false,
        }),
        OpId::MoeRoute => r9v_ir::Op::MoeRoute(r9v_ir::MoeRouteOp {
            top_k: 2,
            scoring: MoeScoring::Softmax,
            renormalize: true,
            group: None,
            scale: 1.0,
        }),
        OpId::MoeFfn => r9v_ir::Op::MoeFfn(r9v_ir::MoeFfnOp {
            act: ActivationKind::Silu,
            out_dtype: DType::F16,
            shared_experts: 0,
        }),
        OpId::StateWriteKv => r9v_ir::Op::StateWriteKv(r9v_ir::StateWriteKvOp {
            cache_dtype: DType::F16,
            scale_granularity: CacheScaleGranularity::PerTokenHead,
            latent: None,
            handle: StateHandle::new(0, StateKind::KvPaged),
        }),
        OpId::Attention => r9v_ir::Op::Attention(r9v_ir::AttentionOp {
            softmax_scale: 1.0f32 / 128.0f32.sqrt(),
            mask: AttentionMask::Causal,
            sinks: 0,
            logit_softcap: None,
            mla: None,
            out_dtype: DType::F16,
            handle: StateHandle::new(0, StateKind::KvPaged),
        }),
        OpId::CausalConv1d => r9v_ir::Op::CausalConv1d(r9v_ir::CausalConv1dOp {
            kernel: 4,
            act: ConvActivation::Silu,
            handle: StateHandle::new(0, StateKind::ConvWindow),
        }),
        OpId::LinearAttnScan => r9v_ir::Op::LinearAttnScan(r9v_ir::LinearAttnScanOp {
            kind: LinearAttnKind::GatedDeltaNet,
            chunk: 64,
            out_dtype: DType::F16,
            handle: StateHandle::new(0, StateKind::Recurrent),
        }),
        OpId::LogitsPostprocess => r9v_ir::Op::LogitsPostprocess(r9v_ir::LogitsPostprocessOp),
        OpId::Sample => r9v_ir::Op::Sample(r9v_ir::SampleOp {
            rng: RngAlgorithm::Philox4x32,
        }),
        OpId::Verify => r9v_ir::Op::Verify(r9v_ir::VerifyOp {
            method: r9v_ir::VerifyMethod::Rejection,
        }),
        OpId::AllReduce => r9v_ir::Op::AllReduce(r9v_ir::AllReduceOp {
            group: GroupId::new(0),
            op: ReduceOp::Sum,
            dtype: DType::F16,
            reduce_in: DType::F32,
        }),
        OpId::AllGather => r9v_ir::Op::AllGather(r9v_ir::AllGatherOp {
            group: GroupId::new(0),
            axis: 0,
            dtype: DType::F16,
        }),
        OpId::ReduceScatter => r9v_ir::Op::ReduceScatter(r9v_ir::ReduceScatterOp {
            group: GroupId::new(0),
            axis: 0,
            op: ReduceOp::Sum,
            dtype: DType::F16,
            reduce_in: DType::F32,
        }),
        OpId::AllToAll => r9v_ir::Op::AllToAll(r9v_ir::AllToAllOp {
            group: GroupId::new(0),
            dtype: DType::F16,
        }),
        OpId::Send => r9v_ir::Op::Send(r9v_ir::SendOp {
            group: GroupId::new(0),
            peer: 1,
            dtype: DType::F16,
        }),
        OpId::Recv => r9v_ir::Op::Recv(r9v_ir::RecvOp {
            group: GroupId::new(0),
            peer: 0,
            shape: vec![Dim::Concrete(128), Dim::Concrete(4096)].into_boxed_slice(),
            dtype: DType::F16,
        }),
        OpId::Barrier => r9v_ir::Op::Barrier(r9v_ir::BarrierOp {
            group: GroupId::new(0),
        }),
    }
}

fn facts_for(op: OpId) -> StaticFacts {
    match op {
        OpId::Matmul => StaticFacts::Matmul(MatmulFacts {
            m_bucket: 16,
            n: 4096,
            k: 4096,
            w_dtype: DType::F16,
            w_scheme: QuantScheme::None,
            w_layout: LayoutId::L1,
            in_dtype: DType::F16,
            act_scheme: QuantScheme::None,
            residual_dtype: None,
            interleave: false,
            sparse: false,
        }),
        OpId::MoeRoute => StaticFacts::MoeRoute(MoeRouteFacts {
            t_bucket: 16,
            e_total: 8,
            has_bias: false,
        }),
        OpId::MoeFfn => StaticFacts::MoeFfn(MoeFfnFacts {
            t_bucket: 16,
            e_local: 8,
            k_topk: 2,
            dm: 2048,
            dff: 1024,
            gate_up: r9v_registry::MoeFfnProjStatic {
                dtype: DType::F16,
                scheme: QuantScheme::None,
                layout: LayoutId::L1,
            },
            down: r9v_registry::MoeFfnProjStatic {
                dtype: DType::F16,
                scheme: QuantScheme::None,
                layout: LayoutId::L1,
            },
            in_dtype: DType::F16,
            act_scheme: QuantScheme::None,
            placement_kind: PlacementKind::Device,
        }),
        OpId::Attention => StaticFacts::Attention(AttentionFacts {
            q_bucket: 16,
            h_local: 32,
            hkv_local: 8,
            d: 128,
            dv: 128,
            q_dtype: DType::F16,
            cache_dtype: DType::F16,
            attention_layout: LayoutId::L1,
        }),
        OpId::StateWriteKv => StaticFacts::StateWriteKv(StateWriteKvFacts {
            hkv_local: 8,
            d: 128,
            dv: 128,
            in_dtype: DType::F16,
            attention_layout: LayoutId::L1,
        }),
        OpId::CausalConv1d => StaticFacts::CausalConv1d(CausalConv1dFacts {
            t_bucket: 16,
            channels: 2048,
            x_dtype: DType::F16,
            w_dtype: DType::F16,
            w_scheme: QuantScheme::None,
            w_layout: LayoutId::L1,
            out_dtype: DType::F16,
            bias_dtype: None,
        }),
        OpId::LinearAttnScan => StaticFacts::LinearAttnScan(LinearAttnScanFacts {
            h_local: 16,
            d: 128,
            dv: 128,
            in_dtype: DType::F16,
            mode: ScanMode::Chunked,
        }),
        OpId::EmbedGather => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::EmbedGather {
                table_placement: PlacementKind::Device,
                table_dtype: DType::F16,
                table_scheme: QuantScheme::None,
                table_layout: LayoutId::L0,
                vocab_size: 32000,
                dim: 4096,
            },
        },
        OpId::NgramGather => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::NgramGather {
                dn: 128,
                table_dtype: DType::F16,
                table_scheme: QuantScheme::None,
                table_layout: LayoutId::L0,
                staging_dtype: DType::F16,
                staging_scheme: QuantScheme::Scheme(r9v_ir::SchemeId::new(1)),
                staging_layout: LayoutId::L0,
                scales_dtype: Some(DType::F32),
            },
        },
        OpId::QuantAct => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::QuantAct {
                in_dtype: DType::F16,
                n: 4096,
            },
        },
        OpId::Cast => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::Cast {
                in_dtype: DType::F16,
                n: 4096,
            },
        },
        OpId::Copy => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::Copy {
                dtype: DType::F16,
                n: 4096,
            },
        },
        OpId::GatherRows => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::GatherRows {
                dtype: DType::F16,
                index_dtype: DType::U32,
                width: 4096,
            },
        },
        OpId::ScatterAddRows => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::ScatterAddRows {
                dtype: DType::F32,
                index_dtype: DType::U32,
                width: 4096,
                has_dest: false,
            },
        },
        OpId::Split => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::Split {
                total: 4096,
                dtype: DType::F16,
            },
        },
        OpId::Concat => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::Concat {
                c0: 2048,
                c1: 2048,
                a_dtype: DType::F16,
                b_dtype: DType::F16,
                out_dtype: DType::F16,
            },
        },
        OpId::Norm => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::Norm {
                in_dtype: DType::F16,
                n: 4096,
                has_bias: false,
            },
        },
        OpId::ResidualAdd => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::ResidualAdd {
                a_dtype: DType::F16,
                b_dtype: DType::F16,
                n: 4096,
            },
        },
        OpId::ActMul => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::ActMul {
                dtype: DType::F16,
                width: 1024,
            },
        },
        OpId::Activation => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::Activation {
                dtype: DType::F16,
                width: 1024,
            },
        },
        OpId::LogitSoftcap => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::LogitSoftcap { v: 32000 },
        },
        OpId::Rope => StaticFacts::Elementwise {
            t_bucket: 16,
            fused_with: None,
            params: ElementwiseFacts::Rope {
                in_dtype: DType::F16,
                h: 32,
                d: 128,
            },
        },
        OpId::LogitsPostprocess => StaticFacts::Sampling(SamplingFacts::LogitsPostprocess {
            s_bucket: 4,
            v: 32000,
            q_bucket: 16,
            has_history_counts: false,
            has_grammar_mask: false,
        }),
        OpId::Sample => StaticFacts::Sampling(SamplingFacts::Sample {
            s_bucket: 4,
            v: 32000,
        }),
        OpId::Verify => StaticFacts::Sampling(SamplingFacts::Verify {
            s_bucket: 4,
            v: 32000,
            q_bucket: 16,
            tree: false,
            has_draft_probs: false,
        }),
        OpId::AllReduce => StaticFacts::Collectives(CollectiveFacts::AllReduce {
            rank: 0,
            world: 1,
            transport: r9v_ir::P2pTransport::Direct,
            bytes_bucket: 65536,
        }),
        OpId::AllGather => StaticFacts::Collectives(CollectiveFacts::AllGather {
            rank: 0,
            world: 1,
            transport: r9v_ir::P2pTransport::Direct,
            bytes_bucket: 65536,
        }),
        OpId::ReduceScatter => StaticFacts::Collectives(CollectiveFacts::ReduceScatter {
            rank: 0,
            world: 1,
            transport: r9v_ir::P2pTransport::Direct,
            bytes_bucket: 65536,
        }),
        OpId::AllToAll => StaticFacts::Collectives(CollectiveFacts::AllToAll {
            rank: 0,
            world: 1,
            transport: r9v_ir::P2pTransport::Direct,
            bytes_bucket: 65536,
        }),
        OpId::Send => StaticFacts::Collectives(CollectiveFacts::Send {
            rank: 0,
            world: 2,
            transport: r9v_ir::P2pTransport::Direct,
            bytes_bucket: 65536,
        }),
        OpId::Recv => StaticFacts::Collectives(CollectiveFacts::Recv {
            rank: 1,
            world: 2,
            transport: r9v_ir::P2pTransport::Direct,
            bytes_bucket: 65536,
            shape: vec![128, 4096],
        }),
        OpId::Barrier => StaticFacts::Collectives(CollectiveFacts::Barrier {
            rank: 0,
            world: 2,
            transport: r9v_ir::P2pTransport::Direct,
        }),
    }
}

/// Total lowering: every one of the 32 IR ops lowers with matching facts.
#[test]
fn test_from_op_total_coverage_all_32_ops() {
    let ops = all_32_op_ids();
    assert_eq!(ops.len(), 32);
    for op in ops {
        let ir = ir_op_for(op);
        let facts = facts_for(op);
        let lowered = OpStatic::from_op(&ir, &facts)
            .unwrap_or_else(|e| panic!("from_op failed for {op}: {e}"));
        assert_eq!(lowered.op_id(), op, "lowered static must carry {op}");
        lowered
            .check_pair(op)
            .unwrap_or_else(|e| panic!("check_pair failed for {op}: {e}"));
        lowered
            .validate()
            .unwrap_or_else(|e| panic!("validate failed for {op}: {e}"));
        // Serde roundtrip preserves identity and hash.
        let json = serde_json::to_string(&lowered).expect("static must serialize");
        let back: OpStatic = serde_json::from_str(&json).expect("static must deserialize");
        assert_eq!(lowered, back, "serde roundtrip must preserve {op}");
        assert_eq!(static_hash(&lowered), static_hash(&back));
        assert_eq!(back.op_id(), op);
        // VariantKey hashing works for the lowered static.
        let key = VariantKey::new(
            op,
            ArchName::from("gfx942"),
            1,
            lowered,
            TileConfig::new(64, 64, 32),
        );
        assert_ne!(key.static_hash(), 0);
    }
}

/// Lowering copies every IR behavior attribute exactly.
#[test]
fn test_from_op_copies_every_behavior_attribute() {
    // Matmul epilogue and transpose.
    let ir = r9v_ir::Op::Matmul(r9v_ir::MatmulOp {
        out_dtype: DType::Bf16,
        epilogue: Epilogue::Bias,
        transpose_w: true,
    });
    let lowered = OpStatic::from_op(&ir, &facts_for(OpId::Matmul)).expect("matmul lowers");
    match lowered {
        OpStatic::Matmul(s) => {
            assert_eq!(s.out_dtype, DType::Bf16);
            assert_eq!(s.epilogue, Epilogue::Bias);
            assert!(s.transpose_w);
        }
        other => panic!("expected matmul, got {other:?}"),
    }

    // MoeRoute scoring, renormalize, group, scale bits.
    let ir = r9v_ir::Op::MoeRoute(r9v_ir::MoeRouteOp {
        top_k: 4,
        scoring: MoeScoring::Sigmoid,
        renormalize: false,
        group: Some(r9v_ir::MoeGroup {
            n_group: 4,
            topk_group: 2,
        }),
        scale: 2.0,
    });
    let lowered = OpStatic::from_op(&ir, &facts_for(OpId::MoeRoute)).expect("moe_route lowers");
    match lowered {
        OpStatic::MoeRoute(s) => {
            assert_eq!(s.top_k, 4);
            assert_eq!(s.scoring, MoeScoring::Sigmoid);
            assert!(!s.renormalize);
            assert_eq!(
                s.group,
                Some(r9v_ir::MoeGroup {
                    n_group: 4,
                    topk_group: 2
                })
            );
            assert_eq!(s.scale(), 2.0);
        }
        other => panic!("expected moe_route, got {other:?}"),
    }

    // Attention softmax scale, sinks, softcap, and full MLA.
    let ir = r9v_ir::Op::Attention(r9v_ir::AttentionOp {
        softmax_scale: 0.25,
        mask: AttentionMask::CausalWindow(512),
        sinks: 3,
        logit_softcap: Some(50.0),
        mla: Some(r9v_ir::MlaAttentionSpec {
            q_lora_rank: Some(64),
            kv_lora_rank: 128,
            qk_nope_dim: 96,
            qk_rope_dim: 32,
            v_dim: 128,
        }),
        out_dtype: DType::Bf16,
        handle: StateHandle::new(1, StateKind::KvLatent),
    });
    let lowered = OpStatic::from_op(&ir, &facts_for(OpId::Attention)).expect("attention lowers");
    match lowered {
        OpStatic::Attention(s) => {
            assert_eq!(s.softmax_scale(), 0.25);
            assert_eq!(s.sinks, 3);
            assert_eq!(s.softcap_f32(), Some(50.0));
            assert_eq!(s.out_dtype, DType::Bf16);
            let mla = s.mla.expect("mla must be preserved");
            assert_eq!(mla.q_lora_rank, Some(64));
            assert_eq!(mla.kv_lora_rank, 128);
            assert_eq!(mla.qk_nope_dim, 96);
            assert_eq!(mla.qk_rope_dim, 32);
            assert_eq!(mla.v_dim, 128);
            assert_eq!(mla.to_ir().kv_lora_rank, 128);
        }
        other => panic!("expected attention, got {other:?}"),
    }

    // StateWriteKv granularity and latent.
    let ir = r9v_ir::Op::StateWriteKv(r9v_ir::StateWriteKvOp {
        cache_dtype: DType::I8,
        scale_granularity: CacheScaleGranularity::PerBlock,
        latent: Some(r9v_ir::MlaLatent {
            kv_lora_rank: 128,
            rope_dim: 64,
        }),
        handle: StateHandle::new(0, StateKind::KvLatent),
    });
    let lowered =
        OpStatic::from_op(&ir, &facts_for(OpId::StateWriteKv)).expect("state_write_kv lowers");
    match lowered {
        OpStatic::StateWriteKv(s) => {
            assert_eq!(s.cache_dtype, DType::I8);
            assert_eq!(s.scale_granularity, CacheScaleGranularity::PerBlock);
            let latent = s.latent.expect("latent must be preserved");
            assert_eq!(latent.kv_lora_rank, 128);
            assert_eq!(latent.rope_dim, 64);
        }
        other => panic!("expected state_write_kv, got {other:?}"),
    }

    // Rope scaling and sections.
    let ir = r9v_ir::Op::Rope(r9v_ir::RopeOp {
        rot_dim: 64,
        theta: 500000.0,
        style: RopeStyle::Interleaved,
        scaling: RopeScaling::Linear(8.0),
        mrope_sections: Some([16, 24, 24]),
        out_dtype: DType::Bf16,
    });
    let lowered = OpStatic::from_op(&ir, &facts_for(OpId::Rope)).expect("rope lowers");
    match lowered {
        OpStatic::Elementwise(e) => match e.op_params {
            ElementwiseParams::Rope(r) => {
                assert_eq!(r.rot_dim, 64);
                assert_eq!(r.theta(), 500000.0);
                assert_eq!(r.style, RopeStyle::Interleaved);
                assert_eq!(r.scaling.to_ir(), RopeScaling::Linear(8.0));
                assert_eq!(r.mrope_sections, Some([16, 24, 24]));
                assert_eq!(r.out_dtype, DType::Bf16);
            }
            other => panic!("expected rope params, got {other:?}"),
        },
        other => panic!("expected elementwise, got {other:?}"),
    }

    // ResidualAdd scale bits.
    let ir = r9v_ir::Op::ResidualAdd(r9v_ir::ResidualAddOp {
        out_dtype: DType::F32,
        scale: 0.5,
    });
    let lowered =
        OpStatic::from_op(&ir, &facts_for(OpId::ResidualAdd)).expect("residual_add lowers");
    match lowered {
        OpStatic::Elementwise(e) => match e.op_params {
            ElementwiseParams::ResidualAdd(r) => {
                assert_eq!(r.out_dtype, DType::F32);
                assert_eq!(r.scale(), 0.5);
            }
            other => panic!("expected residual_add params, got {other:?}"),
        },
        other => panic!("expected elementwise, got {other:?}"),
    }

    // Verify typical acceptance bits and tree.
    let ir = r9v_ir::Op::Verify(r9v_ir::VerifyOp {
        method: r9v_ir::VerifyMethod::TypicalAcceptance {
            eps: 0.05,
            delta: 1.25,
        },
    });
    let facts = StaticFacts::Sampling(SamplingFacts::Verify {
        s_bucket: 4,
        v: 32000,
        q_bucket: 16,
        tree: true,
        has_draft_probs: true,
    });
    let lowered = OpStatic::from_op(&ir, &facts).expect("verify lowers");
    match lowered {
        OpStatic::Sampling(SamplingStatic::Verify(v)) => {
            assert!(v.tree);
            assert!(v.has_draft_probs);
            match v.method {
                VerifyMethodStatic::TypicalAcceptance {
                    eps_bits,
                    delta_bits,
                } => {
                    assert_eq!(f32::from_bits(eps_bits), 0.05);
                    assert_eq!(f32::from_bits(delta_bits), 1.25);
                }
                other => panic!("expected typical acceptance, got {other:?}"),
            }
        }
        other => panic!("expected sampling verify, got {other:?}"),
    }

    // Collectives carry group, peer, axis, reduction from IR.
    let ir = r9v_ir::Op::ReduceScatter(r9v_ir::ReduceScatterOp {
        group: GroupId::new(9),
        axis: 1,
        op: ReduceOp::Sum,
        dtype: DType::Bf16,
        reduce_in: DType::F32,
    });
    let lowered =
        OpStatic::from_op(&ir, &facts_for(OpId::ReduceScatter)).expect("reduce_scatter lowers");
    match lowered {
        OpStatic::Collectives(CollectivesStatic::ReduceScatter(s)) => {
            assert_eq!(s.group, 9);
            assert_eq!(s.axis, 1);
            assert_eq!(s.reduction_op, ReduceOp::Sum);
            assert_eq!(s.dtype, DType::Bf16);
            assert_eq!(s.reduce_in, DType::F32);
        }
        other => panic!("expected reduce_scatter, got {other:?}"),
    }
}

/// Mismatched facts/op pairs are typed errors across the full matrix.
#[test]
fn test_from_op_rejects_every_mismatched_facts_pair() {
    let ops = all_32_op_ids();
    for op in ops {
        let ir = ir_op_for(op);
        for other in ops {
            if other == op {
                continue;
            }
            let wrong = facts_for(other);
            match OpStatic::from_op(&ir, &wrong) {
                Err(RegistryError::FactsOpMismatch { op: got, .. }) => {
                    assert_eq!(got, op, "facts error must name the IR op");
                }
                Err(other_err) => {
                    panic!("mismatched facts for {op} must be FactsOpMismatch, got {other_err:?}")
                }
                Ok(ok) => panic!("mismatched facts for {op} (facts of {other}) lowered to {ok:?}"),
            }
        }
    }

    // Recv shape rank mismatch is a typed error even with the right variant.
    let ir = ir_op_for(OpId::Recv);
    let facts = StaticFacts::Collectives(CollectiveFacts::Recv {
        rank: 1,
        world: 2,
        transport: r9v_ir::P2pTransport::Direct,
        bytes_bucket: 65536,
        shape: vec![128],
    });
    match OpStatic::from_op(&ir, &facts).expect_err("recv rank mismatch must fail") {
        RegistryError::FactsOpMismatch { op, .. } => assert_eq!(op, OpId::Recv),
        other => panic!("expected FactsOpMismatch, got {other:?}"),
    }
}

/// Exact pairing: all 32 cross pairings are typed StaticOpMismatch errors.
#[test]
fn test_all_32_exact_pairing_mismatches_are_typed_errors() {
    let lowered: Vec<(OpId, OpStatic)> = all_32_op_ids()
        .iter()
        .map(|op| {
            (
                *op,
                OpStatic::from_op(&ir_op_for(*op), &facts_for(*op)).expect("lowers"),
            )
        })
        .collect();
    assert_eq!(lowered.len(), 32);
    for (op, _) in &lowered {
        for (other_op, other_stat) in &lowered {
            if op == other_op {
                continue;
            }
            match other_stat.check_pair(*op) {
                Err(RegistryError::StaticOpMismatch { op: got, static_op }) => {
                    assert_eq!(got, *op);
                    assert_eq!(static_op, *other_op);
                }
                Err(other_err) => panic!(
                    "pairing {op} with {other_op} statics must be StaticOpMismatch, got {other_err:?}"
                ),
                Ok(()) => panic!("pairing {op} with {other_op} statics must fail"),
            }
        }
    }
}

/// Invalid shapes are rejected by validate(), never silently accepted.
#[test]
fn test_validate_rejects_invalid_statics() {
    // MoeFfn projections must be legal (dtype, scheme) combinations.
    let mut good = OpStatic::from_op(&ir_op_for(OpId::MoeFfn), &facts_for(OpId::MoeFfn))
        .expect("moe_ffn lowers");
    if let OpStatic::MoeFfn(ref mut s) = good {
        s.gate_up.scheme = QuantScheme::PerRow;
    }
    assert!(
        good.validate().is_err(),
        "f16 gate_up with PerRow scales must fail"
    );

    // Split requires 0 < first < total.
    let mut split =
        OpStatic::from_op(&ir_op_for(OpId::Split), &facts_for(OpId::Split)).expect("split lowers");
    if let OpStatic::Elementwise(ref mut e) = split {
        if let ElementwiseParams::Split(ref mut s) = e.op_params {
            s.first = s.total;
        }
    }
    assert!(split.validate().is_err(), "first == total split must fail");

    // Ngram orders/table_sizes must match heads.
    let mut ngram = OpStatic::from_op(&ir_op_for(OpId::NgramGather), &facts_for(OpId::NgramGather))
        .expect("ngram lowers");
    if let OpStatic::Elementwise(ref mut e) = ngram {
        if let ElementwiseParams::NgramGather(ref mut s) = e.op_params {
            s.orders.pop();
        }
    }
    assert!(ngram.validate().is_err(), "orders/head mismatch must fail");

    // Collective rank must be < world.
    let mut send =
        OpStatic::from_op(&ir_op_for(OpId::Send), &facts_for(OpId::Send)).expect("send lowers");
    if let OpStatic::Collectives(CollectivesStatic::Send(ref mut s)) = send {
        s.rank = s.world;
    }
    assert!(send.validate().is_err(), "rank >= world must fail");

    // AllReduce reduce_in must be f32.
    let mut ar = OpStatic::from_op(&ir_op_for(OpId::AllReduce), &facts_for(OpId::AllReduce))
        .expect("all_reduce lowers");
    if let OpStatic::Collectives(CollectivesStatic::AllReduce(ref mut s)) = ar {
        s.reduce_in = DType::F16;
    }
    assert!(ar.validate().is_err(), "non-f32 reduce_in must fail");

    // Recv shape must be non-empty with all extents > 0.
    let mut recv =
        OpStatic::from_op(&ir_op_for(OpId::Recv), &facts_for(OpId::Recv)).expect("recv lowers");
    if let OpStatic::Collectives(CollectivesStatic::Recv(ref mut s)) = recv {
        s.shape = vec![];
    }
    assert!(recv.validate().is_err(), "empty recv shape must fail");
    let mut recv2 =
        OpStatic::from_op(&ir_op_for(OpId::Recv), &facts_for(OpId::Recv)).expect("recv lowers");
    if let OpStatic::Collectives(CollectivesStatic::Recv(ref mut s)) = recv2 {
        s.shape = vec![128, 0];
    }
    assert!(recv2.validate().is_err(), "zero recv extent must fail");

    // MoeRoute top_k must not exceed e_total.
    let mut route = OpStatic::from_op(&ir_op_for(OpId::MoeRoute), &facts_for(OpId::MoeRoute))
        .expect("moe_route lowers");
    if let OpStatic::MoeRoute(ref mut s) = route {
        s.top_k = s.e_total + 1;
    }
    assert!(route.validate().is_err(), "top_k > e_total must fail");
}

/// Resolution enforces pairing before lookup with a typed error.
#[test]
fn test_resolve_rejects_mismatched_pair_with_typed_error() {
    let registry = Registry::new(RegistryConfig {
        allow_jit: false,
        ..RegistryConfig::default()
    });
    let arch = ArchName::from("gfx942");
    let matmul = OpStatic::from_op(&ir_op_for(OpId::Matmul), &facts_for(OpId::Matmul))
        .expect("matmul lowers");
    match registry.resolve(OpId::Sample, &arch, &matmul) {
        Err(RegistryError::StaticOpMismatch { op, static_op }) => {
            assert_eq!(op, OpId::Sample);
            assert_eq!(static_op, OpId::Matmul);
        }
        Err(other) => panic!("expected StaticOpMismatch, got {other:?}"),
        Ok(_) => panic!("mismatched resolve must fail"),
    }
    // Correct pairing passes the pair gate and reaches arch refusal on an empty registry.
    match registry.resolve(OpId::Matmul, &arch, &matmul) {
        Err(RegistryError::UnlistedArchRefused { .. }) => {}
        Err(other) => panic!("expected UnlistedArchRefused, got {other:?}"),
        Ok(_) => panic!("empty registry must not resolve"),
    }
}

/// Manifest entry selection enforces op-tag agreement with typed errors.
#[test]
fn test_manifest_entry_pair_checks() {
    let mut manifest = BundleManifest::new(1, vec![ArchName::from("gfx942")]);
    let _ = &mut manifest;
    let entry = r9v_registry::ManifestVariantEntry {
        arch: ArchName::from("gfx942"),
        file: "kernels/a.co".to_string(),
        tier: r9v_registry::Tier::T2,
        entry_symbol: "k".to_string(),
        launch_geometry: r9v_registry::LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        workspace_bytes: 0,
        static_bytes: 0,
        static_flops: 0,
        op: Some(OpId::Matmul),
        static_hash: Some(42),
        validated: true,
        validated_on: None,
    };
    assert!(entry.matches_request(OpId::Matmul));
    assert!(!entry.matches_request(OpId::Sample));
    BundleManifest::check_entry_for(&entry, OpId::Matmul).expect("matching tag passes");
    match BundleManifest::check_entry_for(&entry, OpId::Sample) {
        Err(RegistryError::StaticOpMismatch { op, static_op }) => {
            assert_eq!(op, OpId::Sample);
            assert_eq!(static_op, OpId::Matmul);
        }
        Err(other) => panic!("expected StaticOpMismatch, got {other:?}"),
        Ok(()) => panic!("wrong-tag entry must fail"),
    }
}

/// Same-hash-before / different-hash-after flips for every newly represented attribute.
#[test]
fn test_new_attribute_flips_change_hash() {
    let flip = |op: OpId, mutate: &dyn Fn(&mut OpStatic)| {
        let base = OpStatic::from_op(&ir_op_for(op), &facts_for(op)).expect("lowers");
        let before = static_hash(&base);
        let mut after = base.clone();
        mutate(&mut after);
        assert_ne!(
            before,
            static_hash(&after),
            "flip must change hash for {op}"
        );
        assert_eq!(
            static_hash(&base),
            before,
            "same descriptor must hash equal"
        );
    };

    flip(OpId::Matmul, &|s| {
        if let OpStatic::Matmul(m) = s {
            m.in_dtype = DType::Bf16;
        }
    });
    flip(OpId::Matmul, &|s| {
        if let OpStatic::Matmul(m) = s {
            m.w_dtype = DType::I8;
        }
    });
    flip(OpId::MoeRoute, &|s| {
        if let OpStatic::MoeRoute(m) = s {
            m.has_bias = true;
        }
    });
    flip(OpId::MoeFfn, &|s| {
        if let OpStatic::MoeFfn(m) = s {
            m.act = ActivationKind::Gelu;
        }
    });
    flip(OpId::MoeFfn, &|s| {
        if let OpStatic::MoeFfn(m) = s {
            m.shared_experts = 1;
        }
    });
    flip(OpId::MoeFfn, &|s| {
        if let OpStatic::MoeFfn(m) = s {
            m.gate_up.layout = LayoutId::L0;
        }
    });
    flip(OpId::MoeFfn, &|s| {
        if let OpStatic::MoeFfn(m) = s {
            m.down.dtype = DType::I8;
            m.down.scheme = QuantScheme::PerRow;
        }
    });
    flip(OpId::Attention, &|s| {
        if let OpStatic::Attention(a) = s {
            a.set_softmax_scale(0.5);
        }
    });
    flip(OpId::Attention, &|s| {
        if let OpStatic::Attention(a) = s {
            a.sinks = 2;
        }
    });
    flip(OpId::Attention, &|s| {
        if let OpStatic::Attention(a) = s {
            a.q_dtype = DType::Bf16;
        }
    });
    flip(OpId::Attention, &|s| {
        if let OpStatic::Attention(a) = s {
            a.mla = Some(MlaAttentionStatic {
                q_lora_rank: None,
                kv_lora_rank: 128,
                qk_nope_dim: 96,
                qk_rope_dim: 32,
                v_dim: 128,
            });
        }
    });
    flip(OpId::StateWriteKv, &|s| {
        if let OpStatic::StateWriteKv(sw) = s {
            sw.scale_granularity = CacheScaleGranularity::PerBlock;
        }
    });
    flip(OpId::StateWriteKv, &|s| {
        if let OpStatic::StateWriteKv(sw) = s {
            sw.latent = Some(MlaLatentStatic {
                kv_lora_rank: 128,
                rope_dim: 64,
            });
        }
    });
    flip(OpId::CausalConv1d, &|s| {
        if let OpStatic::CausalConv1d(c) = s {
            c.t_bucket = 32;
        }
    });
    flip(OpId::CausalConv1d, &|s| {
        if let OpStatic::CausalConv1d(c) = s {
            c.x_dtype = DType::Bf16;
        }
    });
    flip(OpId::CausalConv1d, &|s| {
        if let OpStatic::CausalConv1d(c) = s {
            c.w_dtype = DType::Bf16;
        }
    });
    flip(OpId::CausalConv1d, &|s| {
        if let OpStatic::CausalConv1d(c) = s {
            c.w_layout = LayoutId::L0;
        }
    });
    flip(OpId::CausalConv1d, &|s| {
        if let OpStatic::CausalConv1d(c) = s {
            c.bias_dtype = Some(DType::F32);
        }
    });
    flip(OpId::LinearAttnScan, &|s| {
        if let OpStatic::LinearAttnScan(l) = s {
            l.out_dtype = DType::Bf16;
        }
    });
    flip(OpId::ResidualAdd, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::ResidualAdd(r) = &mut e.op_params {
                r.set_scale(0.5);
            }
        }
    });
    flip(OpId::ResidualAdd, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::ResidualAdd(r) = &mut e.op_params {
                r.a_dtype = DType::Bf16;
            }
        }
    });
    flip(OpId::ResidualAdd, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::ResidualAdd(r) = &mut e.op_params {
                r.b_dtype = DType::F32;
            }
        }
    });
    flip(OpId::Norm, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::Norm(n) = &mut e.op_params {
                n.has_bias = true;
            }
        }
    });
    flip(OpId::ScatterAddRows, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::ScatterAddRows(p) = &mut e.op_params {
                p.has_dest = true;
            }
        }
    });
    flip(OpId::NgramGather, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::NgramGather(g) = &mut e.op_params {
                g.scales_dtype = Some(DType::F16);
            }
        }
    });
    flip(OpId::NgramGather, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::NgramGather(g) = &mut e.op_params {
                g.staging_scheme = QuantScheme::Scheme(SchemeId::new(2));
            }
        }
    });
    flip(OpId::EmbedGather, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::EmbedGather(g) = &mut e.op_params {
                g.table_dtype = DType::I8;
            }
        }
    });
    flip(OpId::NgramGather, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::NgramGather(g) = &mut e.op_params {
                g.dn = 256;
            }
        }
    });
    flip(OpId::Split, &|s| {
        if let OpStatic::Elementwise(e) = s {
            if let ElementwiseParams::Split(sp) = &mut e.op_params {
                sp.first = 1024;
            }
        }
    });
    flip(OpId::LogitsPostprocess, &|s| {
        if let OpStatic::Sampling(SamplingStatic::LogitsPostprocess(_)) = s {
            // LogitsPostprocess carries no method; mutate the bucket instead.
            *s = OpStatic::Sampling(SamplingStatic::LogitsPostprocess(LogitsPostprocessStatic {
                s_bucket: 8,
                v: 32000,
                q_bucket: 16,
                has_history_counts: false,
                has_grammar_mask: false,
            }));
        }
    });
    flip(OpId::Verify, &|s| {
        if let OpStatic::Sampling(SamplingStatic::Verify(v)) = s {
            v.tree = true;
        }
    });
    flip(OpId::Verify, &|s| {
        if let OpStatic::Sampling(SamplingStatic::Verify(v)) = s {
            v.has_draft_probs = true;
        }
    });
    flip(OpId::LogitsPostprocess, &|s| {
        if let OpStatic::Sampling(SamplingStatic::LogitsPostprocess(p)) = s {
            p.has_history_counts = true;
        }
    });
    flip(OpId::LogitsPostprocess, &|s| {
        if let OpStatic::Sampling(SamplingStatic::LogitsPostprocess(p)) = s {
            p.has_grammar_mask = true;
        }
    });
    flip(OpId::Matmul, &|s| {
        if let OpStatic::Matmul(m) = s {
            m.epilogue = r9v_ir::Epilogue::Residual;
            m.residual_dtype = Some(DType::F16);
        }
    });
    flip(OpId::Send, &|s| {
        if let OpStatic::Collectives(CollectivesStatic::Send(p)) = s {
            p.peer = 0;
            p.rank = 1;
        }
    });
    flip(OpId::Recv, &|s| {
        if let OpStatic::Collectives(CollectivesStatic::Recv(p)) = s {
            p.shape = vec![64, 4096];
        }
    });
    flip(OpId::Barrier, &|s| {
        if let OpStatic::Collectives(CollectivesStatic::Barrier(b)) = s {
            b.group = 5;
        }
    });
}

/// Contradictory IR/facts pairs fail typed at lowering, never silently win.
#[test]
fn test_from_op_rejects_contradictory_facts() {
    let must_mismatch =
        |op: r9v_ir::Op, facts: StaticFacts, why: &str| match OpStatic::from_op(&op, &facts) {
            Err(RegistryError::FactsOpMismatch { .. }) => {}
            Err(other) => panic!("{why} must be FactsOpMismatch, got {other:?}"),
            Ok(_) => panic!("{why} must fail"),
        };

    // Matmul residual presence must match a Residual epilogue.
    let mut residual_facts = facts_for(OpId::Matmul);
    if let StaticFacts::Matmul(ref mut f) = residual_facts {
        f.residual_dtype = Some(DType::F16);
    }
    must_mismatch(
        ir_op_for(OpId::Matmul),
        residual_facts,
        "residual facts with a None epilogue",
    );
    let residual_ir = r9v_ir::Op::Matmul(r9v_ir::MatmulOp {
        out_dtype: DType::F16,
        epilogue: r9v_ir::Epilogue::Residual,
        transpose_w: false,
    });
    must_mismatch(
        residual_ir,
        facts_for(OpId::Matmul),
        "Residual epilogue without residual facts",
    );

    // Ngram scales presence must match Staged source.
    let mut staged_facts = facts_for(OpId::NgramGather);
    if let StaticFacts::Elementwise {
        params: ElementwiseFacts::NgramGather { scales_dtype, .. },
        ..
    } = &mut staged_facts
    {
        *scales_dtype = None;
    }
    must_mismatch(
        ir_op_for(OpId::NgramGather),
        staged_facts,
        "Staged source without scales facts",
    );
    let device_ir = r9v_ir::Op::NgramGather(r9v_ir::NgramGatherOp {
        source: NgramSource::Device,
        orders: vec![2u32, 3u32].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(7),
        table_sizes: vec![1024u32, 1024u32].into_boxed_slice(),
        combine: NgramCombine::Sum,
        out_dtype: DType::F16,
    });
    must_mismatch(
        device_ir,
        facts_for(OpId::NgramGather),
        "Device source with scales facts",
    );

    // Recv rank mismatch still fails typed.
    let mut rank_facts = facts_for(OpId::Recv);
    if let StaticFacts::Collectives(CollectiveFacts::Recv { shape, .. }) = &mut rank_facts {
        shape.push(8);
    }
    must_mismatch(
        ir_op_for(OpId::Recv),
        rank_facts,
        "recv facts rank must match IR rank",
    );

    // Recv concrete extent mismatch fails typed.
    let mut extent_facts = facts_for(OpId::Recv);
    if let StaticFacts::Collectives(CollectiveFacts::Recv { shape, .. }) = &mut extent_facts {
        shape[1] = 2048;
    }
    must_mismatch(
        ir_op_for(OpId::Recv),
        extent_facts,
        "recv facts extent must equal the concrete IR extent",
    );

    // Recv zero extents fail typed.
    let mut zero_facts = facts_for(OpId::Recv);
    if let StaticFacts::Collectives(CollectiveFacts::Recv { shape, .. }) = &mut zero_facts {
        shape[1] = 0;
    }
    must_mismatch(ir_op_for(OpId::Recv), zero_facts, "recv zero extent");

    // Recv overflowing element counts fail typed.
    let wide_ir = r9v_ir::Op::Recv(r9v_ir::RecvOp {
        group: GroupId::new(0),
        peer: 0,
        shape: vec![
            Dim::Symbolic(ShapeSymbol::T),
            Dim::Symbolic(ShapeSymbol::S),
            Dim::Symbolic(ShapeSymbol::Dm),
        ]
        .into_boxed_slice(),
        dtype: DType::F16,
    });
    let wide_facts = StaticFacts::Collectives(CollectiveFacts::Recv {
        rank: 1,
        world: 2,
        transport: r9v_ir::P2pTransport::Direct,
        bytes_bucket: 65536,
        shape: vec![u32::MAX, u32::MAX, u32::MAX],
    });
    must_mismatch(wide_ir, wide_facts, "recv overflowing shape product");
}

/// Symbolic IR extents resolve to the concrete facts extents.
#[test]
fn test_recv_symbolic_extents_resolve_to_facts() {
    let ir = r9v_ir::Op::Recv(r9v_ir::RecvOp {
        group: GroupId::new(0),
        peer: 0,
        shape: vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(4096)].into_boxed_slice(),
        dtype: DType::F16,
    });
    let lowered = OpStatic::from_op(&ir, &facts_for(OpId::Recv)).expect("symbolic recv lowers");
    match lowered {
        OpStatic::Collectives(CollectivesStatic::Recv(s)) => {
            assert_eq!(s.shape, vec![128, 4096]);
        }
        other => panic!("expected recv, got {other:?}"),
    }
}

/// Illegal (dtype, scheme, layout) combinations are rejected by `validate`.
#[test]
fn test_validate_rejects_illegal_semantic_combinations() {
    let must_fail = |s: OpStatic, why: &str| {
        assert!(s.validate().is_err(), "{why} must fail validation");
    };
    let must_pass = |s: OpStatic, why: &str| {
        assert!(s.validate().is_ok(), "{why} must pass validation");
    };

    // Attention q dtype is f16/bf16/f32 only.
    let mut attention = OpStatic::from_op(&ir_op_for(OpId::Attention), &facts_for(OpId::Attention))
        .expect("attention lowers");
    if let OpStatic::Attention(ref mut a) = attention {
        a.q_dtype = DType::I8;
    }
    must_fail(attention, "i8 attention q_dtype");

    // Residual addends are f16/bf16/f32 only, independently.
    let mut residual =
        OpStatic::from_op(&ir_op_for(OpId::ResidualAdd), &facts_for(OpId::ResidualAdd))
            .expect("residual_add lowers");
    if let OpStatic::Elementwise(ref mut e) = residual {
        if let ElementwiseParams::ResidualAdd(ref mut r) = e.op_params {
            r.a_dtype = DType::I8;
        }
    }
    must_fail(residual, "i8 residual a_dtype");

    // Conv weight legality mirrors the IR float/quantized split.
    let conv = |w_dtype: DType, w_scheme: QuantScheme| {
        let mut facts = facts_for(OpId::CausalConv1d);
        if let StaticFacts::CausalConv1d(ref mut f) = facts {
            f.w_dtype = w_dtype;
            f.w_scheme = w_scheme;
        }
        OpStatic::from_op(&ir_op_for(OpId::CausalConv1d), &facts).expect("conv lowers")
    };
    must_fail(
        conv(DType::F16, QuantScheme::PerRow),
        "f16 conv with PerRow",
    );
    must_fail(conv(DType::I8, QuantScheme::None), "i8 conv without scales");
    must_fail(conv(DType::E4m3, QuantScheme::None), "e4m3 conv weight");
    must_pass(
        conv(DType::Bf16, QuantScheme::None),
        "bf16 conv without scales",
    );
    must_pass(
        conv(DType::I8, QuantScheme::PerRow),
        "i8 conv with PerRow scales",
    );
    must_pass(
        conv(DType::I4, QuantScheme::Scheme(SchemeId::new(1))),
        "i4 conv with block scales",
    );
    let mut biased = conv(DType::F16, QuantScheme::None);
    if let OpStatic::CausalConv1d(ref mut c) = biased {
        c.bias_dtype = Some(DType::Bool);
    }
    must_fail(biased, "bool conv bias_dtype");

    // Matmul residual presence tracks the Residual epilogue. These statics are
    // built literally (not lowered) because `from_op` already rejects the
    // contradictory pairs; `validate` must reject them independently for
    // directly constructed descriptors at resolve time.
    let matmul = |epilogue: r9v_ir::Epilogue, residual_dtype: Option<DType>| {
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
            epilogue,
            residual_dtype,
            transpose_w: false,
            interleave: false,
            sparse: false,
        })
    };
    must_pass(
        matmul(r9v_ir::Epilogue::Residual, Some(DType::Bf16)),
        "Residual epilogue with bf16 residual must validate",
    );
    must_fail(
        matmul(r9v_ir::Epilogue::Residual, None),
        "Residual epilogue without residual dtype",
    );
    must_fail(
        matmul(r9v_ir::Epilogue::None, Some(DType::F16)),
        "residual dtype without Residual epilogue",
    );
    must_fail(
        matmul(r9v_ir::Epilogue::Residual, Some(DType::I8)),
        "i8 residual dtype",
    );

    // Moe projections validate independently per the GEMM weight rule.
    let moe = |gate: MoeFfnProjStatic, down: MoeFfnProjStatic| {
        let mut facts = facts_for(OpId::MoeFfn);
        if let StaticFacts::MoeFfn(ref mut f) = facts {
            f.gate_up = gate;
            f.down = down;
        }
        OpStatic::from_op(&ir_op_for(OpId::MoeFfn), &facts).expect("moe_ffn lowers")
    };
    let f16_none = MoeFfnProjStatic {
        dtype: DType::F16,
        scheme: QuantScheme::None,
        layout: LayoutId::L1,
    };
    let i8_row = MoeFfnProjStatic {
        dtype: DType::I8,
        scheme: QuantScheme::PerRow,
        layout: LayoutId::L1,
    };
    must_pass(
        moe(i8_row, f16_none),
        "mixed quantized gate_up and dense down",
    );
    must_fail(
        moe(
            MoeFfnProjStatic {
                scheme: QuantScheme::PerRow,
                ..f16_none
            },
            f16_none,
        ),
        "f16 gate_up with PerRow scales",
    );
    must_fail(
        moe(
            f16_none,
            MoeFfnProjStatic {
                scheme: QuantScheme::None,
                ..i8_row
            },
        ),
        "i8 down without scales",
    );
    must_fail(
        moe(
            MoeFfnProjStatic {
                dtype: DType::F32,
                ..f16_none
            },
            f16_none,
        ),
        "f32 gate_up weight",
    );

    // Ngram Staged mode requires block scales and f32/f16 row scales.
    let mut ngram = OpStatic::from_op(&ir_op_for(OpId::NgramGather), &facts_for(OpId::NgramGather))
        .expect("ngram lowers");
    if let OpStatic::Elementwise(ref mut e) = ngram {
        if let ElementwiseParams::NgramGather(ref mut g) = e.op_params {
            g.staging_scheme = QuantScheme::None;
        }
    }
    must_fail(ngram, "Staged ngram without block staging scheme");
    let mut ngram = OpStatic::from_op(&ir_op_for(OpId::NgramGather), &facts_for(OpId::NgramGather))
        .expect("ngram lowers");
    if let OpStatic::Elementwise(ref mut e) = ngram {
        if let ElementwiseParams::NgramGather(ref mut g) = e.op_params {
            g.scales_dtype = Some(DType::I8);
        }
    }
    must_fail(ngram, "Staged ngram with i8 scales dtype");
    let device_ir = r9v_ir::Op::NgramGather(r9v_ir::NgramGatherOp {
        source: NgramSource::Device,
        orders: vec![2u32, 3u32].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(7),
        table_sizes: vec![1024u32, 1024u32].into_boxed_slice(),
        combine: NgramCombine::Sum,
        out_dtype: DType::F16,
    });
    let mut device_facts = facts_for(OpId::NgramGather);
    if let StaticFacts::Elementwise {
        params: ElementwiseFacts::NgramGather { scales_dtype, .. },
        ..
    } = &mut device_facts
    {
        *scales_dtype = None;
    }
    must_pass(
        OpStatic::from_op(&device_ir, &device_facts).expect("device ngram lowers"),
        "Device-table ngram without scales must validate",
    );
}

/// Gate/up and down projections are independent kernel semantics.
#[test]
fn test_moe_ffn_projections_are_independent() {
    let base = OpStatic::from_op(&ir_op_for(OpId::MoeFfn), &facts_for(OpId::MoeFfn))
        .expect("moe_ffn lowers");
    // Swapping the two projections changes identity: order is semantic.
    let mut swapped = base.clone();
    if let OpStatic::MoeFfn(ref mut m) = swapped {
        m.gate_up = MoeFfnProjStatic {
            dtype: DType::I8,
            scheme: QuantScheme::PerRow,
            layout: LayoutId::L0,
        };
        m.down = MoeFfnProjStatic {
            dtype: DType::F16,
            scheme: QuantScheme::None,
            layout: LayoutId::L1,
        };
    }
    let mut unswapped = base.clone();
    if let OpStatic::MoeFfn(ref mut m) = unswapped {
        m.gate_up = MoeFfnProjStatic {
            dtype: DType::F16,
            scheme: QuantScheme::None,
            layout: LayoutId::L1,
        };
        m.down = MoeFfnProjStatic {
            dtype: DType::I8,
            scheme: QuantScheme::PerRow,
            layout: LayoutId::L0,
        };
    }
    assert_ne!(
        static_hash(&swapped),
        static_hash(&unswapped),
        "swapped projections must hash differently"
    );
    assert!(
        swapped.validate().is_ok() && unswapped.validate().is_ok(),
        "both mixed projection orders must validate"
    );
}

/// Optional input dtypes survive canonical serde round-trips bit-exactly.
#[test]
fn test_optional_dtype_serde_roundtrip() {
    let residual_ir = r9v_ir::Op::Matmul(r9v_ir::MatmulOp {
        out_dtype: DType::F16,
        epilogue: r9v_ir::Epilogue::Residual,
        transpose_w: false,
    });
    let mut residual_facts = facts_for(OpId::Matmul);
    if let StaticFacts::Matmul(ref mut f) = residual_facts {
        f.residual_dtype = Some(DType::Bf16);
    }
    let biased_conv = {
        let mut facts = facts_for(OpId::CausalConv1d);
        if let StaticFacts::CausalConv1d(ref mut f) = facts {
            f.bias_dtype = Some(DType::F16);
        }
        OpStatic::from_op(&ir_op_for(OpId::CausalConv1d), &facts).expect("biased conv lowers")
    };
    for op_s in [
        OpStatic::from_op(&residual_ir, &residual_facts).expect("residual lowers"),
        biased_conv,
        OpStatic::from_op(&ir_op_for(OpId::NgramGather), &facts_for(OpId::NgramGather))
            .expect("ngram lowers"),
    ] {
        let json = serde_json::to_string(&op_s).expect("static must serialize");
        let back: OpStatic = serde_json::from_str(&json).expect("static must deserialize");
        assert_eq!(op_s, back, "optional dtypes must round-trip in {json}");
        assert_eq!(
            static_hash(&op_s),
            static_hash(&back),
            "hash must survive serde round-trip"
        );
    }
}
