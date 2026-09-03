// SPDX-License-Identifier: Apache-2.0
//! API-shape tests for r9v-ir (Spec 1 §2, §3, §4, §5, §6; App. A; r9v-card-work §6; card A1.2).
//!
//! Asserts compilation, visibility boundaries and the `Send`/`Sync` markers
//! downstream crates rely on. Closed-set enums are additionally matched
//! exhaustively (no wildcard) so an RFC-added variant breaks this test until
//! the new surface is handled deliberately.

use std::hash::Hash;

use r9v_ir::{
    bucket_s, bucket_step, bucket_t_dec, bucket_t_pre, compute_contiguous_strides, fusion_table,
    is_permitted_fusion, is_permitted_pair, legal_layout_tuples, legal_layouts, match_chain,
    match_gated_pair, matmul_numerics, moe_ffn_gemm_numerics, ActMulOp, ActivationKind,
    ActivationOp, AllGatherOp, AllReduceOp, AllToAllOp, ArchDescriptor, ArchFamily, AttentionMask,
    AttentionOp, BarrierOp, BatchMeta, BatchMetaBuilder, CacheScaleGranularity, CastOp,
    CausalConv1dOp, Class, ConcatOp, ConvActivation, CopyKind, CopyOp, DType, Dim, EdgeId,
    EmbedGatherOp, Epilogue, ExpertCount, ExternalInput, ExternalInputKind, ExternalOutput,
    ExternalOutputKind, FusionEntry, FusionPattern, GatherRowsOp, Graph, GraphCapture, GraphEdge,
    GraphNode, GraphSummary, GroupId, HashId, HeadCount, InsertedCopy, IrError, IrVersion,
    LayoutId, LinearAttnKind, LinearAttnScanOp, LogitSoftcapOp, LogitsPostprocessOp, MatmulOp,
    MatrixOp, Measured, MlaAttentionSpec, MlaLatent, MoeFfnOp, MoeGroup, MoeRouteOp, MoeScoring,
    NgramCombine, NgramGatherOp, NgramSource, NodeId, NormAxis, NormKind, NormOp, Numerics, Op,
    P2pLink, P2pTransport, Placement, PlanId, Positions, PositionsKind, QuantActOp, QuantScheme,
    RecvOp, ReduceOp, ReduceScatterOp, ReductionOrder, RelRate, ResidualAddOp, RngAlgorithm,
    RopeOp, RopeScaling, RopeStyle, SampleOp, SamplingParams, ScatterAddRowsOp, SchemeId, SendOp,
    ShapeSymbol, ShardLayout, ShardLayoutPattern, ShardingRule, Smoothing, SplitOp, StateHandle,
    StateKind, StateWriteKvOp, StepGraphKey, StrideRequirement, Tensor, TreeMask, ValuDot,
    VerifyMethod, VerifyOp, BLOCK_TABLE_SENTINEL, BUCKET_SIZES, FUSION_TABLE,
};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_copy<T: Copy>() {}
fn assert_clone<T: Clone>() {}
fn assert_debug<T: std::fmt::Debug>() {}
fn assert_display<T: std::fmt::Display>() {}
fn assert_hash<T: Hash>() {}
fn assert_error<T: std::error::Error>() {}
fn assert_ord<T: Ord>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn api_shape_markers_and_errors() {
    assert_send::<DType>();
    assert_sync::<DType>();
    assert_copy::<DType>();
    assert_hash::<DType>();
    assert_display::<DType>();

    assert_send::<SchemeId>();
    assert_sync::<SchemeId>();
    assert_copy::<SchemeId>();
    assert_hash::<SchemeId>();
    assert_display::<SchemeId>();

    assert_send::<QuantScheme>();
    assert_sync::<QuantScheme>();
    assert_copy::<QuantScheme>();
    assert_hash::<QuantScheme>();

    assert_send::<LayoutId>();
    assert_sync::<LayoutId>();
    assert_copy::<LayoutId>();
    assert_hash::<LayoutId>();
    assert_display::<LayoutId>();

    assert_send::<Tensor>();
    assert_sync::<Tensor>();
    assert_clone::<Tensor>();
    assert_debug::<Tensor>();

    assert_send::<BatchMeta>();
    assert_sync::<BatchMeta>();
    assert_clone::<BatchMeta>();

    assert_send::<TreeMask>();
    assert_sync::<TreeMask>();
    assert_clone::<TreeMask>();

    assert_send::<StateHandle>();
    assert_sync::<StateHandle>();
    assert_copy::<StateHandle>();
    assert_hash::<StateHandle>();

    assert_send::<ArchDescriptor>();
    assert_sync::<ArchDescriptor>();
    assert_clone::<ArchDescriptor>();

    assert_send::<IrVersion>();
    assert_sync::<IrVersion>();
    assert_copy::<IrVersion>();
    assert_hash::<IrVersion>();
    assert_display::<IrVersion>();
    assert_ord::<IrVersion>();

    assert_send::<IrError>();
    assert_sync::<IrError>();
    assert_clone::<IrError>();
    assert_error::<IrError>();

    assert_send::<BatchMetaBuilder>();
    assert_sync::<BatchMetaBuilder>();

    assert_send_sync::<ShapeSymbol>();
    assert_send_sync::<Dim>();
    assert_send_sync::<Placement>();
    assert_send_sync::<ShardLayout>();
    assert_send_sync::<Class>();
    assert_send_sync::<Positions>();
    assert_send_sync::<StateKind>();
    assert_send_sync::<ArchFamily>();
    assert_send_sync::<RelRate>();
    assert_send_sync::<MatrixOp>();
    assert_send_sync::<ValuDot>();
    assert_send_sync::<GraphCapture>();
    assert_send_sync::<P2pTransport>();
    assert_send_sync::<P2pLink>();
    assert_send_sync::<Measured>();

    // A1.2 markers
    assert_send_sync::<PlanId>();
    assert_copy::<PlanId>();
    assert_hash::<PlanId>();
    assert_ord::<PlanId>();

    assert_send_sync::<StepGraphKey>();
    assert_copy::<StepGraphKey>();
    assert_hash::<StepGraphKey>();
    assert_ord::<StepGraphKey>();

    assert_send_sync::<Numerics>();
    assert_copy::<Numerics>();
    assert_hash::<Numerics>();

    assert_send_sync::<ReductionOrder>();
    assert_copy::<ReductionOrder>();
    assert_hash::<ReductionOrder>();

    assert_send_sync::<FusionPattern>();
    assert_copy::<FusionPattern>();
    assert_hash::<FusionPattern>();

    assert_send_sync::<FusionEntry>();
    assert_copy::<FusionEntry>();

    assert_send_sync::<ShardingRule>();
    assert_copy::<ShardingRule>();

    assert_send_sync::<ExternalInputKind>();
    assert_send_sync::<PositionsKind>();
    assert_copy::<ExternalInputKind>();
    assert_hash::<ExternalInputKind>();

    assert_send_sync::<ExternalOutputKind>();
    assert_copy::<ExternalOutputKind>();
    assert_hash::<ExternalOutputKind>();

    assert_send_sync::<NodeId>();
    assert_copy::<NodeId>();
    assert_hash::<NodeId>();

    assert_send_sync::<EdgeId>();
    assert_copy::<EdgeId>();
    assert_hash::<EdgeId>();

    assert_send_sync::<ExternalInput>();
    assert_clone::<ExternalInput>();

    assert_send_sync::<ExternalOutput>();
    assert_copy::<ExternalOutput>();
    assert_hash::<ExternalOutput>();

    assert_send_sync::<StrideRequirement>();
    assert_copy::<StrideRequirement>();
    assert_hash::<StrideRequirement>();

    assert_send_sync::<NgramSource>();
    assert_copy::<NgramSource>();
    assert_hash::<NgramSource>();

    assert_send_sync::<CopyKind>();
    assert_copy::<CopyKind>();
    assert_hash::<CopyKind>();

    assert_send_sync::<SamplingParams>();
    assert_clone::<SamplingParams>();

    assert_send_sync::<HeadCount>();
    assert_copy::<HeadCount>();
    assert_hash::<HeadCount>();

    assert_send_sync::<ExpertCount>();
    assert_copy::<ExpertCount>();
    assert_hash::<ExpertCount>();

    assert_send_sync::<ShardLayoutPattern>();
    assert_copy::<ShardLayoutPattern>();
    assert_hash::<ShardLayoutPattern>();

    assert_send_sync::<GraphEdge>();
    assert_send_sync::<GraphNode>();
    assert_send_sync::<InsertedCopy>();
    assert_send_sync::<GraphSummary>();
    assert_send_sync::<Graph>();

    // Closed ops
    assert_send_sync::<Op>();
    assert_send_sync::<EmbedGatherOp>();
    assert_send_sync::<NgramGatherOp>();
    assert_send_sync::<QuantActOp>();
    assert_send_sync::<CastOp>();
    assert_send_sync::<CopyOp>();
    assert_send_sync::<GatherRowsOp>();
    assert_send_sync::<ScatterAddRowsOp>();
    assert_send_sync::<SplitOp>();
    assert_send_sync::<ConcatOp>();
    assert_send_sync::<NormOp>();
    assert_send_sync::<ResidualAddOp>();
    assert_send_sync::<ActMulOp>();
    assert_send_sync::<ActivationOp>();
    assert_send_sync::<LogitSoftcapOp>();
    assert_send_sync::<RopeOp>();
    assert_send_sync::<MatmulOp>();
    assert_send_sync::<MoeRouteOp>();
    assert_send_sync::<MoeFfnOp>();
    assert_send_sync::<StateWriteKvOp>();
    assert_send_sync::<AttentionOp>();
    assert_send_sync::<CausalConv1dOp>();
    assert_send_sync::<LinearAttnScanOp>();
    assert_send_sync::<LogitsPostprocessOp>();
    assert_send_sync::<SampleOp>();
    assert_send_sync::<VerifyOp>();
    assert_send_sync::<AllReduceOp>();
    assert_send_sync::<AllGatherOp>();
    assert_send_sync::<ReduceScatterOp>();
    assert_send_sync::<AllToAllOp>();
    assert_send_sync::<SendOp>();
    assert_send_sync::<RecvOp>();
    assert_send_sync::<BarrierOp>();
    assert_send_sync::<MlaAttentionSpec>();
    assert_send_sync::<MlaLatent>();
    assert_send_sync::<MoeGroup>();

    let _ = BLOCK_TABLE_SENTINEL;
}

#[test]
fn api_shape_closed_sets_match_exhaustively() {
    // No wildcard arms: adding a variant fails this test until handled.
    let dtype = DType::F32;
    let _ = match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::Bf16 => "bf16",
        DType::E4m3 => "e4m3",
        DType::E5m2 => "e5m2",
        DType::I8 => "i8",
        DType::I4 => "i4",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::Bool => "bool",
    };

    let scheme = QuantScheme::None;
    let _ = match scheme {
        QuantScheme::None => 0,
        QuantScheme::PerRow => 1,
        QuantScheme::Scheme(_) => 2,
        QuantScheme::PerToken => 3,
        QuantScheme::PerBlock32 => 4,
    };

    let class = Class::Weight;
    let _ = match class {
        Class::Weight => 0,
        Class::Activation => 1,
        Class::State => 2,
        Class::Staging => 3,
        Class::Param => 4,
    };

    let sharding = ShardLayout::Replicated;
    let _ = match sharding {
        ShardLayout::Replicated => 0,
        ShardLayout::ColShard { axis: _ } => 1,
        ShardLayout::RowShard { axis: _ } => 2,
        ShardLayout::HeadShard { heads: _ } => 3,
        ShardLayout::ExpertShard { experts: _ } => 4,
        ShardLayout::Partial => 5,
    };

    let placement = Placement::Host;
    let _ = match placement {
        Placement::Device { rank: _ } => 0,
        Placement::Host => 1,
        Placement::Tiered => 2,
    };

    let kind = StateKind::KvPaged;
    let _ = match kind {
        StateKind::KvPaged => 0,
        StateKind::KvLatent => 1,
        StateKind::Recurrent => 2,
        StateKind::ConvWindow => 3,
    };

    let family = ArchFamily::Rdna4;
    let _ = match family {
        ArchFamily::Rdna4 => 0,
        ArchFamily::Rdna3 => 1,
        ArchFamily::Cdna3 => 2,
        ArchFamily::Reference => 3,
        ArchFamily::Cpu => 4,
    };

    let dot = ValuDot::Dot4I32I8;
    let _ = match dot {
        ValuDot::Dot4I32I8 => 0,
        ValuDot::Dot2F32F16 => 1,
        ValuDot::Dot2F32Bf16 => 2,
    };

    let capture = GraphCapture::Supported;
    let _ = match capture {
        GraphCapture::Supported => 0,
        GraphCapture::Unstable => 1,
        GraphCapture::None => 2,
    };

    let transport = P2pTransport::Direct;
    let _ = match transport {
        P2pTransport::Direct => 0,
        P2pTransport::HostStaged => 1,
    };

    let dim = Dim::Concrete(1);
    let _ = match dim {
        Dim::Concrete(_) => 0,
        Dim::Symbolic(_) => 1,
    };

    let symbol = ShapeSymbol::T;
    let _ = match symbol {
        ShapeSymbol::T => 0,
        ShapeSymbol::S => 1,
        ShapeSymbol::Dm => 2,
        ShapeSymbol::Dff => 3,
        ShapeSymbol::H => 4,
        ShapeSymbol::Hkv => 5,
        ShapeSymbol::D => 6,
        ShapeSymbol::E => 7,
        ShapeSymbol::K => 8,
        ShapeSymbol::V => 9,
        ShapeSymbol::Np => 10,
        ShapeSymbol::L => 11,
    };

    let pos = Positions::PerToken(vec![]);
    let _ = match pos {
        Positions::PerToken(_) => 0,
        Positions::Mrope(_) => 1,
    };

    // A1.2 closed sets exhaustive matching
    let red_order = ReductionOrder::None;
    let _ = match red_order {
        ReductionOrder::None => 0,
        ReductionOrder::AscendingK => 1,
        ReductionOrder::AscendingBlock => 2,
        ReductionOrder::AscendingAxis => 3,
        ReductionOrder::AscendingRank => 4,
        ReductionOrder::AscendingIndex => 5,
    };

    let fusion_pat = FusionPattern::ResidualAddNorm;
    let _ = match fusion_pat {
        FusionPattern::ResidualAddNorm => 0,
        FusionPattern::NormQuantAct => 1,
        FusionPattern::MatmulEpilogue => 2,
        FusionPattern::GatedMatmulActMul => 3,
        FusionPattern::RopeStateWriteKv => 4,
        FusionPattern::RopeAttentionPrefill => 5,
        FusionPattern::StateWriteKvAttentionDecode => 6,
        FusionPattern::LogitsPostprocessSample => 7,
        FusionPattern::QuantActAllToAll => 8,
    };

    let ext_in = ExternalInputKind::TokenIds;
    let _ = match ext_in {
        ExternalInputKind::TokenIds => 0,
        ExternalInputKind::BatchMeta => 1,
        ExternalInputKind::RngState => 2,
        ExternalInputKind::GatherStaging => 3,
        ExternalInputKind::GrammarMask => 4,
        ExternalInputKind::SamplingParams => 5,
        ExternalInputKind::EmbedOverride => 6,
        ExternalInputKind::EmbedMask => 7,
        ExternalInputKind::SubgraphHidden => 8,
    };

    let ext_out = ExternalOutputKind::Sampled;
    let _ = match ext_out {
        ExternalOutputKind::Sampled => 0,
        ExternalOutputKind::AcceptLen => 1,
        ExternalOutputKind::Logits => 2,
        ExternalOutputKind::Hidden => 3,
        ExternalOutputKind::UpdatedRngState => 4,
    };

    let norm_kind = NormKind::Rms;
    let _ = match norm_kind {
        NormKind::Rms => 0,
        NormKind::Layer => 1,
    };

    let norm_axis = NormAxis::Last;
    let _ = match norm_axis {
        NormAxis::Last => 0,
        NormAxis::Head(_) => 1,
    };

    let act_kind = ActivationKind::Silu;
    let _ = match act_kind {
        ActivationKind::Silu => 0,
        ActivationKind::Gelu => 1,
        ActivationKind::GeluTanh => 2,
        ActivationKind::Relu2 => 3,
        ActivationKind::Identity => 4,
    };

    let rope_style = RopeStyle::Neox;
    let _ = match rope_style {
        RopeStyle::Neox => 0,
        RopeStyle::Interleaved => 1,
    };

    let rope_scale = RopeScaling::None;
    let _ = match rope_scale {
        RopeScaling::None => 0,
        RopeScaling::Linear(_) => 1,
        RopeScaling::Yarn {
            factor: _,
            beta_fast: _,
            beta_slow: _,
            orig_ctx: _,
            mscale: _,
        } => 2,
        RopeScaling::Dynamic => 3,
    };

    let epilogue = Epilogue::None;
    let _ = match epilogue {
        Epilogue::None => 0,
        Epilogue::Bias => 1,
        Epilogue::Residual => 2,
        Epilogue::Act(_) => 3,
    };

    let moe_scoring = MoeScoring::Softmax;
    let _ = match moe_scoring {
        MoeScoring::Softmax => 0,
        MoeScoring::Sigmoid => 1,
    };

    let cache_scale = CacheScaleGranularity::PerTokenHead;
    let _ = match cache_scale {
        CacheScaleGranularity::PerTokenHead => 0,
        CacheScaleGranularity::PerBlock => 1,
    };

    let attn_mask = AttentionMask::Causal;
    let _ = match attn_mask {
        AttentionMask::Causal => 0,
        AttentionMask::CausalWindow(_) => 1,
        AttentionMask::Tree => 2,
    };

    let conv_act = ConvActivation::Silu;
    let _ = match conv_act {
        ConvActivation::Silu => 0,
        ConvActivation::Identity => 1,
    };

    let lin_attn = LinearAttnKind::GLA;
    let _ = match lin_attn {
        LinearAttnKind::GatedDeltaNet => 0,
        LinearAttnKind::GLA => 1,
        LinearAttnKind::Mamba2 => 2,
    };

    let rng = RngAlgorithm::Philox4x32;
    let _ = match rng {
        RngAlgorithm::Philox4x32 => 0,
    };

    let verify = VerifyMethod::Greedy;
    let _ = match verify {
        VerifyMethod::Rejection => 0,
        VerifyMethod::Greedy => 1,
        VerifyMethod::TypicalAcceptance { eps: _, delta: _ } => 2,
    };

    let ext_input = ExternalInput::BatchMeta;
    let _ = match ext_input {
        ExternalInput::Tensor { kind: _, edge: _ } => 0,
        ExternalInput::BatchMeta => 1,
        ExternalInput::SamplingParams => 2,
        ExternalInput::RngState => 3,
    };

    let ext_output = ExternalOutput::UpdatedRngState;
    let _ = match ext_output {
        ExternalOutput::Tensor { kind: _, edge: _ } => 0,
        ExternalOutput::UpdatedRngState => 1,
    };

    let stride_req = StrideRequirement::Contiguous;
    let _ = match stride_req {
        StrideRequirement::Any => 0,
        StrideRequirement::Contiguous => 1,
    };

    let ngram_src = NgramSource::Staged;
    let _ = match ngram_src {
        NgramSource::Staged => 0,
        NgramSource::Device => 1,
    };

    let copy_k = CopyKind::Contiguize;
    let _ = match copy_k {
        CopyKind::Contiguize => 0,
        CopyKind::DeviceToDevice => 1,
        CopyKind::HostToDevice => 2,
        CopyKind::DeviceToHost => 3,
    };

    let head_c = HeadCount::Symbolic;
    let _ = match head_c {
        HeadCount::Concrete(_) => 0,
        HeadCount::Symbolic => 1,
    };

    let expert_c = ExpertCount::Symbolic;
    let _ = match expert_c {
        ExpertCount::Concrete(_) => 0,
        ExpertCount::Symbolic => 1,
    };

    let shard_pat = ShardLayoutPattern::Replicated;
    let _ = match shard_pat {
        ShardLayoutPattern::Replicated => 0,
        ShardLayoutPattern::ColShard { axis: _ } => 1,
        ShardLayoutPattern::RowShard { axis: _ } => 2,
        ShardLayoutPattern::HeadShard { heads: _ } => 3,
        ShardLayoutPattern::ExpertShard { experts: _ } => 4,
        ShardLayoutPattern::Partial => 5,
    };

    let reduce_op = ReduceOp::Sum;
    let _ = match reduce_op {
        ReduceOp::Sum => 0,
    };

    let ngram_comb = NgramCombine::Concat;
    let _ = match ngram_comb {
        NgramCombine::Concat => 0,
        NgramCombine::Sum => 1,
    };

    let smoothing = Smoothing::None;
    let _ = match smoothing {
        Smoothing::None => 0,
        Smoothing::Folded => 1,
    };

    let dummy_op = Op::Copy(CopyOp::default());
    let _ = match dummy_op {
        Op::EmbedGather(_) => 0,
        Op::NgramGather(_) => 1,
        Op::QuantAct(_) => 2,
        Op::Cast(_) => 3,
        Op::Copy(_) => 4,
        Op::GatherRows(_) => 5,
        Op::ScatterAddRows(_) => 6,
        Op::Split(_) => 29,
        Op::Concat(_) => 30,
        Op::Norm(_) => 7,
        Op::ResidualAdd(_) => 8,
        Op::ActMul(_) => 9,
        Op::Activation(_) => 10,
        Op::LogitSoftcap(_) => 31,
        Op::Rope(_) => 11,
        Op::Matmul(_) => 12,
        Op::MoeRoute(_) => 13,
        Op::MoeFfn(_) => 14,
        Op::StateWriteKv(_) => 15,
        Op::Attention(_) => 16,
        Op::CausalConv1d(_) => 17,
        Op::LinearAttnScan(_) => 18,
        Op::LogitsPostprocess(_) => 19,
        Op::Sample(_) => 20,
        Op::Verify(_) => 21,
        Op::AllReduce(_) => 22,
        Op::AllGather(_) => 23,
        Op::ReduceScatter(_) => 24,
        Op::AllToAll(_) => 25,
        Op::Send(_) => 26,
        Op::Recv(_) => 27,
        Op::Barrier(_) => 28,
    };
}

#[test]
fn api_shape_constructors_are_reachable() {
    let scheme = SchemeId::new(7);
    assert_eq!(scheme.as_u64(), 7);

    let layout = LayoutId::new(9);
    assert_eq!(layout.as_u64(), 9);
    assert_eq!(LayoutId::L1, LayoutId::new(2));

    let handle = StateHandle::new(3, StateKind::KvLatent);
    assert_eq!(handle.layer(), 3);
    assert_eq!(handle.kind(), StateKind::KvLatent);

    let tensor = Tensor::new(
        vec![Dim::Concrete(4), Dim::Symbolic(ShapeSymbol::Dm)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect("valid tensor builds");
    assert_eq!(tensor.rank(), 2);

    let rate = RelRate::new(2.0).expect("positive rate builds");
    assert_eq!(rate.as_f32(), 2.0);

    let op = MatrixOp::new([16, 16, 16], DType::I8, DType::I8, DType::I32, rate)
        .expect("valid matrix op builds");
    assert_eq!(op.shape, [16, 16, 16]);

    let measured = Measured::empty();
    assert!(measured.is_empty());

    let link = P2pLink {
        peer_rank: 1,
        transport: P2pTransport::HostStaged,
        measured_gbps: None,
    };
    assert_eq!(link.peer_rank, 1);

    let gfx = ArchDescriptor::gfx1201();
    assert_eq!(gfx.name, "gfx1201");
    let cpu = ArchDescriptor::cpu();
    assert_eq!(cpu.family, ArchFamily::Cpu);

    assert_eq!(IrVersion::CURRENT, IrVersion::new(0, 2, 0));
    assert_eq!(IrVersion::CURRENT.to_string(), "0.2.0");

    let plan = PlanId::new(42);
    assert_eq!(plan.as_u64(), 42);

    let key = StepGraphKey::new(plan, 0, 16, 16, 0, 0).expect("valid key builds");
    assert_eq!(key.s, 16);
    let mut graph = Graph::new(key);
    let source = Tensor::new(
        vec![Dim::Concrete(4), Dim::Symbolic(ShapeSymbol::Dm)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .expect("graph-owned weight source builds");
    let source_edge = graph.add_tensor(source).expect("weight source registers");
    assert_eq!(source_edge, EdgeId(0));
    let reshaped_edge = graph
        .reshape_edge(
            source_edge,
            vec![Dim::Concrete(4), Dim::Symbolic(ShapeSymbol::Dm)],
        )
        .expect("metadata reshape API is reachable");
    assert_eq!(reshaped_edge, EdgeId(1));
    assert!(graph.topological_order().unwrap().is_empty());

    let hash = HashId::new(101);
    assert_eq!(hash.as_u64(), 101);

    let group = GroupId::new(3);
    assert_eq!(group.as_u64(), 3);
    assert_eq!(group.as_u32(), 3);
    assert_eq!(group.to_string(), "3");

    let num = Numerics::f32(ReductionOrder::AscendingK);
    assert_eq!(num.accumulator, Some(DType::F32));
    assert_eq!(num.reduction_order, ReductionOrder::AscendingK);

    let num_matmul = matmul_numerics(
        DType::I8,
        DType::I8,
        QuantScheme::PerToken,
        QuantScheme::PerRow,
    )
    .unwrap();
    assert_eq!(num_matmul.accumulator, Some(DType::I32));
    assert_eq!(num_matmul.reduction_order, ReductionOrder::AscendingK);

    let num_moe = moe_ffn_gemm_numerics(
        DType::I8,
        DType::I8,
        QuantScheme::PerBlock32,
        QuantScheme::PerRow,
    )
    .unwrap();
    assert_eq!(num_moe.accumulator, Some(DType::I32));
    assert_eq!(num_moe.reduction_order, ReductionOrder::AscendingBlock);

    assert!(is_permitted_fusion(FusionPattern::ResidualAddNorm));
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
    assert_eq!(
        is_permitted_pair(&residual, &norm),
        Some(FusionPattern::ResidualAddNorm)
    );
    assert_eq!(
        match_chain(&[&residual, &norm]),
        Some(FusionPattern::ResidualAddNorm)
    );
    let _gated_matcher: fn(&Op, &Op, &Op) -> Option<FusionPattern> = match_gated_pair;

    let sp = SamplingParams {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        min_p: 0.05,
        repetition_penalty: 1.1,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: vec![(10, 1.5)],
    };
    assert!(sp.validate().is_ok());

    assert_eq!(CopyKind::default(), CopyKind::Contiguize);
    assert_eq!(StrideRequirement::default(), StrideRequirement::Any);

    let _ = BUCKET_SIZES;
    let _ = FUSION_TABLE;
    let _ = fusion_table();
    let _ = compute_contiguous_strides(&[Dim::Concrete(4), Dim::Concrete(8)]);
    let _ = bucket_s(4);
    let _ = bucket_t_dec(4);
    let _ = bucket_t_pre(0);
    let _ = bucket_step(4, 4, 0);

    let op_sample = Op::Copy(CopyOp::default());
    let _ = legal_layouts(&op_sample);
    let _ = legal_layout_tuples(&op_sample);
}
