// SPDX-License-Identifier: Apache-2.0
//! R9V Op IR core types (Spec 1 §2, App. A; card A1.1).
//!
//! This crate owns the API-bearing type surface the rest of the engine builds
//! against: dtypes, quant-scheme tags, tensors, batch metadata, state
//! handles, the arch descriptor and the IR version. Ops, graphs, sharding
//! tables and the numerics contract are owned by card A1.2 (Spec 1 §3–§6).
//!
//! Repository standards: `CONVENTIONS.md`; engineering bar:
//! `.agents/skills/r9v-engineering-standards`.

pub mod arch;
pub mod batch;
pub mod dtype;
pub mod error;
pub mod fusion;
pub mod graph;
pub mod layout;
pub mod numerics;
pub mod op;
pub mod quant;
pub mod sharding;
pub mod state;
pub mod tensor;
pub mod version;

pub use arch::{
    ArchDescriptor, ArchFamily, DeviceDescriptor, DeviceFacts, DeviceIdentity, GraphCapture,
    MatrixOp, Measured, P2pLink, P2pTransport, RelRate, ValuDot,
};
pub use batch::{BatchMeta, BatchMetaBuilder, Positions, TreeMask, BLOCK_TABLE_SENTINEL};
pub use dtype::DType;
pub use error::IrError;
pub use fusion::{
    fusion_table, is_permitted_fusion, is_permitted_pair, match_chain, match_gated_pair,
    FusionEntry, FusionPattern, FUSION_TABLE,
};
pub use graph::{
    bucket_s, bucket_step, bucket_t_dec, bucket_t_pre, compute_contiguous_strides, EdgeId,
    ExternalInput, ExternalInputKind, ExternalOutput, ExternalOutputKind, Graph, GraphEdge,
    GraphNode, GraphSummary, InsertedCopy, NodeId, PlanId, PositionsKind, StepGraphKey,
    StrideRequirement, BUCKET_SIZES,
};
pub use layout::LayoutId;
pub use numerics::{matmul_numerics, moe_ffn_gemm_numerics, Numerics, ReductionOrder};
pub use op::{
    ActMulOp, ActivationKind, ActivationOp, AllGatherOp, AllReduceOp, AllToAllOp, AttentionMask,
    AttentionOp, BarrierOp, CacheScaleGranularity, CastOp, CausalConv1dOp, ConcatOp,
    ConvActivation, CopyKind, CopyOp, EmbedGatherOp, Epilogue, GatherRowsOp, GroupId, HashId,
    LinearAttnKind, LinearAttnScanOp, LogitSoftcapOp, LogitsPostprocessOp, MatmulOp,
    MlaAttentionSpec, MlaLatent, MoeFfnOp, MoeGroup, MoeRouteOp, MoeScoring, NgramCombine,
    NgramGatherOp, NgramSource, NormAxis, NormKind, NormOp, Op, QuantActOp, RecvOp, ReduceOp,
    ReduceScatterOp, ResidualAddOp, RngAlgorithm, RopeOp, RopeScaling, RopeStyle, SampleOp,
    SamplingParams, ScatterAddRowsOp, SendOp, Smoothing, SplitOp, StateWriteKvOp, VerifyMethod,
    VerifyOp,
};
pub use quant::{QuantScheme, SchemeId};
pub use sharding::{
    legal_layout_tuples, legal_layouts, ExpertCount, HeadCount, ShardLayoutPattern, ShardingRule,
};
pub use state::{StateHandle, StateKind};
pub use tensor::{Class, Dim, Placement, ShapeSymbol, ShardLayout, Tensor};
pub use version::IrVersion;
