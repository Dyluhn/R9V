// SPDX-License-Identifier: Apache-2.0
//! R9V kernel registry, bundle loader, tune files, resolution, and launch dispatch (Spec 4, Spec 14 §2).
//!
//! This crate implements Card A3.1:
//! - Variant keys and stable variant hashing with deterministic serialization (Spec 4 §3).
//! - Release bundle manifest reader, validation, and fingerprint calculation (Spec 4 §11).
//! - Autotune file reader, writer, and versioned merging (Spec 4 §6.2).
//! - Graph-capture variant resolution with strict tier ordering and validation gating (Spec 4 §9.2, §9.3).
//! - Lazy code object loading via `hipModuleLoadData` on demand (Spec 4 §11).
//! - Kernel launch list recording and deterministic sequential replay against stub and hardware devices (Spec 4 §7).
//! - Zero-overhead disabled-cost profiling sink hook (Spec 4 §12).
//! - Gated online JIT autotuning via `allow_jit` (Spec 4 §9.2, §14).

pub mod error;
pub mod launch;
pub mod manifest;
pub mod resolution;
pub mod serde_helpers;
pub mod tune;
pub mod types;
pub mod variant;

pub use error::{RegistryError, Result};
pub use launch::{
    dispatch_launch, DeviceExecutor, LaunchEntry, LaunchList, LaunchRecord, ProfileSink, StubDevice,
};
pub use manifest::{BundleManifest, ManifestVariantEntry};
pub use r9v_ir::P2pTransport;
pub use resolution::{JitProvider, Registry, RegistryConfig, ResolvedVariant};
pub use tune::{TuneEntry, TuneFile, TuneMeasuredOn};
pub use types::{
    ActMulStatic, ActivationStatic, AllGatherStatic, AllReduceStatic, AllToAllStatic, ArchName,
    ArtifactOrigin, AttentionFacts, AttentionStatic, BarrierStatic, CastStatic, CausalConv1dFacts,
    CausalConv1dStatic, CollectiveFacts, CollectivesStatic, ConcatStatic, CopyStatic,
    ElementwiseFacts, ElementwiseParams, ElementwiseStatic, EmbedGatherStatic, GatherRowsStatic,
    LaunchGeometry, LinearAttnScanFacts, LinearAttnScanStatic, LogitSoftcapStatic,
    LogitsPostprocessStatic, MatmulFacts, MatmulStatic, MlaAttentionStatic, MlaLatentStatic,
    MoeFfnFacts, MoeFfnStatic, MoeRouteFacts, MoeRouteStatic, NgramGatherStatic, NormStatic, OpId,
    OpStatic, PlacementKind, QuantActStatic, RecvStatic, ReduceScatterStatic, ResidualAddStatic,
    RopeScalingStatic, RopeStatic, SampleStatic, SamplingFacts, SamplingStatic, ScanMode,
    ScatterAddRowsStatic, SendStatic, SplitStatic, StateWriteKvFacts, StateWriteKvStatic,
    StaticFacts, Tier, TileConfig, VariantHash, VerifyMethodStatic, VerifyStatic,
};
pub use variant::{static_hash, variant_hash, VariantKey};
