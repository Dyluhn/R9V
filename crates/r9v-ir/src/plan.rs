// SPDX-License-Identifier: Apache-2.0
//! Execution plan types (Spec 5 §5.1).
//!
//! The canonical [`Plan`] is produced by the planner from a
//! [`ModelSummary`](r9v_models::ModelSummary), the topology, and the config.
//! Card A2.6 constructs single-device plans in `r9v-loader`
//! (`plan_single_device`) until the full `r9v-part` planner exists; this
//! module defines the shared type both sides build against.

use std::collections::BTreeMap;

/// Plan strategy (Spec 5 §5.1 `strategy`).
///
/// Closed set: exhaustive matching, no wildcard arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanStrategy {
    /// No GPU selected or explicit CPU execution (Spec 5 §5.1).
    Cpu,
    /// One execution device (Spec 5 §5.1).
    Single,
    /// Pipeline parallel across stages (Spec 5 §5.1).
    Pp,
    /// Tensor parallel (Spec 5 §5.1).
    Tp,
    /// Expert parallel for MoE (Spec 5 §5.1).
    Ep,
    /// Pipeline plus tensor parallel, 4+ devices (Spec 5 §5.1).
    PpTp,
}

/// One pipeline stage: the ranks executing a contiguous layer range
/// (Spec 5 §5.1 `stages`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlanStage {
    /// Ranks executing this stage.
    pub rank_set: Vec<u32>,
    /// Inclusive start, exclusive end layer indices.
    pub layer_range: (u32, u32),
}

/// Per-expert execution placement (Spec 5 §5.1 `expert_map` value).
///
/// Closed set: exhaustive matching, no wildcard arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertPlacement {
    /// Expert resident on the rank's device (Spec 5 §5.1).
    Device,
    /// Expert computed on the host CPU (Spec 5 §3.4, §5.1).
    HostCompute,
    /// Expert fetched to device on demand (Spec 5 §3.4, §5.1).
    HostFetch,
}

/// Assignment of one expert to a rank and execution placement
/// (Spec 5 §5.1 `expert_map` entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertAssign {
    /// Model layer index owning the expert.
    pub layer: u32,
    /// Expert index within the layer.
    pub expert: u32,
    /// Rank executing or fetching the expert.
    pub rank: u32,
    /// Where the expert executes.
    pub placement: ExpertPlacement,
}

/// Inter-rank transport (Spec 5 §5.1 `transport` value).
///
/// Closed set: exhaustive matching, no wildcard arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transport {
    /// Direct device-to-device transfer (Spec 5 §5.1).
    Direct,
    /// Transfer staged through host memory (Spec 5 §5.1).
    HostStaged,
}

/// Transport for one directed rank pair (Spec 5 §5.1 `transport` entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkTransport {
    /// Source rank.
    pub from_rank: u32,
    /// Destination rank.
    pub to_rank: u32,
    /// Transport used on this link.
    pub transport: Transport,
}

/// Expected per-bucket step cost (Spec 5 §5.1 `expected` entry).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BucketCost {
    /// Sequence-count bucket `S`.
    pub s: u32,
    /// Decode-token bucket `T_dec`.
    pub t_dec: u32,
    /// Prefill-token bucket `T_pre`.
    pub t_pre: u32,
    /// Expected step time in microseconds.
    pub step_us: f32,
}

/// Expected costs carried by the plan (Spec 5 §5.1 `expected`).
#[derive(Debug, Clone, PartialEq)]
pub struct PlanExpected {
    /// Expected step microseconds per bucket, ascending `(s, t_dec, t_pre)`.
    pub step_us_by_bucket: Vec<BucketCost>,
    /// Expected cold-expert rate from calibration data (Spec 5 §3.4).
    pub cold_expert_rate: f32,
}

/// Execution plan (Spec 5 §5.1).
///
/// Pure data: construction and selection live in the planner. Vectors that
/// affect decisions (`stages`, `expert_map`, `transport`,
/// `step_us_by_bucket`) are in ascending order so plans compare
/// deterministically.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Selected strategy (Spec 5 §5.1).
    pub strategy: PlanStrategy,
    /// Pipeline stages (Spec 5 §5.1).
    pub stages: Vec<PlanStage>,
    /// Tensor-parallel degree (Spec 5 §5.1).
    pub tp_degree: u32,
    /// Expert assignments in ascending `(layer, expert)` order (Spec 5 §5.1).
    pub expert_map: Vec<ExpertAssign>,
    /// Link transports in ascending `(from_rank, to_rank)` order
    /// (Spec 5 §5.1).
    pub transport: Vec<LinkTransport>,
    /// Pipeline microbatch count (Spec 5 §5.1).
    pub pp_microbatches: u32,
    /// Expected costs (Spec 5 §5.1).
    pub expected: PlanExpected,
}

impl Plan {
    /// Ranks participating in this plan, ascending with no duplicates.
    ///
    /// Spec 5 §5.1. Pure function of the stages.
    pub fn rank_set(&self) -> Vec<u32> {
        let mut ranks: BTreeMap<u32, ()> = BTreeMap::new();
        for stage in &self.stages {
            for rank in &stage.rank_set {
                ranks.insert(*rank, ());
            }
        }
        ranks.into_keys().collect()
    }
}
