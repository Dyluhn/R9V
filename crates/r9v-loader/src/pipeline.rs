// SPDX-License-Identifier: Apache-2.0
//! Steps 1–4 cohesive pipeline: open, bind, plan, budget (Spec 9 §2).
//!
//! [`prepare`] runs metadata-only open (step 1), family resolution,
//! validation, and tensor binding (step 2), single-device planning
//! (step 3), and exact budget checks (step 4). No step reads tensor payload
//! data; infeasible loads refuse with numbers before any I/O.

use std::path::{Path, PathBuf};

use r9v_ir::{Plan, PlanStrategy};
use r9v_models::{ModelGraph, ModelSpec, ModelSummary};
use r9v_state::{group_layer_specs, StateConfig};

use crate::bind::{is_stacked_expert_weight, BindReport};
use crate::budget::{
    check_device_budget, check_host_budget, DeviceBudget, DeviceBudgetInput, HostBudget,
    HostBudgetInput,
};
use crate::error::LoaderError;
use crate::open::{open, open_shard_set_with_file_sizes, open_with_file_size, OpenedCheckpoint};
use crate::open::{GgufFileMeta, ModelFingerprint};
use crate::plan::{plan_single_device, PlannedDevice};
use crate::validate::{model_id_from_meta, resolve_and_validate};

/// Steps 1–4 inputs (Spec 9 §2, §4, §13; Spec 3 §9).
///
/// All device facts are explicit caller inputs: the loader discovers
/// nothing, reads no environment, and touches no HIP runtime, so identical
/// inputs plan identically on any machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareOptions {
    /// Visible devices in any order; empty selects the `Cpu` strategy.
    pub devices: Vec<PlannedDevice>,
    /// Tokens per sequence (Spec 3 §9 `state.max_ctx`, multiple of 32).
    pub max_ctx: u32,
    /// Maximum live sequences (Spec 3 §9 `state.max_seqs`).
    pub max_seqs: u32,
    /// Reserve bytes (Spec 3 §6.3; [`DEFAULT_RESERVE_BYTES`] when unset —
    /// use that constant explicitly, there is no implicit default here).
    pub reserve_bytes: u64,
    /// Activation workspace bytes (Spec 6 §5.3, measured at capture).
    pub workspace_bytes: u64,
    /// Available pinned host bytes (Spec 9 §4.2; explicit, never discovered).
    pub host_pinned_bytes: u64,
    /// I/O staging chunk bytes (Spec 9 §5.1).
    pub chunk_bytes: u64,
    /// I/O queue depth (Spec 9 §5.1).
    pub queue_depth: u64,
    /// Tiered slab bytes (Spec 9 §6).
    pub slab_bytes: u64,
    /// Per-step buffer bytes (Spec 9 §4.2).
    pub per_step_bytes: u64,
}

/// Steps 1–4 output: everything materialization needs, decided before any
/// payload I/O (Spec 9 §2).
#[derive(Debug)]
pub struct PreparedLoad {
    /// First shard path.
    pub path: String,
    /// Every shard path in shard order.
    pub shard_paths: Vec<String>,
    /// Spec 9 §3 merged file fingerprint over the shard set.
    pub file_fp: u128,
    /// Spec 9 §3 model fingerprint lifecycle state.
    pub model_fp: ModelFingerprint,
    /// Disk bytes read while opening (metadata prefixes only).
    pub bytes_read: u64,
    /// Validated model specification (Spec 8 §4–§6).
    pub spec: ModelSpec,
    /// Lowered model graph (Spec 8 §2).
    pub graph: ModelGraph,
    /// Planner summary (Spec 8 §7).
    pub summary: ModelSummary,
    /// Tensor bindings with unused-tensor warnings (Spec 8 §6).
    pub bind: BindReport,
    /// Single-device plan (Spec 5 §5.1).
    pub plan: Plan,
    /// Device budget (`None` under the `Cpu` strategy).
    pub device_budget: Option<DeviceBudget>,
    /// Host budget (Spec 9 §4.2).
    pub host_budget: HostBudget,
}

/// Runs steps 1–4 against the on-disk checkpoint (Spec 9 §2).
///
/// A single path that declares a split set discovers and checks every
/// sibling; use [`prepare_shard_set`] for an explicit path set.
pub fn prepare(path: &Path, options: &PrepareOptions) -> Result<PreparedLoad, LoaderError> {
    let checkpoint = open(path)?;
    prepare_from_checkpoint(checkpoint, options)
}

/// Runs steps 1–4 against a metadata prefix of a logically `file_size`-byte
/// file (Spec 9 §2).
///
/// See [`open_with_file_size`]: tests truncate the payload and pass the
/// pre-truncation size to prove steps 1–4 never touch weight data.
pub fn prepare_with_file_size(
    path: &Path,
    file_size: u64,
    options: &PrepareOptions,
) -> Result<PreparedLoad, LoaderError> {
    let checkpoint = open_with_file_size(path, file_size)?;
    prepare_from_checkpoint(checkpoint, options)
}

/// Runs steps 1–4 against an explicit shard path set in any order
/// (Spec 9 §2 step 1).
///
/// `sizes[i]` is the logical size of `paths[i]` (`None` selects the
/// on-disk size); tests truncate shard payloads and pass pre-truncation
/// sizes to prove steps 1–4 never touch weight data.
pub fn prepare_shard_set(
    paths: &[PathBuf],
    sizes: &[Option<u64>],
    options: &PrepareOptions,
) -> Result<PreparedLoad, LoaderError> {
    let checkpoint = open_shard_set_with_file_sizes(paths, sizes)?;
    prepare_from_checkpoint(checkpoint, options)
}

/// Strategy selection shared by binding and planning (Spec 5 §5.2 rule 1).
pub fn strategy_for(devices: &[PlannedDevice]) -> PlanStrategy {
    if devices.is_empty() {
        PlanStrategy::Cpu
    } else {
        PlanStrategy::Single
    }
}

/// Steps 2–4 over an opened checkpoint.
fn prepare_from_checkpoint(
    checkpoint: OpenedCheckpoint,
    options: &PrepareOptions,
) -> Result<PreparedLoad, LoaderError> {
    StateConfig {
        max_ctx: options.max_ctx,
        max_seqs: options.max_seqs,
    }
    .validate()?;

    let file = checkpoint.file();
    let meta = GgufFileMeta::new(file);
    let model_id = model_id_from_meta(&meta)?;
    let strategy = strategy_for(&options.devices);
    let validated = resolve_and_validate(&meta, &checkpoint, &model_id, strategy)?;
    let bound = validated.bind;

    let plan = plan_single_device(&validated.summary, &options.devices)?;

    let groups = group_layer_specs(&validated.graph.state_declarations())?;

    // Single-device plans keep every weight `Device(0)` (see
    // `intended_placement`); under `Cpu` the execution device is the host,
    // so weights and pools charge the host budget.
    //
    // Destination storage budgets once: tied aliases occupy no additional
    // bytes (Spec 8 §5, Spec 2 §4), so zero-byte bindings never enter a
    // budget input.
    let resident: Vec<(String, u64)> = bound
        .bound
        .iter()
        .filter(|b| b.bytes > 0)
        .map(|b| (b.name.clone(), b.bytes))
        .collect();
    let expert_bytes: u64 = bound
        .bound
        .iter()
        .filter(|b| is_stacked_expert_weight(&validated.graph, &b.name))
        .map(|b| b.bytes)
        .try_fold(0u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| LoaderError::Overflow {
                    what: "expert bytes total".to_string(),
                    detail: format!("{total} + {bytes}"),
                })
        })?;

    let device_budget = match strategy {
        PlanStrategy::Cpu => None,
        _ => {
            // Internal invariant: a non-CPU plan targets exactly one rank,
            // so the plan carries the budgeted device.
            let rank = plan
                .stages
                .first()
                .and_then(|stage| stage.rank_set.first())
                .copied()
                .ok_or_else(|| LoaderError::Validation {
                    problems: vec!["non-CPU plan has no stage rank (Spec 5 §5.1)".to_string()],
                })?;
            let device = options
                .devices
                .iter()
                .find(|d| d.rank == rank)
                .ok_or_else(|| LoaderError::Validation {
                    problems: vec![format!(
                        "plan targets rank {rank} outside the visible device set (Spec 5 §5.1)"
                    )],
                })?;
            Some(check_device_budget(&DeviceBudgetInput {
                rank: device.rank,
                tensors: &resident,
                expert_bytes,
                groups: &groups,
                max_ctx: options.max_ctx,
                max_seqs: options.max_seqs,
                workspace_bytes: options.workspace_bytes,
                comms_bytes: 0,
                reserve_bytes: options.reserve_bytes,
                available_bytes: device.vram_bytes,
            })?)
        }
    };

    let host_tensors: Vec<(String, u64)> = match strategy {
        PlanStrategy::Cpu => resident.clone(),
        _ => Vec::new(),
    };
    let staging_bytes = options
        .chunk_bytes
        .checked_mul(options.queue_depth)
        .ok_or_else(|| LoaderError::Overflow {
            what: "staging ring".to_string(),
            detail: format!("{} * {}", options.chunk_bytes, options.queue_depth),
        })?;
    let host_budget = check_host_budget(&HostBudgetInput {
        tensors: &host_tensors,
        charge_pools_to_host: strategy == PlanStrategy::Cpu,
        groups: &groups,
        max_ctx: options.max_ctx,
        max_seqs: options.max_seqs,
        staging_bytes,
        slab_bytes: options.slab_bytes,
        per_step_bytes: options.per_step_bytes,
        available_bytes: options.host_pinned_bytes,
    })?;

    Ok(PreparedLoad {
        path: checkpoint.path().to_string(),
        shard_paths: checkpoint.shard_paths(),
        file_fp: checkpoint.file_fp(),
        model_fp: checkpoint.model_fp(),
        bytes_read: checkpoint.bytes_read(),
        spec: validated.spec,
        graph: validated.graph,
        summary: validated.summary,
        bind: bound,
        plan,
        device_budget,
        host_budget,
    })
}
