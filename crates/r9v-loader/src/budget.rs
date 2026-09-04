// SPDX-License-Identifier: Apache-2.0
//! Step 4 — exact budgets and refusals (Spec 9 §2 step 4, §4, §4.3;
//! Spec 3 §6.3).
//!
//! Per-device arena (Spec 9 §4.1: weights in graph-consumption order at
//! 256-byte-aligned offsets, state pools, workspace, comms, reserve) and
//! host pinned memory (Spec 9 §4.2: staging ring, host tensors, slab,
//! per-step buffers). All arithmetic is checked: overflow fails closed,
//! never wraps. A shortfall refuses with required, available, shortfall,
//! the top contributors, and the smallest single config change that would
//! fit — settings are never silently lowered.

use r9v_state::{required_pool_bytes, LayerGroup, StateConfig, BLOCK_TOKENS};

use crate::error::{BudgetScope, LoaderError};

/// Tensor placement alignment in arenas (Spec 9 §4.1).
pub const TENSOR_ALIGN_BYTES: u64 = 256;

/// Default reserve (Spec 9 §4.1, Spec 3 §6.3 `state.reserve_bytes`).
pub const DEFAULT_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// Default I/O staging chunk (Spec 9 §5.1, §13 `[io] chunk_mb = 16`).
pub const DEFAULT_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// Default I/O queue depth (Spec 9 §5.1, §13 `[io] queue_depth = 8`).
pub const DEFAULT_QUEUE_DEPTH: u64 = 8;

/// Checked addition for budget totals.
fn checked_add(total: u64, add: u64, what: &str) -> Result<u64, LoaderError> {
    total.checked_add(add).ok_or_else(|| LoaderError::Overflow {
        what: what.to_string(),
        detail: format!("{total} + {add}"),
    })
}

/// Checked round-up to `TENSOR_ALIGN_BYTES` (Spec 9 §4.1).
pub fn align_up_256(bytes: u64) -> Result<u64, LoaderError> {
    bytes
        .checked_add(TENSOR_ALIGN_BYTES - 1)
        .ok_or_else(|| LoaderError::Overflow {
            what: "arena alignment".to_string(),
            detail: format!("{bytes} + {}", TENSOR_ALIGN_BYTES - 1),
        })
        .map(|padded| padded & !(TENSOR_ALIGN_BYTES - 1))
}

/// Device arena layout: `(name, 256-byte-aligned start offset)` per
/// tensor in order, plus the exact arena size (Spec 9 §4.1).
///
/// Offsets advance as `offset = align_up_256(offset); offset += size`:
/// each tensor *starts* aligned; sizes are never blindly rounded. The
/// total is the final offset with no trailing pad.
pub fn arena_layout(tensors: &[(String, u64)]) -> Result<(Vec<(String, u64)>, u64), LoaderError> {
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut offset = 0u64;
    for (name, bytes) in tensors {
        offset = align_up_256(offset)?;
        offsets.push((name.clone(), offset));
        offset = checked_add(offset, *bytes, "arena weights total")?;
    }
    Ok((offsets, offset))
}

/// Device budget input (Spec 9 §4.1).
#[derive(Debug)]
pub struct DeviceBudgetInput<'a> {
    /// Plan rank owning the arena.
    pub rank: u32,
    /// `(name, exact bytes)` per resident tensor, binding order.
    /// Zero-byte tied aliases must already be excluded: the arena holds
    /// destination storage once (Spec 8 §5, Spec 2 §4).
    pub tensors: &'a [(String, u64)],
    /// Exact stacked-expert bytes within `tensors` (a subset of the
    /// pre-alignment total), isolating the `experts.hot_set_vram` knob
    /// (Spec 9 §4.3). Zero for dense models.
    pub expert_bytes: u64,
    /// State layer groups for pool sizing (Spec 3 §6.3).
    pub groups: &'a [LayerGroup],
    /// Tokens per sequence.
    pub max_ctx: u32,
    /// Maximum live sequences.
    pub max_seqs: u32,
    /// Activation workspace bytes (Spec 6 §5.3, measured at capture).
    pub workspace_bytes: u64,
    /// Comms buffer bytes (Spec 5 §6.1; zero without collectives).
    pub comms_bytes: u64,
    /// Reserve bytes (Spec 3 §6.3; default [`DEFAULT_RESERVE_BYTES`]).
    pub reserve_bytes: u64,
    /// Available VRAM bytes.
    pub available_bytes: u64,
}

/// Accepted device budget with exact numbers (Spec 9 §4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBudget {
    /// Plan rank owning the arena.
    pub rank: u32,
    /// Weights at 256-byte-aligned offsets.
    pub weights_bytes: u64,
    /// State pools across groups (Spec 3 §6.3).
    pub state_pool_bytes: u64,
    /// Workspace bytes.
    pub workspace_bytes: u64,
    /// Comms bytes.
    pub comms_bytes: u64,
    /// Reserve bytes.
    pub reserve_bytes: u64,
    /// Total required bytes.
    pub required_bytes: u64,
    /// Available bytes.
    pub available_bytes: u64,
}

/// Host budget input (Spec 9 §4.2).
#[derive(Debug)]
pub struct HostBudgetInput<'a> {
    /// `(name, exact bytes)` per host-resident tensor (CPU plan weights).
    pub tensors: &'a [(String, u64)],
    /// Whether state pools live in host memory (CPU plan).
    pub charge_pools_to_host: bool,
    /// State layer groups for pool sizing (Spec 3 §6.3).
    pub groups: &'a [LayerGroup],
    /// Tokens per sequence.
    pub max_ctx: u32,
    /// Maximum live sequences.
    pub max_seqs: u32,
    /// Staging ring bytes (`chunk × depth`, Spec 9 §4.2).
    pub staging_bytes: u64,
    /// Tiered slab bytes (Spec 9 §6).
    pub slab_bytes: u64,
    /// Per-step buffer bytes (Spec 9 §4.2).
    pub per_step_bytes: u64,
    /// Available pinned bytes.
    pub available_bytes: u64,
}

/// Accepted host budget with exact numbers (Spec 9 §4.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBudget {
    /// Host-resident tensor bytes.
    pub tensor_bytes: u64,
    /// Host-side state pools (CPU plan; else 0).
    pub state_pool_bytes: u64,
    /// Staging ring bytes.
    pub staging_bytes: u64,
    /// Tiered slab bytes.
    pub slab_bytes: u64,
    /// Per-step buffer bytes.
    pub per_step_bytes: u64,
    /// Total required bytes.
    pub required_bytes: u64,
    /// Available bytes.
    pub available_bytes: u64,
}

/// Checks the device budget, refusing with numbers on shortfall
/// (Spec 9 §2 step 4, §4.1, §4.3).
pub fn check_device_budget(input: &DeviceBudgetInput<'_>) -> Result<DeviceBudget, LoaderError> {
    let exact_tensors = exact_total(input.tensors)?;
    if input.expert_bytes > exact_tensors {
        return Err(LoaderError::Validation {
            problems: vec![format!(
                "expert bytes {} exceed exact tensor total {exact_tensors} (Spec 9 §4.3)",
                input.expert_bytes,
            )],
        });
    }
    let weights_bytes = arena_layout(input.tensors)?.1;
    let pools = pool_bytes(input.groups, input.max_ctx, input.max_seqs)?;
    let mut required = checked_add(weights_bytes, pools, "device weights + pools")?;
    required = checked_add(required, input.workspace_bytes, "device + workspace")?;
    required = checked_add(required, input.comms_bytes, "device + comms")?;
    required = checked_add(required, input.reserve_bytes, "device + reserve")?;

    if required <= input.available_bytes {
        return Ok(DeviceBudget {
            rank: input.rank,
            weights_bytes,
            state_pool_bytes: pools,
            workspace_bytes: input.workspace_bytes,
            comms_bytes: input.comms_bytes,
            reserve_bytes: input.reserve_bytes,
            required_bytes: required,
            available_bytes: input.available_bytes,
        });
    }

    let mut contributors = input.tensors.to_vec();
    push_contributor(&mut contributors, "state pools", pools);
    push_contributor(&mut contributors, "workspace", input.workspace_bytes);
    push_contributor(&mut contributors, "comms", input.comms_bytes);
    push_contributor(&mut contributors, "reserve", input.reserve_bytes);
    let largest = top_contributors(contributors);
    let suggestion = suggest_change(&SuggestInput {
        scope: BudgetScope::Device { rank: input.rank },
        groups: input.groups,
        max_ctx: input.max_ctx,
        max_seqs: input.max_seqs,
        weights_bytes,
        expert_bytes: input.expert_bytes,
        fixed_no_pool: required.saturating_sub(pools),
        available: input.available_bytes,
    })?;
    Err(LoaderError::Budget {
        scope: BudgetScope::Device { rank: input.rank },
        required,
        available: input.available_bytes,
        shortfall: required - input.available_bytes,
        largest,
        suggestion,
    })
}

/// Checks the host budget, refusing with numbers on shortfall
/// (Spec 9 §2 step 4, §4.2–§4.3).
pub fn check_host_budget(input: &HostBudgetInput<'_>) -> Result<HostBudget, LoaderError> {
    let tensor_bytes = exact_total(input.tensors)?;
    let pools = if input.charge_pools_to_host {
        pool_bytes(input.groups, input.max_ctx, input.max_seqs)?
    } else {
        0
    };
    let mut required = checked_add(tensor_bytes, pools, "host tensors + pools")?;
    required = checked_add(required, input.staging_bytes, "host + staging")?;
    required = checked_add(required, input.slab_bytes, "host + slab")?;
    required = checked_add(required, input.per_step_bytes, "host + per-step")?;

    if required <= input.available_bytes {
        return Ok(HostBudget {
            tensor_bytes,
            state_pool_bytes: pools,
            staging_bytes: input.staging_bytes,
            slab_bytes: input.slab_bytes,
            per_step_bytes: input.per_step_bytes,
            required_bytes: required,
            available_bytes: input.available_bytes,
        });
    }

    let mut contributors = input.tensors.to_vec();
    push_contributor(&mut contributors, "state pools", pools);
    push_contributor(&mut contributors, "staging ring", input.staging_bytes);
    push_contributor(&mut contributors, "tiered slab", input.slab_bytes);
    push_contributor(&mut contributors, "per-step buffers", input.per_step_bytes);
    let largest = top_contributors(contributors);
    let suggestion = suggest_change(&SuggestInput {
        scope: BudgetScope::Host,
        groups: input.groups,
        max_ctx: input.max_ctx,
        max_seqs: input.max_seqs,
        weights_bytes: tensor_bytes,
        expert_bytes: 0,
        fixed_no_pool: required.saturating_sub(pools),
        available: input.available_bytes,
    })?;
    Err(LoaderError::Budget {
        scope: BudgetScope::Host,
        required,
        available: input.available_bytes,
        shortfall: required - input.available_bytes,
        largest,
        suggestion,
    })
}

/// State pool bytes for a config (Spec 3 §6.3).
fn pool_bytes(groups: &[LayerGroup], max_ctx: u32, max_seqs: u32) -> Result<u64, LoaderError> {
    required_pool_bytes(StateConfig { max_ctx, max_seqs }, groups).map_err(LoaderError::State)
}

/// Exact sum of tensor bytes.
fn exact_total(tensors: &[(String, u64)]) -> Result<u64, LoaderError> {
    let mut total = 0u64;
    for (_, bytes) in tensors {
        total = checked_add(total, *bytes, "tensor bytes total")?;
    }
    Ok(total)
}

/// Adds a pseudo-contributor when nonzero.
fn push_contributor(contributors: &mut Vec<(String, u64)>, name: &str, bytes: u64) {
    if bytes > 0 {
        contributors.push((name.to_string(), bytes));
    }
}

/// Top five contributors by bytes descending, name ascending on ties.
fn top_contributors(mut contributors: Vec<(String, u64)>) -> Vec<(String, u64)> {
    contributors.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    contributors.truncate(5);
    contributors
}

/// Smallest single config change that would fit, with resulting numbers
/// (Spec 9 §4.3).
///
/// Knob order follows the spec text: `state.max_ctx`, then
/// `state.max_seqs`, then `experts.hot_set_vram` (device scope with
/// stacked experts resident), then a smaller quant. Each suggestion quotes
/// the resulting requirement in bytes against the available bytes, and
/// settings are never silently lowered.
// DECISION(A2.6): knob preference order max_ctx, max_seqs, hot_set_vram
// with scope-specific fallbacks; rejected enumerating multi-knob
// combinations because Spec 9 §4.3 asks for the smallest single change.
// Spec 9 §4.3.
/// Suggestion inputs (Spec 9 §4.3): one struct so the knob search stays
/// a three-argument helper instead of growing per knob.
struct SuggestInput<'a> {
    scope: BudgetScope,
    groups: &'a [LayerGroup],
    max_ctx: u32,
    max_seqs: u32,
    weights_bytes: u64,
    expert_bytes: u64,
    fixed_no_pool: u64,
    available: u64,
}

fn suggest_change(input: &SuggestInput<'_>) -> Result<String, LoaderError> {
    let fits = |ctx: u32, seqs: u32| -> Result<Option<u64>, LoaderError> {
        let pool = pool_bytes(input.groups, ctx, seqs)?;
        let req = checked_add(input.fixed_no_pool, pool, "suggestion total")?;
        Ok((req <= input.available).then_some(req))
    };

    // Both searches are binary: the requirement grows monotonically with
    // either knob, so the fitting sets are contiguous ranges.
    let block = BLOCK_TOKENS;
    let top_ctx = (input.max_ctx / block) * block;
    if top_ctx >= block {
        let mut lo: u64 = 1;
        let mut hi: u64 = u64::from(top_ctx / block);
        let mut best: Option<(u32, u64)> = None;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let ctx = (mid * u64::from(block)) as u32;
            match fits(ctx, input.max_seqs)? {
                Some(req) => {
                    best = Some((ctx, req));
                    lo = mid + 1;
                }
                None => {
                    if mid == 0 {
                        break;
                    }
                    hi = mid - 1;
                }
            }
        }
        if let Some((ctx, req)) = best {
            return Ok(format!(
                "state.max_ctx = {ctx} (requires {req} B of {} B available)",
                input.available,
            ));
        }
    }

    if input.max_seqs >= 2 {
        let mut lo: u64 = 1;
        let mut hi: u64 = u64::from(input.max_seqs) - 1;
        let mut best: Option<(u32, u64)> = None;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let seqs = mid as u32;
            match fits(input.max_ctx, seqs)? {
                Some(req) => {
                    best = Some((seqs, req));
                    lo = mid + 1;
                }
                None => {
                    if mid == 0 {
                        break;
                    }
                    hi = mid - 1;
                }
            }
        }
        if let Some((seqs, req)) = best {
            return Ok(format!(
                "state.max_seqs = {seqs} (requires {req} B of {} B available)",
                input.available,
            ));
        }
    }

    // `experts.hot_set_vram` caps device-resident expert bytes under
    // expert-offload planning (Spec 5 §3.4); the remainder stays host-side.
    // Only the device scope with resident experts can use it.
    if matches!(input.scope, BudgetScope::Device { .. }) && input.expert_bytes > 0 {
        // Resident weights with a hot set of `hot`: every non-expert byte
        // plus at most `hot` expert bytes. `weights_bytes` already counts
        // all experts once, so subtract then cap.
        let non_expert = input.weights_bytes.saturating_sub(input.expert_bytes);
        let resident = |hot: u64| -> Result<Option<u64>, LoaderError> {
            let capped = input.expert_bytes.min(hot);
            let weights = checked_add(non_expert, capped, "hot-set weights")?;
            let base = checked_add(
                weights,
                input.fixed_no_pool.saturating_sub(input.weights_bytes),
                "hot-set fixed",
            )?;
            let req = checked_add(
                base,
                pool_bytes(input.groups, input.max_ctx, input.max_seqs)?,
                "hot-set total",
            )?;
            Ok((req <= input.available).then_some(req))
        };
        if resident(0)?.is_some() {
            let mut lo = 0u64;
            let mut hi = input.expert_bytes;
            let mut best: Option<(u64, u64)> = None;
            while lo <= hi {
                let mid = lo + (hi - lo) / 2;
                match resident(mid)? {
                    Some(req) => {
                        best = Some((mid, req));
                        lo = mid + 1;
                    }
                    None => {
                        if mid == 0 {
                            break;
                        }
                        hi = mid - 1;
                    }
                }
            }
            if let Some((hot, req)) = best {
                return Ok(format!(
                    "experts.hot_set_vram = {hot} (requires {req} B of {} B available)",
                    input.available,
                ));
            }
        }
    }

    if input.weights_bytes > input.available {
        return Ok(format!(
            "weights alone require {} B of {} B available on {}; use a smaller quant",
            input.weights_bytes, input.available, input.scope,
        ));
    }
    match input.scope {
        BudgetScope::Host => {
            let need = checked_add(
                input.fixed_no_pool,
                pool_bytes(input.groups, input.max_ctx, input.max_seqs)?,
                "host need",
            )?;
            Ok(format!(
                "host.pinned_budget = {need} (requires {need} B of {} B available)",
                input.available,
            ))
        }
        BudgetScope::Device { .. } => {
            let min = checked_add(
                input.fixed_no_pool,
                pool_bytes(input.groups, BLOCK_TOKENS, 1)?,
                "minimum need",
            )?;
            Ok(format!(
                "even state.max_ctx = {} with state.max_seqs = 1 requires {min} B of {} B available on {}; reduce fixed overhead or use a smaller quant",
                BLOCK_TOKENS,
                input.available,
                input.scope,
            ))
        }
    }
}
