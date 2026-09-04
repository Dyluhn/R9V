// SPDX-License-Identifier: Apache-2.0
//! Step 3 — single-device planning (Spec 9 §2 step 3, Spec 5 §5.1–§5.2).
//!
//! Local `plan_single_device()` behind the canonical [`Plan`] type defined
//! in `r9v-ir`, until the full `r9v-part` planner exists (card A2.6). Zero
//! GPUs select `Cpu`, one GPU selects `Single` (Spec 5 §5.2 rule 1); with
//! several visible devices the single-device plan targets the lowest rank.
//! Pure function of its inputs: no device discovery, no environment reads,
//! no clocks, so plans are identical on any machine given the same inputs.

use r9v_ir::{ExpertAssign, ExpertPlacement, Plan, PlanExpected, PlanStage, PlanStrategy};
use r9v_models::ModelSummary;

use crate::error::LoaderError;

/// One visible device for single-device planning (Spec 5 §5.1 inputs).
///
/// Plain numbers supplied by the caller — the loader never discovers
/// devices itself, which keeps planning GPU-independent and
/// machine-agnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlannedDevice {
    /// Plan rank for this device.
    pub rank: u32,
    /// Available VRAM bytes (budgets run against this, not discovery).
    pub vram_bytes: u64,
}

/// Plans a model on zero or one device (Spec 9 §2 step 3, Spec 5 §5.1).
///
/// `devices` is the visible set in any order; the plan targets the lowest
/// rank. Empty selects the `Cpu` strategy, non-empty selects `Single` with
/// `tp_degree = 1` (Spec 5 §5.2 rule 1). Multi-device enumeration (PP/TP/EP
/// candidates, cost-model scoring) belongs to `r9v-part`.
pub fn plan_single_device(
    summary: &ModelSummary,
    devices: &[PlannedDevice],
) -> Result<Plan, LoaderError> {
    let mut ranks: Vec<u32> = devices.iter().map(|d| d.rank).collect();
    ranks.sort_unstable();
    // Duplicate ranks are a caller error, refused deterministically rather
    // than silently winning by input order (Spec 5 §5.1 inputs are a set
    // of visible devices).
    if let Some(dup) = ranks
        .windows(2)
        .find_map(|w| (w[0] == w[1]).then_some(w[0]))
    {
        return Err(LoaderError::Validation {
            problems: vec![format!(
                "duplicate device rank {dup} in single-device plan inputs (Spec 5 §5.1)"
            )],
        });
    }
    let rank = ranks.first().copied().unwrap_or(0);

    let strategy = if devices.is_empty() {
        PlanStrategy::Cpu
    } else {
        PlanStrategy::Single
    };
    let tp_degree = 1u32;

    // Spec 8 §6 item 3: `hkv % tp_degree == 0` (replication permitted is a
    // multi-device planner decision in Spec 5 §3.2, not taken here).
    if !summary.hkv.is_multiple_of(tp_degree) {
        return Err(LoaderError::Validation {
            problems: vec![format!(
                "hkv ({}) is not divisible by tp_degree ({tp_degree}) (Spec 8 §6)",
                summary.hkv,
            )],
        });
    }

    let num_layers = u32::try_from(summary.layers.len()).map_err(|_| LoaderError::Overflow {
        what: "plan layer count".to_string(),
        detail: format!("layers={}", summary.layers.len()),
    })?;

    // The single-device plan keeps every expert on the execution device
    // (Spec 5 §3.4 hot/cold splits are the multi-device planner's
    // decision in `r9v-part`). Expert identity and counts come from the
    // summary's per-layer facts, not weight names.
    // DECISION(A2.6): no `Host`/`Tiered` expert placements here; rejected a
    // placement mapping through `intended_placement` because it yields
    // `Device` by construction and the match could never select another
    // arm. Spec 5 §5.1, §3.4.
    let mut expert_map = Vec::new();
    for (index, layer) in summary.layers.iter().enumerate() {
        let layer_idx = index as u32;
        if let Some(experts) = &layer.experts {
            for expert in 0..experts.e {
                expert_map.push(ExpertAssign {
                    layer: layer_idx,
                    expert,
                    rank,
                    placement: ExpertPlacement::Device,
                });
            }
        }
    }

    Ok(Plan {
        strategy,
        stages: vec![PlanStage {
            rank_set: vec![rank],
            layer_range: (0, num_layers),
        }],
        tp_degree,
        expert_map,
        // DECISION(A2.6): no link transports on a single-device plan;
        // rejected a rank-0 self-link because Spec 5 §5.1 transports describe
        // inter-rank links and a single rank has none. Spec 5 §5.1.
        transport: Vec::new(),
        pp_microbatches: 1,
        // DECISION(A2.6): empty expected-cost table with a zero cold rate;
        // rejected estimated costs because coefficients come from tune files
        // and topology measurements owned by `r9v-part`'s cost model, and a
        // fabricated number would masquerade as measurement. Spec 5 §5.1-§5.2.
        expected: PlanExpected {
            step_us_by_bucket: Vec::new(),
            cold_expert_rate: 0.0,
        },
    })
}
