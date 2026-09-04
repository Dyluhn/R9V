// SPDX-License-Identifier: Apache-2.0
//! Step cost table, budget resolution, and prefill admission calculations (Spec 6 §4, §10).

use r9v_ir::{bucket_s, bucket_t_dec, bucket_t_pre, BUCKET_SIZES};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{SchedError, SchedResult};

// DECISION(A3.9): step costs fail closed at every scheduler consumption boundary:
// NaN, infinite, or negative values return SchedError::InvalidCost instead of being
// clamped to zero, which would silently admit wrong prefill budgets. Rejected clamping
// because a zeroed cost hides a broken table behind over-admission. The happy path
// performs no heap allocation. Spec 6 §4.1.
fn validate_cost_ms(context: &'static str, value: f32) -> SchedResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(SchedError::InvalidCost { context, value });
    }
    Ok(value)
}

fn validate_budget_ms(value: f32) -> SchedResult<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(SchedError::InvalidCost {
            context: "step_budget",
            value,
        });
    }
    Ok(value)
}

/// Operational profile determining latency vs throughput optimization (Spec 5 §8, Spec 6 §4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProfileMode {
    /// Latency mode: step budget tuned to 1.25 * C(1, 1, 0) (Spec 6 §4.3).
    #[default]
    Latency,
    /// Throughput mode: step budget tuned to 8 * C(1, 1, 0) (Spec 6 §4.3).
    Throughput,
}

/// Configuration setting for step execution budget (Spec 6 §10).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum StepBudgetConfig {
    /// Explicit budget in milliseconds (Spec 6 §4.3, §10).
    Manual(f32),
    /// Automatically resolved from C(1, 1, 0) at load/warmup (Spec 6 §4.3, §10).
    #[default]
    Auto,
}

/// Abstract cost table providing measured step costs per bucket (Spec 6 §1, §4.1).
pub trait CostTable: Send + Sync {
    /// Evaluates step cost C(S, T_dec, T_pre) in milliseconds for the given shape bucket (Spec 6 §4.1).
    ///
    /// Returns a finite, non-negative cost; rejects NaN, infinite, or negative
    /// table values with [`SchedError::InvalidCost`] instead of clamping.
    fn cost_ms(&self, s: u32, t_dec: u32, t_pre: u32) -> SchedResult<f32>;

    /// Resolves `step_budget_ms` according to the active profile mode (Spec 6 §4.3).
    ///
    /// Rejects non-finite or non-positive budgets with [`SchedError::InvalidCost`].
    fn resolve_budget_ms(
        &self,
        config: StepBudgetConfig,
        profile: ProfileMode,
    ) -> SchedResult<f32> {
        let b = match config {
            StepBudgetConfig::Manual(ms) => ms,
            StepBudgetConfig::Auto => {
                let base = self.cost_ms(1, 1, 0)?;
                match profile {
                    ProfileMode::Latency => 1.25 * base,
                    ProfileMode::Throughput => 8.0 * base,
                }
            }
        };
        validate_budget_ms(b)
    }

    /// Evaluates prefill headroom room R = budget - D - pre_step_estimate (Spec 6 §4.1).
    ///
    /// Rejects non-finite inputs or decode costs with [`SchedError::InvalidCost`];
    /// a finite negative room is a legal "no fit" signal, not an error.
    fn prefill_room_ms(
        &self,
        budget_ms: f32,
        s_dec: u32,
        t_dec: u32,
        pre_step_estimate_ms: f32,
    ) -> SchedResult<f32> {
        if !budget_ms.is_finite() || !pre_step_estimate_ms.is_finite() {
            return Err(SchedError::InvalidCost {
                context: "prefill_room_inputs",
                value: if !budget_ms.is_finite() {
                    budget_ms
                } else {
                    pre_step_estimate_ms
                },
            });
        }
        let d_cost = if s_dec > 0 && t_dec > 0 {
            self.cost_ms(s_dec, t_dec, 0)?
        } else {
            0.0
        };
        Ok(budget_ms - d_cost - pre_step_estimate_ms)
    }

    /// Selects the largest valid prefill bucket b <= remaining such that C(S_dec + 1, T_dec, b) - D <= R (Spec 6 §4.1).
    ///
    /// Propagates [`SchedError::InvalidCost`] for non-finite table values or room.
    fn select_prefill_chunk(
        &self,
        remaining_prompt: u32,
        s_dec: u32,
        t_dec: u32,
        room_ms: f32,
        min_chunk: u32,
        max_chunk: u32,
    ) -> SchedResult<Option<u32>> {
        if !room_ms.is_finite() {
            return Err(SchedError::InvalidCost {
                context: "prefill_room",
                value: room_ms,
            });
        }
        if room_ms <= 0.0 || remaining_prompt == 0 {
            return Ok(None);
        }

        let d_cost = if s_dec > 0 && t_dec > 0 {
            self.cost_ms(s_dec, t_dec, 0)?
        } else {
            0.0
        };

        // If remaining prompt is smaller than min_chunk, evaluate if min_chunk enclosing bucket fits room (Spec 6 §4.1)
        if remaining_prompt < min_chunk {
            let next_s = s_dec.checked_add(1).unwrap_or(s_dec);
            let next_cost = self.cost_ms(next_s, t_dec, min_chunk)?;
            let marginal_cost = next_cost - d_cost;
            if marginal_cost <= room_ms {
                return Ok(Some(remaining_prompt));
            } else {
                return Ok(None);
            }
        }

        // Discrete chunk selection per Spec 6 §4.1:
        // Candidates are discrete bucket sizes b in BUCKET_SIZES such that min_chunk <= b <= max_chunk and b <= remaining_prompt.
        for &b in BUCKET_SIZES.iter().rev() {
            if b >= min_chunk && b <= max_chunk && b <= remaining_prompt {
                let next_s = s_dec.checked_add(1).unwrap_or(s_dec);
                let next_cost = self.cost_ms(next_s, t_dec, b)?;
                let marginal_cost = next_cost - d_cost;
                if marginal_cost <= room_ms {
                    return Ok(Some(b));
                }
            }
        }

        Ok(None)
    }
}

// DECISION(A3.9): pre-step host overhead estimate is budgeted as a deterministic 10% fraction of step_budget_ms; rejected wall-clock measurement in scheduling decisions because Spec 6 §1 Principle 6 mandates reproducible schedules and Spec 6 §3.1 budgets host work at <= 10% of step_budget_ms.
/// Computes host pre-step overhead estimate as 10% of step_budget_ms (Spec 6 §3.1).
#[inline]
pub fn estimate_pre_step_ms(budget_ms: f32) -> f32 {
    if !budget_ms.is_finite() || budget_ms <= 0.0 {
        0.0
    } else {
        budget_ms * 0.10
    }
}

/// Typed in-memory stub cost table for testing and simulation (Spec 6 §1, §4.1).
#[derive(Debug, Clone)]
pub struct CostTableStub {
    exact_table: BTreeMap<(u32, u32, u32), f32>,
    base_decode_ms: f32,
    decode_token_ms: f32,
    prefill_token_ms: f32,
}

impl Default for CostTableStub {
    fn default() -> Self {
        Self::new(8.0, 0.05, 0.005)
    }
}

impl CostTableStub {
    /// Constructs a new cost table stub with linear token cost parameters.
    ///
    /// Parameters are stored exactly as given; invalid values surface as
    /// [`SchedError::InvalidCost`] from [`CostTable::cost_ms`], never clamped.
    pub fn new(base_decode_ms: f32, decode_token_ms: f32, prefill_token_ms: f32) -> Self {
        Self {
            exact_table: BTreeMap::new(),
            base_decode_ms,
            decode_token_ms,
            prefill_token_ms,
        }
    }

    /// Overrides or sets an explicit cost for a specific bucket (S, T_dec, T_pre).
    ///
    /// The value is stored exactly as given; invalid values surface as
    /// [`SchedError::InvalidCost`] from [`CostTable::cost_ms`], never clamped.
    pub fn set_bucket_cost(&mut self, s: u32, t_dec: u32, t_pre: u32, cost_ms: f32) {
        let b_s = bucket_s(s.max(1)).unwrap_or(1);
        let b_dec = bucket_t_dec(t_dec.max(1)).unwrap_or(1);
        let b_pre = bucket_t_pre(t_pre).unwrap_or(0);
        self.exact_table.insert((b_s, b_dec, b_pre), cost_ms);
    }
}

impl CostTable for CostTableStub {
    fn cost_ms(&self, s: u32, t_dec: u32, t_pre: u32) -> SchedResult<f32> {
        let b_s = bucket_s(s.max(1)).unwrap_or(1);
        let b_dec = bucket_t_dec(t_dec.max(1)).unwrap_or(1);
        let b_pre = bucket_t_pre(t_pre).unwrap_or(0);

        let cost = if let Some(&c) = self.exact_table.get(&(b_s, b_dec, b_pre)) {
            c
        } else {
            let mut c = self.base_decode_ms;
            if b_dec > b_s {
                c += (b_dec - b_s) as f32 * self.decode_token_ms;
            }
            if b_pre > 0 {
                c += b_pre as f32 * self.prefill_token_ms;
            }
            c
        };

        validate_cost_ms("cost_table", cost)
    }
}
