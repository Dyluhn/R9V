// SPDX-License-Identifier: Apache-2.0
//! Shared fixtures for `r9v-state` integration tests (Spec 3 §8).

// Each integration target compiles this module separately, so allow helpers
// used by only some targets.
#![allow(dead_code)]

use r9v_state::{
    group_layers, required_pool_bytes, CacheDtype, Retain, StateConfig, StateManager, StateSpec,
};

/// Single-layer dense shape used across tests.
pub fn kv_all() -> StateSpec {
    StateSpec::KvPaged {
        hkv: 2,
        d: 16,
        dv: 16,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }
}

/// Windowed dense shape.
pub fn kv_window(w: u32) -> StateSpec {
    StateSpec::KvPaged {
        hkv: 2,
        d: 16,
        dv: 16,
        cache: CacheDtype::E4M3,
        retain: Retain::Window { w },
    }
}

/// Builds a manager with exactly enough pool for full context (Spec 3 §6.3).
pub fn manager_for(config: StateConfig, specs: &[StateSpec]) -> StateManager {
    let groups = group_layers(specs);
    let pool = required_pool_bytes(config, &groups).expect("pool math is exact");
    StateManager::new(config, specs.to_vec(), pool).expect("valid fixture config")
}

/// Standard small config: 128 tokens of context, 8 sequences.
pub fn config_128() -> StateConfig {
    StateConfig {
        max_ctx: 128,
        max_seqs: 8,
    }
}

/// Drives one full step: reserve `tokens.len()`, write them, commit all.
pub fn step_all(m: &mut StateManager, seq: r9v_common::SeqId, tokens: &[u32]) {
    let ctx = m.ctx_len(seq).unwrap();
    m.reserve(seq, tokens.len() as u32).unwrap();
    m.write_tokens(seq, ctx, tokens).unwrap();
    m.commit(seq, tokens.len() as u32).unwrap();
}
