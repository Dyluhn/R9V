// SPDX-License-Identifier: Apache-2.0
//! Public-surface shape tests for the API-bearing A1.11 deliverables
//! (r9v-card-work §6): visibility, `Send`/`Sync`, closed-set exhaustiveness,
//! and the explicitly deferred B1 caches.

mod common;

use common::{config_128, kv_all, manager_for};
use r9v_state::{CacheDtype, Retain, StateError, StateManager, StateSpec};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn public_types_are_send_sync_and_errors_are_std_errors() {
    assert_send_sync::<StateManager>();
    assert_send_sync::<r9v_state::BatchMeta>();
    assert_send_sync::<r9v_state::CompactOp>();
    assert_send_sync::<r9v_state::Budget>();
    assert_send_sync::<r9v_state::SlotRange>();
    fn is_std_error<T: std::error::Error>() {}
    is_std_error::<StateError>();
}

/// Closed sets match exhaustively with no wildcard arm: adding a variant
/// fails compilation at every site (CONVENTIONS.md §3.2).
#[test]
fn closed_sets_are_exhaustive() {
    fn kind_name(spec: StateSpec) -> &'static str {
        match spec {
            StateSpec::KvPaged { .. } => "kv_paged",
            StateSpec::KvLatent { .. } => "kv_latent",
            StateSpec::Recurrent { .. } => "recurrent",
            StateSpec::ConvWindow { .. } => "conv_window",
        }
    }
    fn cache_name(cache: CacheDtype) -> &'static str {
        match cache {
            CacheDtype::E4M3 => "e4m3",
            CacheDtype::I8 => "i8",
            CacheDtype::F16 => "f16",
        }
    }
    fn retain_name(retain: Retain) -> &'static str {
        match retain {
            Retain::All => "all",
            Retain::Window { .. } => "window",
            Retain::SinkWindow { .. } => "sink_window",
        }
    }
    assert_eq!(kind_name(kv_all()), "kv_paged");
    assert_eq!(cache_name(CacheDtype::E4M3), "e4m3");
    assert_eq!(retain_name(Retain::All), "all");
}

/// Prefix/session reuse is explicitly deferred: no hidden sharing.
#[test]
fn prefix_and_session_caches_are_explicitly_deferred() {
    assert_eq!(StateManager::PREFIX_CACHE_DEFERRED_TO, "B1");
    let mut m = manager_for(config_128(), &[kv_all()]);
    let (_, matched) = m.new_seq(&[1, 2, 3]).unwrap();
    assert_eq!(matched, 0);
    let stats = m.stats();
    assert_eq!(stats.prefix_hit_rate, 0.0);
    assert_eq!(stats.evictions, 0);
}
