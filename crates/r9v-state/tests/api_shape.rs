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

/// Proves group_layers and group_layer_specs return StateResult<Vec<LayerGroup>>,
/// never panic, and never silently drop entries.
#[test]
fn grouping_functions_return_state_result_and_never_panic_or_drop() {
    use r9v_state::{group_layer_specs, group_layers, LayerGroup, StateDecl, StateResult};

    // Assert function signatures return StateResult<Vec<LayerGroup>>
    let fn_group_layers: fn(&[StateSpec]) -> StateResult<Vec<LayerGroup>> = group_layers;
    let fn_group_layer_specs: fn(&[StateDecl]) -> StateResult<Vec<LayerGroup>> = group_layer_specs;

    let dummy_spec = kv_all();
    let res1 = fn_group_layers(&[dummy_spec]);
    assert!(res1.is_ok());

    let res2 = fn_group_layer_specs(&[StateDecl::new(0, dummy_spec)]);
    assert!(res2.is_ok());

    // Out of range: neither panics nor silently drops; returns typed StateError::InvalidConfig
    let out_of_range = StateDecl::new(1024, dummy_spec);
    let err = fn_group_layer_specs(&[out_of_range]).unwrap_err();
    match err {
        StateError::InvalidConfig { problems } => {
            assert_eq!(problems.len(), 1);
            assert_eq!(problems[0].index, 1024);
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}
