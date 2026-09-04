// SPDX-License-Identifier: Apache-2.0
//! Public API shape for the loader pipeline (card A2.6; `CONVENTIONS.md`
//! §4.1). Asserts visibility, `Send`/`Sync`, and the canonical `Plan`
//! plumbing without exercising behavior.

use r9v_loader::{
    BindReport, BoundTensor, BudgetScope, DeviceBudget, DeviceBudgetInput, GgufFileMeta,
    HostBudget, HostBudgetInput, LoaderError, ModelFingerprint, OpenedCheckpoint, Plan,
    PlannedDevice, PrepareOptions, PreparedLoad, TensorProblem, TensorProblemKind, ValidatedModel,
};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn public_types_are_send_sync() {
    assert_send_sync::<LoaderError>();
    assert_send_sync::<TensorProblem>();
    assert_send_sync::<TensorProblemKind>();
    assert_send_sync::<BudgetScope>();
    assert_send_sync::<ModelFingerprint>();
    assert_send_sync::<OpenedCheckpoint>();
    assert_send_sync::<GgufFileMeta<'static>>();
    assert_send_sync::<ValidatedModel>();
    assert_send_sync::<BoundTensor>();
    assert_send_sync::<BindReport>();
    assert_send_sync::<PlannedDevice>();
    assert_send_sync::<PrepareOptions>();
    assert_send_sync::<PreparedLoad>();
    assert_send_sync::<DeviceBudget>();
    assert_send_sync::<DeviceBudgetInput<'_>>();
    assert_send_sync::<HostBudget>();
    assert_send_sync::<HostBudgetInput<'_>>();
    // The single-device plan is the canonical `r9v-ir` type, not a local copy.
    assert_send_sync::<Plan>();
    const IS_CANONICAL: fn(&Plan) = |_: &r9v_ir::Plan| {};
    let _ = IS_CANONICAL;
}

#[test]
fn step_1_to_4_surface_is_present() {
    // Split-shard open, shard-set prepare, MTP downgrade, fusion checks,
    // carried expert facts, and arena layout: all referenced so signature
    // drift fails here first.
    let _ = r9v_loader::open as fn(&std::path::Path) -> Result<OpenedCheckpoint, LoaderError>;
    let _ = r9v_loader::open_shard_set
        as fn(&[std::path::PathBuf]) -> Result<OpenedCheckpoint, LoaderError>;
    let _ = r9v_loader::prepare_shard_set
        as fn(
            &[std::path::PathBuf],
            &[Option<u64>],
            &PrepareOptions,
        ) -> Result<PreparedLoad, LoaderError>;
    let _ = r9v_loader::downgrade_absent_mtp
        as fn(
            r9v_models::ModelSpec,
            &OpenedCheckpoint,
            &str,
        ) -> Result<(r9v_models::ModelSpec, r9v_models::ModelGraph), LoaderError>;
    let _ = r9v_loader::check_fusion_decls
        as fn(&r9v_models::ModelGraph, &OpenedCheckpoint) -> Vec<String>;
    let _ = r9v_loader::is_stacked_expert_weight as fn(&r9v_models::ModelGraph, &str) -> bool;
    let _ = r9v_loader::arena_layout
        as fn(&[(String, u64)]) -> Result<(Vec<(String, u64)>, u64), LoaderError>;
    let _ = ModelFingerprint::PendingUntilRepack.as_u128();
}
