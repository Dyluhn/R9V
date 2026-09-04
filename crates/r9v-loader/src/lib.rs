//! R9V model loading pipeline, memory budgeting, and repack cache (Spec 9, Spec 14 §2).
//!
//! Card A2.6 owns pipeline steps 1–4 with a single-device [`Plan`]:
//!
//! 1. [`open`] — metadata-only checkpoint open with the Spec 9 §3
//!    fingerprint; tensor payload bytes are never required.
//! 2. [`resolve_and_validate`] + [`bind`] — family resolution (Spec 8 §4),
//!    model validation collecting all failures (Spec 8 §6), and binding
//!    every required tensor with semantic-role placement legality
//!    (Spec 1 §2.3).
//! 3. [`plan_single_device`] — the canonical `r9v-ir` [`Plan`] for zero or
//!    one device (Spec 5 §5.1–§5.2), until `r9v-part` exists.
//! 4. [`check_device_budget`] / [`check_host_budget`] — exact checked
//!    budgets with byte shortfalls and actionable suggestions (Spec 9 §4).
//!
//! [`prepare`] runs all four steps cohesively. Materialization (card A2.7),
//! repack (card A2.8), tokenizers (card A2.9), and the load report
//! (card A2.10) are explicitly out of scope.
//!
//! Repository standards: `CONVENTIONS.md`; engineering bar:
//! `.agents/skills/r9v-engineering-standards`.

pub mod bind;
pub mod budget;
pub mod error;
pub mod open;
pub mod pipeline;
pub mod plan;
pub mod validate;

pub use bind::{
    bind, intended_placement, is_stacked_expert_weight, placement_is_legal, BindReport, BoundTensor,
};
pub use budget::{
    align_up_256, arena_layout, check_device_budget, check_host_budget, DeviceBudget,
    DeviceBudgetInput, HostBudget, HostBudgetInput, DEFAULT_CHUNK_BYTES, DEFAULT_QUEUE_DEPTH,
    DEFAULT_RESERVE_BYTES, TENSOR_ALIGN_BYTES,
};
pub use error::{BudgetScope, LoaderError, TensorProblem, TensorProblemKind};
pub use open::{
    open, open_shard_set, open_shard_set_with_file_sizes, open_with_file_size, GgufFileMeta,
    ModelFingerprint, OpenedCheckpoint,
};
pub use pipeline::{
    prepare, prepare_shard_set, prepare_with_file_size, strategy_for, PrepareOptions, PreparedLoad,
};
pub use plan::{plan_single_device, PlannedDevice};
pub use r9v_ir::Plan;
pub use validate::{
    check_fusion_decls, downgrade_absent_mtp, model_id_from_meta, resolve_and_validate,
    ValidatedModel,
};
