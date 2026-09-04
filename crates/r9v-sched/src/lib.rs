// SPDX-License-Identifier: Apache-2.0
#![forbid(unsafe_code)]
//! R9V sequence scheduler, step graph execution, and budgeting (Spec 6, Spec 14 §2).
//!
//! This crate implements Card A3.9:
//! - Complete public Request/Sequence/Step API (Spec 6 §2).
//! - Deterministic pre-step -> device -> post-step loop (Spec 6 §3).
//! - Prefill admission, chunk selection, and budget resolution (Spec 6 §4).
//! - Step-graph capture per `(S=1, T_dec, T_pre)` against the kernel registry (Spec 6 §5.1, §5.2).
//! - Checked scratch workspace arena (Spec 6 §5.3).
//! - Three streams (`Compute`, `Comms`, `Copy`) and explicit event chain (Spec 6 §5.4).
//! - Memory pressure handling with sequence pause (Spec 6 §6).
//! - Finish handling for EOS, max_tokens, and stop strings (Spec 6 §7).
//! - Diagnostic schedule log ring buffer (Spec 6 §9).
//! - Internal speculative decoding proposer adapter with k=0 (Spec 7 §2).

pub mod arena;
pub mod cost;
pub mod error;
pub mod graph;
pub mod log;
pub(crate) mod proposer;
pub mod scheduler;
pub mod streams;
pub mod types;

pub use arena::{WorkspaceArena, WORKSPACE_ALIGNMENT};
pub use cost::{estimate_pre_step_ms, CostTable, CostTableStub, ProfileMode, StepBudgetConfig};
pub use error::{SchedError, SchedResult};
pub use graph::{
    ArgsTemplateBuilder, CapturedGraph, GraphCache, StepGraphBuilder, StepGraphProgram,
    StepProgramOp, WARM_S, WARM_T_DEC, WARM_T_PRE,
};
pub use log::{GraphMode, ScheduleLogRing, ScheduleRecord, SCHEDULE_LOG_CAPACITY};
pub use scheduler::{
    DeviceStepSample, Scheduler, SchedulerConfig, StepExecutor, StepInputs, DEFAULT_MAX_OUTSTANDING,
};
// Re-exported reservation/batch types naming the `StepInputs` contract
// (Spec 3 §5, Spec 1 §2.5).
pub use r9v_state::{BatchWorkspace, SlotRange};
pub use streams::{EventId, EventKind, StepEventChain, StepEventRecord, StreamKind};
pub use types::{
    ByteDetokenizer, Detokenizer, FinishReason, InlineVec, Request, Sequence, SequencePhase, Step,
    StepResult, StopCriteria,
};
