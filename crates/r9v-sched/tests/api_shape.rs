// SPDX-License-Identifier: Apache-2.0
//! API shape and trait bound verification for r9v-sched (Spec 14 §2, r9v-card-work §6).

use std::error::Error;
use std::fmt::Debug;

use r9v_sched::{
    BatchWorkspace, ByteDetokenizer, CapturedGraph, CostTable, CostTableStub, Detokenizer,
    DeviceStepSample, EventId, EventKind, FinishReason, GraphCache, GraphMode, InlineVec,
    ProfileMode, Request, SchedError, ScheduleLogRing, ScheduleRecord, Scheduler, SchedulerConfig,
    Sequence, SequencePhase, SlotRange, Step, StepBudgetConfig, StepEventChain, StepEventRecord,
    StepExecutor, StepGraphProgram, StepInputs, StepProgramOp, StepResult, StopCriteria,
    StreamKind, WorkspaceArena, DEFAULT_MAX_OUTSTANDING, SCHEDULE_LOG_CAPACITY,
    WORKSPACE_ALIGNMENT,
};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_copy<T: Copy>() {}
fn assert_clone<T: Clone>() {}
fn assert_debug<T: Debug>() {}
fn assert_error<T: Error>() {}

#[test]
fn api_shape_trait_bounds() {
    // Request & Sequence types
    assert_send::<Request>();
    assert_sync::<Request>();
    assert_clone::<Request>();
    assert_debug::<Request>();

    assert_send::<Sequence>();
    assert_sync::<Sequence>();
    assert_clone::<Sequence>();
    assert_debug::<Sequence>();

    assert_send::<SequencePhase>();
    assert_sync::<SequencePhase>();
    assert_clone::<SequencePhase>();
    assert_debug::<SequencePhase>();

    assert_send::<FinishReason>();
    assert_sync::<FinishReason>();
    assert_clone::<FinishReason>();
    assert_debug::<FinishReason>();

    assert_send::<StopCriteria>();
    assert_sync::<StopCriteria>();
    assert_clone::<StopCriteria>();
    assert_debug::<StopCriteria>();

    // Step types
    assert_send::<Step>();
    assert_sync::<Step>();
    assert_clone::<Step>();
    assert_debug::<Step>();

    assert_send::<StepResult>();
    assert_sync::<StepResult>();
    assert_clone::<StepResult>();
    assert_debug::<StepResult>();

    assert_send::<InlineVec<u32, 1>>();
    assert_sync::<InlineVec<u32, 1>>();
    assert_clone::<InlineVec<u32, 1>>();
    assert_debug::<InlineVec<u32, 1>>();

    // Arena
    assert_send::<WorkspaceArena>();
    assert_sync::<WorkspaceArena>();
    assert_clone::<WorkspaceArena>();
    assert_debug::<WorkspaceArena>();
    assert_eq!(WORKSPACE_ALIGNMENT, 256);

    // Streams & Events
    assert_send::<StreamKind>();
    assert_sync::<StreamKind>();
    assert_copy::<StreamKind>();
    assert_clone::<StreamKind>();
    assert_debug::<StreamKind>();

    assert_send::<EventKind>();
    assert_sync::<EventKind>();
    assert_copy::<EventKind>();
    assert_clone::<EventKind>();
    assert_debug::<EventKind>();

    assert_send::<EventId>();
    assert_sync::<EventId>();
    assert_copy::<EventId>();
    assert_clone::<EventId>();
    assert_debug::<EventId>();

    assert_send::<StepEventRecord>();
    assert_sync::<StepEventRecord>();
    assert_clone::<StepEventRecord>();
    assert_debug::<StepEventRecord>();

    assert_send::<StepEventChain>();
    assert_sync::<StepEventChain>();
    assert_clone::<StepEventChain>();
    assert_debug::<StepEventChain>();

    // Log ring
    assert_send::<ScheduleRecord>();
    assert_sync::<ScheduleRecord>();
    assert_clone::<ScheduleRecord>();
    assert_debug::<ScheduleRecord>();

    assert_send::<ScheduleLogRing>();
    assert_sync::<ScheduleLogRing>();
    assert_clone::<ScheduleLogRing>();
    assert_debug::<ScheduleLogRing>();
    assert_eq!(SCHEDULE_LOG_CAPACITY, 4096);

    assert_send::<GraphMode>();
    assert_sync::<GraphMode>();
    assert_copy::<GraphMode>();
    assert_clone::<GraphMode>();
    assert_debug::<GraphMode>();

    // Cost & Budget
    assert_send::<ProfileMode>();
    assert_sync::<ProfileMode>();
    assert_copy::<ProfileMode>();
    assert_clone::<ProfileMode>();
    assert_debug::<ProfileMode>();

    assert_send::<StepBudgetConfig>();
    assert_sync::<StepBudgetConfig>();
    assert_copy::<StepBudgetConfig>();
    assert_clone::<StepBudgetConfig>();
    assert_debug::<StepBudgetConfig>();

    assert_send::<CostTableStub>();
    assert_sync::<CostTableStub>();
    assert_clone::<CostTableStub>();
    assert_debug::<CostTableStub>();

    // Graphs
    assert_send::<CapturedGraph>();
    assert_sync::<CapturedGraph>();
    assert_clone::<CapturedGraph>();
    assert_debug::<CapturedGraph>();

    assert_send::<GraphCache>();
    assert_sync::<GraphCache>();
    assert_clone::<GraphCache>();
    assert_debug::<GraphCache>();

    assert_send::<StepGraphProgram>();
    assert_sync::<StepGraphProgram>();
    assert_clone::<StepGraphProgram>();
    assert_debug::<StepGraphProgram>();

    assert_send::<StepProgramOp>();
    assert_sync::<StepProgramOp>();
    assert_clone::<StepProgramOp>();
    assert_debug::<StepProgramOp>();

    assert_send::<ByteDetokenizer>();
    assert_sync::<ByteDetokenizer>();
    assert_clone::<ByteDetokenizer>();
    assert_debug::<ByteDetokenizer>();

    assert_send::<DeviceStepSample>();
    assert_sync::<DeviceStepSample>();
    assert_clone::<DeviceStepSample>();
    assert_debug::<DeviceStepSample>();

    assert_send::<StepInputs<'_>>();
    assert_sync::<StepInputs<'_>>();
    assert_clone::<StepInputs<'_>>();
    assert_debug::<StepInputs<'_>>();

    // Reservation/batch types carried by StepInputs (Spec 3 §5, Spec 1 §2.5)
    assert_send::<SlotRange>();
    assert_sync::<SlotRange>();
    assert_copy::<SlotRange>();
    assert_clone::<SlotRange>();
    assert_debug::<SlotRange>();

    assert_send::<BatchWorkspace>();
    assert_sync::<BatchWorkspace>();
    assert_debug::<BatchWorkspace>();
    const {
        assert!(DEFAULT_MAX_OUTSTANDING >= 1);
    }

    // Scheduler and Error
    assert_send::<Scheduler>();
    assert_send::<SchedulerConfig>();
    assert_sync::<SchedulerConfig>();
    assert_clone::<SchedulerConfig>();
    assert_debug::<SchedulerConfig>();

    assert_send::<SchedError>();
    assert_sync::<SchedError>();
    assert_debug::<SchedError>();
    assert_error::<SchedError>();
}

#[test]
fn api_shape_trait_objects() {
    // Verify dynamic polymorphism bounds for traits named in specs
    fn accept_cost_table(_: &dyn CostTable) {}
    fn accept_detokenizer(_: &dyn Detokenizer) {}
    fn accept_executor(_: &mut dyn StepExecutor) {}

    let cost_stub = CostTableStub::default();
    accept_cost_table(&cost_stub);

    let byte_detok = ByteDetokenizer::new();
    accept_detokenizer(&byte_detok);

    struct NoopExecutor;
    impl StepExecutor for NoopExecutor {
        fn upload_inputs(&mut self, _: &StepInputs<'_>) -> Result<(), SchedError> {
            Ok(())
        }
        fn replay_graph(&mut self, _: &CapturedGraph) -> Result<(), SchedError> {
            Ok(())
        }
        fn readback_sample(&mut self) -> Result<DeviceStepSample, SchedError> {
            Ok(DeviceStepSample {
                step_id: r9v_common::StepId::new(0),
                token: 0,
                accept_len: 1,
            })
        }
    }
    let mut exec = NoopExecutor;
    accept_executor(&mut exec);
}
