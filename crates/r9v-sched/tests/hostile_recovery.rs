// SPDX-License-Identifier: Apache-2.0
//! Hostile recovery, boundary, and bit-exactness tests for the A3.9 repair
//! (Spec 6 §§2,3,5,7,9; single sequence S=1, k=0).
//!
//! Each test injects exactly one fault or boundary and proves the scheduler
//! preserves the request, sequence, reservation, counters, and retry state:
//! no silent loss, no leaked tail, no duplicated commit, no reused StepId.

use std::sync::Arc;

use r9v_common::{ReqId, SeqId, StepId};
use r9v_ir::{AttentionMask, DType, Epilogue, LayoutId, PlanId, QuantScheme, SamplingParams};
use r9v_registry::{
    ArchName, AttentionStatic, BundleManifest, ElementwiseStatic, LaunchGeometry,
    ManifestVariantEntry, MatmulStatic, OpId, OpStatic, Registry, RegistryConfig, SamplingMethod,
    SamplingStatic, StubDevice, Tier, VariantHash,
};
use r9v_sched::{
    ByteDetokenizer, CapturedGraph, CostTableStub, Detokenizer, DeviceStepSample, FinishReason,
    GraphMode, ProfileMode, Request, SchedError, SchedResult, ScheduleRecord, Scheduler,
    SchedulerConfig, SlotRange, StepBudgetConfig, StepExecutor, StepGraphProgram, StepInputs,
    StepProgramOp, StopCriteria, WorkspaceArena, DEFAULT_MAX_OUTSTANDING,
};
use r9v_state::{CacheDtype, Retain, StateConfig, StateManager, StateSpec, BLOCK_TABLE_SENTINEL};

fn hostile_registry(arch: &ArchName) -> Registry {
    let mut manifest = BundleManifest::new(1, vec![arch.clone()]);
    for (idx, &op) in [OpId::Norm, OpId::Attention, OpId::Matmul, OpId::Sample]
        .iter()
        .enumerate()
    {
        let vhash = VariantHash::new(0x3000_0000_0000_0000 + (idx as u64) + 1);
        manifest.insert_variant(
            vhash,
            ManifestVariantEntry {
                arch: arch.clone(),
                file: format!("reference/{}.co", op.as_str()),
                tier: Tier::T1,
                entry_symbol: format!("t1_{}", op.as_str()),
                launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
                workspace_bytes: 4096,
                static_bytes: 8192,
                static_flops: 16384,
                op: Some(op),
                static_hash: None,
                validated: true,
                validated_on: Some("ref".to_owned()),
            },
        );
    }
    let mut registry = Registry::new(RegistryConfig {
        gen_version: 1,
        allow_jit: false,
        tune_budget_ms: 2000,
        allow_nondeterministic: false,
    });
    registry.set_manifest(manifest, None).expect("manifest");
    registry
}

fn hostile_program() -> StepGraphProgram {
    let template = vec![0u8; 64];
    let mut program = StepGraphProgram::new();
    program.add_op(StepProgramOp::new(
        OpId::Norm,
        "norm",
        |key| {
            Some(OpStatic::Elementwise(ElementwiseStatic {
                t_bucket: key.t_dec + key.t_pre,
                dims: vec![1024],
                dtypes: vec![DType::F16],
                fused_with: None,
            }))
        },
        template.clone(),
        Some(0),
    ));
    program.add_op(StepProgramOp::new(
        OpId::Attention,
        "attention_decode",
        |key| {
            Some(OpStatic::Attention(AttentionStatic {
                q_bucket: key.t_dec,
                h_local: 32,
                hkv_local: 8,
                d: 128,
                dv: 128,
                cache_dtype: DType::E4m3,
                attention_layout: LayoutId::CONTIGUOUS,
                mask_kind: AttentionMask::Causal,
                latent: None,
                softcap_bits: None,
                sinks: None,
            }))
        },
        template.clone(),
        Some(0),
    ));
    program.add_op(StepProgramOp::new(
        OpId::Matmul,
        "matmul",
        |key| {
            Some(OpStatic::Matmul(MatmulStatic {
                m_bucket: key.t_dec + key.t_pre,
                n: 1024,
                k: 1024,
                w_scheme: QuantScheme::None,
                w_layout: LayoutId::CONTIGUOUS,
                act_scheme: QuantScheme::None,
                out_dtype: DType::F16,
                epilogue: Epilogue::None,
                interleave: false,
                sparse: false,
            }))
        },
        template.clone(),
        Some(0),
    ));
    program.add_op(StepProgramOp::new(
        OpId::Sample,
        "sample",
        |key| {
            Some(OpStatic::Sampling(SamplingStatic {
                s_bucket: key.s,
                v: 32000,
                q_bucket: key.t_dec,
                method: SamplingMethod::InverseCdfSample,
            }))
        },
        template,
        Some(0),
    ));
    program
}

fn hostile_setup() -> (Scheduler, StubDevice) {
    let arch = ArchName::from("gfx1201");
    let registry = hostile_registry(&arch);
    let cost_table = Arc::new(CostTableStub::default());
    let state_manager = StateManager::new(
        StateConfig {
            max_ctx: 4096,
            max_seqs: 16,
        },
        vec![StateSpec::KvPaged {
            hkv: 8,
            d: 128,
            dv: 128,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        }],
        64 * 1024 * 1024,
    )
    .expect("state manager");
    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);
    let config = SchedulerConfig {
        step_budget_ms: StepBudgetConfig::Auto,
        profile: ProfileMode::Latency,
        prefill_min_chunk: 128,
        prefill_max_chunk: 2048,
        max_wait_ms: 500,
        max_seqs: 1,
        k_max: 0,
        min_accept: 0.3,
        graph_mode: GraphMode::List,
        plan_id: PlanId::new(1),
        rank: 0,
        vocab_size: 128,
        max_outstanding: DEFAULT_MAX_OUTSTANDING,
    };
    let scheduler = Scheduler::new(
        config,
        state_manager,
        registry,
        arch,
        cost_table,
        hostile_program(),
        arena,
    )
    .expect("scheduler");
    (scheduler, StubDevice::new())
}

fn hostile_sampling() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: vec![],
    }
}

fn hostile_req(id: u64, prompt_len: usize, max_tokens: u32) -> Request {
    Request::new(
        ReqId::new(id),
        vec![7; prompt_len],
        hostile_sampling(),
        max_tokens,
        StopCriteria::default(),
        false,
    )
    .expect("valid req")
}

/// Single fault-injection executor: captures the exact `StepInputs` upload,
/// echoes the candidate StepId (plus an optional wrong-step offset), and can
/// fail once at exactly one device-phase point (Spec 6 §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultPoint {
    Upload,
    Replay,
    Readback,
}

struct ProbeExecutor<'d> {
    device: &'d StubDevice,
    script: Vec<(u32, u32)>,
    idx: usize,
    fault: Option<FaultPoint>,
    wrong_step_offset: u64,
    uploads: usize,
    seen_step: u64,
    seen_seq: u64,
    seen_res_start: u32,
    seen_res_len: u32,
    seen_seed: u64,
    seen_rng: u64,
    seen_prefill: Option<(u32, u32)>,
    seen_ctx: u32,
    seen_prompt_len: usize,
    seen_generated_len: usize,
    batch_seqs: u32,
    batch_tokens: u32,
    batch_groups: u32,
    batch_seq_ids: Vec<u32>,
    batch_query_lens: Vec<u32>,
    batch_ctx_lens: Vec<u32>,
    batch_positions: Vec<u32>,
    batch_slot_map: Vec<u32>,
    batch_block_table: Vec<u32>,
    seen_temperature: f32,
    seen_reservation: Option<SlotRange>,
    prev_slot_map: Vec<u32>,
    prev_positions: Vec<u32>,
}

impl<'d> ProbeExecutor<'d> {
    fn new(device: &'d StubDevice, script: Vec<(u32, u32)>) -> Self {
        Self {
            device,
            script,
            idx: 0,
            fault: None,
            wrong_step_offset: 0,
            uploads: 0,
            seen_step: 0,
            seen_seq: 0,
            seen_res_start: 0,
            seen_res_len: 0,
            seen_seed: 0,
            seen_rng: 0,
            seen_prefill: None,
            seen_ctx: 0,
            seen_prompt_len: 0,
            seen_generated_len: 0,
            batch_seqs: 0,
            batch_tokens: 0,
            batch_groups: 0,
            batch_seq_ids: Vec::new(),
            batch_query_lens: Vec::new(),
            batch_ctx_lens: Vec::new(),
            batch_positions: Vec::new(),
            batch_slot_map: Vec::new(),
            batch_block_table: Vec::new(),
            seen_temperature: f32::NAN,
            seen_reservation: None,
            prev_slot_map: Vec::new(),
            prev_positions: Vec::new(),
        }
    }
}

impl StepExecutor for ProbeExecutor<'_> {
    fn upload_inputs(&mut self, input: &StepInputs<'_>) -> SchedResult<()> {
        if self.fault == Some(FaultPoint::Upload) {
            self.fault = None;
            return Err(SchedError::ExecutionFailed {
                detail: "injected upload fault".to_owned(),
            });
        }
        // Retire the previous upload's batch rows before capturing the new
        // ones, so consecutive steps can be checked for slot aliasing.
        self.prev_slot_map = std::mem::take(&mut self.batch_slot_map);
        self.prev_positions = std::mem::take(&mut self.batch_positions);
        self.seen_step = input.step_id.as_u64();
        self.seen_seq = input.seq_id.as_u64();
        self.seen_res_start = input.reservation.start();
        self.seen_res_len = input.reservation.len();
        self.seen_seed = input.seed;
        self.seen_rng = input.rng_counter;
        self.seen_prefill = input.prefill;
        self.seen_ctx = input.ctx_len;
        self.seen_prompt_len = input.prompt_tokens.len();
        self.seen_generated_len = input.generated_tokens.len();
        self.batch_seqs = input.batch.seqs();
        self.batch_tokens = input.batch.tokens();
        self.batch_groups = input.batch.groups();
        self.batch_seq_ids = input.batch.seq_ids().to_vec();
        self.batch_query_lens = input.batch.query_lens().to_vec();
        self.batch_ctx_lens = input.batch.ctx_lens().to_vec();
        self.batch_positions = input.batch.positions().to_vec();
        self.batch_slot_map = input.batch.slot_map().to_vec();
        self.batch_block_table = input.batch.block_table().to_vec();
        self.seen_temperature = input.sampling.temperature;
        self.seen_reservation = Some(input.reservation);
        self.uploads += 1;
        Ok(())
    }

    fn replay_graph(&mut self, graph: &CapturedGraph) -> SchedResult<()> {
        if self.fault == Some(FaultPoint::Replay) {
            self.fault = None;
            return Err(SchedError::ExecutionFailed {
                detail: "injected replay fault".to_owned(),
            });
        }
        graph.launches.replay(self.device, None)?;
        Ok(())
    }

    fn readback_sample(&mut self) -> SchedResult<DeviceStepSample> {
        if self.fault == Some(FaultPoint::Readback) {
            self.fault = None;
            return Err(SchedError::ExecutionFailed {
                detail: "injected readback fault".to_owned(),
            });
        }
        let (token, accept_len) = self.script[self.idx % self.script.len()];
        self.idx += 1;
        Ok(DeviceStepSample {
            step_id: StepId::new(self.seen_step.wrapping_add(self.wrong_step_offset)),
            token,
            accept_len,
        })
    }
}

// -----------------------------------------------------------------------------
// StepInputs exact-state + descriptor bit-exactness (Spec 6 §3.1, Spec 3 §5)
// -----------------------------------------------------------------------------

/// The upload carries the live reservation descriptor, the filled batch
/// tensors, sampling params, and the deterministic (seed, step) identity —
/// every value bit-exact against the state manager (Spec 6 §3.1 steps 6-9).
#[test]
fn step_inputs_carries_exact_executable_state() {
    let (mut scheduler, stub) = hostile_setup();
    let req = hostile_req(501, 256, 10).with_seed(0x1234_5678_9ABC_DEF0);
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    let mut exec = ProbeExecutor::new(&stub, vec![(42, 0)]);
    let res = scheduler.step(&mut exec).expect("step").expect("res");
    assert_eq!(res.record.chunk, 128);

    // Reservation descriptor: start is the pre-step ctx, len the chunk.
    assert_eq!(exec.seen_seq, seq_id.as_u64());
    assert_eq!(exec.seen_res_start, 0);
    assert_eq!(exec.seen_res_len, 128);
    assert_eq!(exec.seen_ctx, 0);
    assert_eq!(exec.seen_prefill, Some((0, 128)));
    assert_eq!(exec.seen_prompt_len, 256);
    assert_eq!(exec.seen_generated_len, 0);
    // Candidate identity + deterministic RNG inputs.
    assert_eq!(exec.seen_step, 1);
    assert_eq!(exec.seen_seed, 0x1234_5678_9ABC_DEF0);
    assert_eq!(exec.seen_rng, 1);
    assert_eq!(exec.seen_temperature, 0.0);
    // Filled batch dims for one sequence of 128 tokens in one group.
    assert_eq!(exec.batch_seqs, 1);
    assert_eq!(exec.batch_tokens, 128);
    assert_eq!(exec.batch_groups, 1);
    assert_eq!(exec.batch_seq_ids, vec![seq_id.as_u64() as u32]);
    assert_eq!(exec.batch_query_lens, vec![128]);
    assert_eq!(exec.batch_ctx_lens, vec![0]);
    // Default scalar positions are ctx + k.
    assert_eq!(exec.batch_positions.len(), 128);
    for (k, pos) in exec.batch_positions.iter().enumerate() {
        assert_eq!(*pos, k as u32);
    }
    // Slot layout law (Spec 1 §2.5: slot = block_id * 32 + lane): every
    // filled slot is mapped, all 128 are pairwise distinct, and slots inside
    // one 32-token block run contiguously.
    assert_eq!(exec.batch_slot_map.len(), 128);
    let mut sorted = exec.batch_slot_map.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 128, "filled slots must be pairwise distinct");
    for (k, window) in exec.batch_slot_map.windows(2).enumerate() {
        if (k + 1) % 32 != 0 {
            assert_eq!(
                window[1],
                window[0] + 1,
                "slots inside one block are contiguous at k={k}"
            );
        }
    }
    // The carried descriptor matches the observed upload facts.
    let range = exec.seen_reservation.expect("reservation carried");
    assert_eq!(range.seq(), seq_id);
    assert_eq!((range.start(), range.len()), (0, 128));
    assert!(!range.is_empty());
    // Scoping: the committed range is stale — resolving it post-commit fails
    // typed instead of reading another step's tail (Spec 3 §3.6).
    assert!(scheduler.state_manager().slot(&range, 0, 0).is_err());
}

/// Consecutive steps never alias slots, positions advance as ctx + k, and the
/// block table carries mapped heads with sentinel holes (Spec 1 §2.5,
/// Spec 3 §5).
#[test]
fn consecutive_steps_never_alias_slots() {
    let (mut scheduler, stub) = hostile_setup();
    let seq_id = scheduler
        .enqueue_request(hostile_req(502, 256, 10))
        .expect("enqueue");

    // Step 1 (intermediate): positions 0..128. Step 2 (final): 128..256.
    let mut exec = ProbeExecutor::new(&stub, vec![(42, 0), (43, 1)]);
    scheduler.step(&mut exec).expect("step 1").expect("res");
    scheduler.step(&mut exec).expect("step 2").expect("res");

    let mgr = scheduler.state_manager();
    assert_eq!(mgr.ctx_len(seq_id).expect("ctx"), 256);
    assert_eq!(mgr.tail_len(seq_id).expect("tail"), 0);
    // Second upload positions are ctx + k with ctx = 128.
    assert_eq!(exec.batch_positions.len(), 128);
    for (k, pos) in exec.batch_positions.iter().enumerate() {
        assert_eq!(*pos, 128 + k as u32);
    }
    // No slot aliases the previous step's rows: the reservation advanced.
    assert_eq!(exec.batch_slot_map.len(), 128);
    assert_eq!(exec.prev_slot_map.len(), 128);
    for slot in &exec.batch_slot_map {
        assert!(
            !exec.prev_slot_map.contains(slot),
            "step 2 slot {slot} aliases step 1"
        );
    }
    // Block table is [G, S, max_blocks] with max_blocks = 4096 / 32 = 128.
    // Eight blocks back 256 tokens; the rest are sentinel holes.
    assert_eq!(exec.batch_block_table.len(), 128);
    assert_ne!(exec.batch_block_table[0], BLOCK_TABLE_SENTINEL);
    assert_ne!(exec.batch_block_table[7], BLOCK_TABLE_SENTINEL);
    assert_eq!(exec.batch_block_table[8], BLOCK_TABLE_SENTINEL);
    assert_eq!(exec.batch_block_table[127], BLOCK_TABLE_SENTINEL);
    // The scheduler-owned workspace retains the last filled batch.
    assert_eq!(scheduler.batch_workspace().tokens(), 128);
    assert_eq!(scheduler.batch_workspace().seqs(), 1);
}

// -----------------------------------------------------------------------------
// Transactional device-phase failures (Spec 6 §3.2, §8)
// -----------------------------------------------------------------------------

/// An upload fault preserves the sequence, clears the tail, burns no commit,
/// writes no log — and the retry succeeds with the next StepId (Spec 6 §8).
#[test]
fn upload_failure_preserves_sequence_and_retry_succeeds() {
    let (mut scheduler, stub) = hostile_setup();
    let seq_id = scheduler
        .enqueue_request(hostile_req(503, 128, 10))
        .expect("enqueue");

    let mut exec = ProbeExecutor::new(&stub, vec![(42, 1)]);
    exec.fault = Some(FaultPoint::Upload);
    let err = scheduler.step(&mut exec).unwrap_err();
    match err {
        SchedError::ExecutionFailed { detail } => assert!(detail.contains("upload")),
        other => panic!("expected ExecutionFailed, got {other:?}"),
    }
    // Preserved: still active, no tail stranded, nothing committed or logged.
    assert!(!scheduler.is_idle());
    assert_eq!(scheduler.state_manager().tail_len(seq_id).expect("tail"), 0);
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 0);
    assert!(scheduler.last_committed_step().is_none());
    assert!(scheduler.in_flight_step().is_none());
    assert_eq!(scheduler.schedule_log().total_written(), 0);

    // Retry is a clean rerun with the burned id skipped, never reused.
    let res = scheduler.step(&mut exec).expect("retry").expect("res");
    assert_eq!(res.step_id.as_u64(), 2);
    assert_eq!(scheduler.last_committed_step(), Some(StepId::new(2)));
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 128);
}

/// A replay fault aborts the open tail without committing (Spec 6 §8).
#[test]
fn replay_failure_aborts_without_commit() {
    let (mut scheduler, stub) = hostile_setup();
    let seq_id = scheduler
        .enqueue_request(hostile_req(504, 128, 10))
        .expect("enqueue");

    let mut exec = ProbeExecutor::new(&stub, vec![(42, 1)]);
    exec.fault = Some(FaultPoint::Replay);
    let err = scheduler.step(&mut exec).unwrap_err();
    match err {
        SchedError::ExecutionFailed { detail } => assert!(detail.contains("replay")),
        other => panic!("expected ExecutionFailed, got {other:?}"),
    }
    assert_eq!(scheduler.state_manager().tail_len(seq_id).expect("tail"), 0);
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 0);
    assert!(scheduler.last_committed_step().is_none());
    assert_eq!(scheduler.schedule_log().total_written(), 0);

    let res = scheduler.step(&mut exec).expect("retry").expect("res");
    assert_eq!(res.step_id.as_u64(), 2);
}

/// A readback fault — after the graph already replayed — still commits
/// nothing and strands no tail (Spec 6 §3.2).
#[test]
fn readback_failure_aborts_without_commit() {
    let (mut scheduler, stub) = hostile_setup();
    let seq_id = scheduler
        .enqueue_request(hostile_req(505, 128, 10))
        .expect("enqueue");

    let mut exec = ProbeExecutor::new(&stub, vec![(42, 1)]);
    exec.fault = Some(FaultPoint::Readback);
    let err = scheduler.step(&mut exec).unwrap_err();
    match err {
        SchedError::ExecutionFailed { detail } => assert!(detail.contains("readback")),
        other => panic!("expected ExecutionFailed, got {other:?}"),
    }
    assert_eq!(scheduler.state_manager().tail_len(seq_id).expect("tail"), 0);
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 0);
    assert!(scheduler.last_committed_step().is_none());

    let res = scheduler.step(&mut exec).expect("retry").expect("res");
    assert_eq!(res.step_id.as_u64(), 2);
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 128);
}

// -----------------------------------------------------------------------------
// Stale/wrong StepIds and duplicate-commit safety (Spec 6 §3.2, §8)
// -----------------------------------------------------------------------------

/// A readback echoing the wrong StepId is a `StaleStep` fault with no commit;
/// the retry uses the next candidate, so the stranded id can never
/// double-commit (Spec 6 §3.2, §8).
#[test]
fn wrong_step_id_echo_is_stale_with_no_commit() {
    let (mut scheduler, stub) = hostile_setup();
    let seq_id = scheduler
        .enqueue_request(hostile_req(506, 128, 10))
        .expect("enqueue");

    let mut exec = ProbeExecutor::new(&stub, vec![(42, 1)]);
    exec.wrong_step_offset = 100;
    let err = scheduler.step(&mut exec).unwrap_err();
    match err {
        SchedError::StaleStep { expected, got } => {
            assert_eq!(expected, Some(1));
            assert_eq!(got, 101);
        }
        other => panic!("expected StaleStep, got {other:?}"),
    }
    // Nothing committed, no tail stranded, no log, sequence preserved.
    assert_eq!(scheduler.state_manager().tail_len(seq_id).expect("tail"), 0);
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 0);
    assert!(scheduler.last_committed_step().is_none());
    assert!(scheduler.in_flight_step().is_none());
    assert_eq!(scheduler.schedule_log().total_written(), 0);
    assert!(!scheduler.is_idle());

    // Retry with the correct echo commits exactly once under the next id.
    exec.wrong_step_offset = 0;
    let res = scheduler.step(&mut exec).expect("retry").expect("res");
    assert_eq!(res.step_id.as_u64(), 2);
    assert_eq!(scheduler.last_committed_step(), Some(StepId::new(2)));
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 128);
    assert_eq!(scheduler.schedule_log().total_written(), 1);
}

/// `abort_open_reservation` clears a stranded tail with a zero-commit and
/// reports whether anything was open; it never commits twice (Spec 6 §8).
#[test]
fn abort_open_reservation_recovers_stranded_tail() {
    let (mut scheduler, _stub) = hostile_setup();
    let seq_id = scheduler
        .enqueue_request(hostile_req(507, 128, 10))
        .expect("enqueue");

    // Nothing open: no-op, never a zero-commit.
    assert!(!scheduler.abort_open_reservation(seq_id).expect("abort"));

    // Strand a tail directly, then recover: ctx unchanged, tail cleared.
    scheduler
        .state_manager_mut()
        .reserve(seq_id, 16)
        .expect("reserve");
    assert_eq!(
        scheduler.state_manager().tail_len(seq_id).expect("tail"),
        16
    );
    assert!(scheduler.abort_open_reservation(seq_id).expect("abort"));
    assert_eq!(scheduler.state_manager().tail_len(seq_id).expect("tail"), 0);
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 0);
    assert!(!scheduler.abort_open_reservation(seq_id).expect("abort"));

    // Unknown sequences fail typed, mutating nothing.
    let err = scheduler
        .abort_open_reservation(SeqId::new(0xDEAD))
        .unwrap_err();
    match err {
        SchedError::State(_) => {}
        other => panic!("expected State error, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// Accept-length boundaries (Spec 6 §3.3, §9; k=0)
// -----------------------------------------------------------------------------

/// Intermediate prefill progress reports `accept_len == 0` and advances `done`
/// with no sampled token; impossible values (2, or 1-while-intermediate) are
/// rejected before mutation (Spec 6 §3.3, §9).
#[test]
fn intermediate_prefill_zero_accept_advances_without_token() {
    let (mut scheduler, stub) = hostile_setup();
    let seq_id = scheduler
        .enqueue_request(hostile_req(508, 256, 10))
        .expect("enqueue");

    // Intermediate chunk: device reports 0, done advances, nothing sampled.
    let mut exec = ProbeExecutor::new(&stub, vec![(42, 0), (43, 1)]);
    let res1 = scheduler.step(&mut exec).expect("step 1").expect("res");
    assert_eq!(res1.record.chunk, 128);
    assert_eq!(res1.record.accept_len.get(0).copied(), Some(0));
    assert_eq!(res1.accepted_tokens.len(), 0);
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 128);

    // Final chunk completes the prompt and samples exactly one token.
    let res2 = scheduler.step(&mut exec).expect("step 2").expect("res");
    assert_eq!(res2.record.accept_len.get(0).copied(), Some(1));
    assert_eq!(
        res2.accepted_tokens
            .get(0)
            .and_then(|(_, toks)| toks.get(0).copied()),
        Some(43)
    );
}

/// `accept_len == 2` is impossible with k=0 on every step kind (Spec 6 §9).
#[test]
fn accept_len_two_rejected_on_decode_and_prefill() {
    for prompt_len in [128usize, 256usize] {
        let (mut scheduler, stub) = hostile_setup();
        let seq_id = scheduler
            .enqueue_request(hostile_req(509 + prompt_len as u64, prompt_len, 10))
            .expect("enqueue");
        let mut exec = ProbeExecutor::new(&stub, vec![(42, 2)]);
        let err = scheduler.step(&mut exec).unwrap_err();
        match err {
            SchedError::ExecutionFailed { detail } => assert!(detail.contains("accept_len")),
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
        assert_eq!(scheduler.state_manager().tail_len(seq_id).expect("tail"), 0);
        assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 0);
        assert!(scheduler.last_committed_step().is_none());
    }
}

/// `accept_len == 1` on an intermediate chunk is a contract violation: the
/// device claims a sample the step never took (Spec 6 §3.3).
#[test]
fn accept_len_one_on_intermediate_chunk_rejected() {
    let (mut scheduler, stub) = hostile_setup();
    let seq_id = scheduler
        .enqueue_request(hostile_req(511, 256, 10))
        .expect("enqueue");

    let mut exec = ProbeExecutor::new(&stub, vec![(42, 1)]);
    let err = scheduler.step(&mut exec).unwrap_err();
    match err {
        SchedError::ExecutionFailed { detail } => assert!(detail.contains("accept_len")),
        other => panic!("expected ExecutionFailed, got {other:?}"),
    }
    assert_eq!(scheduler.state_manager().tail_len(seq_id).expect("tail"), 0);
    assert_eq!(scheduler.state_manager().ctx_len(seq_id).expect("ctx"), 0);
}

// -----------------------------------------------------------------------------
// Detokenizer partial-UTF-8 vs genuine-empty distinction (Spec 6 §7)
// -----------------------------------------------------------------------------

/// A buffered multi-byte start byte that equals an EOS id does not finish:
/// EOS/stop evaluation defers until the code point completes (Spec 6 §7).
#[test]
fn buffered_partial_utf8_defers_eos_until_complete() {
    let (mut scheduler, stub) = hostile_setup();
    // 0xC3 (195) opens a 2-byte sequence; it is also the configured EOS id.
    let stop = StopCriteria::new(vec![195], vec![]);
    let req = Request::new(
        ReqId::new(512),
        vec![7; 8],
        hostile_sampling(),
        10,
        stop,
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    // Step 1 (final prefill, 8 tokens): token 195 buffers, no finish.
    let mut exec = ProbeExecutor::new(&stub, vec![(195, 1), (0xA9, 1)]);
    let res1 = scheduler.step(&mut exec).expect("step 1").expect("res");
    assert_eq!(res1.finished_sequences.len(), 0);
    assert!(scheduler.get_finished_result(seq_id).is_none());

    // Step 2 (decode): continuation completes U+00E9; 169 is not EOS.
    let res2 = scheduler.step(&mut exec).expect("step 2").expect("res");
    assert_eq!(res2.finished_sequences.len(), 0);
    assert!(scheduler.get_finished_result(seq_id).is_none());
    let finished = scheduler.take_finished_result(seq_id);
    assert!(finished.is_none(), "sequence must still be live");
}

/// A zero-byte append with nothing buffered is a genuinely empty token text:
/// EOS evaluation still applies normally (Spec 6 §7).
#[test]
fn genuine_empty_token_text_still_applies_eos() {
    struct EmptyDetok;
    impl Detokenizer for EmptyDetok {
        fn append_token(
            &mut self,
            _seq_id: SeqId,
            _token: u32,
            _output: &mut String,
        ) -> SchedResult<usize> {
            Ok(0)
        }
    }

    let (mut scheduler, stub) = hostile_setup();
    scheduler.set_detokenizer(Box::new(EmptyDetok));
    let stop = StopCriteria::new(vec![77], vec![]);
    let req = Request::new(
        ReqId::new(513),
        vec![7; 8],
        hostile_sampling(),
        10,
        stop,
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    // Step 1 (final prefill): token 77 finishes via EOS despite empty text.
    let mut exec = ProbeExecutor::new(&stub, vec![(77, 1)]);
    let res1 = scheduler.step(&mut exec).expect("step 1").expect("res");
    assert_eq!(
        res1.finished_sequences.get(0),
        Some(&(seq_id, FinishReason::Eos(77)))
    );
}

/// The default byte detokenizer reports no buffering for plain ASCII, so
/// normal tokens evaluate EOS/stop exactly as before (Spec 6 §7).
#[test]
fn byte_detokenizer_buffered_len_zero_for_ascii() {
    let mut detok = ByteDetokenizer::new();
    let seq = SeqId::new(9);
    let mut out = String::new();
    let n = detok.append_token(seq, 65, &mut out).expect("ascii");
    assert_eq!(n, 1);
    assert_eq!(out, "A");
    assert_eq!(detok.buffered_len(seq), 0);
    assert_eq!(detok.buffered_len(SeqId::new(10)), 0);
}

// -----------------------------------------------------------------------------
// Seed ownership and replay determinism (Spec 1 §4.F, Spec 6 §1)
// -----------------------------------------------------------------------------

/// `Request::new` defaults the seed to the request id; explicit seeds stick.
#[test]
fn request_seed_defaults_to_req_id_and_explicit_sticks() {
    let a = hostile_req(514, 8, 4);
    assert_eq!(a.seed, 514);
    let b = a.with_seed(999);
    assert_eq!(b.seed, 999);
    assert_eq!(a.seed, 514);
    let c = Request::new_with_seed(
        ReqId::new(515),
        vec![7; 8],
        hostile_sampling(),
        4,
        StopCriteria::default(),
        false,
        4242,
    )
    .expect("valid req");
    assert_eq!(c.seed, 4242);
}

/// Identical seeds replay bit-identically; the upload carries the seed and
/// the (seed, step) RNG identity every step (Spec 1 §4.F, Spec 6 §1).
#[test]
fn identical_seeds_replay_bit_identically() {
    fn run(seed: u64) -> (Vec<u32>, Vec<ScheduleRecord>, Vec<u64>) {
        let (mut scheduler, stub) = hostile_setup();
        let req = hostile_req(516, 128, 6).with_seed(seed);
        let seq_id = scheduler.enqueue_request(req).expect("enqueue");
        let mut exec = ProbeExecutor::new(&stub, vec![(60, 1), (61, 1), (62, 1)]);
        let mut seeds = Vec::new();
        while scheduler.get_finished_result(seq_id).is_none() {
            let res = scheduler.step(&mut exec).expect("step").expect("res");
            seeds.push(exec.seen_seed);
            let _ = res;
        }
        let (req_out, generated, _) = scheduler.take_finished_result(seq_id).expect("finished");
        let _ = req_out;
        let logs = scheduler.schedule_log().to_vec();
        (generated, logs, seeds)
    }

    let (gen_a, logs_a, seeds_a) = run(777);
    let (gen_b, logs_b, seeds_b) = run(777);
    assert_eq!(gen_a, gen_b);
    assert_eq!(logs_a, logs_b);
    assert!(!seeds_a.is_empty());
    assert!(seeds_a.iter().all(|&s| s == 777));
    assert_eq!(seeds_a, seeds_b);

    // A different seed is a different RNG identity on every upload.
    let (_, _, seeds_c) = run(778);
    assert!(seeds_c.iter().all(|&s| s == 778));
    assert!(!seeds_a.is_empty());
}
