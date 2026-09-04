// SPDX-License-Identifier: Apache-2.0
//! Adversarial lifecycle, budget, memory pressure, and error handling tests (Spec 6, Card A3.9).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use r9v_common::{ReqId, SeqId, StepId};
use r9v_ir::{
    AttentionMask, DType, Epilogue, LayoutId, PlanId, QuantScheme, SamplingParams, StepGraphKey,
};
use r9v_registry::{
    ArchName, AttentionStatic, BundleManifest, ElementwiseStatic, LaunchGeometry,
    ManifestVariantEntry, MatmulStatic, OpId, OpStatic, Registry, RegistryConfig, SamplingMethod,
    SamplingStatic, StubDevice, Tier, VariantHash,
};
use r9v_sched::{
    ByteDetokenizer, CapturedGraph, CostTable, CostTableStub, Detokenizer, DeviceStepSample,
    EventKind, FinishReason, GraphCache, GraphMode, InlineVec, ProfileMode, Request, SchedError,
    SchedResult, ScheduleLogRing, ScheduleRecord, Scheduler, SchedulerConfig, Sequence,
    StepBudgetConfig, StepEventChain, StepExecutor, StepGraphBuilder, StepGraphProgram, StepInputs,
    StepProgramOp, StopCriteria, StreamKind, WorkspaceArena, DEFAULT_MAX_OUTSTANDING,
};
use r9v_state::{CacheDtype, Retain, StateConfig, StateManager, StateSpec};

// -----------------------------------------------------------------------------
// Zero-Allocation Counting Harness (Thread-Safe)
// -----------------------------------------------------------------------------

struct CountingAlloc;

std::thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = COUNTING.try_with(|c| {
            if c.get() {
                c.set(false);
                let _ = ALLOC_COUNT.try_with(|cnt| cnt.set(cnt.get() + 1));
                c.set(true);
            }
        });
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = COUNTING.try_with(|c| {
            if c.get() {
                c.set(false);
                let _ = ALLOC_COUNT.try_with(|cnt| cnt.set(cnt.get() + 1));
                c.set(true);
            }
        });
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

fn start_alloc_counting() {
    ALLOC_COUNT.with(|c| c.set(0));
    COUNTING.with(|c| c.set(true));
}

fn stop_alloc_counting() -> usize {
    COUNTING.with(|c| c.set(false));
    ALLOC_COUNT.with(|c| c.get())
}

// Deterministic stub StepExecutor shared by most tests: replays the captured graph
// into the stub device and returns scripted (token, accept_len) pairs in order,
// cycling. Scripted values model the device Sample op output; production has no
// host sampler path.
struct ScriptExecutor<'d> {
    device: &'d StubDevice,
    script: Vec<(u32, u32)>,
    idx: usize,
    last_step: u64,
    last_seq: u64,
    last_prompt_len: usize,
    last_ctx_len: u32,
    last_prefill: Option<(u32, u32)>,
    uploads: usize,
}

impl<'d> ScriptExecutor<'d> {
    fn new(device: &'d StubDevice, script: Vec<(u32, u32)>) -> Self {
        Self {
            device,
            script,
            idx: 0,
            last_step: 0,
            last_seq: 0,
            last_prompt_len: 0,
            last_ctx_len: 0,
            last_prefill: None,
            uploads: 0,
        }
    }

    fn script_token(token: u32) -> Vec<(u32, u32)> {
        vec![(token, 1)]
    }
}

impl StepExecutor for ScriptExecutor<'_> {
    fn upload_inputs(&mut self, input: &StepInputs<'_>) -> SchedResult<()> {
        self.last_step = input.step_id.as_u64();
        self.last_seq = input.seq_id.as_u64();
        self.last_prompt_len = input.prompt_tokens.len();
        self.last_ctx_len = input.ctx_len;
        self.last_prefill = input.prefill;
        self.uploads += 1;
        Ok(())
    }
    fn replay_graph(&mut self, graph: &CapturedGraph) -> SchedResult<()> {
        graph.launches.replay(self.device, None)?;
        Ok(())
    }
    fn readback_sample(&mut self) -> SchedResult<DeviceStepSample> {
        let (token, accept_len) = self.script[self.idx % self.script.len()];
        self.idx += 1;
        Ok(DeviceStepSample {
            step_id: StepId::new(self.last_step),
            token,
            accept_len,
        })
    }
}

// Allocation-free stub for the scheduler-local isolation test: replay only touches
// the captured graph length and readback computes printable ASCII arithmetically,
// so the measured step region performs no heap allocation through the executor.
struct SilentStub {
    counter: u64,
    last_step: u64,
    last_prefill_is_intermediate: bool,
}

impl SilentStub {
    fn new() -> Self {
        Self {
            counter: 0,
            last_step: 0,
            last_prefill_is_intermediate: false,
        }
    }
}

impl StepExecutor for SilentStub {
    fn upload_inputs(&mut self, input: &StepInputs<'_>) -> SchedResult<()> {
        self.last_step = input.step_id.as_u64();
        // An intermediate prefill chunk leaves prompt tokens unprocessed, so
        // the device reports no sampled token for it (Spec 6 §3.3 step 3).
        self.last_prefill_is_intermediate = match input.prefill {
            Some((done, chunk)) => {
                (done as usize).saturating_add(chunk as usize) < input.prompt_tokens.len()
            }
            None => false,
        };
        Ok(())
    }
    fn replay_graph(&mut self, graph: &CapturedGraph) -> SchedResult<()> {
        let _ = graph.launches.len();
        Ok(())
    }
    fn readback_sample(&mut self) -> SchedResult<DeviceStepSample> {
        self.counter = self.counter.wrapping_add(1);
        Ok(DeviceStepSample {
            step_id: StepId::new(self.last_step),
            token: ((self.counter % 96) + 32) as u32,
            accept_len: if self.last_prefill_is_intermediate {
                0
            } else {
                1
            },
        })
    }
}

// -----------------------------------------------------------------------------
// Test Helpers
// -----------------------------------------------------------------------------

fn create_test_registry(arch: &ArchName) -> Registry {
    let mut manifest = BundleManifest::new(1, vec![arch.clone()]);
    let ops = [OpId::Norm, OpId::Attention, OpId::Matmul, OpId::Sample];

    for (idx, &op) in ops.iter().enumerate() {
        let vhash = VariantHash::new(0x2000_0000_0000_0000 + (idx as u64) + 1);
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
    registry
        .set_manifest(manifest, None)
        .expect("valid manifest");
    registry
}

fn adv_args_template() -> Vec<u8> {
    vec![0u8; 64]
}

fn create_test_program() -> StepGraphProgram {
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
        adv_args_template(),
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
        adv_args_template(),
        Some(0),
    ));
    program.add_op(StepProgramOp::new(
        OpId::Attention,
        "attention_prefill",
        |key| {
            if key.t_pre > 0 {
                Some(OpStatic::Attention(AttentionStatic {
                    q_bucket: key.t_pre,
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
            } else {
                None
            }
        },
        adv_args_template(),
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
        adv_args_template(),
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
        adv_args_template(),
        Some(0),
    ));
    program
}

fn default_test_setup() -> (Scheduler, StubDevice) {
    setup_with_outstanding(DEFAULT_MAX_OUTSTANDING)
}

fn setup_with_outstanding(max_outstanding: u32) -> (Scheduler, StubDevice) {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);
    let cost_table = Arc::new(CostTableStub::default());
    let state_config = StateConfig {
        max_ctx: 4096,
        max_seqs: 16,
    };
    let layer_specs = vec![StateSpec::KvPaged {
        hkv: 8,
        d: 128,
        dv: 128,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }];
    let state_manager =
        StateManager::new(state_config, layer_specs, 64 * 1024 * 1024).expect("state manager init");
    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);

    // DECISION(A3.9): stub-tier tokens are byte-domain (vocab 128) so every read-back
    // token is valid ASCII the default ByteDetokenizer decodes exactly; a wider vocab
    // would emit byte values outside any valid UTF-8 stream, which the exact
    // incremental detokenizer contract rejects. Spec 6 §7.
    let scheduler_config = SchedulerConfig {
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
        max_outstanding,
    };

    let program = create_test_program();
    let scheduler = Scheduler::new(
        scheduler_config,
        state_manager,
        registry,
        arch,
        cost_table,
        program,
        arena,
    )
    .expect("scheduler init");

    let stub_device = StubDevice::new();
    (scheduler, stub_device)
}

fn make_sampling_params() -> SamplingParams {
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

// -----------------------------------------------------------------------------
// Baseline Lifecycle Tests
// -----------------------------------------------------------------------------

#[test]
fn test_lifecycle_queued_to_prefill_to_decode_to_eos() {
    let (mut scheduler, stub_device) = default_test_setup();

    // Prompt of 256 tokens (two 128-token chunks)
    let prompt_tokens: Vec<u32> = (0..256).map(|i| i + 1).collect();
    let req = Request::new(
        ReqId::new(1),
        prompt_tokens,
        make_sampling_params(),
        50,
        StopCriteria::new(vec![77], vec![]),
        false,
    )
    .expect("valid request");

    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    // Stub readback emits regular tokens then EOS token (77) on the 4th readback.
    // The EOS id is byte-domain so the default ByteDetokenizer accepts it; every
    // generated token passes through detokenization before the EOS check.
    // The first chunk is intermediate prefill (no token sampled yet), so it
    // reports accept_len 0; later steps report 1 (Spec 6 §3.3).
    let mut exec = ScriptExecutor::new(&stub_device, vec![(42, 0), (42, 1), (42, 1), (77, 1)]);

    // Step 1: Prefill chunk 1 (128 tokens)
    let res1 = scheduler
        .step(&mut exec)
        .expect("step 1")
        .expect("step result");
    assert_eq!(res1.record.chunk, 128);
    assert_eq!(res1.record.t_pre, 128);
    assert_eq!(res1.finished_sequences.len(), 0);

    // Step 2: Prefill chunk 2 (128 tokens) -> prompt complete, transitions to Decoding
    let res2 = scheduler
        .step(&mut exec)
        .expect("step 2")
        .expect("step result");
    assert_eq!(res2.record.chunk, 128);
    assert_eq!(res2.accepted_tokens.len(), 1); // Emits first token from prompt completion

    // Step 3: Decode step 1
    let res3 = scheduler
        .step(&mut exec)
        .expect("step 3")
        .expect("step result");
    assert_eq!(res3.record.t_dec, 1);
    assert_eq!(res3.record.chunk, 0);

    // Step 4: Decode step 2 -> emits token 77 (EOS)
    let res4 = scheduler
        .step(&mut exec)
        .expect("step 4")
        .expect("step result");
    assert_eq!(res4.finished_sequences.len(), 1);
    assert_eq!(
        res4.finished_sequences.get(0),
        Some(&(seq_id, FinishReason::Eos(77)))
    );

    // Verify finished record stored in scheduler
    let finished = scheduler
        .get_finished_result(seq_id)
        .expect("finished result present");
    assert_eq!(finished.2, FinishReason::Eos(77));
    assert!(scheduler.is_idle());
}

#[test]
fn test_lifecycle_finish_max_tokens() {
    let (mut scheduler, stub_device) = default_test_setup();

    // 128 tokens prompt, max_tokens = 3
    let prompt_tokens: Vec<u32> = (0..128).map(|i| i + 1).collect();
    let req = Request::new(
        ReqId::new(2),
        prompt_tokens,
        make_sampling_params(),
        3,
        StopCriteria::new(vec![], vec![]),
        false,
    )
    .expect("valid request");

    let seq_id = scheduler.enqueue_request(req).expect("enqueue");
    let mut exec = ScriptExecutor::new(&stub_device, ScriptExecutor::script_token(42));

    // Step 1: Prefill 128 tokens -> prompt completes, records 1st generated token
    let res1 = scheduler
        .step(&mut exec)
        .expect("step 1")
        .expect("step result");
    assert_eq!(res1.record.chunk, 128);

    // Step 2: Decode token 2
    let res2 = scheduler
        .step(&mut exec)
        .expect("step 2")
        .expect("step result");
    assert_eq!(res2.finished_sequences.len(), 0);

    // Step 3: Decode token 3 -> reaches max_tokens = 3
    let res3 = scheduler
        .step(&mut exec)
        .expect("step 3")
        .expect("step result");
    assert_eq!(res3.finished_sequences.len(), 1);
    assert_eq!(
        res3.finished_sequences.get(0),
        Some(&(seq_id, FinishReason::MaxTokens))
    );

    let finished = scheduler
        .get_finished_result(seq_id)
        .expect("finished present");
    assert_eq!(finished.1.len(), 3);
    assert_eq!(finished.2, FinishReason::MaxTokens);
    assert!(scheduler.is_idle());
}

#[test]
fn test_lifecycle_finish_stop_string_and_trim() {
    let (mut scheduler, stub_device) = default_test_setup();

    let stop = StopCriteria::new(vec![], vec!["STOP".to_owned()]);
    let req = Request::new(
        ReqId::new(3),
        vec![65, 66, 67], // "ABC"
        make_sampling_params(),
        20,
        stop,
        false,
    )
    .expect("valid request");

    struct AsciiDetokenizer;
    impl Detokenizer for AsciiDetokenizer {
        fn append_token(
            &mut self,
            _seq_id: SeqId,
            token: u32,
            output: &mut String,
        ) -> Result<usize, SchedError> {
            let c = (token & 0xFF) as u8 as char;
            output.push(c);
            Ok(c.len_utf8())
        }
    }
    scheduler.set_detokenizer(Box::new(AsciiDetokenizer));

    // Stub readback emits S, T, O, P in order, cycling
    let mut exec = ScriptExecutor::new(&stub_device, vec![(83, 1), (84, 1), (79, 1), (80, 1)]);

    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    let mut finished = false;
    for _ in 0..10 {
        if let Some(res) = scheduler.step(&mut exec).expect("step") {
            if let Some((_, reason)) = res.finished_sequences.first() {
                assert_eq!(*reason, FinishReason::StopString("STOP".to_owned()));
                finished = true;
                break;
            }
        }
    }
    assert!(finished, "sequence must finish via StopString");

    let record = scheduler
        .get_finished_result(seq_id)
        .expect("finished record present");
    assert_eq!(record.2, FinishReason::StopString("STOP".to_owned()));
}

#[test]
fn test_lifecycle_explicit_cancellation() {
    let (mut scheduler, _stub_device) = default_test_setup();

    let req1 = Request::new(
        ReqId::new(10),
        vec![1, 2, 3],
        make_sampling_params(),
        10,
        StopCriteria::default(),
        false,
    )
    .expect("valid request");

    let req2 = Request::new(
        ReqId::new(20),
        vec![4, 5, 6],
        make_sampling_params(),
        10,
        StopCriteria::default(),
        false,
    )
    .expect("valid request");

    let seq1 = scheduler.enqueue_request(req1).expect("enqueue 1");
    let seq2 = scheduler.enqueue_request(req2).expect("enqueue 2");

    let cancelled_req = scheduler
        .cancel_request(ReqId::new(20))
        .expect("cancel req");
    assert!(cancelled_req);

    let cancelled_seq = scheduler.cancel_sequence(seq1).expect("cancel seq");
    assert!(cancelled_seq);

    let double_cancel = scheduler.cancel_sequence(seq2).expect("double cancel");
    assert!(!double_cancel);

    assert!(scheduler.is_idle());
}

// -----------------------------------------------------------------------------
// Baseline Budget & Error Tests
// -----------------------------------------------------------------------------

#[test]
fn test_budget_resolution_latency_vs_throughput_vs_manual() {
    let cost_table = CostTableStub::new(8.0, 0.05, 0.02);

    let latency_budget = cost_table
        .resolve_budget_ms(StepBudgetConfig::Auto, ProfileMode::Latency)
        .expect("latency budget resolves");
    assert!((latency_budget - 10.0).abs() < 1e-5);

    let throughput_budget = cost_table
        .resolve_budget_ms(StepBudgetConfig::Auto, ProfileMode::Throughput)
        .expect("throughput budget resolves");
    assert!((throughput_budget - 64.0).abs() < 1e-5);

    let manual_budget = cost_table
        .resolve_budget_ms(StepBudgetConfig::Manual(25.5), ProfileMode::Latency)
        .expect("manual budget resolves");
    assert!((manual_budget - 25.5).abs() < 1e-5);

    // Invalid budgets fail closed with a typed error, never a clamped zero.
    for bad in [
        StepBudgetConfig::Manual(f32::NAN),
        StepBudgetConfig::Manual(f32::INFINITY),
        StepBudgetConfig::Manual(0.0),
        StepBudgetConfig::Manual(-10.0),
    ] {
        let err = cost_table
            .resolve_budget_ms(bad, ProfileMode::Latency)
            .unwrap_err();
        match err {
            SchedError::InvalidCost { context, .. } => {
                assert_eq!(context, "step_budget");
            }
            other => panic!("expected InvalidCost, got {other:?}"),
        }
    }
}

#[test]
fn test_budget_prefill_chunk_sizing_and_forced_admission() {
    let mut cost_table = CostTableStub::new(8.0, 0.05, 0.02);
    cost_table.set_bucket_cost(1, 0, 512, 20.0);
    cost_table.set_bucket_cost(1, 0, 128, 5.0);

    let room = 6.0;
    let chunk = cost_table
        .select_prefill_chunk(512, 0, 0, room, 128, 2048)
        .expect("chunk selection resolves");
    assert_eq!(chunk, Some(128), "must select largest chunk that fits room");

    let tight_room = 1.0;
    let no_chunk = cost_table
        .select_prefill_chunk(512, 0, 0, tight_room, 128, 2048)
        .expect("chunk selection resolves");
    assert_eq!(no_chunk, None, "no chunk fits under tight budget");

    // Non-finite room fails closed with a typed error.
    let err = cost_table
        .select_prefill_chunk(512, 0, 0, f32::NAN, 128, 2048)
        .unwrap_err();
    match err {
        SchedError::InvalidCost { context, .. } => {
            assert_eq!(context, "prefill_room");
        }
        other => panic!("expected InvalidCost, got {other:?}"),
    }
}

#[test]
fn test_budget_forced_admission_after_max_wait_timeout() {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);

    let mut cost_table = CostTableStub::new(8.0, 0.05, 0.02);
    cost_table.set_bucket_cost(1, 0, 128, 50.0);
    cost_table.set_bucket_cost(1, 0, 256, 50.0);
    let cost_table_arc = Arc::new(cost_table);

    let state_config = StateConfig {
        max_ctx: 4096,
        max_seqs: 16,
    };
    let layer_specs = vec![StateSpec::KvPaged {
        hkv: 8,
        d: 128,
        dv: 128,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }];
    let state_manager =
        StateManager::new(state_config, layer_specs, 64 * 1024 * 1024).expect("state manager init");
    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);

    let scheduler_config = SchedulerConfig {
        step_budget_ms: StepBudgetConfig::Manual(10.0),
        profile: ProfileMode::Latency,
        prefill_min_chunk: 128,
        prefill_max_chunk: 2048,
        max_wait_ms: 45,
        max_seqs: 1,
        k_max: 0,
        min_accept: 0.3,
        graph_mode: GraphMode::List,
        plan_id: PlanId::new(1),
        rank: 0,
        vocab_size: 128,
        max_outstanding: DEFAULT_MAX_OUTSTANDING,
    };

    let program = create_test_program();
    let mut scheduler = Scheduler::new(
        scheduler_config,
        state_manager,
        registry,
        arch,
        cost_table_arc,
        program,
        arena,
    )
    .expect("scheduler init");

    let req = Request::new(
        ReqId::new(99),
        vec![10; 256],
        make_sampling_params(),
        10,
        StopCriteria::default(),
        false,
    )
    .expect("valid req");

    scheduler.enqueue_request(req).expect("enqueue");
    let stub_device = StubDevice::new();
    // The forced 128-token chunk of a 256-token prompt is intermediate
    // prefill, so the device reports accept_len 0 for it (Spec 6 §3.3).
    let mut exec = ScriptExecutor::new(&stub_device, vec![(42, 0), (42, 1)]);

    for _ in 0..4 {
        let res = scheduler.step(&mut exec).expect("step wait");
        assert!(res.is_none(), "step should wait while under wait timeout");
    }

    let forced_res = scheduler.step(&mut exec).expect("step forced");
    assert!(forced_res.is_some());
    let step_res = forced_res.unwrap();
    assert!(
        step_res.record.forced_admission,
        "forced_admission must be true in schedule record"
    );
    assert_eq!(step_res.record.chunk, 128);
}

#[test]
fn test_error_invalid_requests_refused_with_collect_all() {
    let err1 = Request::new(
        ReqId::new(1),
        vec![],
        make_sampling_params(),
        10,
        StopCriteria::default(),
        false,
    )
    .unwrap_err();

    match err1 {
        SchedError::InvalidRequest { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("tokens must not be empty")));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }

    let err2 = Request::new(
        ReqId::new(2),
        vec![1, 2, 3],
        make_sampling_params(),
        0,
        StopCriteria::default(),
        false,
    )
    .unwrap_err();

    match err2 {
        SchedError::InvalidRequest { problems } => {
            assert!(problems
                .iter()
                .any(|p| p.contains("max_tokens must be >= 1")));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }

    let bad_sampling = SamplingParams {
        temperature: -1.0,
        top_k: 0,
        top_p: 1.5,
        min_p: 0.0,
        repetition_penalty: -2.0,
        presence_penalty: f32::NAN,
        frequency_penalty: 0.0,
        logit_bias: vec![],
    };

    let err3 = Request::new(
        ReqId::new(3),
        vec![1, 2, 3],
        bad_sampling,
        10,
        StopCriteria::default(),
        false,
    )
    .unwrap_err();

    match err3 {
        SchedError::InvalidRequest { problems } => {
            assert!(
                problems.len() >= 4,
                "must collect all invalid sampling params, got {problems:?}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn test_error_workspace_arena_overflow_reports_numbers() {
    let mut arena = WorkspaceArena::new(0, 1024);

    let off1 = arena.allocate_slice(512).expect("alloc 1");
    assert_eq!(off1, 0);
    assert_eq!(arena.allocated_bytes(), 512);

    let err = arena.allocate_slice(1024).unwrap_err();
    match err {
        SchedError::ArenaOverflow {
            required,
            available,
            shortfall,
        } => {
            assert_eq!(required, 1536);
            assert_eq!(available, 1024);
            assert_eq!(shortfall, 512);
        }
        other => panic!("expected ArenaOverflow, got {other:?}"),
    }
}

#[test]
fn test_three_streams_and_event_chain_order() {
    let mut chain = StepEventChain::new();
    let step_id = StepId::new(1);
    chain.begin_step(step_id);

    // Canonical five-edge order per Spec 6 §5.4.
    let upload = chain.record_upload_complete(step_id).expect("upload");
    let wait_upload = chain
        .record_wait(step_id, StreamKind::Compute, upload)
        .expect("wait upload");
    let compute = chain.record_compute_complete(step_id).expect("compute");
    let wait_compute = chain
        .record_wait(step_id, StreamKind::Copy, compute)
        .expect("wait compute");
    chain.record_readback_complete(step_id).expect("readback");

    assert!(upload < wait_upload);
    assert!(wait_upload < compute);
    assert!(compute < wait_compute);

    // Wait records store the exact awaited event.
    assert_eq!(chain.get(1).expect("rec 1").awaited_event, Some(upload));
    assert_eq!(chain.get(3).expect("rec 3").awaited_event, Some(compute));
    assert_eq!(chain.get(0).expect("rec 0").awaited_event, None);

    chain
        .validate_step_chain(step_id)
        .expect("valid step chain");

    // Incomplete chain is rejected with the exact-five count.
    let step_id_bad = StepId::new(2);
    chain.begin_step(step_id_bad);
    chain.record_upload_complete(step_id_bad).expect("upload 2");
    let err = chain.validate_step_chain(step_id_bad).unwrap_err();
    match err {
        SchedError::ExecutionFailed { detail } => {
            assert!(detail.contains("expected exactly 5 stream events"));
        }
        other => panic!("expected ExecutionFailed, got {other:?}"),
    }

    // Wrong awaited event is rejected: wait references a foreign event id.
    let step_id_wrong = StepId::new(3);
    chain.begin_step(step_id_wrong);
    let up = chain
        .record_upload_complete(step_id_wrong)
        .expect("upload 3");
    let other = chain
        .record_compute_complete(step_id_wrong)
        .expect("compute 3");
    assert_ne!(up, other);
    chain
        .record_wait(step_id_wrong, StreamKind::Compute, other)
        .expect("wrong wait");
    chain
        .record_wait(step_id_wrong, StreamKind::Copy, up)
        .expect("wrong wait 2");
    chain
        .record_readback_complete(step_id_wrong)
        .expect("readback 3");
    let err2 = chain.validate_step_chain(step_id_wrong).unwrap_err();
    match err2 {
        SchedError::ExecutionFailed { detail } => {
            assert!(detail.contains("event 1 must be Compute wait"));
        }
        other => panic!("expected ExecutionFailed, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// Focused Adversarial Tests (Requirements 1-13)
// -----------------------------------------------------------------------------

/// Adversarial Requirement 1: Paused-sequence retry of oldest decode before queued work; clean stall before replay;
/// no StepId gap; decode never preempted or killed for prefill.
#[test]
fn test_adv_01_paused_sequence_retry_stall_no_gap_decode_never_preempted() {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);
    let cost_table = Arc::new(CostTableStub::default());
    let state_config = StateConfig {
        max_ctx: 256,
        max_seqs: 16,
    };
    // 8 blocks * 64KB + overhead = 532480 bytes pool
    let layer_specs = vec![StateSpec::KvPaged {
        hkv: 8,
        d: 128,
        dv: 128,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }];
    let mut state_manager =
        StateManager::new(state_config, layer_specs, 532480).expect("state manager init");

    // Occupy 4 blocks with a background sequence so exactly 4 blocks remain for req1
    let (blocker_id, _) = state_manager.new_seq(&[1; 128]).expect("new blocker seq");
    state_manager
        .reserve(blocker_id, 128)
        .expect("reserve blocker");
    state_manager
        .commit(blocker_id, 128)
        .expect("commit blocker");

    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);
    let scheduler_config = SchedulerConfig {
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
    let program = create_test_program();
    let mut scheduler = Scheduler::new(
        scheduler_config,
        state_manager,
        registry,
        arch,
        cost_table,
        program,
        arena,
    )
    .expect("scheduler init");
    let stub_device = StubDevice::new();
    let mut exec = ScriptExecutor::new(&stub_device, ScriptExecutor::script_token(42));

    // Enqueue request 1 (128 prompt tokens)
    let req1 = Request::new(
        ReqId::new(101),
        vec![42; 128],
        make_sampling_params(),
        10,
        StopCriteria::default(),
        false,
    )
    .expect("valid req1");
    let seq1 = scheduler.enqueue_request(req1).expect("enqueue req1");

    // Step 1: Prefill req1 consumes the remaining 4 blocks.
    let res1 = scheduler
        .step(&mut exec)
        .expect("step 1")
        .expect("step result 1");
    assert_eq!(res1.step_id.as_u64(), 1);
    assert_eq!(res1.record.step_id.as_u64(), 1);
    assert_eq!(res1.step.step_id.as_u64(), 1);
    assert_eq!(res1.record.chunk, 128);

    // Enqueue request 2 (which should wait and NOT preempt or kill decode)
    let req2 = Request::new(
        ReqId::new(102),
        vec![99; 128],
        make_sampling_params(),
        10,
        StopCriteria::default(),
        false,
    )
    .expect("valid req2");
    let _seq2 = scheduler.enqueue_request(req2).expect("enqueue req2");

    // Step 2: req1 tries to decode token 129, requiring a 5th block.
    // Pool is exhausted, so req1 is paused. Scheduler stalls cleanly before device replay.
    // StepId must NOT be allocated.
    let res2 = scheduler.step(&mut exec).expect("step 2 stall");
    assert!(
        res2.is_none(),
        "step 2 must stall cleanly on memory pressure"
    );
    assert_eq!(scheduler.active_sequence_count(), 2);

    // Step 3: Next step retries the oldest paused decode before queued work.
    // Memory pressure persists, so it stalls cleanly again. Still no StepId allocated.
    let res3 = scheduler.step(&mut exec).expect("step 3 stall");
    assert!(
        res3.is_none(),
        "step 3 must retry oldest paused and stall cleanly"
    );

    // Release memory by freeing the blocker sequence
    scheduler
        .state_manager_mut()
        .free_seq(blocker_id)
        .expect("free blocker seq");

    // Step 4: Retrying paused decode now succeeds!
    let res4 = scheduler
        .step(&mut exec)
        .expect("step 4")
        .expect("step result 4");
    // CRITICAL: StepId allocation was deferred so step 4 has StepId = 2 with NO unlogged gap!
    assert_eq!(
        res4.step_id.as_u64(),
        2,
        "StepId must be strictly sequential with no gap"
    );
    assert_eq!(res4.record.step_id.as_u64(), 2);
    assert_eq!(res4.step.step_id.as_u64(), 2);
    assert_eq!(res4.record.t_dec, 1);
    assert_eq!(res4.step.seqs_decode.get(0), Some(&seq1));
    assert_eq!(res4.accepted_tokens.len(), 1);
    // Pause telemetry: the exact paused sequence ID is retained in the fixed slot and
    // reported on the next completed record, then cleared.
    assert_eq!(res4.record.paused.len(), 1);
    assert_eq!(res4.record.paused.get(0), Some(&seq1));

    // Step 5: the slot cleared, so the following record carries no pause.
    let res5 = scheduler
        .step(&mut exec)
        .expect("step 5")
        .expect("step result 5");
    assert_eq!(res5.step_id.as_u64(), 3);
    assert_eq!(res5.record.paused.len(), 0);
}

/// Adversarial Requirement 2: Incremental bounded stop-tail handling using max_stop_len, exact token boundary mapping,
/// trimming spanning pieces, keeping prefix before STOP, resetting state across sequences.
#[test]
fn test_adv_02_bounded_stop_tail_exact_token_boundary_and_reset() {
    let (mut scheduler, stub_device) = default_test_setup();

    // Test token boundary mapping with multi-byte stop string:
    // Prompt: "PROMPT"
    // Token 10: "alpha "
    // Token 11: "beta "
    // Token 12: "STOP" (spans stop string)
    // Token 13: "extra"
    // Stop string: "STOP"
    let stop = StopCriteria::new(vec![], vec!["STOP".to_owned()]);
    assert_eq!(stop.max_stop_len, 4);

    struct CustomByteDetok;
    impl Detokenizer for CustomByteDetok {
        fn append_token(
            &mut self,
            _seq_id: SeqId,
            token: u32,
            output: &mut String,
        ) -> Result<usize, SchedError> {
            let text = match token {
                10 => "alpha ",
                11 => "beta ",
                12 => "STOP_WORD",
                13 => "tail",
                _ => "x",
            };
            output.push_str(text);
            Ok(text.len())
        }
    }
    scheduler.set_detokenizer(Box::new(CustomByteDetok));

    // 1st token from prefill prompt finish: 10 ("alpha ")
    // 2nd token decode 1: 11 ("beta ")
    // 3rd token decode 2: 12 ("STOP_WORD")
    let mut exec = ScriptExecutor::new(&stub_device, vec![(10, 1), (11, 1), (12, 1), (13, 1)]);

    let req = Request::new(
        ReqId::new(201),
        vec![1; 128],
        make_sampling_params(),
        10,
        stop,
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    // Step 1: Prefill prompt -> emits token 10 ("alpha ")
    let res1 = scheduler.step(&mut exec).expect("step 1").expect("res 1");
    assert_eq!(res1.accepted_tokens.len(), 1);
    assert_eq!(
        res1.accepted_tokens
            .get(0)
            .and_then(|(_, toks)| toks.get(0).copied()),
        Some(10)
    );

    // Step 2: Decode token 11 ("beta ")
    let res2 = scheduler.step(&mut exec).expect("step 2").expect("res 2");
    assert_eq!(res2.accepted_tokens.len(), 1);
    assert_eq!(
        res2.accepted_tokens
            .get(0)
            .and_then(|(_, toks)| toks.get(0).copied()),
        Some(11)
    );

    // Step 3: Decode token 12 ("STOP_WORD") -> matches "STOP"
    // Trims spanning token 12; prefix tokens [10, 11] kept!
    let res3 = scheduler.step(&mut exec).expect("step 3").expect("res 3");
    assert_eq!(res3.finished_sequences.len(), 1);
    assert_eq!(
        res3.finished_sequences.get(0),
        Some(&(seq_id, FinishReason::StopString("STOP".to_owned())))
    );

    let (finished_req, finished_tokens, finish_reason) = scheduler
        .get_finished_result(seq_id)
        .expect("finished result present");
    assert_eq!(*finish_reason, FinishReason::StopString("STOP".to_owned()));
    assert_eq!(finished_req.id, ReqId::new(201));
    // Kept prefix tokens before STOP (10 and 11); token 12 trimmed!
    assert_eq!(*finished_tokens, vec![10, 11]);

    // Verify reset across sequences: subsequent sequence starts with empty detokenizer tail
    let req2 = Request::new(
        ReqId::new(202),
        vec![1; 128],
        make_sampling_params(),
        5,
        StopCriteria::new(vec![], vec!["NEVER".to_owned()]),
        false,
    )
    .expect("req2");
    let seq_id2 = scheduler.enqueue_request(req2).expect("enqueue 2");
    let res_next = scheduler
        .step(&mut exec)
        .expect("step req2")
        .expect("res req2");
    assert_eq!(
        res_next
            .accepted_tokens
            .get(0)
            .and_then(|(_, toks)| toks.get(0).copied()),
        Some(13)
    );
    assert!(scheduler.cancel_sequence(seq_id2).expect("cancel"));
}

/// 3. StepEventChain fixed 16 slots inline storage, monotonic event IDs, and 3 streams + wait edges.
#[test]
fn test_adv_03_event_chain_fixed_16_capacity_and_stream_dependencies() {
    let mut chain = StepEventChain::new();
    let step_id = StepId::new(5);
    chain.begin_step(step_id);

    let e1 = chain.record_upload_complete(step_id).expect("e1");
    let e2 = chain
        .record_wait(step_id, StreamKind::Compute, e1)
        .expect("e2");
    let e3 = chain.record_compute_complete(step_id).expect("e3");
    let e4 = chain
        .record_wait(step_id, StreamKind::Copy, e3)
        .expect("e4");
    let e5 = chain.record_readback_complete(step_id).expect("e5");

    // Enforce strictly monotonic event IDs
    assert!(e1 < e2);
    assert!(e2 < e3);
    assert!(e3 < e4);
    assert!(e4 < e5);

    // Three streams modeled
    assert_eq!(chain.get(0).unwrap().stream, StreamKind::Copy);
    assert_eq!(chain.get(1).unwrap().stream, StreamKind::Compute);
    assert_eq!(chain.get(2).unwrap().stream, StreamKind::Compute);
    assert_eq!(chain.get(3).unwrap().stream, StreamKind::Copy);
    assert_eq!(chain.get(4).unwrap().stream, StreamKind::Copy);

    chain
        .validate_step_chain(step_id)
        .expect("valid step chain");

    // Capacity test: inline array holds up to 16 events
    for _ in 5..16 {
        chain
            .record_comms_event(step_id, EventKind::ComputeComplete)
            .expect("record within 16");
    }
    assert_eq!(chain.len(), 16);

    // 17th event fails with ArithmeticOverflow (storage exceeded)
    let err = chain
        .record_comms_event(step_id, EventKind::ComputeComplete)
        .unwrap_err();
    match err {
        SchedError::ArithmeticOverflow { what, details } => {
            assert_eq!(what, "step_event_storage");
            assert!(details.contains("16 events per step exceeded"));
        }
        other => panic!("expected ArithmeticOverflow, got {other:?}"),
    }
}

/// Adversarial Requirement 4: accept_len derivation (0 for intermediate prefill, 1 for final prefill and decode),
/// and deterministic cost-table-driven timing stubs.
#[test]
fn test_adv_04_accept_len_derivation_and_timing_stubs() {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);
    let mut cost_table = CostTableStub::default();
    cost_table.set_bucket_cost(1, 0, 128, 7.5);
    cost_table.set_bucket_cost(1, 1, 0, 8.0);
    let cost_table_arc = Arc::new(cost_table);

    let state_config = StateConfig {
        max_ctx: 4096,
        max_seqs: 16,
    };
    let layer_specs = vec![StateSpec::KvPaged {
        hkv: 8,
        d: 128,
        dv: 128,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }];
    let state_manager =
        StateManager::new(state_config, layer_specs, 64 * 1024 * 1024).expect("state manager init");
    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);
    let scheduler_config = SchedulerConfig {
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
    let program = create_test_program();
    let mut scheduler = Scheduler::new(
        scheduler_config,
        state_manager,
        registry,
        arch,
        cost_table_arc,
        program,
        arena,
    )
    .expect("scheduler init");
    let stub_device = StubDevice::new();
    // Intermediate prefill reports accept_len 0, final prefill and decode
    // report 1 (Spec 6 §3.3).
    let mut exec = ScriptExecutor::new(&stub_device, vec![(42, 0), (42, 1), (42, 1)]);

    // 256 tokens prompt (two 128-token chunks)
    let req = Request::new(
        ReqId::new(401),
        vec![1; 256],
        make_sampling_params(),
        5,
        StopCriteria::default(),
        false,
    )
    .expect("valid req");
    scheduler.enqueue_request(req).expect("enqueue");

    // Step 1: Intermediate prefill chunk (128 tokens)
    let res1 = scheduler.step(&mut exec).expect("step 1").expect("res 1");
    assert_eq!(
        res1.record.accept_len.get(0).copied(),
        Some(0),
        "intermediate prefill accept_len must be 0"
    );
    assert_eq!(res1.record.accept_len.len(), 1);
    assert_eq!(res1.accepted_tokens.len(), 0);
    assert_eq!(res1.record.t_device_us, 7500, "cost_ms=7.5 -> 7500 us");
    assert_eq!(res1.record.t_post_us, 0);

    // Step 2: Final prefill chunk (128 tokens) -> completes prompt, produces first token
    let res2 = scheduler.step(&mut exec).expect("step 2").expect("res 2");
    assert_eq!(
        res2.record.accept_len.get(0).copied(),
        Some(1),
        "final prefill accept_len must be 1"
    );
    assert_eq!(res2.accepted_tokens.len(), 1);
    assert_eq!(res2.record.t_device_us, 7500);
    assert_eq!(res2.record.t_post_us, 0);

    // Step 3: Decode step 1
    let res3 = scheduler.step(&mut exec).expect("step 3").expect("res 3");
    assert_eq!(
        res3.record.accept_len.get(0).copied(),
        Some(1),
        "decode accept_len must be 1"
    );
    assert_eq!(res3.accepted_tokens.len(), 1);
    assert_eq!(res3.record.t_device_us, 8000, "cost_ms=8.0 -> 8000 us");
    assert_eq!(res3.record.t_post_us, 0);
}

/// 5. Step construction: real public Step struct instantiated in pre-step and embedded in StepResult.
#[test]
fn test_adv_05_step_construction_and_routing() {
    let (mut scheduler, stub_device) = default_test_setup();
    let mut exec = ScriptExecutor::new(&stub_device, ScriptExecutor::script_token(42));

    let req = Request::new(
        ReqId::new(501),
        vec![1; 128],
        make_sampling_params(),
        5,
        StopCriteria::default(),
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    let res = scheduler.step(&mut exec).expect("step").expect("res");
    // Verify Step is populated and embedded in StepResult
    assert_eq!(res.step.step_id, res.step_id);
    assert_eq!(res.step.seqs_prefill.len(), 1);
    assert_eq!(res.step.seqs_prefill.get(0), Some(&(seq_id, 128)));
    assert_eq!(res.step.seqs_decode.len(), 0);
    assert_eq!(res.step.k.len(), 1);
    assert_eq!(res.step.k.get(0).copied(), Some(0));
    assert_eq!(res.step.bucket, res.record.bucket);
}

/// Adversarial Requirement 6: Workspace arena: 256-byte aligned distinct offsets, args binding, non-overlap,
/// no runtime arena mutation during graph capture; lazy growth recaptures every graph (Spec 6 §5.3).
#[test]
fn test_adv_06_workspace_arena_256_align_no_overlap_and_lazy_growth_recapture() {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);
    let program = create_test_program();
    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);

    let key = StepGraphKey {
        plan_id: PlanId::new(1),
        rank: 0,
        s: 1,
        t_dec: 1,
        t_pre: 0,
        segment: 0,
    };

    let allocated_before = arena.allocated_bytes();
    let captured = StepGraphBuilder::capture(&registry, &arch, key, &program, &arena)
        .expect("capture must succeed");
    let allocated_after = arena.allocated_bytes();

    // Capture must never mutate runtime arena
    assert_eq!(
        allocated_before, allocated_after,
        "runtime arena must not be mutated during graph capture"
    );

    // Verify 256-byte aligned distinct offsets and pairwise non-overlap
    let offsets = &captured.workspace_offsets;
    assert!(offsets.len() >= 2);
    for &off in offsets {
        assert_eq!(off % 256, 0, "offset {off} must be 256-byte aligned");
    }

    // Verify pairwise non-overlap
    for i in 0..offsets.len() {
        for j in (i + 1)..offsets.len() {
            let start_i = offsets[i];
            let end_i = start_i + 4096; // ops use 4096 workspace bytes in test registry
            let start_j = offsets[j];
            let end_j = start_j + 4096;
            assert!(
                end_i <= start_j || end_j <= start_i,
                "workspace intervals [{start_i}, {end_i}) and [{start_j}, {end_j}) must not overlap"
            );
        }
    }

    // Verify args_blob binding
    for (i, entry) in captured.launches.entries().iter().enumerate() {
        let bound_offset = u64::from_le_bytes(entry.args_blob[..8].try_into().unwrap());
        assert_eq!(bound_offset, offsets[i]);
    }

    // Lazy growth (Spec 6 §5.3): an undersized arena grows on first use of a larger
    // bucket and every cached graph is recaptured under the new generation.
    let mut small_arena = WorkspaceArena::new(0, 1024);
    assert_eq!(small_arena.generation(), 1);
    let mut cache = GraphCache::new(create_test_program());
    let (graph, was_captured) = cache
        .get_or_capture(&registry, &arch, key, &mut small_arena)
        .expect("lazy capture with growth must succeed");
    assert!(was_captured, "first use must lazily capture");
    assert!(
        small_arena.capacity_bytes() >= graph.required_workspace_bytes,
        "arena must grow to fit the captured graph"
    );
    assert_eq!(
        small_arena.capacity_bytes() % 256,
        0,
        "grown capacity stays 256-byte aligned"
    );
    assert_eq!(small_arena.generation(), 2, "growth bumps the generation");
    assert_eq!(
        cache.arena_generation(),
        2,
        "cache tracks the grown arena generation"
    );

    // A cached hit performs no growth and no recapture.
    let gen_before = small_arena.generation();
    let (_, was_captured_again) = cache
        .get_or_capture(&registry, &arch, key, &mut small_arena)
        .expect("cached hit");
    assert!(!was_captured_again, "cached graph must not recapture");
    assert_eq!(small_arena.generation(), gen_before);

    // Eager warm capture never grows: capacity and generation are untouched.
    let warm_arena = WorkspaceArena::new(0, 32 * 1024 * 1024);
    let cap_before = warm_arena.capacity_bytes();
    let warm_cache = GraphCache::eager_capture_warm_buckets(
        &registry,
        &arch,
        PlanId::new(1),
        0,
        create_test_program(),
        &warm_arena,
    )
    .expect("warm capture");
    assert_eq!(warm_arena.capacity_bytes(), cap_before);
    assert_eq!(warm_arena.generation(), 1);
    assert_eq!(warm_cache.arena_generation(), 1);
    assert!(!warm_cache.is_empty(), "warm buckets must be captured");
}

/// Adversarial Requirement 7: Spec 6 §4.1 discrete prefill chunk selection over BUCKET_SIZES; logical remainder against
/// enclosing bucket when tail < min_chunk; accumulated wait time in ms.
#[test]
fn test_adv_07_discrete_prefill_chunk_selection_and_remainder() {
    let cost_table = CostTableStub::default();

    // Never injects prompt_len into search: chunk selection returns standard discrete bucket
    let chunk = cost_table
        .select_prefill_chunk(300, 0, 0, 100.0, 128, 2048)
        .expect("chunk selection resolves");
    assert_eq!(chunk, Some(256), "must select discrete bucket 256");

    let chunk2 = cost_table
        .select_prefill_chunk(1000, 0, 0, 100.0, 128, 2048)
        .expect("chunk selection resolves");
    assert_eq!(chunk2, Some(512), "must select discrete bucket 512");

    // Tail < min_chunk (50 tokens when min_chunk=128):
    // Test that scheduler runs exact logical remainder 50 against enclosing bucket 128
    let (mut scheduler, stub_device) = default_test_setup();
    let mut exec = ScriptExecutor::new(&stub_device, ScriptExecutor::script_token(42));
    let req = Request::new(
        ReqId::new(701),
        vec![1; 50],
        make_sampling_params(),
        5,
        StopCriteria::default(),
        false,
    )
    .expect("valid req");
    scheduler.enqueue_request(req).expect("enqueue");

    let res = scheduler.step(&mut exec).expect("step").expect("res");
    assert_eq!(
        res.record.chunk, 50,
        "logical chunk is exact remaining tokens"
    );
    assert_eq!(res.record.t_pre, 50, "logical t_pre is 50");
    assert_eq!(
        res.record.bucket.2, 64,
        "enclosing bucket in BUCKET_SIZES is 64"
    );
}

/// 8. Checked arithmetic across the crate and typed ArithmeticOverflow errors.
#[test]
fn test_adv_08_arithmetic_overflow_errors() {
    let mut arena = WorkspaceArena::new(0, u64::MAX);
    let _ = arena.allocate_slice(u64::MAX - 256).expect("first alloc");
    let err = arena.allocate_slice(512).unwrap_err();
    match err {
        SchedError::ArithmeticOverflow { what, details } => {
            assert_eq!(what, "arena_slice");
            assert!(details.contains("offset="));
        }
        other => panic!("expected ArithmeticOverflow, got {other:?}"),
    }

    let mut inline_vec: InlineVec<u32, 2> = InlineVec::new();
    inline_vec.push(1).expect("push 1");
    inline_vec.push(2).expect("push 2");
    let push_err = inline_vec.push(3).unwrap_err();
    match push_err {
        SchedError::ArithmeticOverflow { what, details } => {
            assert_eq!(what, "inline_vec_push");
            assert!(details.contains("capacity 2 exceeded"));
        }
        other => panic!("expected ArithmeticOverflow, got {other:?}"),
    }
}

/// 9. Schedule log 4096-entry ring buffer wrap with monotonic total_written and chronological reads.
#[test]
fn test_adv_09_schedule_log_true_ring_wrap_monotonic() {
    let mut ring = ScheduleLogRing::new(4096);

    for i in 1..=5000 {
        let record = ScheduleRecord {
            step_id: StepId::new(i),
            t_pre_us: 100,
            t_draft_us: 0,
            t_device_us: 500,
            t_post_us: 0,
            s: 1,
            t_dec: 1,
            t_pre: 0,
            chunk: 0,
            k: InlineVec::new(),
            accept_len: InlineVec::new(),
            forced_admission: false,
            budget_ms: 10.0,
            bucket: (1, 1, 0),
            graph_mode: GraphMode::List,
            captured: false,
            paused: InlineVec::new(),
            segment_sync_us: 0,
        };
        ring.push(record).expect("push");
    }

    assert_eq!(ring.total_written(), 5000);
    assert_eq!(ring.len(), 4096);

    let records = ring.to_vec();
    assert_eq!(records.len(), 4096);
    // Oldest surviving record is step 905, newest is step 5000
    assert_eq!(records[0].step_id.as_u64(), 905);
    assert_eq!(records[4095].step_id.as_u64(), 5000);

    // Strictly ascending chronological order
    for w in records.windows(2) {
        assert_eq!(w[0].step_id.as_u64() + 1, w[1].step_id.as_u64());
    }
}

/// 10. Injected concrete StepGraphProgram resolving against Registry per StepGraphKey.
#[test]
fn test_adv_10_injected_step_graph_program_resolution() {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);
    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);

    // Explicit argument templates and workspace slots are required at construction:
    // no generic blob is installed silently (Spec 4 §7).
    fn sample_op() -> StepProgramOp {
        StepProgramOp::new(
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
            vec![0u8; 64],
            Some(0),
        )
    }

    fn norm_op() -> StepProgramOp {
        StepProgramOp::new(
            OpId::Norm,
            "norm",
            |key| {
                Some(OpStatic::Elementwise(ElementwiseStatic {
                    t_bucket: key.t_dec,
                    dims: vec![1024],
                    dtypes: vec![DType::F16],
                    fused_with: None,
                }))
            },
            vec![0u8; 64],
            Some(0),
        )
    }

    let mut prog_a = StepGraphProgram::new();
    prog_a.add_op(norm_op());
    prog_a.add_op(sample_op());

    let mut prog_b = StepGraphProgram::new();
    prog_b.add_op(norm_op());
    prog_b.add_op(StepProgramOp::new(
        OpId::Matmul,
        "matmul",
        |key| {
            Some(OpStatic::Matmul(MatmulStatic {
                m_bucket: key.t_dec,
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
        vec![0u8; 64],
        Some(0),
    ));
    prog_b.add_op(sample_op());

    let key = StepGraphKey {
        plan_id: PlanId::new(1),
        rank: 0,
        s: 1,
        t_dec: 1,
        t_pre: 0,
        segment: 0,
    };

    let cap_a = StepGraphBuilder::capture(&registry, &arch, key, &prog_a, &arena).expect("cap a");
    let cap_b = StepGraphBuilder::capture(&registry, &arch, key, &prog_b, &arena).expect("cap b");

    assert_eq!(cap_a.launches.len(), 2);
    assert_eq!(cap_b.launches.len(), 3);

    // Empty programs are rejected.
    let empty = StepGraphProgram::new();
    let err_empty = StepGraphBuilder::capture(&registry, &arch, key, &empty, &arena).unwrap_err();
    match err_empty {
        SchedError::GraphCaptureFailed { reason, .. } => {
            assert!(reason.contains("empty"), "got: {reason}");
        }
        other => panic!("expected GraphCaptureFailed, got {other:?}"),
    }

    // Programs without a sampling op are rejected.
    let mut no_sample = StepGraphProgram::new();
    no_sample.add_op(norm_op());
    let err_nosample =
        StepGraphBuilder::capture(&registry, &arch, key, &no_sample, &arena).unwrap_err();
    match err_nosample {
        SchedError::GraphCaptureFailed { reason, .. } => {
            assert!(reason.contains("Sample"), "got: {reason}");
        }
        other => panic!("expected GraphCaptureFailed, got {other:?}"),
    }

    // Workspace binding writes only the declared 8-byte slot; every other args
    // byte keeps its sentinel value.
    let mut sentinel_prog = StepGraphProgram::new();
    sentinel_prog.add_op(
        StepProgramOp::new(
            OpId::Norm,
            "norm",
            |key| {
                Some(OpStatic::Elementwise(ElementwiseStatic {
                    t_bucket: key.t_dec,
                    dims: vec![1024],
                    dtypes: vec![DType::F16],
                    fused_with: None,
                }))
            },
            vec![0u8; 64],
            Some(0),
        )
        .with_args_template(vec![0xAB; 64], Some(16)),
    );
    sentinel_prog.add_op(sample_op());
    let cap_sentinel = StepGraphBuilder::capture(&registry, &arch, key, &sentinel_prog, &arena)
        .expect("sentinel capture");
    let first_blob = &cap_sentinel
        .launches
        .entries()
        .first()
        .expect("launch")
        .args_blob;
    assert_eq!(first_blob.len(), 64);
    assert!(
        first_blob[..16].iter().all(|&b| b == 0xAB),
        "prefix sentinel bytes must survive workspace binding"
    );
    assert!(
        first_blob[24..].iter().all(|&b| b == 0xAB),
        "suffix sentinel bytes must survive workspace binding"
    );
    let bound = u64::from_le_bytes(first_blob[16..24].try_into().expect("workspace slot bytes"));
    assert_eq!(bound, cap_sentinel.workspace_offsets[0]);
    assert_eq!(bound % 256, 0);
}

/// 11. Configuration validation rejects max_seqs != 1 and k_max != 0.
#[test]
fn test_adv_11_config_validation_rejects_invalid_s_and_k() {
    let mut config = SchedulerConfig {
        step_budget_ms: StepBudgetConfig::Auto,
        profile: ProfileMode::Latency,
        prefill_min_chunk: 128,
        prefill_max_chunk: 2048,
        max_wait_ms: 500,
        max_seqs: 2, // INVALID for A3.9
        k_max: 0,
        min_accept: 0.3,
        graph_mode: GraphMode::List,
        plan_id: PlanId::new(1),
        rank: 0,
        vocab_size: 128,
        max_outstanding: DEFAULT_MAX_OUTSTANDING,
    };
    let err_s = config.validate().unwrap_err();
    match err_s {
        SchedError::InvalidRequest { problems } => {
            assert!(problems.iter().any(|p| p.contains("max_seqs must be 1")));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }

    config.max_seqs = 1;
    config.k_max = 2; // INVALID for A3.9
    let err_k = config.validate().unwrap_err();
    match err_k {
        SchedError::InvalidRequest { problems } => {
            assert!(problems.iter().any(|p| p.contains("k_max must be 0")));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }

    // Non-finite or non-positive manual budgets are rejected.
    config.k_max = 0;
    for bad in [f32::NAN, f32::INFINITY, 0.0, -10.0] {
        config.step_budget_ms = StepBudgetConfig::Manual(bad);
        let err = config.validate().unwrap_err();
        match err {
            SchedError::InvalidRequest { problems } => {
                assert!(
                    problems.iter().any(|p| p.contains("step_budget_ms")),
                    "budget {bad} must be refused, got {problems:?}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    // Non-finite or out-of-range min_accept is rejected.
    config.step_budget_ms = StepBudgetConfig::Auto;
    for bad in [f32::NAN, -0.5, 1.5] {
        config.min_accept = bad;
        let err = config.validate().unwrap_err();
        match err {
            SchedError::InvalidRequest { problems } => {
                assert!(
                    problems.iter().any(|p| p.contains("min_accept")),
                    "min_accept {bad} must be refused, got {problems:?}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    // Non-positive admission bounds are rejected.
    config.step_budget_ms = StepBudgetConfig::Auto;
    config.min_accept = 0.3;
    config.max_outstanding = 0; // INVALID: backpressure bound must be positive
    let err_cap = config.validate().unwrap_err();
    match err_cap {
        SchedError::InvalidRequest { problems } => {
            assert!(problems.iter().any(|p| p.contains("max_outstanding")));
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
    config.max_outstanding = DEFAULT_MAX_OUTSTANDING;
    assert!(config.validate().is_ok());

    // Invalid cost-table values fail closed with typed errors, never clamped zeros.
    let mut bad_costs = CostTableStub::new(f32::NAN, f32::INFINITY, -3.0);
    bad_costs.set_bucket_cost(1, 1, 0, f32::NAN);
    for (s, t_dec, t_pre) in [(1, 1, 0), (1, 4, 128)] {
        let err = bad_costs.cost_ms(s, t_dec, t_pre).unwrap_err();
        match err {
            SchedError::InvalidCost { .. } => {}
            other => panic!("expected InvalidCost, got {other:?}"),
        }
    }
    let err_budget = bad_costs
        .resolve_budget_ms(StepBudgetConfig::Auto, ProfileMode::Latency)
        .unwrap_err();
    match err_budget {
        SchedError::InvalidCost { .. } => {}
        other => panic!("expected InvalidCost, got {other:?}"),
    }
    bad_costs.set_bucket_cost(1, 1, 0, -5.0);
    let err_neg = bad_costs.cost_ms(1, 1, 0).unwrap_err();
    match err_neg {
        SchedError::InvalidCost { context, value } => {
            assert_eq!(context, "cost_table");
            assert_eq!(value, -5.0);
        }
        other => panic!("expected InvalidCost, got {other:?}"),
    }
}

/// 12. Scheduler-local zero-allocation isolation: zero heap allocations on the hot
///     decode step path measured in nonempty mid-run scheduler state — a long (>256
///     step) decode run with real stop-string tracking through the default byte
///     detokenizer and device readback through the StepExecutor.
///
///     This is scheduler-local isolation, NOT a claim about the whole real-state
///     loop: layer specs stay empty here to isolate scheduler allocations from the
///     state manager's per-call Vec/BTreeSet locals in reserve/commit (r9v-state,
///     owned by the separate state hot-path card, which will add the integrated
///     zero-allocation proof). "Nonempty state" here is the scheduler's mid-run
///     state — live context, 150+ generated tokens, and populated tail/span
///     buffers — never a cold empty scheduler. The companion
///     `test_adv_12b_long_decode_nonempty_paged_state` below runs the same long
///     decode against nonempty paged KV state functionally.
#[test]
fn test_adv_12_scheduler_local_zero_alloc_isolation() {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);
    let cost_table = Arc::new(CostTableStub::default());
    let state_config = StateConfig {
        max_ctx: 4096,
        max_seqs: 1,
    };
    let state_manager =
        StateManager::new(state_config, vec![], 64 * 1024 * 1024).expect("state manager init");
    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);
    let scheduler_config = SchedulerConfig {
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
        // Byte-domain stub tokens so the default ByteDetokenizer decodes exactly.
        vocab_size: 128,
        max_outstanding: DEFAULT_MAX_OUTSTANDING,
    };
    let program = create_test_program();
    let mut scheduler = Scheduler::new(
        scheduler_config,
        state_manager,
        registry,
        arch,
        cost_table,
        program,
        arena,
    )
    .expect("scheduler init");
    let mut exec = SilentStub::new();

    // Real stop tracking: a stop string is configured and evaluated against the
    // incrementally detokenized tail on every decode step (it never matches this
    // ASCII stream, so no finish allocation pollutes the measured region).
    let stop = StopCriteria::new(vec![], vec!["QUACKSTOP-never-matches".to_owned()]);
    let req = Request::new(
        ReqId::new(999),
        vec![1; 128],
        make_sampling_params(),
        600,
        stop,
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    // Step 1: prefill 128 tokens; KV context is live from here on.
    scheduler.step(&mut exec).expect("step 1").expect("res 1");
    assert_eq!(
        scheduler.state_manager().ctx_len(seq_id).expect("ctx"),
        128,
        "sequence state must be nonempty after prefill"
    );

    // Warmup: 150 decode steps settle the graph cache, tail window, and span
    // buffer past every transient reallocation; state is nonempty mid-run.
    for i in 0..150 {
        let res = scheduler
            .step(&mut exec)
            .unwrap_or_else(|e| panic!("warmup step {i}: {e}"))
            .expect("warmup result");
        assert_eq!(res.record.t_dec, 1);
        assert_eq!(res.accepted_tokens.len(), 1);
    }

    // Measured region: 260 consecutive hot decode steps, each allocation-free.
    for i in 0..260 {
        start_alloc_counting();
        let res = scheduler.step(&mut exec).expect("hot step");
        let allocs = stop_alloc_counting();
        assert_eq!(
            allocs, 0,
            "hot decode step {i} must perform 0 heap allocations after warmup, but had {allocs}"
        );
        let res = res.expect("hot result");
        assert_eq!(res.record.t_dec, 1);
        assert_eq!(res.accepted_tokens.len(), 1);
        assert_eq!(res.record.paused.len(), 0);
    }
}

/// Companion to adv_12: the same long decode with real stop tracking and device
/// readback runs functionally against nonempty paged KV state (Spec 6 §3, §7).
#[test]
fn test_adv_12b_long_decode_nonempty_paged_state() {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);
    let cost_table = Arc::new(CostTableStub::default());
    let state_config = StateConfig {
        max_ctx: 4096,
        max_seqs: 1,
    };
    let layer_specs = vec![StateSpec::KvPaged {
        hkv: 8,
        d: 128,
        dv: 128,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }];
    let state_manager =
        StateManager::new(state_config, layer_specs, 64 * 1024 * 1024).expect("state manager init");
    let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);
    let scheduler_config = SchedulerConfig {
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
    let program = create_test_program();
    let mut scheduler = Scheduler::new(
        scheduler_config,
        state_manager,
        registry,
        arch,
        cost_table,
        program,
        arena,
    )
    .expect("scheduler init");
    let mut exec = SilentStub::new();

    let stop = StopCriteria::new(vec![], vec!["QUACKSTOP-never-matches".to_owned()]);
    let req = Request::new(
        ReqId::new(1001),
        vec![1; 128],
        make_sampling_params(),
        600,
        stop,
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    scheduler.step(&mut exec).expect("prefill").expect("res");
    assert_eq!(
        scheduler.state_manager().ctx_len(seq_id).expect("ctx"),
        128,
        "KV state must be nonempty after prefill"
    );

    // 300 decode steps with live paged blocks, tracked stop tail, and device
    // readback on every step; nothing finishes early and every step emits one token.
    let mut decoded = 0u32;
    for i in 0..300 {
        let res = scheduler
            .step(&mut exec)
            .unwrap_or_else(|e| panic!("decode step {i}: {e}"))
            .expect("decode result");
        assert_eq!(res.record.t_dec, 1);
        assert_eq!(res.accepted_tokens.len(), 1);
        assert_eq!(res.finished_sequences.len(), 0);
        decoded += 1;
    }
    assert_eq!(decoded, 300);
    assert!(
        scheduler.state_manager().ctx_len(seq_id).expect("ctx") > 128,
        "KV context must keep growing across the long decode"
    );
}

/// 13. Lifecycle cancellation and Card A6.1 proposer boundary.
#[test]
fn test_adv_13_lifecycle_cancellation_and_card_boundary() {
    let (mut scheduler, stub_device) = default_test_setup();
    let mut exec = ScriptExecutor::new(&stub_device, ScriptExecutor::script_token(42));

    // Test cancellation during decode phase cleans up state manager
    let req = Request::new(
        ReqId::new(1301),
        vec![1; 128],
        make_sampling_params(),
        50,
        StopCriteria::default(),
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    // Step 1: Transitions to Decoding
    scheduler.step(&mut exec).expect("step 1");

    // Cancel active sequence
    let cancelled = scheduler.cancel_sequence(seq_id).expect("cancel active");
    assert!(cancelled);
    assert!(scheduler.is_idle());

    // State manager has released sequence
    assert!(scheduler.state_manager().ctx_len(seq_id).is_err());

    // Idempotent cancellation
    let double_cancel = scheduler.cancel_sequence(seq_id).expect("double cancel");
    assert!(!double_cancel);

    // Finished result is taken exactly once: take returns ownership, then it is gone.
    let taken = scheduler
        .take_finished_result(seq_id)
        .expect("finished result present");
    assert_eq!(taken.2, FinishReason::Cancelled);
    assert!(scheduler.get_finished_result(seq_id).is_none());
    assert!(scheduler.take_finished_result(seq_id).is_none());
}

/// StepExecutor contract: device phase runs upload -> replay -> readback in order,
/// and post-step consumes exactly the read-back device token (never a host value).
#[test]
fn test_step_executor_call_order_and_device_output() {
    struct OrderExecutor {
        calls: Vec<&'static str>,
        replayed_launches: usize,
        next_token: u32,
        last_step: u64,
        last_seq: u64,
        last_prompt_len: usize,
        last_prefill: Option<(u32, u32)>,
    }
    impl StepExecutor for OrderExecutor {
        fn upload_inputs(&mut self, input: &StepInputs<'_>) -> SchedResult<()> {
            self.calls.push("upload");
            self.last_step = input.step_id.as_u64();
            self.last_seq = input.seq_id.as_u64();
            self.last_prompt_len = input.prompt_tokens.len();
            self.last_prefill = input.prefill;
            Ok(())
        }
        fn replay_graph(&mut self, graph: &CapturedGraph) -> SchedResult<()> {
            self.calls.push("replay");
            self.replayed_launches = graph.launches.len();
            Ok(())
        }
        fn readback_sample(&mut self) -> SchedResult<DeviceStepSample> {
            self.calls.push("readback");
            Ok(DeviceStepSample {
                step_id: StepId::new(self.last_step),
                token: self.next_token,
                accept_len: 1,
            })
        }
    }

    let (mut scheduler, _stub) = default_test_setup();
    let req = Request::new(
        ReqId::new(1401),
        vec![1; 128],
        make_sampling_params(),
        10,
        StopCriteria::default(),
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    let mut exec = OrderExecutor {
        calls: Vec::new(),
        replayed_launches: 0,
        next_token: 65,
        last_step: 0,
        last_seq: 0,
        last_prompt_len: 0,
        last_prefill: None,
    };
    let res1 = scheduler.step(&mut exec).expect("step 1").expect("res 1");
    assert_eq!(exec.calls, vec!["upload", "replay", "readback"]);
    assert_eq!(exec.last_step, 1);
    assert_eq!(exec.last_seq, seq_id.as_u64());
    // Upload carries the actual batch facts: full prompt IDs and prefill progress.
    assert_eq!(exec.last_prompt_len, 128);
    assert_eq!(exec.last_prefill, Some((0, 128)));
    assert!(
        exec.replayed_launches >= 1,
        "device must replay the captured graph launches"
    );
    assert_eq!(
        res1.accepted_tokens
            .get(0)
            .and_then(|(_, toks)| toks.get(0).copied()),
        Some(65),
        "post-step must consume the read-back device token"
    );

    // A new device value on the next step replaces the old one: nothing cached.
    exec.calls.clear();
    exec.next_token = 66;
    let res2 = scheduler.step(&mut exec).expect("step 2").expect("res 2");
    assert_eq!(exec.calls, vec!["upload", "replay", "readback"]);
    assert_eq!(
        res2.accepted_tokens
            .get(0)
            .and_then(|(_, toks)| toks.get(0).copied()),
        Some(66)
    );
}

/// Final max-token precedence: a stop-string match completing exactly on the last
/// permitted token reports MaxTokens with no trim (Spec 6 §7).
#[test]
fn test_stop_trim_final_max_token_precedence() {
    struct TwoTokenDetok;
    impl Detokenizer for TwoTokenDetok {
        fn append_token(
            &mut self,
            _seq_id: SeqId,
            token: u32,
            output: &mut String,
        ) -> Result<usize, SchedError> {
            let text = match token {
                10 => "B",
                11 => "C",
                _ => "x",
            };
            output.push_str(text);
            Ok(text.len())
        }
    }

    let (mut scheduler, stub_device) = default_test_setup();
    scheduler.set_detokenizer(Box::new(TwoTokenDetok));
    // Stub readback emits token 10 ("B") then 11 ("C").
    let mut exec = ScriptExecutor::new(&stub_device, vec![(10, 1), (11, 1)]);

    let stop = StopCriteria::new(vec![], vec!["BC".to_owned()]);
    let req = Request::new(
        ReqId::new(1501),
        vec![1; 128],
        make_sampling_params(),
        2,
        stop,
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");

    // Step 1: prefill completes, emits token 10 ("B").
    let res1 = scheduler.step(&mut exec).expect("step 1").expect("res 1");
    assert_eq!(res1.finished_sequences.len(), 0);

    // Step 2: emits token 11; tail "BC" matches the stop string exactly at the
    // max_tokens bound -> MaxTokens wins, no trim.
    let res2 = scheduler.step(&mut exec).expect("step 2").expect("res 2");
    assert_eq!(res2.finished_sequences.len(), 1);
    assert_eq!(
        res2.finished_sequences.get(0),
        Some(&(seq_id, FinishReason::MaxTokens))
    );
    assert_eq!(
        res2.accepted_tokens
            .get(0)
            .and_then(|(_, toks)| toks.get(0).copied()),
        Some(11),
        "final token must be accepted, not trimmed"
    );
    let finished = scheduler
        .get_finished_result(seq_id)
        .expect("finished present");
    assert_eq!(finished.1, vec![10, 11], "no tokens trimmed at the bound");
}

/// ByteDetokenizer: exact incremental multi-byte decoding with no UTF-8
/// corruption, incremental == batch equivalence, and rejection of invalid input.
#[test]
fn test_byte_detokenizer_incremental_utf8_exact() {
    let seq = SeqId::new(7);

    // U+20AC (€) = E2 82 AC split across three tokens: 0, 0, then 3 bytes.
    let mut d = ByteDetokenizer::new();
    let mut s = String::new();
    assert_eq!(d.append_token(seq, 0xE2, &mut s).expect("byte 1"), 0);
    assert_eq!(s, "");
    assert_eq!(d.append_token(seq, 0x82, &mut s).expect("byte 2"), 0);
    assert_eq!(s, "");
    assert_eq!(d.append_token(seq, 0xAC, &mut s).expect("byte 3"), 3);
    assert_eq!(s, "\u{20AC}");

    // Incremental decoding equals batch decoding byte-for-byte.
    let mut batch_detok = ByteDetokenizer::new();
    let mut batch = String::new();
    let n = batch_detok
        .detokenize_to(seq, &[0x41, 0xE2, 0x82, 0xAC], &mut batch)
        .expect("batch");
    assert_eq!(n, 4);
    assert_eq!(batch, "A\u{20AC}");
    let whole = batch_detok
        .detokenize(seq, &[0x41, 0xE2, 0x82, 0xAC])
        .expect("whole");
    assert_eq!(whole, "A\u{20AC}");

    // Invalid inputs are rejected, never corrupted or silently replaced.
    let mut bad = ByteDetokenizer::new();
    assert!(
        bad.append_token(seq, 0x80, &mut String::new()).is_err(),
        "lone continuation byte must be rejected"
    );
    assert!(
        bad.append_token(seq, 256, &mut String::new()).is_err(),
        "token id above the byte range must be rejected"
    );
    let mut bad2 = ByteDetokenizer::new();
    let mut tmp = String::new();
    assert_eq!(bad2.append_token(seq, 0xE2, &mut tmp).expect("start"), 0);
    assert!(
        bad2.append_token(seq, 0x41, &mut tmp).is_err(),
        "non-continuation byte mid-sequence must be rejected"
    );

    // Reset clears pending multi-byte state for the sequence.
    let mut r = ByteDetokenizer::new();
    let mut tmp2 = String::new();
    assert_eq!(r.append_token(seq, 0xE2, &mut tmp2).expect("start"), 0);
    r.reset(seq);
    assert_eq!(
        r.append_token(seq, 0x41, &mut tmp2).expect("after reset"),
        1
    );
    assert_eq!(tmp2, "A");
}

/// k=0 contract: every executed sampling step must report
/// `DeviceStepSample.accept_len == 1`. Values 0 and 99 are rejected with a typed
/// execution error before commit, log, or EMA mutation (Spec 6 §3.3, §9).
#[test]
fn test_accept_len_non_one_rejected_before_mutation() {
    for bad_accept in [0u32, 99u32] {
        let (mut scheduler, stub_device) = default_test_setup();
        // First readback is valid so prefill completes; the decode readback is bad.
        let mut exec = ScriptExecutor::new(&stub_device, vec![(42, 1), (43, bad_accept)]);
        let req = Request::new(
            ReqId::new(1601),
            vec![1; 128],
            make_sampling_params(),
            10,
            StopCriteria::default(),
            false,
        )
        .expect("valid req");
        let seq_id = scheduler.enqueue_request(req).expect("enqueue");
        let res1 = scheduler.step(&mut exec).expect("prefill").expect("res");
        assert_eq!(res1.accepted_tokens.len(), 1);

        let ctx_before = scheduler.state_manager().ctx_len(seq_id).expect("ctx");
        let log_before = scheduler.schedule_log().total_written();
        let err = scheduler.step(&mut exec).unwrap_err();
        match err {
            SchedError::ExecutionFailed { detail } => {
                assert!(
                    detail.contains("accept_len"),
                    "error must name the violated contract, got: {detail}"
                );
            }
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
        // Rejected before mutation: context and log untouched, nothing finished.
        assert_eq!(
            scheduler.state_manager().ctx_len(seq_id).expect("ctx"),
            ctx_before,
            "bad accept_len {bad_accept} must not commit"
        );
        assert_eq!(
            scheduler.schedule_log().total_written(),
            log_before,
            "bad accept_len {bad_accept} must not log"
        );
        assert!(scheduler.get_finished_result(seq_id).is_none());
    }
}

/// Explicit backpressure: enqueue is rejected once finished + queued + active +
/// paused reaches `max_outstanding`; taking a finished result reopens capacity.
/// Nothing is evicted or lost (Spec 6 §2).
#[test]
fn test_backpressure_exact_capacity_and_recovery() {
    let (mut scheduler, _stub_device) = setup_with_outstanding(2);

    let enq = |id: u64| {
        Request::new(
            ReqId::new(id),
            vec![1; 8],
            make_sampling_params(),
            5,
            StopCriteria::default(),
            false,
        )
        .expect("valid req")
    };
    let req_a = enq(1701);
    let req_b = enq(1702);

    let seq_a = scheduler.enqueue_request(req_a).expect("enqueue A");
    assert_eq!(scheduler.outstanding_count(), 1);
    let _seq_b = scheduler.enqueue_request(req_b).expect("enqueue B");
    assert_eq!(scheduler.outstanding_count(), 2);

    // At exact capacity the next enqueue is rejected with the numbers attached.
    let req_c = enq(1703);
    let err = scheduler.enqueue_request(req_c).unwrap_err();
    match err {
        SchedError::CapacityExceeded {
            outstanding,
            maximum,
        } => {
            assert_eq!(outstanding, 2);
            assert_eq!(maximum, 2);
        }
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }
    assert_eq!(scheduler.outstanding_count(), 2);

    // Cancelling A records its finished result without freeing admission: still full.
    assert!(scheduler.cancel_sequence(seq_a).expect("cancel A"));
    assert!(scheduler.get_finished_result(seq_a).is_some());
    assert_eq!(scheduler.outstanding_count(), 2);
    let req_c2 = enq(1703);
    assert!(
        scheduler.enqueue_request(req_c2).is_err(),
        "finished retained counts against capacity"
    );

    // Taking the finished result reopens exactly one slot; nothing was lost.
    let taken = scheduler.take_finished_result(seq_a).expect("taken result");
    assert_eq!(taken.2, FinishReason::Cancelled);
    assert_eq!(scheduler.outstanding_count(), 1);
    let req_c3 = enq(1703);
    scheduler
        .enqueue_request(req_c3)
        .expect("enqueue C after take");
    assert_eq!(scheduler.outstanding_count(), 2);
}

/// Over-capacity paused arrays are rejected on deserialize, never silently
/// truncated (Spec 6 §9).
#[test]
fn test_serde_paused_seqs_rejects_over_capacity() {
    let (mut scheduler, stub_device) = default_test_setup();
    let mut exec = ScriptExecutor::new(&stub_device, ScriptExecutor::script_token(42));
    let req = Request::new(
        ReqId::new(1801),
        vec![1; 128],
        make_sampling_params(),
        5,
        StopCriteria::default(),
        false,
    )
    .expect("valid req");
    scheduler.enqueue_request(req).expect("enqueue");
    let res = scheduler.step(&mut exec).expect("step").expect("res");

    // A real record round-trips exactly.
    let value = serde_json::to_value(&res.record).expect("serialize record");
    let roundtrip: ScheduleRecord =
        serde_json::from_value(value.clone()).expect("roundtrip record");
    assert_eq!(roundtrip, res.record);

    // One paused ID is the structural maximum and deserializes.
    let mut one = value.clone();
    one["paused"] = serde_json::json!([7]);
    let decoded: ScheduleRecord = serde_json::from_value(one).expect("one paused id");
    assert_eq!(decoded.paused.len(), 1);

    // Two paused IDs are rejected, never truncated to one.
    let mut two = value;
    two["paused"] = serde_json::json!([7, 8]);
    assert!(
        serde_json::from_value::<ScheduleRecord>(two).is_err(),
        "over-capacity paused array must be rejected"
    );
}

/// Multi-byte stop trimming truncates from the first contributing token: stop strings
/// split across 2-, 3-, and 4-byte UTF-8 sequences, each with preceding ASCII, plus a
/// cross-token match spanning ASCII and a code point (Spec 6 §7).
#[test]
fn test_multibyte_stop_trim_starts_at_first_contributing_token() {
    // (stop string, preceding ASCII byte tokens, code point byte tokens)
    let cases: &[(&str, &[u32], &[u32])] = &[
        ("é", &[0x41], &[0xC3, 0xA9]),
        ("€", &[0x41, 0x42], &[0xE2, 0x82, 0xAC]),
        ("𝄞", &[0x41], &[0xF0, 0x9D, 0x84, 0x9E]),
    ];
    for (case_idx, (stop_str, prefix, codepoint)) in cases.iter().enumerate() {
        let stop = StopCriteria::new(vec![], vec![(*stop_str).to_owned()]);
        let req = Request::new(
            ReqId::new(1900 + case_idx as u64),
            vec![1; 8],
            make_sampling_params(),
            50,
            stop,
            false,
        )
        .expect("valid req");
        let mut seq = Sequence::new(req, SeqId::new(600 + case_idx as u64), 0);
        let mut detok = ByteDetokenizer::new();

        for &b in *prefix {
            let (finish, trimmed) = seq.append_generated_token(b, &mut detok).expect("prefix");
            assert!(finish.is_none());
            assert!(!trimmed);
        }
        for &b in &codepoint[..codepoint.len() - 1] {
            let (finish, trimmed) = seq.append_generated_token(b, &mut detok).expect("lead");
            assert!(finish.is_none(), "buffered lead bytes finish nothing");
            assert!(!trimmed);
        }
        let last = codepoint[codepoint.len() - 1];
        let (finish, trimmed) = seq.append_generated_token(last, &mut detok).expect("tail");
        assert_eq!(
            finish,
            Some(FinishReason::StopString((*stop_str).to_owned())),
            "stop {stop_str} must match at the code point"
        );
        assert!(trimmed, "the completing token is trimmed");
        assert_eq!(
            seq.generated,
            prefix.to_vec(),
            "stop {stop_str}: only the ASCII prefix survives; all {n} code point tokens trim",
            n = codepoint.len(),
        );
    }

    // Cross-token match: stop "Aé" spans the ASCII token and the 2-byte code point.
    let stop = StopCriteria::new(vec![], vec!["Aé".to_owned()]);
    let req = Request::new(
        ReqId::new(1910),
        vec![1; 8],
        make_sampling_params(),
        50,
        stop,
        false,
    )
    .expect("valid req");
    let mut seq = Sequence::new(req, SeqId::new(610), 0);
    let mut detok = ByteDetokenizer::new();
    for &b in &[0x41u32, 0xC3, 0xA9] {
        let (finish, _) = seq.append_generated_token(b, &mut detok).expect("token");
        if b != 0xA9 {
            assert!(finish.is_none());
        }
    }
    assert!(
        seq.generated.is_empty(),
        "cross-token match trims everything"
    );
}

/// Uniform finish precedence: an EOS token filling the final budget slot reports
/// MaxTokens with no trim, exactly like a simultaneous stop-string match; below the
/// budget EOS still wins over stop strings (Spec 6 §7).
#[test]
fn test_finish_precedence_final_budget_eos_reports_max_tokens() {
    // Final slot: EOS token 90 with max_tokens 2 reports MaxTokens, token kept.
    let (mut scheduler, stub_device) = default_test_setup();
    let mut exec = ScriptExecutor::new(&stub_device, vec![(65, 1), (90, 1)]);
    let stop = StopCriteria::new(vec![90], vec![]);
    let req = Request::new(
        ReqId::new(2001),
        vec![1; 128],
        make_sampling_params(),
        2,
        stop,
        false,
    )
    .expect("valid req");
    let seq_id = scheduler.enqueue_request(req).expect("enqueue");
    let res1 = scheduler.step(&mut exec).expect("step 1").expect("res 1");
    assert_eq!(res1.finished_sequences.len(), 0);
    let res2 = scheduler.step(&mut exec).expect("step 2").expect("res 2");
    assert_eq!(
        res2.finished_sequences.get(0),
        Some(&(seq_id, FinishReason::MaxTokens)),
        "final-budget EOS must report MaxTokens, not Eos"
    );
    assert_eq!(
        res2.accepted_tokens
            .get(0)
            .and_then(|(_, toks)| toks.get(0).copied()),
        Some(90),
        "final token must be accepted, not trimmed"
    );
    let finished = scheduler
        .get_finished_result(seq_id)
        .expect("finished present");
    assert_eq!(finished.1, vec![65, 90]);

    // Below the budget the same EOS token still reports Eos.
    let (mut scheduler2, stub_device2) = default_test_setup();
    let mut exec2 = ScriptExecutor::new(&stub_device2, vec![(65, 1), (90, 1)]);
    let stop2 = StopCriteria::new(vec![90], vec![]);
    let req2 = Request::new(
        ReqId::new(2002),
        vec![1; 128],
        make_sampling_params(),
        5,
        stop2,
        false,
    )
    .expect("valid req");
    let seq_id2 = scheduler2.enqueue_request(req2).expect("enqueue");
    scheduler2.step(&mut exec2).expect("step 1").expect("res 1");
    let res_b2 = scheduler2.step(&mut exec2).expect("step 2").expect("res 2");
    assert_eq!(
        res_b2.finished_sequences.get(0),
        Some(&(seq_id2, FinishReason::Eos(90))),
        "below-budget EOS must still report Eos"
    );
}

/// Scheduler consumption boundaries fail closed on invalid cost tables: a NaN or
/// negative base cost rejects the step with a typed error, never a clamped budget
/// with silent over- or under-admission (Spec 6 §4.1).
#[test]
fn test_step_rejects_invalid_cost_table_values() {
    for bad_base in [f32::NAN, f32::INFINITY, -1.0] {
        let arch = ArchName::from("gfx1201");
        let registry = create_test_registry(&arch);
        let cost_table = Arc::new(CostTableStub::new(bad_base, 0.05, 0.005));
        let state_config = StateConfig {
            max_ctx: 4096,
            max_seqs: 16,
        };
        let layer_specs = vec![StateSpec::KvPaged {
            hkv: 8,
            d: 128,
            dv: 128,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        }];
        let state_manager = StateManager::new(state_config, layer_specs, 64 * 1024 * 1024)
            .expect("state manager init");
        let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);
        let scheduler_config = SchedulerConfig {
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
        let mut scheduler = Scheduler::new(
            scheduler_config,
            state_manager,
            registry,
            arch,
            cost_table,
            create_test_program(),
            arena,
        )
        .expect("scheduler init");
        let req = Request::new(
            ReqId::new(2101),
            vec![1; 128],
            make_sampling_params(),
            5,
            StopCriteria::default(),
            false,
        )
        .expect("valid req");
        scheduler.enqueue_request(req).expect("enqueue");
        let stub_device = StubDevice::new();
        let mut exec = ScriptExecutor::new(&stub_device, ScriptExecutor::script_token(42));
        let err = scheduler.step(&mut exec).unwrap_err();
        match err {
            SchedError::InvalidCost { .. } => {}
            other => panic!("base cost {bad_base} must fail closed, got {other:?}"),
        }
    }
}
