// SPDX-License-Identifier: Apache-2.0
//! Hosted stub-device and fake-cost-table simulation running 1000 steps twice (Spec 6, Card A3.9).
//!
//! Validates bit-identical deterministic reproducibility across two 1000-step runs:
//! - Exact same token outputs per sequence
//! - Exact same schedule log records
//! - Exact same kernel launches replayed on StubDevice

use std::collections::BTreeMap;
use std::sync::Arc;

use r9v_common::{ReqId, SeqId};
use r9v_ir::{AttentionMask, DType, Epilogue, LayoutId, PlanId, QuantScheme, SamplingParams};
use r9v_registry::{
    ArchName, AttentionStatic, BundleManifest, ElementwiseStatic, LaunchGeometry, LaunchRecord,
    ManifestVariantEntry, MatmulStatic, OpId, OpStatic, Registry, RegistryConfig, SamplingMethod,
    SamplingStatic, StubDevice, Tier, VariantHash,
};
use r9v_sched::{
    CapturedGraph, CostTableStub, DeviceStepSample, GraphMode, ProfileMode, Request, SchedResult,
    ScheduleRecord, Scheduler, SchedulerConfig, StepBudgetConfig, StepExecutor, StepGraphProgram,
    StepInputs, StepProgramOp, StopCriteria, WorkspaceArena, DEFAULT_MAX_OUTSTANDING,
};
use r9v_state::{CacheDtype, Retain, StateConfig, StateManager, StateSpec};

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

fn sim_args_template() -> Vec<u8> {
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
        sim_args_template(),
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
        sim_args_template(),
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
        sim_args_template(),
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
        sim_args_template(),
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
        sim_args_template(),
        Some(0),
    ));
    program
}

// Deterministic stub-tier token model: bit-identical LCG over (sequence, prompt,
// readback index). Lives in the simulation only; production has no host sampler.
fn sim_token(seq_id: u64, prompt_len: u32, read_idx: u32, vocab: u32) -> u32 {
    let v = vocab.max(1) as u64;
    let hash = seq_id
        .wrapping_mul(6364136223846793005)
        .wrapping_add((prompt_len as u64).wrapping_mul(1442695040888963407))
        .wrapping_add((read_idx as u64).wrapping_mul(2862933555777941757))
        .wrapping_add(0x9E3779B97F4A7C15);
    let token = (hash % v) as u32;
    if token == 0 {
        1
    } else {
        token
    }
}

/// Deterministic stub [`StepExecutor`] for the 1000-step simulation (Spec 6 §3.2).
///
/// Upload records the actual typed batch facts; replay dispatches the captured graph
/// through the stub device so kernel launches are recorded; readback returns the
/// deterministic stub-tier token with `accept_len == 1`. No production sampler path.
struct SimStepExecutor<'d> {
    device: &'d StubDevice,
    vocab: u32,
    read_count: u32,
    last_seq: u64,
    last_prompt_len: u32,
}

impl<'d> SimStepExecutor<'d> {
    fn new(device: &'d StubDevice, vocab: u32) -> Self {
        Self {
            device,
            vocab,
            read_count: 0,
            last_seq: 0,
            last_prompt_len: 0,
        }
    }
}

impl StepExecutor for SimStepExecutor<'_> {
    fn upload_inputs(&mut self, input: &StepInputs<'_>) -> SchedResult<()> {
        self.last_seq = input.seq_id.as_u64();
        self.last_prompt_len = input.prompt_tokens.len() as u32;
        Ok(())
    }
    fn replay_graph(&mut self, graph: &CapturedGraph) -> SchedResult<()> {
        graph.launches.replay(self.device, None)?;
        Ok(())
    }
    fn readback_sample(&mut self) -> SchedResult<DeviceStepSample> {
        self.read_count = self.read_count.wrapping_add(1);
        Ok(DeviceStepSample {
            token: sim_token(
                self.last_seq,
                self.last_prompt_len,
                self.read_count,
                self.vocab,
            ),
            accept_len: 1,
        })
    }
}

fn create_simulation_workload() -> Vec<Request> {
    let mut requests = Vec::new();
    let prompt_patterns = [
        (101, 256, 40),
        (102, 128, 60),
        (103, 384, 50),
        (104, 512, 80),
        (105, 128, 70),
        (106, 256, 90),
        (107, 300, 45),
        (108, 150, 65),
        (109, 400, 85),
        (110, 200, 75),
        (111, 256, 50),
        (112, 128, 60),
        (113, 384, 70),
        (114, 512, 55),
        (115, 180, 80),
        (116, 240, 95),
        (117, 320, 60),
        (118, 160, 50),
        (119, 280, 70),
        (120, 210, 90),
    ];

    for &(id_raw, prompt_len, max_gen) in &prompt_patterns {
        let req_id = ReqId::new(id_raw);
        let tokens: Vec<u32> = (0..prompt_len)
            .map(|i| ((id_raw.wrapping_mul(37) + i as u64 * 13) % 30000 + 100) as u32)
            .collect();
        let sampling = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: vec![],
        };
        let stop = StopCriteria::new(vec![29999], vec!["END".to_owned()]);
        let req =
            Request::new(req_id, tokens, sampling, max_gen, stop, false).expect("valid request");
        requests.push(req);
    }
    requests
}

struct SimulationResult {
    outputs: BTreeMap<u64, Vec<u32>>,
    logs: Vec<ScheduleRecord>,
    launches: Vec<LaunchRecord>,
}

fn run_1000_steps_simulation() -> SimulationResult {
    let arch = ArchName::from("gfx1201");
    let registry = create_test_registry(&arch);
    let cost_table = Arc::new(CostTableStub::default());
    let program = create_test_program();
    let state_config = StateConfig {
        max_ctx: 4096,
        max_seqs: 32,
    };
    let layer_specs = vec![StateSpec::KvPaged {
        hkv: 8,
        d: 128,
        dv: 128,
        cache: CacheDtype::E4M3,
        retain: Retain::All,
    }];
    let state_manager = StateManager::new(state_config, layer_specs, 128 * 1024 * 1024)
        .expect("state manager init");
    let arena = WorkspaceArena::new(0, 64 * 1024 * 1024);

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
        // Byte-domain stub tokens so the default ByteDetokenizer decodes exactly (Spec 6 §7).
        vocab_size: 128,
        max_outstanding: DEFAULT_MAX_OUTSTANDING,
    };

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
    let mut exec = SimStepExecutor::new(&stub_device, 128);
    let mut workload = create_simulation_workload();
    let mut workload_idx = 0;

    // Enqueue initial batch of requests
    for _ in 0..5 {
        if workload_idx < workload.len() {
            let req = workload[workload_idx].clone();
            workload_idx += 1;
            scheduler.enqueue_request(req).expect("enqueue request");
        }
    }

    let mut step_count = 0;
    let target_steps = 1000;

    while step_count < target_steps {
        // Feed more requests if queue is running low to ensure 1000 continuous steps
        if scheduler.active_sequence_count() < 3 {
            if workload_idx >= workload.len() {
                // Re-cycle workload with deterministic offset
                workload = create_simulation_workload();
                workload_idx = 0;
            }
            let mut req = workload[workload_idx].clone();
            workload_idx += 1;
            // Distinct request ID to allow fresh enqueue
            req.id = ReqId::new(req.id.as_u64() + 1000 * (step_count as u64 + 1));
            let _ = scheduler.enqueue_request(req);
        }

        let res = scheduler.step(&mut exec).expect("step execution failed");
        if res.is_some() {
            step_count += 1;
        }
    }

    assert_eq!(step_count, 1000, "must run exactly 1000 steps");

    // Collect all schedule logs (1000 steps)
    let logs = scheduler.schedule_log().to_vec();
    assert_eq!(logs.len(), 1000, "ring must contain exactly 1000 records");
    assert_eq!(
        scheduler.schedule_log().total_written(),
        1000,
        "total written count must be exactly 1000"
    );
    for (i, rec) in logs.iter().enumerate() {
        assert_eq!(
            rec.step_id.as_u64(),
            (i + 1) as u64,
            "step ids must be strictly contiguous and monotonically increasing without gaps"
        );
    }

    // Collect all recorded launches from stub device
    let launches = stub_device.recorded_launches().expect("recorded launches");

    // Collect all generated output tokens
    let mut outputs = BTreeMap::new();
    // Finished sequences
    for i in 1..=2000 {
        let sid = SeqId::new(i);
        if let Some((_, gen, _)) = scheduler.get_finished_result(sid) {
            outputs.insert(i, gen.clone());
        }
    }

    SimulationResult {
        outputs,
        logs,
        launches,
    }
}

#[test]
fn test_simulation_1000_steps_determinism_twice_bit_identical() {
    // Run 1: 1000 steps simulation
    let run1 = run_1000_steps_simulation();

    // Run 2: 1000 steps simulation from fresh identical state
    let run2 = run_1000_steps_simulation();

    // 1. Outputs must be identical
    assert_eq!(
        run1.outputs.len(),
        run2.outputs.len(),
        "number of finished sequences must match"
    );
    assert!(
        !run1.outputs.is_empty(),
        "simulation must produce finished sequences"
    );
    for (seq_id, tokens1) in &run1.outputs {
        let tokens2 = run2
            .outputs
            .get(seq_id)
            .unwrap_or_else(|| panic!("seq_id {seq_id} missing in run 2"));
        assert_eq!(
            tokens1, tokens2,
            "tokens for sequence {seq_id} must be bit-identical"
        );
    }

    // 2. Schedule logs must be bit-identical across all steps
    assert_eq!(
        run1.logs.len(),
        run2.logs.len(),
        "schedule log length must match exactly"
    );
    assert_eq!(
        run1.logs.len(),
        1000,
        "schedule log must contain exactly 1000 records"
    );
    for i in 0..1000 {
        let r1 = &run1.logs[i];
        let r2 = &run2.logs[i];
        assert_eq!(r1, r2, "schedule log record at step {i} must be identical");
    }

    // 3. Stub device kernel launches must be bit-identical
    assert_eq!(
        run1.launches.len(),
        run2.launches.len(),
        "stub device launch count must match"
    );
    assert!(
        !run1.launches.is_empty(),
        "stub device must have recorded kernel launches"
    );
    for (idx, (l1, l2)) in run1.launches.iter().zip(run2.launches.iter()).enumerate() {
        assert_eq!(
            l1, l2,
            "kernel launch record at launch index {idx} must be bit-identical"
        );
    }
}
