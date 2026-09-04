//! Deterministic pre-step -> device -> post-step scheduler execution loop (Spec 6 §1, §3, §5, §7).
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use r9v_common::{ReqId, SeqId, StepId};
use r9v_ir::{bucket_step, PlanId, StepGraphKey};
use r9v_registry::{ArchName, Registry};
use r9v_state::StateManager;

use crate::arena::WorkspaceArena;
use crate::cost::{estimate_pre_step_ms, CostTable, ProfileMode, StepBudgetConfig};
use crate::error::{SchedError, SchedResult};
use crate::graph::{CapturedGraph, GraphCache, StepGraphProgram};
use crate::log::{GraphMode, ScheduleLogRing, ScheduleRecord};
use crate::proposer::NoOpProposer;
use crate::streams::{StepEventChain, StreamKind};
use crate::types::{
    ByteDetokenizer, Detokenizer, FinishReason, InlineVec, Request, Sequence, SequencePhase, Step,
    StepResult,
};

// DECISION(A3.9): there is no host-side token sampler in production. The only token
// source post-step may consume is the DeviceStepSample read back from the device
// phase; rejected synthesizing tokens on the host after a DeviceExecutor call because
// the accepted token must provably come out of the device Sample op in call order.
// Deterministic drivers live in tests/simulation as stub StepExecutors. Spec 6 §3.2.
/// Fixed-size device step output written by readback (Spec 6 §3.2).
///
/// Stack-allocated; the device phase produces exactly one per executed step with
/// no heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceStepSample {
    /// Sampled token ID produced by the device Sample op (Spec 6 §3.2).
    pub token: u32,
    /// Number of accepted tokens reported by the device for this step (Spec 6 §3.3, §9).
    ///
    /// Card A3.9 runs with k=0, so every executed sampling step reports exactly 1;
    /// any other value is rejected with a typed execution error before post-step
    /// mutation.
    pub accept_len: u32,
}

/// Typed per-step device upload payload carrying the actual token IDs, positions,
/// and batch facts for H2D upload on the Copy stream (Spec 6 §3.2, §5.4).
///
/// Borrowed slices only; constructing or passing this performs no heap allocation.
#[derive(Debug, Clone, Copy)]
pub struct StepInputs<'a> {
    /// Step being executed (Spec 6 §2).
    pub step_id: StepId,
    /// Sequence being stepped (Spec 6 §2).
    pub seq_id: SeqId,
    /// Full prompt token IDs: the actual IDs a prefill chunk uploads (Spec 6 §2).
    pub prompt_tokens: &'a [u32],
    /// Generated token IDs so far: the actual IDs positions derive from (Spec 6 §2).
    pub generated_tokens: &'a [u32],
    /// Prefill progress `(done, chunk_len)` when this step admits a prefill chunk;
    /// `None` for a decode step (Spec 6 §3.1).
    pub prefill: Option<(u32, u32)>,
    /// Verified context length before this step (Spec 3 §3.3, Spec 6 §2).
    pub ctx_len: u32,
}

/// Explicit device-side step execution: upload inputs, replay the captured graph,
/// read back the sampled token into fixed output (Spec 6 §3.2, §5.4).
///
/// The device phase runs upload -> replay -> readback in order; post-step consumes
/// only the [`DeviceStepSample`] read back from the executor. Accepted tokens come
/// from that readback alone, never from a host value.
pub trait StepExecutor: Send + Sync {
    /// Uploads H2D step inputs (actual token IDs, positions, batch facts) on the
    /// Copy stream (Spec 6 §5.4).
    fn upload_inputs(&mut self, input: &StepInputs<'_>) -> SchedResult<()>;

    /// Replays the captured step graph launch list on the Compute stream (Spec 4 §7, Spec 6 §5.2).
    fn replay_graph(&mut self, graph: &CapturedGraph) -> SchedResult<()>;

    /// Reads back the device-sampled token into fixed output on the Copy stream
    /// (Spec 6 §3.2, §5.4).
    ///
    /// Takes no host-derived sampling hints: the token returned is the actual
    /// device Sample op output for the uploaded inputs.
    fn readback_sample(&mut self) -> SchedResult<DeviceStepSample>;
}

/// Configuration parameters for the scheduler (Spec 6 §10).
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerConfig {
    /// Step execution budget setting (Spec 6 §10).
    pub step_budget_ms: StepBudgetConfig,
    /// Optimization profile mode: latency or throughput (Spec 5 §8, Spec 6 §4.3).
    pub profile: ProfileMode,
    /// Minimum prefill chunk size in tokens (Spec 6 §10).
    pub prefill_min_chunk: u32,
    /// Maximum prefill chunk size in tokens (Spec 6 §10).
    pub prefill_max_chunk: u32,
    /// Maximum waiting duration before forced prefill admission in milliseconds (Spec 6 §4.1, §10).
    pub max_wait_ms: u64,
    /// Maximum active sequences admitted simultaneously (Spec 3 §9, Spec 6 §2; for A3.9 S=1).
    pub max_seqs: u32,
    /// Maximum speculative draft length k (Spec 6 §10; for A3.9 k=0).
    pub k_max: u32,
    /// Minimum acceptance threshold turning off speculative drafting (Spec 6 §4.2, §10).
    pub min_accept: f32,
    /// Graph replay mechanism (Spec 6 §10).
    pub graph_mode: GraphMode,
    /// Execution plan identifier (Spec 1 §3.1).
    pub plan_id: PlanId,
    /// Device rank executing this scheduler instance (Spec 6 §5.1).
    pub rank: u32,
    /// Vocabulary size for sampling boundaries (Spec 1 §2.4).
    pub vocab_size: u32,
    /// Maximum total outstanding requests (finished retained + queued + active +
    /// paused) before admission backpressure rejects `enqueue_request` (Spec 6 §2).
    ///
    /// Bounds retained finished results and queue state without silent eviction:
    /// taking a finished result reopens capacity and nothing is lost. Must be positive.
    pub max_outstanding: u32,
}

/// Default bound on total outstanding requests for admission backpressure (Spec 6 §2).
pub const DEFAULT_MAX_OUTSTANDING: u32 = 1024;

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            step_budget_ms: StepBudgetConfig::Auto,
            profile: ProfileMode::Latency,
            prefill_min_chunk: 128,
            prefill_max_chunk: 2048,
            max_wait_ms: 500,
            max_seqs: 1,
            k_max: 0,
            min_accept: 0.3,
            graph_mode: GraphMode::Auto,
            plan_id: PlanId::new(1),
            rank: 0,
            vocab_size: 32000,
            max_outstanding: DEFAULT_MAX_OUTSTANDING,
        }
    }
}

impl SchedulerConfig {
    /// Validates scheduler configuration parameters, accumulating all problems (CONVENTIONS.md §1.4).
    pub fn validate(&self) -> SchedResult<()> {
        let mut problems = Vec::new();
        if self.prefill_min_chunk == 0 {
            problems.push("prefill_min_chunk must be >= 1".to_owned());
        }
        if self.prefill_max_chunk < self.prefill_min_chunk {
            problems.push(format!(
                "prefill_max_chunk ({}) must be >= prefill_min_chunk ({})",
                self.prefill_max_chunk, self.prefill_min_chunk
            ));
        }
        if self.max_seqs != 1 {
            problems.push(format!(
                "max_seqs must be 1 for Card A3.9 minimal scheduler, got {}",
                self.max_seqs
            ));
        }
        if self.k_max != 0 {
            problems.push(format!(
                "k_max must be 0 for Card A3.9 minimal scheduler, got {}",
                self.k_max
            ));
        }
        if self.vocab_size == 0 {
            problems.push("vocab_size must be >= 1".to_owned());
        }
        if let StepBudgetConfig::Manual(ms) = self.step_budget_ms {
            if !ms.is_finite() || ms <= 0.0 {
                problems.push(format!(
                    "step_budget_ms manual budget must be finite and > 0, got {ms}"
                ));
            }
        }
        if !self.min_accept.is_finite() || self.min_accept < 0.0 || self.min_accept > 1.0 {
            problems.push(format!(
                "min_accept must be finite within [0, 1], got {}",
                self.min_accept
            ));
        }
        if self.max_outstanding == 0 {
            problems.push("max_outstanding must be >= 1".to_owned());
        }
        if !problems.is_empty() {
            return Err(SchedError::invalid_request(problems));
        }
        Ok(())
    }
}

/// Core sequence scheduler executing pre-step -> device -> post-step loop (Spec 6 §1, §3).
pub struct Scheduler {
    config: SchedulerConfig,
    state_manager: StateManager,
    registry: Registry,
    arch: ArchName,
    cost_table: Arc<dyn CostTable>,
    arena: WorkspaceArena,
    graph_cache: GraphCache,
    event_chain: StepEventChain,
    schedule_log: ScheduleLogRing,
    proposer: NoOpProposer,
    detokenizer: Box<dyn Detokenizer>,
    step_counter: u64,
    arrival_counter: u64,
    queued: VecDeque<Sequence>,
    active: Option<Sequence>,
    // DECISION(A3.9): paused decode storage is a single Option<Sequence> slot, not a
    // queue; rejected Vec/VecDeque because S=1 admits at most one decode that can be
    // paused at a time, and a fixed slot makes oldest-first resume structural. Spec 6 §6.
    paused: Option<Sequence>,
    // DECISION(A3.9): pause telemetry is a fixed Option<SeqId> slot holding the exact
    // sequence ID that encountered a reserve pause; the next completed schedule record
    // reports it, then the slot clears. Rejected re-reading live paused state because
    // that reports every record while paused instead of exactly once. No heap
    // allocation. Spec 6 §6, §9.
    pending_pause_report: Option<SeqId>,
    finished: BTreeMap<SeqId, (Request, Vec<u32>, FinishReason)>,
}

impl Scheduler {
    /// Constructs a new scheduler instance, eagerly capturing warm buckets against the registry (Spec 6 §3, §5.1).
    pub fn new(
        config: SchedulerConfig,
        state_manager: StateManager,
        registry: Registry,
        arch: ArchName,
        cost_table: Arc<dyn CostTable>,
        program: StepGraphProgram,
        arena: WorkspaceArena,
    ) -> SchedResult<Self> {
        config.validate()?;

        // Eagerly capture warm step graphs at load (Spec 6 §5.1)
        let graph_cache = GraphCache::eager_capture_warm_buckets(
            &registry,
            &arch,
            config.plan_id,
            config.rank,
            program,
            &arena,
        )?;

        Ok(Self {
            config,
            state_manager,
            registry,
            arch,
            cost_table,
            arena,
            graph_cache,
            event_chain: StepEventChain::new(),
            schedule_log: ScheduleLogRing::new(4096),
            proposer: NoOpProposer::new(),
            detokenizer: Box::new(ByteDetokenizer::new()),
            step_counter: 0,
            arrival_counter: 0,
            queued: VecDeque::new(),
            active: None,
            paused: None,
            pending_pause_report: None,
            finished: BTreeMap::new(),
        })
    }

    /// Overrides the detokenizer implementation (Spec 6 §7).
    pub fn set_detokenizer(&mut self, detokenizer: Box<dyn Detokenizer>) {
        self.detokenizer = detokenizer;
    }

    /// Returns a reference to the active scheduler configuration (Spec 6 §10).
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Returns a reference to the sequence state manager (Spec 3 §5).
    pub fn state_manager(&self) -> &StateManager {
        &self.state_manager
    }

    /// Returns a mutable reference to the sequence state manager (Spec 3 §5).
    pub fn state_manager_mut(&mut self) -> &mut StateManager {
        &mut self.state_manager
    }

    /// Returns a reference to the workspace arena (Spec 6 §5.3).
    pub fn workspace_arena(&self) -> &WorkspaceArena {
        &self.arena
    }

    /// Returns a reference to the diagnostic schedule log ring buffer (Spec 6 §9).
    pub fn schedule_log(&self) -> &ScheduleLogRing {
        &self.schedule_log
    }

    /// Returns a reference to the event chain history (Spec 6 §5.4).
    pub fn event_chain(&self) -> &StepEventChain {
        &self.event_chain
    }

    /// Returns the total number of sequences currently queued or active in the engine.
    pub fn active_sequence_count(&self) -> usize {
        let mut count = self.queued.len();
        if self.active.is_some() {
            count = count.saturating_add(1);
        }
        count.saturating_add(usize::from(self.paused.is_some()))
    }

    /// Returns `true` if no sequences are queued, active, or paused.
    pub fn is_idle(&self) -> bool {
        self.active_sequence_count() == 0
    }

    /// Returns the total number of outstanding requests: finished results retained
    /// plus queued, active, and paused sequences (Spec 6 §2).
    ///
    /// Admission backpressure compares this count against
    /// [`SchedulerConfig::max_outstanding`]; taking a finished result reopens
    /// capacity.
    pub fn outstanding_count(&self) -> usize {
        self.finished
            .len()
            .saturating_add(self.queued.len())
            .saturating_add(usize::from(self.active.is_some()))
            .saturating_add(usize::from(self.paused.is_some()))
    }

    /// Enqueues a new request into the scheduler queue in arrival order (Spec 6 §1 Principle 6, §2, §4.1).
    ///
    /// Applies explicit admission backpressure: once
    /// [`Scheduler::outstanding_count`] reaches
    /// [`SchedulerConfig::max_outstanding`], the request is rejected with
    /// [`SchedError::CapacityExceeded`] before any state is allocated, so
    /// retained finished results are bounded without silent eviction.
    pub fn enqueue_request(&mut self, req: Request) -> SchedResult<SeqId> {
        let outstanding = self.outstanding_count();
        let maximum = self.config.max_outstanding as usize;
        if outstanding >= maximum {
            return Err(SchedError::CapacityExceeded {
                outstanding,
                maximum,
            });
        }
        let (seq_id, _) = self.state_manager.new_seq(&req.tokens)?;
        self.arrival_counter = self
            .arrival_counter
            .checked_add(1)
            .ok_or_else(|| SchedError::overflow("arrival_counter", "arrival counter overflow"))?;
        let sequence = Sequence::new(req, seq_id, self.arrival_counter);
        self.queued.push_back(sequence);
        Ok(seq_id)
    }

    /// Explicitly cancels an active, queued, or paused sequence (Spec 6 §7).
    pub fn cancel_sequence(&mut self, seq_id: SeqId) -> SchedResult<bool> {
        // Check active sequence
        if self.active.as_ref().map(|s| s.seq_id) == Some(seq_id) {
            if let Some(mut seq) = self.active.take() {
                self.state_manager.free_seq(seq_id)?;
                self.detokenizer.reset(seq_id);
                self.proposer.reset(seq_id);
                seq.reset_tail_state();
                seq.phase = SequencePhase::Finished(FinishReason::Cancelled);
                tracing::info!(
                    req_id = %seq.req.id.as_u64(),
                    seq_id = %seq_id.as_u64(),
                    "active sequence cancelled"
                );
                self.finished
                    .insert(seq_id, (seq.req, seq.generated, FinishReason::Cancelled));
                self.try_unpause_sequences();
                return Ok(true);
            }
        }

        // Check queued sequences
        if let Some(pos) = self.queued.iter().position(|s| s.seq_id == seq_id) {
            if let Some(mut seq) = self.queued.remove(pos) {
                self.state_manager.free_seq(seq_id)?;
                self.detokenizer.reset(seq_id);
                self.proposer.reset(seq_id);
                seq.reset_tail_state();
                seq.phase = SequencePhase::Finished(FinishReason::Cancelled);
                tracing::info!(
                    req_id = %seq.req.id.as_u64(),
                    seq_id = %seq_id.as_u64(),
                    "queued sequence cancelled"
                );
                self.finished
                    .insert(seq_id, (seq.req, seq.generated, FinishReason::Cancelled));
                return Ok(true);
            }
        }

        // Check paused sequence
        if self.paused.as_ref().map(|s| s.seq_id) == Some(seq_id) {
            if let Some(mut seq) = self.paused.take() {
                self.state_manager.free_seq(seq_id)?;
                self.detokenizer.reset(seq_id);
                self.proposer.reset(seq_id);
                seq.reset_tail_state();
                seq.phase = SequencePhase::Finished(FinishReason::Cancelled);
                // A cancelled pause reports nothing on the next record.
                if self.pending_pause_report == Some(seq_id) {
                    self.pending_pause_report = None;
                }
                tracing::info!(
                    req_id = %seq.req.id.as_u64(),
                    seq_id = %seq_id.as_u64(),
                    "paused sequence cancelled"
                );
                self.finished
                    .insert(seq_id, (seq.req, seq.generated, FinishReason::Cancelled));
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Explicitly cancels a request by its request identifier (Spec 6 §7).
    pub fn cancel_request(&mut self, req_id: ReqId) -> SchedResult<bool> {
        if let Some(ref active) = self.active {
            if active.req.id == req_id {
                return self.cancel_sequence(active.seq_id);
            }
        }
        if let Some(seq) = self.queued.iter().find(|s| s.req.id == req_id) {
            let sid = seq.seq_id;
            return self.cancel_sequence(sid);
        }
        if let Some(sid) = self
            .paused
            .as_ref()
            .filter(|s| s.req.id == req_id)
            .map(|s| s.seq_id)
        {
            return self.cancel_sequence(sid);
        }
        Ok(false)
    }

    /// Returns the finished execution result for a completed sequence if available (Spec 6 §7).
    pub fn get_finished_result(&self, seq_id: SeqId) -> Option<&(Request, Vec<u32>, FinishReason)> {
        self.finished.get(&seq_id)
    }

    /// Takes ownership of the finished execution result for a completed sequence,
    /// removing it from the scheduler so state is released exactly once (Spec 6 §7).
    pub fn take_finished_result(
        &mut self,
        seq_id: SeqId,
    ) -> Option<(Request, Vec<u32>, FinishReason)> {
        self.finished.remove(&seq_id)
    }

    // DECISION(A3.9): unpause is promotion-only. The sole paused sequence moves to
    // active with no state reservation here; the next actual step performs exactly one
    // reserve in its decode branch, which stalls cleanly (no StepId allocated) while
    // memory pressure persists. Rejected reserving inside unpause because the old path
    // reserved once here and again on the next step (dead code that double-reserved on
    // the only path that could reach it). Spec 6 §6.
    /// Moves the sole paused decoding sequence back to active without reserving
    /// state when the active slot is free (Spec 6 §6).
    ///
    /// Promotion performs no reservation and therefore cannot fail: it succeeds even
    /// under memory pressure, and the next actual step performs exactly one reserve.
    fn try_unpause_sequences(&mut self) {
        if self.active.is_none() {
            if let Some(candidate) = self.paused.take() {
                tracing::info!(
                    req_id = %candidate.req.id.as_u64(),
                    seq_id = %candidate.seq_id.as_u64(),
                    "paused decode sequence unpaused"
                );
                self.active = Some(candidate);
            }
        }
    }

    /// Executes a single scheduler step across pre-step, device replay, and post-step phases (Spec 6 §3).
    ///
    /// The device phase runs upload -> replay -> readback in order against the
    /// explicit [`StepExecutor`]; post-step consumes only the [`DeviceStepSample`]
    /// read back from the executor. Accepted tokens come from that readback alone.
    pub fn step(&mut self, exec: &mut dyn StepExecutor) -> SchedResult<Option<StepResult>> {
        if self.is_idle() {
            return Ok(None);
        }

        let budget_ms = self
            .cost_table
            .resolve_budget_ms(self.config.step_budget_ms, self.config.profile)?;
        let pre_step_estimate_ms = estimate_pre_step_ms(budget_ms);

        // ---------------------------------------------------------------------
        // 3.1 Pre-step (Host)
        // ---------------------------------------------------------------------
        let mut forced_admission = false;
        let s_logical = 1u32;
        let mut t_dec_logical = 0u32;
        let mut t_pre_logical = 0u32;
        let mut admitted_chunk = 0u32;
        let mut seqs_decode = InlineVec::<SeqId, 1>::new();
        let mut seqs_prefill = InlineVec::<(SeqId, u32), 1>::new();

        // 1. Promote paused or queued sequence to active if slot is empty (Single sequence S=1)
        // Card A3.9 requirement 1: pre-step retries oldest paused decode before queued work.
        // No state reservation happens here; the decode branch below performs exactly
        // one reserve for the resumed sequence (Spec 6 §6).
        if self.active.is_none() {
            if let Some(candidate) = self.paused.take() {
                // Oldest (only) paused sequence resumes before queued work
                self.active = Some(candidate);
            } else if let Some(mut seq) = self.queued.pop_front() {
                seq.phase = SequencePhase::Prefilling { done: 0 };
                self.active = Some(seq);
            }
        }

        let mut active_seq = match self.active.take() {
            Some(s) => s,
            None => return Ok(None),
        };

        // 2. Schedule active sequence
        match active_seq.phase {
            SequencePhase::Decoding => {
                // Spec 6 §3.1 step 2: Admit decode (never preempted or killed for prefill)
                t_dec_logical = 1;

                // Spec 7 §2: Proposer call site (k=0 in A3.9)
                self.proposer.draft(active_seq.seq_id, 0);

                // Spec 6 §3.1 step 6: Reserve state
                match self.state_manager.reserve(active_seq.seq_id, 1) {
                    Ok(_) => {
                        let _ = seqs_decode.push(active_seq.seq_id);
                    }
                    Err(r9v_state::StateError::PoolExhausted { .. }) => {
                        // Spec 6 §6: Decode reserve failure pauses sequence; never killed.
                        // Stall cleanly before graph replay; StepId is not allocated.
                        // The exact paused ID is retained in the fixed telemetry slot for
                        // the next completed record.
                        tracing::warn!(
                            req_id = %active_seq.req.id.as_u64(),
                            seq_id = %active_seq.seq_id.as_u64(),
                            "decode reserve failed on memory pressure; sequence paused, stalling step"
                        );
                        self.pending_pause_report = Some(active_seq.seq_id);
                        self.paused = Some(active_seq);
                        return Ok(None);
                    }
                    Err(e) => return Err(SchedError::State(e)),
                }
            }
            SequencePhase::Prefilling { done } => {
                // Spec 6 §3.1 step 4: Admit prefill
                let remaining = active_seq.prompt_len().checked_sub(done).ok_or_else(|| {
                    SchedError::overflow("prefill_remaining", "prompt len underflow")
                })?;
                let room =
                    self.cost_table
                        .prefill_room_ms(budget_ms, 0, 0, pre_step_estimate_ms)?;

                let chunk_opt = self.cost_table.select_prefill_chunk(
                    remaining,
                    0,
                    0,
                    room,
                    self.config.prefill_min_chunk,
                    self.config.prefill_max_chunk,
                )?;

                let chunk = match chunk_opt {
                    Some(c) => c,
                    None => {
                        // Spec 6 §4.1: If no chunk fits within budget, accumulate wait time.
                        // If wait exceeds max_wait_ms, force admission.
                        active_seq.accumulated_wait_ms += budget_ms;
                        active_seq.wait_steps =
                            active_seq.wait_steps.checked_add(1).ok_or_else(|| {
                                SchedError::overflow("wait_steps", "wait steps overflow")
                            })?;

                        if active_seq.accumulated_wait_ms >= self.config.max_wait_ms as f32 {
                            forced_admission = true;
                            remaining.min(self.config.prefill_min_chunk)
                        } else {
                            // Prompt waits; clean stall before device replay, StepId not allocated
                            tracing::debug!(
                                req_id = %active_seq.req.id.as_u64(),
                                wait_steps = active_seq.wait_steps,
                                accumulated_wait_ms = active_seq.accumulated_wait_ms,
                                "prefill chunk does not fit within budget; prompt waits"
                            );
                            self.active = Some(active_seq);
                            return Ok(None);
                        }
                    }
                };

                // Spec 6 §3.1 step 6: Reserve state for prefill chunk
                match self.state_manager.reserve(active_seq.seq_id, chunk) {
                    Ok(_) => {
                        t_pre_logical = chunk;
                        admitted_chunk = chunk;
                        let _ = seqs_prefill.push((active_seq.seq_id, chunk));

                        // Spec 7 §2: Notify proposer of prefill chunk
                        let slice_start = done as usize;
                        let slice_end =
                            slice_start.checked_add(chunk as usize).ok_or_else(|| {
                                SchedError::overflow("slice_end", "slice end overflow")
                            })?;
                        if slice_end <= active_seq.req.tokens.len() {
                            self.proposer.on_prefill(
                                active_seq.seq_id,
                                &active_seq.req.tokens[slice_start..slice_end],
                            );
                        }
                    }
                    Err(r9v_state::StateError::PoolExhausted { .. }) => {
                        // Spec 6 §6: Prefill reserve failure: chunk not admitted, prompt waits
                        active_seq.accumulated_wait_ms += budget_ms;
                        active_seq.wait_steps =
                            active_seq.wait_steps.checked_add(1).ok_or_else(|| {
                                SchedError::overflow("wait_steps", "wait steps overflow")
                            })?;
                        tracing::warn!(
                            req_id = %active_seq.req.id.as_u64(),
                            "prefill reserve failed on memory pressure; prompt waits"
                        );
                        self.active = Some(active_seq);
                        return Ok(None);
                    }
                    Err(e) => return Err(SchedError::State(e)),
                }
            }
            _ => {
                self.active = Some(active_seq);
                return Ok(None);
            }
        }

        // Work admitted and device step is guaranteed to execute. Allocate StepId now.
        self.step_counter = self
            .step_counter
            .checked_add(1)
            .ok_or_else(|| SchedError::overflow("step_counter", "step counter overflow"))?;
        let step_id = StepId::new(self.step_counter);

        // Discrete shape bucket resolution (Spec 1 §3.5)
        let (s_bucket, t_dec_bucket, t_pre_bucket) =
            bucket_step(s_logical.max(1), t_dec_logical.max(1), t_pre_logical)?;

        let key = StepGraphKey {
            plan_id: self.config.plan_id,
            rank: self.config.rank,
            s: s_bucket,
            t_dec: t_dec_bucket,
            t_pre: t_pre_bucket,
            segment: 0,
        };

        // Construct public Step in pre-step (Spec 6 §2, §3.1)
        let mut inline_graphs = InlineVec::new();
        let _ = inline_graphs.push(key);
        let mut inline_k = InlineVec::new();
        if s_logical > 0 {
            let _ = inline_k.push(0);
        }

        let step = Step {
            step_id,
            seqs_decode,
            seqs_prefill,
            k: inline_k,
            bucket: (s_bucket, t_dec_bucket, t_pre_bucket),
            graphs: inline_graphs,
        };

        // ---------------------------------------------------------------------
        // 3.2 Device Replay via explicit StepExecutor (Spec 6 §3.2, §5.4)
        // ---------------------------------------------------------------------
        // Reset workspace arena bump state for this step (Spec 6 §5.3)
        self.arena.reset();

        // Begin step on event chain with fixed per-step storage (Spec 6 §5.4)
        self.event_chain.begin_step(step_id);

        // Copy stream uploads the actual token IDs, positions, and batch facts (Spec 6 §5.4)
        let prefill_upload = match active_seq.phase {
            SequencePhase::Prefilling { done } => Some((done, admitted_chunk)),
            _ => None,
        };
        let upload = StepInputs {
            step_id,
            seq_id: active_seq.seq_id,
            prompt_tokens: &active_seq.req.tokens,
            generated_tokens: &active_seq.generated,
            prefill: prefill_upload,
            ctx_len: active_seq.ctx_len,
        };
        exec.upload_inputs(&upload)?;

        // Record UploadComplete on Copy stream (Spec 6 §5.4)
        let upload_eid = self.event_chain.record_upload_complete(step_id)?;

        // Compute stream waits on upload completion (Spec 6 §5.4)
        self.event_chain
            .record_wait(step_id, StreamKind::Compute, upload_eid)?;

        // Retrieve or lazily capture step graph, then replay it on the Compute
        // stream (Spec 4 §7, Spec 6 §5.1, §5.2, §5.3). Lazy capture may grow the
        // arena and recapture every graph; warm buckets never grow (Spec 6 §5.3).
        let captured = {
            let Self {
                graph_cache,
                registry,
                arch,
                arena,
                ..
            } = &mut *self;
            let (graph, was_captured) = graph_cache.get_or_capture(registry, arch, key, arena)?;
            exec.replay_graph(graph)?;
            was_captured
        };

        // Record ComputeComplete on Compute stream (Spec 6 §5.4)
        let compute_eid = self.event_chain.record_compute_complete(step_id)?;

        // Copy stream waits on compute completion (Spec 6 §5.4)
        self.event_chain
            .record_wait(step_id, StreamKind::Copy, compute_eid)?;

        // Device sampling readback into fixed output; this is the ONLY token
        // source post-step may consume (Spec 6 §3.2). The readback takes no
        // host-derived sampling hints: the token is the actual device Sample op
        // output for the uploaded inputs.
        let device_sample = exec.readback_sample()?;

        // Card A3.9 runs with k=0, so every executed sampling step reports exactly
        // one accepted token. Any other value is a device contract violation and is
        // rejected here, before commit, log, and EMA mutation.
        if device_sample.accept_len != 1 {
            return Err(SchedError::ExecutionFailed {
                detail: format!(
                    "k=0 step requires device accept_len == 1, got {} (seq {}, step {})",
                    device_sample.accept_len,
                    active_seq.seq_id.as_u64(),
                    step_id.as_u64(),
                ),
            });
        }

        // Record ReadbackComplete on Copy stream (Spec 6 §5.4)
        self.event_chain.record_readback_complete(step_id)?;

        // Validate complete three-stream event sequence without heap allocation (Spec 6 §5.4)
        self.event_chain.validate_step_chain(step_id)?;

        // ---------------------------------------------------------------------
        // 3.3 Post-step (Host)
        // ---------------------------------------------------------------------
        let mut accepted_tokens: InlineVec<(SeqId, InlineVec<u32, 1>), 1> = InlineVec::new();
        let mut finished_sequences: InlineVec<(SeqId, FinishReason), 1> = InlineVec::new();
        let mut finish_reason: Option<FinishReason> = None;
        let mut accept_len_val: u32 = 0;

        match active_seq.phase {
            SequencePhase::Prefilling { done } => {
                let new_done = done
                    .checked_add(admitted_chunk)
                    .ok_or_else(|| SchedError::overflow("prefill_done", "prefill done overflow"))?;
                self.state_manager
                    .commit(active_seq.seq_id, admitted_chunk)?;
                active_seq.ctx_len = active_seq
                    .ctx_len
                    .checked_add(admitted_chunk)
                    .ok_or_else(|| SchedError::overflow("ctx_len", "ctx len overflow"))?;

                if new_done >= active_seq.prompt_len() {
                    // Prompt complete: transition to Decoding and evaluate the device
                    // readback token from the last position (Spec 6 §3.3 step 3)
                    active_seq.phase = SequencePhase::Decoding;
                    let (opt_finish, is_trimmed) = active_seq
                        .append_generated_token(device_sample.token, &mut *self.detokenizer)?;
                    if !is_trimmed {
                        let mut tok_vec = InlineVec::new();
                        let _ = tok_vec.push(device_sample.token);
                        let _ = accepted_tokens.push((active_seq.seq_id, tok_vec));
                        accept_len_val = device_sample.accept_len;
                    } else {
                        accept_len_val = 0;
                    }
                    finish_reason = opt_finish;
                } else {
                    active_seq.phase = SequencePhase::Prefilling { done: new_done };
                    // Intermediate prefill chunk: accept_len is 0 (Spec 6 §9)
                    accept_len_val = 0;
                }
            }
            SequencePhase::Decoding => {
                // Spec 6 §3.3 step 2: Commit accepted token, evaluate incremental stop criteria
                self.state_manager.commit(active_seq.seq_id, 1)?;
                active_seq.ctx_len = active_seq
                    .ctx_len
                    .checked_add(1)
                    .ok_or_else(|| SchedError::overflow("ctx_len", "ctx len overflow"))?;

                let (opt_finish, is_trimmed) = active_seq
                    .append_generated_token(device_sample.token, &mut *self.detokenizer)?;
                if !is_trimmed {
                    let mut tok_vec = InlineVec::new();
                    let _ = tok_vec.push(device_sample.token);
                    let _ = accepted_tokens.push((active_seq.seq_id, tok_vec));
                    accept_len_val = device_sample.accept_len;
                } else {
                    accept_len_val = 0;
                }
                finish_reason = opt_finish;

                // Update accept_ema with alpha=0.2 (Spec 6 §4.2)
                active_seq.accept_ema = 0.8 * active_seq.accept_ema + 0.2 * (accept_len_val as f32);

                // Update proposer observe call site (Spec 6 §3.3 step 2, Spec 7 §2)
                self.proposer
                    .observe(active_seq.seq_id, &[device_sample.token]);
            }
            _ => {}
        }

        // Handle sequence finish (Spec 6 §7)
        if let Some(reason) = finish_reason {
            self.state_manager.free_seq(active_seq.seq_id)?;
            self.detokenizer.reset(active_seq.seq_id);
            self.proposer.reset(active_seq.seq_id);
            active_seq.reset_tail_state();
            active_seq.phase = SequencePhase::Finished(reason.clone());
            let _ = finished_sequences.push((active_seq.seq_id, reason.clone()));
            tracing::info!(
                req_id = %active_seq.req.id.as_u64(),
                seq_id = %active_seq.seq_id.as_u64(),
                "sequence finished generation"
            );
            self.finished.insert(
                active_seq.seq_id,
                (active_seq.req.clone(), active_seq.generated.clone(), reason),
            );
            // Sequence finished, clear active and unpause any stalled sequence
            self.active = None;
            self.try_unpause_sequences();
        } else {
            self.active = Some(active_seq);
        }

        // 6. Append step record to schedule log (Spec 6 §3.3 step 6, §9)
        let cost_ms = self
            .cost_table
            .cost_ms(s_bucket, t_dec_bucket, t_pre_bucket)?;
        let t_device_us = (cost_ms * 1000.0) as u64;
        let t_post_us = 0u64;

        let mut inline_accept_len = InlineVec::new();
        if s_logical > 0 {
            let _ = inline_accept_len.push(accept_len_val);
        }

        // Pause telemetry: report the exact sequence ID retained at the reserve
        // pause in the fixed pending slot, then clear the slot so exactly the next
        // completed record carries it (Spec 6 §6, §9). Stalled steps allocate no
        // StepId, so there is no gap before this record.
        let mut paused_ids = InlineVec::new();
        if let Some(paused_seq_id) = self.pending_pause_report.take() {
            let _ = paused_ids.push(paused_seq_id);
        }

        let record = ScheduleRecord {
            step_id,
            t_pre_us: (pre_step_estimate_ms * 1000.0) as u64,
            t_draft_us: 0,
            t_device_us,
            t_post_us,
            s: s_logical,
            t_dec: t_dec_logical,
            t_pre: t_pre_logical,
            chunk: admitted_chunk,
            k: inline_k,
            accept_len: inline_accept_len,
            forced_admission,
            budget_ms,
            bucket: (s_bucket, t_dec_bucket, t_pre_bucket),
            graph_mode: self.config.graph_mode,
            captured,
            paused: paused_ids,
            segment_sync_us: 0,
        };
        self.schedule_log.push(record.clone())?;

        tracing::debug!(
            step_id = %step_id.as_u64(),
            s = s_bucket,
            t_dec = t_dec_bucket,
            t_pre = t_pre_bucket,
            "step completed"
        );

        Ok(Some(StepResult {
            step,
            step_id,
            bucket: (s_bucket, t_dec_bucket, t_pre_bucket),
            accepted_tokens,
            finished_sequences,
            record,
        }))
    }
}

#[cfg(test)]
mod promotion_tests {
    // Direct regression for promotion-only unpause (Spec 6 §6).
    //
    // try_unpause_sequences is unreachable through the public S=1 step loop (pre-step
    // always promotes the paused slot itself), so this unit test forces the transition
    // directly: with the pool still exhausted, promotion must move the sole paused
    // sequence to active without reserving. The old reserve-inside-unpause path left
    // the sequence paused here (and would have reserved twice on the next step); the
    // next actual step then performs exactly one reserve and stalls cleanly with no
    // StepId allocated. Uses assert!/assert_eq! only, no unwrap/expect/panic.
    use super::*;
    use std::sync::Arc;

    use r9v_common::ReqId;
    use r9v_ir::{AttentionMask, DType, Epilogue, LayoutId, QuantScheme};
    use r9v_registry::{
        AttentionStatic, BundleManifest, ElementwiseStatic, LaunchGeometry, ManifestVariantEntry,
        MatmulStatic, OpId, OpStatic, RegistryConfig, SamplingMethod, SamplingStatic, Tier,
        VariantHash,
    };

    use crate::graph::StepProgramOp;
    use crate::types::StopCriteria;
    use r9v_state::{CacheDtype, Retain, StateConfig, StateSpec};

    struct FixedStub;
    impl StepExecutor for FixedStub {
        fn upload_inputs(&mut self, _input: &StepInputs<'_>) -> SchedResult<()> {
            Ok(())
        }
        fn replay_graph(&mut self, graph: &CapturedGraph) -> SchedResult<()> {
            let _ = graph.launches.len();
            Ok(())
        }
        fn readback_sample(&mut self) -> SchedResult<DeviceStepSample> {
            Ok(DeviceStepSample {
                token: 42,
                accept_len: 1,
            })
        }
    }

    fn unit_registry(arch: &ArchName) -> Registry {
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
        let installed = registry.set_manifest(manifest, None);
        assert!(installed.is_ok(), "unit manifest must install");
        registry
    }

    fn unit_program() -> StepGraphProgram {
        let template = vec![0u8; 64];
        StepGraphProgram::new()
            .with_op(StepProgramOp::new(
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
            ))
            .with_op(StepProgramOp::new(
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
            ))
            .with_op(StepProgramOp::new(
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
            ))
            .with_op(StepProgramOp::new(
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
            ))
    }

    #[test]
    fn unpause_promotes_paused_to_active_without_reserve_under_pressure() {
        let arch = ArchName::from("gfx1201");
        let registry = unit_registry(&arch);
        let cost_table = Arc::new(crate::cost::CostTableStub::default());
        let state_config = StateConfig {
            max_ctx: 256,
            max_seqs: 16,
        };
        let layer_specs = vec![StateSpec::KvPaged {
            hkv: 8,
            d: 128,
            dv: 128,
            cache: CacheDtype::E4M3,
            retain: Retain::All,
        }];
        let state_manager = StateManager::new(state_config, layer_specs, 532480);
        assert!(state_manager.is_ok(), "unit state manager must init");
        let mut state_manager = match state_manager {
            Ok(m) => m,
            Err(_) => return,
        };
        // Exhaust the pool with a background sequence: 4 of 8 blocks.
        let blocker = state_manager.new_seq(&[1; 128]);
        assert!(blocker.is_ok(), "blocker seq must allocate");
        let (blocker_id, _) = match blocker {
            Ok(v) => v,
            Err(_) => return,
        };
        assert!(state_manager.reserve(blocker_id, 128).is_ok());
        assert!(state_manager.commit(blocker_id, 128).is_ok());

        let arena = WorkspaceArena::new(0, 32 * 1024 * 1024);
        let config = SchedulerConfig {
            max_outstanding: 1024,
            ..SchedulerConfig::default()
        };
        let scheduler = Scheduler::new(
            config,
            state_manager,
            registry,
            arch,
            cost_table,
            unit_program(),
            arena,
        );
        assert!(scheduler.is_ok(), "unit scheduler must init");
        let mut scheduler = match scheduler {
            Ok(s) => s,
            Err(_) => return,
        };

        let sampling = r9v_ir::SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: vec![],
        };
        let req = Request::new(
            ReqId::new(901),
            vec![42; 128],
            sampling,
            10,
            StopCriteria::default(),
            false,
        );
        assert!(req.is_ok(), "unit request must validate");
        let req = match req {
            Ok(r) => r,
            Err(_) => return,
        };
        let enqueued = scheduler.enqueue_request(req);
        assert!(enqueued.is_ok(), "unit enqueue must succeed");
        let seq_id = match enqueued {
            Ok(id) => id,
            Err(_) => return,
        };

        // Step 1 consumes the remaining 4 blocks via prefill.
        let mut exec = FixedStub;
        let first = scheduler.step(&mut exec);
        assert!(first.is_ok(), "prefill step must succeed");
        assert!(first.map(|r| r.is_some()).unwrap_or(false));

        // Step 2: decode reserve fails under pressure; the sequence pauses and the
        // step stalls cleanly with no StepId allocated.
        let stalled = scheduler.step(&mut exec);
        assert!(stalled.is_ok(), "pause stall must not error");
        assert!(
            stalled.map(|r| r.is_none()).unwrap_or(false),
            "step must stall cleanly while paused"
        );
        assert!(
            scheduler.active.is_none(),
            "stalled step holds no active seq"
        );
        assert_eq!(scheduler.paused.as_ref().map(|s| s.seq_id), Some(seq_id));

        // Forced transition while the pool is STILL exhausted: promotion-only unpause
        // moves the sequence to active with no reserve, so no reserve error is
        // possible here by construction.
        scheduler.try_unpause_sequences();
        assert_eq!(
            scheduler.active.as_ref().map(|s| s.seq_id),
            Some(seq_id),
            "unpause must promote the paused sequence without reserving"
        );
        assert!(
            scheduler.paused.is_none(),
            "paused slot must drain on promote"
        );

        // The next actual step performs exactly one reserve: pressure persists, so it
        // stalls cleanly again with no StepId allocated and the sequence paused.
        let stalled_again = scheduler.step(&mut exec);
        assert!(stalled_again.is_ok(), "second stall must not error");
        assert!(
            stalled_again.map(|r| r.is_none()).unwrap_or(false),
            "next step must stall on the single decode reserve"
        );
        assert_eq!(
            scheduler.paused.as_ref().map(|s| s.seq_id),
            Some(seq_id),
            "sequence must be paused again after the single reserve fails"
        );
        assert_eq!(
            scheduler.schedule_log().total_written(),
            1,
            "only the prefill step logged; stalls allocate no StepId"
        );
    }
}
