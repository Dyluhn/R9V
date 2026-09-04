// SPDX-License-Identifier: Apache-2.0
//! Three streams and explicit event chain execution model (Spec 6 §5.4).

use r9v_common::StepId;
use serde::{Deserialize, Serialize};

use crate::error::{SchedError, SchedResult};

/// Maximum synchronization events retained per single scheduler step (Spec 6 §5.4).
pub const MAX_EVENTS_PER_STEP: usize = 16;

/// The three execution streams defined per device rank (Spec 6 §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamKind {
    /// Compute stream replaying the captured step graph (Spec 6 §5.4).
    Compute,
    /// Communications stream handling pipeline parallel boundaries and collectives (Spec 5 §6.3, Spec 6 §5.4).
    Comms,
    /// Copy stream executing H2D uploads and D2H sampled token readbacks (Spec 6 §5.4).
    Copy,
}

/// Synchronization event types within the step execution lifecycle (Spec 6 §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// Recorded on Copy stream when H2D uploads of token IDs, positions, and BatchMeta complete (Spec 6 §5.4).
    UploadComplete,
    /// Recorded when a stream begins waiting on an upload event (Spec 6 §5.4).
    WaitUpload,
    /// Recorded on Compute stream when kernel graph replay finishes (Spec 6 §5.4).
    ComputeComplete,
    /// Recorded when a stream begins waiting on a compute complete event (Spec 6 §5.4).
    WaitCompute,
    /// Recorded on Copy stream when D2H readback of sampled tokens completes (Spec 6 §5.4).
    ReadbackComplete,
    /// Recorded on Comms stream for inter-rank boundary collectives or pipeline transfers (Spec 5 §6.3, Spec 6 §5.4).
    CommsComplete,
    /// Recorded when a stream waits on communication completion (Spec 6 §5.4).
    WaitComms,
}

/// Unique synchronization event identifier (Spec 6 §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EventId(u64);

impl EventId {
    /// Constructs a new EventId (CONVENTIONS.md §3.1).
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying raw integer value.
    pub const fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Recorded event trace entry within the step event chain (Spec 6 §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepEventRecord {
    /// Step during which this event was recorded (Spec 6 §5.4).
    pub step_id: StepId,
    /// Stream where the event was recorded or awaited (Spec 6 §5.4).
    pub stream: StreamKind,
    /// Type of synchronization event (Spec 6 §5.4).
    pub event_kind: EventKind,
    /// Unique event identifier.
    pub event_id: EventId,
    /// Specific event awaited by this wait record, if applicable (Spec 6 §5.4).
    pub awaited_event: Option<EventId>,
}

/// Explicit event chain managing the deterministic three-stream sequence per step (Spec 6 §5.4).
///
/// Order per step:
/// 1. Copy stream uploads -> record `UploadComplete`.
/// 2. Compute stream waits referencing exact `UploadComplete`.
/// 3. Compute stream replays kernel graph -> record `ComputeComplete`.
/// 4. Copy stream waits referencing exact `ComputeComplete`.
/// 5. Copy stream reads back `sampled` and `accept_len` -> record `ReadbackComplete`.
#[derive(Debug, Clone)]
pub struct StepEventChain {
    event_counter: u64,
    current_step: Option<StepId>,
    records: [Option<StepEventRecord>; MAX_EVENTS_PER_STEP],
    count: usize,
}

impl Default for StepEventChain {
    fn default() -> Self {
        Self::new()
    }
}

impl StepEventChain {
    /// Constructs a new step event chain tracker with fixed per-step storage (Spec 6 §5.4).
    pub fn new() -> Self {
        Self {
            event_counter: 0,
            current_step: None,
            records: [None; MAX_EVENTS_PER_STEP],
            count: 0,
        }
    }

    /// Resets fixed per-step event storage once per actual step (Spec 6 §5.4).
    pub fn begin_step(&mut self, step_id: StepId) {
        self.current_step = Some(step_id);
        self.records = [None; MAX_EVENTS_PER_STEP];
        self.count = 0;
    }

    /// Returns the number of events recorded for the current step.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no events have been recorded for the current step.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns a reference to the event at the given index, or None if out of bounds.
    pub fn get(&self, index: usize) -> Option<&StepEventRecord> {
        if index < self.count {
            self.records.get(index)?.as_ref()
        } else {
            None
        }
    }

    /// Allocates the next unique event identifier with checked overflow (Spec 6 §5.4, CONVENTIONS.md §1.3).
    pub fn next_event_id(&mut self) -> SchedResult<EventId> {
        let next = self
            .event_counter
            .checked_add(1)
            .ok_or_else(|| SchedError::overflow("event_id", "u64 event counter overflow"))?;
        self.event_counter = next;
        Ok(EventId::new(next))
    }

    /// Records an event entry in the fixed per-step storage.
    fn record_entry(
        &mut self,
        step_id: StepId,
        stream: StreamKind,
        event_kind: EventKind,
        awaited_event: Option<EventId>,
    ) -> SchedResult<EventId> {
        if self.count >= MAX_EVENTS_PER_STEP {
            return Err(SchedError::overflow(
                "step_event_storage",
                format!("maximum {MAX_EVENTS_PER_STEP} events per step exceeded"),
            ));
        }
        let eid = self.next_event_id()?;
        let slot = self.records.get_mut(self.count).ok_or_else(|| {
            SchedError::Internal("step event storage index out of bounds".to_owned())
        })?;
        *slot = Some(StepEventRecord {
            step_id,
            stream,
            event_kind,
            event_id: eid,
            awaited_event,
        });
        self.count += 1;
        Ok(eid)
    }

    /// Records that H2D inputs were uploaded on the Copy stream and records `UploadComplete` (Spec 6 §5.4).
    pub fn record_upload_complete(&mut self, step_id: StepId) -> SchedResult<EventId> {
        self.record_entry(step_id, StreamKind::Copy, EventKind::UploadComplete, None)
    }

    /// Records that a stream begins waiting on a previous synchronization event (Spec 6 §5.4).
    pub fn record_wait(
        &mut self,
        step_id: StepId,
        stream: StreamKind,
        awaited_event: EventId,
    ) -> SchedResult<EventId> {
        let kind = match stream {
            StreamKind::Compute => EventKind::WaitUpload,
            StreamKind::Copy => EventKind::WaitCompute,
            StreamKind::Comms => EventKind::WaitComms,
        };
        self.record_entry(step_id, stream, kind, Some(awaited_event))
    }

    /// Records that the Compute stream finished replay, recording `ComputeComplete` (Spec 6 §5.4).
    pub fn record_compute_complete(&mut self, step_id: StepId) -> SchedResult<EventId> {
        self.record_entry(
            step_id,
            StreamKind::Compute,
            EventKind::ComputeComplete,
            None,
        )
    }

    /// Records that the Copy stream finished D2H readback, recording `ReadbackComplete` (Spec 6 §5.4).
    pub fn record_readback_complete(&mut self, step_id: StepId) -> SchedResult<EventId> {
        self.record_entry(step_id, StreamKind::Copy, EventKind::ReadbackComplete, None)
    }

    /// Records an explicit event on the Comms stream (Spec 5 §6.3, Spec 6 §5.4).
    pub fn record_comms_event(
        &mut self,
        step_id: StepId,
        event_kind: EventKind,
    ) -> SchedResult<EventId> {
        self.record_entry(step_id, StreamKind::Comms, event_kind, None)
    }

    /// Validates that the event chain contains the exact five required records:
    /// 1. Copy UploadComplete
    /// 2. Compute wait referencing exact UploadComplete
    /// 3. Compute ComputeComplete
    /// 4. Copy wait referencing exact ComputeComplete
    /// 5. Copy ReadbackComplete
    ///
    /// Fails on missing, wrong, or reordered edges without allocating (Spec 6 §5.4).
    pub fn validate_step_chain(&self, step_id: StepId) -> SchedResult<()> {
        if self.count != 5 {
            return Err(SchedError::ExecutionFailed {
                detail: format!(
                    "step {}: expected exactly 5 stream events in event chain, found {}",
                    step_id.as_u64(),
                    self.count
                ),
            });
        }

        let missing = |i: usize| SchedError::ExecutionFailed {
            detail: format!("step {}: missing event record {i}", step_id.as_u64()),
        };
        let r0 = self.get(0).ok_or_else(|| missing(0))?;
        let r1 = self.get(1).ok_or_else(|| missing(1))?;
        let r2 = self.get(2).ok_or_else(|| missing(2))?;
        let r3 = self.get(3).ok_or_else(|| missing(3))?;
        let r4 = self.get(4).ok_or_else(|| missing(4))?;

        // Step ID check across all records
        if r0.step_id != step_id
            || r1.step_id != step_id
            || r2.step_id != step_id
            || r3.step_id != step_id
            || r4.step_id != step_id
        {
            return Err(SchedError::ExecutionFailed {
                detail: format!("step {}: event record step_id mismatch", step_id.as_u64()),
            });
        }

        // Strict monotonic event IDs
        if !(r0.event_id.as_u64() < r1.event_id.as_u64()
            && r1.event_id.as_u64() < r2.event_id.as_u64()
            && r2.event_id.as_u64() < r3.event_id.as_u64()
            && r3.event_id.as_u64() < r4.event_id.as_u64())
        {
            return Err(SchedError::ExecutionFailed {
                detail: format!(
                    "step {}: non-monotonic event IDs in event chain",
                    step_id.as_u64()
                ),
            });
        }

        // 1. Copy UploadComplete
        if r0.stream != StreamKind::Copy || r0.event_kind != EventKind::UploadComplete {
            return Err(SchedError::ExecutionFailed {
                detail: format!(
                    "step {}: event 0 must be Copy UploadComplete, got {:?} on {:?}",
                    step_id.as_u64(),
                    r0.event_kind,
                    r0.stream
                ),
            });
        }
        let upload_eid = r0.event_id;

        // 2. Compute wait referencing exact UploadComplete
        if r1.stream != StreamKind::Compute
            || r1.event_kind != EventKind::WaitUpload
            || r1.awaited_event != Some(upload_eid)
        {
            return Err(SchedError::ExecutionFailed {
                detail: format!(
                    "step {}: event 1 must be Compute wait referencing UploadComplete ({}), got {:?} on {:?} awaiting {:?}",
                    step_id.as_u64(),
                    upload_eid.as_u64(),
                    r1.event_kind,
                    r1.stream,
                    r1.awaited_event.map(|e| e.as_u64())
                ),
            });
        }

        // 3. Compute ComputeComplete
        if r2.stream != StreamKind::Compute || r2.event_kind != EventKind::ComputeComplete {
            return Err(SchedError::ExecutionFailed {
                detail: format!(
                    "step {}: event 2 must be Compute ComputeComplete, got {:?} on {:?}",
                    step_id.as_u64(),
                    r2.event_kind,
                    r2.stream
                ),
            });
        }
        let compute_eid = r2.event_id;

        // 4. Copy wait referencing exact ComputeComplete
        if r3.stream != StreamKind::Copy
            || r3.event_kind != EventKind::WaitCompute
            || r3.awaited_event != Some(compute_eid)
        {
            return Err(SchedError::ExecutionFailed {
                detail: format!(
                    "step {}: event 3 must be Copy wait referencing ComputeComplete ({}), got {:?} on {:?} awaiting {:?}",
                    step_id.as_u64(),
                    compute_eid.as_u64(),
                    r3.event_kind,
                    r3.stream,
                    r3.awaited_event.map(|e| e.as_u64())
                ),
            });
        }

        // 5. Copy ReadbackComplete
        if r4.stream != StreamKind::Copy || r4.event_kind != EventKind::ReadbackComplete {
            return Err(SchedError::ExecutionFailed {
                detail: format!(
                    "step {}: event 4 must be Copy ReadbackComplete, got {:?} on {:?}",
                    step_id.as_u64(),
                    r4.event_kind,
                    r4.stream
                ),
            });
        }

        Ok(())
    }

    /// Clears recorded events.
    pub fn clear(&mut self) {
        self.records = [None; MAX_EVENTS_PER_STEP];
        self.count = 0;
        self.current_step = None;
    }
}
