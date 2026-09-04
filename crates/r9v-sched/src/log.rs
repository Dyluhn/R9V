// SPDX-License-Identifier: Apache-2.0
//! Diagnostic schedule log record and ring buffer (Spec 6 §9).

use r9v_common::{SeqId, StepId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{SchedError, SchedResult};
use crate::types::InlineVec;

/// Maximum capacity of the in-memory schedule log ring buffer (Spec 6 §9).
pub const SCHEDULE_LOG_CAPACITY: usize = 4096;

mod serde_step_id {
    use super::*;
    pub fn serialize<S>(id: &StepId, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(id.as_u64())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<StepId, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = u64::deserialize(deserializer)?;
        Ok(StepId::new(v))
    }
}

mod serde_paused_seqs {
    use super::*;
    use serde::ser::SerializeSeq;

    pub fn serialize<S>(ids: &InlineVec<SeqId, 1>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(ids.len()))?;
        for id in ids.iter() {
            seq.serialize_element(&id.as_u64())?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<InlineVec<SeqId, 1>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // DECISION(A3.9): over-capacity paused arrays are rejected, never silently
        // truncated; rejected let-_ push because dropping sequence IDs hides a
        // malformed log. Spec 6 §9.
        let raw = Vec::<u64>::deserialize(deserializer)?;
        if raw.len() > 1 {
            return Err(serde::de::Error::invalid_length(
                raw.len(),
                &"at most 1 paused sequence",
            ));
        }
        let mut res = InlineVec::default();
        for id in raw {
            res.push(SeqId::new(id)).map_err(serde::de::Error::custom)?;
        }
        Ok(res)
    }
}

/// Replay and execution mode for step graphs (Spec 6 §5.2, §10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GraphMode {
    /// Automatic mode selecting the faster mechanism at warmup (Spec 6 §5.2).
    #[default]
    Auto,
    /// Sequential launch list replay (Spec 6 §5.2).
    List,
    /// Hardware hipGraph replay (Spec 6 §5.2).
    HipGraph,
}

/// Single step execution record in the diagnostic schedule log (Spec 6 §9).
///
/// Flushed to the doctor bundle on request or fault; read by Spec 11 for telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScheduleRecord {
    /// Step identifier (Spec 6 §9).
    #[serde(with = "serde_step_id")]
    pub step_id: StepId,
    /// Host pre-step duration in microseconds (Spec 6 §9).
    pub t_pre_us: u64,
    /// Proposer drafting duration in microseconds (Spec 6 §9).
    pub t_draft_us: u64,
    /// Device graph execution duration in microseconds (Spec 6 §9).
    pub t_device_us: u64,
    /// Host post-step duration in microseconds (Spec 6 §9).
    pub t_post_us: u64,
    /// Logical active sequence count S (Spec 6 §9).
    pub s: u32,
    /// Total logical decode tokens T_dec (Spec 6 §9).
    pub t_dec: u32,
    /// Total logical prefill tokens T_pre (Spec 6 §9).
    pub t_pre: u32,
    /// Prefill chunk size admitted in this step (Spec 6 §9).
    pub chunk: u32,
    /// Draft tokens per sequence (Spec 6 §9).
    pub k: InlineVec<u32, 1>,
    /// Accepted tokens per sequence (Spec 6 §9).
    pub accept_len: InlineVec<u32, 1>,
    /// Whether prefill was admitted exceeding budget due to max_wait_ms timeout (Spec 6 §4.1, §9).
    pub forced_admission: bool,
    /// Effective step budget in milliseconds (Spec 6 §4.3, §9).
    pub budget_ms: f32,
    /// Discrete shape bucket (S, T_dec, T_pre) (Spec 1 §3.5, Spec 6 §9).
    pub bucket: (u32, u32, u32),
    /// Replay mechanism used for this step (Spec 6 §5.2, §9).
    pub graph_mode: GraphMode,
    /// Whether this step triggered lazy graph capture (Spec 6 §5.1, §9).
    pub captured: bool,
    /// Exact sequence ID that encountered a reserve pause, reported on the next
    /// completed record after the pause and cleared thereafter (Spec 6 §6, §9).
    #[serde(with = "serde_paused_seqs")]
    pub paused: InlineVec<SeqId, 1>,
    /// Segment synchronization overhead in microseconds (Spec 6 §9).
    pub segment_sync_us: u64,
}

// DECISION(A3.9): schedule log ring buffer stores the most recent 4096 records in a fixed-capacity ring that overwrites oldest entries without heap reallocation; rejected unbounded growth or discarding recent records because Spec 6 §9 specifies an in-memory ring of the last 4096 steps.
/// Bounded ring buffer holding the most recent 4096 schedule records (Spec 6 §9).
#[derive(Debug, Clone)]
pub struct ScheduleLogRing {
    buffer: Vec<Option<ScheduleRecord>>,
    write_idx: usize,
    total_written: u64,
    capacity: usize,
}

impl Default for ScheduleLogRing {
    fn default() -> Self {
        Self::new(SCHEDULE_LOG_CAPACITY)
    }
}

impl ScheduleLogRing {
    /// Constructs a ring buffer with the specified capacity (Spec 6 §9).
    pub fn new(capacity: usize) -> Self {
        let cap = if capacity == 0 {
            SCHEDULE_LOG_CAPACITY
        } else {
            capacity
        };
        let mut buffer = Vec::with_capacity(cap);
        buffer.resize_with(cap, || None);
        Self {
            buffer,
            write_idx: 0,
            total_written: 0,
            capacity: cap,
        }
    }

    /// Appends a new schedule record into the ring, overwriting oldest entry on overflow (Spec 6 §9).
    pub fn push(&mut self, record: ScheduleRecord) -> SchedResult<()> {
        let next_total = self.total_written.checked_add(1).ok_or_else(|| {
            SchedError::overflow("total_written", "u64 total_written counter overflow")
        })?;
        let idx = self.write_idx;
        let slot = self
            .buffer
            .get_mut(idx)
            .ok_or_else(|| SchedError::Internal("schedule log write index".to_owned()))?;
        *slot = Some(record);
        self.write_idx = (self.write_idx + 1) % self.capacity;
        self.total_written = next_total;
        Ok(())
    }

    /// Returns the number of records currently stored in the ring buffer.
    pub fn len(&self) -> usize {
        if self.total_written < self.capacity as u64 {
            self.total_written as usize
        } else {
            self.capacity
        }
    }

    /// Returns `true` if no records are stored in the ring buffer.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total count of records pushed since initialization.
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Returns the fixed capacity of the ring buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns all stored records in strictly ascending chronological order (Spec 6 §9).
    pub fn to_vec(&self) -> Vec<ScheduleRecord> {
        let mut records = Vec::with_capacity(self.len());
        if self.total_written < self.capacity as u64 {
            let n = self.total_written as usize;
            for rec in self.buffer.iter().take(n).flatten() {
                records.push(rec.clone());
            }
        } else {
            for i in 0..self.capacity {
                let idx = (self.write_idx + i) % self.capacity;
                if let Some(rec) = self.buffer.get(idx).and_then(|o| o.as_ref()) {
                    records.push(rec.clone());
                }
            }
        }
        records
    }

    /// Clears all recorded schedule records from the ring buffer.
    pub fn clear(&mut self) {
        for slot in &mut self.buffer {
            *slot = None;
        }
        self.write_idx = 0;
        self.total_written = 0;
    }
}
