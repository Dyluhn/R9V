// SPDX-License-Identifier: Apache-2.0
//! Checked scratch workspace arena for step graph execution (Spec 6 §5.3).

use crate::error::{SchedError, SchedResult};

/// Mandatory pointer alignment for kernel ABI workspaces (Spec 4 §7).
pub const WORKSPACE_ALIGNMENT: u64 = 256;

// DECISION(A3.9): workspace arena enforces 256-byte alignment and validates capacity bounds with checked arithmetic; lazy capture may grow the arena when a larger bucket requires it, forcing graph recapture with an incremented arena generation. Spec 6 §5.3.
/// Checked scratch workspace arena sized to the maximum workspace over captured buckets (Spec 6 §5.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceArena {
    rank: u32,
    capacity_bytes: u64,
    allocated_bytes: u64,
    peak_bytes: u64,
    generation: u64,
}

impl WorkspaceArena {
    /// Constructs a checked workspace arena with the given capacity on the specified device rank (Spec 6 §5.3).
    pub fn new(rank: u32, capacity_bytes: u64) -> Self {
        Self {
            rank,
            capacity_bytes,
            allocated_bytes: 0,
            peak_bytes: 0,
            generation: 1,
        }
    }

    /// Returns the device rank bound to this arena (Spec 6 §5.3).
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Returns total capacity in bytes (Spec 6 §5.3).
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Returns bytes currently reserved or bound (Spec 6 §5.3).
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Returns available unallocated bytes within capacity.
    pub fn available_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Returns peak allocated bytes observed since construction.
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// Returns the current arena generation, incremented whenever arena capacity grows (Spec 6 §5.3).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Allocates an aligned slice from the arena, returning its byte offset (Spec 4 §7, Spec 6 §5.3).
    pub fn allocate_slice(&mut self, size_bytes: u64) -> SchedResult<u64> {
        let align_mask = WORKSPACE_ALIGNMENT
            .checked_sub(1)
            .ok_or_else(|| SchedError::overflow("arena_alignment", "invalid alignment"))?;
        let aligned_offset = match self.allocated_bytes.checked_add(align_mask) {
            Some(sum) => sum & !align_mask,
            None => {
                return Err(SchedError::overflow(
                    "arena_alignment",
                    format!(
                        "allocated={} align_mask={}",
                        self.allocated_bytes, align_mask
                    ),
                ))
            }
        };

        let new_allocated = match aligned_offset.checked_add(size_bytes) {
            Some(sum) => sum,
            None => {
                return Err(SchedError::overflow(
                    "arena_slice",
                    format!("offset={} size={}", aligned_offset, size_bytes),
                ))
            }
        };

        if new_allocated > self.capacity_bytes {
            let shortfall = new_allocated.saturating_sub(self.capacity_bytes);
            return Err(SchedError::ArenaOverflow {
                required: new_allocated,
                available: self.capacity_bytes,
                shortfall,
            });
        }

        self.allocated_bytes = new_allocated;
        if new_allocated > self.peak_bytes {
            self.peak_bytes = new_allocated;
        }

        Ok(aligned_offset)
    }

    /// Validates that the arena can accommodate a graph requiring `required_bytes` (Spec 6 §5.3).
    pub fn check_requirement(&self, required_bytes: u64) -> SchedResult<()> {
        if required_bytes > self.capacity_bytes {
            let shortfall = required_bytes.saturating_sub(self.capacity_bytes);
            return Err(SchedError::ArenaOverflow {
                required: required_bytes,
                available: self.capacity_bytes,
                shortfall,
            });
        }
        Ok(())
    }

    /// Resets the per-step bump offset back to zero between steps (Spec 6 §5.3).
    ///
    /// Captured graphs bind fixed offsets checked against capacity, so this only
    /// resets transient bump state tracked by [`WorkspaceArena::allocate_slice`].
    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
    }

    /// Grows the arena capacity to at least `new_capacity_bytes` aligned to 256 bytes,
    /// incrementing the arena generation (Spec 6 §5.3).
    pub fn grow(&mut self, new_capacity_bytes: u64) -> SchedResult<()> {
        if new_capacity_bytes <= self.capacity_bytes {
            return Ok(());
        }
        let align_mask = WORKSPACE_ALIGNMENT
            .checked_sub(1)
            .ok_or_else(|| SchedError::overflow("arena_growth", "invalid alignment"))?;
        let aligned_cap = match new_capacity_bytes.checked_add(align_mask) {
            Some(sum) => sum & !align_mask,
            None => {
                return Err(SchedError::overflow(
                    "arena_growth",
                    format!("overflow aligning new capacity {new_capacity_bytes}"),
                ))
            }
        };
        self.capacity_bytes = aligned_cap;
        self.generation = self.generation.checked_add(1).ok_or_else(|| {
            SchedError::overflow("arena_generation", "generation counter overflow")
        })?;
        Ok(())
    }
}
