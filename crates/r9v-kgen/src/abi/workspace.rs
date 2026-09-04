// SPDX-License-Identifier: Apache-2.0
//! Workspace slot encoding per kernel operation (Spec 4 §7).

use r9v_ir::DType;
use serde::{Deserialize, Serialize};

use crate::abi::types::{AbiType, FieldRole, PointeeType};

/// Closed enum of fixed-size workspace slots owned by the scheduler's per-graph arena (Spec 4 §7, Spec 6).
///
/// Workspaces are fixed-size per bucket and owned by the scheduler arena, never allocated by a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceSlotKind {
    /// Split-K partial accumulator buffers for matmul GEMM (Spec 4 §5.1, §7).
    SplitKPartials,
    /// Split-KV partial (m, l, acc) buffers for decode split-KV attention (Spec 4 §5.3, §7).
    SplitKvPartials,
    /// MoE sorted expert-token pairs and prefix sums (Spec 4 §5.6, §7).
    MoeSortBuffers,
    /// Multi-rank collective staging buffer (Spec 4 §5.9, §7).
    CollectiveStaging,
    /// Sampling wave-level bitonic sort scratch in global memory (Spec 4 §5.8).
    BitonicSort,
    /// Linear attention recurrent carry buffer (Spec 4 §5.5).
    ScanCarry,
}

impl WorkspaceSlotKind {
    /// Canonical field name for the workspace pointer in the argument struct.
    pub const fn default_field_name(&self) -> &'static str {
        match self {
            Self::SplitKPartials => "workspace",
            Self::SplitKvPartials => "workspace",
            Self::MoeSortBuffers => "sort_workspace",
            Self::CollectiveStaging => "staging",
            Self::BitonicSort => "workspace",
            Self::ScanCarry => "workspace",
        }
    }

    /// Primary element data type for this workspace slot.
    pub const fn default_dtype(&self) -> DType {
        match self {
            Self::SplitKPartials => DType::F32,
            Self::SplitKvPartials => DType::F32,
            Self::MoeSortBuffers => DType::I8,
            Self::CollectiveStaging => DType::I8,
            Self::BitonicSort => DType::I8,
            Self::ScanCarry => DType::F32,
        }
    }

    /// Documentation comment describing this workspace slot.
    pub const fn description(&self) -> &'static str {
        match self {
            Self::SplitKPartials => "Split-K partial accumulator workspace from scheduler arena (Spec 4 §5.1, §7)",
            Self::SplitKvPartials => "Split-KV partial (m, l, acc) merge workspace from scheduler arena (Spec 4 §5.3, §7)",
            Self::MoeSortBuffers => "MoE expert sort and prefix sum workspace from scheduler arena (Spec 4 §5.6, §7)",
            Self::CollectiveStaging => "Inter-rank collective staging workspace from scheduler arena (Spec 4 §5.9, §7)",
            Self::BitonicSort => "Bitonic sort candidate workspace from scheduler arena (Spec 4 §5.8)",
            Self::ScanCarry => "Linear attention recurrent carry state workspace from scheduler arena (Spec 4 §5.5)",
        }
    }
}

/// Description of an arena-backed workspace slot required by a kernel variant (Spec 4 §7).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkspaceSlot {
    /// Kind of workspace buffer.
    pub kind: WorkspaceSlotKind,
    /// Slot index within the kernel's workspace requirements.
    pub slot_index: u32,
    /// Element data type.
    #[serde(with = "r9v_registry::serde_helpers::serde_dtype")]
    pub element_dtype: DType,
    /// Description and sizing rule note.
    pub description: String,
}

impl WorkspaceSlot {
    /// Constructs a workspace slot specification.
    pub fn new(kind: WorkspaceSlotKind, slot_index: u32) -> Self {
        Self {
            kind,
            slot_index,
            element_dtype: kind.default_dtype(),
            description: kind.description().to_string(),
        }
    }

    /// Constructs a workspace slot with a custom data type.
    pub fn with_dtype(kind: WorkspaceSlotKind, slot_index: u32, dtype: DType) -> Self {
        Self {
            kind,
            slot_index,
            element_dtype: dtype,
            description: kind.description().to_string(),
        }
    }

    /// Converts this workspace slot to an ABI argument pointer type (mutable device pointer, 256-byte aligned).
    pub fn abi_type(&self) -> AbiType {
        AbiType::mut_ptr(PointeeType::from_dtype(self.element_dtype))
    }

    /// Field role for this workspace slot.
    pub const fn role(&self) -> FieldRole {
        FieldRole::Workspace
    }
}
