// SPDX-License-Identifier: Apache-2.0
//! R9V kernel generator, cost model, search space, and leaf wrappers (Spec 4, Spec 14 §2).
//!
//! Card A3.2 delivers the canonical kernel ABI generator (Spec 4 §7).

pub mod abi;
pub mod error;

pub use abi::{
    abi, abi_for_op, canonical_struct_name, AbiField, AbiStruct, AbiType, BatchMetaField,
    FieldRole, PointeeType, ScalarType, WorkspaceSlot, WorkspaceSlotKind,
    KERNEL_PTR_ALIGNMENT_BYTES,
};
pub use error::KgenError;
