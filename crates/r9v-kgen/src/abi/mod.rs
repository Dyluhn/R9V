// SPDX-License-Identifier: Apache-2.0
//! Kernel ABI definitions, types, and generators (Spec 4 §7).

pub mod batch_meta;
pub mod emitter;
pub mod layout;
pub mod per_op;
pub mod types;
pub mod workspace;

pub use batch_meta::BatchMetaField;
pub use emitter::{
    emit_all_hip_header, emit_all_rust_module, emit_hip_assume_aligned, emit_hip_struct,
    emit_rust_struct,
};
pub use layout::{AbiStructBuilder, FieldSpec};
pub use per_op::{abi, abi_for_op, canonical_struct_name, op_static_family};
pub use types::{
    AbiField, AbiStruct, AbiType, FieldRole, PointeeType, ScalarType, KERNEL_PTR_ALIGNMENT_BYTES,
};
pub use workspace::{WorkspaceSlot, WorkspaceSlotKind};
