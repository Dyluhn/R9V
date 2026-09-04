// SPDX-License-Identifier: Apache-2.0
//! API-shape tests for r9v-kgen (Spec 4 §1, §7; card A3.2).
//!
//! Asserts compilation, visibility boundaries, and trait markers (`Send`, `Sync`,
//! etc.) that downstream crates rely on. Closed-set enums are matched exhaustively
//! (without wildcard arms) so any RFC addition breaks this test until explicitly handled.

use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;

mod common;

use r9v_kgen::abi::{
    abi, abi_for_op, canonical_struct_name, op_static_family, AbiField, AbiStruct,
    AbiStructBuilder, AbiType, BatchMetaField, FieldRole, FieldSpec, PointeeType, ScalarType,
    WorkspaceSlot, WorkspaceSlotKind, KERNEL_PTR_ALIGNMENT_BYTES,
};
use r9v_kgen::error::KgenError;
use r9v_registry::OpId;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_copy<T: Copy>() {}
fn assert_clone<T: Clone>() {}
fn assert_debug<T: Debug>() {}
fn assert_display<T: Display>() {}
fn assert_hash<T: Hash>() {}
fn assert_error<T: Error>() {}
fn assert_eq_trait<T: PartialEq + Eq>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn test_api_shape_trait_markers() {
    // ABI core types
    assert_send::<AbiStruct>();
    assert_sync::<AbiStruct>();
    assert_send_sync::<AbiStruct>();
    assert_clone::<AbiStruct>();
    assert_debug::<AbiStruct>();
    assert_eq_trait::<AbiStruct>();

    assert_send_sync::<AbiField>();
    assert_clone::<AbiField>();
    assert_debug::<AbiField>();
    assert_eq_trait::<AbiField>();

    assert_send_sync::<AbiType>();
    assert_clone::<AbiType>();
    assert_debug::<AbiType>();
    assert_eq_trait::<AbiType>();

    assert_send_sync::<FieldRole>();
    assert_copy::<FieldRole>();
    assert_clone::<FieldRole>();
    assert_debug::<FieldRole>();
    assert_hash::<FieldRole>();
    assert_eq_trait::<FieldRole>();

    assert_send_sync::<PointeeType>();
    assert_copy::<PointeeType>();
    assert_clone::<PointeeType>();
    assert_debug::<PointeeType>();
    assert_hash::<PointeeType>();
    assert_eq_trait::<PointeeType>();

    assert_send_sync::<ScalarType>();
    assert_copy::<ScalarType>();
    assert_clone::<ScalarType>();
    assert_debug::<ScalarType>();
    assert_hash::<ScalarType>();
    assert_eq_trait::<ScalarType>();

    assert_send_sync::<BatchMetaField>();
    assert_copy::<BatchMetaField>();
    assert_clone::<BatchMetaField>();
    assert_debug::<BatchMetaField>();
    assert_hash::<BatchMetaField>();
    assert_eq_trait::<BatchMetaField>();

    assert_send_sync::<WorkspaceSlot>();
    assert_clone::<WorkspaceSlot>();
    assert_debug::<WorkspaceSlot>();
    assert_eq_trait::<WorkspaceSlot>();

    assert_send_sync::<WorkspaceSlotKind>();
    assert_copy::<WorkspaceSlotKind>();
    assert_clone::<WorkspaceSlotKind>();
    assert_debug::<WorkspaceSlotKind>();
    assert_hash::<WorkspaceSlotKind>();
    assert_eq_trait::<WorkspaceSlotKind>();

    assert_send_sync::<AbiStructBuilder>();
    assert_clone::<AbiStructBuilder>();
    assert_debug::<AbiStructBuilder>();

    assert_send_sync::<FieldSpec>();
    assert_clone::<FieldSpec>();
    assert_debug::<FieldSpec>();

    // Error type
    assert_send_sync::<KgenError>();
    assert_debug::<KgenError>();
    assert_display::<KgenError>();
    assert_error::<KgenError>();

    // Alignment constant
    assert_eq!(KERNEL_PTR_ALIGNMENT_BYTES, 256);
}

#[test]
fn test_api_shape_all_32_ops_construct() {
    for op in common::ALL_32_OPS {
        let st = common::representative_static_for_op(op);
        let abi = abi_for_op(op, &st).unwrap_or_else(|e| panic!("failed for {op}: {e}"));
        assert_eq!(abi.op(), op);
        assert!(!abi.name().is_empty());
        assert!(!abi.fields().is_empty());
        assert_eq!(abi.size() % abi.alignment(), 0);
    }
}

#[test]
fn test_api_shape_closed_sets_exhaustive() {
    // Exhaustive match on FieldRole: adding a variant fails compilation until handled
    let role = FieldRole::InputTensor;
    let _ = match role {
        FieldRole::InputTensor => 0,
        FieldRole::OutputTensor => 1,
        FieldRole::Weight => 2,
        FieldRole::WeightScale => 3,
        FieldRole::WeightIndices => 4,
        FieldRole::ActivationScale => 5,
        FieldRole::Bias => 6,
        FieldRole::Residual => 7,
        FieldRole::Workspace => 8,
        FieldRole::BatchMeta(_) => 9,
        FieldRole::DynamicScalar => 10,
    };

    // Exhaustive match on PointeeType
    let pt = PointeeType::Void;
    let _ = match pt {
        PointeeType::Void => 0,
        PointeeType::U8 => 1,
        PointeeType::I8 => 2,
        PointeeType::U16 => 3,
        PointeeType::I16 => 4,
        PointeeType::F16 => 5,
        PointeeType::BF16 => 6,
        PointeeType::F32 => 7,
        PointeeType::U32 => 8,
        PointeeType::I32 => 9,
        PointeeType::U64 => 10,
        PointeeType::I64 => 11,
    };

    // Exhaustive match on ScalarType
    let st = ScalarType::U32;
    let _ = match st {
        ScalarType::U32 => 0,
        ScalarType::I32 => 1,
        ScalarType::U64 => 2,
        ScalarType::I64 => 3,
        ScalarType::F32 => 4,
    };

    // Exhaustive match on BatchMetaField
    let bmf = BatchMetaField::SeqIds;
    let _ = match bmf {
        BatchMetaField::SeqIds => 0,
        BatchMetaField::QueryLen => 1,
        BatchMetaField::CtxLen => 2,
        BatchMetaField::Positions => 3,
        BatchMetaField::SlotMap => 4,
        BatchMetaField::BlockTable => 5,
        BatchMetaField::WindowStart => 6,
        BatchMetaField::TreeParents => 7,
        BatchMetaField::TreeAncestors => 8,
    };

    // Exhaustive match on WorkspaceSlotKind
    let wsk = WorkspaceSlotKind::SplitKPartials;
    let _ = match wsk {
        WorkspaceSlotKind::SplitKPartials => 0,
        WorkspaceSlotKind::SplitKvPartials => 1,
        WorkspaceSlotKind::MoeSortBuffers => 2,
        WorkspaceSlotKind::CollectiveStaging => 3,
        WorkspaceSlotKind::BitonicSort => 4,
        WorkspaceSlotKind::ScanCarry => 5,
    };

    // Exhaustive match on KgenError
    let err = KgenError::EmptyStruct { op: OpId::Matmul };
    let _ = match err {
        KgenError::ValidationFailed { .. } => 0,
        KgenError::ArithmeticOverflow { .. } => 1,
        KgenError::UnsupportedOp { .. } => 2,
        KgenError::AlignmentError { .. } => 3,
        KgenError::EmptyStruct { .. } => 4,
        KgenError::CompileError { .. } => 5,
        KgenError::LayoutMismatch { .. } => 6,
        KgenError::MismatchedOpFamily { .. } => 7,
        KgenError::AmbiguousOpFamily { .. } => 8,
        KgenError::InconsistentVariantCollision { .. } => 9,
        KgenError::Ir(_) => 10,
        KgenError::Registry(_) => 11,
        KgenError::Io(_) => 12,
    };
}

#[test]
fn test_api_shape_public_constructors_reachable() {
    let matmul_st = common::representative_matmul_static();
    let matmul_abi = abi_for_op(OpId::Matmul, &matmul_st).expect("matmul abi builds");
    assert_eq!(matmul_abi.op(), OpId::Matmul);
    assert_eq!(
        matmul_abi.name(),
        canonical_struct_name(OpId::Matmul, &matmul_st)
    );
    assert_eq!(op_static_family(&matmul_st), "matmul");

    // Direct abi(&OpStatic) works for unique family Matmul
    let direct_matmul = abi(&matmul_st).expect("unique family dispatches directly");
    assert_eq!(direct_matmul.op(), OpId::Matmul);
}
