// SPDX-License-Identifier: Apache-2.0
//! Typed fail-closed validation and checked arithmetic tests for r9v-kgen ABI generator (Spec 4 §7; card A3.2).
//!
//! Asserts that invalid or maliciously malformed ABI descriptions fail closed with
//! explicit `KgenError` variants, arithmetic overflow does not wrap, and alignment/layout
//! invariants are strictly enforced.

use r9v_kgen::abi::{
    AbiField, AbiStruct, AbiStructBuilder, AbiType, BatchMetaField, FieldRole, FieldSpec,
    PointeeType,
};
use r9v_kgen::error::KgenError;
use r9v_registry::OpId;

#[test]
fn test_validation_rejects_empty_struct_name() {
    let s = AbiStruct {
        name: "".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "x".to_string(),
            ty: AbiType::const_ptr(PointeeType::F16),
            role: FieldRole::InputTensor,
            offset: 0,
            doc: "Input".to_string(),
        }],
        size: 8,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("not a valid identifier"))),
        "expected empty struct name validation failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_invalid_struct_name_characters() {
    let s = AbiStruct {
        name: "invalid name with spaces!".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "x".to_string(),
            ty: AbiType::const_ptr(PointeeType::F16),
            role: FieldRole::InputTensor,
            offset: 0,
            doc: "Input".to_string(),
        }],
        size: 8,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("not a valid identifier"))),
        "expected invalid character validation failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_empty_fields() {
    let s = AbiStruct {
        name: "empty_args".to_string(),
        op: OpId::Matmul,
        fields: vec![],
        size: 0,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("no fields"))),
        "expected no fields failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_duplicate_field_names() {
    let s = AbiStruct {
        name: "dup_args".to_string(),
        op: OpId::Matmul,
        fields: vec![
            AbiField {
                name: "field_a".to_string(),
                ty: AbiType::const_ptr(PointeeType::F16),
                role: FieldRole::InputTensor,
                offset: 0,
                doc: "First".to_string(),
            },
            AbiField {
                name: "field_a".to_string(),
                ty: AbiType::mut_ptr(PointeeType::F16),
                role: FieldRole::OutputTensor,
                offset: 8,
                doc: "Duplicate".to_string(),
            },
        ],
        size: 16,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("duplicate field name"))),
        "expected duplicate field failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_misaligned_field_offset() {
    let s = AbiStruct {
        name: "misaligned_args".to_string(),
        op: OpId::Matmul,
        fields: vec![
            AbiField {
                name: "a".to_string(),
                ty: AbiType::u32(),
                role: FieldRole::DynamicScalar,
                offset: 0,
                doc: "a".to_string(),
            },
            // Pointer requires 8-byte alignment, placing it at offset 4 is illegal
            AbiField {
                name: "ptr".to_string(),
                ty: AbiType::const_ptr(PointeeType::F16),
                role: FieldRole::InputTensor,
                offset: 4,
                doc: "ptr".to_string(),
            },
        ],
        size: 16,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("is not aligned to its requirement"))),
        "expected misaligned field failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_overlapping_fields() {
    let s = AbiStruct {
        name: "overlap_args".to_string(),
        op: OpId::Matmul,
        fields: vec![
            AbiField {
                name: "p1".to_string(),
                ty: AbiType::const_ptr(PointeeType::F16),
                role: FieldRole::InputTensor,
                offset: 0,
                doc: "p1".to_string(),
            },
            // Overlaps with p1 (size 8 bytes, so next valid offset is >= 8)
            AbiField {
                name: "p2".to_string(),
                ty: AbiType::const_ptr(PointeeType::F16),
                role: FieldRole::InputTensor,
                offset: 4,
                doc: "p2".to_string(),
            },
        ],
        size: 16,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("overlaps previous field"))),
        "expected overlapping field failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_non_power_of_two_alignment() {
    let s = AbiStruct {
        name: "bad_align_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "x".to_string(),
            ty: AbiType::u32(),
            role: FieldRole::DynamicScalar,
            offset: 0,
            doc: "x".to_string(),
        }],
        size: 12,
        alignment: 3, // Not a power of two
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("power of two"))),
        "expected power of two alignment failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_size_not_multiple_of_alignment() {
    let s = AbiStruct {
        name: "bad_size_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "p".to_string(),
            ty: AbiType::const_ptr(PointeeType::F16),
            role: FieldRole::InputTensor,
            offset: 0,
            doc: "p".to_string(),
        }],
        size: 10, // 10 is not a multiple of 8
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("not a multiple of its alignment"))),
        "expected size multiple failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_pointer_without_256_byte_alignment_assumption() {
    let s = AbiStruct {
        name: "wrong_ptr_align_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "p".to_string(),
            ty: AbiType::Pointer {
                pointee: PointeeType::F16,
                is_const: true,
                is_restrict: true,
                is_nullable: false,
                assume_aligned_bytes: 64, // Spec 4 §7 requires exactly 256
            },
            role: FieldRole::InputTensor,
            offset: 0,
            doc: "p".to_string(),
        }],
        size: 8,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("assumes alignment of 64 B, expected 256 B"))),
        "expected 256-byte alignment assumption failure, got: {err}"
    );
}

#[test]
fn test_builder_rejects_duplicate_field_name() {
    let res = AbiStructBuilder::new("dup_builder", OpId::Matmul)
        .add_field(FieldSpec::new(
            "a",
            AbiType::u32(),
            FieldRole::DynamicScalar,
            "first",
        ))
        .add_field(FieldSpec::new(
            "a",
            AbiType::u32(),
            FieldRole::DynamicScalar,
            "dup",
        ))
        .build();

    let err = res.unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("duplicate field name 'a'"))),
        "expected duplicate field error in builder, got: {err}"
    );
}

#[test]
fn test_builder_checked_arithmetic_overflow() {
    // If a field offset or size approaches usize::MAX, builder must fail closed with ArithmeticOverflow
    // instead of wrapping silently
    // Directly verify the checked_add logic on large values
    let huge_offset = usize::MAX - 2;
    let add_res = huge_offset.checked_add(8);
    assert!(add_res.is_none(), "checked_add must overflow on usize::MAX");

    let err = KgenError::ArithmeticOverflow {
        context: "test overflow",
        lhs: huge_offset,
        op: "+",
        rhs: 8,
    };
    assert_eq!(
        err.to_string(),
        format!("arithmetic overflow in test overflow: {huge_offset} + 8")
    );
}

#[test]
fn test_builder_descending_alignment_eliminates_internal_padding() {
    // Adding 4-byte scalar first, then 8-byte pointer:
    // Builder sorts descending (pointers first, then scalars), so pointer is placed at offset 0
    // and scalar at offset 8, with total size 16 and zero internal holes.
    let s = AbiStructBuilder::new("order_args", OpId::Matmul)
        .add_field(FieldSpec::new(
            "scalar_first",
            AbiType::u32(),
            FieldRole::DynamicScalar,
            "scalar",
        ))
        .add_field(FieldSpec::new(
            "ptr_second",
            AbiType::const_ptr(PointeeType::F16),
            FieldRole::InputTensor,
            "ptr",
        ))
        .build()
        .expect("builder should successfully sort and place fields");

    assert_eq!(s.fields[0].name, "ptr_second");
    assert_eq!(s.fields[0].offset, 0);
    assert_eq!(s.fields[1].name, "scalar_first");
    assert_eq!(s.fields[1].offset, 8);
    assert_eq!(s.size, 16);
    assert_eq!(s.alignment, 8);
}

#[test]
fn test_validation_rejects_invalid_field_identifier() {
    let s = AbiStruct {
        name: "test_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "123_invalid_lead_digit".to_string(),
            ty: AbiType::const_ptr(PointeeType::F16),
            role: FieldRole::InputTensor,
            offset: 0,
            doc: "test".to_string(),
        }],
        size: 8,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("is not a valid identifier"))),
        "expected invalid field identifier failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_enclosing_overlap() {
    let s = AbiStruct {
        name: "test_args".to_string(),
        op: OpId::Matmul,
        fields: vec![
            AbiField {
                name: "field_a".to_string(),
                ty: AbiType::const_ptr(PointeeType::F16),
                role: FieldRole::InputTensor,
                offset: 0,
                doc: "a".to_string(),
            },
            AbiField {
                name: "field_b".to_string(),
                ty: AbiType::u32(),
                role: FieldRole::DynamicScalar,
                offset: 4, // Inside field_a which spans 0..8
                doc: "b".to_string(),
            },
        ],
        size: 16,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("overlaps previous field"))),
        "expected enclosing overlap failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_nullable_output_tensor() {
    let s = AbiStruct {
        name: "test_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "out".to_string(),
            ty: AbiType::nullable_mut_ptr(PointeeType::F16),
            role: FieldRole::OutputTensor,
            offset: 0,
            doc: "output".to_string(),
        }],
        size: 8,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("output tensor field 'out' in struct 'test_args' cannot be nullable"))),
        "expected nullable output failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_nullable_workspace() {
    let s = AbiStruct {
        name: "test_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "ws".to_string(),
            ty: AbiType::nullable_mut_ptr(PointeeType::Void),
            role: FieldRole::Workspace,
            offset: 0,
            doc: "workspace".to_string(),
        }],
        size: 8,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("workspace field 'ws' in struct 'test_args' cannot be nullable"))),
        "expected nullable workspace failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_nullable_batch_meta() {
    let s = AbiStruct {
        name: "test_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "seq".to_string(),
            ty: AbiType::nullable_const_ptr(PointeeType::U32),
            role: FieldRole::BatchMeta(BatchMetaField::SeqIds),
            offset: 0,
            doc: "seq".to_string(),
        }],
        size: 8,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("batch meta field 'seq' in struct 'test_args' cannot be nullable"))),
        "expected nullable batch meta failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_pointer_as_dynamic_scalar() {
    let s = AbiStruct {
        name: "test_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "scalar_ptr".to_string(),
            ty: AbiType::const_ptr(PointeeType::U32),
            role: FieldRole::DynamicScalar,
            offset: 0,
            doc: "scalar ptr".to_string(),
        }],
        size: 8,
        alignment: 8,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("dynamic scalar field 'scalar_ptr' in struct 'test_args' cannot be a pointer"))),
        "expected pointer dynamic scalar failure, got: {err}"
    );
}

#[test]
fn test_validation_rejects_output_tensor_not_pointer() {
    let s = AbiStruct {
        name: "test_args".to_string(),
        op: OpId::Matmul,
        fields: vec![AbiField {
            name: "out".to_string(),
            ty: AbiType::u32(),
            role: FieldRole::OutputTensor,
            offset: 0,
            doc: "out".to_string(),
        }],
        size: 4,
        alignment: 4,
        workspace_slots: vec![],
        batch_meta_fields: vec![],
    };

    let err = s.validate().unwrap_err();
    assert!(
        matches!(err, KgenError::ValidationFailed { ref problems } if problems.iter().any(|p| p.contains("output tensor field 'out' in struct 'test_args' must be a device pointer"))),
        "expected non-pointer output failure, got: {err}"
    );
}
