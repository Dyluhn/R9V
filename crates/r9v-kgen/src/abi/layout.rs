// SPDX-License-Identifier: Apache-2.0
//! Deterministic C ABI struct layout computation with checked arithmetic (Spec 4 §7).

use r9v_registry::OpId;

use crate::abi::batch_meta::BatchMetaField;
use crate::abi::types::{AbiField, AbiStruct, AbiType, FieldRole};
use crate::abi::workspace::WorkspaceSlot;
use crate::error::KgenError;

// DECISION(A3.2): ABI argument struct fields are placed in descending alignment order (8-byte pointers/u64 first, then 4-byte scalars); rejected arbitrary or insertion-order field placement because descending alignment guarantees zero internal padding holes and identical layout between Rust and HIP without compiler padding divergence. Spec 4 §7.

/// Specification for an unplaced field before layout computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSpec {
    /// Field identifier.
    pub name: String,
    /// Field data type.
    pub ty: AbiType,
    /// Semantic role of the field.
    pub role: FieldRole,
    /// Documentation comment describing the field.
    pub doc: String,
}

impl FieldSpec {
    /// Constructs a new field specification.
    pub fn new(
        name: impl Into<String>,
        ty: AbiType,
        role: FieldRole,
        doc: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            ty,
            role,
            doc: doc.into(),
        }
    }
}

/// Builder for constructing an [`AbiStruct`] with deterministic layout and checked arithmetic (Spec 4 §7).
#[derive(Debug, Clone)]
pub struct AbiStructBuilder {
    name: String,
    op: OpId,
    raw_fields: Vec<FieldSpec>,
    workspace_slots: Vec<WorkspaceSlot>,
    batch_meta_fields: Vec<BatchMetaField>,
}

impl AbiStructBuilder {
    /// Starts building an ABI struct for the given operation identifier.
    pub fn new(name: impl Into<String>, op: OpId) -> Self {
        Self {
            name: name.into(),
            op,
            raw_fields: Vec::new(),
            workspace_slots: Vec::new(),
            batch_meta_fields: Vec::new(),
        }
    }

    /// Appends a field specification to the builder.
    pub fn add_field(mut self, spec: FieldSpec) -> Self {
        self.raw_fields.push(spec);
        self
    }

    /// Registers a workspace slot and adds its corresponding device pointer field.
    pub fn add_workspace_slot(mut self, slot: WorkspaceSlot) -> Self {
        let field_name = slot.kind.default_field_name().to_string();
        let field_ty = slot.abi_type();
        let field_doc = slot.description.clone();
        self.workspace_slots.push(slot);
        self.raw_fields.push(FieldSpec::new(
            field_name,
            field_ty,
            FieldRole::Workspace,
            field_doc,
        ));
        self
    }

    /// Registers a BatchMeta field and adds its corresponding device pointer field.
    pub fn add_batch_meta_field(mut self, field: BatchMetaField) -> Self {
        let field_name = field.as_str().to_string();
        let field_ty = field.abi_type();
        let field_doc = field.doc().to_string();
        self.batch_meta_fields.push(field);
        self.raw_fields.push(FieldSpec::new(
            field_name,
            field_ty,
            field.role(),
            field_doc,
        ));
        self
    }

    /// Computes the layout deterministically using descending-alignment order and checked arithmetic (Spec 4 §7).
    pub fn build(self) -> Result<AbiStruct, KgenError> {
        let mut indexed_fields: Vec<(usize, FieldSpec)> =
            self.raw_fields.into_iter().enumerate().collect();

        // Sort by descending alignment requirement, preserving original declaration order as tie-breaker
        indexed_fields.sort_by(|(idx_a, a), (idx_b, b)| {
            b.ty.align_bytes()
                .cmp(&a.ty.align_bytes())
                .then_with(|| idx_a.cmp(idx_b))
        });

        let mut current_offset: usize = 0;
        let mut max_align: usize = 1;
        let mut placed_fields = Vec::with_capacity(indexed_fields.len());

        for (_, spec) in indexed_fields {
            let field_align = spec.ty.align_bytes();
            max_align = max_align.max(field_align);

            // Checked alignment padding computation
            let rem = current_offset % field_align;
            let pad = if rem == 0 { 0 } else { field_align - rem };

            let field_offset =
                current_offset
                    .checked_add(pad)
                    .ok_or(KgenError::ArithmeticOverflow {
                        context: "field alignment padding",
                        lhs: current_offset,
                        op: "+",
                        rhs: pad,
                    })?;

            let next_offset = field_offset.checked_add(spec.ty.size_bytes()).ok_or(
                KgenError::ArithmeticOverflow {
                    context: "field size addition",
                    lhs: field_offset,
                    op: "+",
                    rhs: spec.ty.size_bytes(),
                },
            )?;

            placed_fields.push(AbiField {
                name: spec.name,
                ty: spec.ty,
                role: spec.role,
                offset: field_offset,
                doc: spec.doc,
            });

            current_offset = next_offset;
        }

        // Struct total size is rounded up to multiple of max_align (Spec 4 §7 C ABI)
        let rem = current_offset % max_align;
        let tail_pad = if rem == 0 { 0 } else { max_align - rem };
        let total_size =
            current_offset
                .checked_add(tail_pad)
                .ok_or(KgenError::ArithmeticOverflow {
                    context: "struct tail padding",
                    lhs: current_offset,
                    op: "+",
                    rhs: tail_pad,
                })?;

        let abi_struct = AbiStruct {
            name: self.name,
            op: self.op,
            fields: placed_fields,
            size: total_size,
            alignment: max_align,
            workspace_slots: self.workspace_slots,
            batch_meta_fields: self.batch_meta_fields,
        };

        abi_struct.validate()?;
        Ok(abi_struct)
    }
}
