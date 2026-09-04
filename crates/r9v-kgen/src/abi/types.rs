// SPDX-License-Identifier: Apache-2.0
//! Core types and representation for the kernel argument ABI (Spec 4 §7).

use std::fmt;

use r9v_ir::DType;
use r9v_registry::OpId;
use serde::{Deserialize, Serialize};

use crate::abi::batch_meta::BatchMetaField;
use crate::abi::workspace::WorkspaceSlot;
use crate::error::KgenError;

/// Canonical alignment in bytes assumed for all kernel device pointer arguments (Spec 4 §7).
///
/// Every device pointer in a kernel argument struct is assumed to be 256-byte aligned.
/// The generator emits `__builtin_assume_aligned(ptr, KERNEL_PTR_ALIGNMENT_BYTES)` for HIP.
// DECISION(A3.2): Device pointers encode explicit 256-byte alignment assumption constants and HIP __builtin_assume_aligned emission; rejected unannotated raw pointers because the memory hierarchy and vector loads require 256-byte alignment guarantees. Spec 4 §7.
pub const KERNEL_PTR_ALIGNMENT_BYTES: usize = 256;

/// Closed enum of data types pointed to by pointer arguments in the kernel ABI (Spec 4 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointeeType {
    /// Untyped byte/void pointer (`const void*` or `void*`).
    Void,
    /// Unsigned 8-bit integer (`uint8_t`).
    U8,
    /// Signed 8-bit integer (`int8_t`).
    I8,
    /// Unsigned 16-bit integer (`uint16_t`).
    U16,
    /// Signed 16-bit integer (`int16_t`).
    I16,
    /// 16-bit IEEE half-precision float (`__half`).
    F16,
    /// 16-bit brain float (`__hip_bfloat16`).
    BF16,
    /// 32-bit single-precision float (`float`).
    F32,
    /// Unsigned 32-bit integer (`uint32_t`).
    U32,
    /// Signed 32-bit integer (`int32_t`).
    I32,
    /// Unsigned 64-bit integer (`uint64_t`).
    U64,
    /// Signed 64-bit integer (`int64_t`).
    I64,
}

impl PointeeType {
    /// Returns the element size in bytes for this pointee type.
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::Void | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::U64 | Self::I64 => 8,
        }
    }

    /// Returns the C++ / HIP type name for this pointee type.
    pub const fn hip_type_name(&self) -> &'static str {
        match self {
            Self::Void => "void",
            Self::U8 => "uint8_t",
            Self::I8 => "int8_t",
            Self::U16 => "uint16_t",
            Self::I16 => "int16_t",
            Self::F16 => "__half",
            Self::BF16 => "hip_bfloat16",
            Self::F32 => "float",
            Self::U32 => "uint32_t",
            Self::I32 => "int32_t",
            Self::U64 => "uint64_t",
            Self::I64 => "int64_t",
        }
    }

    /// Returns the Rust type name for this pointee type.
    pub const fn rust_type_name(&self) -> &'static str {
        match self {
            Self::Void => "std::ffi::c_void",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::F16 => "u16",
            Self::BF16 => "u16",
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
        }
    }

    /// Maps an IR `DType` to its corresponding `PointeeType`.
    pub const fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self::F32,
            DType::F16 => Self::F16,
            DType::Bf16 => Self::BF16,
            DType::I8 => Self::I8,
            DType::I4 => Self::U8,
            DType::I32 => Self::I32,
            DType::U32 => Self::U32,
            DType::Bool => Self::U8,
            DType::E4m3 => Self::U8,
            DType::E5m2 => Self::U8,
        }
    }
}

impl fmt::Display for PointeeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.hip_type_name())
    }
}

/// Closed enum of scalar value types passed by value in the kernel ABI (Spec 4 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarType {
    /// Unsigned 32-bit integer (`uint32_t` / `u32`).
    U32,
    /// Signed 32-bit integer (`int32_t` / `i32`).
    I32,
    /// Unsigned 64-bit integer (`uint64_t` / `u64`).
    U64,
    /// Signed 64-bit integer (`int64_t` / `i64`).
    I64,
    /// 32-bit single-precision float (`float` / `f32`).
    F32,
}

impl ScalarType {
    /// Size in bytes for this scalar type.
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 => 8,
        }
    }

    /// Alignment requirement in bytes for this scalar type.
    pub const fn align_bytes(&self) -> usize {
        self.size_bytes()
    }

    /// C++ / HIP type name for this scalar type.
    pub const fn hip_type_name(&self) -> &'static str {
        match self {
            Self::U32 => "uint32_t",
            Self::I32 => "int32_t",
            Self::U64 => "uint64_t",
            Self::I64 => "int64_t",
            Self::F32 => "float",
        }
    }

    /// Rust type name for this scalar type.
    pub const fn rust_type_name(&self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F32 => "f32",
        }
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.hip_type_name())
    }
}

/// Closed enum representing field types in an argument struct (Spec 4 §7).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AbiType {
    /// Device pointer passed by value with 256-byte alignment assumption (Spec 4 §7).
    Pointer {
        /// Pointee data type.
        pointee: PointeeType,
        /// Const qualifier on pointer target (`const T*`).
        is_const: bool,
        /// Restrict qualifier on pointer (`T* __restrict__`).
        is_restrict: bool,
        /// Whether the pointer argument is nullable (`None` / null).
        is_nullable: bool,
        /// Assumed pointer target alignment in bytes (Spec 4 §7 requires 256).
        assume_aligned_bytes: usize,
    },
    /// Scalar value passed by value (Spec 4 §7).
    Scalar(ScalarType),
}

impl AbiType {
    /// Constructs a const device pointer with canonical 256-byte alignment.
    pub const fn const_ptr(pointee: PointeeType) -> Self {
        Self::Pointer {
            pointee,
            is_const: true,
            is_restrict: true,
            is_nullable: false,
            assume_aligned_bytes: KERNEL_PTR_ALIGNMENT_BYTES,
        }
    }

    /// Constructs a nullable const device pointer with canonical 256-byte alignment.
    pub const fn nullable_const_ptr(pointee: PointeeType) -> Self {
        Self::Pointer {
            pointee,
            is_const: true,
            is_restrict: true,
            is_nullable: true,
            assume_aligned_bytes: KERNEL_PTR_ALIGNMENT_BYTES,
        }
    }

    /// Constructs a mutable device pointer with canonical 256-byte alignment.
    pub const fn mut_ptr(pointee: PointeeType) -> Self {
        Self::Pointer {
            pointee,
            is_const: false,
            is_restrict: true,
            is_nullable: false,
            assume_aligned_bytes: KERNEL_PTR_ALIGNMENT_BYTES,
        }
    }

    /// Constructs a nullable mutable device pointer with canonical 256-byte alignment.
    pub const fn nullable_mut_ptr(pointee: PointeeType) -> Self {
        Self::Pointer {
            pointee,
            is_const: false,
            is_restrict: true,
            is_nullable: true,
            assume_aligned_bytes: KERNEL_PTR_ALIGNMENT_BYTES,
        }
    }

    /// Constructs a 32-bit unsigned scalar.
    pub const fn u32() -> Self {
        Self::Scalar(ScalarType::U32)
    }

    /// Constructs a 32-bit signed scalar.
    pub const fn i32() -> Self {
        Self::Scalar(ScalarType::I32)
    }

    /// Constructs a 64-bit unsigned scalar.
    pub const fn u64() -> Self {
        Self::Scalar(ScalarType::U64)
    }

    /// Constructs a 32-bit float scalar.
    pub const fn f32() -> Self {
        Self::Scalar(ScalarType::F32)
    }

    /// Size in bytes for this field type under 64-bit C ABI.
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::Pointer { .. } => 8,
            Self::Scalar(s) => s.size_bytes(),
        }
    }

    /// Alignment requirement in bytes for this field type under 64-bit C ABI.
    pub const fn align_bytes(&self) -> usize {
        match self {
            Self::Pointer { .. } => 8,
            Self::Scalar(s) => s.align_bytes(),
        }
    }

    /// True if this type is a pointer.
    pub const fn is_pointer(&self) -> bool {
        matches!(self, Self::Pointer { .. })
    }

    /// True if this type is a nullable pointer.
    pub const fn is_nullable(&self) -> bool {
        match self {
            Self::Pointer { is_nullable, .. } => *is_nullable,
            Self::Scalar(_) => false,
        }
    }

    /// Assumed alignment in bytes if this is a pointer with an alignment assumption.
    pub const fn assume_aligned_bytes(&self) -> Option<usize> {
        match self {
            Self::Pointer {
                assume_aligned_bytes,
                ..
            } => Some(*assume_aligned_bytes),
            Self::Scalar(_) => None,
        }
    }
}

/// Semantic role of a field in a kernel argument struct (Spec 4 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldRole {
    /// Primary input tensor data.
    InputTensor,
    /// Primary output tensor data.
    OutputTensor,
    /// Weight tensor data.
    Weight,
    /// Weight quantization scales.
    WeightScale,
    /// Weight sparsity indices (e.g. SWMMAC 2:4).
    WeightIndices,
    /// Activation quantization scales.
    ActivationScale,
    /// Bias tensor data.
    Bias,
    /// Residual tensor data.
    Residual,
    /// Arena-backed workspace buffer.
    Workspace,
    /// Batch metadata field projection.
    BatchMeta(BatchMetaField),
    /// Dynamic runtime scalar (token count, step, sequence count).
    DynamicScalar,
}

/// Description of a single field within an ABI argument struct (Spec 4 §7).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AbiField {
    /// Field identifier in snake_case.
    pub name: String,
    /// Field data type.
    pub ty: AbiType,
    /// Semantic role of the field.
    pub role: FieldRole,
    /// Byte offset within the struct, calculated deterministically.
    pub offset: usize,
    /// Documentation comment describing the field purpose.
    pub doc: String,
}

impl AbiField {
    /// Returns the field name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the field data type.
    pub fn ty(&self) -> &AbiType {
        &self.ty
    }

    /// Returns the field semantic role.
    pub fn role(&self) -> FieldRole {
        self.role
    }

    /// Returns the byte offset of this field within the struct.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the field size in bytes.
    pub fn size(&self) -> usize {
        self.ty.size_bytes()
    }

    /// Returns the field alignment in bytes.
    pub fn align(&self) -> usize {
        self.ty.align_bytes()
    }

    /// Returns the documentation description.
    pub fn doc(&self) -> &str {
        &self.doc
    }
}

/// Canonical per-op argument struct description (Spec 4 §7).
///
/// Single source of truth from which both Rust-side and HIP-side struct definitions
/// are deterministically generated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiStruct {
    /// Struct identifier name (e.g. `matmul_args` or `matmul_7f3a9c_args`).
    pub name: String,
    /// Operation identifier.
    pub op: OpId,
    /// Ordered list of fields in the struct.
    pub fields: Vec<AbiField>,
    /// Total struct size in bytes, including tail padding.
    pub size: usize,
    /// Struct alignment in bytes.
    pub alignment: usize,
    /// Workspace slots required by this kernel variant.
    pub workspace_slots: Vec<WorkspaceSlot>,
    /// BatchMeta fields selected by this kernel variant.
    pub batch_meta_fields: Vec<BatchMetaField>,
}

impl AbiStruct {
    /// Returns the struct name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the operation identifier.
    pub fn op(&self) -> OpId {
        self.op
    }

    /// Returns the slice of fields.
    pub fn fields(&self) -> &[AbiField] {
        &self.fields
    }

    /// Looks up a field by name.
    pub fn field(&self, name: &str) -> Option<&AbiField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Total size in bytes of the struct.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Alignment requirement in bytes of the struct.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Workspace slots required by this op.
    pub fn workspace_slots(&self) -> &[WorkspaceSlot] {
        &self.workspace_slots
    }

    /// BatchMeta fields required by this op.
    pub fn batch_meta_fields(&self) -> &[BatchMetaField] {
        &self.batch_meta_fields
    }

    /// Validates the struct representation using fail-closed collect-all pattern (CONVENTIONS §1.4).
    pub fn validate(&self) -> Result<(), KgenError> {
        let mut problems = Vec::new();

        if !is_valid_ident(&self.name) {
            problems.push(format!(
                "struct name '{}' is not a valid identifier",
                self.name
            ));
        }
        if self.fields.is_empty() {
            problems.push(format!("struct '{}' has no fields", self.name));
        }

        // Validate field names are unique and offsets are strictly non-overlapping and aligned
        let mut seen_names = std::collections::BTreeSet::new();

        for (i, f) in self.fields.iter().enumerate() {
            if !seen_names.insert(&f.name) {
                problems.push(format!(
                    "duplicate field name '{}' in struct '{}'",
                    f.name, self.name
                ));
            }
            if !is_valid_ident(&f.name) {
                problems.push(format!(
                    "field {} ('{}') in struct '{}' is not a valid identifier",
                    i, f.name, self.name
                ));
            }

            let field_align = f.ty.align_bytes();
            if f.offset % field_align != 0 {
                problems.push(format!(
                    "field '{}' at offset {} is not aligned to its requirement of {} bytes",
                    f.name, f.offset, field_align
                ));
            }

            // Pointer checks: natural C pointer alignment in struct vs 256-byte target alignment
            if f.ty.is_pointer() {
                if field_align != 8 {
                    problems.push(format!(
                        "pointer field '{}' in struct '{}' has struct alignment {} B, expected natural C pointer alignment 8 B",
                        f.name, field_align, self.name
                    ));
                }
                if let Some(align_assumed) = f.ty.assume_aligned_bytes() {
                    if align_assumed != KERNEL_PTR_ALIGNMENT_BYTES {
                        problems.push(format!(
                            "pointer field '{}' assumes alignment of {} B, expected {} B per Spec 4 §7",
                            f.name, align_assumed, KERNEL_PTR_ALIGNMENT_BYTES
                        ));
                    }
                }
            }

            // Role and nullability consistency checks
            match f.role {
                FieldRole::DynamicScalar => {
                    if f.ty.is_pointer() {
                        problems.push(format!(
                            "dynamic scalar field '{}' in struct '{}' cannot be a pointer",
                            f.name, self.name
                        ));
                    }
                    if f.ty.is_nullable() {
                        problems.push(format!(
                            "dynamic scalar field '{}' in struct '{}' cannot be nullable",
                            f.name, self.name
                        ));
                    }
                }
                FieldRole::OutputTensor => {
                    if f.ty.is_nullable() {
                        problems.push(format!(
                            "output tensor field '{}' in struct '{}' cannot be nullable",
                            f.name, self.name
                        ));
                    }
                    if !f.ty.is_pointer() {
                        problems.push(format!(
                            "output tensor field '{}' in struct '{}' must be a device pointer",
                            f.name, self.name
                        ));
                    }
                }
                FieldRole::Workspace => {
                    if f.ty.is_nullable() {
                        problems.push(format!(
                            "workspace field '{}' in struct '{}' cannot be nullable",
                            f.name, self.name
                        ));
                    }
                    if !f.ty.is_pointer() {
                        problems.push(format!(
                            "workspace field '{}' in struct '{}' must be a device pointer",
                            f.name, self.name
                        ));
                    }
                }
                FieldRole::BatchMeta(_) => {
                    if f.ty.is_nullable() {
                        problems.push(format!(
                            "batch meta field '{}' in struct '{}' cannot be nullable",
                            f.name, self.name
                        ));
                    }
                    if !f.ty.is_pointer() {
                        problems.push(format!(
                            "batch meta field '{}' in struct '{}' must be a device pointer",
                            f.name, self.name
                        ));
                    }
                }
                _ => {}
            }

            if !f.ty.is_pointer() && f.role != FieldRole::DynamicScalar {
                problems.push(format!(
                    "scalar field '{}' in struct '{}' must have DynamicScalar role",
                    f.name, self.name
                ));
            }
        }

        // Catch enclosing overlaps using max-end on sorted intervals
        let mut spans = Vec::with_capacity(self.fields.len());
        for f in &self.fields {
            match f.offset.checked_add(f.ty.size_bytes()) {
                Some(end) => spans.push((f.offset, end, &f.name)),
                None => problems.push(format!(
                    "field '{}' offset overflow in struct '{}'",
                    f.name, self.name
                )),
            }
        }
        spans.sort_by_key(|s| s.0);
        let mut max_end = 0usize;
        for (start, end, name) in spans {
            if start < max_end {
                problems.push(format!(
                    "field '{}' at offset {}..{} overlaps previous field ending at {} in struct '{}'",
                    name, start, end, max_end, self.name
                ));
            }
            max_end = max_end.max(end);
        }

        if self.alignment == 0 || (self.alignment & (self.alignment - 1)) != 0 {
            problems.push(format!(
                "struct alignment {} is not a power of two",
                self.alignment
            ));
        }

        if self.alignment > 0 && !self.size.is_multiple_of(self.alignment) {
            problems.push(format!(
                "struct size {} is not a multiple of its alignment {}",
                self.size, self.alignment
            ));
        }

        if self.size < max_end {
            problems.push(format!(
                "struct size {} is smaller than total field extent {}",
                self.size, max_end
            ));
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(KgenError::ValidationFailed { problems })
        }
    }
}

fn is_valid_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {
            chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        _ => false,
    }
}
