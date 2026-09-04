// SPDX-License-Identifier: Apache-2.0
//! Canonical per-op ABI descriptions for every closed-set op family (Spec 1 §4, Spec 4 §3, §7).

use r9v_ir::AttentionMask;
use r9v_registry::{OpId, OpStatic};

use crate::abi::batch_meta::BatchMetaField;
use crate::abi::layout::{AbiStructBuilder, FieldSpec};
use crate::abi::types::{AbiStruct, AbiType, FieldRole, PointeeType};
use crate::abi::workspace::{WorkspaceSlot, WorkspaceSlotKind};
use crate::error::KgenError;

// DECISION(A3.API): every OpStatic family maps 1:1 to its own family name (MoeRoute and
// CausalConv1d included); rejected reusing MoeFfn/LinearAttnScan names for those ops because
// shared names let two compile-time-distinct kernels hash to the same family and collide.
// Spec 4 §3.
/// Returns the family name string for an [`OpStatic`] descriptor (Spec 4 §3).
pub const fn op_static_family(op_static: &OpStatic) -> &'static str {
    match op_static {
        OpStatic::Matmul(_) => "matmul",
        OpStatic::MoeRoute(_) => "moe_route",
        OpStatic::MoeFfn(_) => "moe_ffn",
        OpStatic::Attention(_) => "attention",
        OpStatic::StateWriteKv(_) => "state_write_kv",
        OpStatic::CausalConv1d(_) => "causal_conv1d",
        OpStatic::LinearAttnScan(_) => "linear_attn_scan",
        OpStatic::Elementwise(_) => "elementwise",
        OpStatic::Sampling(_) => "sampling",
        OpStatic::Collectives(_) => "collectives",
    }
}

/// Constructs the exact canonical variant struct name: `<op>_<static_hash_16_lower_hex>_args` (Spec 4 §7).
pub fn canonical_struct_name(op: OpId, op_static: &OpStatic) -> String {
    let hash = r9v_registry::static_hash(op_static);
    format!("{}_{:016x}_args", op.as_str(), hash)
}

/// Validates compatibility between an [`OpId`] and an [`OpStatic`] descriptor exhaustively (Spec 4 §3, §7).
// DECISION(A3.API): op-to-static agreement is exact down to the nested descriptor
// (Sample requires Sampling::Sample, Send requires Collectives::Send); rejected
// family-level-only checks because two different compile-time kernel semantics
// must never share an identity. Cross-family mismatches report MismatchedOpFamily,
// within-family mismatches report NestedOpMismatch. Spec 4 §3.
fn validate_op_family(op: OpId, op_static: &OpStatic) -> Result<(), KgenError> {
    let family = op_static_family(op_static);
    let is_valid_family = matches!(
        (op, op_static),
        (OpId::Matmul, OpStatic::Matmul(_))
            | (OpId::MoeRoute, OpStatic::MoeRoute(_))
            | (OpId::MoeFfn, OpStatic::MoeFfn(_))
            | (OpId::Attention, OpStatic::Attention(_))
            | (OpId::StateWriteKv, OpStatic::StateWriteKv(_))
            | (OpId::CausalConv1d, OpStatic::CausalConv1d(_))
            | (OpId::LinearAttnScan, OpStatic::LinearAttnScan(_))
            | (
                OpId::EmbedGather
                    | OpId::NgramGather
                    | OpId::QuantAct
                    | OpId::Cast
                    | OpId::Copy
                    | OpId::GatherRows
                    | OpId::ScatterAddRows
                    | OpId::Split
                    | OpId::Concat
                    | OpId::Norm
                    | OpId::ResidualAdd
                    | OpId::ActMul
                    | OpId::Activation
                    | OpId::LogitSoftcap
                    | OpId::Rope,
                OpStatic::Elementwise(_),
            )
            | (
                OpId::LogitsPostprocess | OpId::Sample | OpId::Verify,
                OpStatic::Sampling(_),
            )
            | (
                OpId::AllReduce
                    | OpId::AllGather
                    | OpId::ReduceScatter
                    | OpId::AllToAll
                    | OpId::Send
                    | OpId::Recv
                    | OpId::Barrier,
                OpStatic::Collectives(_),
            )
    );

    if !is_valid_family {
        return Err(KgenError::MismatchedOpFamily { op, family });
    }
    let static_op = op_static.op_id();
    if static_op != op {
        return Err(KgenError::NestedOpMismatch { op, static_op });
    }
    Ok(())
}

/// Derives the canonical ABI description from an [`OpStatic`] parameter descriptor (Spec 4 §4.1).
///
/// Unique families dispatch directly; shared/ambiguous families fail closed with [`KgenError::AmbiguousOpFamily`].
pub fn abi(op_static: &OpStatic) -> Result<AbiStruct, KgenError> {
    match op_static {
        OpStatic::Matmul(_) => abi_for_op(OpId::Matmul, op_static),
        OpStatic::MoeRoute(_) => abi_for_op(OpId::MoeRoute, op_static),
        OpStatic::MoeFfn(_) => abi_for_op(OpId::MoeFfn, op_static),
        OpStatic::Attention(_) => abi_for_op(OpId::Attention, op_static),
        OpStatic::StateWriteKv(_) => abi_for_op(OpId::StateWriteKv, op_static),
        OpStatic::CausalConv1d(_) => abi_for_op(OpId::CausalConv1d, op_static),
        OpStatic::LinearAttnScan(_) => abi_for_op(OpId::LinearAttnScan, op_static),
        OpStatic::Elementwise(_) => Err(KgenError::AmbiguousOpFamily {
            family: "elementwise",
            valid_ops: vec![
                OpId::EmbedGather,
                OpId::NgramGather,
                OpId::QuantAct,
                OpId::Cast,
                OpId::Copy,
                OpId::GatherRows,
                OpId::ScatterAddRows,
                OpId::Split,
                OpId::Concat,
                OpId::Norm,
                OpId::ResidualAdd,
                OpId::ActMul,
                OpId::Activation,
                OpId::LogitSoftcap,
                OpId::Rope,
            ],
        }),
        OpStatic::Sampling(_) => Err(KgenError::AmbiguousOpFamily {
            family: "sampling",
            valid_ops: vec![OpId::LogitsPostprocess, OpId::Sample, OpId::Verify],
        }),
        OpStatic::Collectives(_) => Err(KgenError::AmbiguousOpFamily {
            family: "collectives",
            valid_ops: vec![
                OpId::AllReduce,
                OpId::AllGather,
                OpId::ReduceScatter,
                OpId::AllToAll,
                OpId::Send,
                OpId::Recv,
                OpId::Barrier,
            ],
        }),
    }
}

/// Authoritative entry point: constructs the canonical ABI description for any of the 32 closed-set operations (Spec 1 §4, Spec 4 §7).
pub fn abi_for_op(op: OpId, op_static: &OpStatic) -> Result<AbiStruct, KgenError> {
    validate_op_family(op, op_static)?;

    let name = canonical_struct_name(op, op_static);
    let b = AbiStructBuilder::new(name, op);

    match op {
        OpId::EmbedGather => b
            .add_field(FieldSpec::new(
                "token_ids",
                AbiType::const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Token IDs [T] u32 (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "table",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Embedding table [V, Dm] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "embed_override",
                AbiType::nullable_const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Optional external embeddings [T, Dm]; null when unused (Spec 1 §3.2)",
            ))
            .add_field(FieldSpec::new(
                "embed_mask",
                AbiType::nullable_const_ptr(PointeeType::U8),
                FieldRole::InputTensor,
                "Optional override mask [T] bool; rows set replace gathered output, null when unused (Spec 1 §3.2)",
            ))
            .add_field(FieldSpec::new(
                "x",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Gathered activations [T, Dm] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        // DECISION(A3.API): one ngram_gather struct carries both source signatures with the
        // inactive pair null (staging/row_scales live for Staged, token_ids/table for Device);
        // rejected two structs per source because the op key must stay one variant per static
        // descriptor. Spec 1 §4.A.
        OpId::NgramGather => b
            .add_field(FieldSpec::new(
                "staging",
                AbiType::nullable_const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Staging buffer [T, Np, Dn]; null in Device-table mode (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "row_scales",
                AbiType::nullable_const_ptr(PointeeType::F32),
                FieldRole::WeightScale,
                "Row scales buffer; null in Device-table mode (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "token_ids",
                AbiType::nullable_const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Token IDs [T] u32 for on-device hashing; null in Staged mode (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "table",
                AbiType::nullable_const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Device n-gram table [TotalEntries, Dn]; null in Staged mode (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "x",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Gathered activations [T, Np*Dn] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::QuantAct => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Input activations [T, N] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "xq",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Quantized activations [T, N] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "scale",
                AbiType::mut_ptr(PointeeType::F32),
                FieldRole::ActivationScale,
                "Per-token scale [T] f32 (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Cast => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Input tensor (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Output tensor (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic element count (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Copy => b
            .add_field(FieldSpec::new(
                "src",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Source buffer (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "dst",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Destination buffer (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic element count (Spec 1 §3.5)",
            ))
            .build(),

        OpId::GatherRows => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Source rows [N, D] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "indices",
                AbiType::const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Row indices [M] u32 (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Gathered rows [M, D] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "m",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic gathered row count M (Spec 1 §3.5)",
            ))
            .build(),

        OpId::ScatterAddRows => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Update rows [M, D] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "indices",
                AbiType::const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Row indices [M] u32 (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Accumulation target rows [N, D] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "m",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic update row count M (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Split => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Input tensor [T, C] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y0",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "First split output [T, C0] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y1",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Second split output [T, C1] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Concat => b
            .add_field(FieldSpec::new(
                "x0",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "First input [T, C0] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "x1",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Second input [T, C1] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Concatenated output [T, C] (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Norm => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Input activations [T, N] (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "weight",
                AbiType::const_ptr(PointeeType::F32),
                FieldRole::Weight,
                "Normalization weight [N] f32 (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "bias",
                AbiType::nullable_const_ptr(PointeeType::F32),
                FieldRole::Bias,
                "Optional normalization bias [N] f32 (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Normalized output activations [T, N] (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::ResidualAdd => b
            .add_field(FieldSpec::new(
                "a",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "First addend tensor (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "b",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::Residual,
                "Second addend (residual) tensor (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Sum output tensor (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic element count (Spec 1 §3.5)",
            ))
            .build(),

        OpId::ActMul => b
            .add_field(FieldSpec::new(
                "gate",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Gate activations [T, Dff] (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "up",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Up activations [T, Dff] (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Gated multiplied output [T, Dff] (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Activation => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Input activations [T, Dff] (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Activated output [T, Dff] (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::LogitSoftcap => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::F32),
                FieldRole::InputTensor,
                "Input logits [S, q, V] f32 (Spec 1 §4.B, SI-28)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::F32),
                FieldRole::OutputTensor,
                "Softcapped output logits [S, q, V] f32 (Spec 1 §4.B, SI-28)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic element count (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Rope => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Input activations [T, H, D] (Spec 1 §4.B)",
            ))
            .add_batch_meta_field(BatchMetaField::Positions)
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Rotated output activations [T, H, D] (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Matmul => {
            // validate_op_family above enforces exact Matmul pairing; re-check here
            // without panicking so input-dependent misuse stays a typed error.
            let s = match op_static {
                OpStatic::Matmul(s) => s,
                _ => {
                    return Err(KgenError::MismatchedOpFamily {
                        op,
                        family: op_static_family(op_static),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "w",
                AbiType::const_ptr(PointeeType::U8),
                FieldRole::Weight,
                "Weight buffer in L1 layout (Spec 2 §2.2)",
            ))
            .add_field(FieldSpec::new(
                "w_scales",
                AbiType::nullable_const_ptr(PointeeType::Void),
                FieldRole::WeightScale,
                "Optional weight quantization scale buffer (Spec 2 §3)",
            ))
            .add_field(FieldSpec::new(
                "w_indices",
                AbiType::nullable_const_ptr(PointeeType::U8),
                FieldRole::WeightIndices,
                "Optional weight sparsity indices buffer (Spec 2 §2.3)",
            ))
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::from_dtype(s.in_dtype)),
                FieldRole::InputTensor,
                "Activation input tensor; pointer type follows activation dtype, not out_dtype (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "x_scale",
                AbiType::nullable_const_ptr(PointeeType::F32),
                FieldRole::ActivationScale,
                "Optional activation quantization scale buffer (Spec 1 §4.A, §4.C)",
            ))
            .add_field(FieldSpec::new(
                "bias",
                AbiType::nullable_const_ptr(PointeeType::F32),
                FieldRole::Bias,
                "Optional fused epilogue bias tensor [N] f32 (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "residual",
                AbiType::nullable_const_ptr(PointeeType::from_dtype(s.out_dtype)),
                FieldRole::Residual,
                "Optional fused epilogue residual tensor [M, N] (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::from_dtype(s.out_dtype)),
                FieldRole::OutputTensor,
                "Output tensor [M, N] (Spec 1 §4.C)",
            ))
            // DECISION(A3.API): split-K partials workspace is part of the common matmul ABI for
            // every variant (sized by k_splits at tune time); rejected emitting it only for
            // k_splits > 1 because the struct shape must not depend on the tuned config.
            // Spec 4 §5.1, §7.
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::SplitKPartials, 0))
            .add_field(FieldSpec::new(
                "m",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic row count M <= m_bucket (Spec 1 §3.5, Spec 4 §7)",
            ))
            .build()
        }

        OpId::MoeRoute => b
            .add_field(FieldSpec::new(
                "logits",
                AbiType::const_ptr(PointeeType::F32),
                FieldRole::InputTensor,
                "Router input logits [T, E] f32 (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "bias",
                AbiType::nullable_const_ptr(PointeeType::F32),
                FieldRole::Bias,
                "Optional router routing correction bias [E] f32 (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "expert_ids",
                AbiType::mut_ptr(PointeeType::U32),
                FieldRole::OutputTensor,
                "Selected expert IDs [T, K] u32 (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "weights",
                AbiType::mut_ptr(PointeeType::F32),
                FieldRole::OutputTensor,
                "Routing weights [T, K] f32 (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T <= T_bucket (Spec 1 §3.5, Spec 4 §7)",
            ))
            .build(),

        OpId::MoeFfn => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Input activations [T, Dm] (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "expert_ids",
                AbiType::const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Routed expert indices [T, K] u32 (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "weights",
                AbiType::const_ptr(PointeeType::F32),
                FieldRole::InputTensor,
                "Routed expert weights [T, K] f32 (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "w_gate_up",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::Weight,
                "Grouped expert gate/up projection weights (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "w_gate_up_scales",
                AbiType::nullable_const_ptr(PointeeType::Void),
                FieldRole::WeightScale,
                "Optional gate/up quantization scales (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "w_down",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::Weight,
                "Grouped expert down projection weights (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "w_down_scales",
                AbiType::nullable_const_ptr(PointeeType::Void),
                FieldRole::WeightScale,
                "Optional down quantization scales (Spec 1 §4.C)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Combined output activations [T, Dm] (Spec 1 §4.C)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::MoeSortBuffers, 0))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T <= T_bucket (Spec 1 §3.5, Spec 4 §7)",
            ))
            .build(),

        OpId::Attention => {
            // validate_op_family above enforces exact Attention pairing; re-check here
            // without panicking so input-dependent misuse stays a typed error.
            let s = match op_static {
                OpStatic::Attention(s) => s,
                _ => {
                    return Err(KgenError::MismatchedOpFamily {
                        op,
                        family: op_static_family(op_static),
                    });
                }
            };
            let mut builder = b
                .add_field(FieldSpec::new(
                    "q",
                    AbiType::const_ptr(PointeeType::Void),
                    FieldRole::InputTensor,
                    "Query activations [T, H, D] (Spec 1 §4.D)",
                ))
                .add_field(FieldSpec::new(
                    "k_cache",
                    AbiType::const_ptr(PointeeType::Void),
                    FieldRole::InputTensor,
                    "Paged/latent key cache buffer (Spec 1 §4.D, Spec 3 §3.2)",
                ))
                .add_field(FieldSpec::new(
                    "v_cache",
                    AbiType::const_ptr(PointeeType::Void),
                    FieldRole::InputTensor,
                    "Paged/latent value cache buffer (Spec 1 §4.D, Spec 3 §3.2)",
                ))
                .add_field(FieldSpec::new(
                    "o",
                    AbiType::mut_ptr(PointeeType::Void),
                    FieldRole::OutputTensor,
                    "Attention output activations [T, H, D] (Spec 1 §4.D)",
                ))
                .add_batch_meta_field(BatchMetaField::BlockTable)
                .add_batch_meta_field(BatchMetaField::CtxLen)
                .add_batch_meta_field(BatchMetaField::QueryLen);

            match s.mask_kind {
                AttentionMask::Causal => {}
                AttentionMask::CausalWindow(_) => {
                    builder = builder.add_batch_meta_field(BatchMetaField::WindowStart);
                }
                AttentionMask::Tree => {
                    builder = builder
                        .add_batch_meta_field(BatchMetaField::TreeParents)
                        .add_batch_meta_field(BatchMetaField::TreeAncestors);
                }
            }

            if s.q_bucket <= 16 {
                builder = builder
                    .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::SplitKvPartials, 0));
            }

            builder
                .add_field(FieldSpec::new(
                    "s",
                    AbiType::u32(),
                    FieldRole::DynamicScalar,
                    "Active sequence count S <= S_bucket (Spec 1 §3.5, Spec 4 §7)",
                ))
                .add_field(FieldSpec::new(
                    "t",
                    AbiType::u32(),
                    FieldRole::DynamicScalar,
                    "Total query token count T <= q_bucket (Spec 1 §3.5, Spec 4 §7)",
                ))
                .build()
        }

        OpId::StateWriteKv => b
            .add_field(FieldSpec::new(
                "k",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Key projections [T, Hkv, D] (Spec 1 §4.D)",
            ))
            .add_field(FieldSpec::new(
                "v",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Value projections [T, Hkv, Dv] (Spec 1 §4.D)",
            ))
            .add_batch_meta_field(BatchMetaField::SlotMap)
            .add_field(FieldSpec::new(
                "k_cache",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "KV cache key storage buffer (Spec 1 §4.D, Spec 3 §3.2)",
            ))
            .add_field(FieldSpec::new(
                "v_cache",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "KV cache value storage buffer (Spec 1 §4.D, Spec 3 §3.2)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::CausalConv1d => b
            .add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Input sequence [T, C] (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "w",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::Weight,
                "Conv1d weight [C, W_k] (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "bias",
                AbiType::nullable_const_ptr(PointeeType::F32),
                FieldRole::Bias,
                "Optional bias [C] f32 (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "conv_state",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "ConvWindow ring buffer state (Spec 1 §4.E, Spec 3 §4)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Convolved output sequence [T, C] (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::LinearAttnScan => b
            .add_field(FieldSpec::new(
                "q",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Query activations [T, H, D] (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "k",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Key activations [T, H, D] (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "v",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Value activations [T, H, Dv] (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "alpha",
                AbiType::const_ptr(PointeeType::F32),
                FieldRole::InputTensor,
                "Decay alpha [T, H] f32 (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "beta",
                AbiType::const_ptr(PointeeType::F32),
                FieldRole::InputTensor,
                "Projection beta [T, H] f32 (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "state",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Recurrent scan state buffer (Spec 1 §4.E, Spec 3 §4)",
            ))
            .add_field(FieldSpec::new(
                "o",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Scan output activations [T, H, Dv] (Spec 1 §4.E)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::ScanCarry, 0))
            .add_batch_meta_field(BatchMetaField::QueryLen)
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build(),

        OpId::LogitsPostprocess => b
            .add_field(FieldSpec::new(
                "logits",
                AbiType::const_ptr(PointeeType::F32),
                FieldRole::InputTensor,
                "Input raw logits [S, q, V] f32 (Spec 1 §4.F)",
            ))
            .add_field(FieldSpec::new(
                "params",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Per-sequence SamplingParams [S] (Spec 1 §4.F)",
            ))
            .add_field(FieldSpec::new(
                "history_counts",
                AbiType::nullable_const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Optional history counts [S, V] u32 (Spec 1 §4.F)",
            ))
            .add_field(FieldSpec::new(
                "grammar_mask",
                AbiType::nullable_const_ptr(PointeeType::U8),
                FieldRole::InputTensor,
                "Optional grammar boolean mask [S, q, V] (Spec 1 §4.F)",
            ))
            .add_field(FieldSpec::new(
                "probs",
                AbiType::mut_ptr(PointeeType::F32),
                FieldRole::OutputTensor,
                "Transformed probabilities [S, q, V] f32 (Spec 1 §4.F)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::BitonicSort, 0))
            .add_field(FieldSpec::new(
                "s",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Active sequence count S <= S_bucket (Spec 1 §3.5)",
            ))
            .add_field(FieldSpec::new(
                "q",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Query tokens per sequence q <= q_bucket (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Sample => b
            .add_field(FieldSpec::new(
                "probs",
                AbiType::const_ptr(PointeeType::F32),
                FieldRole::InputTensor,
                "Normalized token probabilities [S, V] f32 (Spec 1 §4.F)",
            ))
            .add_field(FieldSpec::new(
                "rng_state",
                AbiType::mut_ptr(PointeeType::U64),
                FieldRole::InputTensor,
                "Philox PRNG state array [S] (Spec 1 §4.F, Spec 4 §5.8)",
            ))
            .add_batch_meta_field(BatchMetaField::SeqIds)
            .add_field(FieldSpec::new(
                "tokens",
                AbiType::mut_ptr(PointeeType::U32),
                FieldRole::OutputTensor,
                "Sampled token IDs [S] u32 (Spec 1 §4.F)",
            ))
            .add_field(FieldSpec::new(
                "s",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Active sequence count S <= S_bucket (Spec 1 §3.5)",
            ))
            .build(),

        OpId::Verify => {
            // validate_op_family above enforces Sampling::Verify pairing; re-check
            // without panicking so input-dependent misuse stays a typed error.
            let tree = match op_static {
                OpStatic::Sampling(r9v_registry::SamplingStatic::Verify(v)) => v.tree,
                OpStatic::Sampling(other) => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
                _ => {
                    return Err(KgenError::MismatchedOpFamily {
                        op,
                        family: op_static_family(op_static),
                    });
                }
            };
            let mut builder = b
                .add_field(FieldSpec::new(
                    "draft_tokens",
                    AbiType::const_ptr(PointeeType::U32),
                    FieldRole::InputTensor,
                    "Draft speculative tokens [S, k] u32 (Spec 1 §4.F, Spec 7 §4)",
                ))
                .add_field(FieldSpec::new(
                    "draft_probs",
                    AbiType::nullable_const_ptr(PointeeType::F32),
                    FieldRole::InputTensor,
                    "Optional draft probabilities [S, k, V] f32 (Spec 1 §4.F, Spec 7 §4)",
                ))
                .add_field(FieldSpec::new(
                    "target_probs",
                    AbiType::const_ptr(PointeeType::F32),
                    FieldRole::InputTensor,
                    "Target model probabilities [S, k+1, V] f32 (Spec 1 §4.F, Spec 7 §4)",
                ))
                .add_field(FieldSpec::new(
                    "rng_state",
                    AbiType::mut_ptr(PointeeType::U64),
                    FieldRole::InputTensor,
                    "Philox PRNG state array [S] (Spec 1 §4.F, Spec 7 §4)",
                ))
                .add_batch_meta_field(BatchMetaField::SeqIds);
            // DECISION(A3.API): tree verify inputs ride the static tree flag so the struct shape
            // is compile-time; rejected a runtime tree pointer because shape must come from
            // static_hash. Spec 7 §4.
            if tree {
                builder = builder
                    .add_batch_meta_field(BatchMetaField::TreeParents)
                    .add_batch_meta_field(BatchMetaField::TreeAncestors);
            }
            builder
                .add_field(FieldSpec::new(
                    "accepted",
                    AbiType::mut_ptr(PointeeType::U32),
                    FieldRole::OutputTensor,
                    "Accepted tokens buffer [S, k+1] u32 (Spec 1 §4.F, Spec 7 §4)",
                ))
                .add_field(FieldSpec::new(
                    "accept_len",
                    AbiType::mut_ptr(PointeeType::U32),
                    FieldRole::OutputTensor,
                    "Accepted length per sequence [S] u32 (Spec 1 §4.F, Spec 7 §4)",
                ))
                .add_field(FieldSpec::new(
                    "s",
                    AbiType::u32(),
                    FieldRole::DynamicScalar,
                    "Active sequence count S <= S_bucket (Spec 1 §3.5)",
                ))
                .add_field(FieldSpec::new(
                    "k",
                    AbiType::u32(),
                    FieldRole::DynamicScalar,
                    "Draft speculative token count k (Spec 1 §4.F, Spec 7 §4)",
                ))
                .build()
        }

        // DECISION(A3.API): rank, world_size, peer and group_id are compile-time statics, so
        // count is the only dynamic collective scalar; rejected runtime rank/world/peer scalars
        // because a kernel launched with a different rank than its static descriptor is a wrong
        // launch, not a dynamic shape. Spec 1 §4.G, Spec 4 §3.
        OpId::AllReduce => b
            .add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Send buffer (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Receive buffer (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Element count to reduce (Spec 1 §4.G)",
            ))
            .build(),

        OpId::AllGather => b
            .add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Send buffer (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Receive buffer (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Element count per rank (Spec 1 §4.G)",
            ))
            .build(),

        OpId::ReduceScatter => b
            .add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Send buffer (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Receive buffer (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Element count per rank (Spec 1 §4.G)",
            ))
            .build(),

        OpId::AllToAll => b
            .add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Send buffer (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Receive buffer (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "counts",
                AbiType::const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Per-rank row counts [P] u32 (Spec 1 §4.G, SI-11)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .build(),

        OpId::Send => b
            .add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Send buffer (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Element count to transfer (Spec 1 §4.G)",
            ))
            .build(),

        OpId::Recv => b
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::OutputTensor,
                "Receive buffer (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Expected element count (Spec 1 §4.G)",
            ))
            .build(),

        OpId::Barrier => b
            .add_field(FieldSpec::new(
                "flags",
                AbiType::mut_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Inter-rank barrier synchronization flags (Spec 1 §4.G)",
            ))
            .build(),
    }
}
