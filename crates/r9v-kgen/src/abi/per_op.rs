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

// DECISION(A3.API): ABI pointee policy is exhaustive over five classes (Spec 4 §7).
// Every activation, parameter, index, or output pointer whose exact element dtype
// is carried in OpStatic uses PointeeType::from_dtype; fields with no dtype in the
// static keep their exact U32/U64/F32/U8 spelling. Rejected leaving typed pointers
// Void because an untyped pointer aliases distinct load widths. Void survives only
// for: (a) layout- or scheme-dependent weight storage (matmul byte buffers,
// embed/ngram tables, moe/conv packed weights); (b) scheme-dependent weight scale
// records (matmul/moe scales); (c) paged or fixed state arenas whose element
// interpretation varies by layout (attention/state-write caches, conv/scan state);
// (d) dtype-agnostic raw byte movement (copy src/dst); (e) truly heterogeneous
// records (sampling params blob). Quantized-but-byte-addressable activations
// (i8/e4m3/i4/bool map to I8/U8) stay typed because the byte spelling is exact.
// Spec 1 §2.1, §4, Spec 4 §3, §7.
/// Borrows the closed elementwise parameter descriptor for an elementwise op (Spec 4 §3).
fn elementwise_params(
    op: OpId,
    op_static: &OpStatic,
) -> Result<&r9v_registry::ElementwiseParams, KgenError> {
    match op_static {
        OpStatic::Elementwise(s) => Ok(&s.op_params),
        _ => Err(KgenError::MismatchedOpFamily {
            op,
            family: op_static_family(op_static),
        }),
    }
}

/// Returns the exact communication element dtype for a collective op (Spec 1 §4.G, Spec 4 §3).
fn collective_dtype(op: OpId, op_static: &OpStatic) -> Result<r9v_ir::DType, KgenError> {
    match op_static {
        OpStatic::Collectives(c) => Ok(match c {
            r9v_registry::CollectivesStatic::AllReduce(s) => s.dtype,
            r9v_registry::CollectivesStatic::AllGather(s) => s.dtype,
            r9v_registry::CollectivesStatic::ReduceScatter(s) => s.dtype,
            r9v_registry::CollectivesStatic::AllToAll(s) => s.dtype,
            r9v_registry::CollectivesStatic::Send(s) => s.dtype,
            r9v_registry::CollectivesStatic::Recv(s) => s.dtype,
            r9v_registry::CollectivesStatic::Barrier(_) => {
                return Err(KgenError::NestedOpMismatch {
                    op,
                    static_op: OpId::Barrier,
                });
            }
        }),
        _ => Err(KgenError::MismatchedOpFamily {
            op,
            family: op_static_family(op_static),
        }),
    }
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
        OpId::EmbedGather => {
            let out_dtype = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::EmbedGather(p) => p.out_dtype,
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            let out_pointee = PointeeType::from_dtype(out_dtype);
            b.add_field(FieldSpec::new(
                "token_ids",
                AbiType::const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Token IDs [T] u32 (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "table",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Embedding table [V, Dm] in layout- and scheme-dependent storage (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "embed_override",
                AbiType::nullable_const_ptr(out_pointee),
                FieldRole::InputTensor,
                "Optional external embeddings [T, Dm]; element type follows the static out dtype, null when unused (Spec 1 §3.2)",
            ))
            .add_field(FieldSpec::new(
                "embed_mask",
                AbiType::nullable_const_ptr(PointeeType::U8),
                FieldRole::InputTensor,
                "Optional override mask [T] bool; rows set replace gathered output, null when unused (Spec 1 §3.2)",
            ))
            .add_field(FieldSpec::new(
                "x",
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Gathered activations [T, Dm]; element type follows the static out dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

        // DECISION(A3.API): one ngram_gather struct carries both source signatures with the
        // inactive pair null (staging/row_scales live for Staged, token_ids/table for Device);
        // rejected two structs per source because the op key must stay one variant per static
        // descriptor. Spec 1 §4.A.
        OpId::NgramGather => {
            // Row scales are f32 or f16 in Staged mode (Spec 1 §4.A); the
            // pointer type follows the static scales dtype so f16 scales are
            // never read through an f32-typed field. Device-table mode keeps
            // the historical f32 pointer shape (always null). validate_op_family
            // above already enforces exact NgramGather pairing; re-check here
            // without panicking so input-dependent misuse stays a typed error.
            let params = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::NgramGather(p) => p,
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            let scales_pointee = match params.scales_dtype {
                Some(d) => PointeeType::from_dtype(d),
                None => PointeeType::F32,
            };
            let out_pointee = PointeeType::from_dtype(params.out_dtype);
            b.add_field(FieldSpec::new(
                "staging",
                AbiType::nullable_const_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "Staging buffer [T, Np, Dn]; null in Device-table mode (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "row_scales",
                AbiType::nullable_const_ptr(scales_pointee),
                FieldRole::WeightScale,
                "Row scales buffer; element type follows the static scales dtype, null in Device-table mode (Spec 1 §4.A)",
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
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Gathered activations [T, Np*Dn]; element type follows the static out dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::QuantAct => {
            let (in_pointee, target_pointee) = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::QuantAct(p) => (
                    PointeeType::from_dtype(p.in_dtype),
                    PointeeType::from_dtype(p.target),
                ),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Input activations [T, N]; element type follows the static in dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "xq",
                AbiType::mut_ptr(target_pointee),
                FieldRole::OutputTensor,
                "Quantized activations [T, N]; element type follows the static target dtype (Spec 1 §4.A)",
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
            .build()
        }

        OpId::Cast => {
            let (in_pointee, out_pointee) = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::Cast(p) => (
                    PointeeType::from_dtype(p.in_dtype),
                    PointeeType::from_dtype(p.out_dtype),
                ),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Input tensor; element type follows the static in dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Output tensor; element type follows the static out dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic element count (Spec 1 §3.5)",
            ))
            .build()
        }

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

        OpId::GatherRows => {
            let (elem_pointee, index_pointee) = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::GatherRows(p) => (
                    PointeeType::from_dtype(p.dtype),
                    PointeeType::from_dtype(p.index_dtype),
                ),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Source rows [N, D]; element type follows the static dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "indices",
                AbiType::const_ptr(index_pointee),
                FieldRole::InputTensor,
                "Row indices [M]; element type follows the static index dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Gathered rows [M, D]; element type follows the static dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "m",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic gathered row count M (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::ScatterAddRows => {
            let (elem_pointee, index_pointee) = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::ScatterAddRows(p) => (
                    PointeeType::from_dtype(p.dtype),
                    PointeeType::from_dtype(p.index_dtype),
                ),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Update rows [M, D]; element type follows the static dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "indices",
                AbiType::const_ptr(index_pointee),
                FieldRole::InputTensor,
                "Row indices [M]; element type follows the static index dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "dest",
                AbiType::nullable_const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Distinct base tensor [N, D] for the out-of-place form; element type follows the static dtype, null exactly when the static has_dest is false (Spec 1 §4.A, SI-10). No host-side value packer exists in A3.1/A3.2: LaunchEntry carries opaque args_blob bytes with no per-op host validation, so nullable here is the only contract available; enforcement is the static has_dest flag plus the guarded HIP unpacker, not a launch-layer check.",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Accumulation target rows [N, D]; element type follows the static dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "m",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic update row count M (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::Split => {
            let elem_pointee = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::Split(p) => PointeeType::from_dtype(p.dtype),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Input tensor [T, C]; element type follows the static dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y0",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "First split output [T, C0]; element type follows the static dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y1",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Second split output [T, C1]; element type follows the static dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::Concat => {
            let (a_pointee, b_pointee, out_pointee) = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::Concat(p) => (
                    PointeeType::from_dtype(p.a_dtype),
                    PointeeType::from_dtype(p.b_dtype),
                    PointeeType::from_dtype(p.out_dtype),
                ),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x0",
                AbiType::const_ptr(a_pointee),
                FieldRole::InputTensor,
                "First input [T, C0]; element type follows the static a dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "x1",
                AbiType::const_ptr(b_pointee),
                FieldRole::InputTensor,
                "Second input [T, C1]; element type follows the static b dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Concatenated output [T, C]; element type follows the static out dtype (Spec 1 §4.A)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::Norm => {
            let (in_pointee, out_pointee) = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::Norm(p) => (
                    PointeeType::from_dtype(p.in_dtype),
                    PointeeType::from_dtype(p.out_dtype),
                ),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Input activations [T, N]; element type follows the static in dtype (Spec 1 §4.B)",
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
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Normalized output activations [T, N]; element type follows the static out dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::ResidualAdd => {
            let (a_pointee, b_pointee, out_pointee) = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::ResidualAdd(p) => (
                    PointeeType::from_dtype(p.a_dtype),
                    PointeeType::from_dtype(p.b_dtype),
                    PointeeType::from_dtype(p.out_dtype),
                ),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "a",
                AbiType::const_ptr(a_pointee),
                FieldRole::InputTensor,
                "First addend tensor; element type follows the static a dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "b",
                AbiType::const_ptr(b_pointee),
                FieldRole::Residual,
                "Second addend (residual) tensor; element type follows the static b dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Sum output tensor; element type follows the static out dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic element count (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::ActMul => {
            let elem_pointee = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::ActMul(p) => PointeeType::from_dtype(p.dtype),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "gate",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Gate activations [T, Dff]; element type follows the static dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "up",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Up activations [T, Dff]; element type follows the static dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Gated multiplied output [T, Dff]; element type follows the static dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::Activation => {
            let elem_pointee = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::Activation(p) => PointeeType::from_dtype(p.dtype),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Input activations [T, Dff]; element type follows the static dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Activated output [T, Dff]; element type follows the static dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

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

        OpId::Rope => {
            let (in_pointee, out_pointee) = match elementwise_params(op, op_static)? {
                r9v_registry::ElementwiseParams::Rope(p) => (
                    PointeeType::from_dtype(p.in_dtype),
                    PointeeType::from_dtype(p.out_dtype),
                ),
                other => {
                    return Err(KgenError::NestedOpMismatch {
                        op,
                        static_op: other.op_id(),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Input activations [T, H, D]; element type follows the static in dtype (Spec 1 §4.B)",
            ))
            .add_batch_meta_field(BatchMetaField::Positions)
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Rotated output activations [T, H, D]; element type follows the static out dtype (Spec 1 §4.B)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

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
                AbiType::nullable_const_ptr(PointeeType::from_dtype(
                    s.residual_dtype.unwrap_or(s.out_dtype),
                )),
                FieldRole::Residual,
                "Optional fused epilogue residual tensor [M, N]; pointer type follows the residual input dtype, not out_dtype (Spec 1 §4.C)",
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

        OpId::MoeFfn => {
            // validate_op_family above enforces exact MoeFfn pairing; re-check
            // here without panicking so input-dependent misuse stays a typed error.
            let (in_pointee, out_pointee) = match op_static {
                OpStatic::MoeFfn(s) => (
                    PointeeType::from_dtype(s.in_dtype),
                    PointeeType::from_dtype(s.out_dtype),
                ),
                _ => {
                    return Err(KgenError::MismatchedOpFamily {
                        op,
                        family: op_static_family(op_static),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Input activations [T, Dm]; element type follows the static in dtype (Spec 1 §4.C)",
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
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Combined output activations [T, Dm]; element type follows the static out dtype (Spec 1 §4.C)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::MoeSortBuffers, 0))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T <= T_bucket (Spec 1 §3.5, Spec 4 §7)",
            ))
            .build()
        }

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
            let q_pointee = PointeeType::from_dtype(s.q_dtype);
            let o_pointee = PointeeType::from_dtype(s.out_dtype);
            let mut builder = b
                .add_field(FieldSpec::new(
                    "q",
                    AbiType::const_ptr(q_pointee),
                    FieldRole::InputTensor,
                    "Query activations [T, H, D]; element type follows the static q dtype (Spec 1 §4.D)",
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
                    AbiType::mut_ptr(o_pointee),
                    FieldRole::OutputTensor,
                    "Attention output activations [T, H, D]; element type follows the static out dtype (Spec 1 §4.D)",
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

        OpId::StateWriteKv => {
            // validate_op_family above enforces exact StateWriteKv pairing;
            // re-check here without panicking so input-dependent misuse stays
            // a typed error.
            let in_pointee = match op_static {
                OpStatic::StateWriteKv(s) => PointeeType::from_dtype(s.in_dtype),
                _ => {
                    return Err(KgenError::MismatchedOpFamily {
                        op,
                        family: op_static_family(op_static),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "k",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Key projections [T, Hkv, D]; element type follows the static in dtype (Spec 1 §4.D)",
            ))
            .add_field(FieldSpec::new(
                "v",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Value projections [T, Hkv, Dv]; element type follows the static in dtype (Spec 1 §4.D)",
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
            .build()
        }

        OpId::CausalConv1d => {
            // validate_op_family above enforces exact CausalConv1d pairing;
            // re-check here without panicking so input-dependent misuse stays
            // a typed error.
            let s = match op_static {
                OpStatic::CausalConv1d(s) => s,
                _ => {
                    return Err(KgenError::MismatchedOpFamily {
                        op,
                        family: op_static_family(op_static),
                    });
                }
            };
            // The bias dtype is independent of the input dtype (Spec 1 §4.E);
            // the pointer type follows the static bias dtype so an f16/bf16
            // bias is never read through an f32-typed field. Absent bias keeps
            // the historical f32 pointer shape (always null).
            let bias_pointee = match s.bias_dtype {
                Some(d) => PointeeType::from_dtype(d),
                None => PointeeType::F32,
            };
            let x_pointee = PointeeType::from_dtype(s.x_dtype);
            let y_pointee = PointeeType::from_dtype(s.out_dtype);
            b.add_field(FieldSpec::new(
                "x",
                AbiType::const_ptr(x_pointee),
                FieldRole::InputTensor,
                "Input sequence [T, C]; element type follows the static x dtype (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "w",
                AbiType::const_ptr(PointeeType::Void),
                FieldRole::Weight,
                "Conv1d weight [C, W_k] (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "bias",
                AbiType::nullable_const_ptr(bias_pointee),
                FieldRole::Bias,
                "Optional bias [C]; element type follows the static bias dtype (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "conv_state",
                AbiType::mut_ptr(PointeeType::Void),
                FieldRole::InputTensor,
                "ConvWindow ring buffer state (Spec 1 §4.E, Spec 3 §4)",
            ))
            .add_field(FieldSpec::new(
                "y",
                AbiType::mut_ptr(y_pointee),
                FieldRole::OutputTensor,
                "Convolved output sequence [T, C]; element type follows the static out dtype (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

        OpId::LinearAttnScan => {
            // validate_op_family above enforces exact LinearAttnScan pairing;
            // re-check here without panicking so input-dependent misuse stays
            // a typed error.
            let (in_pointee, out_pointee) = match op_static {
                OpStatic::LinearAttnScan(s) => (
                    PointeeType::from_dtype(s.in_dtype),
                    PointeeType::from_dtype(s.out_dtype),
                ),
                _ => {
                    return Err(KgenError::MismatchedOpFamily {
                        op,
                        family: op_static_family(op_static),
                    });
                }
            };
            b.add_field(FieldSpec::new(
                "q",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Query activations [T, H, D]; element type follows the static in dtype (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "k",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Key activations [T, H, D]; element type follows the static in dtype (Spec 1 §4.E)",
            ))
            .add_field(FieldSpec::new(
                "v",
                AbiType::const_ptr(in_pointee),
                FieldRole::InputTensor,
                "Value activations [T, H, Dv]; element type follows the static in dtype (Spec 1 §4.E)",
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
                AbiType::mut_ptr(out_pointee),
                FieldRole::OutputTensor,
                "Scan output activations [T, H, Dv]; element type follows the static out dtype (Spec 1 §4.E)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::ScanCarry, 0))
            .add_batch_meta_field(BatchMetaField::QueryLen)
            .add_field(FieldSpec::new(
                "t",
                AbiType::u32(),
                FieldRole::DynamicScalar,
                "Dynamic token count T (Spec 1 §3.5)",
            ))
            .build()
        }

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
        OpId::AllReduce => {
            let elem_pointee = PointeeType::from_dtype(collective_dtype(op, op_static)?);
            b.add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Send buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Receive buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Element count to reduce (Spec 1 §4.G)",
            ))
            .build()
        }

        OpId::AllGather => {
            let elem_pointee = PointeeType::from_dtype(collective_dtype(op, op_static)?);
            b.add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Send buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Receive buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Element count per rank (Spec 1 §4.G)",
            ))
            .build()
        }

        OpId::ReduceScatter => {
            let elem_pointee = PointeeType::from_dtype(collective_dtype(op, op_static)?);
            b.add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Send buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Receive buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Element count per rank (Spec 1 §4.G)",
            ))
            .build()
        }

        OpId::AllToAll => {
            let elem_pointee = PointeeType::from_dtype(collective_dtype(op, op_static)?);
            b.add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Send buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Receive buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_field(FieldSpec::new(
                "counts",
                AbiType::const_ptr(PointeeType::U32),
                FieldRole::InputTensor,
                "Per-rank row counts [P] u32 (Spec 1 §4.G, SI-11)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .build()
        }

        OpId::Send => {
            let elem_pointee = PointeeType::from_dtype(collective_dtype(op, op_static)?);
            b.add_field(FieldSpec::new(
                "send_buf",
                AbiType::const_ptr(elem_pointee),
                FieldRole::InputTensor,
                "Send buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Element count to transfer (Spec 1 §4.G)",
            ))
            .build()
        }

        OpId::Recv => {
            let elem_pointee = PointeeType::from_dtype(collective_dtype(op, op_static)?);
            b.add_field(FieldSpec::new(
                "recv_buf",
                AbiType::mut_ptr(elem_pointee),
                FieldRole::OutputTensor,
                "Receive buffer; element type follows the static dtype (Spec 1 §4.G)",
            ))
            .add_workspace_slot(WorkspaceSlot::new(WorkspaceSlotKind::CollectiveStaging, 0))
            .add_field(FieldSpec::new(
                "count",
                AbiType::u64(),
                FieldRole::DynamicScalar,
                "Expected element count (Spec 1 §4.G)",
            ))
            .build()
        }

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
