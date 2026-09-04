// SPDX-License-Identifier: Apache-2.0
//! R9V CPU reference scalar T0 implementations
//! (Spec 1 §4.A–§4.G, §6.1–§6.5, Spec 3 §2–§4, Spec 4 §2, Spec 7 §4,
//! Cards A1.5–A1.9 and A1.14).
//!
//! T0 serves as the primary ground-truth oracle for all engine operations.
//! Implementations are scalar, strictly deterministic, accumulate in f32 (or i32),
//! and cast once on output according to the IR contract.

pub mod act_mul;
pub mod activation;
pub mod attention;
pub mod buffer;
pub mod cast;
pub mod causal_conv1d;
pub mod collectives;
pub mod concat;
pub mod copy;
pub mod decode;
pub mod dtype;
pub mod embed_gather;
pub mod error;
pub mod exec;
pub mod gather_rows;
pub mod harness;
pub mod linear_attn_scan;
pub mod logit_softcap;
pub mod matmul;
pub mod moe_ffn;
pub mod moe_route;
pub mod ngram_gather;
pub mod norm;
pub mod philox;
pub mod quant_act;
pub mod residual_add;
pub mod rope;
pub mod sampling;
pub mod scatter_add_rows;
pub mod segments;
pub mod split;
pub mod synthetic;
pub mod tolerance;

pub use act_mul::{act_mul, act_mul_f64_reference};
pub use activation::{
    activation, activation_f64_reference, erf_f32, erf_f64, eval_activation_f32,
    eval_activation_f64,
};
pub use attention::{
    attention, attention_mla, attention_paged, attention_row_f64_reference, mla_row_f64_reference,
    state_write_kv, state_write_kv_latent, state_write_kv_paged, KvCache, KvLatentCache,
    KvPagedCache,
};
pub use buffer::{TensorData, TensorDataMut, TensorView, TensorViewMut, TypedBuffer};
pub use cast::{cast, cast_f64_reference};
pub use causal_conv1d::{causal_conv1d, causal_conv1d_f64_reference};
pub use collectives::{all_gather, all_reduce, all_to_all, barrier, recv, reduce_scatter, send};
pub use concat::{concat, concat_f64_reference};
pub use copy::{copy, copy_f64_reference};
pub use decode::{decode_greedy, DecodeConfig, DecodeResult};
pub use dtype::{
    bf16_to_f32, dtype_element_size, f16_to_f32, f32_to_bf16, f32_to_f16, fp8_e4m3_decode,
    fp8_e4m3_encode, fp8_e5m2_decode, fp8_e5m2_encode, read_f32_at, read_f64_at, write_f32_at,
};
pub use embed_gather::{embed_gather, embed_gather_f64_reference, embed_gather_with_scales};
pub use error::T0Error;
pub use exec::{CpuExecutor, ExecError, RunArgs};
pub use gather_rows::{gather_rows, gather_rows_f64_reference};
pub use linear_attn_scan::{
    linear_attn_scan_chunked, linear_attn_scan_f64_reference, linear_attn_scan_recurrent,
};
pub use logit_softcap::{logit_softcap, logit_softcap_f64_reference};
pub use matmul::{matmul, matmul_f64_reference, matmul_with_scales};
pub use moe_ffn::{moe_ffn, moe_ffn_f64_reference};
pub use moe_route::{moe_route, moe_route_f64_reference};
pub use ngram_gather::{
    ngram_gather, ngram_gather_device, ngram_gather_f64_reference_rows,
    ngram_gather_f64_reference_staged, NgramHash,
};
pub use norm::{norm, norm_f64_reference};
pub use philox::{philox4x32_10, u32_to_unit_f32, RngState};
pub use quant_act::{fp8_e4m3_encode_f64_oracle, quant_act, quant_act_f64_reference};
pub use residual_add::{residual_add, residual_add_f64_reference};
pub use rope::{rope, rope_f64_reference};
pub use sampling::{
    logits_postprocess, logits_postprocess_f64_reference, sample, verify, VerifyOutput,
};
pub use scatter_add_rows::{scatter_add_rows, scatter_add_rows_f64_reference};
pub use segments::SeqLayout;
pub use split::{split, split_f64_reference};
pub use synthetic::{build as build_synthetic, SyntheticSpec, TinyModel, CACHE_BLOCK_TOKENS};
pub use tolerance::Tolerance;

use r9v_ir::Op;

/// Dispatches and executes an elementwise op using scalar T0 reference implementations (Spec 1 §4.B, Spec 4 §2).
///
/// Enforces exact input and output operand arity for each supported elementwise op.
pub fn execute_elementwise_op(
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
) -> Result<(), T0Error> {
    match op {
        Op::Norm(norm_op) => {
            if (inputs.len() != 2 && inputs.len() != 3) || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "norm",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "norm requires 2 or 3 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            let x = &inputs[0];
            let weight = &inputs[1];
            let bias = if inputs.len() == 3 {
                Some(&inputs[2])
            } else {
                None
            };
            norm(norm_op, x, weight, bias, &mut outputs[0])
        }
        Op::ResidualAdd(res_op) => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "residual_add",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "residual_add requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            residual_add(res_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::ActMul(act_mul_op) => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "act_mul",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "act_mul requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            act_mul(act_mul_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::Activation(act_op) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "activation",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "activation requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            activation(act_op, &inputs[0], &mut outputs[0])
        }
        Op::Rope(rope_op) => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "rope",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "rope requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            rope(rope_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::Cast(cast_op) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "cast",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "cast requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            cast(cast_op, &inputs[0], &mut outputs[0])
        }
        Op::Copy(copy_op) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "copy",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "copy requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            copy(copy_op, &inputs[0], &mut outputs[0])
        }
        Op::QuantAct(quant_op) => {
            if inputs.len() != 1 || outputs.len() != 2 {
                return Err(T0Error::InvalidAttribute {
                    op: "quant_act",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "quant_act requires 1 input and 2 outputs, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            let (xq, rest) = outputs.split_at_mut(1);
            quant_act(quant_op, &inputs[0], &mut xq[0], &mut rest[0])
        }
        Op::Split(split_op) => {
            if inputs.len() != 1 || outputs.len() != 2 {
                return Err(T0Error::InvalidAttribute {
                    op: "split",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "split requires 1 input and 2 outputs, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            let (a, rest) = outputs.split_at_mut(1);
            split(split_op, &inputs[0], &mut a[0], &mut rest[0])
        }
        Op::Concat(concat_op) => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "concat",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "concat requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            concat(concat_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::LogitSoftcap(softcap_op) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "logit_softcap",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "logit_softcap requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            logit_softcap(softcap_op, &inputs[0], &mut outputs[0])
        }
        other => Err(T0Error::InvalidAttribute {
            op: other.op_name(),
            attribute: "op",
            reason: format!(
                "op `{}` is not in the A1.5 elementwise group",
                other.op_name()
            ),
        }),
    }
}

/// Dispatches and executes a matmul op using scalar T0 reference implementation (Spec 1 §4.C, §6.1, §6.2, Card A1.6).
pub fn execute_matmul_op(
    op: &r9v_ir::MatmulOp,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
) -> Result<(), T0Error> {
    if outputs.len() != 1 {
        return Err(T0Error::InvalidAttribute {
            op: "matmul",
            attribute: "outputs",
            reason: format!("matmul requires 1 output, got {}", outputs.len()),
        });
    }

    match op.epilogue {
        r9v_ir::Epilogue::None | r9v_ir::Epilogue::Act(_) => {
            if inputs.len() != 2 {
                return Err(T0Error::InvalidAttribute {
                    op: "matmul",
                    attribute: "inputs",
                    reason: format!(
                        "matmul requires 2 inputs for None/Act epilogue, got {}",
                        inputs.len()
                    ),
                });
            }
            matmul(op, &inputs[0], &inputs[1], None, None, &mut outputs[0])
        }
        r9v_ir::Epilogue::Bias => {
            if inputs.len() != 3 {
                return Err(T0Error::InvalidAttribute {
                    op: "matmul",
                    attribute: "inputs",
                    reason: format!(
                        "matmul requires 3 inputs for Bias epilogue, got {}",
                        inputs.len()
                    ),
                });
            }
            matmul(
                op,
                &inputs[0],
                &inputs[1],
                Some(&inputs[2]),
                None,
                &mut outputs[0],
            )
        }
        r9v_ir::Epilogue::Residual => {
            if inputs.len() != 3 {
                return Err(T0Error::InvalidAttribute {
                    op: "matmul",
                    attribute: "inputs",
                    reason: format!(
                        "matmul requires 3 inputs for Residual epilogue, got {}",
                        inputs.len()
                    ),
                });
            }
            matmul(
                op,
                &inputs[0],
                &inputs[1],
                None,
                Some(&inputs[2]),
                &mut outputs[0],
            )
        }
    }
}

/// Dispatches and executes a lookup or scatter op using scalar T0 reference implementation (Spec 1 §4.A, Card A1.6).
pub fn execute_lookup_op(
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
) -> Result<(), T0Error> {
    match op {
        Op::EmbedGather(embed_op) => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "embed_gather",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "embed_gather requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            embed_gather(embed_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::GatherRows(gather_op) => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "gather_rows",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "gather_rows requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            gather_rows(gather_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::ScatterAddRows(scatter_op) => {
            if (inputs.len() != 2 && inputs.len() != 3) || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "scatter_add_rows",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "scatter_add_rows requires 2 or 3 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            let dest = if inputs.len() == 3 {
                Some(&inputs[2])
            } else {
                None
            };
            scatter_add_rows(scatter_op, &inputs[0], &inputs[1], dest, &mut outputs[0])
        }
        other => Err(T0Error::InvalidAttribute {
            op: other.op_name(),
            attribute: "op",
            reason: format!("op `{}` is not a lookup/scatter op", other.op_name()),
        }),
    }
}

/// Dispatches and executes a MoE op using scalar T0 reference implementations (Spec 1 §4.C, Card A1.9).
///
/// `moe_route` takes 1 or 2 inputs `(logits, bias?)` and 2 outputs `(expert_ids, weights)`.
/// `moe_ffn` takes exactly 5 inputs `(x, expert_ids, weights, w_gate_up, w_down)` and 1 output;
/// out-of-band scales travel attached to the input views (see [`moe_ffn()`]).
pub fn execute_moe_op(
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
) -> Result<(), T0Error> {
    match op {
        Op::MoeRoute(route_op) => {
            if (inputs.len() != 1 && inputs.len() != 2) || outputs.len() != 2 {
                return Err(T0Error::InvalidAttribute {
                    op: "moe_route",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "moe_route requires 1 or 2 inputs and 2 outputs, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            let bias = if inputs.len() == 2 {
                Some(&inputs[1])
            } else {
                None
            };
            let (ids, rest) = outputs.split_at_mut(1);
            moe_route(route_op, &inputs[0], bias, &mut ids[0], &mut rest[0])
        }
        Op::MoeFfn(ffn_op) => {
            if inputs.len() != 5 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "moe_ffn",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "moe_ffn requires 5 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            moe_ffn(
                ffn_op,
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                None,
                &inputs[4],
                None,
                None,
                &mut outputs[0],
            )
        }
        other => Err(T0Error::InvalidAttribute {
            op: other.op_name(),
            attribute: "op",
            reason: format!("op `{}` is not in the A1.9 MoE group", other.op_name()),
        }),
    }
}

/// Dispatches and executes a stateful sequence op using scalar T0 reference implementations
/// (Spec 1 §4.E, Spec 3 §4.2, Card A1.9).
///
/// `causal_conv1d` takes 2 or 3 inputs `(x, w, bias?)` and 1 output; `linear_attn_scan`
/// takes exactly 5 inputs `(q, k, v, alpha, beta)` and 1 output. Both read `state_in`
/// (slot A) and write `state_out` (slot B) over the explicit `seq` segmentation.
/// `chunked` selects the scan form for `linear_attn_scan` (ignored for `causal_conv1d`);
/// form selection is executor scope per spec 4 §5.5.
#[allow(clippy::too_many_arguments)]
pub fn execute_state_scan_op(
    op: &Op,
    inputs: &[TensorView<'_>],
    state_in: &TensorView<'_>,
    seq: &SeqLayout,
    outputs: &mut [TensorViewMut<'_>],
    state_out: &mut TensorViewMut<'_>,
    chunked: bool,
) -> Result<(), T0Error> {
    match op {
        Op::CausalConv1d(conv_op) => {
            if (inputs.len() != 2 && inputs.len() != 3) || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "causal_conv1d",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "causal_conv1d requires 2 or 3 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            let bias = if inputs.len() == 3 {
                Some(&inputs[2])
            } else {
                None
            };
            causal_conv1d(
                conv_op,
                &inputs[0],
                &inputs[1],
                bias,
                state_in,
                seq,
                &mut outputs[0],
                state_out,
            )
        }
        Op::LinearAttnScan(scan_op) => {
            if inputs.len() != 5 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "linear_attn_scan",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "linear_attn_scan requires 5 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            if chunked {
                linear_attn_scan_chunked(
                    scan_op,
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    state_in,
                    seq,
                    &mut outputs[0],
                    state_out,
                )
            } else {
                linear_attn_scan_recurrent(
                    scan_op,
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    state_in,
                    seq,
                    &mut outputs[0],
                    state_out,
                )
            }
        }
        other => Err(T0Error::InvalidAttribute {
            op: other.op_name(),
            attribute: "op",
            reason: format!(
                "op `{}` is not in the A1.9 state/scan group",
                other.op_name()
            ),
        }),
    }
}

/// Dispatches and executes the staged `ngram_gather` via views (Spec 1 §4.A, Card A1.9).
///
/// Takes exactly 2 inputs `(gather_staging, row_scales)` and 1 output; the op must
/// carry `NgramSource::Staged`. Device-table mode needs a hash and table scale,
/// so it runs through [`ngram_gather_device`] instead.
pub fn execute_ngram_op(
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
) -> Result<(), T0Error> {
    match op {
        Op::NgramGather(ngram_op) => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "ngram_gather",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "ngram_gather requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            ngram_gather(ngram_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        other => Err(T0Error::InvalidAttribute {
            op: other.op_name(),
            attribute: "op",
            reason: format!("op `{}` is not in the A1.9 n-gram group", other.op_name()),
        }),
    }
}

/// Dispatches and executes a rank-1 collective using scalar T0 reference implementations
/// (Spec 1 §4.G, Card A1.9).
///
/// Arity: `all_reduce`/`all_gather`/`reduce_scatter` take 1 input and 1 output;
/// `all_to_all` takes 2 inputs `(x, counts)` and 1 output; `send` takes 1 input
/// and no outputs; `recv` takes no inputs and 1 output; `barrier` takes neither.
pub fn execute_collective_op(
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorViewMut<'_>],
) -> Result<(), T0Error> {
    match op {
        Op::AllReduce(reduce_op) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "all_reduce",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "all_reduce requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            all_reduce(reduce_op, &inputs[0], &mut outputs[0])
        }
        Op::AllGather(gather_op) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "all_gather",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "all_gather requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            all_gather(gather_op, &inputs[0], &mut outputs[0])
        }
        Op::ReduceScatter(scatter_op) => {
            if inputs.len() != 1 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "reduce_scatter",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "reduce_scatter requires 1 input and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            reduce_scatter(scatter_op, &inputs[0], &mut outputs[0])
        }
        Op::AllToAll(all_to_all_op) => {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "all_to_all",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "all_to_all requires 2 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            all_to_all(all_to_all_op, &inputs[0], &inputs[1], &mut outputs[0])
        }
        Op::Send(send_op) => {
            if inputs.len() != 1 || !outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "send",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "send requires 1 input and 0 outputs, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            send(send_op, &inputs[0])
        }
        Op::Recv(recv_op) => {
            if !inputs.is_empty() || outputs.len() != 1 {
                return Err(T0Error::InvalidAttribute {
                    op: "recv",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "recv requires 0 inputs and 1 output, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            recv(recv_op, &mut outputs[0])
        }
        Op::Barrier(barrier_op) => {
            if !inputs.is_empty() || !outputs.is_empty() {
                return Err(T0Error::InvalidAttribute {
                    op: "barrier",
                    attribute: "inputs/outputs",
                    reason: format!(
                        "barrier requires 0 inputs and 0 outputs, got {} inputs and {} outputs",
                        inputs.len(),
                        outputs.len()
                    ),
                });
            }
            barrier(barrier_op)
        }
        other => Err(T0Error::InvalidAttribute {
            op: other.op_name(),
            attribute: "op",
            reason: format!(
                "op `{}` is not in the A1.9 collective group",
                other.op_name()
            ),
        }),
    }
}
