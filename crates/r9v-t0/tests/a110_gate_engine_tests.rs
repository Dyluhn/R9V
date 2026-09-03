// SPDX-License-Identifier: Apache-2.0
//! A1.10 generic gate engine coverage (Spec 4 §10, Spec 1 §6.1 / App. B).
//!
//! Every supported `r9v_ir::Op` variant runs through the cohesive engine
//! (`r9v_t0::harness::run_gates`): golden vs an independent f64/source
//! oracle on exactly 32 seeded inputs per legal shape (including
//! single-token, padding-row, and max-bucket edges with tiny non-bucket
//! dimensions), batch invariance over a pinned logical row at differing
//! row indices, twice determinism from fresh state, and legal/illegal
//! shape fuzz with typed refusals. This is the single A1.10 op-level path:
//! every variant is exercised through the shared engine, not a parallel
//! set of bespoke comparison loops.

use r9v_common::ids::{SeqId, StepId};
use r9v_common::rng::SeededRng;
use r9v_format::{GgmlType, SchemeId};
use r9v_ir::{
    ActMulOp, ActivationKind, ActivationOp, AllGatherOp, AllReduceOp, AllToAllOp, AttentionMask,
    AttentionOp, BarrierOp, BatchMeta, CacheScaleGranularity, CastOp, CausalConv1dOp, ConcatOp,
    ConvActivation, CopyKind, CopyOp, DType, EmbedGatherOp, Epilogue, GatherRowsOp, GroupId,
    HashId, LinearAttnKind, LinearAttnScanOp, LogitSoftcapOp, MatmulOp, MoeFfnOp, MoeRouteOp,
    MoeScoring, NgramCombine, NgramSource, NormAxis, NormKind, NormOp, Positions, QuantActOp,
    QuantScheme, RecvOp, ReduceOp, ReduceScatterOp, ResidualAddOp, RopeOp, RopeScaling, RopeStyle,
    SamplingParams, ScatterAddRowsOp, SendOp, Smoothing, SplitOp, StateHandle, StateKind,
    StateWriteKvOp, VerifyMethod,
};
use r9v_t0::harness::{
    self, activation_class_tensor, activation_values, batch_invariant, bucket_edge_counts,
    carrier_for_ggml, check_bits_equal, check_f32_against_f64, check_within, class_tensor,
    deterministic, f32_output_bytes, golden, ids_in_range, is_natively_decoded, logical_row_bytes,
    native_l0_weight, param_class_tensor, positive_scales, run_gates, scheme_weight_carrier,
    shape_fuzz, staging_class_tensor, state_class_tensor, symmetric_i8, tolerance_for,
    u32_output_bytes, uniform_f32, weight_class_tensor, BatchRows, GateBuffers, GateCase,
    HarnessError, ALL_CLASSES, CASES_PER_SHAPE, CLASS_COUNT, MASTER_SEED, MAX_BUCKET,
};
use r9v_t0::{
    act_mul, act_mul_f64_reference, activation, activation_f64_reference, all_gather, all_reduce,
    all_to_all, attention_paged, attention_row_f64_reference, barrier, cast, cast_f64_reference,
    causal_conv1d, causal_conv1d_f64_reference, concat, concat_f64_reference, copy, embed_gather,
    embed_gather_f64_reference, gather_rows, gather_rows_f64_reference, linear_attn_scan_chunked,
    logit_softcap, logit_softcap_f64_reference, logits_postprocess,
    logits_postprocess_f64_reference, matmul, matmul_f64_reference, matmul_with_scales, moe_ffn,
    moe_ffn_f64_reference, moe_route, moe_route_f64_reference, ngram_gather_f64_reference_staged,
    norm, norm_f64_reference, quant_act, quant_act_f64_reference, recv, reduce_scatter,
    residual_add, residual_add_f64_reference, rope, rope_f64_reference, sample, scatter_add_rows,
    scatter_add_rows_f64_reference, send, split, split_f64_reference, state_write_kv_paged, verify,
    RngState, SeqLayout, Tolerance, TypedBuffer,
};

// Pinned-identity stream base: far above any golden base
// (`shape_idx * 32 + case_idx`), so logical-row bytes never share a stream
// with golden draws (SI-59).
const PIN_BASE: u64 = u64::MAX - 65536;

fn pin_rng(op: &str, salt: u64) -> SeededRng {
    SeededRng::new(harness::seed_for(
        op,
        PIN_BASE.wrapping_add(salt),
        MASTER_SEED,
    ))
}

fn filler_rng(op: &str, rows: usize) -> SeededRng {
    SeededRng::new(harness::seed_for(
        op,
        PIN_BASE.wrapping_sub(1).wrapping_sub(rows as u64),
        MASTER_SEED,
    ))
}

fn gate_tol(op: &str) -> Tolerance {
    tolerance_for(op).unwrap_or_else(|e| panic!("tolerance row: {e:?}"))
}

fn gate_ok(result: Result<(), HarnessError>, context: &str) {
    if let Err(e) = result {
        panic!("{context}: {e:?}");
    }
}

/// Row-major `[t, n]` shapes over the bucket edges with tiny `n`.
fn rowwise_shapes(n: usize) -> Vec<Vec<usize>> {
    bucket_edge_counts().iter().map(|&t| vec![t, n]).collect()
}

fn rowwise_fuzz(n: usize) -> Vec<Vec<usize>> {
    vec![vec![3, n], vec![17, n]]
}

fn rowwise_batch(n: usize) -> BatchRows {
    BatchRows {
        alone: vec![1, n],
        padded: vec![8, n],
        embedded: vec![6, n],
        row_alone: 0,
        row: 2,
    }
}

/// Builds `[t, n]` f32 rows with rows `0..=row` pinned to the identity
/// stream and the rest from the shape-dependent filler stream.
///
/// The prefix form matters for history-dependent ops (causal conv, scans):
/// pinning only row `row` leaves different histories before it per mode,
/// which correctly changes the answer. Pinning the prefix keeps the
/// history identical so the logical row must match.
fn pinned_prefix_rows(op: &str, t: usize, n: usize, row: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut pinned = pin_rng(op, 0);
    let prefix = uniform_f32(&mut pinned, (row + 1) * n, lo, hi);
    let mut filler = filler_rng(op, t);
    let mut out = Vec::with_capacity(t * n);
    for r in 0..t {
        if r <= row {
            out.extend_from_slice(&prefix[r * n..(r + 1) * n]);
        } else {
            out.extend_from_slice(&uniform_f32(&mut filler, n, lo, hi));
        }
    }
    out
}

/// Builds `[t, n]` f32 rows with row `row` pinned to the identity stream.
fn pinned_rows(op: &str, t: usize, n: usize, row: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut pinned = pin_rng(op, 0);
    let logical = uniform_f32(&mut pinned, n, lo, hi);
    let mut filler = filler_rng(op, t);
    let mut out = Vec::with_capacity(t * n);
    for r in 0..t {
        if r == row {
            out.extend_from_slice(&logical);
        } else {
            out.extend_from_slice(&uniform_f32(&mut filler, n, lo, hi));
        }
    }
    out
}

fn f32_row_bytes(
    buf: &TypedBuffer,
    row: usize,
    cols: usize,
    context: &str,
) -> Result<Vec<u8>, HarnessError> {
    logical_row_bytes(&buf.to_f32_vec(), row, cols, context)
}

/// Serializes an F16 slice-backed buffer through f16 bits (exact for
/// on-grid values; `byte_data` stays empty for slice-backed buffers).
fn f16_output_bytes(buf: &TypedBuffer) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.num_elements() * 2);
    for v in buf.to_f32_vec() {
        out.extend_from_slice(&r9v_t0::f32_to_f16(v).to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Elementwise gates: copy, cast, activation, act_mul, logit_softcap,
// residual_add, norm (Spec 1 §4.B).
// ---------------------------------------------------------------------------

struct CopyGate;
impl GateCase for CopyGate {
    fn op_name(&self) -> &'static str {
        "copy"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("copy")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(4)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(4)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("copy", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        copy(
            &CopyOp {
                kind: CopyKind::Contiguize,
            },
            &buffers.inputs[0].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        // Oracle is the copy rule itself (identity): input bytes must equal
        // output bytes without calling the implementation again.
        let mut want = Vec::new();
        for v in buffers.inputs[0].to_f32_vec() {
            want.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        check_bits_equal(
            &f32_output_bytes(&buffers.outputs[0], "copy golden")?,
            &want,
            "copy golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "copy determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "copy invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(4)
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = TypedBuffer::from_f32(&[2, 4], &[0.5; 8]);
        let y = match index {
            0 => TypedBuffer::zeros(&[2, 3], DType::F32),
            _ => TypedBuffer::zeros(&[2, 4], DType::F16),
        };
        Ok(GateBuffers::fresh(vec![x], vec![y]))
    }
}

#[test]
fn gate_copy_through_engine() {
    gate_ok(run_gates(&CopyGate), "copy");
}

struct CastGate;
impl GateCase for CastGate {
    fn op_name(&self) -> &'static str {
        "cast"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("cast")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(4)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(4)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F16)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("cast", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F16)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        cast(
            &CastOp { dtype: DType::F16 },
            &buffers.inputs[0].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = cast_f64_reference(&x64);
        // Slice-backed F16 keeps u16s (`byte_data` is for `from_bytes`
        // buffers), so the oracle pairs against decoded f32 values.
        let got64: Vec<f64> = buffers.outputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        check_within(gate_tol("cast"), &got64, &expected, "cast golden")
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        Ok(f16_output_bytes(&buffers.outputs[0]))
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        let raw = f16_output_bytes(&buffers.outputs[0]);
        let start = buffers.logical_row.saturating_mul(n.saturating_mul(2));
        let end = start.saturating_add(n.saturating_mul(2));
        if end > raw.len() {
            return Err(HarnessError::LengthMismatch {
                context: "cast invariance".to_owned(),
                actual: end,
                expected: raw.len(),
            });
        }
        Ok(raw[start..end].to_vec())
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(4)
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = TypedBuffer::from_f32(&[2, 4], &[0.5; 8]);
        let y = match index {
            // Output dtype must match the op dtype (F16).
            0 => TypedBuffer::zeros(&[2, 4], DType::F32),
            // Output shape must match the input shape.
            _ => TypedBuffer::zeros(&[2, 3], DType::F16),
        };
        Ok(GateBuffers::fresh(vec![x], vec![y]))
    }
}

#[test]
fn gate_cast_through_engine() {
    gate_ok(run_gates(&CastGate), "cast");
}

struct ActivationGate;
impl GateCase for ActivationGate {
    fn op_name(&self) -> &'static str {
        "activation"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("activation")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(4)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(4)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("activation", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        activation(
            &ActivationOp {
                act: ActivationKind::Silu,
                clamp: None,
            },
            &buffers.inputs[0].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let op = ActivationOp {
            act: ActivationKind::Silu,
            clamp: None,
        };
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = activation_f64_reference(&op, &x64);
        check_f32_against_f64(
            gate_tol("activation"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "activation golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "activation determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "activation invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(4)
    }
    fn illegal_count(&self) -> usize {
        1
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[2, 4], &[0.5; 8])],
            vec![TypedBuffer::zeros(&[2, 3], DType::F32)],
        ))
    }
}

#[test]
fn gate_activation_through_engine() {
    gate_ok(run_gates(&ActivationGate), "activation");
}

struct ActMulGate;
impl GateCase for ActMulGate {
    fn op_name(&self) -> &'static str {
        "act_mul"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("act_mul")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(4)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(4)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let gate = activation_values(&mut rng, t * n);
        let up = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[t, n], &gate),
                TypedBuffer::from_f32(&[t, n], &up),
            ],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let gate = pinned_rows("act_mul_gate", t, n, row, -2.0, 2.0);
        let up = pinned_rows("act_mul_up", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[t, n], &gate),
                TypedBuffer::from_f32(&[t, n], &up),
            ],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        act_mul(
            &ActMulOp {
                act: ActivationKind::Silu,
                clamp: None,
            },
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let op = ActMulOp {
            act: ActivationKind::Silu,
            clamp: None,
        };
        let g64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let u64_: Vec<f64> = buffers.inputs[1]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = act_mul_f64_reference(&op, &g64, &u64_);
        check_f32_against_f64(
            gate_tol("act_mul"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "act_mul golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "act_mul determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "act_mul invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(4)
    }
    fn illegal_count(&self) -> usize {
        1
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[2, 4], &[0.5; 8]),
                TypedBuffer::from_f32(&[2, 3], &[0.5; 6]),
            ],
            vec![TypedBuffer::zeros(&[2, 4], DType::F32)],
        ))
    }
}

#[test]
fn gate_act_mul_through_engine() {
    gate_ok(run_gates(&ActMulGate), "act_mul");
}

struct SoftcapGate;
impl GateCase for SoftcapGate {
    fn op_name(&self) -> &'static str {
        "logit_softcap"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("logit_softcap")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(4)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(4)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("logit_softcap", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        logit_softcap(
            &LogitSoftcapOp { cap: 30.0 },
            &buffers.inputs[0].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = logit_softcap_f64_reference(&x64, 30.0);
        check_f32_against_f64(
            gate_tol("logit_softcap"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "softcap golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "softcap determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "softcap invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(4)
    }
    fn illegal_count(&self) -> usize {
        1
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[2, 4], &[0.5; 8])],
            vec![TypedBuffer::zeros(&[3, 4], DType::F32)],
        ))
    }
}

#[test]
fn gate_logit_softcap_through_engine() {
    gate_ok(run_gates(&SoftcapGate), "logit_softcap");
}

struct ResidualGate;
impl GateCase for ResidualGate {
    fn op_name(&self) -> &'static str {
        "residual_add"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("residual_add")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(4)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(4)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let a = activation_values(&mut rng, t * n);
        let b = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[t, n], &a),
                TypedBuffer::from_f32(&[t, n], &b),
            ],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let a = pinned_rows("residual_a", t, n, row, -2.0, 2.0);
        let b = pinned_rows("residual_b", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[t, n], &a),
                TypedBuffer::from_f32(&[t, n], &b),
            ],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        residual_add(
            &ResidualAddOp {
                out_dtype: DType::F32,
                scale: 1.0,
            },
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let a64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let b64: Vec<f64> = buffers.inputs[1]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = residual_add_f64_reference(&a64, &b64, 1.0);
        check_f32_against_f64(
            gate_tol("residual_add"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "residual golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "residual determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "residual invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(4)
    }
    fn illegal_count(&self) -> usize {
        1
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[2, 4], &[0.5; 8]),
                TypedBuffer::from_f32(&[2, 5], &[0.5; 10]),
            ],
            vec![TypedBuffer::zeros(&[2, 4], DType::F32)],
        ))
    }
}

#[test]
fn gate_residual_add_through_engine() {
    gate_ok(run_gates(&ResidualGate), "residual_add");
}

struct NormGate;
impl GateCase for NormGate {
    fn op_name(&self) -> &'static str {
        "norm"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("norm")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(8)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(8)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        let w = positive_scales(&mut rng, n);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[t, n], &x),
                TypedBuffer::from_f32(&[n], &w),
            ],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("norm", t, n, row, -2.0, 2.0);
        let mut wrng = pin_rng("norm_w", 0);
        let w = positive_scales(&mut wrng, n);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[t, n], &x),
                TypedBuffer::from_f32(&[n], &w),
            ],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        norm(
            &NormOp {
                kind: NormKind::Rms,
                eps: 1e-5,
                axis: NormAxis::Last,
                weight_offset: 0.0,
                out_dtype: DType::F32,
            },
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            None,
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let op = NormOp {
            kind: NormKind::Rms,
            eps: 1e-5,
            axis: NormAxis::Last,
            weight_offset: 0.0,
            out_dtype: DType::F32,
        };
        let t = buffers.inputs[0].shape()[0];
        let n = buffers.inputs[0].shape()[1];
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let w64: Vec<f64> = buffers.inputs[1]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = norm_f64_reference(&op, &x64, [t, n], &w64, None, 0.0, 1e-5);
        check_f32_against_f64(
            gate_tol("norm"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "norm golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "norm determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "norm invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(8)
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = TypedBuffer::from_f32(&[2, 8], &[0.5; 16]);
        let bad_w = match index {
            0 => TypedBuffer::from_f32(&[7], &[1.0; 7]),
            _ => TypedBuffer::from_f32(&[2, 8], &[1.0; 16]),
        };
        Ok(GateBuffers::fresh(
            vec![x, bad_w],
            vec![TypedBuffer::zeros(&[2, 8], DType::F32)],
        ))
    }
}

#[test]
fn gate_norm_through_engine() {
    gate_ok(run_gates(&NormGate), "norm");
}

// ---------------------------------------------------------------------------
// Reshape/index gates: quant_act, split, concat, rope, gather_rows,
// scatter_add_rows, embed_gather, ngram_gather (Spec 1 §4.A–§4.B).
// ---------------------------------------------------------------------------

struct QuantActGate;
impl GateCase for QuantActGate {
    fn op_name(&self) -> &'static str {
        "quant_act"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("quant_act")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(8)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(8)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![
                TypedBuffer::zeros(&[t, n], DType::I8),
                TypedBuffer::zeros(&[t], DType::F32),
            ],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("quant_act", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![
                TypedBuffer::zeros(&[t, n], DType::I8),
                TypedBuffer::zeros(&[t], DType::F32),
            ],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let (o0, o1) = buffers.outputs.as_mut_slice().split_at_mut(1);
        quant_act(
            &QuantActOp {
                scheme: QuantScheme::PerToken,
                target: DType::I8,
                smoothing: Smoothing::None,
            },
            &buffers.inputs[0].as_view(),
            &mut o0[0].as_view_mut(),
            &mut o1[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let op = QuantActOp {
            scheme: QuantScheme::PerToken,
            target: DType::I8,
            smoothing: Smoothing::None,
        };
        let t = buffers.inputs[0].shape()[0];
        let n = buffers.inputs[0].shape()[1];
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (exp_xq, exp_sc) = quant_act_f64_reference(&op, &x64, [t, n]);
        let got_sc = harness::buffer_to_f64(&buffers.outputs[1]);
        check_within(gate_tol("quant_act"), &got_sc, &exp_sc, "quant_act scales")?;
        // Codes agree exactly except at proven f32/f64 rounding ties:
        // the oracle rounds half-to-even in f64, the impl in f32, so a
        // value within 1e-4 (code units) of a half-integer boundary may
        // round either way. Only adjacent codes with that evidence pass;
        // anything else is a value error. (The bespoke golden checks
        // scales at the floor and only asserts the code count.)
        let got_xq = buffers.outputs[0].to_i8_vec();
        let tol = gate_tol("quant_act");
        let mut index = 0usize;
        for (((got_row, exp_row), x_row), &s) in got_xq
            .chunks_exact(n)
            .zip(exp_xq.chunks_exact(n))
            .zip(x64.chunks_exact(n))
            .zip(exp_sc.iter())
        {
            for ((&g, &e), &x) in got_row.iter().zip(exp_row.iter()).zip(x_row.iter()) {
                let (got, exp) = (f64::from(g), e);
                if got != exp {
                    let unquant = if s == 0.0 { 0.0 } else { x / s };
                    let to_half = (unquant.abs().fract() - 0.5).abs();
                    if !((got - exp).abs() == 1.0 && to_half < 1e-4) {
                        return Err(HarnessError::GoldenMismatch {
                            context: "quant_act codes".to_owned(),
                            index,
                            actual: got,
                            expected: exp,
                            abs: tol.abs,
                            rel: tol.rel,
                        });
                    }
                }
                index += 1;
            }
        }
        Ok(())
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let mut out = harness::i8_output_bytes(&buffers.outputs[0], "quant_act codes");
        out.extend_from_slice(&f32_output_bytes(&buffers.outputs[1], "quant_act scales")?);
        Ok(out)
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.inputs[0].shape()[1];
        let row = buffers.logical_row;
        let codes = buffers.outputs[0].to_i8_vec();
        let start = row.saturating_mul(n);
        let end = start.saturating_add(n);
        if end > codes.len() {
            return Err(HarnessError::LengthMismatch {
                context: "quant_act invariance".to_owned(),
                actual: end,
                expected: codes.len(),
            });
        }
        let mut out: Vec<u8> = codes[start..end].iter().map(|&v| v as u8).collect();
        out.extend_from_slice(&f32_row_bytes(
            &buffers.outputs[1],
            row,
            1,
            "quant_act scales",
        )?);
        Ok(out)
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(8)
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = TypedBuffer::from_f32(&[2, 8], &[0.5; 16]);
        let (xq, sc) = match index {
            0 => (
                TypedBuffer::zeros(&[2, 7], DType::I8),
                TypedBuffer::zeros(&[2], DType::F32),
            ),
            _ => (
                TypedBuffer::zeros(&[2, 8], DType::F32),
                TypedBuffer::zeros(&[2], DType::F32),
            ),
        };
        Ok(GateBuffers::fresh(vec![x], vec![xq, sc]))
    }
}

#[test]
fn gate_quant_act_through_engine() {
    gate_ok(run_gates(&QuantActGate), "quant_act");
}

struct SplitGate;
impl GateCase for SplitGate {
    fn op_name(&self) -> &'static str {
        "split"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("split")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * 8);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, 1, 8], &x)],
            vec![
                TypedBuffer::zeros(&[t, 1, 3], DType::F32),
                TypedBuffer::zeros(&[t, 1, 5], DType::F32),
            ],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let x = pinned_rows("split", t, 8, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, 1, 8], &x)],
            vec![
                TypedBuffer::zeros(&[t, 1, 3], DType::F32),
                TypedBuffer::zeros(&[t, 1, 5], DType::F32),
            ],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let (o0, o1) = buffers.outputs.as_mut_slice().split_at_mut(1);
        split(
            &SplitOp { first: 3 },
            &buffers.inputs[0].as_view(),
            &mut o0[0].as_view_mut(),
            &mut o1[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let t = buffers.inputs[0].shape()[0];
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (exp_a, exp_b) = split_f64_reference(&x64, [t, 1, 8], 3);
        check_within(
            Tolerance::exact(),
            &harness::buffer_to_f64(&buffers.outputs[0]),
            &exp_a,
            "split a",
        )?;
        check_within(
            Tolerance::exact(),
            &harness::buffer_to_f64(&buffers.outputs[1]),
            &exp_b,
            "split b",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let mut out = f32_output_bytes(&buffers.outputs[0], "split a")?;
        out.extend_from_slice(&f32_output_bytes(&buffers.outputs[1], "split b")?);
        Ok(out)
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let row = buffers.logical_row;
        let mut out = f32_row_bytes(&buffers.outputs[0], row, 3, "split a row")?;
        out.extend_from_slice(&f32_row_bytes(&buffers.outputs[1], row, 5, "split b row")?);
        Ok(out)
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = TypedBuffer::from_f32(&[2, 1, 8], &[0.5; 16]);
        let (a, b) = match index {
            0 => (
                TypedBuffer::zeros(&[2, 1, 2], DType::F32),
                TypedBuffer::zeros(&[2, 1, 5], DType::F32),
            ),
            _ => (
                TypedBuffer::zeros(&[2, 1, 3], DType::F32),
                TypedBuffer::zeros(&[2, 1, 4], DType::F32),
            ),
        };
        Ok(GateBuffers::fresh(vec![x], vec![a, b]))
    }
}

#[test]
fn gate_split_through_engine() {
    gate_ok(run_gates(&SplitGate), "split");
}

struct ConcatGate;
impl GateCase for ConcatGate {
    fn op_name(&self) -> &'static str {
        "concat"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("concat")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let a = activation_values(&mut rng, t * 3);
        let b = activation_values(&mut rng, t * 5);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[t, 1, 3], &a),
                TypedBuffer::from_f32(&[t, 1, 5], &b),
            ],
            vec![TypedBuffer::zeros(&[t, 1, 8], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let a = pinned_rows("concat_a", t, 3, row, -2.0, 2.0);
        let b = pinned_rows("concat_b", t, 5, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[t, 1, 3], &a),
                TypedBuffer::from_f32(&[t, 1, 5], &b),
            ],
            vec![TypedBuffer::zeros(&[t, 1, 8], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        concat(
            &ConcatOp,
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let t = buffers.inputs[0].shape()[0];
        let a64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let b64: Vec<f64> = buffers.inputs[1]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = concat_f64_reference(&a64, &b64, t, 1);
        check_within(
            Tolerance::exact(),
            &harness::buffer_to_f64(&buffers.outputs[0]),
            &expected,
            "concat golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "concat determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            8,
            "concat invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        1
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[2, 1, 3], &[0.5; 6]),
                TypedBuffer::from_f32(&[3, 1, 5], &[0.5; 15]),
            ],
            vec![TypedBuffer::zeros(&[2, 1, 8], DType::F32)],
        ))
    }
}

#[test]
fn gate_concat_through_engine() {
    gate_ok(run_gates(&ConcatGate), "concat");
}

fn rope_op() -> RopeOp {
    RopeOp {
        rot_dim: 8,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F32,
    }
}

struct RopeGate;
impl GateCase for RopeGate {
    fn op_name(&self) -> &'static str {
        "rope"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("rope")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * 8);
        let positions = ids_in_range(&mut rng, t, 128);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[t, 1, 8], &x),
                TypedBuffer::from_u32(&[t], &positions),
            ],
            vec![TypedBuffer::zeros(&[t, 1, 8], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let x = pinned_rows("rope", t, 8, row, -2.0, 2.0);
        // The logical row keeps its position value across modes (SI-61):
        // moving a row without its position changes the correct answer.
        let mut prng = pin_rng("rope_pos", 0);
        let logical_pos = ids_in_range(&mut prng, 1, 128)[0];
        let mut frng = filler_rng("rope_pos", t);
        let mut positions = ids_in_range(&mut frng, t, 128);
        positions[row] = logical_pos;
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[t, 1, 8], &x),
                TypedBuffer::from_u32(&[t], &positions),
            ],
            vec![TypedBuffer::zeros(&[t, 1, 8], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        rope(
            &rope_op(),
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let t = buffers.inputs[0].shape()[0];
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = rope_f64_reference(
            &rope_op(),
            &x64,
            [t, 1, 8],
            &buffers.inputs[1].to_u32_vec(),
            false,
        );
        check_f32_against_f64(
            gate_tol("rope"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "rope golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "rope determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            8,
            "rope invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = TypedBuffer::from_f32(&[2, 1, 8], &[0.5; 16]);
        let bad_pos = match index {
            // Positions must be u32 token positions, not f32.
            0 => TypedBuffer::from_f32(&[2], &[1.0, 2.0]),
            // Position count must match the token count.
            _ => TypedBuffer::from_u32(&[3], &[1, 2, 3]),
        };
        Ok(GateBuffers::fresh(
            vec![x, bad_pos],
            vec![TypedBuffer::zeros(&[2, 1, 8], DType::F32)],
        ))
    }
}

#[test]
fn gate_rope_through_engine() {
    gate_ok(run_gates(&RopeGate), "rope");
}

const GATE_N: usize = 8;
const GATE_D: usize = 4;

struct GatherGate;

/// Shared `[8, 4]` source table from the fixed stream: context is
/// identical across modes, only the gathered ids vary.
fn gather_table() -> (Vec<f32>, TypedBuffer) {
    let mut trng = pin_rng("gather_table", 0);
    let x = activation_values(&mut trng, GATE_N * GATE_D);
    let buf = TypedBuffer::from_f32(&[GATE_N, GATE_D], &x);
    (x, buf)
}
impl GateCase for GatherGate {
    fn op_name(&self) -> &'static str {
        "gather_rows"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("gather_rows")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&m| vec![m]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let m = shape[0];
        let mut rng = SeededRng::new(seed);
        let idx = ids_in_range(&mut rng, m, GATE_N as u32);
        let (_, table) = gather_table();
        Ok(GateBuffers::fresh(
            vec![table, TypedBuffer::from_u32(&[m], &idx)],
            vec![TypedBuffer::zeros(&[m, GATE_D], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let m = shape[0];
        let mut prng = pin_rng("gather_ids", 0);
        let logical_id = ids_in_range(&mut prng, 1, GATE_N as u32)[0];
        let mut frng = filler_rng("gather_ids", m);
        let mut idx = ids_in_range(&mut frng, m, GATE_N as u32);
        idx[row] = logical_id;
        // The gathered source row is pinned too, so the logical output row
        // is byte-identical across modes by construction of the oracle.
        let (mut x, _) = gather_table();
        let mut xrng = pin_rng("gather_ids", 1);
        let logical_row = activation_values(&mut xrng, GATE_D);
        x[logical_id as usize * GATE_D..(logical_id as usize + 1) * GATE_D]
            .copy_from_slice(&logical_row);
        let table = TypedBuffer::from_f32(&[GATE_N, GATE_D], &x);
        Ok(GateBuffers::pinned(
            vec![table, TypedBuffer::from_u32(&[m], &idx)],
            vec![TypedBuffer::zeros(&[m, GATE_D], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        gather_rows(
            &GatherRowsOp,
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected =
            gather_rows_f64_reference(&x64, GATE_N, GATE_D, &buffers.inputs[1].to_u32_vec());
        check_within(
            Tolerance::exact(),
            &harness::buffer_to_f64(&buffers.outputs[0]),
            &expected,
            "gather golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "gather determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            GATE_D,
            "gather invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let (_, table) = gather_table();
        let (ids, y) = match index {
            0 => (
                TypedBuffer::from_u32(&[2], &[0, 99]),
                TypedBuffer::zeros(&[2, GATE_D], DType::F32),
            ),
            _ => (
                TypedBuffer::from_u32(&[3], &[0, 1, 2]),
                TypedBuffer::zeros(&[2, GATE_D], DType::F32),
            ),
        };
        Ok(GateBuffers::fresh(vec![table, ids], vec![y]))
    }
}

#[test]
fn gate_gather_rows_through_engine() {
    gate_ok(run_gates(&GatherGate), "gather_rows");
}

struct ScatterGate;
impl GateCase for ScatterGate {
    fn op_name(&self) -> &'static str {
        "scatter_add_rows"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("scatter_add_rows")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&m| vec![m]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let m = shape[0];
        let mut rng = SeededRng::new(seed);
        let updates = activation_values(&mut rng, m * GATE_D);
        let idx = ids_in_range(&mut rng, m, GATE_N as u32);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[m, GATE_D], &updates),
                TypedBuffer::from_u32(&[m], &idx),
            ],
            vec![TypedBuffer::zeros(&[GATE_N, GATE_D], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let m = shape[0];
        let updates = pinned_rows("scatter_upd", m, GATE_D, row, -2.0, 2.0);
        let mut prng = pin_rng("scatter_ids", 0);
        let logical_id = ids_in_range(&mut prng, 1, GATE_N as u32)[0];
        let mut frng = filler_rng("scatter_ids", m);
        // Filler ids avoid the logical slot so no other row accumulates
        // into it; accumulation order over the remaining rows is ascending.
        let mut idx = ids_in_range(&mut frng, m, GATE_N as u32 - 1);
        for v in idx.iter_mut() {
            if *v >= logical_id {
                *v += 1;
            }
        }
        idx[row] = logical_id;
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[m, GATE_D], &updates),
                TypedBuffer::from_u32(&[m], &idx),
            ],
            vec![TypedBuffer::zeros(&[GATE_N, GATE_D], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        scatter_add_rows(
            &ScatterAddRowsOp,
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            None,
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let m = buffers.inputs[0].shape()[0];
        let u64_: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected = scatter_add_rows_f64_reference(
            &u64_,
            m,
            GATE_D,
            &buffers.inputs[1].to_u32_vec(),
            None,
            GATE_N,
        );
        check_within(
            gate_tol("scatter_add_rows"),
            &harness::buffer_to_f64(&buffers.outputs[0]),
            &expected,
            "scatter golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "scatter determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        // The logical slot holds exactly the pinned update row: no other
        // update targets it by construction.
        let logical_id = buffers.inputs[1].to_u32_vec()[buffers.logical_row] as usize;
        f32_row_bytes(
            &buffers.outputs[0],
            logical_id,
            GATE_D,
            "scatter invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let (updates, ids) = match index {
            0 => (
                TypedBuffer::from_f32(&[2, GATE_D], &[0.5; 8]),
                TypedBuffer::from_u32(&[2], &[0, 99]),
            ),
            _ => (
                TypedBuffer::from_f32(&[3, GATE_D], &[0.5; 12]),
                TypedBuffer::from_u32(&[2], &[0, 1]),
            ),
        };
        Ok(GateBuffers::fresh(
            vec![updates, ids],
            vec![TypedBuffer::zeros(&[GATE_N, GATE_D], DType::F32)],
        ))
    }
}

#[test]
fn gate_scatter_add_rows_through_engine() {
    gate_ok(run_gates(&ScatterGate), "scatter_add_rows");
}

/// Shared `[8, 4]` f16 table from the fixed stream.
fn embed_table() -> TypedBuffer {
    let mut trng = pin_rng("embed_table", 0);
    harness::f16_bytes_tensor(&mut trng, &[GATE_N, GATE_D])
}

struct EmbedGate;
impl GateCase for EmbedGate {
    fn op_name(&self) -> &'static str {
        "embed_gather"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("embed_gather")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let ids = ids_in_range(&mut rng, t, GATE_N as u32);
        Ok(GateBuffers::fresh(
            vec![embed_table(), TypedBuffer::from_u32(&[t], &ids)],
            vec![TypedBuffer::zeros(&[t, GATE_D], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut prng = pin_rng("embed_ids", 0);
        let logical_id = ids_in_range(&mut prng, 1, GATE_N as u32)[0];
        let mut frng = filler_rng("embed_ids", t);
        let mut ids = ids_in_range(&mut frng, t, GATE_N as u32);
        ids[row] = logical_id;
        Ok(GateBuffers::pinned(
            vec![embed_table(), TypedBuffer::from_u32(&[t], &ids)],
            vec![TypedBuffer::zeros(&[t, GATE_D], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        embed_gather(
            &EmbedGatherOp {
                scale: 1.0,
                out_dtype: DType::F32,
            },
            &buffers.inputs[1].as_view(),
            &buffers.inputs[0].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let table64 = harness::bytes_f16_to_f64(&buffers.inputs[0]);
        let expected = embed_gather_f64_reference(
            &buffers.inputs[1].to_u32_vec(),
            &table64,
            GATE_N,
            GATE_D,
            1.0,
        );
        check_f32_against_f64(
            gate_tol("embed_gather"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "embed golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "embed determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            GATE_D,
            "embed invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let (ids, y) = match index {
            0 => (
                TypedBuffer::from_u32(&[2], &[0, 99]),
                TypedBuffer::zeros(&[2, GATE_D], DType::F32),
            ),
            _ => (
                TypedBuffer::from_u32(&[3], &[0, 1, 2]),
                TypedBuffer::zeros(&[2, GATE_D], DType::F32),
            ),
        };
        Ok(GateBuffers::fresh(vec![embed_table(), ids], vec![y]))
    }
}

#[test]
fn gate_embed_gather_through_engine() {
    gate_ok(run_gates(&EmbedGate), "embed_gather");
}

use r9v_ir::Op as T0Op;
use r9v_t0::execute_ngram_op;

fn ngram_op() -> r9v_ir::NgramGatherOp {
    r9v_ir::NgramGatherOp {
        source: NgramSource::Staged,
        orders: vec![1; 2].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(0),
        table_sizes: vec![64; 2].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F32,
    }
}

struct NgramGate;
impl GateCase for NgramGate {
    fn op_name(&self) -> &'static str {
        "ngram_gather"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("ngram_gather")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let staging = symmetric_i8(&mut rng, t * 16);
        let scales = positive_scales(&mut rng, t * 2);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_i8(&[t, 2, 8], &staging)
                    .with_quant(QuantScheme::Scheme(SchemeId::I8R.to_ir())),
                TypedBuffer::from_f32(&[t, 2], &scales),
            ],
            vec![TypedBuffer::zeros(&[t, 16], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut prng = pin_rng("ngram", 0);
        let logical_staging = symmetric_i8(&mut prng, 16);
        let logical_scales = positive_scales(&mut prng, 2);
        let mut frng = filler_rng("ngram", t);
        let mut staging = symmetric_i8(&mut frng, t * 16);
        let mut scales = positive_scales(&mut frng, t * 2);
        staging[row * 16..(row + 1) * 16].copy_from_slice(&logical_staging);
        scales[row * 2..(row + 1) * 2].copy_from_slice(&logical_scales);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_i8(&[t, 2, 8], &staging)
                    .with_quant(QuantScheme::Scheme(SchemeId::I8R.to_ir())),
                TypedBuffer::from_f32(&[t, 2], &scales),
            ],
            vec![TypedBuffer::zeros(&[t, 16], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        execute_ngram_op(
            &T0Op::NgramGather(ngram_op()),
            &[buffers.inputs[0].as_view(), buffers.inputs[1].as_view()],
            &mut [buffers.outputs[0].as_view_mut()],
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let t = buffers.inputs[0].shape()[0];
        let staging = buffers.inputs[0].to_i8_vec();
        let scales64: Vec<f64> = buffers.inputs[1]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected =
            ngram_gather_f64_reference_staged(&staging, &scales64, t, 2, 8, NgramCombine::Concat)
                .map_err(|e| HarnessError::UnexpectedRefusal {
                context: "ngram oracle".to_owned(),
                error: format!("{e:?}"),
            })?;
        check_f32_against_f64(
            gate_tol("ngram_gather"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "ngram golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "ngram determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            16,
            "ngram invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let (staging, scales) = match index {
            0 => (
                TypedBuffer::from_f32(&[2, 2, 8], &[0.5; 32]),
                TypedBuffer::from_f32(&[2, 2], &[1.0; 4]),
            ),
            _ => (
                TypedBuffer::from_i8(&[2, 2, 8], &[1i8; 32])
                    .with_quant(QuantScheme::Scheme(SchemeId::I8R.to_ir())),
                TypedBuffer::from_f32(&[2, 3], &[1.0; 6]),
            ),
        };
        Ok(GateBuffers::fresh(
            vec![staging, scales],
            vec![TypedBuffer::zeros(&[2, 16], DType::F32)],
        ))
    }
}

#[test]
fn gate_ngram_gather_through_engine() {
    gate_ok(run_gates(&NgramGate), "ngram_gather");
}

// ---------------------------------------------------------------------------
// Matmul + MoE gates (Spec 1 §4.A, §4.C).
// ---------------------------------------------------------------------------

const GATE_K: usize = 16;
const GATE_NN: usize = 8;

fn gate_matmul_op() -> MatmulOp {
    MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    }
}

/// Shared `[8, 16]` f16 weight from the fixed stream.
fn matmul_weight() -> TypedBuffer {
    let mut wrng = pin_rng("matmul_w", 0);
    harness::f16_bytes_tensor(&mut wrng, &[GATE_NN, GATE_K])
}

struct MatmulGate;
impl GateCase for MatmulGate {
    fn op_name(&self) -> &'static str {
        "matmul"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("matmul")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&m| vec![m]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let m = shape[0];
        let mut rng = SeededRng::new(seed);
        let x = harness::activation_tensor_as(&mut rng, &[m, GATE_K], DType::F16);
        Ok(GateBuffers::fresh(
            vec![x, matmul_weight()],
            vec![TypedBuffer::zeros(&[m, GATE_NN], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let m = shape[0];
        // f16 values pin through the f32 pre-image so golden pairing stays
        // in the same value family as `activation_tensor_as`.
        let pre = pinned_rows("matmul_x", m, GATE_K, row, -2.0, 2.0);
        let bits: Vec<u16> = pre.iter().map(|&v| r9v_t0::f32_to_f16(v)).collect();
        let mut bytes = Vec::with_capacity(m * GATE_K * 2);
        for b in bits {
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        let x = TypedBuffer::from_bytes(&[m, GATE_K], DType::F16, &bytes);
        Ok(GateBuffers::pinned(
            vec![x, matmul_weight()],
            vec![TypedBuffer::zeros(&[m, GATE_NN], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        matmul(
            &gate_matmul_op(),
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            None,
            None,
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let m = buffers.inputs[0].shape()[0];
        let x64 = harness::buffer_to_f64(&buffers.inputs[0]);
        let w64 = harness::bytes_f16_to_f64(&buffers.inputs[1]);
        let expected = matmul_f64_reference(
            &x64,
            m,
            GATE_K,
            &w64,
            GATE_NN,
            None,
            None,
            Epilogue::None,
            false,
        );
        check_f32_against_f64(
            gate_tol("matmul"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "matmul golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "matmul determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            GATE_NN,
            "matmul invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let mut rng = SeededRng::new(7);
        let x = harness::activation_tensor_as(&mut rng, &[2, GATE_K], DType::F16);
        let (w, y) = match index {
            // K mismatch between activation and weight.
            0 => (
                harness::f16_bytes_tensor(&mut rng, &[GATE_NN, GATE_K - 1]),
                TypedBuffer::zeros(&[2, GATE_NN], DType::F32),
            ),
            // Output width must match N.
            _ => (
                matmul_weight(),
                TypedBuffer::zeros(&[2, GATE_NN - 1], DType::F32),
            ),
        };
        Ok(GateBuffers::fresh(vec![x, w], vec![y]))
    }
}

#[test]
fn gate_matmul_through_engine() {
    gate_ok(run_gates(&MatmulGate), "matmul");
}

fn moe_route_op() -> MoeRouteOp {
    MoeRouteOp {
        top_k: 2,
        scoring: MoeScoring::Softmax,
        renormalize: true,
        group: None,
        scale: 1.0,
    }
}

struct MoeRouteGate;
impl GateCase for MoeRouteGate {
    fn op_name(&self) -> &'static str {
        "moe_route"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("moe_route")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let logits = activation_values(&mut rng, t * 4);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, 4], &logits)],
            vec![
                TypedBuffer::zeros(&[t, 2], DType::U32),
                TypedBuffer::zeros(&[t, 2], DType::F32),
            ],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let logits = pinned_rows("moe_route", t, 4, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, 4], &logits)],
            vec![
                TypedBuffer::zeros(&[t, 2], DType::U32),
                TypedBuffer::zeros(&[t, 2], DType::F32),
            ],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let (o0, o1) = buffers.outputs.as_mut_slice().split_at_mut(1);
        moe_route(
            &moe_route_op(),
            &buffers.inputs[0].as_view(),
            None,
            &mut o0[0].as_view_mut(),
            &mut o1[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let t = buffers.inputs[0].shape()[0];
        let l64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (exp_ids, exp_w) =
            moe_route_f64_reference(&l64, t, 4, None, 2, MoeScoring::Softmax, true, 1.0).map_err(
                |e| HarnessError::UnexpectedRefusal {
                    context: "moe_route oracle".to_owned(),
                    error: format!("{e:?}"),
                },
            )?;
        let mut want_ids = Vec::with_capacity(t * 2 * 4);
        for id in &exp_ids {
            want_ids.extend_from_slice(&id.to_le_bytes());
        }
        check_bits_equal(
            &u32_output_bytes(&buffers.outputs[0], "route ids")?,
            &want_ids,
            "route ids",
        )?;
        check_within(
            gate_tol("moe_route"),
            &harness::buffer_to_f64(&buffers.outputs[1]),
            &exp_w,
            "route weights",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let mut out = u32_output_bytes(&buffers.outputs[0], "route ids")?;
        out.extend_from_slice(&f32_output_bytes(&buffers.outputs[1], "route weights")?);
        Ok(out)
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let row = buffers.logical_row;
        let ids = buffers.outputs[0].to_u32_vec();
        let mut out = Vec::new();
        for id in &ids[row * 2..row * 2 + 2] {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&f32_row_bytes(
            &buffers.outputs[1],
            row,
            2,
            "route weights",
        )?);
        Ok(out)
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let logits = TypedBuffer::from_f32(&[2, 4], &[0.5; 8]);
        let (ids, weights) = match index {
            // top_k slots must be 2.
            0 => (
                TypedBuffer::zeros(&[2, 1], DType::U32),
                TypedBuffer::zeros(&[2, 2], DType::F32),
            ),
            // Weights must be f32.
            _ => (
                TypedBuffer::zeros(&[2, 2], DType::U32),
                TypedBuffer::zeros(&[2, 2], DType::F16),
            ),
        };
        Ok(GateBuffers::fresh(vec![logits], vec![ids, weights]))
    }
}

#[test]
fn gate_moe_route_through_engine() {
    gate_ok(run_gates(&MoeRouteGate), "moe_route");
}

fn moe_ffn_op() -> MoeFfnOp {
    MoeFfnOp {
        act: ActivationKind::Silu,
        out_dtype: DType::F32,
        shared_experts: 0,
    }
}

/// Shared expert weights from the fixed stream (f32 pre-image for the
/// oracle, f16 bytes for the implementation).
fn moe_experts() -> (Vec<f32>, Vec<f32>, TypedBuffer, TypedBuffer) {
    let mut erng = pin_rng("moe_ffn_exp", 0);
    let gu = activation_values(&mut erng, 2 * 8 * 4);
    let wd = activation_values(&mut erng, 2 * 4 * 4);
    let gu_bytes: Vec<u8> = gu
        .iter()
        .flat_map(|&v| r9v_t0::f32_to_f16(v).to_le_bytes())
        .collect();
    let wd_bytes: Vec<u8> = wd
        .iter()
        .flat_map(|&v| r9v_t0::f32_to_f16(v).to_le_bytes())
        .collect();
    let gu_buf = TypedBuffer::from_bytes(&[2, 8, 4], DType::F16, &gu_bytes);
    let wd_buf = TypedBuffer::from_bytes(&[2, 4, 4], DType::F16, &wd_bytes);
    (gu, wd, gu_buf, wd_buf)
}

struct MoeFfnGate;
impl GateCase for MoeFfnGate {
    fn op_name(&self) -> &'static str {
        "moe_ffn"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("moe_ffn")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let x = harness::activation_tensor_as(&mut rng, &[t, 4], DType::F16);
        let ids = ids_in_range(&mut rng, t, 2);
        let weights = positive_scales(&mut rng, t);
        let (_, _, gu_buf, wd_buf) = moe_experts();
        Ok(GateBuffers::fresh(
            vec![
                x,
                TypedBuffer::from_u32(&[t, 1], &ids),
                TypedBuffer::from_f32(&[t, 1], &weights),
                gu_buf,
                wd_buf,
            ],
            vec![TypedBuffer::zeros(&[t, 4], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let pre = pinned_rows("moe_ffn_x", t, 4, row, -2.0, 2.0);
        let bits: Vec<u16> = pre.iter().map(|&v| r9v_t0::f32_to_f16(v)).collect();
        let mut bytes = Vec::with_capacity(t * 8);
        for b in &bits {
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        let x = TypedBuffer::from_bytes(&[t, 4], DType::F16, &bytes);
        let mut prng = pin_rng("moe_ffn_ids", 0);
        let logical_id = ids_in_range(&mut prng, 1, 2)[0];
        let logical_w = positive_scales(&mut prng, 1)[0];
        let mut frng = filler_rng("moe_ffn_ids", t);
        let mut ids = ids_in_range(&mut frng, t, 2);
        let mut weights = positive_scales(&mut frng, t);
        ids[row] = logical_id;
        weights[row] = logical_w;
        let (_, _, gu_buf, wd_buf) = moe_experts();
        Ok(GateBuffers::pinned(
            vec![
                x,
                TypedBuffer::from_u32(&[t, 1], &ids),
                TypedBuffer::from_f32(&[t, 1], &weights),
                gu_buf,
                wd_buf,
            ],
            vec![TypedBuffer::zeros(&[t, 4], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        moe_ffn(
            &moe_ffn_op(),
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &buffers.inputs[2].as_view(),
            &buffers.inputs[3].as_view(),
            None,
            &buffers.inputs[4].as_view(),
            None,
            None,
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let t = buffers.inputs[0].shape()[0];
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let w64: Vec<f64> = buffers.inputs[2]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        // Oracle over the f16-rounded weights T0 actually decodes, not the
        // f32 pre-image: grid rounding is fixture error, not impl error.
        let gu64 = harness::bytes_f16_to_f64(&buffers.inputs[3]);
        let wd64 = harness::bytes_f16_to_f64(&buffers.inputs[4]);
        let expected = moe_ffn_f64_reference(
            &x64,
            t,
            4,
            &buffers.inputs[1].to_u32_vec(),
            &w64,
            1,
            &gu64,
            2,
            4,
            &wd64,
            ActivationKind::Silu,
        )
        .map_err(|e| HarnessError::UnexpectedRefusal {
            context: "moe_ffn oracle".to_owned(),
            error: format!("{e:?}"),
        })?;
        check_f32_against_f64(
            gate_tol("moe_ffn"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "moe_ffn golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "moe_ffn determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            4,
            "moe_ffn invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let mut rng = SeededRng::new(11);
        let x = harness::activation_tensor_as(&mut rng, &[2, 4], DType::F16);
        let (_, _, gu_buf, wd_buf) = moe_experts();
        let (ids, weights) = match index {
            // Expert id 5 does not exist (E = 2).
            0 => (
                TypedBuffer::from_u32(&[2, 1], &[0, 5]),
                TypedBuffer::from_f32(&[2, 1], &[0.5, 0.5]),
            ),
            // Routing count must be k = 1.
            _ => (
                TypedBuffer::from_u32(&[2, 2], &[0, 1, 1, 0]),
                TypedBuffer::from_f32(&[2, 1], &[0.5, 0.5]),
            ),
        };
        Ok(GateBuffers::fresh(
            vec![x, ids, weights, gu_buf, wd_buf],
            vec![TypedBuffer::zeros(&[2, 4], DType::F32)],
        ))
    }
}

#[test]
fn gate_moe_ffn_through_engine() {
    gate_ok(run_gates(&MoeFfnGate), "moe_ffn");
}

// ---------------------------------------------------------------------------
// Stateful gates: state_write_kv, attention (Spec 1 §4.D, Spec 3 §3).
//
// Caches are reconstructed from the input buffers on every `execute`
// call, so the determinism gate proves fresh-state determinism. The
// attention oracle runs over the cache-readback snapshot (outputs[1]),
// exactly like the bespoke MLA golden: values on the f16 grid are never
// compared against pre-quant floats.
// ---------------------------------------------------------------------------

use r9v_t0::KvPagedCache;

const GATE_HKV: usize = 1;
const GATE_AD: usize = 8;
const GATE_ADV: usize = 8;

fn kv_write_op() -> StateWriteKvOp {
    StateWriteKvOp {
        cache_dtype: DType::F16,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: None,
        handle: StateHandle::new(0, StateKind::KvPaged),
    }
}

/// Single-sequence write meta for `rows` rows with ascending slots.
fn write_meta(rows: usize) -> BatchMeta {
    let mb = rows.div_ceil(32).max(1) as u32;
    let need = rows.div_ceil(32);
    let mut table = vec![u32::MAX; mb as usize];
    for (block, slot) in table.iter_mut().enumerate().take(need) {
        *slot = block as u32;
    }
    let slots: Vec<u32> = (0..rows as u32).collect();
    BatchMeta::builder(1, 1, rows as u32, mb)
        .seq_ids(vec![7])
        .query_len(vec![rows as u32])
        .ctx_len(vec![0])
        .positions(Positions::PerToken(vec![0u32; rows]))
        .slot_map(slots)
        .block_table(table)
        .window_start(vec![0])
        .tree(None)
        .build()
        .expect("gate write meta builds")
}

struct StateWriteKvGate;
impl GateCase for StateWriteKvGate {
    fn op_name(&self) -> &'static str {
        "state_write_kv"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("state_write_kv")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&r| vec![r]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let r = shape[0];
        let mut rng = SeededRng::new(seed);
        let k = activation_values(&mut rng, r * GATE_AD);
        let v = activation_values(&mut rng, r * GATE_ADV);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[r, GATE_HKV, GATE_AD], &k),
                TypedBuffer::from_f32(&[r, GATE_HKV, GATE_ADV], &v),
            ],
            vec![TypedBuffer::zeros(&[r, GATE_AD + GATE_ADV], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let r = shape[0];
        let k = pinned_rows("swkv_k", r, GATE_AD, row, -2.0, 2.0);
        let v = pinned_rows("swkv_v", r, GATE_ADV, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[r, GATE_HKV, GATE_AD], &k),
                TypedBuffer::from_f32(&[r, GATE_HKV, GATE_ADV], &v),
            ],
            vec![TypedBuffer::zeros(&[r, GATE_AD + GATE_ADV], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let r = buffers.inputs[0].shape()[0];
        let meta = write_meta(r);
        let mb = r.div_ceil(32).max(1);
        let mut cache = KvPagedCache::new(mb, GATE_HKV, GATE_AD, GATE_ADV, DType::F16)
            .expect("gate cache builds");
        state_write_kv_paged(
            &kv_write_op(),
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &meta,
            0,
            &mut cache,
        )?;
        // Snapshot the written slots back into outputs[0] in write order.
        let snap = &mut buffers.outputs[0];
        let mut flat = snap.to_f32_vec();
        for p in 0..r {
            for dd in 0..GATE_AD {
                flat[p * (GATE_AD + GATE_ADV) + dd] =
                    cache.read_k_f32(p, 0, dd).expect("gate read k");
            }
            for dd in 0..GATE_ADV {
                flat[p * (GATE_AD + GATE_ADV) + GATE_AD + dd] =
                    cache.read_v_f32(p, 0, dd).expect("gate read v");
            }
        }
        let fresh = TypedBuffer::from_f32(&[r, GATE_AD + GATE_ADV], &flat);
        *snap = fresh;
        Ok(())
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let k64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let v64: Vec<f64> = buffers.inputs[1]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let snap = buffers.outputs[0].to_f32_vec();
        let r = buffers.inputs[0].shape()[0];
        let mut got_k = Vec::with_capacity(r * GATE_AD);
        let mut got_v = Vec::with_capacity(r * GATE_ADV);
        for p in 0..r {
            got_k.extend_from_slice(&snap[p * 16..p * 16 + 8]);
            got_v.extend_from_slice(&snap[p * 16 + 8..p * 16 + 16]);
        }
        let got_k64: Vec<f64> = got_k.iter().map(|&v| v as f64).collect();
        let got_v64: Vec<f64> = got_v.iter().map(|&v| v as f64).collect();
        check_within(Tolerance::f16_bf16(), &got_k64, &k64, "swkv k roundtrip")?;
        check_within(Tolerance::f16_bf16(), &got_v64, &v64, "swkv v roundtrip")
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "swkv determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            GATE_AD + GATE_ADV,
            "swkv invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let (k, v) = match index {
            // K and V row counts must agree.
            0 => (
                TypedBuffer::from_f32(&[2, GATE_HKV, GATE_AD], &[0.5; 16]),
                TypedBuffer::from_f32(&[3, GATE_HKV, GATE_ADV], &[0.5; 24]),
            ),
            // K must be rank 3 [T, Hkv, D].
            _ => (
                TypedBuffer::from_f32(&[2, GATE_AD], &[0.5; 16]),
                TypedBuffer::from_f32(&[2, GATE_HKV, GATE_ADV], &[0.5; 16]),
            ),
        };
        Ok(GateBuffers::fresh(
            vec![k, v],
            vec![TypedBuffer::zeros(&[2, GATE_AD + GATE_ADV], DType::F32)],
        ))
    }
}

#[test]
fn gate_state_write_kv_through_engine() {
    gate_ok(run_gates(&StateWriteKvGate), "state_write_kv");
}

fn causal_attn_op() -> AttentionOp {
    AttentionOp {
        softmax_scale: 0.25,
        mask: AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: None,
        out_dtype: DType::F32,
        handle: StateHandle::new(0, StateKind::KvPaged),
    }
}

fn attn_rows(shape: &[usize]) -> (usize, usize, usize) {
    let (s, ctx) = (shape[0], shape[1]);
    (s, ctx, s * (ctx + 1))
}

struct AttentionGate;
impl GateCase for AttentionGate {
    fn op_name(&self) -> &'static str {
        "attention"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("attention")
    }
    /// `[S, ctx]`: S decode sequences with `qlen = 1` and `ctx` prefix
    /// rows each. S covers the bucket edges including 4096 with tiny ctx.
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![1, 0], vec![5, 1], vec![33, 3], vec![MAX_BUCKET, 2]]
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3, 1], vec![7, 0]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (s, _ctx, r) = attn_rows(shape);
        let mut rng = SeededRng::new(seed);
        let q = activation_values(&mut rng, s * GATE_AD);
        let kf = activation_values(&mut rng, r * GATE_AD);
        let vf = activation_values(&mut rng, r * GATE_ADV);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[s, 1, GATE_AD], &q),
                TypedBuffer::from_f32(&[r, 1, GATE_AD], &kf),
                TypedBuffer::from_f32(&[r, 1, GATE_ADV], &vf),
            ],
            vec![
                TypedBuffer::zeros(&[s, 1, GATE_ADV], DType::F32),
                TypedBuffer::zeros(&[r, GATE_AD + GATE_ADV], DType::F32),
            ],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (s, ctx, r) = attn_rows(shape);
        let q = pinned_rows("attn_q", s, GATE_AD, row, -2.0, 2.0);
        // The logical sequence's KV rows are pinned; filler sequences use
        // the shape-dependent filler stream.
        let mut prng = pin_rng("attn_kv", 0);
        let logical_k = activation_values(&mut prng, (ctx + 1) * GATE_AD);
        let logical_v = activation_values(&mut prng, (ctx + 1) * GATE_ADV);
        let mut frng = filler_rng("attn_kv", s * 7 + ctx);
        let mut kf = activation_values(&mut frng, r * GATE_AD);
        let mut vf = activation_values(&mut frng, r * GATE_ADV);
        kf[row * (ctx + 1) * GATE_AD..(row + 1) * (ctx + 1) * GATE_AD].copy_from_slice(&logical_k);
        vf[row * (ctx + 1) * GATE_ADV..(row + 1) * (ctx + 1) * GATE_ADV]
            .copy_from_slice(&logical_v);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[s, 1, GATE_AD], &q),
                TypedBuffer::from_f32(&[r, 1, GATE_AD], &kf),
                TypedBuffer::from_f32(&[r, 1, GATE_ADV], &vf),
            ],
            vec![
                TypedBuffer::zeros(&[s, 1, GATE_ADV], DType::F32),
                TypedBuffer::zeros(&[r, GATE_AD + GATE_ADV], DType::F32),
            ],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let s = buffers.inputs[0].shape()[0];
        let r = buffers.inputs[1].shape()[0];
        let ctx = r.checked_div(s.max(1)).unwrap_or(0).saturating_sub(1);
        // Write phase: S sequences of (ctx+1) rows, one 32-slot block per
        // sequence, so every sequence's rows are block-aligned and
        // `block_table[seq, 0] == seq` maps positions to slots exactly.
        let wmeta = {
            let mut slots = Vec::with_capacity(r);
            for i in 0..s {
                for k in 0..ctx + 1 {
                    slots.push(i as u32 * 32 + k as u32);
                }
            }
            BatchMeta::builder(1, s as u32, r as u32, 1)
                .seq_ids(vec![7; s])
                .query_len(vec![ctx as u32 + 1; s])
                .ctx_len(vec![0; s])
                .positions(Positions::PerToken(vec![0u32; r]))
                .slot_map(slots)
                .block_table((0..s as u32).collect())
                .window_start(vec![0; s])
                .tree(None)
                .build()
                .expect("gate write meta builds")
        };
        let mut cache = KvPagedCache::new(s.max(1), GATE_HKV, GATE_AD, GATE_ADV, DType::F16)
            .expect("gate cache builds");
        state_write_kv_paged(
            &kv_write_op(),
            &buffers.inputs[1].as_view(),
            &buffers.inputs[2].as_view(),
            &wmeta,
            0,
            &mut cache,
        )?;
        // Decode phase: S sequences of qlen 1 over their own blocks.
        let mut slots = Vec::with_capacity(s);
        let mut seq_ids = Vec::with_capacity(s);
        for i in 0..s {
            slots.push(i as u32 * 32 + ctx as u32);
            seq_ids.push(if i == buffers.logical_row {
                7
            } else {
                100 + i as u32
            });
        }
        let dmeta = BatchMeta::builder(1, s as u32, s as u32, 1)
            .seq_ids(seq_ids)
            .query_len(vec![1; s])
            .ctx_len(vec![ctx as u32; s])
            .positions(Positions::PerToken(vec![0u32; s]))
            .slot_map(slots)
            .block_table((0..s as u32).collect())
            .window_start(vec![0; s])
            .tree(None)
            .build()
            .expect("gate decode meta builds");
        attention_paged(
            &causal_attn_op(),
            &buffers.inputs[0].as_view(),
            &dmeta,
            0,
            &cache,
            &mut buffers.outputs[0].as_view_mut(),
        )?;
        // Snapshot the cache rows backing the oracle, in per-sequence
        // row order (sequence i owns block i, rows [i*32, i*32+ctx]).
        let mut flat = buffers.outputs[1].to_f32_vec();
        for i in 0..s {
            for k in 0..ctx + 1 {
                let slot = i * 32 + k;
                let base = (i * (ctx + 1) + k) * 16;
                for dd in 0..GATE_AD {
                    flat[base + dd] = cache.read_k_f32(slot, 0, dd).expect("gate read k");
                }
                for dd in 0..GATE_ADV {
                    flat[base + GATE_AD + dd] = cache.read_v_f32(slot, 0, dd).expect("gate read v");
                }
            }
        }
        let fresh = TypedBuffer::from_f32(&[r, GATE_AD + GATE_ADV], &flat);
        buffers.outputs[1] = fresh;
        Ok(())
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let s = buffers.inputs[0].shape()[0];
        let r = buffers.inputs[1].shape()[0];
        let ctx = r.checked_div(s.max(1)).unwrap_or(0).saturating_sub(1);
        let snap = buffers.outputs[1].to_f32_vec();
        let qflat = buffers.inputs[0].to_f32_vec();
        let mut expected = Vec::with_capacity(s * GATE_ADV);
        for i in 0..s {
            let mut ks: Vec<Vec<f64>> = Vec::with_capacity(ctx + 1);
            let mut vs: Vec<Vec<f64>> = Vec::with_capacity(ctx + 1);
            for p in 0..ctx + 1 {
                let base = (i * (ctx + 1) + p) * 16;
                ks.push(snap[base..base + 8].iter().map(|&v| v as f64).collect());
                vs.push(
                    snap[base + 8..base + 16]
                        .iter()
                        .map(|&v| v as f64)
                        .collect(),
                );
            }
            let q_row: Vec<f64> = qflat[i * 8..(i + 1) * 8]
                .iter()
                .map(|&v| v as f64)
                .collect();
            expected.extend_from_slice(&attention_row_f64_reference(&q_row, &ks, &vs, 0.25, None));
        }
        check_f32_against_f64(
            gate_tol("attention"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "attention golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let mut out = f32_output_bytes(&buffers.outputs[0], "attention o")?;
        out.extend_from_slice(&f32_output_bytes(&buffers.outputs[1], "attention snap")?);
        Ok(out)
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            GATE_ADV,
            "attention invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        // All three modes use ctx = 1: the logical sequence attends the
        // same two rows everywhere (modes with different ctx would
        // correctly change the answer).
        BatchRows {
            alone: vec![1, 1],
            padded: vec![8, 1],
            embedded: vec![6, 1],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let (s, _ctx, r) = (2usize, 1usize, 4usize);
        let q = match index {
            // Q head dim must match the cache D.
            0 => TypedBuffer::from_f32(&[s, 1, GATE_AD + 1], &[0.5; 18]),
            // Q must be rank 3 [T, H, D].
            _ => TypedBuffer::from_f32(&[s, GATE_AD], &[0.5; 16]),
        };
        Ok(GateBuffers::fresh(
            vec![
                q,
                TypedBuffer::from_f32(&[r, 1, GATE_AD], &[0.5; 32]),
                TypedBuffer::from_f32(&[r, 1, GATE_ADV], &[0.5; 32]),
            ],
            vec![
                TypedBuffer::zeros(&[s, 1, GATE_ADV], DType::F32),
                TypedBuffer::zeros(&[r, 16], DType::F32),
            ],
        ))
    }
}

#[test]
fn gate_attention_through_engine() {
    gate_ok(run_gates(&AttentionGate), "attention");
}

// ---------------------------------------------------------------------------
// Scan gates: causal_conv1d, linear_attn_scan (Spec 1 §4.E).
// ---------------------------------------------------------------------------

const GATE_C: usize = 4;
const GATE_WK: usize = 3;

fn conv_op() -> CausalConv1dOp {
    CausalConv1dOp {
        kernel: GATE_WK as u32,
        act: ConvActivation::Identity,
        handle: StateHandle::new(0, StateKind::ConvWindow),
    }
}

struct ConvGate;
impl GateCase for ConvGate {
    fn op_name(&self) -> &'static str {
        "causal_conv1d"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("causal_conv1d")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * GATE_C);
        let w = activation_values(&mut rng, GATE_C * GATE_WK);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[t, GATE_C], &x),
                TypedBuffer::from_f32(&[GATE_C, GATE_WK], &w),
                TypedBuffer::zeros(&[1, GATE_WK - 1, GATE_C], DType::F16),
            ],
            vec![
                TypedBuffer::zeros(&[t, GATE_C], DType::F32),
                TypedBuffer::zeros(&[1, GATE_WK - 1, GATE_C], DType::F16),
            ],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let x = pinned_prefix_rows("conv_x", t, GATE_C, row, -2.0, 2.0);
        let mut wrng = pin_rng("conv_w", 0);
        let w = activation_values(&mut wrng, GATE_C * GATE_WK);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[t, GATE_C], &x),
                TypedBuffer::from_f32(&[GATE_C, GATE_WK], &w),
                TypedBuffer::zeros(&[1, GATE_WK - 1, GATE_C], DType::F16),
            ],
            vec![
                TypedBuffer::zeros(&[t, GATE_C], DType::F32),
                TypedBuffer::zeros(&[1, GATE_WK - 1, GATE_C], DType::F16),
            ],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let t = buffers.inputs[0].shape()[0];
        let seq = SeqLayout::new(&[t as u32]).expect("gate layout builds");
        let (o0, o1) = buffers.outputs.as_mut_slice().split_at_mut(1);
        causal_conv1d(
            &conv_op(),
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            None,
            &buffers.inputs[2].as_view(),
            &seq,
            &mut o0[0].as_view_mut(),
            &mut o1[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let t = buffers.inputs[0].shape()[0];
        let x64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let w64: Vec<f64> = buffers.inputs[1]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let zeros = vec![0.0f64; (GATE_WK - 1) * GATE_C];
        let (expected, _) = causal_conv1d_f64_reference(
            &x64,
            t,
            GATE_C,
            &w64,
            GATE_WK,
            None,
            ConvActivation::Identity,
            &zeros,
            1,
            &[t as u32],
        )
        .map_err(|e| HarnessError::UnexpectedRefusal {
            context: "conv oracle".to_owned(),
            error: format!("{e:?}"),
        })?;
        check_f32_against_f64(
            gate_tol("causal_conv1d"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "conv golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let mut out = f32_output_bytes(&buffers.outputs[0], "conv y")?;
        out.extend_from_slice(&f16_output_bytes(&buffers.outputs[1]));
        Ok(out)
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        // Causal conv looks back only: row 0 depends on the pinned first
        // row plus the zero history in every mode, so suffix rows (later
        // tokens, larger batches) cannot change it.
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            GATE_C,
            "conv invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 0,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = TypedBuffer::from_f32(&[4, GATE_C], &[0.5; 16]);
        let w = match index {
            // Quantized conv weights fail closed (SI-55).
            0 => TypedBuffer::from_i8(&[GATE_C, GATE_WK], &[1i8; 12]),
            // Kernel width must match the op.
            _ => TypedBuffer::from_f32(&[GATE_C, GATE_WK + 1], &[0.5; 16]),
        };
        Ok(GateBuffers::fresh(
            vec![
                x,
                w,
                TypedBuffer::zeros(&[1, GATE_WK - 1, GATE_C], DType::F16),
            ],
            vec![
                TypedBuffer::zeros(&[4, GATE_C], DType::F32),
                TypedBuffer::zeros(&[1, GATE_WK - 1, GATE_C], DType::F16),
            ],
        ))
    }
}

#[test]
fn gate_causal_conv1d_through_engine() {
    gate_ok(run_gates(&ConvGate), "causal_conv1d");
}

fn scan_op() -> LinearAttnScanOp {
    LinearAttnScanOp {
        kind: LinearAttnKind::GLA,
        chunk: 2,
        out_dtype: DType::F32,
        handle: StateHandle::new(0, StateKind::Recurrent),
    }
}

struct ScanGate;
impl GateCase for ScanGate {
    fn op_name(&self) -> &'static str {
        "linear_attn_scan"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("linear_attn_scan")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let q = uniform_f32(&mut rng, t * 4, -1.0, 1.0);
        let k = uniform_f32(&mut rng, t * 4, -1.0, 1.0);
        let v = uniform_f32(&mut rng, t * 4, -1.0, 1.0);
        let a = uniform_f32(&mut rng, t, 0.8, 1.0);
        let b = uniform_f32(&mut rng, t, 0.0, 0.5);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[t, 1, 4], &q),
                TypedBuffer::from_f32(&[t, 1, 4], &k),
                TypedBuffer::from_f32(&[t, 1, 4], &v),
                TypedBuffer::from_f32(&[t, 1], &a),
                TypedBuffer::from_f32(&[t, 1], &b),
                TypedBuffer::zeros(&[1, 1, 4, 4], DType::F32),
            ],
            vec![
                TypedBuffer::zeros(&[t, 1, 4], DType::F32),
                TypedBuffer::zeros(&[1, 1, 4, 4], DType::F32),
            ],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let q = pinned_prefix_rows("scan_q", t, 4, row, -1.0, 1.0);
        let k = pinned_prefix_rows("scan_k", t, 4, row, -1.0, 1.0);
        let v = pinned_prefix_rows("scan_v", t, 4, row, -1.0, 1.0);
        let a = pinned_prefix_rows("scan_a", t, 1, row, 0.8, 1.0);
        let b = pinned_prefix_rows("scan_b", t, 1, row, 0.0, 0.5);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[t, 1, 4], &q),
                TypedBuffer::from_f32(&[t, 1, 4], &k),
                TypedBuffer::from_f32(&[t, 1, 4], &v),
                TypedBuffer::from_f32(&[t, 1], &a),
                TypedBuffer::from_f32(&[t, 1], &b),
                TypedBuffer::zeros(&[1, 1, 4, 4], DType::F32),
            ],
            vec![
                TypedBuffer::zeros(&[t, 1, 4], DType::F32),
                TypedBuffer::zeros(&[1, 1, 4, 4], DType::F32),
            ],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let t = buffers.inputs[0].shape()[0];
        let seq = SeqLayout::new(&[t as u32]).expect("gate layout builds");
        let (o0, o1) = buffers.outputs.as_mut_slice().split_at_mut(1);
        linear_attn_scan_chunked(
            &scan_op(),
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &buffers.inputs[2].as_view(),
            &buffers.inputs[3].as_view(),
            &buffers.inputs[4].as_view(),
            &buffers.inputs[5].as_view(),
            &seq,
            &mut o0[0].as_view_mut(),
            &mut o1[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        use r9v_t0::linear_attn_scan_f64_reference as scan_oracle;
        let t = buffers.inputs[0].shape()[0];
        let f = |buf: &TypedBuffer| {
            buf.to_f32_vec()
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>()
        };
        let (expected, _) = scan_oracle(
            &f(&buffers.inputs[0]),
            &f(&buffers.inputs[1]),
            &f(&buffers.inputs[2]),
            &f(&buffers.inputs[3]),
            &f(&buffers.inputs[4]),
            t,
            1,
            4,
            4,
            &f(&buffers.inputs[5]),
            1,
            &[t as u32],
        )
        .map_err(|e| HarnessError::UnexpectedRefusal {
            context: "scan oracle".to_owned(),
            error: format!("{e:?}"),
        })?;
        check_f32_against_f64(
            gate_tol("linear_attn_scan"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "scan golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let mut out = f32_output_bytes(&buffers.outputs[0], "scan o")?;
        out.extend_from_slice(&f32_output_bytes(&buffers.outputs[1], "scan state")?);
        Ok(out)
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        // Single-sequence scan from zero state: row 0 depends only on the
        // pinned first row, so suffix rows (later tokens, larger batches)
        // cannot change it. Later rows legitimately depend on history, so
        // the gate compares row 0 in every mode.
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            4,
            "scan invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 0,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let t = 4usize;
        let good = |n: usize| TypedBuffer::from_f32(&[t, 1, 4], &vec![0.5; n * 4]);
        let (a, state) = match index {
            // Gate scales must be [T, H].
            0 => (
                TypedBuffer::from_f32(&[t, 2], &[0.9; 8]),
                TypedBuffer::zeros(&[1, 1, 4, 4], DType::F32),
            ),
            // State must be rank 4 [1, H, D, Dv].
            _ => (
                TypedBuffer::from_f32(&[t, 1], &[0.9; 4]),
                TypedBuffer::zeros(&[1, 4, 4], DType::F32),
            ),
        };
        Ok(GateBuffers::fresh(
            vec![
                good(t),
                good(t),
                good(t),
                a,
                TypedBuffer::from_f32(&[t, 1], &[0.1; 4]),
                state,
            ],
            vec![
                TypedBuffer::zeros(&[t, 1, 4], DType::F32),
                TypedBuffer::zeros(&[1, 1, 4, 4], DType::F32),
            ],
        ))
    }
}

#[test]
fn gate_linear_attn_scan_through_engine() {
    gate_ok(run_gates(&ScanGate), "linear_attn_scan");
}

// ---------------------------------------------------------------------------
// Sampling gates: logits_postprocess, sample, verify (Spec 1 §4.F).
// ---------------------------------------------------------------------------

fn gate_params(temp: f32) -> SamplingParams {
    SamplingParams {
        temperature: temp,
        top_k: 4,
        top_p: 0.9,
        min_p: 0.05,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: vec![],
    }
}

const GATE_V: usize = 8;

/// Keep-mask bits stashed in a u32 buffer (bit j keeps token j).
fn pp_mask_bits(mask: &[bool]) -> Vec<u32> {
    mask.chunks(GATE_V)
        .map(|row| {
            row.iter().enumerate().fold(
                0u32,
                |acc, (j, &keep)| {
                    if keep {
                        acc | (1 << j)
                    } else {
                        acc
                    }
                },
            )
        })
        .collect()
}
fn pp_expand_mask(words: &[u32], s: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(s * GATE_V);
    for &w in words {
        for j in 0..GATE_V {
            out.push(w & (1 << j) != 0);
        }
    }
    out
}

struct PostprocessGate;
impl GateCase for PostprocessGate {
    fn op_name(&self) -> &'static str {
        "logits_postprocess"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("logits_postprocess")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&s| vec![s]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let s = shape[0];
        let mut rng = SeededRng::new(seed);
        let logits = activation_values(&mut rng, s * GATE_V);
        let mask = harness::keep_mask(&mut rng, s * GATE_V, 0.9);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[s, GATE_V], &logits),
                TypedBuffer::from_u32(&[s], &pp_mask_bits(&mask)),
            ],
            vec![TypedBuffer::zeros(&[s, GATE_V], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let s = shape[0];
        let logits = pinned_rows("postprocess", s, GATE_V, row, -2.0, 2.0);
        // The logical row keeps its mask bits; every row keeps at least
        // token 0 so no row is fully masked (a refusal case, not golden).
        let mut prng = pin_rng("postprocess_mask", 0);
        let mut logical = harness::keep_mask(&mut prng, GATE_V, 0.9);
        logical[0] = true;
        let mut frng = filler_rng("postprocess_mask", s);
        let mut mask = harness::keep_mask(&mut frng, s * GATE_V, 0.9);
        mask[row * GATE_V..(row + 1) * GATE_V].copy_from_slice(&logical);
        for r in 0..s {
            mask[r * GATE_V] = true;
        }
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[s, GATE_V], &logits),
                TypedBuffer::from_u32(&[s], &pp_mask_bits(&mask)),
            ],
            vec![TypedBuffer::zeros(&[s, GATE_V], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let s = buffers.inputs[0].shape()[0];
        let params = vec![gate_params(0.7); s];
        let mask = pp_expand_mask(&buffers.inputs[1].to_u32_vec(), s);
        let mut out = buffers.outputs[0].to_f32_vec();
        logits_postprocess(
            &buffers.inputs[0].to_f32_vec(),
            s,
            1,
            GATE_V,
            &params,
            None,
            Some(&mask),
            &mut out,
        )?;
        let fresh = TypedBuffer::from_f32(&[s, GATE_V], &out);
        buffers.outputs[0] = fresh;
        Ok(())
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let s = buffers.inputs[0].shape()[0];
        let params = vec![gate_params(0.7); s];
        let mask = pp_expand_mask(&buffers.inputs[1].to_u32_vec(), s);
        let l64: Vec<f64> = buffers.inputs[0]
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect();
        let expected =
            logits_postprocess_f64_reference(&l64, s, 1, GATE_V, &params, None, Some(&mask));
        check_f32_against_f64(
            gate_tol("logits_postprocess"),
            &buffers.outputs[0].to_f32_vec(),
            &expected,
            "postprocess golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "postprocess determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            GATE_V,
            "postprocess invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let s = 2usize;
        let logits = match index {
            // NaN logits refuse with (seq, query, token) locations.
            0 => {
                let mut l = vec![0.1f32; s * GATE_V];
                l[3] = f32::NAN;
                l
            }
            _ => vec![0.1f32; s * GATE_V],
        };
        let mask_words = match index {
            // A fully masked row refuses (empty distribution).
            0 => vec![0xFF; s],
            _ => vec![0xFF, 0x00],
        };
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[s, GATE_V], &logits),
                TypedBuffer::from_u32(&[s], &mask_words),
            ],
            vec![TypedBuffer::zeros(&[s, GATE_V], DType::F32)],
        ))
    }
}

#[test]
fn gate_logits_postprocess_through_engine() {
    gate_ok(run_gates(&PostprocessGate), "logits_postprocess");
}

fn gate_rng(seq: u64, step: u64) -> RngState {
    RngState::new(0xA110, SeqId::new(seq), StepId::new(step)).expect("gate rng builds")
}

/// One-hot rows peaked at `peaks[i]`: the draw is the peak for every
/// RNG state, so the argmax oracle is exact and RNG-independent. Draw
/// distribution shape (non-degenerate probs) stays with the bespoke
/// L2 test; the gate proves wiring, refusal, and determinism.
fn sample_one_hot(s: usize, peaks: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; s * GATE_V];
    for (i, &p) in peaks.iter().enumerate() {
        out[i * GATE_V + p % GATE_V] = 1.0;
    }
    out
}

fn sample_states_for(buffers: &GateBuffers) -> Vec<RngState> {
    buffers.inputs[1]
        .to_u32_vec()
        .chunks_exact(2)
        .map(|w| gate_rng(w[0] as u64, w[1] as u64))
        .collect()
}

struct SampleGate;
impl GateCase for SampleGate {
    fn op_name(&self) -> &'static str {
        "sample"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("sample")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&s| vec![s]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let s = shape[0];
        let mut rng = SeededRng::new(seed);
        let peaks = ids_in_range(&mut rng, s, GATE_V as u32);
        let words: Vec<u32> = (0..s).flat_map(|i| [5 + i as u32, 9]).collect();
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(
                    &[s, GATE_V],
                    &sample_one_hot(s, &peaks.iter().map(|&p| p as usize).collect::<Vec<_>>()),
                ),
                TypedBuffer::from_u32(&[s, 2], &words),
            ],
            vec![TypedBuffer::zeros(&[s], DType::U32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let s = shape[0];
        // Logical row: peak token 3 with RNG state (7, 3) in every mode.
        let mut frng = filler_rng("sample", s);
        let mut peaks = ids_in_range(&mut frng, s, GATE_V as u32);
        peaks[row] = 3;
        let mut words: Vec<u32> = (0..s).flat_map(|i| [5 + i as u32, 9]).collect();
        words[row * 2] = 7;
        words[row * 2 + 1] = 3;
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(
                    &[s, GATE_V],
                    &sample_one_hot(s, &peaks.iter().map(|&p| p as usize).collect::<Vec<_>>()),
                ),
                TypedBuffer::from_u32(&[s, 2], &words),
            ],
            vec![TypedBuffer::zeros(&[s], DType::U32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let s = buffers.inputs[0].shape()[0];
        let mut states = sample_states_for(buffers);
        let tokens = sample(&buffers.inputs[0].to_f32_vec(), s, GATE_V, &mut states)?;
        let fresh = TypedBuffer::from_u32(&[s], &tokens);
        buffers.outputs[0] = fresh;
        Ok(())
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let s = buffers.inputs[0].shape()[0];
        let probs = buffers.inputs[0].to_f32_vec();
        let mut want = Vec::with_capacity(s * 4);
        for i in 0..s {
            let peak = probs[i * GATE_V..(i + 1) * GATE_V]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j as u32)
                .unwrap_or(0);
            want.extend_from_slice(&peak.to_le_bytes());
        }
        check_bits_equal(
            &u32_output_bytes(&buffers.outputs[0], "sample tokens")?,
            &want,
            "sample golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        u32_output_bytes(&buffers.outputs[0], "sample determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let toks = buffers.outputs[0].to_u32_vec();
        Ok(toks[buffers.logical_row].to_le_bytes().to_vec())
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        // Buffer shapes always match their data (`from_*` asserts); the
        // refusal comes from the op call, never from construction.
        match index {
            // One RNG state for two sequences refuses.
            0 => Ok(GateBuffers::fresh(
                vec![
                    TypedBuffer::from_f32(&[2, GATE_V], &[1.0; 16]),
                    TypedBuffer::from_u32(&[1, 2], &[5, 9]),
                ],
                vec![TypedBuffer::zeros(&[2], DType::U32)],
            )),
            // S must be nonzero.
            _ => Ok(GateBuffers::fresh(
                vec![
                    TypedBuffer::from_f32(&[0, GATE_V], &[]),
                    TypedBuffer::from_u32(&[0, 2], &[]),
                ],
                vec![TypedBuffer::zeros(&[0], DType::U32)],
            )),
        }
    }
}

#[test]
fn gate_sample_through_engine() {
    gate_ok(run_gates(&SampleGate), "sample");
}

fn verify_fixture(s: usize, seed: u64) -> (Vec<u32>, Vec<f32>, Vec<u32>) {
    let mut rng = SeededRng::new(seed);
    let peaks = ids_in_range(&mut rng, s, GATE_V as u32);
    let mut target = Vec::with_capacity(s * 3 * GATE_V);
    for _ in 0..s {
        for _ in 0..3 {
            for _ in 0..GATE_V {
                target.push(0.1 / 7.0);
            }
        }
    }
    for (i, &p) in peaks.iter().enumerate() {
        for k in 0..3 {
            target[(i * 3 + k) * GATE_V + p as usize] = 0.9;
        }
    }
    let mut draft = Vec::with_capacity(s * 2);
    for (i, &p) in peaks.iter().enumerate() {
        if i % 2 == 0 {
            draft.push(p);
            draft.push(p);
        } else {
            draft.push((p + 1) % GATE_V as u32);
            draft.push(p);
        }
    }
    let words: Vec<u32> = (0..s).flat_map(|i| [2 + i as u32, 4]).collect();
    (draft, target, words)
}

struct VerifyGate;
impl GateCase for VerifyGate {
    fn op_name(&self) -> &'static str {
        "verify"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("verify")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&s| vec![s]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    /// Peaked target rows (0.9 at `peak`, 0.1/7 elsewhere) and drafts that
    /// fully match on even rows and mismatch at position 0 on odd rows.
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let s = shape[0];
        let (draft, target, words) = verify_fixture(s, seed);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_u32(&[s * 2], &draft),
                TypedBuffer::from_f32(&[s * 3 * GATE_V], &target),
                TypedBuffer::from_u32(&[s, 2], &words),
            ],
            vec![
                TypedBuffer::zeros(&[s * 3], DType::U32),
                TypedBuffer::zeros(&[s], DType::U32),
            ],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        // Pinned mode always uses the all-matching fixture (peak 3) so the
        // logical row's decision is identical across modes.
        let s = shape[0];
        let target = vec![0.1f32 / 7.0; s * 3 * GATE_V];
        let mut target = target;
        for i in 0..s {
            for k in 0..3 {
                target[(i * 3 + k) * GATE_V + 3] = 0.9;
            }
        }
        let draft = vec![3u32; s * 2];
        let words: Vec<u32> = (0..s).flat_map(|i| [2 + i as u32, 4]).collect();
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_u32(&[s * 2], &draft),
                TypedBuffer::from_f32(&[s * 3 * GATE_V], &target),
                TypedBuffer::from_u32(&[s, 2], &words),
            ],
            vec![
                TypedBuffer::zeros(&[s * 3], DType::U32),
                TypedBuffer::zeros(&[s], DType::U32),
            ],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        let s = buffers.outputs[1].shape()[0];
        let mut states: Vec<RngState> = buffers.inputs[2]
            .to_u32_vec()
            .chunks_exact(2)
            .map(|w| gate_rng(w[0] as u64, w[1] as u64))
            .collect();
        let out = verify(
            &buffers.inputs[0].to_u32_vec(),
            None,
            &buffers.inputs[1].to_f32_vec(),
            s,
            2,
            GATE_V,
            &VerifyMethod::Greedy,
            &mut states,
            None,
        )?;
        buffers.outputs[0] = TypedBuffer::from_u32(&[s * 3], &out.accepted);
        buffers.outputs[1] = TypedBuffer::from_u32(&[s], &out.accept_len);
        Ok(())
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let s = buffers.outputs[1].shape()[0];
        let draft = buffers.inputs[0].to_u32_vec();
        let target = buffers.inputs[1].to_f32_vec();
        // Independent oracle: per-row argmax plus the greedy prefix rule
        // (accept while draft == peak; terminal token is the peak).
        let mut want_acc = Vec::with_capacity(s * 3);
        let mut want_len: Vec<u32> = Vec::with_capacity(s);
        for i in 0..s {
            let row0 = &target[i * 3 * GATE_V..(i + 1) * 3 * GATE_V];
            let peak = row0[..GATE_V]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j as u32)
                .unwrap_or(0);
            let (d0, d1) = (draft[i * 2], draft[i * 2 + 1]);
            if d0 == peak && d1 == peak {
                want_acc.extend_from_slice(&[d0, d1, peak]);
                want_len.push(2);
            } else if d0 == peak {
                want_acc.extend_from_slice(&[d0, peak, 0]);
                want_len.push(1);
            } else {
                want_acc.extend_from_slice(&[peak, 0, 0]);
                want_len.push(0);
            }
        }
        let mut want_acc_bytes = Vec::with_capacity(s * 12);
        for a in &want_acc {
            want_acc_bytes.extend_from_slice(&a.to_le_bytes());
        }
        let mut want_len_bytes = Vec::with_capacity(s * 4);
        for l in &want_len {
            want_len_bytes.extend_from_slice(&l.to_le_bytes());
        }
        check_bits_equal(
            &u32_output_bytes(&buffers.outputs[0], "verify accepted")?,
            &want_acc_bytes,
            "verify accepted",
        )?;
        check_bits_equal(
            &u32_output_bytes(&buffers.outputs[1], "verify accept_len")?,
            &want_len_bytes,
            "verify accept_len",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let mut out = u32_output_bytes(&buffers.outputs[0], "verify accepted")?;
        out.extend_from_slice(&u32_output_bytes(&buffers.outputs[1], "verify accept_len")?);
        Ok(out)
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let row = buffers.logical_row;
        let acc = buffers.outputs[0].to_u32_vec();
        let len = buffers.outputs[1].to_u32_vec();
        let mut out = Vec::new();
        for a in &acc[row * 3..row * 3 + 3] {
            out.extend_from_slice(&a.to_le_bytes());
        }
        out.extend_from_slice(&len[row].to_le_bytes());
        Ok(out)
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let s = 1usize;
        let (draft, target) = match index {
            // Draft length must be S * K.
            0 => (vec![3u32; s * 2 - 1], vec![0.1f32; s * 3 * GATE_V]),
            // Target length must be S * (K + 1) * V.
            _ => (vec![3u32; s * 2], vec![0.1f32; s * 3 * GATE_V - 1]),
        };
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_u32(&[draft.len()], &draft),
                TypedBuffer::from_f32(&[target.len()], &target),
                TypedBuffer::from_u32(&[s, 2], &[2, 4]),
            ],
            vec![
                TypedBuffer::zeros(&[s * 3], DType::U32),
                TypedBuffer::zeros(&[s], DType::U32),
            ],
        ))
    }
}

#[test]
fn gate_verify_through_engine() {
    gate_ok(run_gates(&VerifyGate), "verify");
}

// ---------------------------------------------------------------------------
// Collective gates at ranks = 1 (Spec 1 §4.G, SI-54).
//
// all_reduce / all_gather / reduce_scatter / all_to_all are bit-exact
// identity transfers through the copy core; send and barrier are no-ops;
// recv has no T0 transport and always refuses with a typed error.
// ---------------------------------------------------------------------------

fn gate_group() -> GroupId {
    GroupId::new(0)
}

fn all_reduce_op() -> AllReduceOp {
    AllReduceOp {
        group: gate_group(),
        op: ReduceOp::Sum,
        dtype: DType::F32,
        reduce_in: DType::F32,
    }
}

struct AllReduceGate;
impl GateCase for AllReduceGate {
    fn op_name(&self) -> &'static str {
        "all_reduce"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("all_reduce")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(2)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(2)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("all_reduce", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        all_reduce(
            &all_reduce_op(),
            &buffers.inputs[0].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        // Rank-1 Sum reduces over one rank: bit-exact identity (SI-54).
        check_bits_equal(
            &f32_output_bytes(&buffers.outputs[0], "all_reduce golden")?,
            &f32_output_bytes(&buffers.inputs[0], "all_reduce input")?,
            "all_reduce golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "all_reduce determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "all_reduce invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(2)
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        // reduce_in must stay f32 per spec even though unused at rank 1
        // (bespoke refusal test); the gate's buffer-driven cases are the
        // output-shape and input-dtype mismatches below.
        let (x, y) = match index {
            0 => (
                TypedBuffer::from_f32(&[2, 2], &[0.5; 4]),
                TypedBuffer::zeros(&[2, 3], DType::F32),
            ),
            _ => (
                TypedBuffer::from_f16(&[2, 2], &[0u16; 4]),
                TypedBuffer::zeros(&[2, 2], DType::F32),
            ),
        };
        Ok(GateBuffers::fresh(vec![x], vec![y]))
    }
}

#[test]
fn gate_all_reduce_through_engine() {
    gate_ok(run_gates(&AllReduceGate), "all_reduce");
}

fn all_gather_op() -> AllGatherOp {
    AllGatherOp {
        group: gate_group(),
        axis: 0,
        dtype: DType::F32,
    }
}

struct AllGatherGate;
impl GateCase for AllGatherGate {
    fn op_name(&self) -> &'static str {
        "all_gather"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("all_gather")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(2)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(2)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("all_gather", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        all_gather(
            &all_gather_op(),
            &buffers.inputs[0].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        check_bits_equal(
            &f32_output_bytes(&buffers.outputs[0], "all_gather golden")?,
            &f32_output_bytes(&buffers.inputs[0], "all_gather input")?,
            "all_gather golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "all_gather determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "all_gather invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(2)
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
        let y = match index {
            0 => TypedBuffer::zeros(&[2, 3], DType::F32),
            _ => TypedBuffer::zeros(&[2, 2], DType::F16),
        };
        Ok(GateBuffers::fresh(vec![x], vec![y]))
    }
}

#[test]
fn gate_all_gather_through_engine() {
    gate_ok(run_gates(&AllGatherGate), "all_gather");
}

fn reduce_scatter_op() -> ReduceScatterOp {
    ReduceScatterOp {
        group: gate_group(),
        axis: 0,
        op: ReduceOp::Sum,
        dtype: DType::F32,
        reduce_in: DType::F32,
    }
}

struct ReduceScatterGate;
impl GateCase for ReduceScatterGate {
    fn op_name(&self) -> &'static str {
        "reduce_scatter"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("reduce_scatter")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(2)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(2)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("reduce_scatter", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        reduce_scatter(
            &reduce_scatter_op(),
            &buffers.inputs[0].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        check_bits_equal(
            &f32_output_bytes(&buffers.outputs[0], "reduce_scatter golden")?,
            &f32_output_bytes(&buffers.inputs[0], "reduce_scatter input")?,
            "reduce_scatter golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "reduce_scatter determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let n = buffers.outputs[0].shape()[1];
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            n,
            "reduce_scatter invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(2)
    }
    fn illegal_count(&self) -> usize {
        1
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[2, 2], &[0.5; 4])],
            vec![TypedBuffer::zeros(&[3, 2], DType::F32)],
        ))
    }
}

#[test]
fn gate_reduce_scatter_through_engine() {
    gate_ok(run_gates(&ReduceScatterGate), "reduce_scatter");
}

struct AllToAllGate;
impl GateCase for AllToAllGate {
    fn op_name(&self) -> &'static str {
        "all_to_all"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("all_to_all")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        bucket_edge_counts().iter().map(|&t| vec![t]).collect()
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3], vec![17]]
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * 2);
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[t, 2], &x),
                TypedBuffer::from_u32(&[1], &[t as u32]),
            ],
            vec![TypedBuffer::zeros(&[t, 2], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let t = shape[0];
        let x = pinned_rows("all_to_all", t, 2, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![
                TypedBuffer::from_f32(&[t, 2], &x),
                TypedBuffer::from_u32(&[1], &[t as u32]),
            ],
            vec![TypedBuffer::zeros(&[t, 2], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        all_to_all(
            &AllToAllOp {
                group: gate_group(),
                dtype: DType::F32,
            },
            &buffers.inputs[0].as_view(),
            &buffers.inputs[1].as_view(),
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        // counts[0] covers all rows: identity at rank 1 (SI-54).
        check_bits_equal(
            &f32_output_bytes(&buffers.outputs[0], "all_to_all golden")?,
            &f32_output_bytes(&buffers.inputs[0], "all_to_all input")?,
            "all_to_all golden",
        )
    }
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_output_bytes(&buffers.outputs[0], "all_to_all determinism")
    }
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        f32_row_bytes(
            &buffers.outputs[0],
            buffers.logical_row,
            2,
            "all_to_all invariance",
        )
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![8],
            embedded: vec![6],
            row_alone: 0,
            row: 2,
        }
    }
    fn illegal_count(&self) -> usize {
        1
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::fresh(
            vec![
                TypedBuffer::from_f32(&[4, 2], &[0.5; 8]),
                TypedBuffer::from_u32(&[1], &[3]),
            ],
            vec![TypedBuffer::zeros(&[4, 2], DType::F32)],
        ))
    }
}

#[test]
fn gate_all_to_all_through_engine() {
    gate_ok(run_gates(&AllToAllGate), "all_to_all");
}

struct SendGate;
impl GateCase for SendGate {
    fn op_name(&self) -> &'static str {
        "send"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("send")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(2)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(2)
    }
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let mut rng = SeededRng::new(seed);
        let x = activation_values(&mut rng, t * n);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        let x = pinned_rows("send", t, n, row, -2.0, 2.0);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[t, n], &x)],
            vec![],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        send(
            &SendOp {
                group: gate_group(),
                peer: 0,
                dtype: DType::F32,
            },
            &buffers.inputs[0].as_view(),
        )
    }
    fn verify(&self, _buffers: &GateBuffers) -> Result<(), HarnessError> {
        // Rank-1 send carries no data: acceptance IS the contract. The
        // illegal cases below prove refusal is still enforced.
        Ok(())
    }
    fn output_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        Ok(Vec::new())
    }
    fn logical_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        Ok(Vec::new())
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(2)
    }
    fn illegal_count(&self) -> usize {
        2
    }
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        let x = match index {
            0 => TypedBuffer::from_f16(&[2, 2], &[0u16; 4]),
            _ => TypedBuffer::from_u32(&[2, 2], &[1; 4]),
        };
        Ok(GateBuffers::fresh(vec![x], vec![]))
    }
}

#[test]
fn gate_send_through_engine() {
    gate_ok(run_gates(&SendGate), "send");
}

struct RecvGate;
impl GateCase for RecvGate {
    fn op_name(&self) -> &'static str {
        "recv"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("recv")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        rowwise_shapes(2)
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        rowwise_fuzz(2)
    }
    fn build(&self, shape: &[usize], _seed: u64) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        Ok(GateBuffers::fresh(
            vec![],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
        ))
    }
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        let (t, n) = (shape[0], shape[1]);
        Ok(GateBuffers::pinned(
            vec![],
            vec![TypedBuffer::zeros(&[t, n], DType::F32)],
            row,
        ))
    }
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        recv(
            &RecvOp {
                group: gate_group(),
                peer: 0,
                shape: vec![r9v_ir::Dim::Concrete(2), r9v_ir::Dim::Concrete(2)].into_boxed_slice(),
                dtype: DType::F32,
            },
            &mut buffers.outputs[0].as_view_mut(),
        )
    }
    fn verify(&self, _buffers: &GateBuffers) -> Result<(), HarnessError> {
        // Unreachable: the engine requires refusal before calling verify.
        Ok(())
    }
    fn output_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        Ok(Vec::new())
    }
    fn logical_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        Ok(Vec::new())
    }
    fn batch_rows(&self) -> BatchRows {
        rowwise_batch(2)
    }
    fn always_refuses(&self) -> bool {
        true
    }
    fn illegal_count(&self) -> usize {
        // Documented exception: every input refuses, so there is no
        // distinct illegal input beyond the legal refusal itself.
        0
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Err(HarnessError::MissingRefusal {
            context: "recv has no illegal inputs".to_owned(),
            detail: "always-refusing op".to_owned(),
        })
    }
}

#[test]
fn gate_recv_through_engine() {
    gate_ok(run_gates(&RecvGate), "recv");
}

struct BarrierGate;
impl GateCase for BarrierGate {
    fn op_name(&self) -> &'static str {
        "barrier"
    }
    fn tolerance(&self) -> Tolerance {
        gate_tol("barrier")
    }
    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![1]]
    }
    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![]
    }
    fn build(&self, _shape: &[usize], _seed: u64) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::fresh(vec![], vec![]))
    }
    fn build_pinned(&self, _shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        Ok(GateBuffers::pinned(vec![], vec![], row))
    }
    fn execute(&self, _buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
        barrier(&BarrierOp {
            group: gate_group(),
        })
    }
    fn verify(&self, _buffers: &GateBuffers) -> Result<(), HarnessError> {
        // Single rank has nothing to synchronize: acceptance is the
        // contract (Spec 1 §4.G).
        Ok(())
    }
    fn output_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        Ok(Vec::new())
    }
    fn logical_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        Ok(Vec::new())
    }
    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1],
            padded: vec![1],
            embedded: vec![1],
            row_alone: 0,
            row: 0,
        }
    }
    fn illegal_count(&self) -> usize {
        // Documented exception: barrier takes no operands, so no input
        // can be illegal.
        0
    }
    fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
        Err(HarnessError::MissingRefusal {
            context: "barrier has no illegal inputs".to_owned(),
            detail: "operand-free op".to_owned(),
        })
    }
}

#[test]
fn gate_barrier_through_engine() {
    gate_ok(run_gates(&BarrierGate), "barrier");
}

#[test]
fn gate_single_gate_api_surface() {
    // The documented convenient API: each gate runs standalone as well as
    // through `run_gates` (used by later cards against T1/T2).
    gate_ok(golden(&CopyGate), "golden");
    gate_ok(batch_invariant(&CopyGate), "batch");
    gate_ok(deterministic(&CopyGate), "det");
    gate_ok(shape_fuzz(&CopyGate), "fuzz");
    gate_ok(golden(&RopeGate), "rope golden");
    gate_ok(batch_invariant(&RopeGate), "rope batch");
}

// ---------------------------------------------------------------------------
// Mechanical exhaustiveness: all 32 Op variants through the engine.
// ---------------------------------------------------------------------------

#[test]
fn gate_engine_covers_all_32_op_variants() {
    let gates: Vec<&dyn GateCase> = vec![
        &CopyGate,
        &CastGate,
        &ActivationGate,
        &ActMulGate,
        &SoftcapGate,
        &ResidualGate,
        &NormGate,
        &QuantActGate,
        &SplitGate,
        &ConcatGate,
        &RopeGate,
        &GatherGate,
        &ScatterGate,
        &EmbedGate,
        &NgramGate,
        &MatmulGate,
        &MoeRouteGate,
        &MoeFfnGate,
        &StateWriteKvGate,
        &AttentionGate,
        &ConvGate,
        &ScanGate,
        &PostprocessGate,
        &SampleGate,
        &VerifyGate,
        &AllReduceGate,
        &AllGatherGate,
        &ReduceScatterGate,
        &AllToAllGate,
        &SendGate,
        &RecvGate,
        &BarrierGate,
    ];
    assert_eq!(gates.len(), 32, "one gate per Op variant");
    let mut names: Vec<&str> = gates.iter().map(|g| g.op_name()).collect();
    names.sort_unstable();
    let mut rows = Tolerance::ALL_OP_NAMES.to_vec();
    rows.sort_unstable();
    assert_eq!(names, rows, "gate names match the tolerance rows exactly");
    for gate in &gates {
        let name = gate.op_name();
        assert!(
            !gate.legal_shapes().is_empty(),
            "{name} declares no legal shapes"
        );
        // Every gate except the operand-free barrier covers a true
        // max-bucket edge (Spec 1 §3.5, Spec 4 §10).
        if name != "barrier" {
            assert!(
                gate.legal_shapes().iter().any(|s| s.contains(&MAX_BUCKET)),
                "{name} has no max-bucket edge shape"
            );
        }
        // Every gate except recv (always refuses) and barrier (no
        // operands) declares explicit illegal inputs.
        if name != "recv" && name != "barrier" {
            assert!(
                gate.illegal_count() >= 1,
                "{name} declares no illegal cases"
            );
        }
        // Every case consumes the fail-closed table row; no case may hide a
        // local widening behind a custom tolerance.
        assert_eq!(
            gate.tolerance(),
            harness::tolerance_for(name).expect("closed op has a tolerance row"),
            "{name} gate bypasses the tolerance table"
        );
    }
}

// ---------------------------------------------------------------------------
// Engine self-checks: counts, edges, fail-closed lookups.
// ---------------------------------------------------------------------------

#[test]
fn gate_engine_constants_and_lookups() {
    assert_eq!(CASES_PER_SHAPE, 32, "Spec 4 §10 count");
    assert_eq!(MAX_BUCKET, 4096, "Spec 1 §3.5 max bucket");
    let edges = bucket_edge_counts();
    assert!(edges.contains(&1), "single-token edge");
    assert!(edges.contains(&MAX_BUCKET), "max-bucket edge");
    assert!(
        edges.iter().any(|&t| t != 1 && t != MAX_BUCKET),
        "padding-row edge"
    );
    // Fail-closed tolerance lookup (SI-60).
    assert!(harness::tolerance_for("no_such_op").is_err());
    for name in Tolerance::ALL_OP_NAMES {
        assert!(harness::tolerance_for(name).is_ok());
    }
    // Empty shape lists and unknown seeds fail closed, never panic.
    struct EmptyGate;
    impl GateCase for EmptyGate {
        fn op_name(&self) -> &'static str {
            "copy"
        }
        fn tolerance(&self) -> Tolerance {
            gate_tol("copy")
        }
        fn legal_shapes(&self) -> Vec<Vec<usize>> {
            vec![]
        }
        fn fuzz_legal(&self) -> Vec<Vec<usize>> {
            vec![]
        }
        fn build(&self, _shape: &[usize], _seed: u64) -> Result<GateBuffers, HarnessError> {
            Err(HarnessError::NoLegalShapes {
                op: "empty".to_owned(),
            })
        }
        fn build_pinned(&self, _shape: &[usize], _row: usize) -> Result<GateBuffers, HarnessError> {
            Err(HarnessError::NoLegalShapes {
                op: "empty".to_owned(),
            })
        }
        fn execute(&self, _buffers: &mut GateBuffers) -> Result<(), r9v_t0::T0Error> {
            Ok(())
        }
        fn verify(&self, _buffers: &GateBuffers) -> Result<(), HarnessError> {
            Ok(())
        }
        fn output_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
            Ok(Vec::new())
        }
        fn logical_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
            Ok(Vec::new())
        }
        fn batch_rows(&self) -> BatchRows {
            BatchRows {
                alone: vec![1],
                padded: vec![1],
                embedded: vec![1],
                row_alone: 0,
                row: 0,
            }
        }
        fn illegal_count(&self) -> usize {
            0
        }
        fn build_illegal(&self, _index: usize) -> Result<GateBuffers, HarnessError> {
            Err(HarnessError::NoLegalShapes {
                op: "empty".to_owned(),
            })
        }
    }
    assert!(matches!(
        golden(&EmptyGate),
        Err(HarnessError::NoLegalShapes { .. })
    ));
    assert!(matches!(
        deterministic(&EmptyGate),
        Err(HarnessError::NoLegalShapes { .. })
    ));
    assert!(harness::case_seed("copy", usize::MAX, u64::MAX).is_err());
}

// ---------------------------------------------------------------------------
// Class generators: all five values, byte-identical (Spec 1 §2.3).
// ---------------------------------------------------------------------------

#[test]
fn gate_class_generators_cover_all_five_classes_byte_identical() {
    assert_eq!(CLASS_COUNT, 5);
    assert_eq!(ALL_CLASSES.len(), 5);
    let shape = [3, 4];
    for class in ALL_CLASSES {
        let a = class_tensor(
            &mut SeededRng::new(harness::seed_for("class", 1, MASTER_SEED)),
            class,
            &shape,
        );
        let b = class_tensor(
            &mut SeededRng::new(harness::seed_for("class", 1, MASTER_SEED)),
            class,
            &shape,
        );
        gate_ok(
            check_bits_equal(
                &f32_output_bytes(&a, "class determinism").expect("bytes"),
                &f32_output_bytes(&b, "class determinism").expect("bytes"),
                "class determinism",
            ),
            "class",
        );
        // Dispatch matches the named generator.
        let mut rng_c = SeededRng::new(harness::seed_for("class", 2, MASTER_SEED));
        let mut rng_n = SeededRng::new(harness::seed_for("class", 2, MASTER_SEED));
        let named = match class {
            r9v_ir::Class::Activation => activation_class_tensor(&mut rng_n, &shape),
            r9v_ir::Class::Weight => weight_class_tensor(&mut rng_n, &shape),
            r9v_ir::Class::State => state_class_tensor(&mut rng_n, &shape),
            r9v_ir::Class::Staging => staging_class_tensor(&mut rng_n, &shape),
            r9v_ir::Class::Param => param_class_tensor(&mut rng_n, &shape),
        };
        let dispatched = class_tensor(&mut rng_c, class, &shape);
        gate_ok(
            check_bits_equal(
                &f32_output_bytes(&named, "class dispatch").expect("bytes"),
                &f32_output_bytes(&dispatched, "class dispatch").expect("bytes"),
                "class dispatch",
            ),
            "class",
        );
        // Value contracts per class.
        let vals = dispatched.to_f32_vec();
        match class {
            r9v_ir::Class::Activation => {
                assert!(vals.iter().all(|&v| (-2.0..=2.0).contains(&v)))
            }
            r9v_ir::Class::Weight => assert!(vals.iter().all(|&v| (-1.0..=1.0).contains(&v))),
            r9v_ir::Class::State | r9v_ir::Class::Staging => {
                assert!(vals.iter().all(|&v| v == 0.0))
            }
            r9v_ir::Class::Param => assert!(vals.iter().all(|&v| (-0.5..=0.5).contains(&v))),
        }
    }
}

// ---------------------------------------------------------------------------
// Native wire fixtures: all four natives decode through T0; every other
// scheme fails closed (Spec 2 §3.2, Card A1.10).
// ---------------------------------------------------------------------------

#[test]
fn gate_native_l0_weights_decode_through_t0_matmul() {
    let cases = [
        (SchemeId::I8R, 64usize, Tolerance::i8_weight()),
        (SchemeId::I8B128, 128usize, Tolerance::i8_weight()),
        (SchemeId::I4K, 256usize, Tolerance::i8_weight()),
        (SchemeId::E4M3B128, 128usize, Tolerance::e4m3_cache()),
    ];
    for (scheme, k, tol) in cases {
        let mut rng = SeededRng::new(harness::seed_for("native_l0", scheme.code(), MASTER_SEED));
        let (m, n) = (2usize, 2usize);
        let carrier = native_l0_weight(&mut rng, scheme, n, k).expect("native builds");
        assert_eq!(carrier.values.shape(), &[n, k]);
        assert_eq!(carrier.expected.len(), n * k);
        // Same stream regenerates byte-identical fixtures.
        let mut rng2 = SeededRng::new(harness::seed_for("native_l0", scheme.code(), MASTER_SEED));
        let again = native_l0_weight(&mut rng2, scheme, n, k).expect("native rebuilds");
        gate_ok(
            check_bits_equal(
                again.values.byte_data(),
                carrier.values.byte_data(),
                "native determinism",
            ),
            "native",
        );
        // Decode through T0: PerToken activations against the inline-L0
        // weight, golden vs f64 over the format-oracle dequant.
        let mut xrng = SeededRng::new(harness::seed_for("native_x", scheme.code(), MASTER_SEED));
        let op = MatmulOp {
            out_dtype: DType::F32,
            epilogue: Epilogue::None,
            transpose_w: false,
        };
        let mut y_buf = TypedBuffer::zeros(&[m, n], DType::F32);
        let mut expected = vec![0.0f64; m * n];
        if scheme == SchemeId::E4M3B128 {
            let x_bytes: Vec<u8> = (0..m * k)
                .map(|_| {
                    r9v_t0::dtype::fp8_e4m3_encode(harness::activation_values(&mut xrng, 1)[0])
                })
                .collect();
            let x_buf = TypedBuffer::from_bytes(&[m, k], DType::E4m3, &x_bytes)
                .with_quant(QuantScheme::PerToken);
            let xs = vec![1.0f32; m];
            let xs_buf = TypedBuffer::from_f32(&[m], &xs);
            matmul_with_scales(
                &op,
                &x_buf.as_view(),
                Some(&xs_buf.as_view()),
                &carrier.values.as_view(),
                None,
                None,
                None,
                &mut y_buf.as_view_mut(),
            )
            .expect("e4m3 matmul runs");
            let xdeq: Vec<f64> = x_bytes
                .iter()
                .map(|&b| r9v_t0::dtype::fp8_e4m3_decode(b) as f64)
                .collect();
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0;
                    for kk in 0..k {
                        acc += xdeq[i * k + kk] * carrier.expected[j * k + kk] as f64;
                    }
                    expected[i * n + j] = acc;
                }
            }
        } else {
            let xq = symmetric_i8(&mut xrng, m * k);
            let xs = positive_scales(&mut xrng, m);
            let x_buf = TypedBuffer::from_i8(&[m, k], &xq).with_quant(QuantScheme::PerToken);
            let xs_buf = TypedBuffer::from_f32(&[m], &xs);
            matmul_with_scales(
                &op,
                &x_buf.as_view(),
                Some(&xs_buf.as_view()),
                &carrier.values.as_view(),
                None,
                None,
                None,
                &mut y_buf.as_view_mut(),
            )
            .expect("i-family matmul runs");
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0;
                    for kk in 0..k {
                        acc += xq[i * k + kk] as f64
                            * xs[i] as f64
                            * carrier.expected[j * k + kk] as f64;
                    }
                    expected[i * n + j] = acc;
                }
            }
        }
        gate_ok(
            check_f32_against_f64(tol, &y_buf.to_f32_vec(), &expected, "native golden"),
            "native",
        );
    }
}

#[test]
fn gate_scheme_fixtures_fail_closed_outside_natives() {
    // Every non-native SchemeId fails closed with a typed contract.
    for scheme in SchemeId::ALL {
        if is_natively_decoded(scheme) {
            continue;
        }
        let mut rng = SeededRng::new(harness::seed_for("repack", scheme.code(), MASTER_SEED));
        let err = native_l0_weight(&mut rng, scheme, 2, 256).expect_err("must refuse");
        assert!(
            matches!(err, HarnessError::UnsupportedScheme { .. }),
            "typed refusal for {}, got {err:?}",
            scheme.name()
        );
        // Geometry carriers still cover every mapping deterministically.
        let mut grng = SeededRng::new(harness::seed_for("repack", scheme.code(), MASTER_SEED));
        let carrier = scheme_weight_carrier(&mut grng, scheme, 2, 64);
        assert!(!carrier.native, "repack carrier must not claim native");
        let mut hrng = SeededRng::new(harness::seed_for("repack", scheme.code(), MASTER_SEED));
        let again = scheme_weight_carrier(&mut hrng, scheme, 2, 64);
        gate_ok(
            check_bits_equal(
                again.values.byte_data(),
                carrier.values.byte_data(),
                "carrier determinism",
            ),
            "carrier",
        );
    }
    // Block-multiple violations fail closed, never panic.
    let mut rng = SeededRng::new(harness::seed_for("repack", 0, MASTER_SEED));
    assert!(native_l0_weight(&mut rng, SchemeId::I4K, 2, 128).is_err());
    assert!(native_l0_weight(&mut rng, SchemeId::I8B128, 2, 64).is_err());
    // Ggml mapping: unquantized halves yield no carrier; Q4_K is native;
    // every quantized type has a deterministic geometry carrier.
    assert!(carrier_for_ggml(&mut rng, GgmlType::F16, 2, 64).is_none());
    assert!(carrier_for_ggml(&mut rng, GgmlType::BF16, 2, 64).is_none());
    for ggml in GgmlType::ALL {
        if !ggml.is_quantized() {
            continue;
        }
        let carrier = carrier_for_ggml(&mut rng, ggml, 2, 64).expect("quantized maps");
        assert_eq!(
            carrier.native,
            is_natively_decoded(ggml.scheme().expect("maps"))
        );
    }
    // A repack-scheme carrier through T0 refuses typed (fail-closed
    // decode): I8B32F-tagged bytes are not a wire record.
    let mut rrng = SeededRng::new(harness::seed_for("repack_t0", 0, MASTER_SEED));
    let repack = scheme_weight_carrier(&mut rrng, SchemeId::I8B32F, 2, 16);
    let x = TypedBuffer::from_f32(&[2, 16], &[0.5; 32]);
    let mut y = TypedBuffer::zeros(&[2, 2], DType::F32);
    let op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    assert!(
        matmul(
            &op,
            &x.as_view(),
            &repack.values.as_view(),
            None,
            None,
            &mut y.as_view_mut()
        )
        .is_err(),
        "repack carrier must fail closed at T0 decode"
    );
}
