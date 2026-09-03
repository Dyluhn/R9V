// SPDX-License-Identifier: Apache-2.0
//! Generic T0 test harness (Card A1.10; Spec 4 §10, Spec 1 §6.1/ App. B).
//!
//! One cohesive gate engine over every T0 op: each op family implements
//! [`GateCase`] (fresh seeded inputs, the implementation under test, an
//! independent f64/source oracle, stable logical-row extraction, and
//! explicit illegal cases) and [`run_gates`] drives the four Spec 4 §10
//! gates — [`golden`] vs the oracle on exactly [`CASES_PER_SHAPE`] seeded
//! inputs per legal shape (including single-token, padding-row, and
//! [`MAX_BUCKET`] edges), [`batch_invariant`] (alone / padded / embedded
//! over the same pinned logical row), [`deterministic`] (twice, from fresh
//! state), and [`shape_fuzz`] (legal shapes run, illegal shapes refuse with
//! typed errors). The four single-gate functions are also public so later
//! cards (A3.3–A3.7) can reuse one gate at a time against T1/T2.
//!
//! Seeded fixture generators cover every [`r9v_ir::Class`] tensor kind and
//! every [`r9v_format::SchemeId`] / [`r9v_format::GgmlType`] mapping; the
//! four native schemes additionally have wire-valid inline-L0 fixtures via
//! [`native_l0_weight`] that decode through T0. Lower-level comparison and
//! row-extraction helpers remain available for specialized gate cases.
//!
//! ```rust,no_run
//! use r9v_t0::harness::{GateCase, HarnessError, run_gates};
//!
//! fn check_every_gate<C: GateCase>(case: &C) -> Result<(), HarnessError> {
//!     run_gates(case)
//! }
//! ```
//!
//! The harness itself is pure test infrastructure: no I/O, no globals, no
//! clocks, no `HashMap` iteration. All fallible paths return typed errors;
//! there are no panics, unwraps, expects, or debug asserts anywhere below.

use r9v_common::hash::xxh3_64;
use r9v_common::rng::SeededRng;
use r9v_ir::{Class, DType, QuantScheme};

use crate::buffer::TypedBuffer;
use crate::error::T0Error;
use crate::tolerance::Tolerance;

/// Master seed for every fixture in this harness (Spec 4 §10, Card A1.10).
///
/// One fixed location; per-case independence comes from [`seed_for`].
// DECISION(A1.10): single `u64` master seed plus xxh3 domain separation
// (`"a1.10" | op | case | master`); rejected per-file ad-hoc seeds because
// correlated streams across op families would hide shared-mode bugs.
pub const MASTER_SEED: u64 = 0x4131305F54303075;

const DOMAIN_TAG: &[u8] = b"a1.10";

/// Typed harness failure (Spec 4 §2, CONVENTIONS.md §1.1).
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// A golden comparison exceeded its tolerance floor.
    #[error("golden mismatch in {context}: index {index}: actual={actual}, expected={expected}, tol(abs={abs}, rel={rel})")]
    GoldenMismatch {
        /// Which gate/case failed.
        context: String,
        /// Flat element index.
        index: usize,
        /// Observed value.
        actual: f64,
        /// Oracle value.
        expected: f64,
        /// Absolute floor used.
        abs: f64,
        /// Relative floor used.
        rel: f64,
    },
    /// Two byte strings that must be bit-identical differ.
    #[error(
        "bit mismatch in {context}: index {index}: actual={actual:#x}, expected={expected:#x}"
    )]
    BitMismatch {
        /// Which gate/case failed.
        context: String,
        /// Flat byte index.
        index: usize,
        /// Observed byte.
        actual: u8,
        /// Expected byte.
        expected: u8,
    },
    /// Slice lengths disagree where the contract requires agreement.
    #[error("length mismatch in {context}: actual={actual}, expected={expected}")]
    LengthMismatch {
        /// Which gate/case failed.
        context: String,
        /// Observed length.
        actual: usize,
        /// Required length.
        expected: usize,
    },
    /// Checked extent arithmetic overflowed.
    #[error("arithmetic overflow in {context}: {detail}")]
    ArithmeticOverflow {
        /// Which gate/case failed.
        context: String,
        /// What overflowed.
        detail: String,
    },
    /// The implementation under test refused a legal case or accepted an illegal one.
    #[error("fuzz verdict failure in {context}: {detail}")]
    FuzzVerdict {
        /// Which gate/case failed.
        context: String,
        /// What went wrong.
        detail: String,
    },
    /// Wraps the implementation's typed error for unexpected failures.
    #[error(transparent)]
    T0(#[from] T0Error),
    /// Wraps a format-layer typed error (wire decode, record packing).
    #[error(transparent)]
    Format(#[from] r9v_format::FormatError),
    /// Wraps an IR-layer typed error (batch meta construction in gates).
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),
    /// An op name with no [`Tolerance::for_op`] row (fail-closed lookup).
    #[error("unknown op `{op}`: no tolerance row")]
    UnknownOp {
        /// The unrecognized op name.
        op: String,
    },
    /// A wire-valid carrier was requested for a scheme that has none.
    #[error("scheme `{scheme}` has no wire-valid native L0 fixture: {detail}")]
    UnsupportedScheme {
        /// The scheme display name.
        scheme: String,
        /// Why no fixture exists.
        detail: String,
    },
    /// Fixture dimensions violate the scheme's block geometry.
    #[error("unsupported shape in {context}: {detail}")]
    UnsupportedShape {
        /// Which gate/case failed.
        context: String,
        /// What was wrong.
        detail: String,
    },
    /// The implementation refused a legal gate case.
    #[error("unexpected refusal in {context}: {error}")]
    UnexpectedRefusal {
        /// Which gate/case failed.
        context: String,
        /// The implementation's typed error.
        error: String,
    },
    /// An illegal case (or an always-refusing op) ran successfully.
    #[error("missing refusal in {context}: {detail}")]
    MissingRefusal {
        /// Which gate/case failed.
        context: String,
        /// What ran that must refuse.
        detail: String,
    },
    /// A gate case declares no legal shapes.
    #[error("op `{op}` declares no legal shapes")]
    NoLegalShapes {
        /// The op name.
        op: String,
    },
    /// A seeded value could not be projected onto its target grid.
    #[error("encode failure in {context}: {detail}")]
    EncodeFailure {
        /// Which gate/case failed.
        context: String,
        /// What could not be encoded.
        detail: String,
    },
}

/// Derives an independent per-case seed (Card A1.10; SI-59).
///
/// `seed = xxh3("a1.10" | op_name | case_idx LE | master LE)`, so cases are
/// reproducible across runs and independent across ops.
pub fn seed_for(op_name: &str, case_idx: u64, master: u64) -> u64 {
    let mut bytes = Vec::with_capacity(
        DOMAIN_TAG
            .len()
            .saturating_add(op_name.len())
            .saturating_add(17),
    );
    bytes.extend_from_slice(DOMAIN_TAG);
    bytes.push(0x7C);
    bytes.extend_from_slice(op_name.as_bytes());
    bytes.push(0x7C);
    bytes.extend_from_slice(&case_idx.to_le_bytes());
    bytes.extend_from_slice(&master.to_le_bytes());
    xxh3_64(&bytes)
}

/// Builds the seeded RNG for one harness case (Spec 1 §6.1, Card A1.10).
pub fn rng_for(op_name: &str, case_idx: u64, master: u64) -> SeededRng {
    SeededRng::new(seed_for(op_name, case_idx, master))
}

/// Compares `actual` against an independent oracle within `tol` (Spec 4 §10
/// gate 1, Spec 1 §6.1). NaN pairs pass only when both sides are NaN.
pub fn check_within(
    tol: Tolerance,
    actual: &[f64],
    expected: &[f64],
    context: &str,
) -> Result<(), HarnessError> {
    if actual.len() != expected.len() {
        return Err(HarnessError::LengthMismatch {
            context: context.to_owned(),
            actual: actual.len(),
            expected: expected.len(),
        });
    }
    for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() && e.is_nan() {
            continue;
        }
        let diff = (a - e).abs();
        let limit = tol.abs + tol.rel * e.abs();
        if diff > limit {
            return Err(HarnessError::GoldenMismatch {
                context: context.to_owned(),
                index,
                actual: a,
                expected: e,
                abs: tol.abs,
                rel: tol.rel,
            });
        }
    }
    Ok(())
}

/// Compares f32 implementation output against an f64 oracle (Spec 4 §10).
pub fn check_f32_against_f64(
    tol: Tolerance,
    actual: &[f32],
    expected: &[f64],
    context: &str,
) -> Result<(), HarnessError> {
    if actual.len() != expected.len() {
        return Err(HarnessError::LengthMismatch {
            context: context.to_owned(),
            actual: actual.len(),
            expected: expected.len(),
        });
    }
    for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a64 = a as f64;
        if a64.is_nan() && e.is_nan() {
            continue;
        }
        let diff = (a64 - e).abs();
        let limit = tol.abs + tol.rel * e.abs();
        if diff > limit {
            return Err(HarnessError::GoldenMismatch {
                context: context.to_owned(),
                index,
                actual: a64,
                expected: e,
                abs: tol.abs,
                rel: tol.rel,
            });
        }
    }
    Ok(())
}

/// Requires bit-identical bytes (Spec 4 §10 gates 2–3, L0 in Spec 1 App. B).
pub fn check_bits_equal(a: &[u8], b: &[u8], context: &str) -> Result<(), HarnessError> {
    if a.len() != b.len() {
        return Err(HarnessError::LengthMismatch {
            context: context.to_owned(),
            actual: a.len(),
            expected: b.len(),
        });
    }
    for (index, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            return Err(HarnessError::BitMismatch {
                context: context.to_owned(),
                index,
                actual: x,
                expected: y,
            });
        }
    }
    Ok(())
}

/// Extracts one logical row as bytes from a row-major f32 matrix (Spec 1
/// §6.1 batch invariance). Used to compare the same logical row/sequence
/// across the alone / padded / embedded runs with stable ids and state.
pub fn logical_row_bytes(
    full: &[f32],
    row: usize,
    cols: usize,
    context: &str,
) -> Result<Vec<u8>, HarnessError> {
    let total = full.len();
    if cols == 0 {
        return Err(HarnessError::ArithmeticOverflow {
            context: context.to_owned(),
            detail: "column count is zero".to_owned(),
        });
    }
    if !total.is_multiple_of(cols) {
        return Err(HarnessError::LengthMismatch {
            context: context.to_owned(),
            actual: total,
            expected: total.saturating_sub(total % cols),
        });
    }
    let rows = total / cols;
    if row >= rows {
        return Err(HarnessError::LengthMismatch {
            context: context.to_owned(),
            actual: row,
            expected: rows,
        });
    }
    let start = match row.checked_mul(cols) {
        Some(v) => v,
        None => {
            return Err(HarnessError::ArithmeticOverflow {
                context: context.to_owned(),
                detail: "row * cols overflows usize".to_owned(),
            });
        }
    };
    let end = match start.checked_add(cols) {
        Some(v) => v,
        None => {
            return Err(HarnessError::ArithmeticOverflow {
                context: context.to_owned(),
                detail: "row end overflows usize".to_owned(),
            });
        }
    };
    if end > total {
        return Err(HarnessError::LengthMismatch {
            context: context.to_owned(),
            actual: end,
            expected: total,
        });
    }
    let mut out = Vec::with_capacity(cols.saturating_mul(4));
    for &v in &full[start..end] {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    Ok(out)
}

/// Checked `a * b * c` extent (no panics on fuzz shapes, Card A1.10).
pub fn checked_extent3(a: usize, b: usize, c: usize, context: &str) -> Result<usize, HarnessError> {
    a.checked_mul(b)
        .and_then(|v| v.checked_mul(c))
        .ok_or_else(|| HarnessError::ArithmeticOverflow {
            context: context.to_owned(),
            detail: "a * b * c overflows usize".to_owned(),
        })
}

/// Advances a `u64` draw/case counter with a typed error instead of wrap.
pub fn checked_add_u64(a: u64, b: u64, context: &str) -> Result<u64, HarnessError> {
    a.checked_add(b)
        .ok_or_else(|| HarnessError::ArithmeticOverflow {
            context: context.to_owned(),
            detail: "u64 counter overflows".to_owned(),
        })
}

// ---------------------------------------------------------------------------
// Seeded fixture generators (Spec 4 §10, Card A1.10).
//
// Every generator takes `&mut SeededRng` plus a shape and returns owned
// buffers. Same `(op_name, case_idx, master)` seed => byte-identical output;
// different op names => independent streams. Value ranges mirror the bounds
// the T0 implementations assume (activations in [-2, 2], i8 in
// [-127, 127] so -128 never appears per Spec 1 §6.2).
// ---------------------------------------------------------------------------

/// Uniform f32 values in `[lo, hi]` from the low 24 bits of the RNG.
pub fn uniform_f32(rng: &mut SeededRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    let span = hi - lo;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let u = ((rng.next_u64() >> 40) as f32) / 16777216.0;
        out.push(lo + span * u);
    }
    out
}

/// Activation-like f32 values in `[-2, 2]` (Spec 1 §6.1 golden inputs).
pub fn activation_values(rng: &mut SeededRng, len: usize) -> Vec<f32> {
    uniform_f32(rng, len, -2.0, 2.0)
}

/// Weight-like f32 values in `[-1, 1]`.
pub fn weight_values(rng: &mut SeededRng, len: usize) -> Vec<f32> {
    uniform_f32(rng, len, -1.0, 1.0)
}

/// Symmetric i8 values in `[-127, 127]`; `-128` never appears (Spec 1 §6.2).
pub fn symmetric_i8(rng: &mut SeededRng, len: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let v = (rng.next_u64() % 255) as i64 - 127;
        out.push(v as i8);
    }
    out
}

/// Token/row ids in `0..bound` (bound of 0 yields zeros; never divides).
pub fn ids_in_range(rng: &mut SeededRng, len: usize, bound: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(if bound == 0 {
            0
        } else {
            (rng.next_u64() % bound as u64) as u32
        });
    }
    out
}

/// Positive f32 scales around 1.0 (never zero, finite by construction).
pub fn positive_scales(rng: &mut SeededRng, len: usize) -> Vec<f32> {
    uniform_f32(rng, len, 0.25, 2.0)
}

/// Grammar/keep mask with the given keep probability plus a forced-false row
/// variant available to callers (refusal path, Spec 1 §4.F).
pub fn keep_mask(rng: &mut SeededRng, len: usize, keep_prob: f32) -> Vec<bool> {
    let threshold = (keep_prob.clamp(0.0, 1.0) * 16777216.0) as u64;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push((rng.next_u64() >> 40) < threshold);
    }
    out
}

/// f32 activation tensor of `shape` (Class::Activation, Spec 1 §2.3).
pub fn activation_tensor(rng: &mut SeededRng, shape: &[usize]) -> TypedBuffer {
    let len = shape
        .iter()
        .copied()
        .fold(1usize, |a, b| a.saturating_mul(b));
    TypedBuffer::from_f32(shape, &activation_values(rng, len))
}

/// f32 weight tensor of `shape` (Class::Weight, Spec 1 §2.3).
pub fn weight_tensor(rng: &mut SeededRng, shape: &[usize]) -> TypedBuffer {
    let len = shape
        .iter()
        .copied()
        .fold(1usize, |a, b| a.saturating_mul(b));
    TypedBuffer::from_f32(shape, &weight_values(rng, len))
}

/// Activation tensor requantized to `dtype` through the f32 values above, so
/// golden-vs-f64 pairing stays exact (Card A1.10).
pub fn activation_tensor_as(rng: &mut SeededRng, shape: &[usize], dtype: DType) -> TypedBuffer {
    use crate::dtype::{f32_to_bf16, f32_to_f16, fp8_e4m3_encode, fp8_e5m2_encode};
    let len = shape
        .iter()
        .copied()
        .fold(1usize, |a, b| a.saturating_mul(b));
    let vals = activation_values(rng, len);
    match dtype {
        DType::F32 => TypedBuffer::from_f32(shape, &vals),
        DType::F16 => {
            let bits: Vec<u16> = vals.iter().map(|&v| f32_to_f16(v)).collect();
            TypedBuffer::from_f16(shape, &bits)
        }
        DType::Bf16 => {
            let bits: Vec<u16> = vals.iter().map(|&v| f32_to_bf16(v)).collect();
            TypedBuffer::from_bf16(shape, &bits)
        }
        DType::E4m3 => {
            let bytes: Vec<u8> = vals.iter().map(|&v| fp8_e4m3_encode(v)).collect();
            TypedBuffer::from_e4m3_bytes(shape, &bytes)
        }
        DType::E5m2 => {
            let bytes: Vec<u8> = vals.iter().map(|&v| fp8_e5m2_encode(v)).collect();
            TypedBuffer::from_bytes(shape, DType::E5m2, &bytes)
        }
        DType::I8 => TypedBuffer::from_i8(shape, &symmetric_i8(rng, len)),
        DType::U32 => TypedBuffer::from_u32(shape, &ids_in_range(rng, len, 1024)),
        DType::I32 => {
            let vals: Vec<i32> = ids_in_range(rng, len, 1024)
                .iter()
                .map(|&v| v as i32)
                .collect();
            TypedBuffer::from_i32(shape, &vals)
        }
        DType::I4 | DType::Bool => TypedBuffer::zeros(shape, dtype),
    }
}

/// Token-id tensor `[t]` with ids in `0..vocab` (Spec 1 §4.A).
pub fn token_ids_tensor(rng: &mut SeededRng, t: usize, vocab: u32) -> TypedBuffer {
    let ids = ids_in_range(rng, t, vocab.max(1));
    TypedBuffer::from_u32(&[t], &ids)
}

/// Row-index tensor `[m]` with entries in `0..rows` (Spec 1 §4.A).
pub fn row_ids_tensor(rng: &mut SeededRng, m: usize, rows: u32) -> TypedBuffer {
    let ids = ids_in_range(rng, m, rows.max(1));
    TypedBuffer::from_u32(&[m], &ids)
}

/// Scalar positions `[t]` (never token ids; SI-30).
pub fn positions_tensor(rng: &mut SeededRng, t: usize, max_pos: u32) -> TypedBuffer {
    let ids = ids_in_range(rng, t, max_pos.max(1));
    TypedBuffer::from_u32(&[t], &ids)
}

/// Per-token scales `[t]` f32 (QuantScheme::PerToken carrier, Spec 1 §2.2).
pub fn per_token_scales(rng: &mut SeededRng, t: usize) -> TypedBuffer {
    let s = positive_scales(rng, t.max(1));
    TypedBuffer::from_f32(&[t.max(1)], &s)
}

/// Per-block scales `[t, n/32]` f32 (QuantScheme::PerBlock32 carrier).
pub fn per_block_scales(rng: &mut SeededRng, t: usize, n: usize) -> TypedBuffer {
    let blocks = n.div_ceil(32).max(1);
    let s = positive_scales(rng, t.max(1).saturating_mul(blocks));
    TypedBuffer::from_f32(&[t.max(1), blocks], &s)
}

/// Read back f32 values as f64 for oracle pairing (independent of T0).
pub fn buffer_to_f64(buf: &TypedBuffer) -> Vec<f64> {
    (0..buf.num_elements())
        .map(|i| buf.read_f32(i) as f64)
        .collect()
}

/// Decodes a bytes-backed F16 buffer to f64 (independent of T0).
///
/// `TypedBuffer::from_bytes` keeps raw bytes only, so `read_f32` cannot see
/// them; this decodes the LE pairs directly for oracle pairing.
pub fn bytes_f16_to_f64(buf: &TypedBuffer) -> Vec<f64> {
    use crate::dtype::f16_to_f32;
    let raw = buf.byte_data();
    let n = buf.num_elements();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = raw.get(i.saturating_mul(2)).copied().unwrap_or(0);
        let hi = raw
            .get(i.saturating_mul(2).saturating_add(1))
            .copied()
            .unwrap_or(0);
        out.push(f16_to_f32(u16::from_le_bytes([lo, hi])) as f64);
    }
    out
}

/// F16 tensor with raw-bytes backing (Spec 1 §2.3, Card A1.10).
///
/// T0 `matmul` weights and `embed_gather` tables require
/// `TensorData::Bytes` backing; slice-backed `from_f16` buffers fail closed
/// there with `BackingRepresentationMismatch`. Values match
/// [`activation_tensor_as`] bit-for-bit so oracle pairing is unchanged.
pub fn f16_bytes_tensor(rng: &mut SeededRng, shape: &[usize]) -> TypedBuffer {
    use crate::dtype::f32_to_f16;
    let len = shape
        .iter()
        .copied()
        .fold(1usize, |a, b| a.saturating_mul(b));
    let vals = activation_values(rng, len);
    let mut bytes = Vec::with_capacity(len.saturating_mul(2));
    for &v in &vals {
        bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    TypedBuffer::from_bytes(shape, DType::F16, &bytes)
}

/// Quant scheme tag carried by a generated weight fixture.
pub fn quant_of(buf: &TypedBuffer) -> QuantScheme {
    buf.quant()
}

// ---------------------------------------------------------------------------
// Scheme carriers: one deterministic generator per `SchemeId` (Spec 2 §3)
// plus the `GgmlType` mapping (Spec 2 §7 step 1, card A2.3).
// ---------------------------------------------------------------------------

/// A deterministic weight fixture for one [`r9v_format::SchemeId`].
pub struct SchemeCarrier {
    /// Raw value bytes tagged `[rows, cols]` with `QuantScheme::Scheme(id)`.
    pub values: TypedBuffer,
    /// One positive f32 scale per `(row, block)`; block structure per scheme.
    pub scales: TypedBuffer,
    /// True for the 4 native schemes T0 decodes (Spec 2 §3.2); repack/IQ
    /// carriers are byte fixtures that fail closed at decode (Card A1.10).
    pub native: bool,
}

/// K-block width governing the scale geometry of `scheme` (Spec 2 §3).
/// Returns 0 for row-scoped `I8R` (one scale per row).
pub const fn scheme_block(scheme: r9v_format::SchemeId) -> usize {
    use r9v_format::SchemeId::*;
    match scheme {
        I8R => 0,
        I8B128 | E4M3B128 => 128,
        I4K | I5K | I6K | I3K | I2K => 256,
        I8B32F | I4B32F | I4B32FM | I5B32F | I5B32FM => 32,
        I4Nl | I4Xs | Iq3Xxs | Iq3S | Iq2Xxs | Iq2Xs | Iq2S | Iq1S | Iq1M => 32,
    }
}

/// True exactly for the native schemes with T0 decode paths (Spec 2 §3.2).
pub const fn is_natively_decoded(scheme: r9v_format::SchemeId) -> bool {
    use r9v_format::SchemeId::*;
    matches!(scheme, I8R | I8B128 | I4K | E4M3B128)
}

/// Deterministic carrier bytes + scales for `scheme` (Card A1.10).
///
/// Same `(rng state)` => byte-identical buffers. Value bytes come straight
/// from the seeded stream; scales are positive f32 around 1.0 with the
/// scheme's block geometry ([`scheme_block`]). Only the four
/// [`is_natively_decoded`] schemes carry wire-valid values — and then only
/// through [`native_l0_weight`], not through this function. Repack/IQ
/// carriers from this function are geometry fixtures (right shapes, valid
/// scale counts, deterministic fill): they are NOT wire records and must
/// fail closed at decode. See [`native_l0_weight`] for the fail-closed
/// wire-valid contract.
pub fn scheme_weight_carrier(
    rng: &mut SeededRng,
    scheme: r9v_format::SchemeId,
    rows: usize,
    cols: usize,
) -> SchemeCarrier {
    use r9v_format::SchemeId::*;
    let len = rows.saturating_mul(cols);
    let mut bytes = vec![0u8; len.max(1)];
    if len > 0 {
        rng.fill_bytes(&mut bytes);
    }
    let dtype = match scheme {
        E4M3B128 => DType::E4m3,
        _ => DType::I8,
    };
    let shape_rows = rows.max(1);
    let shape_cols = cols.max(1);
    let values = TypedBuffer::from_bytes(&[shape_rows, shape_cols], dtype, &bytes)
        .with_quant(QuantScheme::Scheme(scheme.to_ir()));
    let block = scheme_block(scheme);
    let scales_per_row = if block == 0 {
        1
    } else {
        shape_cols.div_ceil(block).max(1)
    };
    let scales = positive_scales(rng, shape_rows.saturating_mul(scales_per_row));
    let scales_buf = TypedBuffer::from_f32(&[shape_rows, scales_per_row], &scales);
    SchemeCarrier {
        values,
        scales: scales_buf,
        native: is_natively_decoded(scheme),
    }
}

/// Carrier for a `GgmlType` via its §7 step-1 scheme mapping (Card A2.3).
/// `F16`/`BF16` map to `None` (unquantized; SI-26) and yield no carrier.
pub fn carrier_for_ggml(
    rng: &mut SeededRng,
    ggml: r9v_format::GgmlType,
    rows: usize,
    cols: usize,
) -> Option<SchemeCarrier> {
    ggml.scheme()
        .map(|s| scheme_weight_carrier(rng, s, rows, cols))
}

/// Exact `I8_R` L0 rows: `cols` i8 values plus one trailing f16 scale per
/// row (Spec 2 §3.2). This is the geometry the landed T0 decoders consume.
pub fn i8r_l0_weight(rng: &mut SeededRng, rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    use crate::dtype::f32_to_f16;
    let mut bytes = Vec::with_capacity(rows.saturating_mul(cols.saturating_add(2)));
    let mut scales = Vec::with_capacity(rows);
    for _ in 0..rows {
        let vals = symmetric_i8(rng, cols);
        for &v in &vals {
            bytes.push(v as u8);
        }
        let s = positive_scales(rng, 1)[0];
        bytes.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        scales.push(s);
    }
    (bytes, scales)
}

// ---------------------------------------------------------------------------
// Generic gate engine (Spec 4 §10, Card A1.10).
//
// Every op family implements [`GateCase`]; [`run_gates`] runs the four
// gates. The engine owns all loop structure (exactly [`CASES_PER_SHAPE`]
// seeds per legal shape, the three batch modes, rebuild-twice determinism,
// legal/illegal fuzz dispatch) so a passing run proves the shape/seed
// coverage mechanically rather than by per-test convention. Cases own the
// op-specific parts: fresh seeded inputs and state, the implementation
// call, the independent oracle comparison, logical-row extraction, and the
// explicit illegal inputs.
// ---------------------------------------------------------------------------

/// Seeded golden inputs per legal shape (Spec 4 §10 gate 1).
///
/// One fixed count for every op and every shape; per-case independence
/// comes from [`seed_for`] via [`case_seed`].
// DECISION(A1.10): single shared count rather than per-op counts;
// rejected per-op tuning because §10 states one number and uneven counts
// would let a weak op hide behind fewer draws. Spec 4 §10.
pub const CASES_PER_SHAPE: u64 = 32;

/// Maximum shape-bucket value (Spec 1 §3.5).
///
/// Legal gate shapes must include this edge with tiny non-bucket
/// dimensions so CI stays sane (a `[4096, 1]` golden is 4K elements, not
/// a full modelDim matrix).
pub const MAX_BUCKET: usize = 4096;

/// Token-count edges every token-batched gate covers (Spec 4 §10 "edge
/// shapes", Spec 1 §3.5 buckets): single token, two padding rows (bucket
/// 4 + 1 and bucket 32 + 1, both land in a larger bucket), and the maximum
/// bucket.
pub fn bucket_edge_counts() -> [usize; 4] {
    [1, 5, 33, MAX_BUCKET]
}

/// Legal-fuzz draws per gate beyond the golden shapes (Spec 4 §10).
pub const LEGAL_FUZZ_DRAWS: u64 = 8;

/// Derives the seed for `(shape_idx, case_idx)` under `op_name` (SI-59).
///
/// Returns [`HarnessError::ArithmeticOverflow`] instead of wrapping.
pub fn case_seed(op_name: &str, shape_idx: usize, case_idx: u64) -> Result<u64, HarnessError> {
    let context = op_name.to_owned();
    let base = (shape_idx as u64)
        .checked_mul(CASES_PER_SHAPE)
        .and_then(|v| v.checked_add(case_idx))
        .ok_or_else(|| HarnessError::ArithmeticOverflow {
            context,
            detail: "shape_idx * CASES_PER_SHAPE + case_idx overflows u64".to_owned(),
        })?;
    Ok(seed_for(op_name, base, MASTER_SEED))
}

/// Fail-closed tolerance lookup for a gate case (Spec 1 §6.1, SI-60).
pub fn tolerance_for(op_name: &str) -> Result<Tolerance, HarnessError> {
    Tolerance::for_op(op_name).ok_or_else(|| HarnessError::UnknownOp {
        op: op_name.to_owned(),
    })
}

/// Fresh input and output buffers for one gate run (Spec 4 §10).
///
/// `build` constructs these from scratch on every call — including any
/// op state (KV caches, recurrent slots, RNG vectors) — so the
/// determinism gate proves fresh-state determinism, not buffer reuse.
pub struct GateBuffers {
    /// Owned inputs (fresh per run).
    pub inputs: Vec<TypedBuffer>,
    /// Owned outputs (zeroed; filled by `execute`).
    pub outputs: Vec<TypedBuffer>,
    /// Pinned logical row for [`GateCase::logical_bytes`]; set by
    /// `build_pinned`, ignored otherwise.
    pub logical_row: usize,
}

impl GateBuffers {
    /// Buffers for a plain (unpinned) run.
    pub fn fresh(inputs: Vec<TypedBuffer>, outputs: Vec<TypedBuffer>) -> Self {
        Self {
            inputs,
            outputs,
            logical_row: 0,
        }
    }

    /// Buffers for a pinned run over logical row `row`.
    pub fn pinned(inputs: Vec<TypedBuffer>, outputs: Vec<TypedBuffer>, row: usize) -> Self {
        Self {
            inputs,
            outputs,
            logical_row: row,
        }
    }
}

/// The three batch shapes plus the pinned logical row (Spec 1 §6.1).
///
/// The case pins the logical row's identity (positions, seq ids, slots,
/// query bytes, RNG state) across all three shapes; the engine compares
/// only that row's bytes. `row_alone` (usually 0) is the logical row's
/// index in `alone`; `row` (usually nonzero) is its index in
/// `padded`/`embedded`, so the comparison also proves row-index
/// independence. Each shape's row count must exceed its row index;
/// extraction reports a typed length error otherwise.
pub struct BatchRows {
    /// One logical row alone.
    pub alone: Vec<usize>,
    /// The logical row plus padding rows in one bucket.
    pub padded: Vec<usize>,
    /// The logical row embedded among random neighbor rows.
    pub embedded: Vec<usize>,
    /// Index of the pinned logical row in `alone`.
    pub row_alone: usize,
    /// Index of the pinned logical row in `padded` and `embedded`.
    pub row: usize,
}

/// One op's contract with the generic gate engine (Spec 4 §10, Card A1.10).
///
/// The engine calls `build`/`build_pinned` afresh for every run (never
/// reusing buffers or state across runs); `verify` compares the run's
/// outputs against an independent f64/source oracle — never against a
/// second call of the implementation under test.
pub trait GateCase {
    /// [`Tolerance::ALL_OP_NAMES`] entry, also the [`seed_for`] domain.
    fn op_name(&self) -> &'static str;
    /// The Spec 1 §6.1 floor for this op (use [`tolerance_for`]).
    fn tolerance(&self) -> Tolerance;
    /// Legal shapes including the [`bucket_edge_counts`] edges with tiny
    /// non-bucket dimensions. The engine runs exactly
    /// [`CASES_PER_SHAPE`] seeds per shape.
    fn legal_shapes(&self) -> Vec<Vec<usize>>;
    /// Deterministic legal shapes beyond [`GateCase::legal_shapes`]
    /// (engine runs each once through execute + verify).
    fn fuzz_legal(&self) -> Vec<Vec<usize>>;
    /// Fresh inputs/outputs for `shape` drawn from `seed`.
    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError>;
    /// Fresh inputs/outputs for `shape` with row `row` pinned to the
    /// logical identity stream (same bytes in alone/padded/embedded).
    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError>;
    /// The implementation under test, once, on fresh buffers.
    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), T0Error>;
    /// Golden comparison of `buffers.outputs` against the independent
    /// oracle for the inputs in `buffers`.
    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError>;
    /// Every output buffer serialized for twice-determinism comparison.
    fn output_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError>;
    /// The pinned logical row's bytes for batch-invariance comparison.
    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError>;
    /// The three batch shapes plus the pinned row.
    fn batch_rows(&self) -> BatchRows;
    /// Number of explicit illegal inputs.
    fn illegal_count(&self) -> usize;
    /// The `index`-th illegal input; `execute` must refuse it with a
    /// typed error (never panic).
    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError>;
    /// True when even legal inputs refuse (rank-1 `recv` has no data
    /// source, SI-54): golden/determinism require the typed refusal
    /// instead of outputs.
    fn always_refuses(&self) -> bool {
        false
    }
}

/// Gate 1: golden vs the independent oracle (Spec 4 §10).
///
/// Exactly [`CASES_PER_SHAPE`] seeds per legal shape from
/// [`GateCase::legal_shapes`]. A legal refusal is an
/// [`HarnessError::UnexpectedRefusal`]; an always-refusing case must
/// refuse every seed.
pub fn golden<C: GateCase + ?Sized>(case: &C) -> Result<(), HarnessError> {
    let shapes = case.legal_shapes();
    if shapes.is_empty() {
        return Err(HarnessError::NoLegalShapes {
            op: case.op_name().to_owned(),
        });
    }
    for (shape_idx, shape) in shapes.iter().enumerate() {
        for case_idx in 0..CASES_PER_SHAPE {
            let seed = case_seed(case.op_name(), shape_idx, case_idx)?;
            let context = format!("{} golden shape {shape:?} seed {seed}", case.op_name(),);
            let mut buffers = case.build(shape, seed)?;
            match case.execute(&mut buffers) {
                Ok(()) => {
                    if case.always_refuses() {
                        return Err(HarnessError::MissingRefusal {
                            context,
                            detail: "always-refusing op ran successfully".to_owned(),
                        });
                    }
                    case.verify(&buffers)?;
                }
                Err(error) => {
                    if case.always_refuses() {
                        continue;
                    }
                    return Err(HarnessError::UnexpectedRefusal {
                        context,
                        error: format!("{error:?}"),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Gate 2: batch invariance over the pinned logical row (Spec 1 §6.1).
///
/// Builds alone/padded/embedded via [`GateCase::build_pinned`] with the
/// same logical identity and requires bit-equal
/// [`GateCase::logical_bytes`]. Always-refusing cases must refuse all
/// three modes.
pub fn batch_invariant<C: GateCase + ?Sized>(case: &C) -> Result<(), HarnessError> {
    let rows = case.batch_rows();
    let context = format!("{} batch invariance", case.op_name());
    let mut pinned = Vec::with_capacity(3);
    for (shape, row) in [
        (&rows.alone, rows.row_alone),
        (&rows.padded, rows.row),
        (&rows.embedded, rows.row),
    ] {
        let mut buffers = case.build_pinned(shape, row)?;
        match case.execute(&mut buffers) {
            Ok(()) => {
                if case.always_refuses() {
                    return Err(HarnessError::MissingRefusal {
                        context: context.clone(),
                        detail: "always-refusing op ran successfully".to_owned(),
                    });
                }
                pinned.push(case.logical_bytes(&buffers)?);
            }
            Err(error) => {
                if case.always_refuses() {
                    pinned.push(format!("refused:{error:?}").into_bytes());
                    continue;
                }
                return Err(HarnessError::UnexpectedRefusal {
                    context: context.clone(),
                    error: format!("{error:?}"),
                });
            }
        }
    }
    check_bits_equal(&pinned[0], &pinned[1], &context)?;
    check_bits_equal(&pinned[0], &pinned[2], &context)?;
    Ok(())
}

/// Gate 3: run-to-run determinism from fresh state (Spec 4 §10).
///
/// Rebuilds every legal shape from its first seed twice (fresh inputs,
/// fresh state) and requires bit-equal [`GateCase::output_bytes`].
/// Always-refusing cases must refuse both rebuilds.
pub fn deterministic<C: GateCase + ?Sized>(case: &C) -> Result<(), HarnessError> {
    let shapes = case.legal_shapes();
    if shapes.is_empty() {
        return Err(HarnessError::NoLegalShapes {
            op: case.op_name().to_owned(),
        });
    }
    for (shape_idx, shape) in shapes.iter().enumerate() {
        let seed = case_seed(case.op_name(), shape_idx, 0)?;
        let context = format!("{} determinism shape {shape:?} seed {seed}", case.op_name(),);
        let first = run_to_bytes(case, shape, seed, &context)?;
        let second = run_to_bytes(case, shape, seed, &context)?;
        check_bits_equal(&first, &second, &context)?;
    }
    Ok(())
}

/// Runs execute once on a fresh build and returns the serialized outputs.
///
/// For always-refusing cases both the refusal marker and `Ok` encode
/// deterministically: two identical refusals match, a flaky
/// refuse/accept pair does not.
fn run_to_bytes<C: GateCase + ?Sized>(
    case: &C,
    shape: &[usize],
    seed: u64,
    context: &str,
) -> Result<Vec<u8>, HarnessError> {
    let mut buffers = case.build(shape, seed)?;
    match case.execute(&mut buffers) {
        Ok(()) => {
            if case.always_refuses() {
                return Err(HarnessError::MissingRefusal {
                    context: context.to_owned(),
                    detail: "always-refusing op ran successfully".to_owned(),
                });
            }
            case.output_bytes(&buffers)
        }
        Err(error) => {
            if case.always_refuses() {
                Ok(format!("refused:{error:?}").into_bytes())
            } else {
                Err(HarnessError::UnexpectedRefusal {
                    context: context.to_owned(),
                    error: format!("{error:?}"),
                })
            }
        }
    }
}

/// Gate 4: shape fuzz — legal shapes run, illegal shapes refuse
/// (Spec 4 §10).
///
/// Each [`GateCase::fuzz_legal`] shape executes and verifies once; each
/// of the [`GateCase::illegal_count`] illegal inputs must refuse with a
/// typed error ([`HarnessError::MissingRefusal`] if one runs).
pub fn shape_fuzz<C: GateCase + ?Sized>(case: &C) -> Result<(), HarnessError> {
    for (fuzz_idx, shape) in case.fuzz_legal().iter().enumerate() {
        let seed = seed_for(
            case.op_name(),
            LEGAL_FUZZ_DRAWS
                .saturating_mul(4096)
                .saturating_add(fuzz_idx as u64),
            MASTER_SEED,
        );
        let context = format!("{} legal fuzz {shape:?} #{fuzz_idx}", case.op_name());
        let mut buffers = case.build(shape, seed)?;
        match case.execute(&mut buffers) {
            Ok(()) => {
                if case.always_refuses() {
                    return Err(HarnessError::MissingRefusal {
                        context,
                        detail: "always-refusing op ran successfully".to_owned(),
                    });
                }
                case.verify(&buffers)?;
            }
            Err(error) => {
                if case.always_refuses() {
                    continue;
                }
                return Err(HarnessError::UnexpectedRefusal {
                    context,
                    error: format!("{error:?}"),
                });
            }
        }
    }
    for index in 0..case.illegal_count() {
        let context = format!("{} illegal fuzz #{index}", case.op_name());
        let mut buffers = case.build_illegal(index)?;
        if case.execute(&mut buffers).is_ok() {
            return Err(HarnessError::MissingRefusal {
                context,
                detail: "illegal input ran successfully".to_owned(),
            });
        }
    }
    Ok(())
}

/// All four Spec 4 §10 gates for one case (Card A1.10).
///
/// Runs [`golden`], [`batch_invariant`], [`deterministic`], then
/// [`shape_fuzz`] in order; the first failure reports with its gate's
/// context. Records no timing: the §10 perf/rate gates need a stored
/// baseline and a device (SI-62).
// DECISION(A1.10): fixed gate order (golden first) so a wrong-value bug
// reports before an invariance bug on the same case; rejected parallel
// gates because the first-failure context must be reproducible. Spec 4
// §10 lists golden first.
pub fn run_gates<C: GateCase + ?Sized>(case: &C) -> Result<(), HarnessError> {
    golden(case)?;
    batch_invariant(case)?;
    deterministic(case)?;
    shape_fuzz(case)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Output serialization helpers for `GateCase` implements.
// ---------------------------------------------------------------------------

/// Serializes an f32 output buffer as LE bits (exact bytes, Spec 1 App. B).
pub fn f32_output_bytes(buf: &TypedBuffer, context: &str) -> Result<Vec<u8>, HarnessError> {
    let vals = buf.to_f32_vec();
    let mut out = Vec::with_capacity(vals.len().saturating_mul(4));
    for v in vals {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    let _ = context;
    Ok(out)
}

/// Serializes a u32 output buffer as LE bytes (ids, Spec 1 App. B).
pub fn u32_output_bytes(buf: &TypedBuffer, context: &str) -> Result<Vec<u8>, HarnessError> {
    let vals = buf.to_u32_vec();
    let mut out = Vec::with_capacity(vals.len().saturating_mul(4));
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let _ = context;
    Ok(out)
}

/// Serializes an i8 output buffer as raw bytes (codes, Spec 1 App. B).
pub fn i8_output_bytes(buf: &TypedBuffer, _context: &str) -> Vec<u8> {
    buf.to_i8_vec().iter().map(|&v| v as u8).collect()
}

/// Serializes every buffer in `buffers` with `per_buffer`.
pub fn concat_output_bytes(
    buffers: &[TypedBuffer],
    per_buffer: impl Fn(&TypedBuffer) -> Vec<u8>,
) -> Vec<u8> {
    let mut out = Vec::new();
    for buf in buffers {
        out.extend_from_slice(&per_buffer(buf));
    }
    out
}

// ---------------------------------------------------------------------------
// Class fixture generators: one deterministic generator per
// `r9v_ir::Class` value (Spec 1 §2.3, Card A1.10).
//
// Same `(rng state)` => byte-identical buffers. Distributions mirror what
// each class carries: activations and weights are live signal, params are
// small compile-time values, fresh state and staging scratch start zeroed.
// ---------------------------------------------------------------------------

/// Number of [`Class`] values covered (Spec 1 §2.3).
pub const CLASS_COUNT: usize = 5;

/// All five [`Class`] values in declaration order.
pub const ALL_CLASSES: [Class; CLASS_COUNT] = [
    Class::Activation,
    Class::Weight,
    Class::State,
    Class::Staging,
    Class::Param,
];

/// f32 activation tensor (Class::Activation, Spec 1 §2.3): live signal in
/// `[-2, 2]`.
pub fn activation_class_tensor(rng: &mut SeededRng, shape: &[usize]) -> TypedBuffer {
    activation_tensor(rng, shape)
}

/// f32 weight tensor (Class::Weight, Spec 1 §2.3): values in `[-1, 1]`.
pub fn weight_class_tensor(rng: &mut SeededRng, shape: &[usize]) -> TypedBuffer {
    weight_tensor(rng, shape)
}

/// f32 state tensor (Class::State, Spec 1 §2.3): fresh state starts
/// zeroed; content arrives via state-write ops, never via seeding.
pub fn state_class_tensor(_rng: &mut SeededRng, shape: &[usize]) -> TypedBuffer {
    TypedBuffer::zeros(shape, DType::F32)
}

/// f32 staging tensor (Class::Staging, Spec 1 §2.3): scratch starts
/// zeroed; producers overwrite every element they read.
pub fn staging_class_tensor(_rng: &mut SeededRng, shape: &[usize]) -> TypedBuffer {
    TypedBuffer::zeros(shape, DType::F32)
}

/// f32 param tensor (Class::Param, Spec 1 §2.3): small compile-time
/// values (e.g. routing biases) in `[-0.5, 0.5]`.
pub fn param_class_tensor(rng: &mut SeededRng, shape: &[usize]) -> TypedBuffer {
    let len = shape
        .iter()
        .copied()
        .fold(1usize, |a, b| a.saturating_mul(b));
    TypedBuffer::from_f32(shape, &uniform_f32(rng, len, -0.5, 0.5))
}

/// Deterministic fixture for any [`Class`] value (Spec 1 §2.3).
///
/// Dispatches to the per-class generator above; every arm is exhaustive
/// with no wildcard so a new `Class` variant fails to compile here.
pub fn class_tensor(rng: &mut SeededRng, class: Class, shape: &[usize]) -> TypedBuffer {
    match class {
        Class::Activation => activation_class_tensor(rng, shape),
        Class::Weight => weight_class_tensor(rng, shape),
        Class::State => state_class_tensor(rng, shape),
        Class::Staging => staging_class_tensor(rng, shape),
        Class::Param => param_class_tensor(rng, shape),
    }
}

// ---------------------------------------------------------------------------
// Native wire-valid L0 fixtures (Spec 2 §3.2, Card A1.10).
//
// `scheme_weight_carrier` above tags deterministic fill bytes with any
// `SchemeId`: its scale geometry is valid but repack/IQ value bytes are
// NOT wire records and must fail closed at decode. The builders below are
// the opposite contract: wire-valid inline-L0 rows for exactly the four
// natively decoded schemes, each paired with its `r9v_format` decode
// oracle so tests prove the bytes decode through T0. Any other scheme
// fails closed with [`HarnessError::UnsupportedScheme`].
// ---------------------------------------------------------------------------

/// A wire-valid inline-L0 weight plus its independent decode oracle.
#[derive(Debug, Clone)]
pub struct NativeL0Weight {
    /// Bytes-backed `[rows, cols]` values with `QuantScheme::Scheme(id)`
    /// in the inline-L0 row stride T0 decodes (scales appended per row).
    pub values: TypedBuffer,
    /// Row-major `[rows, cols]` f32 from the `r9v_format` decode oracle.
    pub expected: Vec<f32>,
}

/// Inline-L0 stride for `scheme` with `cols` values per row (Spec 2 §3.2).
///
/// `I8R`: `cols + 2` (one trailing f16 scale); `I8B128`/`E4M3B128`:
/// `cols + 2 * (cols / 128)`; `I4K`: `cols / 2 + 16 * (cols / 256)`
/// (low nibble first, matching T0 and `decode_i4k_superblock`).
fn native_l0_stride(scheme: r9v_format::SchemeId, cols: usize) -> Result<usize, ()> {
    use r9v_format::SchemeId::*;
    match scheme {
        I8R => cols.checked_add(2).ok_or(()),
        I8B128 | E4M3B128 => {
            let scales = cols.checked_div(128).ok_or(())?;
            cols.checked_add(scales.saturating_mul(2)).ok_or(())
        }
        I4K => {
            let supers = cols.checked_div(256).ok_or(())?;
            cols.checked_div(2)
                .and_then(|v| v.checked_add(supers.saturating_mul(16)))
                .ok_or(())
        }
        _ => Err(()),
    }
}

/// Wire-valid inline-L0 weight for exactly the four native schemes.
///
/// `cols` must be a multiple of the scheme block (128 for
/// `I8B128`/`E4M3B128`, 256 for `I4K`); any other scheme returns
/// [`HarnessError::UnsupportedScheme`]. Same `(rng state)` =>
/// byte-identical buffers.
pub fn native_l0_weight(
    rng: &mut SeededRng,
    scheme: r9v_format::SchemeId,
    rows: usize,
    cols: usize,
) -> Result<NativeL0Weight, HarnessError> {
    use r9v_format::SchemeId::*;
    let context = format!("native_l0_weight({})", scheme.name());
    let fail = |detail: &str| HarnessError::UnsupportedScheme {
        scheme: scheme.name().to_owned(),
        detail: detail.to_owned(),
    };
    match scheme {
        I8R => {
            let (bytes, _) = i8r_l0_weight(rng, rows, cols);
            let expected = decode_i8r_l0(&bytes, rows, cols)?;
            Ok(NativeL0Weight {
                values: TypedBuffer::from_bytes(&[rows.max(1), cols.max(1)], DType::I8, &bytes)
                    .with_quant(QuantScheme::Scheme(scheme.to_ir())),
                expected,
            })
        }
        I8B128 => {
            if !cols.is_multiple_of(128) {
                return Err(fail("I8B128 needs cols % 128 == 0"));
            }
            let stride = native_l0_stride(scheme, cols).map_err(|()| fail("stride overflow"))?;
            let mut bytes = vec![0u8; rows.saturating_mul(stride)];
            let mut expected = Vec::with_capacity(rows.saturating_mul(cols));
            for r in 0..rows {
                let q = symmetric_i8(rng, cols);
                let scales_f32 = positive_scales(rng, cols / 128);
                let base = r.saturating_mul(stride);
                for (c, &v) in q.iter().enumerate() {
                    bytes[base.saturating_add(c)] = v as u8;
                }
                for (b, &s) in scales_f32.iter().enumerate() {
                    let bits = crate::dtype::f32_to_f16(s);
                    let off = base.saturating_add(cols).saturating_add(b.saturating_mul(2));
                    bytes[off] = bits.to_le_bytes()[0];
                    bytes[off.saturating_add(1)] = bits.to_le_bytes()[1];
                }
                let oracle = decode_i8b128_l0(&q, &scales_f32, &context)?;
                expected.extend_from_slice(&oracle);
            }
            Ok(NativeL0Weight {
                values: TypedBuffer::from_bytes(&[rows.max(1), cols.max(1)], DType::I8, &bytes)
                    .with_quant(QuantScheme::Scheme(scheme.to_ir())),
                expected,
            })
        }
        E4M3B128 => {
            if !cols.is_multiple_of(128) {
                return Err(fail("E4M3B128 needs cols % 128 == 0"));
            }
            let stride = native_l0_stride(scheme, cols).map_err(|()| fail("stride overflow"))?;
            let mut bytes = vec![0u8; rows.saturating_mul(stride)];
            let mut expected = Vec::with_capacity(rows.saturating_mul(cols));
            for r in 0..rows {
                let vals = activation_values(rng, cols);
                let scales_f32 = positive_scales(rng, cols / 128);
                let base = r.saturating_mul(stride);
                let mut codes = Vec::with_capacity(cols);
                for (c, &v) in vals.iter().enumerate() {
                    let code = crate::dtype::fp8_e4m3_encode(v);
                    bytes[base.saturating_add(c)] = code;
                    codes.push(code);
                }
                for (b, &s) in scales_f32.iter().enumerate() {
                    let bits = crate::dtype::f32_to_f16(s);
                    let off = base.saturating_add(cols).saturating_add(b.saturating_mul(2));
                    bytes[off] = bits.to_le_bytes()[0];
                    bytes[off.saturating_add(1)] = bits.to_le_bytes()[1];
                }
                let oracle = decode_e4m3b128_l0(&codes, &scales_f32, &context)?;
                expected.extend_from_slice(&oracle);
            }
            Ok(NativeL0Weight {
                values: TypedBuffer::from_bytes(&[rows.max(1), cols.max(1)], DType::E4m3, &bytes)
                    .with_quant(QuantScheme::Scheme(scheme.to_ir())),
                expected,
            })
        }
        I4K => {
            if !cols.is_multiple_of(256) {
                return Err(fail("I4K needs cols % 256 == 0"));
            }
            let stride = native_l0_stride(scheme, cols).map_err(|()| fail("stride overflow"))?;
            let mut bytes = vec![0u8; rows.saturating_mul(stride)];
            let mut expected = Vec::with_capacity(rows.saturating_mul(cols));
            for r in 0..rows {
                let base = r.saturating_mul(stride);
                let end = base.saturating_add(stride);
                let row_oracle = pack_i4k_row(rng, &mut bytes[base..end], cols, &context)?;
                expected.extend_from_slice(&row_oracle);
            }
            Ok(NativeL0Weight {
                values: TypedBuffer::from_bytes(&[rows.max(1), cols.max(1)], DType::I4, &bytes)
                    .with_quant(QuantScheme::Scheme(scheme.to_ir())),
                expected,
            })
        }
        _ => Err(fail(
            "only I8R, I8B128, I4K, E4M3B128 have T0 decode paths; repack/IQ carriers are geometry fixtures, not wire records",
        )),
    }
}

/// Decodes `I8R` inline-L0 rows via the format oracle (Spec 2 §3.2).
fn decode_i8r_l0(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>, HarnessError> {
    use r9v_format::decode::decode_i8_row;
    use r9v_format::records::I8RowScale;
    let mut out = Vec::with_capacity(rows.saturating_mul(cols));
    for r in 0..rows {
        let base = r.saturating_mul(cols.saturating_add(2));
        let mut q = Vec::with_capacity(cols);
        for c in 0..cols {
            q.push(bytes[base.saturating_add(c)] as i8);
        }
        let scale = I8RowScale::from_bytes([
            bytes[base.saturating_add(cols)],
            bytes[base.saturating_add(cols).saturating_add(1)],
        ]);
        out.extend_from_slice(&decode_i8_row(&q, &scale)?);
    }
    Ok(out)
}

/// Decodes one `I8B128` row's 128-blocks via the format oracle.
fn decode_i8b128_l0(q: &[i8], scales_f32: &[f32], context: &str) -> Result<Vec<f32>, HarnessError> {
    use r9v_format::decode::decode_i8_block128;
    use r9v_format::records::I8Block128Scale;
    let mut scales = Vec::with_capacity(scales_f32.len());
    for &s in scales_f32 {
        scales.push(I8Block128Scale::from_bits(crate::dtype::f32_to_f16(s)));
    }
    let _ = context;
    decode_i8_block128(q, &scales).map_err(HarnessError::Format)
}

/// Decodes one `E4M3B128` row's 128-blocks via the format oracle.
fn decode_e4m3b128_l0(
    codes: &[u8],
    scales_f32: &[f32],
    context: &str,
) -> Result<Vec<f32>, HarnessError> {
    use r9v_format::decode::decode_e4m3_block128;
    use r9v_format::records::E4M3Block128Scale;
    use r9v_format::scales::E4m3;
    let _ = context;
    let mut q = Vec::with_capacity(codes.len());
    for &b in codes {
        q.push(E4m3::new(b));
    }
    let mut scales = Vec::with_capacity(scales_f32.len());
    for &s in scales_f32 {
        scales.push(E4M3Block128Scale::from_bits(crate::dtype::f32_to_f16(s)));
    }
    decode_e4m3_block128(&q, &scales).map_err(HarnessError::Format)
}

/// Packs one `I4K` inline-L0 row (low nibble first) and returns its
/// format-oracle dequant (Spec 2 §3.2; llama.cpp `dequantize_row_q4_K`
/// order).
fn pack_i4k_row(
    rng: &mut SeededRng,
    row_bytes: &mut [u8],
    cols: usize,
    context: &str,
) -> Result<Vec<f32>, HarnessError> {
    use r9v_format::decode::decode_i4k_superblock;
    use r9v_format::records::I4KSuperblock;
    let fail = |detail: String| HarnessError::UnsupportedShape {
        context: context.to_owned(),
        detail,
    };
    let supers = cols / 256;
    if row_bytes.len() < cols / 2 + supers * 16 {
        return Err(fail("I4K row buffer too short".to_owned()));
    }
    let mut out = Vec::with_capacity(cols);
    for sb in 0..supers {
        let mut nibbles = [0u8; 256];
        for v in nibbles.iter_mut() {
            *v = (rng.next_u64() % 16) as u8;
        }
        let val_base = sb * 128;
        for (i, chunk) in nibbles.chunks_exact(2).enumerate() {
            row_bytes[val_base + i] = chunk[0] | (chunk[1] << 4);
        }
        let d_bits = crate::dtype::f32_to_f16(positive_scales(rng, 1)[0]);
        let dmin_bits = crate::dtype::f32_to_f16(positive_scales(rng, 1)[0]);
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for v in sc.iter_mut() {
            *v = (rng.next_u64() % 64) as u8;
        }
        for v in mn.iter_mut() {
            *v = (rng.next_u64() % 64) as u8;
        }
        let header = I4KSuperblock::pack(d_bits, dmin_bits, sc, mn)?;
        let hdr_base = cols / 2 + sb * 16;
        row_bytes[hdr_base..hdr_base + 16].copy_from_slice(&header.to_bytes());
        let parsed = I4KSuperblock::from_bytes(&header.to_bytes());
        out.extend_from_slice(&decode_i4k_superblock(&nibbles, &parsed)?);
    }
    Ok(out)
}
