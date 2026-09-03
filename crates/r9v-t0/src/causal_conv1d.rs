// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Scalar T0 implementation of `causal_conv1d` op (Spec 1 §4.E, Card A1.9).
//!
//! Depthwise causal 1D convolution with explicit sequence segmentation and
//! fail-before-mutation state continuity: every arithmetic input is the
//! decoded `f16` bit pattern. Split-vs-one-shot is bit-exact only when the
//! carried boundary history is already `f16`-representable; for general
//! `f32`/`bf16` inputs the boundary rows round through the specified `f16`
//! state quantization, so continuity there is approximate within tolerance.

use r9v_ir::{CausalConv1dOp, ConvActivation, DType, LayoutId};

use crate::activation::eval_activation_f32;
use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;
use crate::segments::SeqLayout;
use r9v_ir::ActivationKind;

/// Executes scalar T0 causal depthwise 1D convolution (Spec 1 §4.E, Card A1.9).
///
/// Signature:
/// - `x`: `[T, C]` (`f16|bf16|f32`)
/// - `w`: `[C, Wk]` (`f16|bf16|f32`; `Wk == op.kernel`)
/// - `bias`: optional `[C]`
/// - `state_in`/`state_out`: `[S, Wk-1, C]` (`f16`, `Wk == 1` gives zero rows)
/// - `y`: `[T, C]` (dtype matches `x`)
/// - `seq`: explicit per-sequence token counts summing to `T` (SI-48)
///
/// Recurrence per channel `c`, ascending `t` within each segment, all MAC in
/// `f32`: `acc = bias[c] + Σ_{i<Wk} w[c,i]·hist[t−Wk+1+i][c]` with
/// `hist = [state rows; segment x rows]`, every element decoded to `f32` at
/// use; `y[t,c] = Silu(acc)` or `acc`. `state_out[s]` holds the last `Wk−1`
/// rows of segment `s`'s history, encoded `f16`.
///
/// Continuity guarantee by construction: chunked runs with carried state
/// consume the same decoded `f16` bit patterns in the same order as a
/// one-shot run, so split-vs-oneshot agrees bit-exactly (L0) whenever the
/// carried boundary history is already `f16`-representable. For general
/// `f32`/`bf16` inputs the boundary rows pass through the specified `f16`
/// state quantization and continuity holds only within tolerance.
///
/// State inputs are never mutated; `y` and `state_out` are staged in owned
/// buffers and committed only after all validation and compute succeed.
///
/// Fail-closed (SI-55): quantized weights (`i8`/`i4`, IR-permitted but with no
/// scale input in the signature) are rejected with [`T0Error::QuantMismatch`].
///
/// DECISION(A1.9): state slots are `f16` exactly with staged `y`/state commit;
/// rejected wider state and in-place updates because the spec fixes `f16`
/// windows and A/B slots must never alias. Per SI-55.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d(
    op: &CausalConv1dOp,
    x: &TensorView<'_>,
    w: &TensorView<'_>,
    bias: Option<&TensorView<'_>>,
    state_in: &TensorView<'_>,
    seq: &SeqLayout,
    y: &mut TensorViewMut<'_>,
    state_out: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    w.validate_backing("w")?;
    if let Some(b) = bias {
        b.validate_backing("bias")?;
    }
    state_in.validate_backing("state_in")?;
    y.validate_backing("y")?;
    state_out.validate_backing("state_out")?;

    let mut problems = Vec::new();

    if x.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "x",
            expected: 2,
            got: x.rank(),
            shape: x.shape().to_vec(),
        });
    }
    if w.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "w",
            expected: 2,
            got: w.rank(),
            shape: w.shape().to_vec(),
        });
    }
    if y.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "y",
            expected: 2,
            got: y.rank(),
            shape: y.shape().to_vec(),
        });
    }
    if state_in.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "state_in",
            expected: 3,
            got: state_in.rank(),
            shape: state_in.shape().to_vec(),
        });
    }
    if state_out.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "state_out",
            expected: 3,
            got: state_out.rank(),
            shape: state_out.shape().to_vec(),
        });
    }

    for (name, v) in [("x", x), ("w", w), ("state_in", state_in)] {
        let vv: &TensorView<'_> = v;
        if vv.layout() != LayoutId::CONTIGUOUS && vv.layout() != LayoutId::L0 {
            problems.push(T0Error::LayoutMismatch {
                tensor: name,
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
                got: vv.layout(),
            });
        }
    }
    if let Some(b) = bias {
        if b.layout() != LayoutId::CONTIGUOUS && b.layout() != LayoutId::L0 {
            problems.push(T0Error::LayoutMismatch {
                tensor: "bias",
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
                got: b.layout(),
            });
        }
    }
    if y.layout() != LayoutId::CONTIGUOUS && y.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "y",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: y.layout(),
        });
    }
    if state_out.layout() != LayoutId::CONTIGUOUS && state_out.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "state_out",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: state_out.layout(),
        });
    }

    if !matches!(x.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: x.dtype(),
        });
    }
    match w.dtype() {
        DType::F16 | DType::Bf16 | DType::F32 => {}
        DType::I8 | DType::I4 => problems.push(T0Error::QuantMismatch {
            tensor: "w",
            expected: vec![r9v_ir::QuantScheme::None],
            got: w.quant(),
        }),
        other => problems.push(T0Error::DTypeMismatch {
            tensor: "w",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: other,
        }),
    }
    if state_in.dtype() != DType::F16 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "state_in",
            expected: vec![DType::F16],
            got: state_in.dtype(),
        });
    }
    if state_out.dtype() != DType::F16 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "state_out",
            expected: vec![DType::F16],
            got: state_out.dtype(),
        });
    }
    if y.dtype() != x.dtype() {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![x.dtype()],
            got: y.dtype(),
        });
    }
    if op.kernel == 0 {
        problems.push(T0Error::InvalidAttribute {
            op: "causal_conv1d",
            attribute: "kernel",
            reason: "kernel must be > 0".to_string(),
        });
    }
    match op.act {
        ConvActivation::Silu | ConvActivation::Identity => {}
    }
    match op.handle.kind() {
        r9v_ir::StateKind::ConvWindow => {}
        other => problems.push(T0Error::InvalidAttribute {
            op: "causal_conv1d",
            attribute: "handle",
            reason: format!("state handle must be ConvWindow, got {other:?}"),
        }),
    }

    T0Error::from_problems(problems)?;

    let t = x.shape()[0];
    let c = x.shape()[1];
    let wk = op.kernel as usize;
    let s = seq.seq_count();

    if t == 0 || c == 0 {
        return Err(T0Error::EmptyInput {
            op: "causal_conv1d",
            tensor: "x",
        });
    }
    for v in [t, c, wk, s] {
        u32::try_from(v).map_err(|_| T0Error::ArithmeticOverflow {
            op: "causal_conv1d",
            detail: format!("dimension exceeds u32: {v}"),
        })?;
    }
    seq.check_total("state_in", t)?;

    let mut problems = Vec::new();
    if w.shape()[0] != c {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "C",
            expected_from: "x",
            expected: c,
            tensor: "w",
            got: w.shape()[0],
        });
    }
    if w.shape()[1] != wk {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Wk",
            expected_from: "kernel",
            expected: wk,
            tensor: "w",
            got: w.shape()[1],
        });
    }
    if let Some(b) = bias {
        if b.rank() != 1 {
            problems.push(T0Error::RankMismatch {
                tensor: "bias",
                expected: 1,
                got: b.rank(),
                shape: b.shape().to_vec(),
            });
        } else if b.shape()[0] != c {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "C",
                expected_from: "x",
                expected: c,
                tensor: "bias",
                got: b.shape()[0],
            });
        }
        if !matches!(b.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(T0Error::DTypeMismatch {
                tensor: "bias",
                expected: vec![DType::F16, DType::Bf16, DType::F32],
                got: b.dtype(),
            });
        }
    }
    if y.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "x",
            expected: t,
            tensor: "y",
            got: y.shape()[0],
        });
    }
    if y.shape()[1] != c {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "C",
            expected_from: "x",
            expected: c,
            tensor: "y",
            got: y.shape()[1],
        });
    }
    let hist_rows = wk.saturating_sub(1);
    if state_in.shape()[0] != s {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "S",
            expected_from: "seq",
            expected: s,
            tensor: "state_in",
            got: state_in.shape()[0],
        });
    }
    if state_in.shape()[1] != hist_rows {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Wk-1",
            expected_from: "kernel",
            expected: hist_rows,
            tensor: "state_in",
            got: state_in.shape()[1],
        });
    }
    if state_in.shape()[2] != c {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "C",
            expected_from: "x",
            expected: c,
            tensor: "state_in",
            got: state_in.shape()[2],
        });
    }
    if state_out.shape()[0] != s {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "S",
            expected_from: "seq",
            expected: s,
            tensor: "state_out",
            got: state_out.shape()[0],
        });
    }
    if state_out.shape()[1] != hist_rows {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Wk-1",
            expected_from: "kernel",
            expected: hist_rows,
            tensor: "state_out",
            got: state_out.shape()[1],
        });
    }
    if state_out.shape()[2] != c {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "C",
            expected_from: "x",
            expected: c,
            tensor: "state_out",
            got: state_out.shape()[2],
        });
    }
    T0Error::from_problems(problems)?;

    // Validated: compute into staged buffers, then commit.
    let y_len = t
        .checked_mul(c)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "causal_conv1d",
            detail: format!("T * C overflows usize for T={t}, C={c}"),
        })?;
    let mut y_tmp = vec![0.0f32; y_len];
    let s_len = s
        .checked_mul(hist_rows)
        .and_then(|v| v.checked_mul(c))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "causal_conv1d",
            detail: "state buffer size overflows usize".to_string(),
        })?;
    let mut s_tmp = vec![0u16; s_len];

    let act_kind = match op.act {
        ConvActivation::Silu => ActivationKind::Silu,
        ConvActivation::Identity => ActivationKind::Identity,
    };
    // Absolute history rows [0, hist) are the segment's input state rows;
    // rows [hist, hist + len) are the segment's x rows. Since wk == hist + 1,
    // tap i of output row sits at absolute history row (row + i).
    let mut base = 0usize;
    for (slot, &len_u32) in seq.seq_lens().iter().enumerate() {
        let len = len_u32 as usize;
        for row in 0..len {
            let gt = base + row;
            for ch in 0..c {
                let mut acc = match bias {
                    Some(b) => b.read_f32(ch),
                    None => 0.0f32,
                };
                for i in 0..wk {
                    let abs = row + i;
                    let h_val = if abs < hist_rows {
                        state_in.read_f32(slot * hist_rows * c + abs * c + ch)
                    } else {
                        x.read_f32((base + abs - hist_rows) * c + ch)
                    };
                    acc += w.read_f32(ch * wk + i) * h_val;
                }
                y_tmp[gt * c + ch] = eval_activation_f32(acc, act_kind);
            }
        }
        // New tail: the last `hist` history rows, at absolute rows [len, len + hist).
        for h_row in 0..hist_rows {
            let abs = len + h_row;
            for ch in 0..c {
                let v = if abs < hist_rows {
                    state_in.read_f32(slot * hist_rows * c + abs * c + ch)
                } else {
                    x.read_f32((base + abs - hist_rows) * c + ch)
                };
                s_tmp[slot * hist_rows * c + h_row * c + ch] = crate::dtype::f32_to_f16(v);
            }
        }
        base += len;
    }

    for (idx, &v) in y_tmp.iter().enumerate() {
        y.write_f32(idx, v);
    }
    // Commit staged f16 state bits.
    for (idx, &bits) in s_tmp.iter().enumerate() {
        write_f16_bits(state_out, idx, bits)?;
    }
    Ok(())
}

/// 64-bit reference causal convolution for testing (Spec 1 §4.E).
///
/// Independent `f64` path: plain nested loops over `&[f64]` slices, never
/// calling [`causal_conv1d`]. `state_in` arrives as decoded `f64` rows
/// (the test decodes the `f16` bits); the recurrence mirrors the T0
/// algorithm in `f64`. Returns `(y [T, C], state_out [S, hist, C])` with the
/// state tail kept in full precision (the test compares against T0 through
/// the `f32` tolerance, which absorbs the `f16` tail rounding). Slice
/// lengths and every extent product are validated with typed errors; there
/// is no silent empty fallback.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_f64_reference(
    x: &[f64],
    t: usize,
    c: usize,
    w: &[f64],
    wk: usize,
    bias: Option<&[f64]>,
    act: ConvActivation,
    state_in: &[f64],
    s: usize,
    seq_lens: &[u32],
) -> Result<(Vec<f64>, Vec<f64>), T0Error> {
    const OP: &str = "causal_conv1d";
    let hist = wk.saturating_sub(1);
    let tc = t
        .checked_mul(c)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("T * C overflows usize for T={t}, C={c}"),
        })?;
    let cwk = c
        .checked_mul(wk)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("C * Wk overflows usize for C={c}, Wk={wk}"),
        })?;
    let shc = s
        .checked_mul(hist)
        .and_then(|v| v.checked_mul(c))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "S * hist * C overflows usize".to_string(),
        })?;
    let mut problems = Vec::new();
    if x.len() != tc {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "x",
            expected: tc,
            got: x.len(),
            detail: "x length must equal T * C".to_string(),
        });
    }
    if w.len() != cwk {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "w",
            expected: cwk,
            got: w.len(),
            detail: "w length must equal C * Wk".to_string(),
        });
    }
    if let Some(b) = bias {
        if b.len() != c {
            problems.push(T0Error::ShapeLengthMismatch {
                op: OP,
                tensor: "bias",
                expected: c,
                got: b.len(),
                detail: "bias length must equal C".to_string(),
            });
        }
    }
    if state_in.len() != shc {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "state_in",
            expected: shc,
            got: state_in.len(),
            detail: "state_in length must equal S * hist * C".to_string(),
        });
    }
    if seq_lens.len() != s {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "seq_lens",
            expected: s,
            got: seq_lens.len(),
            detail: "seq_lens length must equal S".to_string(),
        });
    }
    T0Error::from_problems(problems)?;
    let mut y = vec![0.0f64; tc];
    let mut s_out = vec![0.0f64; shc];
    let silu = |a: f64| match act {
        ConvActivation::Silu => a / (1.0 + (-a).exp()),
        ConvActivation::Identity => a,
    };
    let mut base = 0usize;
    for (slot, &len_u32) in seq_lens.iter().enumerate() {
        let len = len_u32 as usize;
        for row in 0..len {
            for ch in 0..c {
                let mut acc = bias.map(|b| b[ch]).unwrap_or(0.0);
                for i in 0..wk {
                    let abs = row + i;
                    let h_val = if abs < hist {
                        state_in[slot * hist * c + abs * c + ch]
                    } else {
                        x[(base + abs - hist) * c + ch]
                    };
                    acc += w[ch * wk + i] * h_val;
                }
                y[(base + row) * c + ch] = silu(acc);
            }
        }
        for h_row in 0..hist {
            let abs = len + h_row;
            for ch in 0..c {
                let v = if abs < hist {
                    state_in[slot * hist * c + abs * c + ch]
                } else {
                    x[(base + abs - hist) * c + ch]
                };
                s_out[slot * hist * c + h_row * c + ch] = v;
            }
        }
        base += len;
    }
    Ok((y, s_out))
}

/// Writes raw `f16` bits to a `f16` output view (Spec 1 §4.E).
///
/// Accepts typed `F16` and raw-byte `F16` backings bit-exactly.
fn write_f16_bits(view: &mut TensorViewMut<'_>, index: usize, bits: u16) -> Result<(), T0Error> {
    match view.data {
        crate::buffer::TensorDataMut::F16(ref mut slice) => {
            if index >= slice.len() {
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "state_out",
                    buffer_len: slice.len(),
                    expected_len: index + 1,
                    shape: view.shape().to_vec(),
                });
            }
            slice[index] = bits;
            Ok(())
        }
        crate::buffer::TensorDataMut::Bytes(_, ref mut slice) => {
            let end = index
                .checked_mul(2)
                .and_then(|off| off.checked_add(2))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "causal_conv1d",
                    detail: format!("f16 byte range for index {index} overflows usize"),
                })?;
            if end > slice.len() {
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "state_out",
                    buffer_len: slice.len(),
                    expected_len: end,
                    shape: view.shape().to_vec(),
                });
            }
            slice[index * 2..end].copy_from_slice(&bits.to_le_bytes());
            Ok(())
        }
        _ => Err(T0Error::BackingRepresentationMismatch {
            op: "causal_conv1d",
            dtype: view.dtype(),
        }),
    }
}
