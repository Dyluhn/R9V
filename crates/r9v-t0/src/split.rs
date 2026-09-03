// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of the last-axis channel split (card A1.14, SI-20, Spec 4 §2).

use r9v_ir::{DType, SplitOp};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Executes scalar T0 channel split: `x [T, H, D]` into `a [T, H, first]`
/// and `b [T, H, D - first]`, copying values unchanged in ascending index
/// order (card A1.14, SI-20, Spec 4 §2).
pub fn split(
    op: &SplitOp,
    x: &TensorView<'_>,
    a: &mut TensorViewMut<'_>,
    b: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    a.validate_backing("a")?;
    b.validate_backing("b")?;

    let mut problems = Vec::new();

    if op.first == 0 {
        problems.push("attribute first: split width must be > 0".to_string());
    }
    if x.rank() != 3 {
        problems.push(format!(
            "input x: expected rank 3 [T, H, D], got rank {} with shape {:?}",
            x.rank(),
            x.shape()
        ));
    }
    if a.rank() != 3 {
        problems.push(format!(
            "output a: expected rank 3 [T, H, first], got rank {} with shape {:?}",
            a.rank(),
            a.shape()
        ));
    }
    if b.rank() != 3 {
        problems.push(format!(
            "output b: expected rank 3 [T, H, D - first], got rank {} with shape {:?}",
            b.rank(),
            b.shape()
        ));
    }
    for (name, view) in [("x", x.dtype()), ("a", a.dtype()), ("b", b.dtype())] {
        if !matches!(view, DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(format!(
                "tensor {name}: expected f16, bf16, or f32, got {view:?}"
            ));
        }
    }
    if a.dtype() != x.dtype() {
        problems.push(format!(
            "output a: expected input x dtype {:?}, got {:?}",
            x.dtype(),
            a.dtype()
        ));
    }
    if b.dtype() != x.dtype() {
        problems.push(format!(
            "output b: expected input x dtype {:?}, got {:?}",
            x.dtype(),
            b.dtype()
        ));
    }
    if x.rank() == 3 && a.rank() == 3 && b.rank() == 3 {
        let (t, h, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if a.shape()[0] != t || a.shape()[1] != h {
            problems.push(format!(
                "output a shape {:?} does not share input [T, H] = [{t}, {h}]",
                a.shape()
            ));
        }
        if b.shape()[0] != t || b.shape()[1] != h {
            problems.push(format!(
                "output b shape {:?} does not share input [T, H] = [{t}, {h}]",
                b.shape()
            ));
        }
        if a.shape()[2] != op.first as usize {
            problems.push(format!(
                "output a width {} does not match split attr first={}",
                a.shape()[2],
                op.first
            ));
        }
        match (op.first as usize).checked_add(b.shape()[2]) {
            Some(sum) if sum == d => {}
            Some(sum) => problems.push(format!(
                "output widths {} + {} = {sum} do not reconstruct input dim {d}",
                op.first,
                b.shape()[2]
            )),
            None => problems.push("output widths overflow usize".to_string()),
        }
    }

    T0Error::from_problems("split", problems)?;

    let (t, h, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let first = op.first as usize;
    for token in 0..t {
        for head in 0..h {
            let base = (token * h + head) * d;
            for k in 0..first {
                a.write_f32((token * h + head) * first + k, x.read_f32(base + k));
            }
            let second = d - first;
            for k in 0..second {
                b.write_f32(
                    (token * h + head) * second + k,
                    x.read_f32(base + first + k),
                );
            }
        }
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0 (card A1.14, SI-20).
pub fn split_f64_reference(x: &[f64], shape: [usize; 3], first: usize) -> (Vec<f64>, Vec<f64>) {
    let [t, h, d] = shape;
    assert_eq!(x.len(), t * h * d);
    assert!(first > 0 && first < d);
    let mut a = vec![0.0f64; t * h * first];
    let mut b = vec![0.0f64; t * h * (d - first)];
    for token in 0..t {
        for head in 0..h {
            let base = (token * h + head) * d;
            a[(token * h + head) * first..(token * h + head) * first + first]
                .copy_from_slice(&x[base..base + first]);
            b[(token * h + head) * (d - first)..(token * h + head + 1) * (d - first)]
                .copy_from_slice(&x[base + first..base + d]);
        }
    }
    (a, b)
}
