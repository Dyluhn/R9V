// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of the last-axis channel concatenation (card A1.14, SI-20, Spec 4 §2).

use r9v_ir::{ConcatOp, DType};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Executes scalar T0 channel concatenation: `(a [T, H, Da], b [T, H, Db])`
/// into `y [T, H, Da + Db]`, copying values unchanged in ascending index
/// order (card A1.14, SI-20, Spec 4 §2).
pub fn concat(
    _op: &ConcatOp,
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    a.validate_backing("a")?;
    b.validate_backing("b")?;
    y.validate_backing("y")?;

    let mut problems = Vec::new();

    for (name, view) in [("a", a), ("b", b)] {
        if view.rank() != 3 {
            problems.push(format!(
                "input {name}: expected rank 3 [T, H, D], got rank {} with shape {:?}",
                view.rank(),
                view.shape()
            ));
        }
        if !matches!(view.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(format!(
                "input {name}: expected f16, bf16, or f32, got {:?}",
                view.dtype()
            ));
        }
    }
    if y.rank() != 3 {
        problems.push(format!(
            "output y: expected rank 3 [T, H, Da + Db], got rank {} with shape {:?}",
            y.rank(),
            y.shape()
        ));
    }
    if !matches!(y.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(format!(
            "output y: expected f16, bf16, or f32, got {:?}",
            y.dtype()
        ));
    }
    if b.dtype() != a.dtype() {
        problems.push(format!(
            "input b: expected input a dtype {:?}, got {:?}",
            a.dtype(),
            b.dtype()
        ));
    }
    if y.dtype() != a.dtype() {
        problems.push(format!(
            "output y: expected input a dtype {:?}, got {:?}",
            a.dtype(),
            y.dtype()
        ));
    }
    if a.rank() == 3 && b.rank() == 3 && y.rank() == 3 {
        let (t, h, da) = (a.shape()[0], a.shape()[1], a.shape()[2]);
        if b.shape()[0] != t || b.shape()[1] != h {
            problems.push(format!(
                "input b shape {:?} does not share input a [T, H] = [{t}, {h}]",
                b.shape()
            ));
        }
        if y.shape()[0] != t || y.shape()[1] != h {
            problems.push(format!(
                "output y shape {:?} does not share input a [T, H] = [{t}, {h}]",
                y.shape()
            ));
        }
        match da.checked_add(b.shape()[2]) {
            Some(sum) if sum == y.shape()[2] => {}
            Some(sum) => problems.push(format!(
                "input widths {da} + {} = {sum} do not match output dim {}",
                b.shape()[2],
                y.shape()[2]
            )),
            None => problems.push("input widths overflow usize".to_string()),
        }
    }

    T0Error::from_problems("concat", problems)?;

    let (t, h, da) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let db = b.shape()[2];
    for token in 0..t {
        for head in 0..h {
            let out_base = (token * h + head) * (da + db);
            for k in 0..da {
                y.write_f32(out_base + k, a.read_f32((token * h + head) * da + k));
            }
            for k in 0..db {
                y.write_f32(out_base + da + k, b.read_f32((token * h + head) * db + k));
            }
        }
    }

    Ok(())
}

/// Straightforward 64-bit floating point reference implementation for testing against T0 (card A1.14, SI-20).
pub fn concat_f64_reference(a: &[f64], b: &[f64], t: usize, h: usize) -> Vec<f64> {
    assert_eq!(a.len() % (t * h), 0);
    assert_eq!(b.len() % (t * h), 0);
    let da = a.len() / (t * h);
    let db = b.len() / (t * h);
    let mut y = vec![0.0f64; t * h * (da + db)];
    for token in 0..t {
        for head in 0..h {
            y[(token * h + head) * (da + db)..(token * h + head) * (da + db) + da]
                .copy_from_slice(&a[(token * h + head) * da..(token * h + head + 1) * da]);
            y[(token * h + head) * (da + db) + da..(token * h + head + 1) * (da + db)]
                .copy_from_slice(&b[(token * h + head) * db..(token * h + head + 1) * db]);
        }
    }
    y
}
