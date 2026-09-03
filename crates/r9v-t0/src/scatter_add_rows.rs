// SPDX-License-Identifier: Apache-2.0
//! Scalar T0 implementation of `scatter_add_rows` op (Spec 1 §4.A, §6.1, Card A1.6).

use r9v_ir::{DType, LayoutId, ScatterAddRowsOp};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::T0Error;

/// Scatters and adds rows from input `x` according to integer `indices` into output `y` (Spec 1 §4.A, Card A1.6).
///
/// Signature:
/// - `x`: `[M, D]` (`f16`, `bf16`, or `f32`)
/// - `indices`: `[M]` (`u32`)
/// - `dest`: optional `[N, D]` (dtype matches `x.dtype()`)
/// - `y`: `[N, D]` (dtype matches `x.dtype()`)
///
/// Numerics:
/// - Deterministic sorted-index order with sequential accumulation in `f32` per destination (Spec 1 §4.A, §6.1).
/// - DECISION(A1.6): duplicate/tied indices targeting the same destination row are accumulated in
///   strictly ascending source index order `m` (0..M), satisfying `ReductionOrder::AscendingIndex` and
///   guaranteeing bit-identical run-to-run determinism.
pub fn scatter_add_rows(
    _op: &ScatterAddRowsOp,
    x: &TensorView<'_>,
    indices: &TensorView<'_>,
    dest: Option<&TensorView<'_>>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    indices.validate_backing("indices")?;
    if let Some(d) = dest {
        d.validate_backing("dest")?;
    }
    y.validate_backing("y")?;

    let mut problems = Vec::new();

    if x.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "x",
            expected: 2,
            got: x.rank(),
            shape: x.shape().to_vec(),
        });
    }

    if indices.rank() != 1 {
        problems.push(T0Error::RankMismatch {
            tensor: "indices",
            expected: 1,
            got: indices.rank(),
            shape: indices.shape().to_vec(),
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

    if let Some(d) = dest {
        if d.rank() != 2 {
            problems.push(T0Error::RankMismatch {
                tensor: "dest",
                expected: 2,
                got: d.rank(),
                shape: d.shape().to_vec(),
            });
        }
    }

    // Validate exact supported layouts
    if x.layout() != LayoutId::CONTIGUOUS && x.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "x",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: x.layout(),
        });
    }

    if indices.layout() != LayoutId::CONTIGUOUS && indices.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "indices",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: indices.layout(),
        });
    }

    if let Some(d) = dest {
        if d.layout() != LayoutId::CONTIGUOUS && d.layout() != LayoutId::L0 {
            problems.push(T0Error::LayoutMismatch {
                tensor: "dest",
                expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
                got: d.layout(),
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

    if !matches!(x.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: x.dtype(),
        });
    }

    if indices.dtype() != DType::U32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "indices",
            expected: vec![DType::U32],
            got: indices.dtype(),
        });
    }

    if y.dtype() != x.dtype() {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![x.dtype()],
            got: y.dtype(),
        });
    }

    if let Some(d) = dest {
        if d.dtype() != x.dtype() {
            problems.push(T0Error::DTypeMismatch {
                tensor: "dest",
                expected: vec![x.dtype()],
                got: d.dtype(),
            });
        }
    }

    T0Error::from_problems(problems)?;

    let m = x.shape()[0];
    let d = x.shape()[1];
    let n = y.shape()[0];

    if m == 0 {
        return Err(T0Error::EmptyInput {
            op: "scatter_add_rows",
            tensor: "x",
        });
    }
    if n == 0 || d == 0 {
        return Err(T0Error::EmptyInput {
            op: "scatter_add_rows",
            tensor: "y",
        });
    }

    let _m_u32 = u32::try_from(m).map_err(|_| T0Error::ArithmeticOverflow {
        op: "scatter_add_rows",
        detail: format!("dimension M exceeds u32: {m}"),
    })?;
    let _d_u32 = u32::try_from(d).map_err(|_| T0Error::ArithmeticOverflow {
        op: "scatter_add_rows",
        detail: format!("dimension D exceeds u32: {d}"),
    })?;
    let _n_u32 = u32::try_from(n).map_err(|_| T0Error::ArithmeticOverflow {
        op: "scatter_add_rows",
        detail: format!("dimension N exceeds u32: {n}"),
    })?;

    let mut problems = Vec::new();

    if indices.shape()[0] != m {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "M",
            expected_from: "x",
            expected: m,
            tensor: "indices",
            got: indices.shape()[0],
        });
    }

    if y.shape()[1] != d {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "D",
            expected_from: "x",
            expected: d,
            tensor: "y",
            got: y.shape()[1],
        });
    }

    if let Some(dest_view) = dest {
        if dest_view.shape()[0] != n {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "N",
                expected_from: "y",
                expected: n,
                tensor: "dest",
                got: dest_view.shape()[0],
            });
        }
        if dest_view.shape()[1] != d {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "D",
                expected_from: "x",
                expected: d,
                tensor: "dest",
                got: dest_view.shape()[1],
            });
        }
    }

    // Bounds check every index before modifying output. Skipped when the
    // index count already mismatches M (recorded above): reading 0..m
    // would panic on the short buffer instead of returning the typed
    // DimensionMismatch (A1.10 illegal fuzz).
    if indices.shape()[0] == m {
        for pos in 0..m {
            let idx = indices.read_u32(pos);
            if idx as usize >= n {
                problems.push(T0Error::RowIndexOutOfRange {
                    op: "scatter_add_rows",
                    tensor: "indices",
                    position: pos,
                    index: idx,
                    upper_bound: n,
                });
            }
        }
    }

    T0Error::from_problems(problems)?;

    // Initialize y with dest or zeros
    if let Some(dest_view) = dest {
        for row in 0..n {
            for col in 0..d {
                y.write_f32(row * d + col, dest_view.read_f32(row * d + col));
            }
        }
    } else {
        for row in 0..n {
            for col in 0..d {
                y.write_f32(row * d + col, 0.0f32);
            }
        }
    }

    // Flat M-sized stable ordering (avoids allocating Vec<Vec<_>> proportional to N)
    // Stable sort preserves ascending source index order for duplicate/tied indices.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by_key(|&src_m| indices.read_u32(src_m));

    let mut i = 0;
    while i < m {
        let dest_row = indices.read_u32(order[i]) as usize;
        let mut j = i + 1;
        while j < m && indices.read_u32(order[j]) as usize == dest_row {
            j += 1;
        }
        let sources = &order[i..j];

        for col in 0..d {
            let mut acc = y.read_f32(dest_row * d + col);
            for &src_m in sources {
                acc += x.read_f32(src_m * d + col);
            }
            y.write_f32(dest_row * d + col, acc);
        }

        i = j;
    }

    Ok(())
}

/// 64-bit reference implementation of `scatter_add_rows` for testing (Spec 1 §4.A, §6.1).
pub fn scatter_add_rows_f64_reference(
    x: &[f64],
    m: usize,
    d: usize,
    indices: &[u32],
    dest: Option<&[f64]>,
    n: usize,
) -> Vec<f64> {
    assert_eq!(x.len(), m * d);
    assert_eq!(indices.len(), m);
    if let Some(dst) = dest {
        assert_eq!(dst.len(), n * d);
    }

    let mut y = if let Some(dst) = dest {
        dst.to_vec()
    } else {
        vec![0.0f64; n * d]
    };

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by_key(|&src_m| indices[src_m]);

    let mut i = 0;
    while i < m {
        let dest_row = indices[order[i]] as usize;
        assert!(dest_row < n, "index {dest_row} out of bounds 0..{n}");
        let mut j = i + 1;
        while j < m && (indices[order[j]] as usize) == dest_row {
            j += 1;
        }

        let sources = &order[i..j];
        for col in 0..d {
            let mut acc = y[dest_row * d + col];
            for &src_m in sources {
                acc += x[src_m * d + col];
            }
            y[dest_row * d + col] = acc;
        }

        i = j;
    }

    y
}
