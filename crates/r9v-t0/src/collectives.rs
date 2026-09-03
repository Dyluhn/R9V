// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Scalar T0 rank-1 collective implementations (Spec 1 §4.G, Card A1.9).
//!
//! At `ranks = 1` every data-movement collective is a bit-exact identity
//! transfer (SI-54): `all_reduce(Sum)`, `all_gather`, `reduce_scatter`, and
//! `all_to_all` copy `x` to `y` with no `f32` round-trip by delegating to the
//! [`copy()`] core. `send` to peer 0 is a no-op, `barrier` is a
//! no-op, and `recv` fails closed (no data source exists in T0 at rank 1).
//!
//! DECISION(A1.9): rank-1 data collectives are bit-exact identities through
//! the `copy` core; rejected `f32` round-trip copies and zero-filled `recv`.
//! Per SI-54.

use r9v_ir::{
    AllGatherOp, AllReduceOp, AllToAllOp, BarrierOp, CopyKind, CopyOp, DType, LayoutId, RecvOp,
    ReduceOp, ReduceScatterOp, SendOp,
};

use crate::buffer::{TensorView, TensorViewMut};
use crate::copy::copy;
use crate::error::T0Error;

/// Bit-exact identity transfer shared by the rank-1 data collectives.
///
/// Validates that both views share `dtype` and delegates the bytes to the
/// [`copy()`] core, which transfers storage without floating-point
/// conversion.
fn identity_transfer(
    _op: &'static str,
    x: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    y.validate_backing("y")?;
    let mut problems = Vec::new();
    if y.dtype() != x.dtype() {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![x.dtype()],
            got: y.dtype(),
        });
    }
    T0Error::from_problems(problems)?;
    // The `copy` core reports shape/dtype/backing problems with the same
    // `x`/`y` tensor names, so its typed errors propagate unchanged.
    copy(
        &CopyOp {
            kind: CopyKind::DeviceToDevice,
        },
        x,
        y,
    )
}

/// Executes scalar T0 rank-1 `all_reduce` (Spec 1 §4.G, Card A1.9).
///
/// `y = x` bit-exact copy; `reduce_in` must still be `f32` per spec (SI-54
/// records the bit-exact identity choice). Multi-rank ascending-rank `f32`
/// reduction is executor scope, not T0 scope.
pub fn all_reduce(
    op: &AllReduceOp,
    x: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();
    match op.op {
        ReduceOp::Sum => {}
    }
    if op.reduce_in != DType::F32 {
        problems.push(T0Error::InvalidAttribute {
            op: "all_reduce",
            attribute: "reduce_in",
            reason: format!("must be f32 per Spec 1 §4.G, got {:?}", op.reduce_in),
        });
    }
    if x.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![op.dtype],
            got: x.dtype(),
        });
    }
    if y.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.dtype],
            got: y.dtype(),
        });
    }
    T0Error::from_problems(problems)?;
    identity_transfer("all_reduce", x, y)
}

/// Executes scalar T0 rank-1 `all_gather` (Spec 1 §4.G, Card A1.9).
///
/// `y = x` bit-exact copy; `axis` must address a real axis of `x`. At rank 1
/// the gathered extent equals the input extent on every axis.
pub fn all_gather(
    op: &AllGatherOp,
    x: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();
    if x.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![op.dtype],
            got: x.dtype(),
        });
    }
    if y.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.dtype],
            got: y.dtype(),
        });
    }
    if op.axis as usize >= x.rank() {
        problems.push(T0Error::InvalidAttribute {
            op: "all_gather",
            attribute: "axis",
            reason: format!("axis {} out of bounds for rank {}", op.axis, x.rank()),
        });
    }
    T0Error::from_problems(problems)?;
    identity_transfer("all_gather", x, y)
}

/// Executes scalar T0 rank-1 `reduce_scatter` (Spec 1 §4.G, Card A1.9).
///
/// `y = x` bit-exact copy; `axis` must address a real axis of `x` and
/// `reduce_in` must be `f32` per spec.
pub fn reduce_scatter(
    op: &ReduceScatterOp,
    x: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();
    match op.op {
        ReduceOp::Sum => {}
    }
    if op.reduce_in != DType::F32 {
        problems.push(T0Error::InvalidAttribute {
            op: "reduce_scatter",
            attribute: "reduce_in",
            reason: format!("must be f32 per Spec 1 §4.G, got {:?}", op.reduce_in),
        });
    }
    if x.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![op.dtype],
            got: x.dtype(),
        });
    }
    if y.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.dtype],
            got: y.dtype(),
        });
    }
    if op.axis as usize >= x.rank() {
        problems.push(T0Error::InvalidAttribute {
            op: "reduce_scatter",
            attribute: "axis",
            reason: format!("axis {} out of bounds for rank {}", op.axis, x.rank()),
        });
    }
    T0Error::from_problems(problems)?;
    identity_transfer("reduce_scatter", x, y)
}

/// Executes scalar T0 rank-1 `all_to_all` (Spec 1 §4.G, Card A1.9).
///
/// Validates `counts [1] u32` with `counts[0] == x.shape[0]`, then `y = x`
/// bit-exact copy. Variable per-peer counts for EP are resolved in the
/// pre-step phase (spec 1 §4.G); at rank 1 the single count covers all rows.
pub fn all_to_all(
    op: &AllToAllOp,
    x: &TensorView<'_>,
    counts: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    counts.validate_backing("counts")?;
    y.validate_backing("y")?;

    let mut problems = Vec::new();
    if x.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![op.dtype],
            got: x.dtype(),
        });
    }
    if y.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.dtype],
            got: y.dtype(),
        });
    }
    if counts.rank() != 1 {
        problems.push(T0Error::RankMismatch {
            tensor: "counts",
            expected: 1,
            got: counts.rank(),
            shape: counts.shape().to_vec(),
        });
    }
    if counts.dtype() != DType::U32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "counts",
            expected: vec![DType::U32],
            got: counts.dtype(),
        });
    }
    if x.rank() == 0 {
        problems.push(T0Error::RankMismatch {
            tensor: "x",
            expected: 1,
            got: 0,
            shape: x.shape().to_vec(),
        });
    }
    T0Error::from_problems(problems)?;

    if counts.shape()[0] != 1 {
        return Err(T0Error::DimensionMismatch {
            dim_name: "P",
            expected_from: "ranks_1",
            expected: 1,
            tensor: "counts",
            got: counts.shape()[0],
        });
    }
    let first = counts
        .try_read_u32(0, "counts")
        .map_err(|_| T0Error::BufferLengthMismatch {
            tensor: "counts",
            buffer_len: counts.backing_len(),
            expected_len: 1,
            shape: counts.shape().to_vec(),
        })?;
    if first as usize != x.shape()[0] {
        return Err(T0Error::DimensionMismatch {
            dim_name: "rows",
            expected_from: "x",
            expected: x.shape()[0],
            tensor: "counts",
            got: first as usize,
        });
    }
    identity_transfer("all_to_all", x, y)
}

/// Executes scalar T0 rank-1 `send` (Spec 1 §4.G, Card A1.9).
///
/// Validates the input backing and dtype; `peer == 0` is a no-op `Ok`, any
/// other peer fails closed (no transport exists in T0 at rank 1).
pub fn send(op: &SendOp, x: &TensorView<'_>) -> Result<(), T0Error> {
    x.validate_backing("x")?;
    let mut problems = Vec::new();
    if x.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "x",
            expected: vec![op.dtype],
            got: x.dtype(),
        });
    }
    if x.layout() != LayoutId::CONTIGUOUS && x.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "x",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: x.layout(),
        });
    }
    if op.peer != 0 {
        problems.push(T0Error::InvalidAttribute {
            op: "send",
            attribute: "peer",
            reason: format!("rank-1 send supports peer 0 only, got peer {}", op.peer),
        });
    }
    T0Error::from_problems(problems)?;
    Ok(())
}

/// Fails closed scalar T0 rank-1 `recv` (Spec 1 §4.G, Card A1.9, SI-54).
///
/// No data source exists in T0 at rank 1, so `recv` always returns a typed
/// error after validating the output descriptor. The error carries every
/// descriptor problem plus the fail-closed refusal.
pub fn recv(op: &RecvOp, y: &mut TensorViewMut<'_>) -> Result<(), T0Error> {
    y.validate_backing("y")?;
    let mut problems = Vec::new();
    if y.dtype() != op.dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.dtype],
            got: y.dtype(),
        });
    }
    problems.push(T0Error::InvalidAttribute {
        op: "recv",
        attribute: "peer",
        reason: format!(
            "rank-1 recv has no data source in T0 (peer {}); SI-54",
            op.peer
        ),
    });
    T0Error::from_problems(problems)?;
    Ok(())
}

/// Executes scalar T0 `barrier` (Spec 1 §4.G, Card A1.9).
///
/// No-op `Ok`: a single rank has nothing to synchronize.
pub fn barrier(_op: &BarrierOp) -> Result<(), T0Error> {
    Ok(())
}
