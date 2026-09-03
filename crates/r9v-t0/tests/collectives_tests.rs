// SPDX-License-Identifier: Apache-2.0
//! Tests for scalar T0 rank-1 collectives (Spec 1 §4.G, Card A1.9).

use r9v_common::SeededRng;
use r9v_ir::{
    AllGatherOp, AllReduceOp, AllToAllOp, BarrierOp, DType, GroupId, Op, RecvOp, ReduceOp,
    ReduceScatterOp, SendOp,
};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::error::T0Error;
use r9v_t0::{
    all_gather, all_reduce, all_to_all, barrier, execute_collective_op, recv, reduce_scatter, send,
};

fn group() -> GroupId {
    GroupId::new(0)
}

fn next_f32(rng: &mut SeededRng, lo: f32, hi: f32) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
    lo + (u as f32) * (hi - lo)
}

#[test]
fn all_reduce_rank1_is_bit_exact_identity() {
    let mut rng = SeededRng::new(0xC011);
    // Include values that would NOT survive an f32 round-trip as other types.
    let vals: Vec<f32> = (0..24).map(|_| next_f32(&mut rng, -10.0, 10.0)).collect();
    let op = AllReduceOp {
        group: group(),
        op: ReduceOp::Sum,
        dtype: DType::F32,
        reduce_in: DType::F32,
    };
    let x_buf = TypedBuffer::from_f32(&[4, 6], &vals);
    let mut y_buf = TypedBuffer::zeros(&[4, 6], DType::F32);
    all_reduce(&op, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap();
    assert_eq!(y_buf.to_f32_vec(), vals);

    // U32 with values above 2^24: only a byte-exact copy survives.
    let big = vec![16_777_217u32, u32::MAX, 0, 1, 123_456_789, 4_000_000_011];
    let op_u = AllReduceOp {
        group: group(),
        op: ReduceOp::Sum,
        dtype: DType::U32,
        reduce_in: DType::F32,
    };
    let xu = TypedBuffer::from_u32(&[2, 3], &big);
    let mut yu = TypedBuffer::zeros(&[2, 3], DType::U32);
    all_reduce(&op_u, &xu.as_view(), &mut yu.as_view_mut()).unwrap();
    assert_eq!(yu.to_u32_vec(), big);
}

#[test]
fn gather_scatter_alltoall_rank1_copy_inputs() {
    let mut rng = SeededRng::new(0x6A7);
    let vals: Vec<f32> = (0..12).map(|_| next_f32(&mut rng, -5.0, 5.0)).collect();
    let x_buf = TypedBuffer::from_f32(&[3, 4], &vals);

    let ag = AllGatherOp {
        group: group(),
        axis: 0,
        dtype: DType::F32,
    };
    let mut y = TypedBuffer::zeros(&[3, 4], DType::F32);
    all_gather(&ag, &x_buf.as_view(), &mut y.as_view_mut()).unwrap();
    assert_eq!(y.to_f32_vec(), vals);

    let rs = ReduceScatterOp {
        group: group(),
        axis: 1,
        op: ReduceOp::Sum,
        dtype: DType::F32,
        reduce_in: DType::F32,
    };
    let mut y2 = TypedBuffer::zeros(&[3, 4], DType::F32);
    reduce_scatter(&rs, &x_buf.as_view(), &mut y2.as_view_mut()).unwrap();
    assert_eq!(y2.to_f32_vec(), vals);

    let a2a = AllToAllOp {
        group: group(),
        dtype: DType::F32,
    };
    let counts = TypedBuffer::from_u32(&[1], &[3]);
    let mut y3 = TypedBuffer::zeros(&[3, 4], DType::F32);
    all_to_all(
        &a2a,
        &x_buf.as_view(),
        &counts.as_view(),
        &mut y3.as_view_mut(),
    )
    .unwrap();
    assert_eq!(y3.to_f32_vec(), vals);
}

#[test]
fn bad_axis_counts_and_peers_rejected() {
    let x_buf = TypedBuffer::from_f32(&[3, 4], &[1.0; 12]);
    let mut y_buf = TypedBuffer::zeros(&[3, 4], DType::F32);

    let ag = AllGatherOp {
        group: group(),
        axis: 2,
        dtype: DType::F32,
    };
    let err = all_gather(&ag, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap_err();
    assert!(matches!(
        err,
        T0Error::InvalidAttribute {
            op: "all_gather",
            attribute: "axis",
            ..
        }
    ));

    let rs = ReduceScatterOp {
        group: group(),
        axis: 9,
        op: ReduceOp::Sum,
        dtype: DType::F32,
        reduce_in: DType::F32,
    };
    let err = reduce_scatter(&rs, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap_err();
    assert!(matches!(
        err,
        T0Error::InvalidAttribute {
            op: "reduce_scatter",
            ..
        }
    ));

    let a2a = AllToAllOp {
        group: group(),
        dtype: DType::F32,
    };
    let bad_counts = TypedBuffer::from_u32(&[1], &[2]);
    let err = all_to_all(
        &a2a,
        &x_buf.as_view(),
        &bad_counts.as_view(),
        &mut y_buf.as_view_mut(),
    )
    .unwrap_err();
    assert!(matches!(err, T0Error::DimensionMismatch { .. }));

    let send_bad = SendOp {
        group: group(),
        peer: 1,
        dtype: DType::F32,
    };
    let err = send(&send_bad, &x_buf.as_view()).unwrap_err();
    assert!(matches!(err, T0Error::InvalidAttribute { op: "send", .. }));
}

#[test]
fn send_peer0_ok_barrier_ok_recv_fails_closed() {
    let x_buf = TypedBuffer::from_f32(&[2, 2], &[1.0; 4]);
    let send_ok = SendOp {
        group: group(),
        peer: 0,
        dtype: DType::F32,
    };
    assert!(send(&send_ok, &x_buf.as_view()).is_ok());
    assert!(barrier(&BarrierOp { group: group() }).is_ok());

    let recv_op = RecvOp {
        group: group(),
        peer: 0,
        shape: vec![r9v_ir::Dim::Concrete(2), r9v_ir::Dim::Concrete(2)].into_boxed_slice(),
        dtype: DType::F32,
    };
    let mut y_buf = TypedBuffer::zeros(&[2, 2], DType::F32);
    let err = recv(&recv_op, &mut y_buf.as_view_mut()).unwrap_err();
    assert!(matches!(err, T0Error::InvalidAttribute { op: "recv", .. }));
    assert_eq!(y_buf.to_f32_vec(), vec![0.0; 4]);
}

#[test]
fn dtype_mismatches_report_op_tensors() {
    let x_buf = TypedBuffer::from_f32(&[2, 2], &[1.0; 4]);
    let mut y_buf = TypedBuffer::zeros(&[2, 2], DType::F32);
    let op = AllReduceOp {
        group: group(),
        op: ReduceOp::Sum,
        dtype: DType::F16,
        reduce_in: DType::F32,
    };
    let err = all_reduce(&op, &x_buf.as_view(), &mut y_buf.as_view_mut()).unwrap_err();
    assert!(matches!(err, T0Error::Multiple { .. }));
}

#[test]
fn dispatch_enforces_collective_arity() {
    let x_buf = TypedBuffer::from_f32(&[2, 2], &[1.0; 4]);
    let counts = TypedBuffer::from_u32(&[1], &[2]);
    let mut y_buf = TypedBuffer::zeros(&[2, 2], DType::F32);

    let ar = Op::AllReduce(AllReduceOp {
        group: group(),
        op: ReduceOp::Sum,
        dtype: DType::F32,
        reduce_in: DType::F32,
    });
    assert!(execute_collective_op(&ar, &[], &mut [y_buf.as_view_mut()]).is_err());
    assert!(execute_collective_op(&ar, &[x_buf.as_view()], &mut [y_buf.as_view_mut()]).is_ok());

    let a2a = Op::AllToAll(AllToAllOp {
        group: group(),
        dtype: DType::F32,
    });
    assert!(execute_collective_op(&a2a, &[x_buf.as_view()], &mut [y_buf.as_view_mut()]).is_err());
    assert!(execute_collective_op(
        &a2a,
        &[x_buf.as_view(), counts.as_view()],
        &mut [y_buf.as_view_mut()]
    )
    .is_ok());

    let send_op = Op::Send(SendOp {
        group: group(),
        peer: 0,
        dtype: DType::F32,
    });
    assert!(execute_collective_op(&send_op, &[x_buf.as_view()], &mut []).is_ok());
    assert!(
        execute_collective_op(&send_op, &[x_buf.as_view()], &mut [y_buf.as_view_mut()]).is_err()
    );

    let barrier_op = Op::Barrier(BarrierOp { group: group() });
    assert!(execute_collective_op(&barrier_op, &[], &mut []).is_ok());
    assert!(execute_collective_op(&barrier_op, &[x_buf.as_view()], &mut []).is_err());

    let recv_op = Op::Recv(RecvOp {
        group: group(),
        peer: 0,
        shape: vec![r9v_ir::Dim::Concrete(2), r9v_ir::Dim::Concrete(2)].into_boxed_slice(),
        dtype: DType::F32,
    });
    assert!(execute_collective_op(&recv_op, &[], &mut [y_buf.as_view_mut()]).is_err());
}
