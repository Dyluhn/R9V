// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! API shape verification for r9v-t0 crate (CONVENTIONS.md §3).

use r9v_t0::*;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn test_trait_bounds_and_api_shapes() {
    assert_send::<T0Error>();
    assert_sync::<T0Error>();

    assert_send::<Tolerance>();
    assert_sync::<Tolerance>();

    assert_send::<TypedBuffer>();
    assert_sync::<TypedBuffer>();

    assert_send::<TensorView<'_>>();
    assert_sync::<TensorView<'_>>();

    assert_send::<TensorViewMut<'_>>();
    assert_sync::<TensorViewMut<'_>>();
}

#[test]
fn test_execute_elementwise_op_dispatch() {
    use r9v_ir::{ActivationKind, ActivationOp, DType, Op};

    let op = Op::Activation(ActivationOp {
        act: ActivationKind::Identity,
        clamp: None,
    });

    let x = TypedBuffer::from_f32(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut y = TypedBuffer::zeros(&[2, 4], DType::F32);

    let inputs = [x.as_view()];
    let mut outputs = [y.as_view_mut()];

    execute_elementwise_op(&op, &inputs, &mut outputs).unwrap();

    assert_eq!(y.to_f32_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}
