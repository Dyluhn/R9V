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

#[test]
fn test_execute_elementwise_op_exact_arity_enforcement() {
    use r9v_ir::{
        ActMulOp, ActivationKind, ActivationOp, CastOp, CopyKind, CopyOp, DType, NormAxis,
        NormKind, NormOp, Op, QuantActOp, QuantScheme, ResidualAddOp, RopeOp, RopeScaling,
        RopeStyle, Smoothing,
    };

    let x = TypedBuffer::zeros(&[2, 4], DType::F32);
    let mut y = TypedBuffer::zeros(&[2, 4], DType::F32);
    let mut y2 = TypedBuffer::zeros(&[2, 4], DType::F32);
    let mut y3 = TypedBuffer::zeros(&[2, 4], DType::F32);

    // Norm: requires 2 or 3 inputs and 1 output
    let norm_op = Op::Norm(NormOp {
        kind: NormKind::Rms,
        axis: NormAxis::Last,
        eps: 1e-5,
        weight_offset: 0.0,
        out_dtype: DType::F32,
    });
    assert!(execute_elementwise_op(&norm_op, &[], &mut [y.as_view_mut()]).is_err());
    assert!(execute_elementwise_op(&norm_op, &[x.as_view()], &mut [y.as_view_mut()]).is_err());
    assert!(execute_elementwise_op(
        &norm_op,
        &[x.as_view(), x.as_view(), x.as_view(), x.as_view()],
        &mut [y.as_view_mut()]
    )
    .is_err());
    assert!(execute_elementwise_op(
        &norm_op,
        &[x.as_view(), x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut()]
    )
    .is_err());

    // ResidualAdd: requires 2 inputs and 1 output
    let res_op = Op::ResidualAdd(ResidualAddOp {
        out_dtype: DType::F32,
    });
    assert!(execute_elementwise_op(&res_op, &[x.as_view()], &mut [y.as_view_mut()]).is_err());
    assert!(execute_elementwise_op(
        &res_op,
        &[x.as_view(), x.as_view(), x.as_view()],
        &mut [y.as_view_mut()]
    )
    .is_err());
    assert!(execute_elementwise_op(
        &res_op,
        &[x.as_view(), x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut()]
    )
    .is_err());

    // ActMul: requires 2 inputs and 1 output
    let act_mul_op = Op::ActMul(ActMulOp {
        act: ActivationKind::Silu,
        clamp: None,
    });
    assert!(execute_elementwise_op(&act_mul_op, &[x.as_view()], &mut [y.as_view_mut()]).is_err());
    assert!(execute_elementwise_op(
        &act_mul_op,
        &[x.as_view(), x.as_view(), x.as_view()],
        &mut [y.as_view_mut()]
    )
    .is_err());
    assert!(execute_elementwise_op(
        &act_mul_op,
        &[x.as_view(), x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut()]
    )
    .is_err());

    // Activation: requires 1 input and 1 output
    let act_op = Op::Activation(ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    });
    assert!(execute_elementwise_op(&act_op, &[], &mut [y.as_view_mut()]).is_err());
    assert!(
        execute_elementwise_op(&act_op, &[x.as_view(), x.as_view()], &mut [y.as_view_mut()])
            .is_err()
    );
    assert!(execute_elementwise_op(
        &act_op,
        &[x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut()]
    )
    .is_err());

    // Rope: requires 2 inputs and 1 output
    let rope_op = Op::Rope(RopeOp {
        rot_dim: 4,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F32,
    });
    assert!(execute_elementwise_op(&rope_op, &[x.as_view()], &mut [y.as_view_mut()]).is_err());
    assert!(execute_elementwise_op(
        &rope_op,
        &[x.as_view(), x.as_view(), x.as_view()],
        &mut [y.as_view_mut()]
    )
    .is_err());
    assert!(execute_elementwise_op(
        &rope_op,
        &[x.as_view(), x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut()]
    )
    .is_err());

    // Cast: requires 1 input and 1 output
    let cast_op = Op::Cast(CastOp { dtype: DType::F32 });
    assert!(execute_elementwise_op(&cast_op, &[], &mut [y.as_view_mut()]).is_err());
    assert!(execute_elementwise_op(
        &cast_op,
        &[x.as_view(), x.as_view()],
        &mut [y.as_view_mut()]
    )
    .is_err());
    assert!(execute_elementwise_op(
        &cast_op,
        &[x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut()]
    )
    .is_err());

    // Copy: requires 1 input and 1 output
    let copy_op = Op::Copy(CopyOp {
        kind: CopyKind::DeviceToDevice,
    });
    assert!(execute_elementwise_op(&copy_op, &[], &mut [y.as_view_mut()]).is_err());
    assert!(execute_elementwise_op(
        &copy_op,
        &[x.as_view(), x.as_view()],
        &mut [y.as_view_mut()]
    )
    .is_err());
    assert!(execute_elementwise_op(
        &copy_op,
        &[x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut()]
    )
    .is_err());

    // QuantAct: requires 1 input and 2 outputs
    let quant_op = Op::QuantAct(QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    });
    assert!(
        execute_elementwise_op(&quant_op, &[], &mut [y.as_view_mut(), y2.as_view_mut()]).is_err()
    );
    assert!(execute_elementwise_op(
        &quant_op,
        &[x.as_view(), x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut()]
    )
    .is_err());
    assert!(execute_elementwise_op(&quant_op, &[x.as_view()], &mut [y.as_view_mut()]).is_err());
    assert!(execute_elementwise_op(
        &quant_op,
        &[x.as_view()],
        &mut [y.as_view_mut(), y2.as_view_mut(), y3.as_view_mut()]
    )
    .is_err());
}

#[test]
fn test_backing_length_validation_rejects_undersized_and_overflowed_storage() {
    use r9v_ir::{ActivationKind, ActivationOp, DType};

    let op = ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    };

    // Undersized backing slice: shape claims [2, 4] (8 elements) but slice has only 4 elements
    let short_data = [1.0f32, 2.0, 3.0, 4.0];
    let view = TensorView::from_f32_slice(&[2, 4], &short_data);
    let mut y = TypedBuffer::zeros(&[2, 4], DType::F32);

    let err = activation(&op, &view, &mut y.as_view_mut()).unwrap_err();
    match err {
        T0Error::BufferLengthMismatch {
            tensor,
            buffer_len,
            expected_len,
            shape,
        } => {
            assert_eq!(tensor, "x");
            assert_eq!(buffer_len, 4);
            assert_eq!(expected_len, 8);
            assert_eq!(shape, vec![2, 4]);
        }
        other => panic!("expected BufferLengthMismatch, got {other:?}"),
    }

    // Overflowed shape elements: shape dimensions whose product overflows usize::MAX
    let overflow_view = TensorView::from_f32_slice(&[usize::MAX, 2], &[]);
    let overflow_err = overflow_view.validate_backing("overflow").unwrap_err();
    match overflow_err {
        T0Error::BufferLengthMismatch { tensor, .. } => {
            assert_eq!(tensor, "overflow");
        }
        other => panic!("expected BufferLengthMismatch, got {other:?}"),
    }
}
