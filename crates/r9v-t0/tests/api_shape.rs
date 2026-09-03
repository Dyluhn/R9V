// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! API shape verification for r9v-t0 crate (CONVENTIONS.md §3, Cards A1.5 and A1.8).

use r9v_t0::*;
use r9v_ir::{SamplingParams, VerifyMethod};

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

#[test]
fn test_type_markers_and_traits() {
    assert_send::<RngState>();
    assert_sync::<RngState>();

    assert_send::<VerifyOutput>();
    assert_sync::<VerifyOutput>();

    assert_send::<T0Error>();
    assert_sync::<T0Error>();

    // Check trait implementations
    let rng = RngState::new(42, 1, 0);
    let rng_clone = rng.clone();
    assert_eq!(rng, rng_clone);
    assert_eq!(rng.seed(), 42);
    assert_eq!(rng.seq_id(), 1);
    assert_eq!(rng.step(), 0);
    assert_eq!(rng.draw_index(), 0);

    let output = VerifyOutput {
        accepted: vec![1, 2, 3],
        accept_len: vec![2],
    };
    let output_clone = output.clone();
    assert_eq!(output, output_clone);

    let err = T0Error::EmptyInput {
        op: "sample",
        tensor: "probs",
    };
    let display_str = format!("{err}");
    assert!(display_str.contains("empty tensor in sample: probs"));
}

#[test]
fn test_public_function_signatures() {
    // Verify logits_postprocess signature compiles and is accessible
    let logits = vec![1.0, 2.0];
    let params = vec![SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: vec![],
    }];
    let mut probs = vec![0.0; 2];
    let res = logits_postprocess(&logits, 1, 1, 2, &params, None, None, &mut probs);
    assert!(res.is_ok());

    // Verify sample signature
    let mut rng_states = vec![RngState::new(1, 0, 0)];
    let res_sample = sample(&probs, 1, 2, &mut rng_states);
    assert!(res_sample.is_ok());

    // Verify verify signature
    let draft_tokens = vec![1];
    let target_probs = vec![0.1, 0.9, 0.8, 0.2];
    let res_verify = verify(
        &draft_tokens,
        None,
        &target_probs,
        1,
        1,
        2,
        &VerifyMethod::Greedy,
        &mut rng_states,
        None,
    );
    assert!(res_verify.is_ok());

    // Verify low-level Philox primitives
    let words = philox4x32_10([0, 0, 0, 0], [0, 0]);
    assert_eq!(words.len(), 4);
    let u = u32_to_unit_f32(words[0]);
    assert!(u > 0.0 && u < 1.0);
}
