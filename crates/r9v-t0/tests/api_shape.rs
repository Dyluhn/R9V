// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! API shape verification for r9v-t0 crate (CONVENTIONS.md §3, Cards A1.5 and A1.8).

use r9v_common::{SeqId, StepId};
use r9v_ir::{DType, SamplingParams, VerifyMethod};
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

    assert_send::<KvPagedCache>();
    assert_sync::<KvPagedCache>();

    assert_send::<KvLatentCache>();
    assert_sync::<KvLatentCache>();

    assert_send::<KvCache>();
    assert_sync::<KvCache>();
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
        scale: 1.0,
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
    let rng = RngState::from_u64(42, 1, 0).unwrap();
    let rng_clone = rng.clone();
    assert_eq!(rng, rng_clone);
    assert_eq!(rng.seed(), 42);
    assert_eq!(rng.seq_id(), SeqId::new(1));
    assert_eq!(rng.step(), StepId::new(0));
    assert_eq!(rng.draw_index(), 0);
    assert_eq!(rng.seq_id_u32(), 1);

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
    let mut rng_states = vec![RngState::from_u64(1, 0, 0).unwrap()];
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

    // Verify A1.6 op signatures
    let x_mat = TypedBuffer::from_f16(&[2, 4], &[r9v_t0::dtype::f32_to_f16(1.0); 8]);
    let x = TypedBuffer::from_f32(&[2, 4], &[1.0; 8]);
    let w_bytes: Vec<u8> = vec![0u8; 4 * 4 * 2];
    let w = TypedBuffer::from_bytes(&[4, 4], DType::F16, &w_bytes);
    let mut y = TypedBuffer::zeros(&[2, 4], DType::F32);
    let mat_op = r9v_ir::MatmulOp {
        out_dtype: DType::F32,
        epilogue: r9v_ir::Epilogue::None,
        transpose_w: false,
    };
    assert!(matmul(
        &mat_op,
        &x_mat.as_view(),
        &w.as_view(),
        None,
        None,
        &mut y.as_view_mut()
    )
    .is_ok());

    let indices = TypedBuffer::from_u32(&[2], &[0u32, 1]);
    let gather_op = r9v_ir::GatherRowsOp;
    assert!(gather_rows(
        &gather_op,
        &x.as_view(),
        &indices.as_view(),
        &mut y.as_view_mut()
    )
    .is_ok());

    let scatter_op = r9v_ir::ScatterAddRowsOp;
    let mut y_scatter = TypedBuffer::zeros(&[2, 4], DType::F32);
    assert!(scatter_add_rows(
        &scatter_op,
        &x.as_view(),
        &indices.as_view(),
        None,
        &mut y_scatter.as_view_mut()
    )
    .is_ok());

    let embed_op = r9v_ir::EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F32,
    };
    let table = TypedBuffer::from_bytes(&[4, 4], DType::F16, &w_bytes);
    let mut y_embed = TypedBuffer::zeros(&[2, 4], DType::F32);
    assert!(embed_gather(
        &embed_op,
        &indices.as_view(),
        &table.as_view(),
        &mut y_embed.as_view_mut()
    )
    .is_ok());
}

#[test]
fn test_execute_matmul_and_lookup_op_dispatch() {
    use r9v_ir::{DType, EmbedGatherOp, Epilogue, GatherRowsOp, MatmulOp, Op, ScatterAddRowsOp};

    let x_mat = TypedBuffer::from_f16(&[2, 4], &[r9v_t0::dtype::f32_to_f16(1.0); 8]);
    let x = TypedBuffer::from_f32(&[2, 4], &[1.0; 8]);
    let w_bytes: Vec<u8> = vec![0u8; 4 * 4 * 2];
    let w = TypedBuffer::from_bytes(&[4, 4], DType::F16, &w_bytes);
    let mut y = TypedBuffer::zeros(&[2, 4], DType::F32);

    // execute_matmul_op
    let mat_op = MatmulOp {
        out_dtype: DType::F32,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    let inputs = [x_mat.as_view(), w.as_view()];
    let mut outputs = [y.as_view_mut()];
    assert!(execute_matmul_op(&mat_op, &inputs, &mut outputs).is_ok());

    // Arity mismatch for execute_matmul_op
    assert!(execute_matmul_op(&mat_op, &[], &mut outputs).is_err());

    // execute_lookup_op: GatherRows
    let indices = TypedBuffer::from_u32(&[2], &[0u32, 1]);
    let gather_op = Op::GatherRows(GatherRowsOp);
    let gather_inputs = [x.as_view(), indices.as_view()];
    let mut gather_outputs = [y.as_view_mut()];
    assert!(execute_lookup_op(&gather_op, &gather_inputs, &mut gather_outputs).is_ok());

    // execute_lookup_op: EmbedGather
    let embed_op = Op::EmbedGather(EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F32,
    });
    let table = TypedBuffer::from_bytes(&[4, 4], DType::F16, &w_bytes);
    let embed_inputs = [indices.as_view(), table.as_view()];
    let mut embed_outputs = [y.as_view_mut()];
    assert!(execute_lookup_op(&embed_op, &embed_inputs, &mut embed_outputs).is_ok());

    // execute_lookup_op: ScatterAddRows
    let scatter_op = Op::ScatterAddRows(ScatterAddRowsOp);
    let scatter_inputs = [x.as_view(), indices.as_view()];
    let mut scatter_outputs = [y.as_view_mut()];
    assert!(execute_lookup_op(&scatter_op, &scatter_inputs, &mut scatter_outputs).is_ok());
}

#[test]
fn test_a19_public_surface_markers_and_dispatch() {
    use r9v_ir::{
        AllReduceOp, BarrierOp, CausalConv1dOp, ConvActivation, LinearAttnKind, LinearAttnScanOp,
        MoeFfnOp, MoeRouteOp, MoeScoring, NgramCombine, NgramGatherOp, NgramSource, Op, ReduceOp,
        StateHandle, StateKind,
    };

    assert_send::<SeqLayout>();
    assert_sync::<SeqLayout>();

    // MoE dispatch: route runs through execute_moe_op.
    let route_op = Op::MoeRoute(MoeRouteOp {
        top_k: 1,
        scoring: MoeScoring::Softmax,
        renormalize: false,
        group: None,
        scale: 1.0,
    });
    let logits = TypedBuffer::from_f32(&[2, 3], &[0.1, 0.2, 0.3, 0.3, 0.2, 0.1]);
    let mut ids = TypedBuffer::zeros(&[2, 1], DType::U32);
    let mut weights = TypedBuffer::zeros(&[2, 1], DType::F32);
    assert!(execute_moe_op(
        &route_op,
        &[logits.as_view()],
        &mut [ids.as_view_mut(), weights.as_view_mut()],
    )
    .is_ok());
    // MoE dispatch: ffn arity is enforced here; behavior is covered in moe_ffn_tests.
    let ffn_op = Op::MoeFfn(MoeFfnOp {
        act: r9v_ir::ActivationKind::Silu,
        out_dtype: DType::F32,
        shared_experts: 0,
    });
    let z = TypedBuffer::zeros(&[1, 2], DType::F32);
    let mut y_ffn = TypedBuffer::zeros(&[1, 2], DType::F32);
    assert!(execute_moe_op(&ffn_op, &[z.as_view()], &mut [y_ffn.as_view_mut()]).is_err());

    // State/scan dispatch: conv runs; scan form flag is accepted.
    let conv_op = Op::CausalConv1d(CausalConv1dOp {
        kernel: 2,
        act: ConvActivation::Identity,
        handle: StateHandle::new(0, StateKind::ConvWindow),
    });
    let xc = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
    let wc = TypedBuffer::from_f32(&[2, 2], &[0.5; 4]);
    let sc = TypedBuffer::from_f16(&[1, 1, 2], &[0; 2]);
    let mut yc = TypedBuffer::zeros(&[2, 2], DType::F32);
    let mut soc = TypedBuffer::zeros(&[1, 1, 2], DType::F16);
    let seq = SeqLayout::new(&[2]).unwrap();
    assert!(execute_state_scan_op(
        &conv_op,
        &[xc.as_view(), wc.as_view()],
        &sc.as_view(),
        &seq,
        &mut [yc.as_view_mut()],
        &mut soc.as_view_mut(),
        false,
    )
    .is_ok());
    let scan_op = Op::LinearAttnScan(LinearAttnScanOp {
        kind: LinearAttnKind::GLA,
        chunk: 2,
        out_dtype: DType::F32,
        handle: StateHandle::new(0, StateKind::Recurrent),
    });
    assert!(execute_state_scan_op(
        &scan_op,
        &[xc.as_view()],
        &sc.as_view(),
        &seq,
        &mut [yc.as_view_mut()],
        &mut soc.as_view_mut(),
        true,
    )
    .is_err());

    // N-gram dispatch: staged runs through views.
    let ngram_op = Op::NgramGather(NgramGatherOp {
        source: NgramSource::Staged,
        orders: vec![1u32].into_boxed_slice(),
        heads: 1,
        hash: r9v_ir::HashId::new(0),
        table_sizes: vec![8u32].into_boxed_slice(),
        combine: NgramCombine::Sum,
        out_dtype: DType::F32,
    });
    let staging = TypedBuffer::from_i8(&[2, 1, 4], &[1i8; 8]).with_quant(
        r9v_ir::QuantScheme::Scheme(r9v_format::SchemeId::I8R.to_ir()),
    );
    let scales = TypedBuffer::from_f32(&[2, 1], &[0.5; 2]);
    let mut yn = TypedBuffer::zeros(&[2, 4], DType::F32);
    assert!(execute_ngram_op(
        &ngram_op,
        &[staging.as_view(), scales.as_view()],
        &mut [yn.as_view_mut()]
    )
    .is_ok());

    // Collective dispatch: barrier and send run; recv fails closed.
    let barrier_op = Op::Barrier(BarrierOp {
        group: r9v_ir::GroupId::new(0),
    });
    assert!(execute_collective_op(&barrier_op, &[], &mut []).is_ok());
    let reduce_op = Op::AllReduce(AllReduceOp {
        group: r9v_ir::GroupId::new(0),
        op: ReduceOp::Sum,
        dtype: DType::F32,
        reduce_in: DType::F32,
    });
    let xr = TypedBuffer::from_f32(&[2, 2], &[1.0; 4]);
    let mut yr = TypedBuffer::zeros(&[2, 2], DType::F32);
    assert!(execute_collective_op(&reduce_op, &[xr.as_view()], &mut [yr.as_view_mut()]).is_ok());
}
