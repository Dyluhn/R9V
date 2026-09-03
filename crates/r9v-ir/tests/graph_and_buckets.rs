// SPDX-License-Identifier: Apache-2.0
//! Tests for step-graph keys, bucket functions, and copy insertion on stride mismatch (Spec 1 §3; card A1.2).

use r9v_ir::{
    bucket_s, bucket_step, bucket_t_dec, bucket_t_pre, ActivationKind, ActivationOp, CastOp, Class,
    DType, Dim, EdgeId, ExternalInput, ExternalInputKind, ExternalOutput, ExternalOutputKind,
    Graph, IrError, LayoutId, NodeId, NormAxis, NormKind, NormOp, Op, Placement, PlanId,
    QuantScheme, ShardLayout, StepGraphKey, StrideRequirement, Tensor, BUCKET_SIZES,
};

#[test]
fn bucket_functions_exact_edges_and_rounding() {
    // 1. Exact bucket edges map to themselves
    for &b in &BUCKET_SIZES {
        assert_eq!(bucket_s(b).unwrap(), b);
        assert_eq!(bucket_t_dec(b).unwrap(), b);
        assert_eq!(bucket_t_pre(b).unwrap(), b);
    }

    // 2. Intermediate values round up to next discrete power-of-two bucket
    assert_eq!(bucket_s(3).unwrap(), 4);
    assert_eq!(bucket_s(5).unwrap(), 8);
    assert_eq!(bucket_s(33).unwrap(), 64);
    assert_eq!(bucket_s(1025).unwrap(), 2048);
    assert_eq!(bucket_s(2049).unwrap(), 4096);

    // 3. T_pre = 0 is legal (decode-only step)
    assert_eq!(bucket_t_pre(0).unwrap(), 0);

    // 4. Values exceeding 4096 fail with BucketExceeded
    assert!(matches!(
        bucket_s(4097),
        Err(IrError::BucketExceeded {
            axis: "S",
            value: 4097,
            max: 4096
        })
    ));
    assert!(matches!(
        bucket_t_dec(4097),
        Err(IrError::BucketExceeded {
            axis: "T_dec",
            value: 4097,
            max: 4096
        })
    ));
    assert!(matches!(
        bucket_t_pre(4097),
        Err(IrError::BucketExceeded {
            axis: "T_pre",
            value: 4097,
            max: 4096
        })
    ));

    // 5. Zero S or T_dec report the actual invalid bucket axis/value.
    assert!(matches!(
        bucket_s(0),
        Err(IrError::InvalidBucket {
            axis: "S",
            value: 0
        })
    ));
    assert!(matches!(
        bucket_t_dec(0),
        Err(IrError::InvalidBucket {
            axis: "T_dec",
            value: 0
        })
    ));
}

#[test]
fn bucket_step_collects_errors() {
    // Both s=0 and t_dec=0 are invalid
    let res = bucket_step(0, 0, 0);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(matches!(err, IrError::Multiple { .. }));
}

#[test]
fn step_graph_key_validation() {
    let plan = PlanId::new(1);
    // Exact bucket sizes build cleanly
    let key = StepGraphKey::new(plan, 0, 16, 32, 0, 0).expect("valid buckets build");
    assert_eq!(key.s, 16);
    assert_eq!(key.t_dec, 32);
    assert_eq!(key.t_pre, 0);

    // Unbucketed values fail with InvalidBucket
    let bad_key = StepGraphKey::new(plan, 0, 15, 32, 0, 0);
    assert!(matches!(
        bad_key,
        Err(IrError::InvalidBucket {
            axis: "S",
            value: 15
        })
    ));

    assert!(matches!(
        StepGraphKey::new(plan, 0, 4097, 32, 0, 0),
        Err(IrError::BucketExceeded {
            axis: "S",
            value: 4097,
            max: 4096
        })
    ));

    // from_unbucketed automatically rounds up
    let unbucketed = StepGraphKey::from_unbucketed(plan, 0, 15, 20, 100, 0).unwrap();
    assert_eq!(unbucketed.s, 16);
    assert_eq!(unbucketed.t_dec, 32);
    assert_eq!(unbucketed.t_pre, 128);
}

#[test]
fn graph_dag_cycle_detection() {
    let plan = PlanId::new(1);
    let key = StepGraphKey::new(plan, 0, 1, 1, 0, 0).unwrap();
    let mut graph = Graph::new(key);

    let t = Tensor::new(
        vec![Dim::Concrete(1), Dim::Concrete(16)],
        DType::F32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();

    let e0 = graph
        .add_external_input(ExternalInputKind::EmbedOverride, t.clone())
        .unwrap();

    let act_op = Op::Activation(ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    });

    // Add node 0: in=[e0], out=[e1]
    let n0 = graph
        .add_op(act_op.clone(), &[e0], std::slice::from_ref(&t))
        .unwrap();
    let e1 = graph.nodes()[n0.0].outputs[0];

    // Add node 1: in=[e1], out=[e2]
    let n1 = graph
        .add_op(act_op.clone(), &[e1], std::slice::from_ref(&t))
        .unwrap();
    let e2 = graph.nodes()[n1.0].outputs[0];

    // Graph is a linear chain, no cycles
    assert!(graph.validate().is_ok());

    // Create a cycle by rewiring n0's input to n1's output e2
    let mut cyclic_graph = graph.clone();
    cyclic_graph.rewire_node_input(n0, 0, e2).unwrap();
    let val_res = cyclic_graph.validate();
    assert!(val_res.is_err());
    assert!(matches!(val_res.unwrap_err(), IrError::GraphCycle { .. }));
}

#[test]
fn graph_detects_cycle_through_metadata_view() {
    let key = StepGraphKey::new(PlanId::new(1), 0, 1, 1, 0, 0).unwrap();
    let mut graph = Graph::new(key);
    let input = Tensor::new(
        vec![Dim::Concrete(1), Dim::Concrete(2)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let input_edge = graph
        .add_external_input(ExternalInputKind::EmbedOverride, input.clone())
        .unwrap();
    let activation = Op::Activation(ActivationOp {
        act: ActivationKind::Identity,
        clamp: None,
    });
    let first = graph
        .add_op(activation.clone(), &[input_edge], &[input])
        .unwrap();
    let first_output = graph.nodes()[first.0].outputs[0];
    let view = graph.transpose_edge(first_output, &[1, 0]).unwrap();
    let view_tensor = graph.edges()[view.0].tensor.clone();
    let second = graph.add_op(activation, &[view], &[view_tensor]).unwrap();
    let second_output = graph.nodes()[second.0].outputs[0];
    graph.rewire_node_input(first, 0, second_output).unwrap();

    assert!(matches!(
        graph.topological_order(),
        Err(IrError::GraphCycle { .. })
    ));
}

#[test]
fn stride_mismatch_materializes_exactly_one_copy() {
    let plan = PlanId::new(1);
    let key = StepGraphKey::new(plan, 0, 1, 16, 0, 0).unwrap();
    let mut graph = Graph::new(key);

    // Initial activation tensor [16, 64]
    let t_in = Tensor::new(
        vec![Dim::Concrete(16), Dim::Concrete(64)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();

    let e_in = graph
        .add_external_input(ExternalInputKind::EmbedOverride, t_in)
        .unwrap();

    // Node 0: Activation produces [16, 64]
    let act_op = Op::Activation(ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    });
    let t_act = Tensor::new(
        vec![Dim::Concrete(16), Dim::Concrete(64)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let n0 = graph.add_op(act_op.clone(), &[e_in], &[t_act]).unwrap();
    let e_act = graph.nodes()[n0.0].outputs[0];

    // Transpose view permuting [16, 64] -> [64, 16]
    // Strides become [1, 64] instead of contiguous [16, 1]
    let e_transposed = graph.transpose_edge(e_act, &[1, 0]).unwrap();
    assert!(!graph.edges()[e_transposed.0].is_contiguous());
    assert_eq!(graph.edges()[e_transposed.0].strides, vec![1, 64]);

    // Weight tensor for NormOp [16]
    let t_weight = Tensor::new(
        vec![Dim::Concrete(16)],
        DType::F32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap();
    let e_weight = graph.add_tensor(t_weight).unwrap();

    // Node 1: NormOp consumes the transposed edge [64, 16] and weight [16]
    let norm_op = Op::Norm(NormOp {
        kind: NormKind::Rms,
        eps: 1e-5,
        axis: NormAxis::Last,
        weight_offset: 0.0,
        out_dtype: DType::F16,
    });
    let t_norm_out = Tensor::new(
        vec![Dim::Concrete(64), Dim::Concrete(16)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();

    let n1 = graph
        .add_op_with_requirements(
            norm_op,
            &[e_transposed, e_weight],
            &[StrideRequirement::Contiguous, StrideRequirement::Contiguous],
            &[t_norm_out],
        )
        .unwrap();
    // Before materialize_copies: consumer input has stride mismatch
    assert_eq!(graph.nodes()[n1.0].inputs[0], e_transposed);
    assert_eq!(graph.inserted_copies().len(), 0);

    // Call materialize_copies(): must detect stride mismatch and insert EXACTLY ONE CopyOp
    let inserted_count = graph
        .materialize_copies()
        .expect("copy materialization succeeds");
    assert_eq!(
        inserted_count, 1,
        "Exactly one copy must be inserted for the stride mismatch"
    );

    // Inspect GraphSummary
    let summary = graph.summary();
    assert_eq!(summary.inserted_copy_count(), 1);
    assert_eq!(summary.inserted_copies.len(), 1);

    let copy_record = &summary.inserted_copies[0];
    assert_eq!(copy_record.source_edge, e_transposed);
    assert_eq!(copy_record.consumer_nodes, vec![n1]);
    assert_eq!(copy_record.actual_strides, vec![1, 64]);
    assert_eq!(copy_record.expected_strides, vec![16, 1]);

    // Consumer input was rewired to the contiguous copy output
    let copy_out_edge = copy_record.dest_edge;
    assert_eq!(graph.nodes()[n1.0].inputs[0], copy_out_edge);
    assert!(graph.edges()[copy_out_edge.0].is_contiguous());
    let execution_order = graph.topological_order().unwrap();
    let copy_position = execution_order
        .iter()
        .position(|&node| node == copy_record.copy_node)
        .unwrap();
    let consumer_position = execution_order.iter().position(|&node| node == n1).unwrap();
    assert!(copy_position < consumer_position);

    // Second call to materialize_copies is idempotent (0 new copies inserted)
    let second_run = graph.materialize_copies().expect("second run succeeds");
    assert_eq!(second_run, 0, "Second run must not insert duplicate copies");

    // Validating the contiguized graph succeeds
    assert!(graph.validate().is_ok());
}

#[test]
fn noncontiguous_any_requirement_inserts_no_copy() {
    let plan = PlanId::new(1);
    let key = StepGraphKey::new(plan, 0, 1, 16, 0, 0).unwrap();
    let mut graph = Graph::new(key);

    let t_in = Tensor::new(
        vec![Dim::Concrete(16), Dim::Concrete(64)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();

    let e_in = graph
        .add_external_input(ExternalInputKind::EmbedOverride, t_in)
        .unwrap();

    let act_op = Op::Activation(ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    });
    let t_act = Tensor::new(
        vec![Dim::Concrete(16), Dim::Concrete(64)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap();
    let n0 = graph.add_op(act_op.clone(), &[e_in], &[t_act]).unwrap();
    let e_act = graph.nodes()[n0.0].outputs[0];

    // Transpose view: [16, 64] -> [64, 16] with non-contiguous strides [1, 64]
    let e_transposed = graph.transpose_edge(e_act, &[1, 0]).unwrap();
    assert!(!graph.edges()[e_transposed.0].is_contiguous());

    let t_out = Tensor::new(
        vec![Dim::Concrete(64), Dim::Concrete(16)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();

    // Consumer op added with explicit StrideRequirement::Any
    let n1 = graph
        .add_op_with_requirements(act_op, &[e_transposed], &[StrideRequirement::Any], &[t_out])
        .unwrap();

    // Verify requirement on node
    assert_eq!(
        graph.nodes()[n1.0].input_requirements,
        vec![StrideRequirement::Any]
    );

    // materialize_copies must NOT insert a copy because requirement is Any
    let inserted = graph
        .materialize_copies()
        .expect("materialize copies succeeds");
    assert_eq!(
        inserted, 0,
        "materialize_copies must not insert copies for StrideRequirement::Any"
    );
    assert_eq!(graph.summary().inserted_copy_count(), 0);
    assert_eq!(graph.inserted_copies().len(), 0);
    assert_eq!(
        graph.nodes()[n1.0].inputs[0],
        e_transposed,
        "Consumer input must not be rewired"
    );
}

#[test]
fn transpose_edge_rejects_duplicate_and_missing_axes() {
    let plan = PlanId::new(1);
    let key = StepGraphKey::new(plan, 0, 1, 1, 0, 0).unwrap();
    let mut graph = Graph::new(key);

    let t = Tensor::new(
        vec![Dim::Concrete(4), Dim::Concrete(8), Dim::Concrete(16)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap();

    let e = graph.add_tensor(t).unwrap();

    // Valid permutation: [2, 0, 1] succeeds
    let e_trans = graph.transpose_edge(e, &[2, 0, 1]).unwrap();
    assert_eq!(
        graph.edges()[e_trans.0].tensor.shape(),
        &[Dim::Concrete(16), Dim::Concrete(4), Dim::Concrete(8)]
    );

    // Duplicate axis [0, 0, 1] rejected
    let res_dup = graph.transpose_edge(e, &[0, 0, 1]);
    assert!(res_dup.is_err());
    assert!(matches!(
        res_dup.unwrap_err(),
        IrError::OpAttributeInvalid {
            attribute: "perm",
            ..
        }
    ));

    // Missing axis and out-of-bounds axis [0, 1, 5] rejected
    let res_oob = graph.transpose_edge(e, &[0, 1, 5]);
    assert!(res_oob.is_err());
    assert!(matches!(
        res_oob.unwrap_err(),
        IrError::OpAttributeInvalid {
            attribute: "perm",
            ..
        }
    ));

    // Rank mismatch (len 2 vs rank 3) rejected
    let res_rank = graph.transpose_edge(e, &[0, 1]);
    assert!(res_rank.is_err());
    assert!(matches!(
        res_rank.unwrap_err(),
        IrError::OpRankMismatch {
            expected: 3,
            got: 2,
            ..
        }
    ));
}

#[test]
fn reshape_edge_preserves_storage_size_and_rejects_noncontiguous_sources() {
    let key = StepGraphKey::new(PlanId::new(1), 0, 1, 4, 0, 0).unwrap();
    let mut graph = Graph::new(key);
    let source = Tensor::new(
        vec![Dim::Concrete(4), Dim::Concrete(8)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let source_edge = graph
        .add_external_input(ExternalInputKind::EmbedOverride, source)
        .unwrap();

    let reshaped = graph
        .reshape_edge(
            source_edge,
            vec![Dim::Concrete(2), Dim::Concrete(4), Dim::Concrete(4)],
        )
        .expect("equal-size contiguous reshape is a view");
    assert_eq!(
        graph.edges()[reshaped.0].tensor.shape(),
        &[Dim::Concrete(2), Dim::Concrete(4), Dim::Concrete(4)]
    );
    assert!(graph.edges()[reshaped.0].is_contiguous());

    assert!(matches!(
        graph.reshape_edge(source_edge, vec![Dim::Concrete(31)]),
        Err(IrError::OpShapeMismatch { .. })
    ));

    let transposed = graph.transpose_edge(source_edge, &[1, 0]).unwrap();
    assert!(matches!(
        graph.reshape_edge(transposed, vec![Dim::Concrete(32)]),
        Err(IrError::StrideMismatch { .. })
    ));
}

#[test]
fn external_inputs_distinguish_tensor_and_non_tensor() {
    let plan = PlanId::new(1);
    let key = StepGraphKey::new(plan, 0, 1, 1, 0, 0).unwrap();
    let mut graph = Graph::new(key);

    let t = Tensor::new(
        vec![Dim::Concrete(1)],
        DType::U32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();

    // Tensor-backed input succeeds
    let e_tok = graph
        .add_external_input(ExternalInputKind::TokenIds, t.clone())
        .unwrap();
    assert_eq!(e_tok.0, 0);

    // Non-tensor inputs cannot be added via add_external_input with fake Tensor
    let bad_bm = graph.add_external_input(ExternalInputKind::BatchMeta, t.clone());
    assert!(bad_bm.is_err());
    assert!(matches!(
        bad_bm.unwrap_err(),
        IrError::OpAttributeInvalid {
            attribute: "kind",
            ..
        }
    ));

    let bad_sp = graph.add_external_input(ExternalInputKind::SamplingParams, t.clone());
    assert!(bad_sp.is_err());
    assert!(matches!(
        bad_sp.unwrap_err(),
        IrError::OpAttributeInvalid {
            attribute: "kind",
            ..
        }
    ));

    // Non-tensor inputs registered cleanly without fake Tensor descriptors
    graph.add_batch_meta_input().expect("batch meta registers");
    graph
        .add_sampling_params_input()
        .expect("sampling params registers");

    // Duplicate non-tensor inputs are rejected
    assert!(graph.add_batch_meta_input().is_err());
    assert!(graph.add_sampling_params_input().is_err());

    // Registering tensor-backed input as non-tensor is rejected
    let bad_non_tensor = graph.add_external_non_tensor(ExternalInputKind::TokenIds);
    assert!(bad_non_tensor.is_err());
    assert!(matches!(
        bad_non_tensor.unwrap_err(),
        IrError::OpAttributeInvalid {
            attribute: "kind",
            ..
        }
    ));

    // Verify external inputs list and exact ExternalInput bindings
    assert_eq!(graph.external_inputs().len(), 3);
    assert_eq!(
        graph.external_inputs()[0],
        ExternalInput::Tensor {
            kind: ExternalInputKind::TokenIds,
            edge: e_tok,
        }
    );
    assert_eq!(
        graph.external_inputs()[0].kind(),
        ExternalInputKind::TokenIds
    );
    assert_eq!(graph.external_inputs()[0].edge_id(), Some(e_tok));
    assert_eq!(graph.external_inputs()[1], ExternalInput::BatchMeta);
    assert_eq!(
        graph.external_inputs()[1].kind(),
        ExternalInputKind::BatchMeta
    );
    assert_eq!(graph.external_inputs()[1].edge_id(), None);
    assert_eq!(graph.external_inputs()[2], ExternalInput::SamplingParams);
    assert_eq!(
        graph.external_inputs()[2].kind(),
        ExternalInputKind::SamplingParams
    );
    assert_eq!(graph.external_inputs()[2].edge_id(), None);

    // Summary correctly reflects all external inputs
    let summary = graph.summary();
    assert_eq!(
        summary.external_inputs,
        vec![
            ExternalInputKind::TokenIds,
            ExternalInputKind::BatchMeta,
            ExternalInputKind::SamplingParams,
        ]
    );

    // External outputs are typed and bound only to semantically matching edges.
    let hidden = Tensor::new(
        vec![Dim::Concrete(1), Dim::Concrete(16)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let e_hidden_input = graph
        .add_external_input(ExternalInputKind::EmbedOverride, hidden.clone())
        .unwrap();
    let act_op = Op::Activation(ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    });
    let n0 = graph.add_op(act_op, &[e_hidden_input], &[hidden]).unwrap();
    let e_hidden = graph.nodes()[n0.0].outputs[0];
    let out_res = graph.add_external_output(ExternalOutputKind::Hidden, e_hidden);
    assert!(out_res.is_ok());
    assert_eq!(graph.external_outputs().len(), 1);
    assert_eq!(
        graph.external_outputs()[0],
        ExternalOutput::Tensor {
            kind: ExternalOutputKind::Hidden,
            edge: e_hidden,
        }
    );

    // Nonexistent edge rejected
    let bad_out = graph.add_external_output(ExternalOutputKind::Hidden, EdgeId(999));
    assert!(bad_out.is_err());
    assert!(matches!(
        bad_out.unwrap_err(),
        IrError::GraphEdgeNotFound { edge: 999 }
    ));

    // Stride requirement mutation on node
    assert_eq!(
        graph.nodes()[n0.0].input_requirements[0],
        StrideRequirement::Any
    );
    graph
        .set_node_input_requirement(n0, 0, StrideRequirement::Contiguous)
        .expect("set requirement succeeds");
    assert_eq!(
        graph.nodes()[n0.0].input_requirements[0],
        StrideRequirement::Contiguous
    );
}

#[test]
fn external_tensor_bindings_enforce_bucket_shape_dtype_and_role() {
    let key = StepGraphKey::new(PlanId::new(2), 0, 4, 4, 0, 0).unwrap();
    let mut graph = Graph::new(key);

    let token_ids = Tensor::new(
        vec![Dim::Concrete(4)],
        DType::U32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let token_edge = graph
        .add_external_input(ExternalInputKind::TokenIds, token_ids)
        .expect("bucket-matched token IDs register");
    assert!(graph
        .add_external_input(
            ExternalInputKind::TokenIds,
            Tensor::new(
                vec![Dim::Concrete(4)],
                DType::U32,
                QuantScheme::None,
                LayoutId::CONTIGUOUS,
                Placement::Device { rank: 0 },
                ShardLayout::Replicated,
                Class::Activation,
            )
            .unwrap(),
        )
        .is_err());

    let wrong_token_shape = Tensor::new(
        vec![Dim::Concrete(2)],
        DType::U32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let mut other_graph = Graph::new(key);
    assert!(matches!(
        other_graph.add_external_input(ExternalInputKind::TokenIds, wrong_token_shape),
        Err(IrError::OpShapeMismatch { .. })
    ));

    let hidden = Tensor::new(
        vec![Dim::Concrete(4), Dim::Concrete(32)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let hidden_input = graph
        .add_external_input(ExternalInputKind::EmbedOverride, hidden.clone())
        .unwrap();
    let hidden_node = graph
        .add_op(
            Op::Activation(ActivationOp {
                act: ActivationKind::Identity,
                clamp: None,
            }),
            &[hidden_input],
            &[hidden],
        )
        .unwrap();
    let hidden_edge = graph.nodes()[hidden_node.0].outputs[0];
    graph
        .add_external_output(ExternalOutputKind::Hidden, hidden_edge)
        .expect("bucket-matched hidden output registers");

    let sampled_token = Tensor::new(
        vec![Dim::Concrete(4)],
        DType::U32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let sampled_node = graph
        .add_op(
            Op::Cast(CastOp { dtype: DType::U32 }),
            &[token_edge],
            &[sampled_token],
        )
        .unwrap();
    let sampled_token_edge = graph.nodes()[sampled_node.0].outputs[0];
    let sampled_matrix_edge = graph
        .reshape_edge(sampled_token_edge, vec![Dim::Concrete(4), Dim::Concrete(1)])
        .unwrap();
    graph
        .add_external_output(ExternalOutputKind::Sampled, sampled_matrix_edge)
        .expect("sample token view satisfies the external sampled matrix contract");
    assert!(matches!(
        graph.add_external_output(ExternalOutputKind::AcceptLen, token_edge),
        Err(IrError::GraphExternalOutputUnproduced { .. })
    ));

    let weight = Tensor::new(
        vec![Dim::Concrete(32), Dim::Concrete(32)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap();
    let _weight_edge = graph.add_tensor(weight).unwrap();
    let invalid_source = Tensor::new(
        vec![Dim::Concrete(4), Dim::Concrete(8)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    assert!(matches!(
        graph.add_tensor(invalid_source),
        Err(IrError::GraphSourceClassInvalid {
            class: Class::Activation
        })
    ));
    assert_eq!(graph.external_inputs().len(), 2);
}

#[test]
fn public_mutations_validate_and_collect_multiple_errors() {
    let plan = PlanId::new(1);
    let key = StepGraphKey::new(plan, 0, 1, 1, 0, 0).unwrap();
    let mut graph = Graph::new(key);

    let act_op = Op::Activation(ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    });
    let t = Tensor::new(
        vec![Dim::Concrete(16)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();

    // add_op with multiple nonexistent edges collects multiple errors
    let bad_edges = [EdgeId(10), EdgeId(20)];
    let res_edges = graph.add_op(act_op.clone(), &bad_edges, std::slice::from_ref(&t));
    assert!(res_edges.is_err());
    let err = res_edges.unwrap_err();
    assert!(matches!(err, IrError::Multiple { .. }));

    // rewire_node_input with bad node and bad edge collects multiple errors
    let res_rewire = graph.rewire_node_input(NodeId(50), 0, EdgeId(50));
    assert!(res_rewire.is_err());
    assert!(matches!(res_rewire.unwrap_err(), IrError::Multiple { .. }));
}
