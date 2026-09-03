// SPDX-License-Identifier: Apache-2.0
//! Tests for Op sharding tables and propagation rules (Spec 1 §5.2; card A1.2).

use r9v_ir::{
    legal_layout_tuples, legal_layouts, ActMulOp, ActivationKind, ActivationOp, AllGatherOp,
    AllReduceOp, AllToAllOp, AttentionMask, AttentionOp, BarrierOp, CacheScaleGranularity, CastOp,
    CausalConv1dOp, Class, ConvActivation, CopyOp, DType, Dim, EmbedGatherOp, Epilogue,
    ExpertCount, GatherRowsOp, GroupId, HashId, HeadCount, LayoutId, LinearAttnKind,
    LinearAttnScanOp, LogitsPostprocessOp, MatmulOp, MoeFfnOp, MoeRouteOp, MoeScoring,
    NgramCombine, NgramGatherOp, NgramSource, NormAxis, NormKind, NormOp, Op, Placement,
    QuantActOp, QuantScheme, RecvOp, ReduceOp, ReduceScatterOp, ResidualAddOp, RngAlgorithm,
    RopeOp, RopeScaling, RopeStyle, SampleOp, ScatterAddRowsOp, SendOp, ShardLayout,
    ShardLayoutPattern, ShardingRule, Smoothing, StateHandle, StateKind, StateWriteKvOp, Tensor,
    VerifyMethod, VerifyOp,
};

fn all_sample_ops() -> Vec<Op> {
    vec![
        Op::EmbedGather(EmbedGatherOp {
            scale: 1.0,
            out_dtype: DType::F16,
        }),
        Op::NgramGather(NgramGatherOp {
            source: NgramSource::Device,
            orders: vec![2, 3].into_boxed_slice(),
            heads: 2,
            hash: HashId::new(1),
            table_sizes: vec![1024, 1024].into_boxed_slice(),
            combine: NgramCombine::Concat,
            out_dtype: DType::F16,
        }),
        Op::QuantAct(QuantActOp {
            scheme: QuantScheme::PerToken,
            target: DType::I8,
            smoothing: Smoothing::None,
        }),
        Op::Cast(CastOp { dtype: DType::F16 }),
        Op::Copy(CopyOp::default()),
        Op::GatherRows(GatherRowsOp),
        Op::ScatterAddRows(ScatterAddRowsOp),
        Op::Norm(NormOp {
            kind: NormKind::Rms,
            eps: 1e-5,
            axis: NormAxis::Last,
            weight_offset: 0.0,
            out_dtype: DType::F16,
        }),
        Op::ResidualAdd(ResidualAddOp {
            out_dtype: DType::F16,
        }),
        Op::ActMul(ActMulOp {
            act: ActivationKind::Silu,
            clamp: None,
        }),
        Op::Activation(ActivationOp {
            act: ActivationKind::Silu,
            clamp: None,
        }),
        Op::Rope(RopeOp {
            rot_dim: 64,
            theta: 10000.0,
            style: RopeStyle::Neox,
            scaling: RopeScaling::None,
            mrope_sections: None,
            out_dtype: DType::F16,
        }),
        Op::Matmul(MatmulOp {
            out_dtype: DType::F16,
            epilogue: Epilogue::None,
            transpose_w: false,
        }),
        Op::MoeRoute(MoeRouteOp {
            top_k: 2,
            scoring: MoeScoring::Softmax,
            renormalize: true,
            group: None,
            scale: 1.0,
        }),
        Op::MoeFfn(MoeFfnOp {
            act: ActivationKind::Silu,
            out_dtype: DType::F16,
            shared_experts: 0,
        }),
        Op::StateWriteKv(StateWriteKvOp {
            cache_dtype: DType::F16,
            scale_granularity: CacheScaleGranularity::PerTokenHead,
            latent: None,
            handle: StateHandle::new(0, StateKind::KvPaged),
        }),
        Op::Attention(AttentionOp {
            softmax_scale: 0.125,
            mask: AttentionMask::Causal,
            sinks: 0,
            logit_softcap: None,
            mla: None,
            out_dtype: DType::F16,
            handle: StateHandle::new(0, StateKind::KvPaged),
        }),
        Op::CausalConv1d(CausalConv1dOp {
            kernel: 4,
            act: ConvActivation::Silu,
            handle: StateHandle::new(0, StateKind::ConvWindow),
        }),
        Op::LinearAttnScan(LinearAttnScanOp {
            kind: LinearAttnKind::GLA,
            chunk: 64,
            out_dtype: DType::F16,
            handle: StateHandle::new(0, StateKind::Recurrent),
        }),
        Op::LogitsPostprocess(LogitsPostprocessOp),
        Op::Sample(SampleOp {
            rng: RngAlgorithm::Philox4x32,
        }),
        Op::Verify(VerifyOp {
            method: VerifyMethod::Greedy,
        }),
        Op::AllReduce(AllReduceOp {
            group: GroupId::new(0),
            op: ReduceOp::Sum,
            dtype: DType::F16,
            reduce_in: DType::F32,
        }),
        Op::AllGather(AllGatherOp {
            group: GroupId::new(0),
            axis: 0,
            dtype: DType::F16,
        }),
        Op::ReduceScatter(ReduceScatterOp {
            group: GroupId::new(0),
            axis: 0,
            op: ReduceOp::Sum,
            dtype: DType::F16,
            reduce_in: DType::F32,
        }),
        Op::AllToAll(AllToAllOp {
            group: GroupId::new(0),
            dtype: DType::F16,
        }),
        Op::Send(SendOp {
            group: GroupId::new(0),
            peer: 1,
            dtype: DType::F16,
        }),
        Op::Recv(RecvOp {
            group: GroupId::new(0),
            peer: 0,
            shape: vec![].into_boxed_slice(),
            dtype: DType::F16,
        }),
        Op::Barrier(BarrierOp {
            group: GroupId::new(0),
        }),
    ]
}

#[test]
fn every_op_has_at_least_one_legal_sharding_tuple() {
    let ops = all_sample_ops();
    assert_eq!(ops.len(), 29, "Must cover all 29 closed ops in Spec 1 §4");

    for op in &ops {
        let rules = legal_layouts(op);
        assert!(
            !rules.is_empty(),
            "Op {} must have at least one legal sharding rule",
            op.op_name()
        );

        let tuples = legal_layout_tuples(op);
        assert_eq!(
            rules.len(),
            tuples.len(),
            "Rules count must match tuples count for op {}",
            op.op_name()
        );

        for rule in rules {
            // Verify rule has valid input/output shard layout patterns without magic zero
            for in_layout in rule.inputs {
                assert!(matches!(
                    in_layout,
                    ShardLayoutPattern::Replicated
                        | ShardLayoutPattern::ColShard { axis: _ }
                        | ShardLayoutPattern::RowShard { axis: _ }
                        | ShardLayoutPattern::HeadShard { heads: _ }
                        | ShardLayoutPattern::ExpertShard { experts: _ }
                        | ShardLayoutPattern::Partial
                ));
                match in_layout {
                    ShardLayoutPattern::HeadShard { heads } => match heads {
                        HeadCount::Concrete(c) => {
                            assert!(*c > 0, "Concrete head count must be > 0, no magic zero");
                        }
                        HeadCount::Symbolic => {}
                    },
                    ShardLayoutPattern::ExpertShard { experts } => match experts {
                        ExpertCount::Concrete(c) => {
                            assert!(*c > 0, "Concrete expert count must be > 0, no magic zero");
                        }
                        ExpertCount::Symbolic => {}
                    },
                    ShardLayoutPattern::Replicated
                    | ShardLayoutPattern::ColShard { .. }
                    | ShardLayoutPattern::RowShard { .. }
                    | ShardLayoutPattern::Partial => {}
                }
            }
            for out_layout in rule.outputs {
                assert!(matches!(
                    out_layout,
                    ShardLayoutPattern::Replicated
                        | ShardLayoutPattern::ColShard { axis: _ }
                        | ShardLayoutPattern::RowShard { axis: _ }
                        | ShardLayoutPattern::HeadShard { heads: _ }
                        | ShardLayoutPattern::ExpertShard { experts: _ }
                        | ShardLayoutPattern::Partial
                ));
                match out_layout {
                    ShardLayoutPattern::HeadShard { heads } => match heads {
                        HeadCount::Concrete(c) => {
                            assert!(*c > 0, "Concrete head count must be > 0, no magic zero");
                        }
                        HeadCount::Symbolic => {}
                    },
                    ShardLayoutPattern::ExpertShard { experts } => match experts {
                        ExpertCount::Concrete(c) => {
                            assert!(*c > 0, "Concrete expert count must be > 0, no magic zero");
                        }
                        ExpertCount::Symbolic => {}
                    },
                    ShardLayoutPattern::Replicated
                    | ShardLayoutPattern::ColShard { .. }
                    | ShardLayoutPattern::RowShard { .. }
                    | ShardLayoutPattern::Partial => {}
                }
            }
        }
    }
}

fn make_test_tensor(shape: Vec<Dim>, sharding: ShardLayout) -> Tensor {
    Tensor::new(
        shape,
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        sharding,
        Class::Activation,
    )
    .expect("test tensor builds")
}

#[test]
fn matmul_sharding_rules_match_spec() {
    let matmul = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    let rules = matmul.legal_layouts();
    assert_eq!(
        rules.len(),
        3,
        "Matmul declares exactly the three principal two-input rows in Spec 1 §4.C"
    );

    // 1. Column parallel: x: Replicated, w: ColShard(0) -> y: ColShard(1)
    assert_eq!(
        rules[0].inputs,
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::ColShard { axis: 0 }
        ]
    );
    assert_eq!(
        rules[0].outputs,
        &[ShardLayoutPattern::ColShard { axis: 1 }]
    );

    // 2. Row parallel: x: ColShard(1), w: RowShard(1) -> y: Partial
    assert_eq!(
        rules[1].inputs,
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::RowShard { axis: 1 }
        ]
    );
    assert_eq!(rules[1].outputs, &[ShardLayoutPattern::Partial]);

    // 3. Replicated: x: Replicated, w: Replicated -> y: Replicated
    assert_eq!(
        rules[2].inputs,
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated
        ]
    );
    assert_eq!(rules[2].outputs, &[ShardLayoutPattern::Replicated]);
}

#[test]
fn residual_add_supports_spec5_residual_trick() {
    let res = ResidualAddOp {
        out_dtype: DType::F16,
    };
    let rules = res.legal_layouts();
    // Spec 5 §3.1 / §4.2 residual trick:
    // residual_add(Partial, Replicated) -> Partial
    // residual_add(Replicated, Partial) -> Partial
    let has_partial_rep = rules.iter().any(|r| {
        r.inputs == [ShardLayoutPattern::Partial, ShardLayoutPattern::Replicated]
            && r.outputs == [ShardLayoutPattern::Partial]
    });
    let has_rep_partial = rules.iter().any(|r| {
        r.inputs == [ShardLayoutPattern::Replicated, ShardLayoutPattern::Partial]
            && r.outputs == [ShardLayoutPattern::Partial]
    });
    assert!(
        has_partial_rep,
        "Must support Spec 5 §4.2 residual trick: (Partial, Replicated) -> Partial"
    );
    assert!(
        has_rep_partial,
        "Must support Spec 5 §4.2 residual trick: (Replicated, Partial) -> Partial"
    );
}

#[test]
fn embed_gather_rules_has_no_invented_colshard() {
    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F16,
    };
    let rules = op.legal_layouts();
    assert_eq!(
        rules.len(),
        2,
        "EmbedGather has exactly 2 rules: Replicated and RowShard(0)"
    );
    assert_eq!(
        rules[0].inputs,
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated
        ]
    );
    assert_eq!(rules[0].outputs, &[ShardLayoutPattern::Replicated]);
    assert_eq!(
        rules[1].inputs,
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::RowShard { axis: 0 }
        ]
    );
    assert_eq!(rules[1].outputs, &[ShardLayoutPattern::Partial]);

    for rule in rules {
        assert!(
            !rule
                .inputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::ColShard { .. })),
            "No invented ColShard on embed inputs"
        );
        assert!(
            !rule
                .outputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::ColShard { .. })),
            "No invented ColShard on embed outputs"
        );
    }
}

#[test]
fn ngram_gather_rules_has_no_stale_one_input_rows() {
    let op = NgramGatherOp {
        source: NgramSource::Device,
        orders: vec![2, 3].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(1),
        table_sizes: vec![1024, 1024].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F16,
    };
    let rules = op.legal_layouts();
    assert_eq!(rules.len(), 3, "NgramGather has exactly 3 rules");
    for rule in rules {
        assert_eq!(
            rule.inputs.len(),
            2,
            "NgramGather must have 2 inputs (no stale 1-input rule)"
        );
        assert_eq!(rule.outputs.len(), 1, "NgramGather must have 1 output");
    }
}

#[test]
fn quant_act_rules_has_no_batch_axis_rowshard() {
    let op = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };
    let rules = op.legal_layouts();
    assert_eq!(rules.len(), 2);
    for rule in rules {
        assert_eq!(rule.inputs.len(), 1);
        assert_eq!(rule.outputs.len(), 2);
        assert!(
            !rule
                .inputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::RowShard { axis: 0 })),
            "No RowShard(0) on quant_act inputs"
        );
    }
}

#[test]
fn state_sampling_and_verify_rules_exclude_structured_non_tensor_values() {
    let state_write = StateWriteKvOp {
        cache_dtype: DType::F16,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: None,
        handle: StateHandle::new(0, StateKind::KvPaged),
    };
    for rule in state_write.legal_layouts() {
        assert_eq!(
            rule.inputs.len(),
            2,
            "StateWriteKv tensor inputs are k and v"
        );
        assert!(rule.outputs.is_empty());
    }

    let sample = SampleOp {
        rng: RngAlgorithm::Philox4x32,
    };
    let sample_rules = sample.legal_layouts();
    assert_eq!(sample_rules.len(), 1);
    assert_eq!(
        sample_rules[0].inputs.len(),
        1,
        "Sample tensor input is only probs"
    );
    assert_eq!(
        sample_rules[0].outputs.len(),
        1,
        "Sample tensor output is only token (no stale 2-output)"
    );

    let verify = VerifyOp {
        method: VerifyMethod::Greedy,
    };
    let verify_rules = verify.legal_layouts();
    assert_eq!(verify_rules.len(), 2);
    for rule in verify_rules {
        assert!(rule.inputs.len() == 2 || rule.inputs.len() == 3);
        assert_eq!(
            rule.outputs.len(),
            2,
            "Verify tensor outputs are accepted and accept_len (no stale 3-output)"
        );
    }
}

#[test]
fn all_to_all_rules_exactly_inputs_x_and_counts_to_y() {
    let op = AllToAllOp {
        group: GroupId::new(0),
        dtype: DType::F16,
    };
    let rules = op.legal_layouts();
    assert_eq!(rules.len(), 2, "AllToAll declares exactly 2 rules");
    for rule in rules {
        assert_eq!(
            rule.inputs.len(),
            2,
            "AllToAll inputs are exactly x and counts"
        );
        assert_eq!(rule.outputs.len(), 1, "AllToAll output is exactly y");
    }
    assert_eq!(
        rules[0].inputs,
        &[
            ShardLayoutPattern::ExpertShard {
                experts: ExpertCount::Symbolic,
            },
            ShardLayoutPattern::Replicated,
        ]
    );
    assert_eq!(
        rules[0].outputs,
        &[ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Symbolic,
        }]
    );
    assert_eq!(
        rules[1].inputs,
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ]
    );
    assert_eq!(rules[1].outputs, &[ShardLayoutPattern::Replicated]);
}

#[test]
fn partial_may_flow_only_through_residual_add_and_matmul() {
    // Check PASSTHROUGH_RULES
    let cast = CastOp { dtype: DType::F16 };
    for rule in cast.legal_layouts() {
        assert!(
            !rule
                .inputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::Partial)),
            "Cast/Copy passthrough must not accept Partial"
        );
        assert!(
            !rule
                .outputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::Partial)),
            "Cast/Copy passthrough must not emit Partial"
        );
    }

    // Check SEND_RULES and RECV_RULES
    let send = SendOp {
        group: GroupId::new(0),
        peer: 1,
        dtype: DType::F16,
    };
    for rule in send.legal_layouts() {
        assert!(
            !rule
                .inputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::Partial)),
            "Send must not accept Partial across pipeline boundaries"
        );
    }
    let recv = RecvOp {
        group: GroupId::new(0),
        peer: 0,
        shape: vec![].into_boxed_slice(),
        dtype: DType::F16,
    };
    for rule in recv.legal_layouts() {
        assert!(
            !rule
                .outputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::Partial)),
            "Recv must not emit Partial across pipeline boundaries"
        );
    }
}

#[test]
fn matches_tensors_validates_arity() {
    let t1 = make_test_tensor(
        vec![Dim::Concrete(1), Dim::Concrete(8)],
        ShardLayout::Replicated,
    );
    let t2 = make_test_tensor(
        vec![Dim::Concrete(1), Dim::Concrete(8)],
        ShardLayout::Replicated,
    );

    let rule_2_in_1_out = ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    );

    // Exact arity: 2 in, 1 out -> true
    assert!(rule_2_in_1_out.matches_tensors(&[t1.clone(), t2.clone()], std::slice::from_ref(&t1)));
    // Arity mismatch: 1 in, 1 out -> false
    assert!(!rule_2_in_1_out.matches_tensors(std::slice::from_ref(&t1), std::slice::from_ref(&t1)));
    // Arity mismatch: 3 in, 1 out -> false
    assert!(!rule_2_in_1_out.matches_tensors(
        &[t1.clone(), t2.clone(), t1.clone()],
        std::slice::from_ref(&t1)
    ));
    // Arity mismatch: 2 in, 0 out -> false
    assert!(!rule_2_in_1_out.matches_tensors(&[t1.clone(), t2.clone()], &[]));
    // Arity mismatch: 2 in, 2 out -> false
    assert!(!rule_2_in_1_out.matches_tensors(&[t1.clone(), t2.clone()], &[t1.clone(), t2.clone()]));
}

#[test]
fn matches_tensors_validates_actual_sharding_bounds() {
    // Rank 2 tensor
    let rank2_shape = vec![Dim::Concrete(4), Dim::Concrete(16)];
    // Axis 2 is out of bounds for rank 2 tensor
    let bad_axis_col = make_test_tensor(rank2_shape.clone(), ShardLayout::ColShard { axis: 2 });
    let bad_axis_row = make_test_tensor(rank2_shape.clone(), ShardLayout::RowShard { axis: 2 });

    let col_rule = ShardingRule::new(
        &[ShardLayoutPattern::ColShard { axis: 2 }],
        &[ShardLayoutPattern::Replicated],
    );
    let row_rule = ShardingRule::new(
        &[ShardLayoutPattern::RowShard { axis: 2 }],
        &[ShardLayoutPattern::Replicated],
    );
    let rep_out = make_test_tensor(vec![Dim::Concrete(4)], ShardLayout::Replicated);

    // Out-of-bounds sharded axis must be rejected
    assert!(!col_rule.matches_tensors(&[bad_axis_col], std::slice::from_ref(&rep_out)));
    assert!(!row_rule.matches_tensors(&[bad_axis_row], std::slice::from_ref(&rep_out)));

    // Magic zero heads/experts must be rejected
    let zero_heads = make_test_tensor(rank2_shape.clone(), ShardLayout::HeadShard { heads: 0 });
    let head_rule = ShardingRule::new(
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
        &[ShardLayoutPattern::Replicated],
    );
    assert!(!head_rule.matches_tensors(&[zero_heads], std::slice::from_ref(&rep_out)));

    let zero_experts =
        make_test_tensor(rank2_shape.clone(), ShardLayout::ExpertShard { experts: 0 });
    let exp_rule = ShardingRule::new(
        &[ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Symbolic,
        }],
        &[ShardLayoutPattern::Replicated],
    );
    assert!(!exp_rule.matches_tensors(&[zero_experts], std::slice::from_ref(&rep_out)));
}

#[test]
fn matches_tensors_symbolic_head_and_expert_cardinalities_must_match() {
    let shape = vec![Dim::Concrete(1), Dim::Concrete(8), Dim::Concrete(64)];
    let q8 = make_test_tensor(shape.clone(), ShardLayout::HeadShard { heads: 8 });
    let o8 = make_test_tensor(shape.clone(), ShardLayout::HeadShard { heads: 8 });
    let o16 = make_test_tensor(shape.clone(), ShardLayout::HeadShard { heads: 16 });

    let attn_rule = ShardingRule::new(
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
    );

    // Matching symbolic head count (8 == 8) -> true
    assert!(attn_rule.matches_tensors(std::slice::from_ref(&q8), &[o8]));
    // Mismatched symbolic head count (8 != 16) -> false
    assert!(!attn_rule.matches_tensors(std::slice::from_ref(&q8), &[o16]));

    // Multi-input head shard rule on state_write_kv (k, v); slot_map is BatchMeta.
    let k8 = make_test_tensor(shape.clone(), ShardLayout::HeadShard { heads: 8 });
    let v8 = make_test_tensor(shape.clone(), ShardLayout::HeadShard { heads: 8 });
    let v4 = make_test_tensor(shape.clone(), ShardLayout::HeadShard { heads: 4 });

    let kv_rule = &r9v_ir::sharding::STATE_WRITE_KV_RULES[1];
    assert!(kv_rule.matches_tensors(&[k8.clone(), v8], &[]));
    assert!(!kv_rule.matches_tensors(&[k8, v4], &[]));

    // ExpertShard symbolic cardinality check
    let x16 = make_test_tensor(
        vec![Dim::Concrete(1), Dim::Concrete(64)],
        ShardLayout::ExpertShard { experts: 16 },
    );
    let counts = make_test_tensor(vec![Dim::Concrete(16)], ShardLayout::Replicated);
    let y16 = make_test_tensor(
        vec![Dim::Concrete(1), Dim::Concrete(64)],
        ShardLayout::ExpertShard { experts: 16 },
    );
    let y8 = make_test_tensor(
        vec![Dim::Concrete(1), Dim::Concrete(64)],
        ShardLayout::ExpertShard { experts: 8 },
    );

    let all_to_all_rule = ShardingRule::new(
        &[
            ShardLayoutPattern::ExpertShard {
                experts: ExpertCount::Symbolic,
            },
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Symbolic,
        }],
    );
    assert!(all_to_all_rule.matches_tensors(&[x16.clone(), counts.clone()], &[y16]));
    assert!(!all_to_all_rule.matches_tensors(&[x16.clone(), counts.clone()], &[y8]));
}
