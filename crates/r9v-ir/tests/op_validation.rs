// SPDX-License-Identifier: Apache-2.0
//! Exhaustive acceptance and rejection validation tests for all 32 ops in Spec 1 §4 (card A1.2, extended by A1.14).
//!
//! Tests verify structural shape, dtype, class, placement, and attribute rules,
//! checking that errors are strictly typed and that collect-all behavior surfaces
//! every independent problem when multiple violations coexist (CONVENTIONS.md §1.4).

use r9v_ir::{
    ActMulOp, ActivationKind, ActivationOp, AllGatherOp, AllReduceOp, AllToAllOp, AttentionMask,
    AttentionOp, BarrierOp, CacheScaleGranularity, CastOp, CausalConv1dOp, Class, ConvActivation,
    CopyKind, CopyOp, DType, Dim, EmbedGatherOp, Epilogue, GatherRowsOp, GroupId, HashId, IrError,
    LayoutId, LinearAttnKind, LinearAttnScanOp, LogitsPostprocessOp, MatmulOp, MlaAttentionSpec,
    MlaLatent, MoeFfnOp, MoeGroup, MoeRouteOp, MoeScoring, NgramCombine, NgramGatherOp,
    NgramSource, NormAxis, NormKind, NormOp, Numerics, Op, Placement, QuantActOp, QuantScheme,
    RecvOp, ReduceOp, ReduceScatterOp, ReductionOrder, ResidualAddOp, RngAlgorithm, RopeOp,
    RopeScaling, RopeStyle, SampleOp, ScatterAddRowsOp, SchemeId, SendOp, ShardLayout, Smoothing,
    StateHandle, StateKind, StateWriteKvOp, Tensor, VerifyMethod, VerifyOp,
};

fn act_tensor(shape: Vec<u32>, dtype: DType) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        dtype,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap()
}

fn weight_tensor(shape: Vec<u32>, dtype: DType) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        dtype,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap()
}

fn quant_weight_tensor(shape: Vec<u32>, dtype: DType, quant: QuantScheme) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        dtype,
        quant,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap()
}

fn device_weight_tensor(shape: Vec<u32>, dtype: DType, rank: u32) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        dtype,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap()
}

fn staging_tensor(shape: Vec<u32>, dtype: DType) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        dtype,
        QuantScheme::Scheme(SchemeId::new(1)),
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Staging,
    )
    .unwrap()
}

fn host_tensor(shape: Vec<u32>, dtype: DType) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        dtype,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Host,
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap()
}

fn param_tensor(shape: Vec<u32>, dtype: DType) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        dtype,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Param,
    )
    .unwrap()
}

fn bool_tensor(shape: Vec<u32>) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        DType::Bool,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap()
}

fn quant_act_tensor(shape: Vec<u32>, dtype: DType, quant: QuantScheme) -> Tensor {
    Tensor::new(
        shape.into_iter().map(Dim::Concrete).collect(),
        dtype,
        quant,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap()
}

fn assert_multiple_problems(err: IrError, min_count: usize) -> Box<[IrError]> {
    match err {
        IrError::Multiple { problems } => {
            assert!(
                problems.len() >= min_count,
                "expected at least {min_count} problems, got {}",
                problems.len()
            );
            problems
        }
        other => panic!("expected IrError::Multiple, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// §4.A Data movement and lookup
// -----------------------------------------------------------------------------

#[test]
fn embed_gather_accepts_valid_dimensions_and_dtypes() {
    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F16,
    };

    let token_ids = act_tensor(vec![128], DType::U32);
    let table = quant_weight_tensor(vec![32000, 4096], DType::I8, QuantScheme::PerRow);
    let x = act_tensor(vec![128, 4096], DType::F16);

    assert!(op.validate(&[token_ids, table], &[x]).is_ok());
}

#[test]
fn embed_gather_rejects_malformed_ranks_classes_and_attributes() {
    let op = EmbedGatherOp {
        scale: 1.0,
        out_dtype: DType::F16,
    };

    let token_ids = act_tensor(vec![128], DType::U32);
    let table = quant_weight_tensor(vec![32000, 4096], DType::I8, QuantScheme::PerRow);
    let x = act_tensor(vec![128, 4096], DType::F16);

    // 1. token_ids wrong dtype
    let bad_token_ids = act_tensor(vec![128], DType::F32);
    let err = op
        .validate(&[bad_token_ids, table.clone()], std::slice::from_ref(&x))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "embed_gather",
                tensor: "token_ids",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. table wrong rank
    let bad_table_rank = weight_tensor(vec![4096], DType::F16);
    let err = op
        .validate(
            &[token_ids.clone(), bad_table_rank],
            std::slice::from_ref(&x),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "embed_gather",
                tensor: "table",
                expected: 2,
                got: 1
            }
        ),
        "got: {err:?}"
    );

    // 3. table wrong class (Activation instead of Weight)
    let bad_table_class = act_tensor(vec![32000, 4096], DType::F16);
    let err = op
        .validate(
            &[token_ids.clone(), bad_table_class],
            std::slice::from_ref(&x),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpClassMismatch {
                op: "embed_gather",
                tensor: "table",
                expected: Class::Weight,
                got: Class::Activation
            }
        ),
        "got: {err:?}"
    );

    // 4. scale <= 0
    let bad_scale_op = EmbedGatherOp {
        scale: 0.0,
        out_dtype: DType::F16,
    };
    let err = bad_scale_op
        .validate(&[token_ids, table], &[x])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "embed_gather",
                attribute: "scale",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn embed_gather_exposes_collect_all_on_coexisting_violations() {
    let op = EmbedGatherOp {
        scale: -1.0,
        out_dtype: DType::I32,
    };

    let bad_token_ids = act_tensor(vec![128], DType::F32);
    let bad_table = act_tensor(vec![32000], DType::F32);
    let bad_x = act_tensor(vec![128, 4096], DType::F16);

    let err = op
        .validate(&[bad_token_ids, bad_table], &[bad_x])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "scale",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "out_dtype",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "token_ids",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpRankMismatch {
            tensor: "table",
            ..
        }
    )));
}

#[test]
fn ngram_gather_staged_accepts_one_dimensional_and_two_dimensional_scales() {
    let op = NgramGatherOp {
        source: NgramSource::Staged,
        orders: vec![2, 3].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(42),
        table_sizes: vec![1024, 1024].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F16,
    };

    let staging = staging_tensor(vec![128, 2, 64], DType::I8);
    let row_scales_2d = act_tensor(vec![128, 2], DType::F32);
    let row_scales_1d = act_tensor(vec![128], DType::F32);
    let x = act_tensor(vec![128, 128], DType::F16);

    assert!(op
        .validate(&[staging.clone(), row_scales_2d], std::slice::from_ref(&x))
        .is_ok());
    assert!(op.validate(&[staging, row_scales_1d], &[x]).is_ok());
}

#[test]
fn ngram_gather_device_accepts_direct_table_and_token_ids() {
    let op = NgramGatherOp {
        source: NgramSource::Device,
        orders: vec![2, 3].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(42),
        table_sizes: vec![1024, 1024].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F16,
    };

    let token_ids = act_tensor(vec![128], DType::U32);
    let table = weight_tensor(vec![2048, 64], DType::F16);
    let x = act_tensor(vec![128, 128], DType::F16);

    assert!(op.validate(&[token_ids, table], &[x]).is_ok());
}

#[test]
fn ngram_gather_staged_rejects_rank_head_dtype_and_quant_mismatches() {
    let op = NgramGatherOp {
        source: NgramSource::Staged,
        orders: vec![2, 3].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(42),
        table_sizes: vec![1024, 1024].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F16,
    };

    let staging = staging_tensor(vec![128, 2, 64], DType::I8);
    let row_scales = act_tensor(vec![128, 2], DType::F32);
    let x = act_tensor(vec![128, 128], DType::F16);

    // 1. staging rank mismatch
    let bad_staging_rank = staging_tensor(vec![128, 128], DType::I8);
    let err = op
        .validate(
            &[bad_staging_rank, row_scales.clone()],
            std::slice::from_ref(&x),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "ngram_gather",
                tensor: "gather_staging",
                expected: 3,
                got: 2
            }
        ),
        "got: {err:?}"
    );

    // 2. staging bytes must carry a block quantization scheme.
    let unquantized_staging = Tensor::new(
        vec![Dim::Concrete(128), Dim::Concrete(2), Dim::Concrete(64)],
        DType::I8,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Staging,
    )
    .unwrap();
    let row_scales = act_tensor(vec![128, 2], DType::F32);
    let x = act_tensor(vec![128, 128], DType::F16);
    let err = op
        .validate(&[unquantized_staging, row_scales], std::slice::from_ref(&x))
        .unwrap_err();
    assert!(matches!(
        err,
        IrError::OpQuantMismatch {
            op: "ngram_gather",
            tensor: "gather_staging",
            ..
        }
    ));

    // 3. staging heads mismatch (row_scales also matching 4 heads to isolate staging heads violation)
    let bad_staging_heads = staging_tensor(vec![128, 4, 64], DType::I8);
    let row_scales_4 = act_tensor(vec![128, 4], DType::F32);
    let err = op
        .validate(&[bad_staging_heads, row_scales_4], std::slice::from_ref(&x))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "ngram_gather",
                tensor: "gather_staging",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 4. row_scales wrong dtype
    let bad_scales_dtype = act_tensor(vec![128, 2], DType::I32);
    let err = op.validate(&[staging, bad_scales_dtype], &[x]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "ngram_gather",
                tensor: "row_scales",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn ngram_gather_device_rejects_rank_class_and_placement_mismatches() {
    let op = NgramGatherOp {
        source: NgramSource::Device,
        orders: vec![2, 3].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(42),
        table_sizes: vec![1024, 1024].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F16,
    };

    let token_ids = act_tensor(vec![128], DType::U32);
    let table = weight_tensor(vec![2048, 64], DType::F16);
    let x = act_tensor(vec![128, 128], DType::F16);

    // 1. token_ids wrong rank
    let bad_token_ids = act_tensor(vec![128, 1], DType::U32);
    let err = op
        .validate(&[bad_token_ids, table.clone()], std::slice::from_ref(&x))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "ngram_gather",
                tensor: "token_ids",
                expected: 1,
                got: 2
            }
        ),
        "got: {err:?}"
    );

    // 2. table wrong class (Activation instead of Weight)
    let bad_table_class = act_tensor(vec![2048, 64], DType::F16);
    let err = op
        .validate(&[token_ids, bad_table_class], &[x])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpClassMismatch {
                op: "ngram_gather",
                tensor: "table",
                expected: Class::Weight,
                got: Class::Activation
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn ngram_gather_exposes_collect_all_on_coexisting_violations() {
    let op = NgramGatherOp {
        source: NgramSource::Staged,
        orders: vec![2, 3].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(42),
        table_sizes: vec![1024, 1024].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F16,
    };

    let bad_staging = act_tensor(vec![128, 4, 64], DType::F32);
    let bad_scales = act_tensor(vec![128, 2, 1], DType::I32);
    let bad_x = act_tensor(vec![128, 128], DType::F32);

    let err = op
        .validate(&[bad_staging, bad_scales], &[bad_x])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "gather_staging",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpClassMismatch {
            tensor: "gather_staging",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpRankMismatch {
            tensor: "row_scales",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
}

#[test]
fn quant_act_accepts_per_token_i8_fp8_and_per_block32_i8() {
    let x = act_tensor(vec![128, 4096], DType::F16);

    // 1. PerToken with I8 target
    let op_token_i8 = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };
    let xq_i8 = quant_act_tensor(vec![128, 4096], DType::I8, QuantScheme::PerToken);
    let scale_1d = act_tensor(vec![128], DType::F32);
    assert!(op_token_i8
        .validate(std::slice::from_ref(&x), &[xq_i8, scale_1d.clone()])
        .is_ok());

    // 2. PerToken with E4m3 target
    let op_token_fp8 = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::E4m3,
        smoothing: Smoothing::Folded,
    };
    let xq_fp8 = quant_act_tensor(vec![128, 4096], DType::E4m3, QuantScheme::PerToken);
    assert!(op_token_fp8
        .validate(std::slice::from_ref(&x), &[xq_fp8, scale_1d])
        .is_ok());

    // 3. PerBlock32 with I8 target
    let op_block_i8 = QuantActOp {
        scheme: QuantScheme::PerBlock32,
        target: DType::I8,
        smoothing: Smoothing::None,
    };
    let xq_block_i8 = quant_act_tensor(vec![128, 4096], DType::I8, QuantScheme::PerBlock32);
    let scale_2d = act_tensor(vec![128, 128], DType::F32);
    assert!(op_block_i8.validate(&[x], &[xq_block_i8, scale_2d]).is_ok());
}

#[test]
fn quant_act_rejects_per_block32_fp8_and_target_scale_mismatches() {
    let x = act_tensor(vec![128, 4096], DType::F16);
    let xq_i8 = quant_act_tensor(vec![128, 4096], DType::I8, QuantScheme::PerToken);
    let scale_1d = act_tensor(vec![128], DType::F32);

    // 1. PerBlock32 with E4m3 target rejected
    let bad_block_fp8 = QuantActOp {
        scheme: QuantScheme::PerBlock32,
        target: DType::E4m3,
        smoothing: Smoothing::None,
    };
    let xq_fp8 = quant_act_tensor(vec![128, 4096], DType::E4m3, QuantScheme::PerToken);
    let scale_2d = act_tensor(vec![128, 128], DType::F32);
    let err = bad_block_fp8
        .validate(std::slice::from_ref(&x), &[xq_fp8, scale_2d])
        .unwrap_err();
    let problems = match err {
        IrError::Multiple { problems } => problems,
        single => vec![single].into_boxed_slice(),
    };
    assert!(
        problems.iter().any(|p| matches!(
            p,
            IrError::OpAttributeInvalid {
                op: "quant_act",
                attribute: "target",
                ..
            }
        )),
        "got: {problems:?}"
    );

    // 2. Invalid target dtype (F32)
    let bad_target = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::F32,
        smoothing: Smoothing::None,
    };
    let err = bad_target
        .validate(std::slice::from_ref(&x), &[xq_i8.clone(), scale_1d.clone()])
        .unwrap_err();
    let problems = match err {
        IrError::Multiple { problems } => problems,
        single => vec![single].into_boxed_slice(),
    };
    assert!(
        problems.iter().any(|p| matches!(
            p,
            IrError::OpAttributeInvalid {
                op: "quant_act",
                attribute: "target",
                ..
            }
        )),
        "got: {problems:?}"
    );

    // 3. Scale rank mismatch for PerToken (2D instead of 1D)
    let op_token_i8 = QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: Smoothing::None,
    };
    let bad_scale_rank = act_tensor(vec![128, 1], DType::F32);
    let err = op_token_i8
        .validate(&[x], &[xq_i8, bad_scale_rank])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "quant_act",
                tensor: "scale",
                expected: 1,
                got: 2
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn quant_act_exposes_collect_all_on_coexisting_violations() {
    let op = QuantActOp {
        scheme: QuantScheme::PerRow,
        target: DType::F32,
        smoothing: Smoothing::None,
    };

    let x = act_tensor(vec![128, 4096], DType::F16);
    let bad_xq = act_tensor(vec![128, 4096], DType::F16);
    let bad_scale = act_tensor(vec![128, 1], DType::I32);

    let err = op.validate(&[x], &[bad_xq, bad_scale]).unwrap_err();
    let problems = assert_multiple_problems(err, 3);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "scheme",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "target",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "scale",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpRankMismatch {
            tensor: "scale",
            ..
        }
    )));
}

#[test]
fn cast_accepts_matching_shapes_with_target_dtype() {
    let op = CastOp { dtype: DType::Bf16 };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::Bf16);

    assert!(op.validate(&[x], &[y]).is_ok());

    for shape in [vec![8], vec![2, 3, 4], vec![2, 3, 4, 5]] {
        let x = act_tensor(shape.clone(), DType::F16);
        let y = act_tensor(shape, DType::Bf16);
        assert!(op.validate(&[x], &[y]).is_ok());
    }
}

#[test]
fn cast_rejects_dtype_and_shape_mismatches() {
    let op = CastOp { dtype: DType::Bf16 };
    let x = act_tensor(vec![128, 4096], DType::F16);

    // 1. Output dtype mismatch
    let bad_y_dtype = act_tensor(vec![128, 4096], DType::F32);
    let err = op
        .validate(std::slice::from_ref(&x), &[bad_y_dtype])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "cast",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. Shape mismatch
    let bad_y_shape = act_tensor(vec![128, 2048], DType::Bf16);
    let err = op
        .validate(std::slice::from_ref(&x), &[bad_y_shape])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "cast",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. Input count mismatch
    let err = op.validate(&[x.clone(), x], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::Multiple { .. } | IrError::OpInputCountMismatch { .. }
        ),
        "got: {err:?}"
    );
}

#[test]
fn cast_exposes_collect_all_on_coexisting_violations() {
    let op = CastOp { dtype: DType::Bf16 };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let bad_y = device_weight_tensor(vec![128, 2048, 1], DType::F32, 1);

    let err = op.validate(&[x], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpRankMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpPlacementMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "y", .. })));
}

#[test]
fn copy_accepts_contiguize_d2d_h2d_and_d2h_boundaries() {
    let x_dev0 = act_tensor(vec![128, 4096], DType::F16);
    let y_dev0 = act_tensor(vec![128, 4096], DType::F16);
    let y_dev1 = Tensor::new(
        vec![Dim::Concrete(128), Dim::Concrete(4096)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 1 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();

    let x_host = host_tensor(vec![128, 4096], DType::F16);
    let y_host_dev = device_weight_tensor(vec![128, 4096], DType::F16, 0);

    // 1. Contiguize
    let op_contig = CopyOp {
        kind: CopyKind::Contiguize,
    };
    assert!(op_contig
        .validate(std::slice::from_ref(&x_dev0), &[y_dev0])
        .is_ok());

    // 2. DeviceToDevice
    let op_d2d = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };
    assert!(op_d2d
        .validate(std::slice::from_ref(&x_dev0), &[y_dev1])
        .is_ok());

    // 3. HostToDevice
    let op_h2d = CopyOp {
        kind: CopyKind::HostToDevice,
    };
    assert!(op_h2d
        .validate(
            std::slice::from_ref(&x_host),
            std::slice::from_ref(&y_host_dev)
        )
        .is_ok());

    // 4. DeviceToHost
    let op_d2h = CopyOp {
        kind: CopyKind::DeviceToHost,
    };
    assert!(op_d2h.validate(&[y_host_dev], &[x_host]).is_ok());
}

#[test]
fn copy_rejects_placement_class_and_dtype_mismatches() {
    let x_dev0 = act_tensor(vec![128, 4096], DType::F16);
    let x_host = host_tensor(vec![128, 4096], DType::F16);

    // 1. Contiguize across placements
    let op_contig = CopyOp {
        kind: CopyKind::Contiguize,
    };
    let y_dev1 = Tensor::new(
        vec![Dim::Concrete(128), Dim::Concrete(4096)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 1 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap();
    let err = op_contig
        .validate(std::slice::from_ref(&x_dev0), &[y_dev1])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpPlacementMismatch {
                op: "copy",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. DeviceToDevice with same rank
    let op_d2d = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };
    let err = op_d2d
        .validate(std::slice::from_ref(&x_dev0), std::slice::from_ref(&x_dev0))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpPlacementMismatch {
                op: "copy",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. Class mismatch (Weight host to Activation device)
    let op_h2d = CopyOp {
        kind: CopyKind::HostToDevice,
    };
    let err = op_h2d
        .validate(std::slice::from_ref(&x_host), std::slice::from_ref(&x_dev0))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpClassMismatch {
                op: "copy",
                tensor: "y",
                expected: Class::Weight,
                got: Class::Activation
            }
        ),
        "got: {err:?}"
    );

    // 4. Output dtype mismatch
    let bad_dtype_dev0 = act_tensor(vec![128, 4096], DType::F32);
    let err = op_contig
        .validate(&[x_dev0], &[bad_dtype_dev0])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "copy",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn copy_exposes_collect_all_on_coexisting_violations() {
    let op = CopyOp {
        kind: CopyKind::DeviceToDevice,
    };
    let x_host = host_tensor(vec![128, 4096], DType::F16);
    let bad_y = act_tensor(vec![128, 2048], DType::F32);

    let err = op.validate(&[x_host], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpPlacementMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpShapeMismatch { tensor: "y", .. })));
}

#[test]
fn gather_rows_accepts_valid_indices_and_activation_table() {
    let op = GatherRowsOp;
    let x = act_tensor(vec![1000, 128], DType::F16);
    let indices = act_tensor(vec![50], DType::U32);
    let y = act_tensor(vec![50, 128], DType::F16);

    assert!(op.validate(&[x, indices], &[y]).is_ok());
}

#[test]
fn gather_rows_rejects_dtype_rank_and_shape_mismatches() {
    let op = GatherRowsOp;
    let x = act_tensor(vec![1000, 128], DType::F16);
    let indices = act_tensor(vec![50], DType::U32);
    let y = act_tensor(vec![50, 128], DType::F16);

    // 1. indices not U32
    let bad_indices = act_tensor(vec![50], DType::F32);
    let err = op
        .validate(&[x.clone(), bad_indices], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "gather_rows",
                tensor: "indices",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. y shape mismatch
    let bad_y = act_tensor(vec![60, 128], DType::F16);
    let err = op
        .validate(&[x.clone(), indices.clone()], &[bad_y])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "gather_rows",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. x class != Activation
    let bad_x_class = weight_tensor(vec![1000, 128], DType::F16);
    let err = op.validate(&[bad_x_class, indices], &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpClassMismatch {
                op: "gather_rows",
                tensor: "x",
                expected: Class::Activation,
                got: Class::Weight
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn gather_rows_exposes_collect_all_on_coexisting_violations() {
    let op = GatherRowsOp;
    let bad_x = weight_tensor(vec![1000, 128], DType::I32);
    let bad_indices = act_tensor(vec![50, 1], DType::F32);
    let bad_y = act_tensor(vec![60, 128], DType::F32);

    let err = op.validate(&[bad_x, bad_indices], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "x", .. })));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpRankMismatch {
            tensor: "indices",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "indices",
            ..
        }
    )));
}

#[test]
fn scatter_add_rows_accepts_two_and_three_input_arities() {
    let op = ScatterAddRowsOp;
    let x = act_tensor(vec![50, 128], DType::F16);
    let indices = act_tensor(vec![50], DType::U32);
    let dest = act_tensor(vec![1000, 128], DType::F16);
    let y = act_tensor(vec![1000, 128], DType::F16);

    // 1. Two inputs (without dest)
    assert!(op
        .validate(&[x.clone(), indices.clone()], std::slice::from_ref(&y))
        .is_ok());

    // 2. Three inputs (with dest)
    assert!(op.validate(&[x, indices, dest], &[y]).is_ok());
}

#[test]
fn scatter_add_rows_rejects_input_count_shape_and_dtype_violations() {
    let op = ScatterAddRowsOp;
    let x = act_tensor(vec![50, 128], DType::F16);
    let indices = act_tensor(vec![50], DType::U32);
    let dest = act_tensor(vec![1000, 128], DType::F16);
    let y = act_tensor(vec![1000, 128], DType::F16);

    // 1. Input count candidates mismatch (1 input)
    let err = op
        .validate(std::slice::from_ref(&x), std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpInputCountCandidatesMismatch {
                op: "scatter_add_rows",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. indices length mismatch with x
    let bad_indices = act_tensor(vec![40], DType::U32);
    let err = op
        .validate(
            &[x.clone(), bad_indices, dest.clone()],
            std::slice::from_ref(&y),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "scatter_add_rows",
                tensor: "x",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. dest shape mismatch with y
    let bad_dest = act_tensor(vec![900, 128], DType::F16);
    let err = op
        .validate(&[x.clone(), indices, bad_dest], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "scatter_add_rows",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 4. y dtype mismatch with x
    let bad_y_dtype = act_tensor(vec![1000, 128], DType::F32);
    let err = op.validate(&[x.clone(), dest], &[bad_y_dtype]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::Multiple { .. } | IrError::OpDTypeMismatch { .. }
        ),
        "got: {err:?}"
    );
}

#[test]
fn scatter_add_rows_exposes_collect_all_on_coexisting_violations() {
    let op = ScatterAddRowsOp;
    let x = act_tensor(vec![50, 128], DType::F16);
    let bad_indices = act_tensor(vec![40], DType::F32);
    let bad_dest = act_tensor(vec![900, 64], DType::F32);
    let bad_y = act_tensor(vec![1000, 128], DType::F32);

    let err = op
        .validate(&[x, bad_indices, bad_dest], &[bad_y])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "indices",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "dest", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpShapeMismatch { tensor: "x", .. })));
}

// -----------------------------------------------------------------------------
// §4.B Normalization and elementwise
// -----------------------------------------------------------------------------

#[test]
fn norm_accepts_two_and_three_input_arities_and_head_axis() {
    let op_last = NormOp {
        kind: NormKind::Rms,
        eps: 1e-5,
        axis: NormAxis::Last,
        weight_offset: 0.0,
        out_dtype: DType::F16,
    };

    let x = act_tensor(vec![128, 4096], DType::F16);
    let weight = weight_tensor(vec![4096], DType::F32);
    let bias = param_tensor(vec![4096], DType::F32);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. Two inputs (without bias)
    assert!(op_last
        .validate(&[x.clone(), weight.clone()], std::slice::from_ref(&y))
        .is_ok());

    // 2. Three inputs (with bias)
    assert!(op_last
        .validate(&[x.clone(), weight.clone(), bias], std::slice::from_ref(&y))
        .is_ok());

    // 3. Head axis where 4096 is divisible by 64
    let op_head = NormOp {
        axis: NormAxis::Head(64),
        ..op_last
    };
    assert!(op_head.validate(&[x, weight], &[y]).is_ok());
}

#[test]
fn norm_rejects_eps_weight_rank_and_head_divisibility_violations() {
    let op = NormOp {
        kind: NormKind::Layer,
        eps: 1e-5,
        axis: NormAxis::Last,
        weight_offset: 0.0,
        out_dtype: DType::F16,
    };

    let x = act_tensor(vec![128, 4096], DType::F16);
    let weight = weight_tensor(vec![4096], DType::F32);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. eps <= 0
    let bad_eps = NormOp { eps: 0.0, ..op };
    let err = bad_eps
        .validate(&[x.clone(), weight.clone()], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "norm",
                attribute: "eps",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. weight rank != 1
    let bad_weight = weight_tensor(vec![4096, 1], DType::F32);
    let err = op
        .validate(&[x.clone(), bad_weight], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "norm",
                tensor: "weight",
                expected: 1,
                got: 2
            }
        ),
        "got: {err:?}"
    );

    // 3. Head axis indivisible: 4096 not divisible by 50
    let bad_head_op = NormOp {
        axis: NormAxis::Head(50),
        ..op
    };
    let err = bad_head_op
        .validate(&[x.clone(), weight], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "norm",
                tensor: "x",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 4. A zero head dimension is rejected without reaching modulo arithmetic.
    let zero_head_op = NormOp {
        axis: NormAxis::Head(0),
        ..op
    };
    let err = zero_head_op
        .validate(
            &[x.clone(), weight_tensor(vec![4096], DType::F32)],
            std::slice::from_ref(&y),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        IrError::OpAttributeInvalid {
            op: "norm",
            attribute: "axis",
            ..
        }
    ));

    // 5. Output dtype mismatch
    let bad_y_dtype = act_tensor(vec![128, 4096], DType::F32);
    let err = op.validate(&[x], &[bad_y_dtype]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::Multiple { .. } | IrError::OpDTypeMismatch { .. }
        ),
        "got: {err:?}"
    );
}

#[test]
fn norm_exposes_collect_all_on_coexisting_violations() {
    let op = NormOp {
        kind: NormKind::Rms,
        eps: -1.0,
        axis: NormAxis::Head(0),
        weight_offset: f32::NAN,
        out_dtype: DType::U32,
    };

    let bad_x = act_tensor(vec![128, 4096], DType::I32);
    let bad_weight = act_tensor(vec![4096, 1], DType::F16);
    let bad_y = act_tensor(vec![128, 4096], DType::F32);

    let err = op.validate(&[bad_x, bad_weight], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 5);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "eps",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "axis",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "weight_offset",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "out_dtype",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
}

#[test]
fn residual_add_accepts_matching_activation_operands() {
    let op = ResidualAddOp {
        out_dtype: DType::F16,
        scale: 1.0,
    };
    let a = act_tensor(vec![128, 4096], DType::F16);
    let b = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    assert!(op.validate(&[a, b], &[y]).is_ok());
}

#[test]
fn residual_add_rejects_shape_dtype_and_class_mismatches() {
    let op = ResidualAddOp {
        out_dtype: DType::F16,
        scale: 1.0,
    };
    let a = act_tensor(vec![128, 4096], DType::F16);
    let b = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. Shape mismatch
    let bad_b_shape = act_tensor(vec![128, 2048], DType::F16);
    let err = op
        .validate(&[a.clone(), bad_b_shape], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "residual_add",
                tensor: "b",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. Class mismatch
    let bad_b_class = weight_tensor(vec![128, 4096], DType::F16);
    let err = op
        .validate(&[a.clone(), bad_b_class], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpClassMismatch {
                op: "residual_add",
                tensor: "b",
                expected: Class::Activation,
                got: Class::Weight
            }
        ),
        "got: {err:?}"
    );

    // 3. Output dtype mismatch
    let bad_y_dtype = act_tensor(vec![128, 4096], DType::F32);
    let err = op.validate(&[a, b], &[bad_y_dtype]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "residual_add",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn residual_add_exposes_collect_all_on_coexisting_violations() {
    let op = ResidualAddOp {
        out_dtype: DType::F16,
        scale: 1.0,
    };
    let bad_a = act_tensor(vec![128, 4096], DType::U32);
    let bad_b = weight_tensor(vec![128, 2048], DType::F32);
    let bad_y = act_tensor(vec![128, 4096], DType::F32);

    let err = op.validate(&[bad_a, bad_b], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "a", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "b", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpShapeMismatch { tensor: "b", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
}

#[test]
fn act_mul_accepts_matching_gate_and_up_projections() {
    let op = ActMulOp {
        act: ActivationKind::Silu,
        clamp: Some(10.0),
    };
    let gate = act_tensor(vec![128, 4096], DType::F16);
    let up = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    assert!(op.validate(&[gate, up], &[y]).is_ok());
}

#[test]
fn act_mul_rejects_dimension_mismatch_and_non_positive_clamp() {
    let op = ActMulOp {
        act: ActivationKind::Gelu,
        clamp: None,
    };
    let gate = act_tensor(vec![128, 4096], DType::F16);
    let up = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. Gate and up shape mismatch
    let bad_up = act_tensor(vec![128, 2048], DType::F16);
    let err = op
        .validate(&[gate.clone(), bad_up], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "act_mul",
                tensor: "up",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. Clamp <= 0.0
    let bad_clamp_op = ActMulOp {
        clamp: Some(0.0),
        ..op
    };
    let err = bad_clamp_op.validate(&[gate, up], &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "act_mul",
                attribute: "clamp",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn act_mul_exposes_collect_all_on_coexisting_violations() {
    let op = ActMulOp {
        act: ActivationKind::GeluTanh,
        clamp: Some(-1.0),
    };
    let bad_gate = act_tensor(vec![128, 4096], DType::I32);
    let bad_up = act_tensor(vec![128, 2048], DType::F16);
    let bad_y = act_tensor(vec![128, 4096], DType::F32);

    let err = op.validate(&[bad_gate, bad_up], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "clamp",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "gate", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpShapeMismatch { tensor: "up", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
}

#[test]
fn activation_accepts_elementwise_transforms_with_optional_clamp() {
    let op_plain = ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    };
    let op_clamped = ActivationOp {
        act: ActivationKind::Relu2,
        clamp: Some(6.0),
    };

    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    assert!(op_plain
        .validate(std::slice::from_ref(&x), std::slice::from_ref(&y))
        .is_ok());
    assert!(op_clamped.validate(&[x], &[y]).is_ok());
}

#[test]
fn activation_rejects_output_rank_and_invalid_clamp() {
    let op = ActivationOp {
        act: ActivationKind::Gelu,
        clamp: None,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);

    // 1. Output rank mismatch
    let bad_y_rank = act_tensor(vec![128, 4096, 1], DType::F16);
    let err = op
        .validate(std::slice::from_ref(&x), &[bad_y_rank])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "activation",
                tensor: "y",
                expected: 2,
                got: 3
            }
        ),
        "got: {err:?}"
    );

    // 2. Non-positive clamp
    let bad_clamp = ActivationOp {
        clamp: Some(0.0),
        ..op
    };
    let y = act_tensor(vec![128, 4096], DType::F16);
    let err = bad_clamp.validate(&[x], &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "activation",
                attribute: "clamp",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn activation_exposes_collect_all_on_coexisting_violations() {
    let op = ActivationOp {
        act: ActivationKind::Identity,
        clamp: Some(-5.0),
    };
    let bad_x = act_tensor(vec![128, 4096], DType::U32);
    let bad_y = act_tensor(vec![128, 2048], DType::F32);

    let err = op.validate(&[bad_x], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "clamp",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpShapeMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
}

#[test]
fn rope_accepts_one_dimensional_standard_and_two_dimensional_mrope_positions() {
    let x = act_tensor(vec![128, 32, 128], DType::F16);
    let y = act_tensor(vec![128, 32, 128], DType::F16);

    // 1. Standard 1D positions
    let op_std = RopeOp {
        rot_dim: 64,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F16,
    };
    let pos_1d = act_tensor(vec![128], DType::U32);
    assert!(op_std
        .validate(&[x.clone(), pos_1d], std::slice::from_ref(&y))
        .is_ok());

    // 2. MRoPE 2D positions with 3 coordinates
    let op_mrope = RopeOp {
        rot_dim: 64,
        theta: 10000.0,
        style: RopeStyle::Interleaved,
        scaling: RopeScaling::None,
        mrope_sections: Some([16, 24, 24]),
        out_dtype: DType::F16,
    };
    let pos_2d = act_tensor(vec![128, 3], DType::U32);
    assert!(op_mrope.validate(&[x, pos_2d], &[y]).is_ok());
}

#[test]
fn rope_rejects_odd_rot_dim_and_positions_rank_or_dtype_violations() {
    let op = RopeOp {
        rot_dim: 64,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F16,
    };

    let x = act_tensor(vec![128, 32, 128], DType::F16);
    let positions = act_tensor(vec![128], DType::U32);
    let y = act_tensor(vec![128, 32, 128], DType::F16);

    // 1. rot_dim odd
    let bad_rot = RopeOp { rot_dim: 65, ..op };
    let err = bad_rot
        .validate(&[x.clone(), positions.clone()], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "rope",
                attribute: "rot_dim",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. positions wrong dtype
    let bad_pos_dtype = act_tensor(vec![128], DType::F32);
    let err = op
        .validate(&[x.clone(), bad_pos_dtype], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "rope",
                tensor: "positions",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. MRoPE with width != 3
    let bad_mrope = RopeOp {
        mrope_sections: Some([16, 24, 24]),
        ..op
    };
    let bad_pos_width = act_tensor(vec![128, 4], DType::U32);
    let err = bad_mrope.validate(&[x, bad_pos_width], &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "rope",
                tensor: "positions",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn rope_exposes_collect_all_on_coexisting_violations() {
    let op = RopeOp {
        rot_dim: 65,
        theta: -100.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::Linear(0.0),
        mrope_sections: None,
        out_dtype: DType::U32,
    };

    let bad_x = act_tensor(vec![128, 32, 128], DType::I32);
    let bad_pos = act_tensor(vec![128], DType::F32);
    let bad_y = act_tensor(vec![128, 32, 128], DType::F32);

    let err = op.validate(&[bad_x, bad_pos], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 6);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "rot_dim",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "theta",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "scaling.factor",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "out_dtype",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "positions",
            ..
        }
    )));
}

// -----------------------------------------------------------------------------
// §4.C Matmul family
// -----------------------------------------------------------------------------

#[test]
fn matmul_accepts_none_act_bias_and_residual_epilogues() {
    let x = act_tensor(vec![128, 4096], DType::F16);
    let w = weight_tensor(vec![2048, 4096], DType::F16);
    let y = act_tensor(vec![128, 2048], DType::F16);

    // 1. None epilogue (2 inputs)
    let op_none = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    assert!(op_none
        .validate(&[x.clone(), w.clone()], std::slice::from_ref(&y))
        .is_ok());

    // 2. Act epilogue (2 inputs)
    let op_act = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::Act(ActivationKind::Silu),
        transpose_w: false,
    };
    assert!(op_act
        .validate(&[x.clone(), w.clone()], std::slice::from_ref(&y))
        .is_ok());

    // 3. Bias epilogue (3 inputs)
    let op_bias = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::Bias,
        transpose_w: false,
    };
    let bias = param_tensor(vec![2048], DType::F32);
    assert!(op_bias
        .validate(&[x.clone(), w.clone(), bias], std::slice::from_ref(&y))
        .is_ok());

    // 4. Residual epilogue (3 inputs)
    let op_res = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::Residual,
        transpose_w: false,
    };
    let residual = act_tensor(vec![128, 2048], DType::F16);
    assert!(op_res.validate(&[x, w, residual], &[y]).is_ok());
}

#[test]
fn matmul_rejects_inner_k_mismatch_and_epilogue_signature_violations() {
    let op_none = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        transpose_w: false,
    };

    let x = act_tensor(vec![128, 4096], DType::F16);
    let w = weight_tensor(vec![2048, 4096], DType::F16);
    let y = act_tensor(vec![128, 2048], DType::F16);

    // 1. Inner K mismatch
    let bad_w_k = weight_tensor(vec![2048, 2048], DType::F16);
    let err = op_none
        .validate(&[x.clone(), bad_w_k], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "matmul",
                tensor: "w",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. Bias epilogue missing bias input
    let op_bias = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::Bias,
        transpose_w: false,
    };
    let err = op_bias
        .validate(&[x.clone(), w.clone()], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpInputCountMismatch {
                op: "matmul",
                expected: 3,
                got: 2
            }
        ),
        "got: {err:?}"
    );

    // 3. Bias epilogue with non-Param class
    let bad_bias_class = weight_tensor(vec![2048], DType::F32);
    let err = op_bias
        .validate(
            &[x.clone(), w.clone(), bad_bias_class],
            std::slice::from_ref(&y),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpClassMismatch {
                op: "matmul",
                tensor: "bias",
                expected: Class::Param,
                got: Class::Weight
            }
        ),
        "got: {err:?}"
    );

    // 4. None epilogue with extraneous third input
    let extra = act_tensor(vec![128, 2048], DType::F16);
    let err = op_none.validate(&[x, w, extra], &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpInputCountMismatch {
                op: "matmul",
                expected: 2,
                got: 3
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn matmul_exposes_collect_all_on_coexisting_violations() {
    let op = MatmulOp {
        out_dtype: DType::I32,
        epilogue: Epilogue::Bias,
        transpose_w: false,
    };

    let bad_x = weight_tensor(vec![128, 4096], DType::F16);
    let bad_w = act_tensor(vec![2048, 2048], DType::F16);
    let bad_bias = act_tensor(vec![1024], DType::F16);
    let bad_y = act_tensor(vec![128, 2048], DType::F32);

    let err = op
        .validate(&[bad_x, bad_w, bad_bias], &[bad_y])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 6);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "out_dtype",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "w", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "bias", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpShapeMismatch { tensor: "w", .. })));
}

#[test]
fn matmul_rejects_invalid_gemm_operands_and_dense_host_weights() {
    let op = MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        transpose_w: false,
    };
    let y = act_tensor(vec![8, 16], DType::F16);

    let bad_x = act_tensor(vec![8, 32], DType::F32);
    let bad_w = weight_tensor(vec![16, 32], DType::Bf16);
    let problems = assert_multiple_problems(
        op.validate(&[bad_x, bad_w], std::slice::from_ref(&y))
            .unwrap_err(),
        2,
    );
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "w", .. })));

    let x = act_tensor(vec![8, 32], DType::F16);
    let host_w = host_tensor(vec![16, 32], DType::F16);
    assert!(matches!(
        op.validate(&[x, host_w], std::slice::from_ref(&y))
            .unwrap_err(),
        IrError::OpPlacementMismatch {
            op: "matmul",
            tensor: "w",
            ..
        }
    ));

    let unquantized_x = act_tensor(vec![8, 32], DType::I8);
    let w = weight_tensor(vec![16, 32], DType::F16);
    assert!(matches!(
        op.validate(&[unquantized_x, w], std::slice::from_ref(&y))
            .unwrap_err(),
        IrError::OpQuantMismatch {
            op: "matmul",
            tensor: "x",
            ..
        }
    ));

    let x = act_tensor(vec![8, 32], DType::F16);
    let unquantized_w = weight_tensor(vec![16, 32], DType::I8);
    assert!(matches!(
        op.validate(&[x, unquantized_w], &[y]).unwrap_err(),
        IrError::OpQuantMismatch {
            op: "matmul",
            tensor: "w",
            ..
        }
    ));
}

#[test]
fn moe_route_accepts_one_and_two_input_signatures() {
    let op = MoeRouteOp {
        top_k: 2,
        scoring: MoeScoring::Softmax,
        renormalize: true,
        group: Some(MoeGroup {
            n_group: 2,
            topk_group: 1,
        }),
        scale: 1.0,
    };

    let logits = act_tensor(vec![128, 8], DType::F32);
    let bias = param_tensor(vec![8], DType::F32);
    let expert_ids = act_tensor(vec![128, 2], DType::U32);
    let weights = act_tensor(vec![128, 2], DType::F32);

    // 1. One input (without bias)
    assert!(op
        .validate(
            std::slice::from_ref(&logits),
            &[expert_ids.clone(), weights.clone()]
        )
        .is_ok());

    // 2. Two inputs (with bias)
    assert!(op.validate(&[logits, bias], &[expert_ids, weights]).is_ok());
}

#[test]
fn moe_route_rejects_zero_top_k_and_weight_dtype_mismatches() {
    let op = MoeRouteOp {
        top_k: 2,
        scoring: MoeScoring::Sigmoid,
        renormalize: false,
        group: None,
        scale: 1.0,
    };

    let logits = act_tensor(vec![128, 8], DType::F32);
    let expert_ids = act_tensor(vec![128, 2], DType::U32);
    let weights = act_tensor(vec![128, 2], DType::F32);

    // 1. top_k == 0 (collect-all surfaces OpAttributeInvalid and OpShapeMismatch against expert_ids)
    let bad_k = MoeRouteOp { top_k: 0, ..op };
    let err = bad_k
        .validate(
            std::slice::from_ref(&logits),
            &[expert_ids.clone(), weights.clone()],
        )
        .unwrap_err();
    let problems = match err {
        IrError::Multiple { problems } => problems,
        single => vec![single].into_boxed_slice(),
    };
    assert!(
        problems.iter().any(|p| matches!(
            p,
            IrError::OpAttributeInvalid {
                op: "moe_route",
                attribute: "top_k",
                ..
            }
        )),
        "got: {problems:?}"
    );

    // 2. weights wrong dtype
    let bad_weights = act_tensor(vec![128, 2], DType::U32);
    let err = op
        .validate(
            std::slice::from_ref(&logits),
            &[expert_ids.clone(), bad_weights],
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "moe_route",
                tensor: "weights",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. group topk_group > top_k
    let bad_group = MoeRouteOp {
        group: Some(MoeGroup {
            n_group: 2,
            topk_group: 3,
        }),
        ..op
    };
    let err = bad_group
        .validate(&[logits], &[expert_ids, weights])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "moe_route",
                attribute: "group.topk_group",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn moe_route_exposes_collect_all_on_coexisting_violations() {
    let op = MoeRouteOp {
        top_k: 0,
        scoring: MoeScoring::Softmax,
        renormalize: true,
        group: None,
        scale: -1.0,
    };

    let bad_logits = act_tensor(vec![128, 8], DType::I32);
    let bad_bias = act_tensor(vec![8], DType::F16);
    let bad_expert_ids = act_tensor(vec![128, 2], DType::F32);
    let bad_weights = act_tensor(vec![128, 2], DType::I32);

    let err = op
        .validate(&[bad_logits, bad_bias], &[bad_expert_ids, bad_weights])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 6);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "top_k",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "scale",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "logits",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "bias", .. })));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "expert_ids",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "weights",
            ..
        }
    )));
}

#[test]
fn moe_ffn_accepts_grouped_expert_gemm_operands() {
    let op = MoeFfnOp {
        act: ActivationKind::Silu,
        out_dtype: DType::F16,
        shared_experts: 0,
    };

    let x = act_tensor(vec![128, 4096], DType::F16);
    let expert_ids = act_tensor(vec![128, 2], DType::U32);
    let weights = act_tensor(vec![128, 2], DType::F32);
    let w_gate_up = weight_tensor(vec![8, 8192, 4096], DType::F16);
    let w_down = weight_tensor(vec![8, 4096, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    assert!(op
        .validate(&[x, expert_ids, weights, w_gate_up, w_down], &[y])
        .is_ok());

    let x = quant_act_tensor(vec![128, 4096], DType::I8, QuantScheme::PerToken);
    let expert_ids = act_tensor(vec![128, 2], DType::U32);
    let weights = act_tensor(vec![128, 2], DType::F32);
    let w_gate_up = quant_weight_tensor(vec![8, 8192, 4096], DType::I8, QuantScheme::PerRow);
    let w_down = quant_weight_tensor(vec![8, 4096, 4096], DType::I8, QuantScheme::PerRow);
    let y = act_tensor(vec![128, 4096], DType::F16);
    assert!(op
        .validate(&[x, expert_ids, weights, w_gate_up, w_down], &[y])
        .is_ok());
}

#[test]
fn moe_ffn_rejects_expert_id_dtype_and_w_down_rank_mismatches() {
    let op = MoeFfnOp {
        act: ActivationKind::Silu,
        out_dtype: DType::F16,
        shared_experts: 0,
    };

    let x = act_tensor(vec![128, 4096], DType::F16);
    let expert_ids = act_tensor(vec![128, 2], DType::U32);
    let weights = act_tensor(vec![128, 2], DType::F32);
    let w_gate_up = weight_tensor(vec![8, 8192, 4096], DType::F16);
    let w_down = weight_tensor(vec![8, 4096, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. expert_ids not U32
    let bad_expert_ids = act_tensor(vec![128, 2], DType::F32);
    let err = op
        .validate(
            &[
                x.clone(),
                bad_expert_ids,
                weights.clone(),
                w_gate_up.clone(),
                w_down.clone(),
            ],
            std::slice::from_ref(&y),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "moe_ffn",
                tensor: "expert_ids",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. w_down rank != 3
    let bad_w_down = weight_tensor(vec![8, 4096], DType::F16);
    let err = op
        .validate(&[x, expert_ids, weights, w_gate_up, bad_w_down], &[y])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "moe_ffn",
                tensor: "w_down",
                expected: 3,
                got: 2
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn moe_ffn_exposes_collect_all_on_coexisting_violations() {
    let op = MoeFfnOp {
        act: ActivationKind::Silu,
        out_dtype: DType::I32,
        shared_experts: 0,
    };

    let bad_x = weight_tensor(vec![128, 4096], DType::F16);
    let bad_expert_ids = act_tensor(vec![128, 2], DType::F32);
    let bad_weights = act_tensor(vec![128, 2], DType::I32);
    let bad_w_gate_up = act_tensor(vec![8, 8192], DType::F16);
    let bad_w_down = weight_tensor(vec![8, 4096, 4096], DType::I32);
    let bad_y = act_tensor(vec![128, 4096], DType::F32);

    let err = op
        .validate(
            &[
                bad_x,
                bad_expert_ids,
                bad_weights,
                bad_w_gate_up,
                bad_w_down,
            ],
            &[bad_y],
        )
        .unwrap_err();
    let problems = assert_multiple_problems(err, 6);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "out_dtype",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "x", .. })));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "expert_ids",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "weights",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpRankMismatch {
            tensor: "w_gate_up",
            ..
        }
    )));
}

#[test]
fn moe_ffn_rejects_operands_outside_matmul_numerics_contract() {
    let op = MoeFfnOp {
        act: ActivationKind::Silu,
        out_dtype: DType::F16,
        shared_experts: 0,
    };
    let expert_ids = act_tensor(vec![4, 2], DType::U32);
    let weights = act_tensor(vec![4, 2], DType::F32);
    let y = act_tensor(vec![4, 32], DType::F16);

    let bad_x = act_tensor(vec![4, 32], DType::F32);
    let bad_gate = weight_tensor(vec![8, 64, 32], DType::Bf16);
    let bad_down = weight_tensor(vec![8, 32, 32], DType::I8);
    let problems = assert_multiple_problems(
        op.validate(&[bad_x, expert_ids, weights, bad_gate, bad_down], &[y])
            .unwrap_err(),
        3,
    );
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "w_gate_up",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpQuantMismatch {
            tensor: "w_down",
            ..
        }
    )));
}

// -----------------------------------------------------------------------------
// §4.D Attention
// -----------------------------------------------------------------------------

#[test]
fn state_write_kv_accepts_paged_and_mla_latent_signatures() {
    let k = act_tensor(vec![128, 8, 128], DType::F16);
    let v = act_tensor(vec![128, 8, 128], DType::F16);

    // 1. Standard paged attention cache write
    let op_paged = StateWriteKvOp {
        cache_dtype: DType::F16,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: None,
        handle: StateHandle::new(0, StateKind::KvPaged),
    };
    assert!(op_paged.validate(&[k.clone(), v.clone()], &[]).is_ok());

    // 2. MLA latent cache write
    let op_mla = StateWriteKvOp {
        cache_dtype: DType::F16,
        scale_granularity: CacheScaleGranularity::PerBlock,
        latent: Some(MlaLatent {
            kv_lora_rank: 512,
            rope_dim: 64,
        }),
        handle: StateHandle::new(1, StateKind::KvLatent),
    };
    let k_mla = act_tensor(vec![128, 1, 576], DType::F16);
    let v_mla = act_tensor(vec![128, 1, 512], DType::F16);
    assert!(op_mla.validate(&[k_mla, v_mla], &[]).is_ok());
}

#[test]
fn state_write_kv_rejects_state_handle_mismatch_and_tensorized_slot_map() {
    let op = StateWriteKvOp {
        cache_dtype: DType::F16,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: None,
        handle: StateHandle::new(0, StateKind::KvPaged),
    };

    let k = act_tensor(vec![128, 8, 128], DType::F16);
    let v = act_tensor(vec![128, 8, 128], DType::F16);
    let slot_map = act_tensor(vec![1, 128], DType::U32);

    // 1. State handle kind mismatch (Recurrent instead of KvPaged)
    let bad_handle_op = StateWriteKvOp {
        handle: StateHandle::new(0, StateKind::Recurrent),
        ..op.clone()
    };
    let err = bad_handle_op
        .validate(&[k.clone(), v.clone()], &[])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::StateHandleKindMismatch {
                op: "state_write_kv",
                expected: StateKind::KvPaged,
                got: StateKind::Recurrent
            }
        ),
        "got: {err:?}"
    );

    // 2. slot_map belongs to structured BatchMeta and is rejected as a tensor operand.
    let err = op
        .validate(&[k.clone(), v.clone(), slot_map], &[])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpInputCountMismatch {
                op: "state_write_kv",
                expected: 2,
                got: 3
            }
        ),
        "got: {err:?}"
    );

    // 3. Non-empty output
    let bad_out = act_tensor(vec![128], DType::F16);
    let err = op.validate(&[k, v], &[bad_out]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpOutputCountMismatch {
                op: "state_write_kv",
                expected: 0,
                got: 1
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn state_write_kv_exposes_collect_all_on_coexisting_violations() {
    let op = StateWriteKvOp {
        cache_dtype: DType::F32,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: None,
        handle: StateHandle::new(0, StateKind::Recurrent),
    };

    let bad_k = act_tensor(vec![128, 8], DType::I32);
    let bad_v = act_tensor(vec![128, 8, 128], DType::F16);
    let bad_out = act_tensor(vec![1], DType::F32);

    let err = op.validate(&[bad_k, bad_v], &[bad_out]).unwrap_err();
    let problems = assert_multiple_problems(err, 5);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "cache_dtype",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::StateHandleKindMismatch { .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpRankMismatch { tensor: "k", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "k", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpOutputCountMismatch { .. })));
}

#[test]
fn attention_accepts_paged_mla_and_windowed_masks_without_batch_meta_as_tensor() {
    let q = act_tensor(vec![128, 32, 128], DType::F16);
    let o = act_tensor(vec![128, 32, 128], DType::F16);

    // 1. Standard Causal paged attention (SI-12: BatchMeta is non-tensor metadata, not in slice)
    let op_causal = AttentionOp {
        softmax_scale: 0.125,
        mask: AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: None,
        out_dtype: DType::F16,
        handle: StateHandle::new(0, StateKind::KvPaged),
    };
    assert!(op_causal
        .validate(std::slice::from_ref(&q), std::slice::from_ref(&o))
        .is_ok());

    // 2. Sliding window causal mask
    let op_window = AttentionOp {
        mask: AttentionMask::CausalWindow(512),
        ..op_causal.clone()
    };
    assert!(op_window
        .validate(std::slice::from_ref(&q), std::slice::from_ref(&o))
        .is_ok());

    // 3. MLA attention with KvLatent state handle
    let op_mla = AttentionOp {
        mla: Some(MlaAttentionSpec {
            q_lora_rank: None,
            kv_lora_rank: 512,
            qk_nope_dim: 128,
            qk_rope_dim: 64,
            v_dim: 128,
        }),
        handle: StateHandle::new(1, StateKind::KvLatent),
        ..op_causal
    };
    let q_mla = act_tensor(vec![128, 32, 192], DType::F16);
    let o_mla = act_tensor(vec![128, 32, 128], DType::F16);
    assert!(op_mla.validate(&[q_mla], &[o_mla]).is_ok());
}

#[test]
fn attention_rejects_invalid_scale_and_mismatched_state_kind() {
    let op = AttentionOp {
        softmax_scale: 0.125,
        mask: AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: None,
        out_dtype: DType::F16,
        handle: StateHandle::new(0, StateKind::KvPaged),
    };

    let q = act_tensor(vec![128, 32, 128], DType::F16);
    let o = act_tensor(vec![128, 32, 128], DType::F16);

    // 1. softmax_scale <= 0
    let bad_scale_op = AttentionOp {
        softmax_scale: 0.0,
        ..op.clone()
    };
    let err = bad_scale_op
        .validate(std::slice::from_ref(&q), std::slice::from_ref(&o))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "attention",
                attribute: "softmax_scale",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. State handle kind mismatch (KvLatent without mla spec)
    let bad_handle_op = AttentionOp {
        handle: StateHandle::new(0, StateKind::KvLatent),
        ..op.clone()
    };
    let err = bad_handle_op
        .validate(std::slice::from_ref(&q), std::slice::from_ref(&o))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::StateHandleKindMismatch {
                op: "attention",
                expected: StateKind::KvPaged,
                got: StateKind::KvLatent
            }
        ),
        "got: {err:?}"
    );

    // 3. q rank != 3
    let bad_q_rank = act_tensor(vec![128, 32], DType::F16);
    let err = op.validate(&[bad_q_rank], &[o]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "attention",
                tensor: "q",
                expected: 3,
                got: 2
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn attention_exposes_collect_all_on_coexisting_violations() {
    let op = AttentionOp {
        softmax_scale: -0.1,
        mask: AttentionMask::CausalWindow(0),
        sinks: 0,
        logit_softcap: Some(0.0),
        mla: None,
        out_dtype: DType::F16,
        handle: StateHandle::new(0, StateKind::Recurrent),
    };

    let bad_q = act_tensor(vec![128, 32], DType::I32);
    let bad_o = act_tensor(vec![128, 32, 128], DType::F32);

    let err = op.validate(&[bad_q], &[bad_o]).unwrap_err();
    let problems = assert_multiple_problems(err, 6);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "softmax_scale",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "mask",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "logit_softcap",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::StateHandleKindMismatch { .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpRankMismatch { tensor: "q", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "o", .. })));
}

// -----------------------------------------------------------------------------
// §4.E Sequence-state ops beyond attention
// -----------------------------------------------------------------------------

#[test]
fn causal_conv1d_accepts_two_and_three_input_signatures() {
    let handle = StateHandle::new(0, StateKind::ConvWindow);
    let op = CausalConv1dOp {
        kernel: 4,
        act: ConvActivation::Silu,
        handle,
    };

    let x = act_tensor(vec![128, 256], DType::F16);
    let w = weight_tensor(vec![256, 4], DType::F16);
    let bias = param_tensor(vec![256], DType::F16);
    let y = act_tensor(vec![128, 256], DType::F16);

    // 1. Two inputs (without bias)
    assert!(op
        .validate(&[x.clone(), w.clone()], std::slice::from_ref(&y))
        .is_ok());

    // 2. Three inputs (with bias)
    assert!(op.validate(&[x, w, bias], &[y]).is_ok());
}

#[test]
fn causal_conv1d_rejects_kernel_zero_and_state_handle_kind_mismatches() {
    let handle = StateHandle::new(0, StateKind::ConvWindow);
    let op = CausalConv1dOp {
        kernel: 4,
        act: ConvActivation::Silu,
        handle,
    };

    let x = act_tensor(vec![128, 256], DType::F16);
    let w = weight_tensor(vec![256, 4], DType::F16);
    let y = act_tensor(vec![128, 256], DType::F16);

    // 1. kernel == 0 (collects attribute error and shape mismatch against w width)
    let bad_kernel_op = CausalConv1dOp { kernel: 0, ..op };
    let err = bad_kernel_op
        .validate(&[x.clone(), w.clone()], std::slice::from_ref(&y))
        .unwrap_err();
    let problems = match err {
        IrError::Multiple { problems } => problems,
        single => vec![single].into_boxed_slice(),
    };
    assert!(
        problems.iter().any(|p| matches!(
            p,
            IrError::OpAttributeInvalid {
                op: "causal_conv1d",
                attribute: "kernel",
                ..
            }
        )),
        "got: {problems:?}"
    );

    // 2. State handle kind mismatch
    let bad_handle_op = CausalConv1dOp {
        handle: StateHandle::new(0, StateKind::KvPaged),
        ..op
    };
    let err = bad_handle_op
        .validate(&[x.clone(), w.clone()], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::StateHandleKindMismatch {
                op: "causal_conv1d",
                expected: StateKind::ConvWindow,
                got: StateKind::KvPaged
            }
        ),
        "got: {err:?}"
    );

    // 3. w kernel dimension mismatch
    let bad_w_k = weight_tensor(vec![256, 5], DType::F16);
    let err = op.validate(&[x, bad_w_k], &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "causal_conv1d",
                tensor: "w",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn causal_conv1d_exposes_collect_all_on_coexisting_violations() {
    let op = CausalConv1dOp {
        kernel: 0,
        act: ConvActivation::Identity,
        handle: StateHandle::new(0, StateKind::KvPaged),
    };

    let bad_x = act_tensor(vec![128, 256], DType::I32);
    let bad_w = act_tensor(vec![256, 5], DType::F16);
    let bad_y = act_tensor(vec![128, 256], DType::F32);

    let err = op.validate(&[bad_x, bad_w], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 5);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "kernel",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::StateHandleKindMismatch { .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpClassMismatch { tensor: "w", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpShapeMismatch { tensor: "w", .. })));
}

#[test]
fn linear_attn_scan_accepts_recurrent_five_input_operands() {
    let handle = StateHandle::new(0, StateKind::Recurrent);
    let op = LinearAttnScanOp {
        kind: LinearAttnKind::GLA,
        chunk: 64,
        out_dtype: DType::F16,
        handle,
    };

    let q = act_tensor(vec![128, 16, 64], DType::F16);
    let k = act_tensor(vec![128, 16, 64], DType::F16);
    let v = act_tensor(vec![128, 16, 64], DType::F16);
    let alpha = act_tensor(vec![128, 16], DType::F32);
    let beta = act_tensor(vec![128, 16], DType::F32);
    let o = act_tensor(vec![128, 16, 64], DType::F16);

    assert!(op.validate(&[q, k, v, alpha, beta], &[o]).is_ok());
}

#[test]
fn linear_attn_scan_rejects_alpha_dtype_and_state_handle_mismatches() {
    let handle = StateHandle::new(0, StateKind::Recurrent);
    let op = LinearAttnScanOp {
        kind: LinearAttnKind::Mamba2,
        chunk: 64,
        out_dtype: DType::F16,
        handle,
    };

    let q = act_tensor(vec![128, 16, 64], DType::F16);
    let k = act_tensor(vec![128, 16, 64], DType::F16);
    let v = act_tensor(vec![128, 16, 64], DType::F16);
    let alpha = act_tensor(vec![128, 16], DType::F32);
    let beta = act_tensor(vec![128, 16], DType::F32);
    let o = act_tensor(vec![128, 16, 64], DType::F16);

    // 1. alpha not F32
    let bad_alpha = act_tensor(vec![128, 16], DType::F16);
    let err = op
        .validate(
            &[q.clone(), k.clone(), v.clone(), bad_alpha, beta.clone()],
            std::slice::from_ref(&o),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "linear_attn_scan",
                tensor: "alpha",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. handle kind mismatch
    let bad_handle_op = LinearAttnScanOp {
        handle: StateHandle::new(0, StateKind::KvPaged),
        ..op
    };
    let err = bad_handle_op
        .validate(&[q, k, v, alpha, beta], &[o])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::StateHandleKindMismatch {
                op: "linear_attn_scan",
                expected: StateKind::Recurrent,
                got: StateKind::KvPaged
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn linear_attn_scan_exposes_collect_all_on_coexisting_violations() {
    let op = LinearAttnScanOp {
        kind: LinearAttnKind::GatedDeltaNet,
        chunk: 0,
        out_dtype: DType::I32,
        handle: StateHandle::new(0, StateKind::KvPaged),
    };

    let bad_q = act_tensor(vec![128, 16], DType::F16);
    let bad_k = act_tensor(vec![128, 16, 64], DType::I32);
    let bad_v = act_tensor(vec![128, 16, 64], DType::F16);
    let bad_alpha = act_tensor(vec![128, 16], DType::F16);
    let bad_beta = act_tensor(vec![128, 16], DType::I32);
    let bad_o = act_tensor(vec![128, 16, 64], DType::F32);

    let err = op
        .validate(&[bad_q, bad_k, bad_v, bad_alpha, bad_beta], &[bad_o])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 6);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "chunk",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "out_dtype",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::StateHandleKindMismatch { .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpRankMismatch { tensor: "q", .. })));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "alpha",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "o", .. })));
}

// -----------------------------------------------------------------------------
// §4.F Sampling and verification
// -----------------------------------------------------------------------------

#[test]
fn logits_postprocess_accepts_one_two_and_three_input_signatures_without_sampling_params_tensor() {
    let op = LogitsPostprocessOp;
    let logits_3d = act_tensor(vec![1, 1, 32000], DType::F32);
    let history = act_tensor(vec![1, 32000], DType::U32);
    let mask = bool_tensor(vec![1, 1, 32000]);
    let probs_3d = act_tensor(vec![1, 1, 32000], DType::F32);

    // 1. One input (3D logits)
    assert!(op
        .validate(
            std::slice::from_ref(&logits_3d),
            std::slice::from_ref(&probs_3d)
        )
        .is_ok());

    // 2. Two inputs (logits + history_counts)
    assert!(op
        .validate(
            &[logits_3d.clone(), history.clone()],
            std::slice::from_ref(&probs_3d)
        )
        .is_ok());

    // 3. Two inputs (logits + grammar_mask)
    assert!(op
        .validate(
            &[logits_3d.clone(), mask.clone()],
            std::slice::from_ref(&probs_3d)
        )
        .is_ok());

    // 4. Three inputs (logits + history_counts + grammar_mask; SamplingParams is non-tensor per SI-12)
    assert!(op
        .validate(&[logits_3d, history, mask], &[probs_3d])
        .is_ok());
}

#[test]
fn logits_postprocess_rejects_logits_rank_and_grammar_mask_dtype() {
    let op = LogitsPostprocessOp;
    let logits = act_tensor(vec![1, 1, 32000], DType::F32);
    let probs = act_tensor(vec![1, 1, 32000], DType::F32);

    // 1. Logits wrong rank (rank 4)
    let bad_logits_rank = act_tensor(vec![1, 1, 1, 32000], DType::F32);
    let err = op
        .validate(&[bad_logits_rank], std::slice::from_ref(&probs))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpRankMismatch {
                op: "logits_postprocess",
                tensor: "logits",
                expected: 3,
                got: 4
            }
        ),
        "got: {err:?}"
    );

    // 2. Logits wrong dtype
    let bad_logits_dtype = act_tensor(vec![1, 1, 32000], DType::F16);
    let err = op
        .validate(&[bad_logits_dtype], std::slice::from_ref(&probs))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "logits_postprocess",
                tensor: "logits",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. Grammar mask wrong dtype (F32 instead of Bool)
    let bad_mask_dtype = act_tensor(vec![1, 1, 32000], DType::F32);
    let history = act_tensor(vec![1, 32000], DType::U32);
    let err = op
        .validate(&[logits, history, bad_mask_dtype], &[probs])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "logits_postprocess",
                tensor: "grammar_mask",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn logits_postprocess_exposes_collect_all_on_coexisting_violations() {
    let op = LogitsPostprocessOp;
    let bad_logits = act_tensor(vec![1, 1, 1, 32000], DType::F16);
    let bad_history = act_tensor(vec![1, 32000], DType::F32);
    let bad_mask = act_tensor(vec![1, 32000], DType::F32);
    let bad_probs = act_tensor(vec![1, 1, 32000], DType::F16);

    let err = op
        .validate(&[bad_logits, bad_history, bad_mask], &[bad_probs])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpRankMismatch {
            tensor: "logits",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "logits",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "history_counts",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "probs",
            ..
        }
    )));
}

#[test]
fn sample_accepts_prob_distribution_without_rng_as_tensor() {
    let op = SampleOp {
        rng: RngAlgorithm::Philox4x32,
    };
    // SI-12: rng_state is typed external value, not passed in tensor slice
    let probs = act_tensor(vec![1, 32000], DType::F32);
    let token = act_tensor(vec![1], DType::U32);

    assert!(op.validate(&[probs], &[token]).is_ok());
}

#[test]
fn sample_rejects_rank_dtype_and_batch_dimension_mismatches() {
    let op = SampleOp {
        rng: RngAlgorithm::Philox4x32,
    };
    let probs = act_tensor(vec![1, 32000], DType::F32);
    let token = act_tensor(vec![1], DType::U32);

    // 1. token not U32
    let bad_token_dtype = act_tensor(vec![1], DType::F32);
    let err = op
        .validate(std::slice::from_ref(&probs), &[bad_token_dtype])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "sample",
                tensor: "token",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. probs not F32
    let bad_probs_dtype = act_tensor(vec![1, 32000], DType::F16);
    let err = op
        .validate(&[bad_probs_dtype], std::slice::from_ref(&token))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "sample",
                tensor: "probs",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. Batch dimension S mismatch
    let bad_token_batch = act_tensor(vec![2], DType::U32);
    let err = op.validate(&[probs], &[bad_token_batch]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "sample",
                tensor: "token",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn sample_exposes_collect_all_on_coexisting_violations() {
    let op = SampleOp {
        rng: RngAlgorithm::Philox4x32,
    };
    let bad_probs = act_tensor(vec![1, 32000, 1], DType::F16);
    let bad_token = act_tensor(vec![2, 1], DType::F32);

    let err = op.validate(&[bad_probs], &[bad_token]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpRankMismatch {
            tensor: "probs",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "probs",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpRankMismatch {
            tensor: "token",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "token",
            ..
        }
    )));
}

#[test]
fn verify_accepts_two_and_three_input_arities_without_tree_as_tensor() {
    let op = VerifyOp {
        method: VerifyMethod::Greedy,
    };

    let draft_tokens = act_tensor(vec![1, 4], DType::U32);
    let target_probs = act_tensor(vec![1, 5, 32000], DType::F32);
    let draft_probs = act_tensor(vec![1, 4, 32000], DType::F32);
    let accepted = act_tensor(vec![1, 5], DType::U32);
    let accept_len = act_tensor(vec![1], DType::U32);

    // 1. Two inputs (deterministic proposer, draft_probs omitted)
    assert!(op
        .validate(
            &[draft_tokens.clone(), target_probs.clone()],
            &[accepted.clone(), accept_len.clone()]
        )
        .is_ok());

    // 2. Three inputs (stochastic proposer with draft_probs; SI-12: tree is non-tensor)
    assert!(op
        .validate(
            &[draft_tokens, draft_probs, target_probs],
            &[accepted, accept_len]
        )
        .is_ok());
}

#[test]
fn verify_rejects_draft_token_dtype_and_output_count_mismatches() {
    let op = VerifyOp {
        method: VerifyMethod::Rejection,
    };

    let draft_tokens = act_tensor(vec![1, 4], DType::U32);
    let target_probs = act_tensor(vec![1, 5, 32000], DType::F32);
    let accepted = act_tensor(vec![1, 5], DType::U32);
    let accept_len = act_tensor(vec![1], DType::U32);

    // 1. draft_tokens not U32
    let bad_draft = act_tensor(vec![1, 4], DType::F32);
    let err = op
        .validate(
            &[bad_draft, target_probs.clone()],
            &[accepted.clone(), accept_len.clone()],
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "verify",
                tensor: "draft_tokens",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. Output count mismatch (only 1 output instead of 2)
    let err = op
        .validate(&[draft_tokens, target_probs], &[accepted])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpOutputCountMismatch {
                op: "verify",
                expected: 2,
                got: 1
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn verify_exposes_collect_all_on_coexisting_violations() {
    let op = VerifyOp {
        method: VerifyMethod::TypicalAcceptance {
            eps: 0.0,
            delta: -1.0,
        },
    };

    let bad_draft = act_tensor(vec![1, 4], DType::F32);
    let bad_target = act_tensor(vec![1, 5, 32000], DType::F16);
    let bad_accepted = act_tensor(vec![1, 5], DType::F32);

    let err = op
        .validate(&[bad_draft, bad_target], &[bad_accepted])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 5);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "method.eps",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "method.delta",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "draft_tokens",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "target_probs",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpOutputCountMismatch { .. })));
}

// -----------------------------------------------------------------------------
// §4.G Collectives
// -----------------------------------------------------------------------------

#[test]
fn all_reduce_accepts_matching_element_dtype() {
    let op = AllReduceOp {
        group: GroupId::new(0),
        op: ReduceOp::Sum,
        dtype: DType::F16,
        reduce_in: DType::F32,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    assert!(op.validate(&[x], &[y]).is_ok());
}

#[test]
fn all_reduce_rejects_invalid_reduce_in_and_dtype_mismatches() {
    let op = AllReduceOp {
        group: GroupId::new(0),
        op: ReduceOp::Sum,
        dtype: DType::F16,
        reduce_in: DType::F32,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. reduce_in != F32
    let bad_reduce_in = AllReduceOp {
        reduce_in: DType::F16,
        ..op
    };
    let err = bad_reduce_in
        .validate(std::slice::from_ref(&x), std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "all_reduce",
                attribute: "reduce_in",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. output dtype mismatch
    let bad_y = act_tensor(vec![128, 4096], DType::F32);
    let err = op.validate(&[x], &[bad_y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "all_reduce",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn all_reduce_exposes_collect_all_on_coexisting_violations() {
    let op = AllReduceOp {
        group: GroupId::new(0),
        op: ReduceOp::Sum,
        dtype: DType::F16,
        reduce_in: DType::F16,
    };

    let bad_x = act_tensor(vec![128, 4096], DType::I32);
    let bad_y = act_tensor(vec![128, 4096], DType::F32);

    let err = op.validate(&[bad_x], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 3);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "reduce_in",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
}

#[test]
fn all_gather_accepts_gathered_axis_dimension() {
    let op = AllGatherOp {
        group: GroupId::new(0),
        axis: 0,
        dtype: DType::F16,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![256, 4096], DType::F16);

    assert!(op.validate(&[x], &[y]).is_ok());
}

#[test]
fn all_gather_rejects_out_of_bounds_axis_and_rank_mismatch() {
    let op = AllGatherOp {
        group: GroupId::new(0),
        axis: 2,
        dtype: DType::F16,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. axis >= rank
    let err = op.validate(std::slice::from_ref(&x), &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "all_gather",
                attribute: "axis",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. non-gathered dimension mismatch
    let op_valid_axis = AllGatherOp { axis: 0, ..op };
    let bad_y_dim = act_tensor(vec![256, 2048], DType::F16);
    let err = op_valid_axis.validate(&[x], &[bad_y_dim]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "all_gather",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn all_gather_exposes_collect_all_on_coexisting_violations() {
    let op = AllGatherOp {
        group: GroupId::new(0),
        axis: 4,
        dtype: DType::F16,
    };
    let bad_x = act_tensor(vec![128, 4096], DType::I32);
    let bad_y = act_tensor(vec![256, 4096, 1], DType::F32);

    let err = op.validate(&[bad_x], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "axis",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpRankMismatch { tensor: "y", .. })));
}

#[test]
fn reduce_scatter_accepts_partitioned_axis_dimension() {
    let op = ReduceScatterOp {
        group: GroupId::new(0),
        axis: 0,
        op: ReduceOp::Sum,
        dtype: DType::F16,
        reduce_in: DType::F32,
    };
    let x = act_tensor(vec![256, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    assert!(op.validate(&[x], &[y]).is_ok());
}

#[test]
fn reduce_scatter_rejects_invalid_reduce_in_and_axis_bounds() {
    let op = ReduceScatterOp {
        group: GroupId::new(0),
        axis: 0,
        op: ReduceOp::Sum,
        dtype: DType::F16,
        reduce_in: DType::F32,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. reduce_in != F32
    let bad_reduce = ReduceScatterOp {
        reduce_in: DType::F16,
        ..op
    };
    let err = bad_reduce
        .validate(std::slice::from_ref(&x), std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "reduce_scatter",
                attribute: "reduce_in",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. axis >= rank
    let bad_axis = ReduceScatterOp { axis: 3, ..op };
    let err = bad_axis.validate(&[x], &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpAttributeInvalid {
                op: "reduce_scatter",
                attribute: "axis",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn reduce_scatter_exposes_collect_all_on_coexisting_violations() {
    let op = ReduceScatterOp {
        group: GroupId::new(0),
        axis: 5,
        op: ReduceOp::Sum,
        dtype: DType::F16,
        reduce_in: DType::F16,
    };
    let bad_x = act_tensor(vec![256, 4096], DType::I32);
    let bad_y = act_tensor(vec![128, 4096], DType::F32);

    let err = op.validate(&[bad_x], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "reduce_in",
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpAttributeInvalid {
            attribute: "axis",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
}

#[test]
fn all_to_all_requires_explicit_counts_and_one_output() {
    let op = AllToAllOp {
        group: GroupId::new(0),
        dtype: DType::F16,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);
    let counts = act_tensor(vec![4], DType::U32);
    assert!(op.validate(&[x, counts], &[y]).is_ok());
}

#[test]
fn all_to_all_rejects_counts_dtype_and_count_candidate_violations() {
    let op = AllToAllOp {
        group: GroupId::new(0),
        dtype: DType::F16,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. counts not U32
    let bad_counts = act_tensor(vec![4], DType::F32);
    let err = op
        .validate(&[x.clone(), bad_counts], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "all_to_all",
                tensor: "counts",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 2. output dtype mismatch
    let bad_y = act_tensor(vec![128, 4096], DType::F32);
    let counts = act_tensor(vec![4], DType::U32);
    let err = op.validate(&[x.clone(), counts], &[bad_y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "all_to_all",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. Empty inputs
    let err = op.validate(&[], &[y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpInputCountMismatch {
                op: "all_to_all",
                expected: 2,
                got: 0
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn all_to_all_exposes_collect_all_on_coexisting_violations() {
    let op = AllToAllOp {
        group: GroupId::new(0),
        dtype: DType::F16,
    };
    let bad_x = act_tensor(vec![128, 4096], DType::I32);
    let bad_counts = act_tensor(vec![4], DType::F32);
    let bad_y = act_tensor(vec![128, 4096], DType::F32);
    let bad_recv = act_tensor(vec![4], DType::F32);

    let err = op
        .validate(&[bad_x, bad_counts], &[bad_y, bad_recv])
        .unwrap_err();
    let problems = assert_multiple_problems(err, 4);
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpDTypeMismatch {
            tensor: "counts",
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpOutputCountMismatch {
            expected: 1,
            got: 2,
            ..
        }
    )));
}

#[test]
fn send_accepts_valid_tensor_and_empty_output() {
    let op = SendOp {
        group: GroupId::new(0),
        peer: 1,
        dtype: DType::F16,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);

    assert!(op.validate(&[x], &[]).is_ok());
}

#[test]
fn send_rejects_non_empty_output_and_dtype_mismatch() {
    let op = SendOp {
        group: GroupId::new(0),
        peer: 1,
        dtype: DType::F16,
    };
    let x = act_tensor(vec![128, 4096], DType::F16);

    // 1. non-empty output
    let bad_out = act_tensor(vec![128, 4096], DType::F16);
    let err = op
        .validate(std::slice::from_ref(&x), &[bad_out])
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpOutputCountMismatch {
                op: "send",
                expected: 0,
                got: 1
            }
        ),
        "got: {err:?}"
    );

    // 2. x dtype mismatch
    let bad_x = act_tensor(vec![128, 4096], DType::F32);
    let err = op.validate(&[bad_x], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "send",
                tensor: "x",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn send_exposes_collect_all_on_coexisting_violations() {
    let op = SendOp {
        group: GroupId::new(0),
        peer: 1,
        dtype: DType::F16,
    };
    let bad_x = act_tensor(vec![128, 4096], DType::F32);
    let bad_out = act_tensor(vec![1], DType::F32);

    let err = op.validate(&[bad_x], &[bad_out]).unwrap_err();
    let problems = assert_multiple_problems(err, 2);
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "x", .. })));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpOutputCountMismatch {
            expected: 0,
            got: 1,
            ..
        }
    )));
}

#[test]
fn recv_accepts_empty_input_and_matching_shape() {
    let op = RecvOp {
        group: GroupId::new(0),
        peer: 0,
        shape: vec![Dim::Concrete(128), Dim::Concrete(4096)].into_boxed_slice(),
        dtype: DType::F16,
    };
    let y = act_tensor(vec![128, 4096], DType::F16);

    assert!(op.validate(&[], &[y]).is_ok());
}

#[test]
fn recv_rejects_non_empty_input_and_shape_mismatches() {
    let op = RecvOp {
        group: GroupId::new(0),
        peer: 0,
        shape: vec![Dim::Concrete(128), Dim::Concrete(4096)].into_boxed_slice(),
        dtype: DType::F16,
    };
    let y = act_tensor(vec![128, 4096], DType::F16);

    // 1. non-empty input
    let bad_in = act_tensor(vec![128, 4096], DType::F16);
    let err = op
        .validate(&[bad_in], std::slice::from_ref(&y))
        .unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpInputCountMismatch {
                op: "recv",
                expected: 0,
                got: 1
            }
        ),
        "got: {err:?}"
    );

    // 2. y shape mismatch
    let bad_y = act_tensor(vec![128, 2048], DType::F16);
    let err = op.validate(&[], &[bad_y]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpShapeMismatch {
                op: "recv",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );

    // 3. y dtype mismatch
    let bad_y_dtype = act_tensor(vec![128, 4096], DType::F32);
    let err = op.validate(&[], &[bad_y_dtype]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpDTypeMismatch {
                op: "recv",
                tensor: "y",
                ..
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn recv_exposes_collect_all_on_coexisting_violations() {
    let op = RecvOp {
        group: GroupId::new(0),
        peer: 0,
        shape: vec![Dim::Concrete(128), Dim::Concrete(4096)].into_boxed_slice(),
        dtype: DType::F16,
    };
    let bad_in = act_tensor(vec![1], DType::F16);
    let bad_y = act_tensor(vec![128, 2048], DType::F32);

    let err = op.validate(&[bad_in], &[bad_y]).unwrap_err();
    let problems = assert_multiple_problems(err, 3);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpInputCountMismatch {
            expected: 0,
            got: 1,
            ..
        }
    )));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpDTypeMismatch { tensor: "y", .. })));
    assert!(problems
        .iter()
        .any(|p| matches!(p, IrError::OpShapeMismatch { tensor: "y", .. })));
}

#[test]
fn barrier_accepts_empty_inputs_and_outputs() {
    let op = BarrierOp {
        group: GroupId::new(0),
    };

    assert!(op.validate(&[], &[]).is_ok());
}

#[test]
fn barrier_rejects_non_empty_inputs_or_outputs() {
    let op = BarrierOp {
        group: GroupId::new(0),
    };
    let tensor = act_tensor(vec![1], DType::F32);

    // 1. non-empty input
    let err = op.validate(std::slice::from_ref(&tensor), &[]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpInputCountMismatch {
                op: "barrier",
                expected: 0,
                got: 1
            }
        ),
        "got: {err:?}"
    );

    // 2. non-empty output
    let err = op.validate(&[], &[tensor]).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::OpOutputCountMismatch {
                op: "barrier",
                expected: 0,
                got: 1
            }
        ),
        "got: {err:?}"
    );
}

#[test]
fn barrier_exposes_collect_all_on_coexisting_violations() {
    let op = BarrierOp {
        group: GroupId::new(0),
    };
    let bad_in = act_tensor(vec![1], DType::F32);
    let bad_out = act_tensor(vec![1], DType::F32);

    let err = op.validate(&[bad_in], &[bad_out]).unwrap_err();
    let problems = assert_multiple_problems(err, 2);
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpInputCountMismatch {
            expected: 0,
            got: 1,
            ..
        }
    )));
    assert!(problems.iter().any(|p| matches!(
        p,
        IrError::OpOutputCountMismatch {
            expected: 0,
            got: 1,
            ..
        }
    )));
}

// -----------------------------------------------------------------------------
// Op Enum Dispatch
// -----------------------------------------------------------------------------

#[test]
fn op_enum_dispatch_validates_variants_consistently() {
    let cast = Op::Cast(CastOp { dtype: DType::Bf16 });
    let x = act_tensor(vec![128, 4096], DType::F16);
    let y = act_tensor(vec![128, 4096], DType::Bf16);
    let bad_y = act_tensor(vec![128, 4096], DType::F32);

    assert!(cast.validate(std::slice::from_ref(&x), &[y]).is_ok());
    assert!(cast.validate(&[x], &[bad_y]).is_err());
}

#[test]
fn op_enum_dispatches_validation_across_groups_d_through_g() {
    let representatives = [
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
        Op::LogitsPostprocess(LogitsPostprocessOp),
        Op::AllReduce(AllReduceOp {
            group: GroupId::new(0),
            op: ReduceOp::Sum,
            dtype: DType::F16,
            reduce_in: DType::F32,
        }),
    ];
    assert_eq!(
        representatives.map(|op| (op.op_name(), op.validate(&[], &[]).is_err())),
        [
            ("attention", true),
            ("causal_conv1d", true),
            ("logits_postprocess", true),
            ("all_reduce", true),
        ]
    );
}

#[test]
fn op_enum_dispatches_public_numerics_contracts() {
    let attention = Op::Attention(AttentionOp {
        softmax_scale: 0.125,
        mask: AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: None,
        out_dtype: DType::F16,
        handle: StateHandle::new(0, StateKind::KvPaged),
    });
    assert_eq!(
        attention.numerics(&[]).unwrap(),
        Numerics::f32(ReductionOrder::AscendingBlock)
    );

    let logits = Op::LogitsPostprocess(LogitsPostprocessOp);
    assert_eq!(
        logits.numerics(&[]).unwrap(),
        Numerics::f32(ReductionOrder::AscendingIndex)
    );

    let all_reduce = Op::AllReduce(AllReduceOp {
        group: GroupId::new(0),
        op: ReduceOp::Sum,
        dtype: DType::F16,
        reduce_in: DType::F32,
    });
    assert_eq!(
        all_reduce.numerics(&[]).unwrap(),
        Numerics::f32(ReductionOrder::AscendingRank)
    );

    let all_to_all = Op::AllToAll(AllToAllOp {
        group: GroupId::new(0),
        dtype: DType::F16,
    });
    assert_eq!(all_to_all.numerics(&[]).unwrap(), Numerics::none());
}

#[test]
fn validation_reports_dimension_overflow_without_panicking() {
    let ngram_device = NgramGatherOp {
        source: NgramSource::Device,
        orders: vec![2, 3].into_boxed_slice(),
        heads: 2,
        hash: HashId::new(1),
        table_sizes: vec![u32::MAX, 1].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F16,
    };
    let token_ids = act_tensor(vec![1], DType::U32);
    let table = weight_tensor(vec![1, 1], DType::F16);
    let x = act_tensor(vec![1, 2], DType::F16);
    assert!(ngram_device.validate(&[token_ids, table], &[x]).is_err());

    let ngram_staged = NgramGatherOp {
        source: NgramSource::Staged,
        orders: vec![2].into_boxed_slice(),
        heads: u32::MAX,
        hash: HashId::new(1),
        table_sizes: vec![1].into_boxed_slice(),
        combine: NgramCombine::Concat,
        out_dtype: DType::F16,
    };
    let staging = staging_tensor(vec![1, 1, 2], DType::I8);
    let scales = act_tensor(vec![1], DType::F32);
    let x = act_tensor(vec![1, 1], DType::F16);
    assert!(ngram_staged.validate(&[staging, scales], &[x]).is_err());

    let rope = RopeOp {
        rot_dim: u32::MAX,
        theta: 10_000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: Some([u32::MAX, 2, 2]),
        out_dtype: DType::F16,
    };
    let x = act_tensor(vec![1, 1, 2], DType::F16);
    let positions = act_tensor(vec![1, 3], DType::U32);
    let y = act_tensor(vec![1, 1, 2], DType::F16);
    assert!(rope.validate(&[x, positions], &[y]).is_err());

    let moe = MoeFfnOp {
        act: ActivationKind::Silu,
        out_dtype: DType::F16,
        shared_experts: 0,
    };
    let x = act_tensor(vec![1, 1], DType::F16);
    let expert_ids = act_tensor(vec![1, 1], DType::U32);
    let weights = act_tensor(vec![1, 1], DType::F32);
    let w_gate_up = weight_tensor(vec![1, 1, 1], DType::F16);
    let w_down = weight_tensor(vec![1, 1, u32::MAX], DType::F16);
    let y = act_tensor(vec![1, 1], DType::F16);
    assert!(moe
        .validate(&[x, expert_ids, weights, w_gate_up, w_down], &[y])
        .is_err());

    let state_write = StateWriteKvOp {
        cache_dtype: DType::F16,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: Some(MlaLatent {
            kv_lora_rank: u32::MAX,
            rope_dim: 2,
        }),
        handle: StateHandle::new(0, StateKind::KvLatent),
    };
    let k = act_tensor(vec![1, 1, 3], DType::F16);
    let v = act_tensor(vec![1, 1, 3], DType::F16);
    assert!(state_write.validate(&[k, v], &[]).is_err());

    let attention = AttentionOp {
        softmax_scale: 1.0,
        mask: AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: Some(MlaAttentionSpec {
            q_lora_rank: None,
            kv_lora_rank: 1,
            qk_nope_dim: u32::MAX,
            qk_rope_dim: 2,
            v_dim: 1,
        }),
        out_dtype: DType::F16,
        handle: StateHandle::new(0, StateKind::KvLatent),
    };
    let q = act_tensor(vec![1, 1, 1], DType::F16);
    let o = act_tensor(vec![1, 1, 1], DType::F16);
    assert!(attention.validate(&[q], &[o]).is_err());

    let verify = VerifyOp {
        method: VerifyMethod::Greedy,
    };
    let draft_tokens = act_tensor(vec![1, u32::MAX], DType::U32);
    let target_probs = act_tensor(vec![1, 1, 2], DType::F32);
    let accepted = act_tensor(vec![1, 1], DType::U32);
    let accept_len = act_tensor(vec![1], DType::U32);
    let problems = assert_multiple_problems(
        verify
            .validate(&[draft_tokens, target_probs], &[accepted, accept_len])
            .unwrap_err(),
        2,
    );
    assert!(problems.iter().any(|problem| matches!(
        problem,
        IrError::OpShapeMismatch {
            tensor: "target_probs",
            ..
        }
    )));
    assert!(problems.iter().any(|problem| matches!(
        problem,
        IrError::OpShapeMismatch {
            tensor: "accepted",
            ..
        }
    )));
}
