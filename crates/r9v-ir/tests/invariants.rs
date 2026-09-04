// SPDX-License-Identifier: Apache-2.0
//! Behavioral constructor and invariant tests for r9v-ir (card A1.1).
//!
//! Each test names the behavior it proves. Random inputs come from
//! `r9v_common::SeededRng` with fixed seeds (CONVENTIONS.md §4.3): no
//! wall-clock, no environment, deterministic across runs.

use r9v_common::SeededRng;
use r9v_ir::{
    ArchDescriptor, ArchFamily, BatchMeta, Class, DType, Dim, GraphCapture, IrError, IrVersion,
    LayoutId, Placement, Positions, QuantScheme, RelRate, SchemeId, ShapeSymbol, ShardLayout,
    StateHandle, StateKind, Tensor, TreeMask, ValuDot, BLOCK_TABLE_SENTINEL,
};

fn weight_tensor() -> Tensor {
    Tensor::new(
        vec![Dim::Concrete(64), Dim::Concrete(64)],
        DType::I8,
        QuantScheme::Scheme(SchemeId::new(11)),
        LayoutId::L1,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .expect("valid weight tensor builds")
}

type BatchParts = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

fn batch_parts(g: usize, s: usize, t: usize, max_blocks: usize) -> BatchParts {
    assert!(
        t >= s,
        "each of S sequences must contribute at least one token"
    );
    let mut query_len = vec![1; s];
    query_len[0] += (t - s) as u32;
    (
        (0..s as u32).collect(),
        query_len,
        vec![0; s],
        (0..(g * t) as u32).collect(),
        vec![BLOCK_TABLE_SENTINEL; g * s * max_blocks],
        vec![0; g * s],
    )
}

fn valid_batch(g: u32, s: u32, t: u32, max_blocks: u32) -> BatchMeta {
    let (seq, q, c, slots, blocks, win) =
        batch_parts(g as usize, s as usize, t as usize, max_blocks as usize);
    BatchMeta::builder(g, s, t, max_blocks)
        .seq_ids(seq)
        .query_len(q)
        .ctx_len(c)
        .positions(Positions::PerToken(vec![0; t as usize]))
        .slot_map(slots)
        .block_table(blocks)
        .window_start(win)
        .build()
        .expect("valid batch builds")
}

#[test]
fn tensor_rank_0_rejected() {
    let err = Tensor::new(
        vec![],
        DType::F32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect_err("rank 0 must be rejected");
    assert_eq!(err, IrError::InvalidRank { got: 0 });
}

#[test]
fn tensor_rank_5_rejected() {
    let err = Tensor::new(
        vec![Dim::Concrete(1); 5],
        DType::F32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect_err("rank 5 must be rejected");
    assert_eq!(err, IrError::InvalidRank { got: 5 });
}

#[test]
fn tensor_zero_extent_rejected() {
    let err = Tensor::new(
        vec![Dim::Concrete(8), Dim::Concrete(0)],
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect_err("zero extent must be rejected");
    assert_eq!(err, IrError::ZeroExtent { axis: 1 });
}

#[test]
fn tensor_host_placement_requires_weight_class() {
    for class in [
        Class::Activation,
        Class::State,
        Class::Staging,
        Class::Param,
    ] {
        let err = Tensor::new(
            vec![Dim::Concrete(4)],
            DType::F16,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            Placement::Host,
            ShardLayout::Replicated,
            class,
        )
        .expect_err("Host on non-Weight must be rejected");
        assert_eq!(
            err,
            IrError::PlacementForClass {
                placement: Placement::Host,
                class,
            }
        );
    }
    let w = Tensor::new(
        vec![Dim::Concrete(4)],
        DType::I8,
        QuantScheme::PerRow,
        LayoutId::L0,
        Placement::Tiered,
        ShardLayout::Replicated,
        Class::Weight,
    );
    assert!(w.is_ok(), "Tiered Weight must build");
}

#[test]
fn tensor_reports_every_problem_at_once() {
    // Rank 6 AND zero extent AND Host-on-Activation: one error, all problems.
    let err = Tensor::new(
        vec![Dim::Concrete(0); 6],
        DType::F32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Host,
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect_err("must be rejected");
    match err {
        IrError::Multiple { problems } => {
            assert!(problems.contains(&IrError::InvalidRank { got: 6 }));
            for axis in 0..6 {
                assert!(problems.contains(&IrError::ZeroExtent { axis }));
            }
            assert!(problems.contains(&IrError::PlacementForClass {
                placement: Placement::Host,
                class: Class::Activation,
            }));
            assert_eq!(problems.len(), 8);
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
}

#[test]
fn tensor_quantization_class_rules_are_enforced() {
    let activation = Tensor::new(
        vec![Dim::Concrete(8)],
        DType::I8,
        QuantScheme::PerRow,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect_err("PerRow is weights-only");
    assert_eq!(
        activation,
        IrError::QuantForClass {
            quant: QuantScheme::PerRow,
            class: Class::Activation,
        }
    );

    let weight = Tensor::new(
        vec![Dim::Concrete(8)],
        DType::I8,
        QuantScheme::PerToken,
        LayoutId::L1,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .expect_err("PerToken is activations-only");
    assert_eq!(
        weight,
        IrError::QuantForClass {
            quant: QuantScheme::PerToken,
            class: Class::Weight,
        }
    );
}

#[test]
fn tensor_quantized_staging_represents_ngram_rows() {
    let staging = Tensor::new(
        vec![
            Dim::Symbolic(ShapeSymbol::T),
            Dim::Symbolic(ShapeSymbol::Np),
            Dim::Concrete(64),
        ],
        DType::I4,
        QuantScheme::Scheme(SchemeId::new(11)),
        LayoutId::L0,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Staging,
    )
    .expect("Spec 1 §4.A quantized gather_staging must be representable");
    assert_eq!(staging.class(), Class::Staging);
    assert_eq!(staging.dtype(), DType::I4);
}

#[test]
fn tensor_activation_layout_and_quantized_dtype_are_enforced() {
    let layout = Tensor::new(
        vec![Dim::Concrete(8)],
        DType::F16,
        QuantScheme::None,
        LayoutId::L1,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect_err("activations are contiguous");
    assert_eq!(
        layout,
        IrError::LayoutForClass {
            layout: LayoutId::L1,
            class: Class::Activation,
        }
    );

    let dtype = Tensor::new(
        vec![Dim::Concrete(8)],
        DType::F16,
        QuantScheme::PerToken,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect_err("PerToken stores i8 or e4m3 values");
    assert_eq!(
        dtype,
        IrError::QuantDType {
            quant: QuantScheme::PerToken,
            dtype: DType::F16,
        }
    );
}

#[test]
fn tensor_valid_construction_exposes_fields() {
    let t = weight_tensor();
    assert_eq!(t.rank(), 2);
    assert_eq!(t.dtype(), DType::I8);
    assert_eq!(t.quant(), QuantScheme::Scheme(SchemeId::new(11)));
    assert_eq!(t.layout(), LayoutId::L1);
    assert_eq!(t.placement(), Placement::Device { rank: 0 });
    assert_eq!(t.sharding(), ShardLayout::Replicated);
    assert_eq!(t.class(), Class::Weight);
}

#[test]
fn quant_scheme_scale_dtypes_follow_spec() {
    assert_eq!(QuantScheme::None.scale_dtype(), None);
    assert_eq!(QuantScheme::PerRow.scale_dtype(), Some(DType::F16));
    assert_eq!(QuantScheme::PerToken.scale_dtype(), Some(DType::F32));
    assert_eq!(QuantScheme::PerBlock32.scale_dtype(), Some(DType::F32));
    assert_eq!(QuantScheme::Scheme(SchemeId::new(0)).scale_dtype(), None);
}

#[test]
fn batch_zero_dims_rejected_with_numbers() {
    let err = BatchMeta::builder(0, 2, 4, 8)
        .seq_ids(vec![0, 1])
        .query_len(vec![1, 1])
        .ctx_len(vec![0, 0])
        .positions(Positions::PerToken(vec![0; 4]))
        .slot_map(vec![])
        .block_table(vec![BLOCK_TABLE_SENTINEL; 0])
        .window_start(vec![])
        .build()
        .expect_err("G=0 must be rejected");
    // Length errors for the zero-sized fields accompany the dim error.
    match err {
        IrError::Multiple { problems } => {
            assert!(problems.contains(&IrError::ZeroBatchDim {
                g: 0,
                s: 2,
                t: 4,
                max_blocks: 8,
            }));
        }
        IrError::ZeroBatchDim {
            g: 0,
            s: 2,
            t: 4,
            max_blocks: 8,
        } => {}
        other => panic!("expected dim error, got {other:?}"),
    }
}

#[test]
fn batch_builder_missing_fields_reports_all() {
    let err = BatchMeta::builder(1, 1, 1, 1)
        .build()
        .expect_err("empty builder must be rejected");
    match err {
        IrError::Multiple { problems } => {
            for field in [
                "seq_ids",
                "query_len",
                "ctx_len",
                "positions",
                "slot_map",
                "block_table",
                "window_start",
            ] {
                assert!(
                    problems.contains(&IrError::MissingField { field }),
                    "missing {field} not reported: {problems:?}",
                );
            }
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
}

#[test]
fn batch_length_mismatch_reports_every_field() {
    let err = BatchMeta::builder(2, 3, 4, 5)
        .seq_ids(vec![0])
        .query_len(vec![1])
        .ctx_len(vec![0])
        .positions(Positions::PerToken(vec![0]))
        .slot_map(vec![0])
        .block_table(vec![0])
        .window_start(vec![0])
        .build()
        .expect_err("short fields must be rejected");
    match err {
        IrError::Multiple { problems } => {
            assert!(problems.contains(&IrError::BatchLength {
                field: "seq_ids",
                expected: 3,
                actual: 1,
            }));
            assert!(problems.contains(&IrError::BatchLength {
                field: "slot_map",
                expected: 8,
                actual: 1,
            }));
            assert!(problems.contains(&IrError::BatchLength {
                field: "block_table",
                expected: 30,
                actual: 1,
            }));
            assert!(problems.contains(&IrError::BatchLength {
                field: "window_start",
                expected: 6,
                actual: 1,
            }));
            assert!(problems.contains(&IrError::BatchLength {
                field: "positions",
                expected: 4,
                actual: 1,
            }));
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
}

#[test]
fn batch_query_lengths_must_sum_to_t() {
    let (seq, _, c, slots, blocks, win) = batch_parts(1, 2, 3, 1);
    let err = BatchMeta::builder(1, 2, 3, 1)
        .seq_ids(seq)
        .query_len(vec![1, 1])
        .ctx_len(c)
        .positions(Positions::PerToken(vec![0; 3]))
        .slot_map(slots)
        .block_table(blocks)
        .window_start(win)
        .build()
        .expect_err("query lengths totaling 2 cannot describe T=3");
    assert_eq!(
        err,
        IrError::QueryTokenCount {
            expected: 3,
            actual: 2,
        }
    );
}

#[test]
fn batch_empty_query_len_rejected() {
    let (seq, _, c, slots, blocks, win) = batch_parts(1, 2, 2, 1);
    let err = BatchMeta::builder(1, 2, 2, 1)
        .seq_ids(seq)
        .query_len(vec![1, 0])
        .ctx_len(c)
        .positions(Positions::PerToken(vec![0, 1]))
        .slot_map(slots)
        .block_table(blocks)
        .window_start(win)
        .build()
        .expect_err("query_len 0 must be rejected");
    assert_eq!(
        err,
        IrError::Multiple {
            problems: vec![
                IrError::EmptyQuery { seq: 1 },
                IrError::QueryTokenCount {
                    expected: 2,
                    actual: 1,
                },
            ]
            .into_boxed_slice(),
        }
    );
}

#[test]
fn batch_mrope_positions_accepted_at_t() {
    let (seq, q, c, slots, blocks, win) = batch_parts(1, 2, 3, 2);
    let b = BatchMeta::builder(1, 2, 3, 2)
        .seq_ids(seq)
        .query_len(q)
        .ctx_len(c)
        .positions(Positions::Mrope(vec![[0, 1, 2]; 3]))
        .slot_map(slots)
        .block_table(blocks)
        .window_start(win)
        .build()
        .expect("mrope positions at T must build");
    assert_eq!(b.positions().len(), 3);
}

#[test]
fn batch_index_helpers_read_row_major() {
    // G=2, S=2, T=3, max_blocks=2. slot(g,t)=100g+10t... distinct per cell.
    let g = 2u32;
    let s = 2u32;
    let t = 3u32;
    let mb = 2u32;
    let slot_map: Vec<u32> = (0..g)
        .flat_map(|gg| (0..t).map(move |tt| gg * 100 + tt))
        .collect();
    let block_table: Vec<u32> = (0..g * s * mb).map(|i| 1000 + i).collect();
    let window_start: Vec<u32> = (0..g * s).map(|i| 50 + i).collect();
    let (seq, q, c, _, _, _) = batch_parts(g as usize, s as usize, t as usize, mb as usize);
    let b = BatchMeta::builder(g, s, t, mb)
        .seq_ids(seq)
        .query_len(q)
        .ctx_len(c)
        .positions(Positions::PerToken(vec![0; t as usize]))
        .slot_map(slot_map)
        .block_table(block_table)
        .window_start(window_start)
        .build()
        .expect("valid batch builds");
    assert_eq!(b.slot(1, 2), 102);
    assert_eq!(b.slot(0, 0), 0);
    // block(1,0,1): ((1*2+0)*2+1) = 5 -> 1005.
    assert_eq!(b.block(1, 0, 1), 1005);
    // window(1,1): 1*2+1 = 3 -> 53.
    assert_eq!(b.window(1, 1), 53);
}

#[test]
fn batch_tree_t_mismatch_rejected() {
    let tree = TreeMask::new(vec![-1, 0], 3, vec![true; 6]).expect("tree builds");
    let (seq, q, c, slots, blocks, win) = batch_parts(1, 1, 3, 1);
    let err = BatchMeta::builder(1, 1, 3, 1)
        .seq_ids(seq)
        .query_len(q)
        .ctx_len(c)
        .positions(Positions::PerToken(vec![0; 3]))
        .slot_map(slots)
        .block_table(blocks)
        .window_start(win)
        .tree(Some(tree))
        .build()
        .expect_err("tree T=2 on batch T=3 must be rejected");
    assert_eq!(
        err,
        IrError::TreeBatchMismatch {
            tree_t: 2,
            batch_t: 3,
        }
    );
}

#[test]
fn tree_bad_parent_rejected() {
    let err = TreeMask::new(vec![-1, 5], 1, vec![true; 2]).expect_err("parent 5 >= T=2 rejected");
    assert_eq!(
        err,
        IrError::BadParent {
            token: 1,
            parent: 5,
            t: 2,
        }
    );
    let err = TreeMask::new(vec![-2, -1], 1, vec![true; 2]).expect_err("parent -2 rejected");
    assert_eq!(
        err,
        IrError::BadParent {
            token: 0,
            parent: -2,
            t: 2,
        }
    );
}

#[test]
fn tree_self_parent_rejected() {
    let err = TreeMask::new(vec![0, -1], 1, vec![true; 2]).expect_err("self-parent rejected");
    assert_eq!(err, IrError::SelfParent { token: 0 });
}

#[test]
fn tree_parent_cycle_rejected() {
    let err = TreeMask::new(vec![-1, 2, 1], 3, vec![true; 9]).expect_err("parent cycle rejected");
    assert_eq!(err, IrError::TreeCycle { token: 1 });
}

#[test]
fn tree_zero_column_width_rejected() {
    let err = TreeMask::new(vec![-1], 0, vec![]).expect_err("non-empty tree needs columns");
    assert_eq!(err, IrError::ZeroTreeMax { t: 1 });
}

#[test]
fn tree_parent_cannot_cross_sequence_boundary() {
    let tree = TreeMask::new(vec![-1, 0, 0, -1], 2, vec![true; 8]).expect("forest builds");
    let (seq, _, c, slots, blocks, win) = batch_parts(1, 2, 4, 1);
    let err = BatchMeta::builder(1, 2, 4, 1)
        .seq_ids(seq)
        .query_len(vec![2, 2])
        .ctx_len(c)
        .positions(Positions::PerToken(vec![0; 4]))
        .slot_map(slots)
        .block_table(blocks)
        .window_start(win)
        .tree(Some(tree))
        .build()
        .expect_err("a parent cannot point into another sequence");
    assert_eq!(
        err,
        IrError::TreeParentCrossesSequence {
            token: 2,
            parent: 0,
            seq: 1,
            seq_start: 2,
            seq_end: 4,
        }
    );
}

#[test]
fn tree_columns_cover_longest_query() {
    let tree = TreeMask::new(vec![-1, 0, 1], 2, vec![true; 6]).expect("tree builds");
    let (seq, q, c, slots, blocks, win) = batch_parts(1, 1, 3, 1);
    let err = BatchMeta::builder(1, 1, 3, 1)
        .seq_ids(seq)
        .query_len(q)
        .ctx_len(c)
        .positions(Positions::PerToken(vec![0; 3]))
        .slot_map(slots)
        .block_table(blocks)
        .window_start(win)
        .tree(Some(tree))
        .build()
        .expect_err("t_max must cover the query");
    assert_eq!(
        err,
        IrError::TreeMaxTooSmall {
            required: 3,
            actual: 2,
        }
    );
}

#[test]
fn tree_ancestor_length_rejected() {
    let err = TreeMask::new(vec![-1, 0], 3, vec![true; 5]).expect_err("5 != 2*3 rejected");
    assert_eq!(
        err,
        IrError::AncestorLength {
            t: 2,
            t_max: 3,
            expected: 6,
            actual: 5,
        }
    );
}

#[test]
fn tree_valid_construction_and_lookup() {
    // Chain 0 <- 1 <- 2, t_max=3, row i marks self + ancestors.
    let ancestors = vec![
        true, false, false, // tok 0: {0}
        true, true, false, // tok 1: {0,1}
        true, true, true, // tok 2: {0,1,2}
    ];
    let tree = TreeMask::new(vec![-1, 0, 1], 3, ancestors).expect("chain builds");
    assert_eq!(tree.t(), 3);
    assert_eq!(tree.t_max(), 3);
    assert_eq!(tree.parents(), &[-1, 0, 1]);
    assert!(tree.is_ancestor(2, 0));
    assert!(!tree.is_ancestor(0, 2));
}

#[test]
fn relrate_rejects_nonpositive_nonnumeric() {
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = RelRate::new(bad).expect_err("bad rate rejected");
        match err {
            IrError::NonPositiveRate { .. } => {}
            other => panic!("expected NonPositiveRate, got {other:?}"),
        }
    }
}

#[test]
fn matrix_op_rejects_invalid_accumulator() {
    let rate = RelRate::new(1.0).expect("positive rate builds");
    let err = r9v_ir::MatrixOp::new([16, 16, 16], DType::F16, DType::F16, DType::Bf16, rate)
        .expect_err("matrix accumulation must be f32 or i32");
    assert_eq!(err, IrError::InvalidAccumulator { got: DType::Bf16 });
}

#[test]
fn gfx1201_initial_values_match_spec() {
    let a = ArchDescriptor::gfx1201();
    assert_eq!(a.family, ArchFamily::Rdna4);
    assert_eq!(a.wave_size, 32);
    assert_eq!(a.lds_bytes_per_wg, 64 * 1024);
    assert_eq!(a.vgprs_per_lane, 256);
    assert_eq!(a.max_wg_size, 1024);
    assert!(a.fp8_convert);
    assert!(a.sparse_matrix);
    assert_eq!(a.valu_dot, vec![ValuDot::Dot4I32I8]);
    assert_eq!(a.fragment_layout, LayoutId::L1);
    // f16/bf16 at 1x, fp8/iu8/iu4 at 2x nominal (Spec 1 App. A).
    let rates: Vec<f32> = a.matrix_ops.iter().map(|m| m.rate.as_f32()).collect();
    assert_eq!(rates, vec![1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
    for m in &a.matrix_ops {
        assert!(matches!(m.acc, DType::F32 | DType::I32));
    }
    assert_eq!(a.matrix_ops.last().map(|op| op.shape), Some([16, 16, 32]));
}

#[test]
fn cpu_reports_reference_identity() {
    let c = ArchDescriptor::cpu();
    assert_eq!(c.family, ArchFamily::Cpu);
    assert_eq!(c.name, "cpu");
    assert!(c.matrix_ops.is_empty());
    assert!(c.valu_dot.is_empty());
    assert!(!c.fp8_convert);
    assert!(!c.sparse_matrix);

    let device = r9v_ir::DeviceDescriptor::cpu();
    assert_eq!(device.arch, c);
    assert_eq!(device.facts.identity, r9v_ir::DeviceIdentity::Cpu);
    assert_eq!(device.facts.graph_capture, GraphCapture::None);
    assert!(device.measured.is_empty());
    assert!(device.p2p.is_empty());
}

#[test]
fn ir_version_orders_and_displays() {
    assert_eq!(IrVersion::CURRENT.to_string(), "0.2.0");
    assert!(IrVersion::new(0, 3, 0) > IrVersion::CURRENT);
    assert!(IrVersion::new(1, 0, 0) > IrVersion::new(0, 99, 99));
}

#[test]
fn state_handle_names_layer_and_kind() {
    let h = StateHandle::new(7, StateKind::Recurrent);
    assert_eq!(h.layer(), 7);
    assert_eq!(h.kind(), StateKind::Recurrent);
    assert_eq!(h, StateHandle::new(7, StateKind::Recurrent));
    assert_ne!(h, StateHandle::new(7, StateKind::ConvWindow));
}

#[test]
fn seeded_batches_build_identically_twice() {
    // Determinism: same seed, same dims and ids, bit-identical metadata.
    fn build(seed: u64) -> BatchMeta {
        let mut rng = SeededRng::new(seed);
        let g = (1 + rng.next_u64() % 3) as u32;
        let s = (1 + rng.next_u64() % 7) as u32;
        let t = s + (rng.next_u64() % 15) as u32;
        let mb = (1 + rng.next_u64() % 7) as u32;
        let (seq, q, c, slots, blocks, win) =
            batch_parts(g as usize, s as usize, t as usize, mb as usize);
        BatchMeta::builder(g, s, t, mb)
            .seq_ids(seq)
            .query_len(q)
            .ctx_len(c)
            .positions(Positions::PerToken(vec![0; t as usize]))
            .slot_map(slots)
            .block_table(blocks)
            .window_start(win)
            .build()
            .expect("seeded batch builds")
    }
    assert_eq!(build(0xC10C), build(0xC10C));
    assert_eq!(valid_batch(2, 4, 8, 4), valid_batch(2, 4, 8, 4));
}

#[test]
fn seeded_tensor_shapes_vary_without_breaking_invariants() {
    // Same seed replays the same shape stream; every shape still validates.
    let mut rng = SeededRng::new(0xA11CE);
    for _ in 0..32 {
        let rank = (1 + rng.next_u64() % 4) as usize;
        let shape: Vec<Dim> = (0..rank)
            .map(|_| Dim::Concrete((1 + rng.next_u64() % 128) as u32))
            .collect();
        let t = Tensor::new(
            shape.clone(),
            DType::F16,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )
        .expect("seeded shape builds");
        assert_eq!(t.shape(), shape.as_slice());
    }
    // Symbolic dims ride along untouched.
    let t = Tensor::new(
        vec![
            Dim::Symbolic(ShapeSymbol::T),
            Dim::Symbolic(ShapeSymbol::Dm),
        ],
        DType::I8,
        QuantScheme::PerToken,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .expect("symbolic shape builds");
    assert_eq!(t.rank(), 2);
}
