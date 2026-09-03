// SPDX-License-Identifier: Apache-2.0
//! IR-level oracles for card A1.14: BatchMeta.positions projections, strict
//! rope positions wiring, the split/concat/softcap extension ops, scaled
//! residuals, and the exact MLA state-write pair (SI-18 through SI-23).

use r9v_ir::{
    CacheScaleGranularity, Class, DType, Dim, EdgeId, EmbedGatherOp, ExternalInputKind, Graph,
    IrError, LayoutId, LogitSoftcapOp, MlaLatent, Op, Placement, PlanId, PositionsKind,
    QuantScheme, ResidualAddOp, RopeOp, RopeScaling, RopeStyle, ShapeSymbol, ShardLayout, SplitOp,
    StateHandle, StateKind, StateWriteKvOp, StepGraphKey, Tensor,
};

fn act_tensor(shape: Vec<Dim>, dtype: DType) -> Tensor {
    Tensor::new(
        shape,
        dtype,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap()
}

fn batch_meta_graph() -> Graph {
    let key = StepGraphKey::new(PlanId::new(0), 0, 1, 1, 0, 0).unwrap();
    let mut graph = Graph::new(key);
    graph
        .add_external_non_tensor(ExternalInputKind::BatchMeta)
        .unwrap();
    graph
}

fn rope_op() -> RopeOp {
    RopeOp {
        rot_dim: 8,
        theta: 10000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F16,
    }
}

/// Builds a graph with BatchMeta plus a `[T, 2, 8]` activation `x` produced
/// the honest way (token IDs, embedding weight, gather, reshape), returning
/// the graph and `x`'s edge for rope wiring.
fn rope_graph_with_x() -> (Graph, EdgeId) {
    let mut graph = batch_meta_graph();
    let tokens = graph
        .add_external_input(
            ExternalInputKind::TokenIds,
            act_tensor(vec![Dim::Symbolic(ShapeSymbol::T)], DType::U32),
        )
        .unwrap();
    let embed = graph
        .add_tensor(
            Tensor::new(
                vec![Dim::Concrete(16), Dim::Concrete(16)],
                DType::F16,
                QuantScheme::None,
                LayoutId::CONTIGUOUS,
                Placement::Device { rank: 0 },
                ShardLayout::Replicated,
                Class::Weight,
            )
            .unwrap(),
        )
        .unwrap();
    let gathered = act_tensor(
        vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(16)],
        DType::F16,
    );
    graph
        .add_op(
            Op::EmbedGather(EmbedGatherOp {
                scale: 1.0,
                out_dtype: DType::F16,
            }),
            &[tokens, embed],
            &[gathered],
        )
        .unwrap();
    let gathered_edge = graph.nodes()[0].outputs[0];
    let x = graph
        .reshape_edge(
            gathered_edge,
            vec![
                Dim::Symbolic(ShapeSymbol::T),
                Dim::Concrete(2),
                Dim::Concrete(8),
            ],
        )
        .unwrap();
    (graph, x)
}

#[test]
fn positions_scalar_binding_has_exact_descriptor() {
    let mut graph = batch_meta_graph();
    let edge = graph.bind_positions(PositionsKind::Scalar).unwrap();
    let tensor = &graph.edges()[edge.0].tensor;
    assert_eq!(tensor.shape(), &[Dim::Symbolic(ShapeSymbol::T)]);
    assert_eq!(tensor.dtype(), DType::U32);
    assert_eq!(tensor.class(), Class::Activation);
    assert_eq!(
        graph.positions_binding(),
        Some((PositionsKind::Scalar, edge))
    );
}

#[test]
fn positions_mrope_binding_has_exact_descriptor() {
    let mut graph = batch_meta_graph();
    let edge = graph.bind_positions(PositionsKind::Mrope).unwrap();
    let tensor = &graph.edges()[edge.0].tensor;
    assert_eq!(
        tensor.shape(),
        &[Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(3)]
    );
    assert_eq!(tensor.dtype(), DType::U32);
    assert_eq!(
        graph.positions_binding(),
        Some((PositionsKind::Mrope, edge))
    );
}

#[test]
fn positions_duplicate_and_conflicting_bindings_rejected() {
    let mut graph = batch_meta_graph();
    graph.bind_positions(PositionsKind::Scalar).unwrap();
    let err = graph.bind_positions(PositionsKind::Scalar).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::PositionsConflict {
                existing: PositionsKind::Scalar,
                requested: PositionsKind::Scalar
            }
        ),
        "duplicate scalar binding: {err}"
    );

    let mut graph = batch_meta_graph();
    graph.bind_positions(PositionsKind::Scalar).unwrap();
    let err = graph.bind_positions(PositionsKind::Mrope).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::PositionsConflict {
                existing: PositionsKind::Scalar,
                requested: PositionsKind::Mrope
            }
        ),
        "conflicting binding: {err}"
    );

    // Binding without the structured BatchMeta input names it.
    let key = StepGraphKey::new(PlanId::new(0), 0, 1, 1, 0, 0).unwrap();
    let mut bare = Graph::new(key);
    let err = bare.bind_positions(PositionsKind::Scalar).unwrap_err();
    assert!(
        matches!(
            err,
            IrError::GraphExternalInputMissing {
                kind: ExternalInputKind::BatchMeta,
                ..
            }
        ),
        "missing BatchMeta: {err}"
    );
}

fn rope_y() -> Tensor {
    act_tensor(
        vec![
            Dim::Symbolic(ShapeSymbol::T),
            Dim::Concrete(2),
            Dim::Concrete(8),
        ],
        DType::F16,
    )
}

/// Returns the token-IDs edge of a helper-built graph.
fn token_edge(graph: &Graph) -> EdgeId {
    graph
        .external_inputs()
        .iter()
        .find_map(|i| match i {
            r9v_ir::ExternalInput::Tensor { kind, edge }
                if *kind == ExternalInputKind::TokenIds =>
            {
                Some(*edge)
            }
            _ => None,
        })
        .expect("token input present")
}

#[test]
fn rope_without_projection_fails_graph_validation() {
    let (mut graph, x) = rope_graph_with_x();
    let tokens = token_edge(&graph);
    graph
        .add_op(Op::Rope(rope_op()), &[x, tokens], &[rope_y()])
        .unwrap();
    let err = graph.validate().unwrap_err();
    let flat = format!("{err:?}");
    assert!(
        flat.contains("GraphPositionsMissing"),
        "unbound projection named: {err}"
    );
}

#[test]
fn rope_reading_non_projection_edge_fails_graph_validation() {
    let (mut graph, x) = rope_graph_with_x();
    graph.bind_positions(PositionsKind::Scalar).unwrap();
    // A token-IDs edge has a plausible descriptor but the wrong identity.
    let tokens = token_edge(&graph);
    graph
        .add_op(Op::Rope(rope_op()), &[x, tokens], &[rope_y()])
        .unwrap();
    let err = graph.validate().unwrap_err();
    assert!(
        matches!(err, IrError::GraphRopePositionsMismatch { .. }),
        "token-IDs positions rejected: {err}"
    );
}

#[test]
fn rope_reading_projection_edge_validates() {
    let (mut graph, x) = rope_graph_with_x();
    let pos = graph.bind_positions(PositionsKind::Scalar).unwrap();
    graph
        .add_op(Op::Rope(rope_op()), &[x, pos], &[rope_y()])
        .unwrap();
    graph.validate().expect("projection-fed rope validates");
}

#[test]
fn split_concat_validate_exact_widths() {
    let x = act_tensor(
        vec![Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(8)],
        DType::F16,
    );
    let a = act_tensor(
        vec![Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(5)],
        DType::F16,
    );
    let b = act_tensor(
        vec![Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(3)],
        DType::F16,
    );
    SplitOp { first: 5 }
        .validate(std::slice::from_ref(&x), &[a.clone(), b.clone()])
        .expect("exact split validates");
    r9v_ir::ConcatOp
        .validate(&[a.clone(), b.clone()], std::slice::from_ref(&x))
        .expect("exact concat validates");

    // Wrong first width.
    let err = SplitOp { first: 4 }
        .validate(std::slice::from_ref(&x), &[a.clone(), b.clone()])
        .unwrap_err();
    assert!(!format!("{err:?}").is_empty(), "wrong split width rejected");

    // Widths that do not reconstruct the input.
    let wide = act_tensor(
        vec![Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(9)],
        DType::F16,
    );
    let err = r9v_ir::ConcatOp
        .validate(&[a, b], std::slice::from_ref(&wide))
        .unwrap_err();
    assert!(
        !format!("{err:?}").is_empty(),
        "concat width mismatch rejected"
    );

    // Degenerate split widths.
    for first in [0u32, 8] {
        let err = SplitOp { first }
            .validate(
                std::slice::from_ref(&x),
                &[
                    act_tensor(
                        vec![Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(4)],
                        DType::F16,
                    ),
                    act_tensor(
                        vec![Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(4)],
                        DType::F16,
                    ),
                ],
            )
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("first"),
            "split first={first} rejected: {err}"
        );
    }
}

#[test]
fn logit_softcap_validates_cap_and_f32_rank2() {
    let x = act_tensor(vec![Dim::Concrete(4), Dim::Concrete(16)], DType::F32);
    LogitSoftcapOp { cap: 30.0 }
        .validate(std::slice::from_ref(&x), std::slice::from_ref(&x))
        .expect("valid softcap validates");
    for cap in [0.0, -2.0, f32::NAN, f32::INFINITY] {
        let err = LogitSoftcapOp { cap }
            .validate(std::slice::from_ref(&x), std::slice::from_ref(&x))
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("cap"),
            "cap {cap} rejected: {err}"
        );
    }
    let x16 = act_tensor(vec![Dim::Concrete(4), Dim::Concrete(16)], DType::F16);
    let err = LogitSoftcapOp { cap: 30.0 }
        .validate(std::slice::from_ref(&x16), std::slice::from_ref(&x16))
        .unwrap_err();
    assert!(!format!("{err:?}").is_empty(), "non-f32 softcap rejected");
}

#[test]
fn residual_add_scale_validates_finite_nonzero() {
    let a = act_tensor(vec![Dim::Concrete(4), Dim::Concrete(8)], DType::F16);
    ResidualAddOp {
        out_dtype: DType::F16,
        scale: 2.5,
    }
    .validate(&[a.clone(), a.clone()], std::slice::from_ref(&a))
    .expect("non-unit scale validates");
    for scale in [0.0, -0.0, f32::NAN, f32::INFINITY] {
        let err = ResidualAddOp {
            out_dtype: DType::F16,
            scale,
        }
        .validate(&[a.clone(), a.clone()], std::slice::from_ref(&a))
        .unwrap_err();
        assert!(
            format!("{err:?}").contains("scale"),
            "scale {scale} rejected: {err}"
        );
    }
}

#[test]
fn state_write_accepts_exact_mla_split_pair() {
    let latent = MlaLatent {
        kv_lora_rank: 16,
        rope_dim: 8,
    };
    let k = act_tensor(
        vec![Dim::Concrete(2), Dim::Concrete(1), Dim::Concrete(8)],
        DType::F16,
    );
    let v = act_tensor(
        vec![Dim::Concrete(2), Dim::Concrete(1), Dim::Concrete(16)],
        DType::F16,
    );
    let op = || StateWriteKvOp {
        cache_dtype: DType::F16,
        scale_granularity: CacheScaleGranularity::PerTokenHead,
        latent: Some(latent),
        handle: StateHandle::new(0, StateKind::KvLatent),
    };
    op().validate(&[k.clone(), v.clone()], &[])
        .expect("exact (rope, latent) split pair validates");

    // Swapped or wrong widths are rejected.
    let err = op().validate(&[v, k], &[]).unwrap_err();
    assert!(
        !format!("{err:?}").is_empty(),
        "mismatched split pair rejected: {err}"
    );
}

#[test]
fn subgraph_hidden_input_validates_as_token_activation() {
    let mut graph = batch_meta_graph();
    let hidden = graph
        .add_external_input(
            ExternalInputKind::SubgraphHidden,
            act_tensor(
                vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(64)],
                DType::F16,
            ),
        )
        .expect("subgraph hidden input binds");
    assert!(graph.edges()[hidden.0].tensor.shape().len() == 2);
    graph.validate().expect("lone capture input validates");
}
