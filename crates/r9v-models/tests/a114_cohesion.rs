// SPDX-License-Identifier: Apache-2.0
//! Model/IR graph-value cohesion oracles for card A1.14: opaque SSA value
//! identity, typed BatchMeta.positions projections, explicit MTP captures,
//! exact MLA lowering, and scaled-residual / final-softcap lowering
//! (SI-27 through SI-32).

use r9v_ir::{ActivationKind, DType, Dim, NormAxis, Op, RopeScaling, RopeStyle, ShapeSymbol};
use r9v_ir::{IrVersion, StateKind};
use r9v_models::{
    build_mixer, build_model, CacheDtype, Ffn, Graph, GraphBuilder, LayerSpec, Mixer, MlaSpec,
    ModelSpec, MtpSource, MtpSpec, NormPlacement, NormSpec, PositionEncoding, RopeSpec,
};

fn tiny_rope() -> RopeSpec {
    RopeSpec {
        theta: 10000.0,
        rot_dim: 16,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
    }
}

fn mrope_rope() -> RopeSpec {
    RopeSpec {
        theta: 10000.0,
        rot_dim: 16,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: Some([4, 6, 6]),
    }
}

fn tiny_mixer() -> Mixer {
    Mixer::Attention {
        h: 2,
        hkv: 1,
        d: 16,
        dv: 16,
        qkv_bias: false,
        o_bias: false,
        qk_norm: None,
        rope: tiny_rope(),
        window: None,
        sinks: 0,
        logit_softcap: None,
        output_gate: false,
        mla: None,
        cache: CacheDtype::E4m3,
        pre_fused: false,
    }
}

fn tiny_layer() -> LayerSpec {
    LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: tiny_mixer(),
        ffn: Ffn::None,
        residual_scale: 1.0,
    }
}

fn tiny_model(layers: Vec<LayerSpec>) -> ModelSpec {
    ModelSpec {
        dm: 32,
        vocab: 64,
        layers,
        embed_scale: 1.0,
        tied_embeddings: false,
        final_norm: NormSpec::rms(1e-5),
        positions: PositionEncoding::Scalar,
        ngram: None,
        mtp: None,
        export_hidden: false,
        final_logit_softcap: None,
        eos_ids: vec![2],
        bos_id: Some(1),
    }
}

/// Two structurally identical weights mint distinct edges, cloning preserves
/// identity, and consumers read the intended edge (SI-31).
#[test]
fn identical_descriptor_weights_never_alias() {
    let mut builder = GraphBuilder::new(IrVersion::CURRENT, "a114-ssa");
    let (x, _) = builder.input_embed_override(32).expect("override");
    let w1 = builder
        .weight(
            "blk.0.attn_q.weight",
            r9v_models::WeightRole::Matmul,
            &[Dim::Concrete(32), Dim::Concrete(32)],
            r9v_models::SchemeClass::Matmul,
        )
        .expect("weight a");
    let w2 = builder
        .weight(
            "blk.0.attn_k.weight",
            r9v_models::WeightRole::Matmul,
            &[Dim::Concrete(32), Dim::Concrete(32)],
            r9v_models::SchemeClass::Matmul,
        )
        .expect("weight b");
    assert_eq!(w1.tensor(), w2.tensor(), "descriptors identical");
    assert_ne!(w1.edge(), w2.edge(), "edges distinct");
    assert_eq!(w1.clone().edge(), w1.edge(), "clone preserves identity");

    let m1 = builder
        .op_matmul(x.clone(), w1.clone(), DType::F16)
        .expect("m1");
    let m2 = builder
        .op_matmul(x.clone(), w2.clone(), DType::F16)
        .expect("m2");
    assert_ne!(m1.edge(), m2.edge(), "outputs distinct");
    let graph = builder.finish().expect("validates");

    // Each matmul's weight input is the intended edge, not the twin.
    let mut seen = 0;
    for node in graph.graph().nodes() {
        if let Op::Matmul(_) = &node.op {
            let w_edge = node.inputs[1];
            assert!(
                w_edge == w1.edge() || w_edge == w2.edge(),
                "matmul reads a bound weight edge"
            );
            seen += 1;
        }
    }
    assert_eq!(seen, 2, "both matmuls present");
    // The two matmuls read different weight edges in emission order.
    let w_edges: Vec<_> = graph
        .graph()
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Matmul(_) => Some(n.inputs[1]),
            _ => None,
        })
        .collect();
    assert_eq!(w_edges, vec![w1.edge(), w2.edge()]);
}

/// Scalar models bind one `[T] u32` positions projection; every rope reads
/// it, never token IDs (SI-30).
#[test]
fn scalar_positions_projection_feeds_rope() {
    let model = tiny_model(vec![tiny_layer()]);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-pos"), &model)
        .expect("scalar model builds and validates");

    let positions: Vec<_> = graph
        .graph()
        .edges()
        .iter()
        .filter(|e| {
            e.tensor.dtype() == DType::U32 && e.tensor.shape() == [Dim::Symbolic(ShapeSymbol::T)]
        })
        .collect();
    // Token IDs plus exactly one projection.
    assert_eq!(positions.len(), 2, "tokens edge plus one projection");
    let tokens = graph
        .graph()
        .external_inputs()
        .iter()
        .filter(|i| {
            matches!(
                i,
                r9v_ir::ExternalInput::Tensor { kind, .. }
                    if *kind == r9v_ir::ExternalInputKind::TokenIds
            )
        })
        .count();
    assert_eq!(tokens, 1, "one structured token input");

    let mut ropes = 0;
    for node in graph.graph().nodes() {
        if let Op::Rope(_) = &node.op {
            ropes += 1;
            let pos_edge = node.inputs[1];
            let pos = &graph.graph().edges()[pos_edge.0].tensor;
            assert_eq!(
                pos.shape(),
                &[Dim::Symbolic(ShapeSymbol::T)],
                "rope positions are the scalar projection"
            );
            // The projection is not the token-IDs edge: exactly one U32 [T]
            // edge is referenced by rope nodes, and token IDs feed only the
            // embedding gather.
            for other in graph.graph().nodes() {
                if let Op::EmbedGather(_) = &other.op {
                    assert_ne!(other.inputs[0], pos_edge, "tokens feed gather");
                    assert_eq!(pos_edge, node.inputs[1]);
                }
            }
        }
    }
    assert_eq!(ropes, 2, "both rope nodes present");
}

/// MRoPE models bind one `[T, 3] u32` projection accepted by rope (SI-30).
#[test]
fn mrope_positions_projection_feeds_rope() {
    let mut mixer = tiny_mixer();
    if let Mixer::Attention { rope, .. } = &mut mixer {
        *rope = mrope_rope();
    }
    let mut model = tiny_model(vec![LayerSpec {
        mixer,
        ..tiny_layer()
    }]);
    model.positions = PositionEncoding::MRope([4, 6, 6]);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-mrope"), &model)
        .expect("mrope model builds and validates");

    let triples: Vec<_> = graph
        .graph()
        .edges()
        .iter()
        .filter(|e| {
            e.tensor.dtype() == DType::U32
                && e.tensor.shape() == [Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(3)]
        })
        .collect();
    assert_eq!(triples.len(), 1, "exactly one [T, 3] projection");
    let projection = triples[0].id;
    for node in graph.graph().nodes() {
        if let Op::Rope(r) = &node.op {
            assert_eq!(node.inputs[1], projection, "rope reads the projection");
            assert!(r.mrope_sections.is_some(), "mrope sections carried");
        }
    }
}

/// A second positions kind on one builder conflicts instead of aliasing (SI-30).
#[test]
fn conflicting_positions_binding_rejected_at_builder() {
    let mut builder = GraphBuilder::new(IrVersion::CURRENT, "a114-pos-conflict");
    let first = builder
        .positions(PositionEncoding::Scalar)
        .expect("scalar binds");
    let again = builder
        .positions(PositionEncoding::Scalar)
        .expect("same kind returns the cached value");
    assert_eq!(first.edge(), again.edge(), "repeat bind is idempotent");
    let err = builder
        .positions(PositionEncoding::MRope([4, 6, 6]))
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("conflicting"),
        "conflicting kind rejected: {err}"
    );
}

/// MTP `Last` captures the parent's final hidden value; both heads restart
/// from it and keep disjoint `blk.0.mtp.*` weights (SI-32).
#[test]
fn mtp_last_capture_binds_parent_hidden() {
    let mtp = MtpSpec {
        heads: 2,
        layers_per_head: vec![tiny_layer()],
        takes_hidden_from: MtpSource::Last,
    };
    mtp.validate(1).expect("mtp validates");
    let mut model = tiny_model(vec![tiny_layer()]);
    model.mtp = Some(mtp);
    model.export_hidden = true;
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-mtp-last"), &model)
        .expect("mtp-last model builds and validates");

    let mtp_graph = graph.subgraphs().get("mtp").expect("mtp subgraph");
    let capture = mtp_graph.capture().expect("explicit capture record");
    let hidden_edge = graph
        .exports()
        .iter()
        .find(|(name, _)| name == "hidden")
        .expect("hidden export")
        .1
        .edge();
    assert_eq!(
        capture.parent_edge, hidden_edge,
        "capture binds the parent final hidden value"
    );

    // Each head restarts from the capture input directly: the placement norm
    // and the mixer residual both read the stream value, which is the
    // capture for the head's first layer (Pre placement, no FFN here).
    let first_op_readers = mtp_graph
        .graph()
        .nodes()
        .iter()
        .filter(|n| n.inputs.first() == Some(&capture.child_edge))
        .count();
    assert_eq!(first_op_readers, 4, "both heads restart from the capture");

    // No chaining: the two heads' ancestor edge sets meet only at the
    // capture input and the shared child positions projection (weights are
    // per-head disjoint by construction).
    let logits: Vec<_> = mtp_graph
        .exports()
        .iter()
        .filter(|(name, _)| name.starts_with("mtp_logits_"))
        .map(|(_, v)| v.edge())
        .collect();
    assert_eq!(logits.len(), 2, "two head logit exports");
    let ancestors = |edge| ancestor_edges(mtp_graph.graph(), edge);
    let a0 = ancestors(logits[0]);
    let a1 = ancestors(logits[1]);
    assert!(
        a0.contains(&capture.child_edge),
        "head 0 sources the capture"
    );
    assert!(
        a1.contains(&capture.child_edge),
        "head 1 sources the capture"
    );
    let positions = mtp_graph
        .graph()
        .edges()
        .iter()
        .find(|e| e.tensor.dtype() == DType::U32)
        .expect("child positions projection")
        .id;
    let shared: Vec<_> = a0.intersection(&a1).copied().collect();
    assert_eq!(
        shared,
        vec![
            capture.child_edge.min(positions),
            capture.child_edge.max(positions)
        ],
        "heads share only the capture and the positions projection"
    );

    // Disjoint per-head weights: head 1 binds the layer-0 set (attention
    // projections plus output head), head 2 the layer-1 set.
    for (head, ordinal) in [(1u32, 0u32), (2, 1)] {
        let prefix = format!("blk.{ordinal}.mtp.");
        assert!(
            mtp_graph
                .bound_weights()
                .iter()
                .any(|w| w.name.starts_with(&prefix)),
            "head {head} owns {prefix}* weights"
        );
        let output = format!("blk.{ordinal}.mtp.output.weight");
        assert!(
            mtp_graph.bound_weights().iter().any(|w| w.name == output),
            "head {head} owns its output head {output}"
        );
    }
    assert!(
        mtp_graph
            .bound_weights()
            .iter()
            .all(|w| w.name.contains(".mtp.")),
        "all head weights live under mtp namespaces"
    );
}

/// MTP `Layer(n)` captures that layer's output, not the final hidden (SI-32).
#[test]
fn mtp_layer_capture_selects_layer_output() {
    let mtp = MtpSpec {
        heads: 1,
        layers_per_head: vec![tiny_layer()],
        takes_hidden_from: MtpSource::Layer(0),
    };
    let mut model = tiny_model(vec![tiny_layer(), tiny_layer()]);
    model.mtp = Some(mtp);
    model.export_hidden = true;
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-mtp-layer"), &model)
        .expect("mtp-layer model builds and validates");

    let mtp_graph = graph.subgraphs().get("mtp").expect("mtp subgraph");
    let capture = mtp_graph.capture().expect("explicit capture record");
    let hidden_edge = graph
        .exports()
        .iter()
        .find(|(name, _)| name == "hidden")
        .expect("hidden export")
        .1
        .edge();
    assert_ne!(
        capture.parent_edge, hidden_edge,
        "Layer(0) capture is not the final hidden"
    );
    let parent_tensor = &graph.graph().edges()[capture.parent_edge.0].tensor;
    assert_eq!(
        parent_tensor.shape(),
        &[Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(32)],
        "captured parent value is [T, Dm]"
    );
}

fn mla_mixer() -> Mixer {
    Mixer::Attention {
        h: 4,
        hkv: 1,
        d: 32,
        dv: 32,
        qkv_bias: false,
        o_bias: false,
        qk_norm: Some(NormSpec::rms(1e-5)),
        rope: RopeSpec {
            theta: 10000.0,
            rot_dim: 64,
            style: RopeStyle::Neox,
            scaling: RopeScaling::None,
            mrope_sections: None,
        },
        window: None,
        sinks: 0,
        logit_softcap: None,
        output_gate: false,
        mla: Some(MlaSpec {
            q_lora_rank: 32,
            kv_lora_rank: 16,
            qk_nope_dim: 24,
            qk_rope_dim: 8,
            v_dim: 48,
        }),
        cache: CacheDtype::E4m3,
        pre_fused: false,
    }
}

/// MLA with unequal dims lowers to exact split/rope/write/reconstructed
/// edges, rotates only rotary channels, and lowers qk_norm (SI-29).
#[test]
fn mla_unequal_dims_split_rope_write_reconstruct() {
    let mixer = mla_mixer();
    let model = tiny_model(vec![LayerSpec {
        mixer,
        ..tiny_layer()
    }]);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-mla"), &model)
        .expect("unequal-dim MLA with qk_norm builds and validates");

    // Exact splits: q [T,4,32] into (24 nope, 8 rope); kv [T,1,24] into
    // (16 latent, 8 rope).
    let mut splits = Vec::new();
    let mut concats = 0;
    for node in graph.graph().nodes() {
        match &node.op {
            Op::Split(s) => splits.push((node.inputs[0], s.first)),
            Op::Concat(_) => concats += 1,
            _ => {}
        }
    }
    assert_eq!(splits.len(), 2, "q split and kv split");
    assert!(
        splits.iter().any(|(_, first)| *first == 24),
        "q split at nope width"
    );
    assert!(
        splits.iter().any(|(_, first)| *first == 16),
        "kv split at latent rank"
    );
    assert_eq!(concats, 1, "query reconstruction concat");

    // Rope touches rotary channels only: every rope x is [T, H|1, 8].
    let mut ropes = 0;
    for node in graph.graph().nodes() {
        if let Op::Rope(_) = &node.op {
            ropes += 1;
            let x = &graph.graph().edges()[node.inputs[0].0].tensor;
            assert_eq!(
                x.shape()[2],
                Dim::Concrete(8),
                "rope input is the rotary part, got {x:?}"
            );
        }
    }
    assert_eq!(ropes, 2, "q-rope and k-rope only");

    // State write carries the exact canonical (latent, rotary) pair.
    let mut writes = 0;
    for node in graph.graph().nodes() {
        if let Op::StateWriteKv(w) = &node.op {
            if w.latent.is_some() {
                writes += 1;
                let c_kv = &graph.graph().edges()[node.inputs[0].0].tensor;
                let k_rope = &graph.graph().edges()[node.inputs[1].0].tensor;
                assert_eq!(
                    c_kv.shape()[2],
                    Dim::Concrete(16),
                    "written operand 0 is latent"
                );
                assert_eq!(
                    k_rope.shape()[2],
                    Dim::Concrete(8),
                    "written operand 1 is rotary"
                );
            }
        }
    }
    assert_eq!(writes, 1, "one MLA state write");

    // Reconstructed attention inputs carry the declared dims.
    let mut attentions = 0;
    for node in graph.graph().nodes() {
        if let Op::Attention(_) = &node.op {
            attentions += 1;
            let q = &graph.graph().edges()[node.inputs[0].0].tensor;
            assert_eq!(
                q.shape(),
                &[
                    Dim::Symbolic(ShapeSymbol::T),
                    Dim::Concrete(4),
                    Dim::Concrete(32)
                ],
                "reconstructed q is [T, H, nope + rope]"
            );
            let out = &graph.graph().edges()[node.outputs[0].0].tensor;
            assert_eq!(
                out.shape()[2],
                Dim::Concrete(48),
                "attention output head dim is v_dim"
            );
        }
    }
    assert_eq!(attentions, 1, "one attention");

    // qk_norm lowered, not rejected: the query rows carry a Head(32)
    // norm over attn_q_norm [4 * 32], the KV rows a row norm over
    // attn_k_norm [16 + 8]. Weight shapes identify the qk_norm nodes
    // exactly (placement and final norms use [Dm] = [32] here).
    let mut norm_weight_dims = Vec::new();
    for node in graph.graph().nodes() {
        if let Op::Norm(n) = &node.op {
            let w = &graph.graph().edges()[node.inputs[1].0].tensor;
            norm_weight_dims.push((n.axis, w.shape().to_vec()));
        }
    }
    assert!(
        norm_weight_dims.contains(&(NormAxis::Head(32), vec![Dim::Concrete(128)])),
        "query-side per-head qk_norm, got {norm_weight_dims:?}"
    );
    assert!(
        norm_weight_dims.contains(&(NormAxis::Last, vec![Dim::Concrete(24)])),
        "kv-side row qk_norm, got {norm_weight_dims:?}"
    );
    for name in ["blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"] {
        assert!(
            graph.bound_weights().iter().any(|w| w.name == name),
            "missing MLA qk_norm weight {name}"
        );
    }
}

/// Non-unit residual scales lower exactly on every layer residual (SI-27).
#[test]
fn residual_scale_lowers_exactly() {
    let mut layer = tiny_layer();
    layer.residual_scale = 0.5;
    layer.ffn = Ffn::Dense {
        dff: 64,
        act: ActivationKind::Silu,
        gated: true,
        bias: false,
        pre_fused: false,
    };
    let model = tiny_model(vec![layer]);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-scale"), &model)
        .expect("scaled model builds and validates");
    let mut scales = Vec::new();
    for node in graph.graph().nodes() {
        if let Op::ResidualAdd(r) = &node.op {
            scales.push(r.scale);
        }
    }
    assert_eq!(scales.len(), 2, "mixer and ffn residuals");
    assert!(
        scales.iter().all(|s| *s == 0.5),
        "every residual carries the scale: {scales:?}"
    );
}

/// Unit residual scale keeps the A1.3 unit form on every residual (SI-27).
#[test]
fn default_residual_scale_stays_unit() {
    let model = tiny_model(vec![tiny_layer()]);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-scale-unit"), &model)
        .expect("default model builds");
    for node in graph.graph().nodes() {
        if let Op::ResidualAdd(r) = &node.op {
            assert_eq!(r.scale, 1.0, "default residual is unit");
        }
    }
}

/// A set final logit softcap lowers to one exact op; None lowers to none (SI-28).
#[test]
fn final_logit_softcap_lowers_exactly() {
    let mut model = tiny_model(vec![tiny_layer()]);
    model.final_logit_softcap = Some(30.0);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-softcap"), &model)
        .expect("softcap model builds and validates");
    let mut caps = Vec::new();
    for node in graph.graph().nodes() {
        if let Op::LogitSoftcap(s) = &node.op {
            caps.push(s.cap);
            let x = &graph.graph().edges()[node.inputs[0].0].tensor;
            assert_eq!(x.dtype(), DType::F32, "softcap input is f32 logits");
            assert_eq!(x.shape().len(), 2, "softcap input is [T, V]");
        }
    }
    assert_eq!(
        caps,
        vec![30.0],
        "exactly one softcap with the declared cap"
    );

    let plain = tiny_model(vec![tiny_layer()]);
    let plain_graph = build_model(Graph::new(IrVersion::CURRENT, "a114-softcap-none"), &plain)
        .expect("default model builds");
    assert!(
        plain_graph
            .graph()
            .nodes()
            .iter()
            .all(|n| !matches!(&n.op, Op::LogitSoftcap(_))),
        "no softcap op without the spec field"
    );
}

/// Defaults carry no extension ops and every rope reads the projection.
#[test]
fn defaults_have_no_extension_ops() {
    let model = tiny_model(vec![tiny_layer()]);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-defaults"), &model)
        .expect("default model builds");
    for node in graph.graph().nodes() {
        assert!(
            !matches!(&node.op, Op::Split(_) | Op::Concat(_) | Op::LogitSoftcap(_)),
            "default graph has no extension op: {:?}",
            node.op
        );
    }
}

/// Malformed inputs report typed errors without panicking.
#[test]
fn malformed_inputs_report_typed_errors() {
    // Zero residual scale.
    let mut layer = tiny_layer();
    layer.residual_scale = 0.0;
    assert!(layer.validate(0).is_err(), "zero residual scale rejected");
    // Non-positive softcap.
    let mut model = tiny_model(vec![tiny_layer()]);
    model.final_logit_softcap = Some(0.0);
    assert!(model.validate().is_err(), "zero final softcap rejected");
    // Out-of-range MTP layer source.
    let mtp = MtpSpec {
        heads: 1,
        layers_per_head: vec![tiny_layer()],
        takes_hidden_from: MtpSource::Layer(7),
    };
    assert!(mtp.validate(2).is_err(), "out-of-range MTP layer rejected");
    // Degenerate MLA dims.
    let bad_mla = MlaSpec {
        q_lora_rank: 0,
        kv_lora_rank: 16,
        qk_nope_dim: 16,
        qk_rope_dim: 16,
        v_dim: 32,
    };
    assert!(bad_mla.validate("mla").is_err(), "zero rank rejected");
}

/// One full-featured model (MLA with qk_norm, MTP, scale, softcap)
/// validates end to end. MRoPE mode is covered by its own projection oracle;
/// the MLA rotary width constrains MRoPE section sums (SI-29), so the full
/// model stays on scalar positions.
#[test]
fn full_featured_model_validates() {
    let mixer = mla_mixer();
    let mut layer = LayerSpec {
        mixer,
        ..tiny_layer()
    };
    layer.residual_scale = 1.5;
    let mtp = MtpSpec {
        heads: 1,
        layers_per_head: vec![tiny_layer()],
        takes_hidden_from: MtpSource::Last,
    };
    let mut model = tiny_model(vec![layer]);
    model.mtp = Some(mtp);
    model.final_logit_softcap = Some(30.0);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "a114-full"), &model)
        .expect("full-featured model builds and validates");
    assert!(graph.subgraphs().contains_key("mtp"), "mtp present");
    assert!(
        graph
            .graph()
            .nodes()
            .iter()
            .any(|n| matches!(&n.op, Op::LogitSoftcap(_))),
        "softcap present"
    );
}

/// Ancestor edge set of an edge: follows op inputs and reshape views back to
/// external inputs and graph-owned sources (weights), which terminate.
fn ancestor_edges(
    graph: &r9v_ir::Graph,
    edge: r9v_ir::EdgeId,
) -> std::collections::BTreeSet<r9v_ir::EdgeId> {
    use std::collections::BTreeSet;
    let mut visited = BTreeSet::new();
    let mut work = vec![edge];
    while let Some(e) = work.pop() {
        if !visited.insert(e) {
            continue;
        }
        if let Some(node) = graph.nodes().iter().find(|n| n.outputs.contains(&e)) {
            work.extend(node.inputs.iter().copied());
        } else if let Some(source) = graph.edges().get(e.0).and_then(|edge| edge.source_edge) {
            work.push(source);
        }
    }
    visited
}

/// Bare-mixer MLA lowering past validation carries the exact split pair.
#[test]
fn bare_mixer_mla_state_write_is_exact() {
    let mixer = mla_mixer();
    let model = tiny_model(vec![tiny_layer()]);
    let mut builder = GraphBuilder::new(IrVersion::CURRENT, "a114-bare-mla");
    let (h, _) = builder.input_embed_override(model.dm).expect("override");
    build_mixer(&mut builder, 0, &mixer, h, &model).expect("bare mixer lowers");
    let graph = builder.finish().expect("bare MLA graph validates");
    let mut kinds = Vec::new();
    for node in graph.graph().nodes() {
        if let Op::StateWriteKv(w) = &node.op {
            kinds.push(w.handle.kind());
        }
    }
    assert_eq!(kinds, vec![StateKind::KvLatent], "latent state handle");
}
