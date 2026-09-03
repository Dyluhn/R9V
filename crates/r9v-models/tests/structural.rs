// SPDX-License-Identifier: Apache-2.0
//! Exhaustive structural tests for all norm × mixer × ffn combinations (Spec 8 §3, §3.1; card A1.3).
//!
//! Verifies that the generic layer builder emits the exact Op IR structures required
//! and that the resulting graph validates under the Op IR specification.

use r9v_ir::op::{
    ActivationKind, Epilogue, HashId, LinearAttnKind, MoeScoring, NgramCombine, Op, RopeScaling,
    RopeStyle,
};
use r9v_ir::tensor::{Dim, ShapeSymbol};
use r9v_ir::version::IrVersion;
use r9v_ir::DType;
use r9v_models::{
    build_layer, build_model, CacheDtype, Ffn, Graph, GraphBuilder, LayerSpec, Mixer, MlaSpec,
    ModelSpec, MoeGroupSpec, MoeSharedSpec, MtpSource, MtpSpec, NgramSpec, NormPlacement, NormSpec,
    PositionEncoding, RopeSpec,
};

fn base_rope() -> RopeSpec {
    RopeSpec {
        theta: 10000.0,
        rot_dim: 64,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
    }
}

fn dummy_model_spec(layers: Vec<LayerSpec>) -> ModelSpec {
    ModelSpec {
        dm: 512,
        layers,
        vocab: 1000,
        embed_scale: 1.0,
        tied_embeddings: false,
        final_norm: NormSpec::rms(1e-5),
        final_logit_softcap: None,
        positions: PositionEncoding::Scalar,
        ngram: None,
        mtp: None,
        export_hidden: false,
        eos_ids: vec![2],
        bos_id: Some(1),
    }
}

/// Hand-derived expected tallies for one `build_layer` graph (Spec 8 §3.1).
///
/// Restated from the spec pattern, not copied from the builder, so a dropped
/// or duplicated op, weight, or edge fails the matrix. Reshapes are edges,
/// not nodes. Bias flags change weights and epilogues, never op counts.
#[derive(Debug, Default, PartialEq, Eq)]
struct LayerTallies {
    matmul: usize,
    norm: usize,
    rope: usize,
    state_write_kv: usize,
    attention: usize,
    residual_add: usize,
    act_mul: usize,
    activation: usize,
    moe_route: usize,
    moe_ffn: usize,
    causal_conv1d: usize,
    linear_attn_scan: usize,
    split: usize,
    concat: usize,
    /// Bound weight edges by dtype (matmul weights F16, vectors/biases F32).
    w_f16: usize,
    w_f32: usize,
    /// Op-output and reshape-view edges by dtype (externals added separately).
    o_f16: usize,
    o_f32: usize,
    o_u32: usize,
    /// Bound `BatchMeta.positions` projection edges (one per rope model).
    positions_u32: usize,
}

impl LayerTallies {
    fn total_nodes(&self) -> usize {
        self.matmul
            + self.norm
            + self.rope
            + self.state_write_kv
            + self.attention
            + self.residual_add
            + self.act_mul
            + self.activation
            + self.moe_route
            + self.moe_ffn
            + self.causal_conv1d
            + self.linear_attn_scan
            + self.split
            + self.concat
    }
}

/// Expected mixer contribution: op nodes, weight edges, and output edges.
fn mixer_tallies(m: &Mixer) -> LayerTallies {
    let mut t = LayerTallies::default();
    match m {
        Mixer::Attention {
            qk_norm,
            output_gate,
            qkv_bias,
            o_bias,
            mla,
            ..
        } => {
            let gate = usize::from(*output_gate);
            let qkv = usize::from(*qkv_bias);
            let ob = usize::from(*o_bias);
            // Q/K/V/output projections, Q/K rotary pair, one cache write, one
            // attention read, optional output gate. Reshapes (q, k, v, a) are
            // edges; MLA reshapes (q, latent, a) likewise.
            t.matmul = 4 + gate;
            t.rope = 2;
            t.state_write_kv = 1;
            t.attention = 1;
            t.act_mul = gate;
            t.w_f16 = 4 + gate;
            // Every attention mixer binds one BatchMeta.positions projection
            // edge feeding both rope nodes (Spec 1 §2.5; card A1.14).
            t.positions_u32 = 1;
            if mla.is_some() {
                // MLA splits q into (nope, rope) and the KV rows into
                // (latent, rope), rotates the rotary parts only, and
                // reconstructs the query by concatenation (card A1.14); its
                // biases cover the down/up projections (q_a, q_b, kv_a).
                t.split = 2;
                t.concat = 1;
                t.w_f32 = 3 * qkv + ob;
                t.o_f16 = 15 + 2 * gate;
            } else {
                let qk = usize::from(qk_norm.is_some());
                t.norm = 2 * qk;
                t.w_f32 = 2 * qk + 3 * qkv + ob;
                t.o_f16 = 11 + 2 * qk + 2 * gate;
            }
        }
        Mixer::LinearAttention {
            conv,
            output_norm,
            output_gate,
            ..
        } => {
            let conv = usize::from(conv.is_some());
            let on = usize::from(output_norm.is_some());
            let gate = usize::from(*output_gate);
            // Q/K/V/alpha/beta/output projections, one scan, optional conv,
            // output norm, and gate. Alpha/beta matmuls emit F32.
            t.matmul = 6 + gate;
            t.causal_conv1d = conv;
            t.linear_attn_scan = 1;
            t.norm = on;
            t.act_mul = gate;
            t.w_f16 = 6 + gate;
            t.w_f32 = conv + on;
            t.o_f16 = 9 + conv + on + 2 * gate;
            t.o_f32 = 2;
        }
        Mixer::None => {}
    }
    t
}

/// Expected FFN contribution: op nodes, weight edges, and output edges.
fn ffn_tallies(f: &Ffn) -> LayerTallies {
    let mut t = LayerTallies::default();
    match f {
        Ffn::Dense { gated, bias, .. } => {
            if *gated {
                t.matmul = 3;
                t.act_mul = 1;
                t.w_f16 = 3;
                // Gated bias lowers as bias epilogues on gate and up.
                t.w_f32 = 2 * usize::from(*bias);
                t.o_f16 = 4;
            } else {
                t.matmul = 2;
                t.activation = 1;
                t.w_f16 = 2;
                t.w_f32 = usize::from(*bias);
                t.o_f16 = 3;
            }
        }
        Ffn::Moe {
            route_bias,
            shared,
            shared_gate,
            ..
        } => {
            // Router matmul (F32 logits), route, expert execution, plus an
            // optional shared-expert path combined by residual_add.
            let shared_gate = usize::from(*shared_gate);
            t.matmul = 1;
            t.moe_route = 1;
            t.moe_ffn = 1;
            t.w_f16 = 3;
            t.w_f32 = usize::from(*route_bias);
            t.o_f16 = 1;
            t.o_f32 = 2;
            t.o_u32 = 1;
            if shared.is_some() {
                t.matmul += 3 + shared_gate;
                t.act_mul += 1 + shared_gate;
                t.residual_add += 1;
                t.w_f16 += 3 + shared_gate;
                t.o_f16 += 5 + 2 * shared_gate;
            }
        }
        Ffn::None => {}
    }
    t
}

/// Full expected tallies for one layer graph, including placement norms and
/// residuals plus the three external inputs (embed F16, mask Bool, tokens U32).
fn expected_layer(norm: NormPlacement, mixer: &Mixer, ffn: &Ffn) -> LayerTallies {
    let m = mixer_tallies(mixer);
    let f = ffn_tallies(ffn);
    let has_mixer = *mixer != Mixer::None;
    let has_ffn = *ffn != Ffn::None;
    let (n_norms, n_res) = match norm {
        NormPlacement::Pre => (
            usize::from(has_mixer) + usize::from(has_ffn),
            usize::from(has_mixer) + usize::from(has_ffn),
        ),
        NormPlacement::Sandwich => (
            2 * usize::from(has_mixer) + 2 * usize::from(has_ffn),
            usize::from(has_mixer) + usize::from(has_ffn),
        ),
        NormPlacement::Parallel => (1, usize::from(has_mixer) + usize::from(has_ffn)),
    };
    LayerTallies {
        matmul: m.matmul + f.matmul,
        norm: m.norm + f.norm + n_norms,
        rope: m.rope,
        state_write_kv: m.state_write_kv,
        attention: m.attention,
        residual_add: m.residual_add + f.residual_add + n_res,
        act_mul: m.act_mul + f.act_mul,
        activation: f.activation,
        moe_route: f.moe_route,
        moe_ffn: f.moe_ffn,
        causal_conv1d: m.causal_conv1d,
        linear_attn_scan: m.linear_attn_scan,
        split: m.split,
        concat: m.concat,
        w_f16: m.w_f16 + f.w_f16,
        w_f32: m.w_f32 + f.w_f32 + n_norms,
        o_f16: m.o_f16 + f.o_f16 + n_norms + n_res,
        o_f32: m.o_f32 + f.o_f32,
        o_u32: m.o_u32 + f.o_u32,
        positions_u32: m.positions_u32,
    }
}

/// Actual op-node tallies folded from a built graph; any unexpected op kind
/// fails the caller via the returned `other` count.
fn actual_nodes(graph: &r9v_models::ModelGraph) -> (LayerTallies, usize) {
    let mut t = LayerTallies::default();
    let mut other = 0usize;
    for node in graph.graph().nodes() {
        match &node.op {
            Op::Matmul(_) => t.matmul += 1,
            Op::Norm(_) => t.norm += 1,
            Op::Rope(_) => t.rope += 1,
            Op::StateWriteKv(_) => t.state_write_kv += 1,
            Op::Attention(_) => t.attention += 1,
            Op::ResidualAdd(_) => t.residual_add += 1,
            Op::ActMul(_) => t.act_mul += 1,
            Op::Activation(_) => t.activation += 1,
            Op::MoeRoute(_) => t.moe_route += 1,
            Op::MoeFfn(_) => t.moe_ffn += 1,
            Op::CausalConv1d(_) => t.causal_conv1d += 1,
            Op::LinearAttnScan(_) => t.linear_attn_scan += 1,
            Op::Split(_) => t.split += 1,
            Op::Concat(_) => t.concat += 1,
            _ => other += 1,
        }
    }
    (t, other)
}

/// Actual edge-dtype histogram folded from a built graph.
fn actual_dtypes(graph: &r9v_models::ModelGraph) -> (usize, usize, usize, usize, usize) {
    let (mut f16, mut f32, mut u32, mut bool, mut other) = (0, 0, 0, 0, 0);
    for edge in graph.graph().edges() {
        match edge.tensor.dtype() {
            DType::F16 => f16 += 1,
            DType::F32 => f32 += 1,
            DType::U32 => u32 += 1,
            DType::Bool => bool += 1,
            _ => other += 1,
        }
    }
    (f16, f32, u32, bool, other)
}

/// Asserts exact per-op counts for one matrix cell.
fn assert_op_counts(actual: &LayerTallies, expected: &LayerTallies, tag: &str) {
    assert_eq!(actual.matmul, expected.matmul, "matmul nodes for {tag}");
    assert_eq!(actual.norm, expected.norm, "norm nodes for {tag}");
    assert_eq!(actual.rope, expected.rope, "rope nodes for {tag}");
    assert_eq!(
        actual.state_write_kv, expected.state_write_kv,
        "state_write_kv nodes for {tag}"
    );
    assert_eq!(
        actual.attention, expected.attention,
        "attention nodes for {tag}"
    );
    assert_eq!(
        actual.residual_add, expected.residual_add,
        "residual_add nodes for {tag}"
    );
    assert_eq!(actual.act_mul, expected.act_mul, "act_mul nodes for {tag}");
    assert_eq!(
        actual.activation, expected.activation,
        "activation nodes for {tag}"
    );
    assert_eq!(
        actual.moe_route, expected.moe_route,
        "moe_route nodes for {tag}"
    );
    assert_eq!(actual.moe_ffn, expected.moe_ffn, "moe_ffn nodes for {tag}");
    assert_eq!(
        actual.causal_conv1d, expected.causal_conv1d,
        "causal_conv1d nodes for {tag}"
    );
    assert_eq!(
        actual.linear_attn_scan, expected.linear_attn_scan,
        "linear_attn_scan nodes for {tag}"
    );
    assert_eq!(actual.split, expected.split, "split nodes for {tag}");
    assert_eq!(actual.concat, expected.concat, "concat nodes for {tag}");
    assert_eq!(
        actual.total_nodes(),
        expected.total_nodes(),
        "total nodes for {tag}"
    );
}

#[test]
fn test_exhaustive_layer_combinations_matrix() {
    let norm_placements = [
        NormPlacement::Pre,
        NormPlacement::Sandwich,
        NormPlacement::Parallel,
    ];

    let mixers = [
        // 1. Standard Attention
        Mixer::Attention {
            h: 8,
            hkv: 4,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: base_rope(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4m3,
        },
        // 2. Attention with MLA
        Mixer::Attention {
            h: 8,
            hkv: 1,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: base_rope(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: Some(MlaSpec {
                q_lora_rank: 128,
                kv_lora_rank: 64,
                qk_nope_dim: 32,
                qk_rope_dim: 32,
                v_dim: 64,
            }),
            cache: CacheDtype::E4m3,
        },
        // 3. Attention with output_gate
        Mixer::Attention {
            h: 8,
            hkv: 4,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: base_rope(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: true,
            mla: None,
            cache: CacheDtype::E4m3,
        },
        // 4. Attention with qk_norm
        Mixer::Attention {
            h: 8,
            hkv: 4,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: Some(NormSpec::rms(1e-5)),
            rope: base_rope(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4m3,
        },
        // 5. Attention with window + sinks + logit_softcap
        Mixer::Attention {
            h: 8,
            hkv: 4,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: base_rope(),
            window: Some(512),
            sinks: 4,
            logit_softcap: Some(50.0),
            output_gate: false,
            mla: None,
            cache: CacheDtype::F16,
        },
        // 6. LinearAttention with Conv + output_norm + output_gate
        Mixer::LinearAttention {
            kind: LinearAttnKind::GatedDeltaNet,
            h: 4,
            d: 32,
            dv: 32,
            conv: Some(4),
            gate_act: ActivationKind::Silu,
            output_norm: Some(NormSpec::rms(1e-5)),
            output_gate: true,
        },
        // 7. LinearAttention without conv
        Mixer::LinearAttention {
            kind: LinearAttnKind::Mamba2,
            h: 4,
            d: 32,
            dv: 32,
            conv: None,
            gate_act: ActivationKind::Silu,
            output_norm: None,
            output_gate: false,
        },
        // 8. Mixer::None
        Mixer::None,
    ];

    let ffns = [
        // 1. Dense gated
        Ffn::Dense {
            dff: 1024,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
        },
        // 2. Dense ungated with bias
        Ffn::Dense {
            dff: 1024,
            act: ActivationKind::Gelu,
            gated: false,
            bias: true,
        },
        // 3. Moe with shared experts + shared gate + group
        Ffn::Moe {
            e: 8,
            k: 2,
            dff_e: 512,
            act: ActivationKind::Silu,
            scoring: MoeScoring::Softmax,
            renormalize: true,
            group: Some(MoeGroupSpec {
                n_group: 2,
                topk_group: 1,
            }),
            route_bias: true,
            route_scale: 1.0,
            shared: Some(MoeSharedSpec { n: 1, dff: 512 }),
            shared_gate: true,
        },
        // 4. Moe standard without shared
        Ffn::Moe {
            e: 4,
            k: 1,
            dff_e: 256,
            act: ActivationKind::Silu,
            scoring: MoeScoring::Sigmoid,
            renormalize: false,
            group: None,
            route_bias: false,
            route_scale: 2.0,
            shared: None,
            shared_gate: false,
        },
        // 5. Ffn::None
        Ffn::None,
    ];

    let mut tested_combinations = 0;

    for norm in norm_placements {
        for (m_idx, mixer) in mixers.iter().enumerate() {
            for (f_idx, ffn) in ffns.iter().enumerate() {
                // Skip combinations where both mixer and ffn are None
                if *mixer == Mixer::None && *ffn == Ffn::None {
                    continue;
                }

                let expected = expected_layer(norm, mixer, ffn);
                let tag = format!("norm={norm:?}, mixer={m_idx}, ffn={f_idx}");

                let layer_spec = LayerSpec {
                    norm,
                    norm_kind: NormSpec::rms(1e-5),
                    mixer: mixer.clone(),
                    ffn: ffn.clone(),
                    residual_scale: 1.0,
                };

                let model = dummy_model_spec(vec![layer_spec.clone()]);
                let mut builder = GraphBuilder::new(
                    IrVersion::CURRENT,
                    format!("test-norm-{norm:?}-m{m_idx}-f{f_idx}"),
                );

                let (x, _) = builder.input_embed_override(model.dm).unwrap();
                let _ = builder.input_tokens().unwrap();

                let out = build_layer(&mut builder, 0, &layer_spec, x, &model);
                assert!(out.is_ok(), "build_layer failed for {tag}: {:?}", out.err());

                let model_graph = builder
                    .finish()
                    .unwrap_or_else(|e| panic!("builder.finish() failed for {tag}: {e:?}"));

                // Exact op counts per §3.1: no missing, duplicated, or alien ops.
                let (actual, other) = actual_nodes(&model_graph);
                assert_eq!(other, 0, "unexpected op kind for {tag}");
                assert_op_counts(&actual, &expected, &tag);

                // Exact edge-dtype table: weights + op outputs + reshape views
                // + the three external inputs (embed F16, mask Bool, tokens U32)
                // + the bound positions projection on rope models (U32).
                let (f16, f32, u32, boolean, other_dt) = actual_dtypes(&model_graph);
                assert_eq!(other_dt, 0, "unexpected edge dtype for {tag}");
                assert_eq!(
                    f16,
                    expected.o_f16 + expected.w_f16 + 1,
                    "F16 edge count for {tag}"
                );
                assert_eq!(
                    f32,
                    expected.o_f32 + expected.w_f32,
                    "F32 edge count for {tag}"
                );
                assert_eq!(
                    u32,
                    expected.o_u32 + 1 + expected.positions_u32,
                    "U32 edge count for {tag}"
                );
                assert_eq!(boolean, 1, "Bool edge count for {tag}");

                tested_combinations += 1;
            }
        }
    }

    // 3 norm placements * (8 mixers * 5 ffns - 1 both none) = 3 * 39 = 117 combinations
    assert_eq!(tested_combinations, 117);
}

#[test]
fn test_ngram_speculative_injection() {
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 4,
            hkv: 2,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: base_rope(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4m3,
        },
        ffn: Ffn::Dense {
            dff: 512,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
        },
        residual_scale: 1.0,
    };

    let mut model = dummy_model_spec(vec![layer.clone(), layer.clone()]);
    model.ngram = Some(NgramSpec {
        orders: vec![2, 3],
        heads: 2,
        dim: 32,
        table_sizes: vec![1024, 2048],
        hash: HashId::new(1),
        combine: NgramCombine::Sum,
        inject_at: 1,
    });

    let builder = Graph::new(IrVersion::CURRENT, "test-ngram");
    let model_graph = build_model(builder, &model).expect("ngram model must build successfully");

    // Verify NgramGatherOp is present in the graph
    let has_ngram_gather = model_graph
        .graph()
        .nodes()
        .iter()
        .any(|node| matches!(node.op, r9v_ir::op::Op::NgramGather(_)));
    assert!(has_ngram_gather, "graph must contain NgramGatherOp");

    // Verify ngram table weight was bound
    let has_ngram_table = model_graph
        .bound_weights()
        .iter()
        .any(|w| w.role == r9v_models::WeightRole::NgramTable);
    assert!(has_ngram_table, "must contain bound NgramTable weight");
}

#[test]
fn test_multi_token_prediction_subgraph() {
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 4,
            hkv: 2,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: base_rope(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4m3,
        },
        ffn: Ffn::Dense {
            dff: 512,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
        },
        residual_scale: 1.0,
    };

    let mut model = dummy_model_spec(vec![layer.clone()]);
    model.mtp = Some(MtpSpec {
        heads: 2,
        layers_per_head: vec![layer.clone()],
        takes_hidden_from: MtpSource::Last,
    });

    let builder = Graph::new(IrVersion::CURRENT, "test-mtp");
    let model_graph = build_model(builder, &model).expect("mtp model must build successfully");

    assert!(
        model_graph.subgraphs().contains_key("mtp"),
        "model graph must register mtp subgraph"
    );
    let mtp_subgraph = &model_graph.subgraphs()["mtp"];
    assert!(
        mtp_subgraph
            .exports()
            .iter()
            .any(|(name, _)| name == "mtp_logits_1"),
        "mtp subgraph must export mtp_logits_1"
    );
    assert!(
        mtp_subgraph
            .exports()
            .iter()
            .any(|(name, _)| name == "mtp_logits_2"),
        "mtp subgraph must export mtp_logits_2"
    );

    // Every subgraph weight lives in the `blk.N.mtp.*` namespace (Spec 8 §5).
    let sub_names: Vec<&str> = mtp_subgraph
        .bound_weights()
        .iter()
        .map(|w| w.name.as_str())
        .collect();
    assert!(
        sub_names.iter().all(|n| n.contains(".mtp.")),
        "all MTP weights must use the mtp namespace: {sub_names:?}"
    );
    let mut unique = sub_names.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        sub_names.len(),
        "MTP head weights must be independent, not shared: {sub_names:?}"
    );
    assert!(
        sub_names.contains(&"blk.0.mtp.output.weight"),
        "head 1 output weight: {sub_names:?}"
    );
    assert!(
        sub_names.contains(&"blk.1.mtp.output.weight"),
        "head 2 output weight: {sub_names:?}"
    );
    // No phantom layers: two ordinals built, two layers summarized.
    assert_eq!(mtp_subgraph.summary().expect("mtp summary").layers.len(), 2);
    assert_eq!(
        model_graph.summary().expect("model summary").layers.len(),
        1
    );
}

#[test]
fn test_tied_embeddings_and_fusions() {
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 4,
            hkv: 2,
            d: 64,
            dv: 64,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: base_rope(),
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4m3,
        },
        ffn: Ffn::Dense {
            dff: 512,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
        },
        residual_scale: 1.0,
    };

    let mut model = dummy_model_spec(vec![layer]);
    model.tied_embeddings = true;
    model.export_hidden = true;

    let builder = Graph::new(IrVersion::CURRENT, "test-tied");
    let model_graph = build_model(builder, &model).expect("model must build successfully");

    // Verify tied embeddings declaration
    assert_eq!(model_graph.tied_decls().len(), 1);
    assert_eq!(model_graph.tied_decls()[0].embed_name, "token_embd.weight");
    assert_eq!(model_graph.tied_decls()[0].head_name, "output.weight");

    // Verify fusion declarations: Qkv and GateUp
    assert!(model_graph
        .fusion_decls()
        .iter()
        .any(|f| matches!(f, r9v_models::FusionDecl::Qkv { .. })));
    assert!(model_graph
        .fusion_decls()
        .iter()
        .any(|f| matches!(f, r9v_models::FusionDecl::GateUp { .. })));

    // Verify exports
    assert!(model_graph
        .exports()
        .iter()
        .any(|(name, _)| name == "hidden"));
    assert!(model_graph
        .exports()
        .iter()
        .any(|(name, _)| name == "logits"));
}

/// Attention bias flags bind canonical GGUF-convention bias weights and lower
/// through the matmul bias epilogue without changing op counts (Spec 8 §3).
#[test]
fn test_attention_bias_weights_and_epilogues() {
    let bias_rope = RopeSpec {
        theta: 10000.0,
        rot_dim: 32,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
    };
    let biased_mixer = || Mixer::Attention {
        h: 4,
        hkv: 2,
        d: 32,
        dv: 32,
        qkv_bias: true,
        o_bias: true,
        qk_norm: None,
        rope: bias_rope.clone(),
        window: None,
        sinks: 0,
        logit_softcap: None,
        output_gate: false,
        mla: None,
        cache: CacheDtype::E4m3,
    };
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: biased_mixer(),
        ffn: Ffn::Dense {
            dff: 128,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
        },
        residual_scale: 1.0,
    };
    let model = dummy_model_spec(vec![layer]);
    let graph = build_model(Graph::new(IrVersion::CURRENT, "test-bias"), &model)
        .expect("biased model must build");

    // Exact op counts match the unbiased pattern; biases ride the epilogue.
    // A full model adds the embedding lookup, the final norm, and the LM head
    // around the layer pattern.
    let mut expected = expected_layer(
        NormPlacement::Pre,
        &biased_mixer(),
        &Ffn::Dense {
            dff: 128,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
        },
    );
    expected.matmul += 1;
    expected.norm += 1;
    let (actual, other) = actual_nodes(&graph);
    assert_eq!(other, 1, "only the embedding lookup is outside the tallies");
    assert_eq!(
        graph
            .graph()
            .nodes()
            .iter()
            .filter(|n| matches!(&n.op, Op::EmbedGather(_)))
            .count(),
        1,
        "the uncounted node is the embedding lookup"
    );
    assert_op_counts(&actual, &expected, "biased attention");

    // Canonical bias weights with exact shapes and F32 vector dtype.
    for (name, dim) in [
        ("blk.0.attn_q.bias", 128u32),
        ("blk.0.attn_k.bias", 64u32),
        ("blk.0.attn_v.bias", 64u32),
        ("blk.0.attn_output.bias", 512u32),
    ] {
        let w = graph
            .bound_weights()
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("missing bias weight {name}"));
        assert_eq!(w.shape, vec![Dim::Concrete(dim)], "shape of {name}");
        assert_eq!(w.tensor.dtype(), DType::F32, "dtype of {name}");
    }

    // Exactly the Q/K/V/output projections carry the bias epilogue.
    let mut bias_epilogues = 0;
    let mut plain = 0;
    for node in graph.graph().nodes() {
        if let Op::Matmul(m) = &node.op {
            match m.epilogue {
                Epilogue::Bias => bias_epilogues += 1,
                Epilogue::None => plain += 1,
                _ => panic!("unexpected matmul epilogue {:?}", m.epilogue),
            }
        }
    }
    assert_eq!(bias_epilogues, 4, "q, k, v, and output projections");
    assert!(plain > 0, "ffn projections stay epilogue-free");

    // Negative case: flags off binds no bias weights and no Bias epilogue.
    let plain_mixer = Mixer::Attention {
        h: 4,
        hkv: 2,
        d: 32,
        dv: 32,
        qkv_bias: false,
        o_bias: false,
        qk_norm: None,
        rope: bias_rope,
        window: None,
        sinks: 0,
        logit_softcap: None,
        output_gate: false,
        mla: None,
        cache: CacheDtype::E4m3,
    };
    let plain_layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: plain_mixer,
        ffn: Ffn::Dense {
            dff: 128,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
        },
        residual_scale: 1.0,
    };
    let plain_model = dummy_model_spec(vec![plain_layer]);
    let plain_graph = build_model(Graph::new(IrVersion::CURRENT, "test-nobias"), &plain_model)
        .expect("unbiased model must build");
    assert!(
        plain_graph
            .bound_weights()
            .iter()
            .all(|w| !w.name.ends_with(".bias")),
        "no bias weights when flags are off"
    );
    assert!(
        plain_graph.graph().nodes().iter().all(|n| !matches!(
            &n.op,
            Op::Matmul(m) if m.epilogue == Epilogue::Bias
        )),
        "no Bias epilogue when flags are off"
    );
}

/// MLA with query/value dimensions that differ must build, validate, and emit
/// an attention output of head dim `v_dim` (Spec 1 §4.D; Spec 8 §3).
#[test]
fn test_mla_unequal_dims_and_bias() {
    let mla_rope = RopeSpec {
        theta: 10000.0,
        rot_dim: 32,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
    };
    let mixer = Mixer::Attention {
        h: 4,
        hkv: 1,
        d: 32,
        dv: 32,
        qkv_bias: true,
        o_bias: true,
        qk_norm: None,
        rope: mla_rope,
        window: None,
        sinks: 0,
        logit_softcap: None,
        output_gate: false,
        mla: Some(MlaSpec {
            q_lora_rank: 32,
            kv_lora_rank: 16,
            qk_nope_dim: 16,
            qk_rope_dim: 16,
            v_dim: 48,
        }),
        cache: CacheDtype::E4m3,
    };
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: mixer.clone(),
        ffn: Ffn::None,
        residual_scale: 1.0,
    };
    let model = dummy_model_spec(vec![layer]);
    // Unequal qk (16 + 16) and v (48) dims previously failed IR validation
    // (`o head dim 32 != v_dim 48`); the builder now emits `[T, H, v_dim]`.
    let graph = build_model(Graph::new(IrVersion::CURRENT, "test-mla-unequal"), &model)
        .expect("unequal-dim MLA model must build and validate");

    let mut expected = expected_layer(NormPlacement::Pre, &mixer, &Ffn::None);
    expected.matmul += 1;
    expected.norm += 1;
    let (actual, other) = actual_nodes(&graph);
    assert_eq!(other, 1, "only the embedding lookup is outside the tallies");
    assert_op_counts(&actual, &expected, "unequal-dim mla");

    // The single attention op carries v_dim and emits a v_dim head.
    let mut attentions = 0;
    for node in graph.graph().nodes() {
        if let Op::Attention(a) = &node.op {
            attentions += 1;
            let mla = a.mla.as_ref().expect("mla config on attention op");
            assert_eq!(mla.v_dim, 48);
            let out_id = node.outputs.first().expect("attention output edge");
            let out_shape = graph.graph().edges()[out_id.0].tensor.shape().to_vec();
            assert_eq!(
                out_shape,
                vec![
                    Dim::Symbolic(ShapeSymbol::T),
                    Dim::Concrete(4),
                    Dim::Concrete(48),
                ],
                "attention output is [T, H, v_dim]"
            );
        }
    }
    assert_eq!(attentions, 1);

    // Up/down projections reflect the split dimensions.
    for (name, rows, cols) in [
        ("blk.0.attn_q_b.weight", 128u32, 32u32),
        ("blk.0.attn_output.weight", 512u32, 192u32),
    ] {
        let w = graph
            .bound_weights()
            .iter()
            .find(|w| w.name == name)
            .unwrap_or_else(|| panic!("missing weight {name}"));
        assert_eq!(
            w.shape,
            vec![Dim::Concrete(rows), Dim::Concrete(cols)],
            "shape of {name}"
        );
    }
    for name in [
        "blk.0.attn_q_a.bias",
        "blk.0.attn_q_b.bias",
        "blk.0.attn_kv_a.bias",
        "blk.0.attn_output.bias",
    ] {
        assert!(
            graph.bound_weights().iter().any(|w| w.name == name),
            "missing MLA bias weight {name}"
        );
    }
    // Latent cache reports a single stream instead of hkv 0.
    assert_eq!(graph.summary().expect("mla summary").hkv, 1);
}

/// MTP heads bind disjoint, complete per-head weight sets in their own
/// ordinal namespace instead of sharing or chaining weights (Spec 8 §2, §5).
///
/// Edge-level capture identity (which physical edge each head reads) is
/// A1.14's scope: `GraphBuilder` resolves tensors by descriptor equality, so
/// same-shaped activations alias until SSA identity lands there.
#[test]
fn test_mtp_heads_bind_disjoint_weights() {
    let mtp_rope = RopeSpec {
        theta: 10000.0,
        rot_dim: 32,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
    };
    let layer = LayerSpec {
        norm: NormPlacement::Pre,
        norm_kind: NormSpec::rms(1e-5),
        mixer: Mixer::Attention {
            h: 4,
            hkv: 2,
            d: 32,
            dv: 32,
            qkv_bias: false,
            o_bias: false,
            qk_norm: None,
            rope: mtp_rope,
            window: None,
            sinks: 0,
            logit_softcap: None,
            output_gate: false,
            mla: None,
            cache: CacheDtype::E4m3,
        },
        ffn: Ffn::Dense {
            dff: 64,
            act: ActivationKind::Silu,
            gated: true,
            bias: false,
        },
        residual_scale: 1.0,
    };
    let mut model = dummy_model_spec(vec![layer.clone()]);
    model.dm = 64;
    model.mtp = Some(MtpSpec {
        heads: 2,
        layers_per_head: vec![layer],
        takes_hidden_from: MtpSource::Last,
    });
    let graph = build_model(Graph::new(IrVersion::CURRENT, "test-mtp-shared"), &model)
        .expect("mtp model must build");
    let sub = &graph.subgraphs()["mtp"];

    // Each head owns a complete layer's weights under its own ordinal; the
    // two sets are disjoint, so no head reuses the other's projections.
    let names: Vec<&str> = sub
        .bound_weights()
        .iter()
        .map(|w| w.name.as_str())
        .collect();
    for head in [0u32, 1u32] {
        for leaf in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "output.weight",
        ] {
            let name = format!("blk.{head}.mtp.{leaf}");
            assert!(
                names.contains(&name.as_str()),
                "head {head} owns {name}: {names:?}"
            );
        }
    }
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 20, "two disjoint complete sets: {names:?}");
    assert_eq!(names.len(), 20, "no extra weights beyond the two heads");
}
