// SPDX-License-Identifier: Apache-2.0
//! Executor dispatch gates over all 32 ops (Card A1.12; Spec 4 §2, §10).
//!
//! Each op runs as a one- or few-node step graph through [`CpuExecutor`]
//! and is checked with the shared A1.10 harness comparisons
//! ([`check_f32_against_f64`] at [`tolerance_for`], [`check_bits_equal`])
//! against its independent f64 oracle — never against a second executor
//! run, and with no bespoke comparison loops. Failure paths assert typed
//! [`ExecError`] variants.

use r9v_common::{SeqId, StepId};
use r9v_ir::{
    ActMulOp, ActivationKind, ActivationOp, AllReduceOp, BarrierOp, BatchMeta, CastOp, Class,
    ConcatOp, CopyKind, CopyOp, DType, Dim, EdgeId, EmbedGatherOp, Epilogue, ExternalInputKind,
    Graph, GroupId, LayoutId, LogitSoftcapOp, LogitsPostprocessOp, MatmulOp, NormAxis, NormKind,
    NormOp, Op, Placement, Positions, QuantScheme, ReduceOp, ResidualAddOp, RngAlgorithm, RopeOp,
    RopeScaling, RopeStyle, SampleOp, SamplingParams, ShapeSymbol, ShardLayout, SplitOp,
    StepGraphKey, VerifyMethod, VerifyOp,
};
use r9v_t0::buffer::TypedBuffer;
use r9v_t0::dtype::f32_to_f16;
use r9v_t0::exec::{CpuExecutor, ExecError, RunArgs};
use r9v_t0::harness::{
    check_bits_equal, check_f32_against_f64, rng_for, tolerance_for, uniform_f32, MASTER_SEED,
};
use r9v_t0::ngram_gather::NgramHash;
use r9v_t0::philox::RngState;
use r9v_t0::{
    act_mul_f64_reference, activation_f64_reference, attention_row_f64_reference,
    cast_f64_reference, causal_conv1d_f64_reference, concat_f64_reference, copy_f64_reference,
    embed_gather_f64_reference, gather_rows_f64_reference, linear_attn_scan_f64_reference,
    logit_softcap_f64_reference, logits_postprocess_f64_reference, matmul_f64_reference,
    moe_ffn_f64_reference, moe_route_f64_reference, ngram_gather_f64_reference_rows,
    ngram_gather_f64_reference_staged, norm_f64_reference, quant_act_f64_reference,
    residual_add_f64_reference, rope_f64_reference, scatter_add_rows_f64_reference,
    split_f64_reference,
};

// ---------------------------------------------------------------------------
// Shared scaffolding
// ---------------------------------------------------------------------------

fn key(s: u32, t: u32) -> StepGraphKey {
    StepGraphKey::from_unbucketed(r9v_ir::graph::PlanId::new(0xA112), 0, s, t, 0, 0).unwrap()
}

fn act_d(shape: Vec<Dim>, dtype: DType) -> r9v_ir::Tensor {
    r9v_ir::Tensor::new(
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

fn concat_dims(shape: &[usize]) -> Vec<Dim> {
    shape.iter().map(|&d| Dim::Concrete(d as u32)).collect()
}

fn weight_d(shape: &[usize], dtype: DType) -> r9v_ir::Tensor {
    r9v_ir::Tensor::new(
        concat_dims(shape),
        dtype,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )
    .unwrap()
}

fn act_q(shape: Vec<Dim>, dtype: DType, quant: QuantScheme) -> r9v_ir::Tensor {
    r9v_ir::Tensor::new(
        shape,
        dtype,
        quant,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )
    .unwrap()
}

fn param_d(shape: &[usize]) -> r9v_ir::Tensor {
    r9v_ir::Tensor::new(
        concat_dims(shape),
        DType::F32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Param,
    )
    .unwrap()
}

/// Seeded f32 values from the A1.12 stream (independent per `case`).
fn fvec(case: u64, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut rng = rng_for("a1.12-exec", case, MASTER_SEED);
    uniform_f32(&mut rng, len, lo, hi)
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

/// Adds an external input edge to the graph.
fn ext(graph: &mut Graph, kind: ExternalInputKind, desc: r9v_ir::Tensor) -> EdgeId {
    graph.add_external_input(kind, desc).unwrap()
}

/// Binds weights/inputs and runs state setup without executing.
fn fresh_exec(
    binds: Vec<(EdgeId, TypedBuffer)>,
    setup: impl FnOnce(&mut CpuExecutor),
) -> CpuExecutor {
    let mut exec = CpuExecutor::new();
    for (edge, buffer) in binds {
        exec.bind(edge, buffer);
    }
    setup(&mut exec);
    exec
}

/// Builds the S=1 step batch at context position `ctx`.
fn step_batch(t: u32, ctx: u32) -> BatchMeta {
    let positions: Vec<u32> = (ctx..ctx + t).collect();
    BatchMeta::builder(1, 1, t, 2)
        .seq_ids(vec![0])
        .query_len(vec![t])
        .ctx_len(vec![ctx])
        .positions(Positions::PerToken(positions.clone()))
        .slot_map(positions)
        .block_table(vec![0, 1])
        .window_start(vec![0])
        .tree(None)
        .build()
        .unwrap()
}

/// Binds weights/inputs, registers no state, and runs one S=1 step at ctx 0.
fn run_step(
    graph: &Graph,
    binds: Vec<(EdgeId, TypedBuffer)>,
    t: u32,
    setup: impl FnOnce(&mut CpuExecutor),
) -> CpuExecutor {
    let mut exec = fresh_exec(binds, setup);
    step_run(&mut exec, graph, t, 0, &[], &mut Vec::new(), None);
    exec
}

/// Runs one step on a live executor (multi-step state tests).
fn step_run(
    exec: &mut CpuExecutor,
    graph: &Graph,
    t: u32,
    ctx: u32,
    params: &[r9v_ir::SamplingParams],
    rng: &mut Vec<r9v_t0::philox::RngState>,
    ngram_hash: Option<&dyn r9v_t0::ngram_gather::NgramHash>,
) {
    let batch = step_batch(t, ctx);
    exec.run(
        graph,
        RunArgs {
            batch: &batch,
            params,
            rng,
            ngram_hash,
        },
    )
    .unwrap();
}

fn no_setup(_: &mut CpuExecutor) {}

fn add_node(
    graph: &mut Graph,
    op: Op,
    inputs: &[EdgeId],
    outputs: Vec<r9v_ir::Tensor>,
) -> Vec<EdgeId> {
    let id = graph.add_op(op, inputs, &outputs).unwrap();
    graph.nodes()[id.0].outputs.clone()
}

/// Registers and binds the shared `[T]` token-ids edge.
fn token_ids(
    graph: &mut Graph,
    binds: &mut Vec<(EdgeId, TypedBuffer)>,
    t: u32,
    vocab: usize,
) -> EdgeId {
    let ids: Vec<u32> = (0..t).map(|i| (i as usize % vocab) as u32).collect();
    let token_edge = ext(
        graph,
        ExternalInputKind::TokenIds,
        act_d(vec![Dim::Symbolic(ShapeSymbol::T)], DType::U32),
    );
    binds.push((token_edge, TypedBuffer::from_u32(&[t as usize], &ids)));
    token_edge
}

/// Embed front-end producing `[T, Dm]` F16 activations from token ids.
fn embed_front(
    graph: &mut Graph,
    binds: &mut Vec<(EdgeId, TypedBuffer)>,
    token_edge: EdgeId,
    t: u32,
    vocab: usize,
    dm: usize,
    case: u64,
) -> EdgeId {
    let table_vals = fvec(case, vocab * dm, -1.0, 1.0);
    let table_edge = graph
        .add_tensor(weight_d(&[vocab, dm], DType::F16))
        .unwrap();
    binds.push((
        table_edge,
        TypedBuffer::from_bytes(&[vocab, dm], DType::F16, &f16_bytes(&table_vals)),
    ));
    add_node(
        graph,
        Op::EmbedGather(EmbedGatherOp {
            scale: 1.0,
            out_dtype: DType::F16,
        }),
        &[token_edge, table_edge],
        vec![act_d(
            vec![Dim::Concrete(t), Dim::Concrete(dm as u32)],
            DType::F16,
        )],
    )[0]
}

fn reshape_to(graph: &mut Graph, edge: EdgeId, shape: &[usize]) -> EdgeId {
    graph.reshape_edge(edge, concat_dims(shape)).unwrap()
}

// ---------------------------------------------------------------------------
// Elementwise and data-movement ops (Spec 1 §4.A–§4.B)
// ---------------------------------------------------------------------------

#[test]
fn executor_copy_is_bit_identical() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let x = embed_front(&mut graph, &mut binds, tok, 2, 8, 6, 1);
    let y = add_node(
        &mut graph,
        Op::Copy(CopyOp {
            kind: CopyKind::Contiguize,
        }),
        &[x],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(6)], DType::F16)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let (a, b) = (
        exec.edge(x).unwrap().to_f32_vec(),
        exec.edge(y).unwrap().to_f32_vec(),
    );
    check_bits_equal(&f32s_to_bytes(&a), &f32s_to_bytes(&b), "copy identity").unwrap();
    let expected = copy_f64_reference(
        &CopyOp {
            kind: CopyKind::Contiguize,
        },
        &a.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    );
    check_f32_against_f64(tolerance_for("copy").unwrap(), &b, &expected, "copy golden").unwrap();
}

#[test]
fn executor_cast_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let input = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F32),
    );
    let vals = fvec(2, 6, -2.0, 2.0);
    let y = add_node(
        &mut graph,
        Op::Cast(CastOp { dtype: DType::F16 }),
        &[input],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F16)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(
        &graph,
        vec![(input, TypedBuffer::from_f32(&[2, 3], &vals))],
        2,
        no_setup,
    );
    let out = exec.edge(y).unwrap().to_f32_vec();
    let expected = cast_f64_reference(&vals.iter().map(|&v| v as f64).collect::<Vec<_>>());
    check_f32_against_f64(
        tolerance_for("cast").unwrap(),
        &out,
        &expected,
        "cast golden",
    )
    .unwrap();
}

#[test]
fn executor_activation_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let x = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 3);
    let op = ActivationOp {
        act: ActivationKind::Silu,
        clamp: None,
    };
    let y = add_node(
        &mut graph,
        Op::Activation(op),
        &[x],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(4)], DType::F16)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let out = exec.edge(y).unwrap().to_f32_vec();
    let input = exec.edge(x).unwrap().to_f32_vec();
    let expected =
        activation_f64_reference(&op, &input.iter().map(|&v| v as f64).collect::<Vec<_>>());
    check_f32_against_f64(
        tolerance_for("activation").unwrap(),
        &out,
        &expected,
        "activation golden",
    )
    .unwrap();
}

#[test]
fn executor_act_mul_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let g = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 4);
    let u = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 5);
    let op = ActMulOp {
        act: ActivationKind::Silu,
        clamp: None,
    };
    let y = add_node(
        &mut graph,
        Op::ActMul(op),
        &[g, u],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(4)], DType::F16)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let out = exec.edge(y).unwrap().to_f32_vec();
    let to64 = |e: EdgeId| {
        exec.edge(e)
            .unwrap()
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>()
    };
    let expected = act_mul_f64_reference(&op, &to64(g), &to64(u));
    check_f32_against_f64(
        tolerance_for("act_mul").unwrap(),
        &out,
        &expected,
        "act_mul golden",
    )
    .unwrap();
}

#[test]
fn executor_norm_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let x = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 6);
    let w_vals = fvec(7, 4, 0.5, 1.5);
    let w = graph.add_tensor(param_d(&[4])).unwrap();
    binds.push((w, TypedBuffer::from_f32(&[4], &w_vals)));
    let op = NormOp {
        kind: NormKind::Rms,
        eps: 1e-5,
        axis: NormAxis::Last,
        weight_offset: 0.0,
        out_dtype: DType::F16,
    };
    let y = add_node(
        &mut graph,
        Op::Norm(op),
        &[x, w],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(4)], DType::F16)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let out = exec.edge(y).unwrap().to_f32_vec();
    let input: Vec<f64> = exec
        .edge(x)
        .unwrap()
        .to_f32_vec()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let expected = norm_f64_reference(
        &op,
        &input,
        [2, 4],
        &w_vals.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        None,
        0.0,
        1e-5,
    );
    check_f32_against_f64(
        tolerance_for("norm").unwrap(),
        &out,
        &expected,
        "norm golden",
    )
    .unwrap();
}

#[test]
fn executor_residual_add_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let a = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 8);
    let b = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 9);
    let y = add_node(
        &mut graph,
        Op::ResidualAdd(ResidualAddOp {
            out_dtype: DType::F16,
            scale: 1.0,
        }),
        &[a, b],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(4)], DType::F16)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let out = exec.edge(y).unwrap().to_f32_vec();
    let to64 = |e: EdgeId| {
        exec.edge(e)
            .unwrap()
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>()
    };
    let expected = residual_add_f64_reference(&to64(a), &to64(b), 1.0);
    check_f32_against_f64(
        tolerance_for("residual_add").unwrap(),
        &out,
        &expected,
        "residual_add golden",
    )
    .unwrap();
}

#[test]
fn executor_split_concat_round_trip_exact() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let x = embed_front(&mut graph, &mut binds, tok, 2, 8, 6, 10);
    let x3 = reshape_to(&mut graph, x, &[2, 2, 3]);
    let outs = add_node(
        &mut graph,
        Op::Split(SplitOp { first: 1 }),
        &[x3],
        vec![
            act_d(
                vec![Dim::Concrete(2), Dim::Concrete(2), Dim::Concrete(1)],
                DType::F16,
            ),
            act_d(
                vec![Dim::Concrete(2), Dim::Concrete(2), Dim::Concrete(2)],
                DType::F16,
            ),
        ],
    );
    let joined = add_node(
        &mut graph,
        Op::Concat(ConcatOp),
        &[outs[0], outs[1]],
        vec![act_d(
            vec![Dim::Concrete(2), Dim::Concrete(2), Dim::Concrete(3)],
            DType::F16,
        )],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let (a, b) = (
        exec.edge(x).unwrap().to_f32_vec(),
        exec.edge(joined).unwrap().to_f32_vec(),
    );
    check_bits_equal(
        &f32s_to_bytes(&a),
        &f32s_to_bytes(&b),
        "split/concat identity",
    )
    .unwrap();
    let input64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let (ea, eb) = split_f64_reference(&input64, [2, 2, 3], 1);
    check_f32_against_f64(
        tolerance_for("split").unwrap(),
        &exec.edge(outs[0]).unwrap().to_f32_vec(),
        &ea,
        "split golden",
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("split").unwrap(),
        &exec.edge(outs[1]).unwrap().to_f32_vec(),
        &eb,
        "split golden b",
    )
    .unwrap();
    let expected = concat_f64_reference(&ea, &eb, 2, 2);
    check_f32_against_f64(
        tolerance_for("concat").unwrap(),
        &b,
        &expected,
        "concat golden",
    )
    .unwrap();
}

#[test]
fn executor_logit_softcap_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let vals = fvec(11, 8, -20.0, 20.0);
    let x = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(vec![Dim::Concrete(2), Dim::Concrete(4)], DType::F32),
    );
    let y = add_node(
        &mut graph,
        Op::LogitSoftcap(LogitSoftcapOp { cap: 15.0 }),
        &[x],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(4)], DType::F32)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(
        &graph,
        vec![(x, TypedBuffer::from_f32(&[2, 4], &vals))],
        2,
        no_setup,
    );
    let out = exec.edge(y).unwrap().to_f32_vec();
    let expected =
        logit_softcap_f64_reference(&vals.iter().map(|&v| v as f64).collect::<Vec<_>>(), 15.0);
    check_f32_against_f64(
        tolerance_for("logit_softcap").unwrap(),
        &out,
        &expected,
        "logit_softcap golden",
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Matmul and lookup group (Spec 1 §4.A, §4.C)
// ---------------------------------------------------------------------------

#[test]
fn executor_matmul_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let x = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 20);
    let w_vals = fvec(21, 12, -1.0, 1.0);
    let w = graph.add_tensor(weight_d(&[3, 4], DType::F16)).unwrap();
    binds.push((
        w,
        TypedBuffer::from_bytes(&[3, 4], DType::F16, &f16_bytes(&w_vals)),
    ));
    let y = add_node(
        &mut graph,
        Op::Matmul(MatmulOp {
            out_dtype: DType::F16,
            epilogue: Epilogue::None,
            transpose_w: false,
        }),
        &[x, w],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F16)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let out = exec.edge(y).unwrap().to_f32_vec();
    let to64 = |e: EdgeId| {
        exec.edge(e)
            .unwrap()
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>()
    };
    // Oracle sees the F16-rounded inputs the executor actually multiplied.
    let w_f16: Vec<f64> = w_vals
        .iter()
        .map(|&v| r9v_t0::dtype::f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let expected =
        matmul_f64_reference(&to64(x), 2, 4, &w_f16, 3, None, None, Epilogue::None, false);
    check_f32_against_f64(
        tolerance_for("matmul").unwrap(),
        &out,
        &expected,
        "matmul golden",
    )
    .unwrap();
}

#[test]
fn executor_embed_gather_matches_oracle() {
    let mut graph = Graph::new(key(1, 3));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 3, 7);
    let table_vals = fvec(22, 35, -1.0, 1.0);
    let table = graph.add_tensor(weight_d(&[7, 5], DType::F16)).unwrap();
    binds.push((
        table,
        TypedBuffer::from_bytes(&[7, 5], DType::F16, &f16_bytes(&table_vals)),
    ));
    let y = add_node(
        &mut graph,
        Op::EmbedGather(EmbedGatherOp {
            scale: 1.0,
            out_dtype: DType::F16,
        }),
        &[tok, table],
        vec![act_d(vec![Dim::Concrete(3), Dim::Concrete(5)], DType::F16)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 3, no_setup);
    let out = exec.edge(y).unwrap().to_f32_vec();
    let table_f16: Vec<f64> = table_vals
        .iter()
        .map(|&v| r9v_t0::dtype::f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let expected = embed_gather_f64_reference(&[0, 1, 2], &table_f16, 7, 5, 1.0);
    check_f32_against_f64(
        tolerance_for("embed_gather").unwrap(),
        &out,
        &expected,
        "embed_gather golden",
    )
    .unwrap();
}

#[test]
fn executor_gather_rows_is_exact() {
    let mut graph = Graph::new(key(1, 2));
    let x_vals = fvec(23, 6, -2.0, 2.0);
    let x = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F32),
    );
    let idx = ext(
        &mut graph,
        ExternalInputKind::TokenIds,
        act_d(vec![Dim::Concrete(2)], DType::U32),
    );
    let y = add_node(
        &mut graph,
        Op::GatherRows(r9v_ir::GatherRowsOp),
        &[x, idx],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F32)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(
        &graph,
        vec![
            (x, TypedBuffer::from_f32(&[2, 3], &x_vals)),
            (idx, TypedBuffer::from_u32(&[2], &[1, 0])),
        ],
        2,
        no_setup,
    );
    let out = exec.edge(y).unwrap().to_f32_vec();
    let expected = gather_rows_f64_reference(
        &x_vals.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        2,
        3,
        &[1, 0],
    );
    check_f32_against_f64(
        tolerance_for("gather_rows").unwrap(),
        &out,
        &expected,
        "gather_rows golden",
    )
    .unwrap();
}

#[test]
fn executor_scatter_add_rows_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let x_vals = fvec(24, 6, -2.0, 2.0);
    let x = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F32),
    );
    let idx = ext(
        &mut graph,
        ExternalInputKind::TokenIds,
        act_d(vec![Dim::Concrete(2)], DType::U32),
    );
    let y = add_node(
        &mut graph,
        Op::ScatterAddRows(r9v_ir::ScatterAddRowsOp),
        &[x, idx],
        vec![act_d(vec![Dim::Concrete(4), Dim::Concrete(3)], DType::F32)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(
        &graph,
        vec![
            (x, TypedBuffer::from_f32(&[2, 3], &x_vals)),
            (idx, TypedBuffer::from_u32(&[2], &[1, 3])),
        ],
        2,
        no_setup,
    );
    let out = exec.edge(y).unwrap().to_f32_vec();
    let expected = scatter_add_rows_f64_reference(
        &x_vals.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        2,
        3,
        &[1, 3],
        None,
        4,
    );
    check_f32_against_f64(
        tolerance_for("scatter_add_rows").unwrap(),
        &out,
        &expected,
        "scatter_add_rows golden",
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Rope, quant_act, MoE (Spec 1 §4.B–§4.C)
// ---------------------------------------------------------------------------

/// Registers BatchMeta and binds the scalar rope-positions projection.
fn positions_binding(graph: &mut Graph, binds: &mut Vec<(EdgeId, TypedBuffer)>, t: u32) -> EdgeId {
    graph.add_batch_meta_input().unwrap();
    let edge = graph.bind_positions(r9v_ir::PositionsKind::Scalar).unwrap();
    let positions: Vec<u32> = (0..t).collect();
    binds.push((edge, TypedBuffer::from_u32(&[t as usize], &positions)));
    edge
}

#[test]
fn executor_rope_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let flat = embed_front(&mut graph, &mut binds, tok, 2, 8, 8, 30);
    let x = reshape_to(&mut graph, flat, &[2, 2, 4]);
    let pos = positions_binding(&mut graph, &mut binds, 2);
    let op = RopeOp {
        rot_dim: 4,
        theta: 10_000.0,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F16,
    };
    let y = add_node(
        &mut graph,
        Op::Rope(op),
        &[x, pos],
        vec![act_d(
            vec![Dim::Concrete(2), Dim::Concrete(2), Dim::Concrete(4)],
            DType::F16,
        )],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let out = exec.edge(y).unwrap().to_f32_vec();
    let input: Vec<f64> = exec
        .edge(flat)
        .unwrap()
        .to_f32_vec()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let expected = rope_f64_reference(&op, &input, [2, 2, 4], &[0, 1], false);
    check_f32_against_f64(
        tolerance_for("rope").unwrap(),
        &out,
        &expected,
        "rope golden",
    )
    .unwrap();
}

#[test]
fn executor_quant_act_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let x = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 31);
    let op = r9v_ir::QuantActOp {
        scheme: QuantScheme::PerToken,
        target: DType::I8,
        smoothing: r9v_ir::Smoothing::None,
    };
    let outs = add_node(
        &mut graph,
        Op::QuantAct(op),
        &[x],
        vec![
            act_q(
                vec![Dim::Concrete(2), Dim::Concrete(4)],
                DType::I8,
                QuantScheme::PerToken,
            ),
            act_d(vec![Dim::Concrete(2)], DType::F32),
        ],
    );
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let xq = exec.edge(outs[0]).unwrap().to_i8_vec();
    let scales = exec.edge(outs[1]).unwrap().to_f32_vec();
    let input: Vec<f64> = exec
        .edge(x)
        .unwrap()
        .to_f32_vec()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let (exp_xq, exp_scales) = quant_act_f64_reference(&op, &input, [2, 4]);
    let xq_f64: Vec<f64> = xq.iter().map(|&v| v as f64).collect();
    check_f32_against_f64(
        tolerance_for("quant_act").unwrap(),
        &xq_f64.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        &exp_xq,
        "quant_act values",
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("quant_act").unwrap(),
        &scales,
        &exp_scales,
        "quant_act scales",
    )
    .unwrap();
}

#[test]
fn executor_moe_route_matches_oracle() {
    let mut graph = Graph::new(key(1, 3));
    let vals = fvec(32, 12, -2.0, 2.0);
    let logits = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(4)],
            DType::F32,
        ),
    );
    let op = r9v_ir::MoeRouteOp {
        top_k: 2,
        scoring: r9v_ir::MoeScoring::Softmax,
        renormalize: true,
        group: None,
        scale: 1.0,
    };
    let outs = add_node(
        &mut graph,
        Op::MoeRoute(op),
        &[logits],
        vec![
            act_d(vec![Dim::Concrete(3), Dim::Concrete(2)], DType::U32),
            act_d(vec![Dim::Concrete(3), Dim::Concrete(2)], DType::F32),
        ],
    );
    graph.validate().unwrap();
    let exec = run_step(
        &graph,
        vec![(logits, TypedBuffer::from_f32(&[3, 4], &vals))],
        3,
        no_setup,
    );
    let ids = exec.edge(outs[0]).unwrap().to_u32_vec();
    let weights = exec.edge(outs[1]).unwrap().to_f32_vec();
    let input: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
    let (exp_ids, exp_w) = moe_route_f64_reference(
        &input,
        3,
        4,
        None,
        2,
        r9v_ir::MoeScoring::Softmax,
        true,
        1.0,
    )
    .unwrap();
    assert_eq!(ids, exp_ids, "moe_route expert ids are exact");
    check_f32_against_f64(
        tolerance_for("moe_route").unwrap(),
        &weights,
        &exp_w,
        "moe_route golden",
    )
    .unwrap();
}

#[test]
fn executor_moe_ffn_matches_oracle() {
    // T=2, Dm=3, E=4, K=2, Dff=2, Silu, dense F16 experts.
    let mut graph = Graph::new(key(1, 2));
    let route_vals = fvec(33, 8, -2.0, 2.0);
    let logits = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(4)],
            DType::F32,
        ),
    );
    let route = add_node(
        &mut graph,
        Op::MoeRoute(r9v_ir::MoeRouteOp {
            top_k: 2,
            scoring: r9v_ir::MoeScoring::Softmax,
            renormalize: true,
            group: None,
            scale: 1.0,
        }),
        &[logits],
        vec![
            act_d(vec![Dim::Concrete(2), Dim::Concrete(2)], DType::U32),
            act_d(vec![Dim::Concrete(2), Dim::Concrete(2)], DType::F32),
        ],
    );
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let x = embed_front(&mut graph, &mut binds, tok, 2, 8, 3, 34);
    let gu_vals = fvec(35, 48, -1.0, 1.0);
    let wd_vals = fvec(36, 24, -1.0, 1.0);
    let gu = graph.add_tensor(weight_d(&[4, 4, 3], DType::F16)).unwrap();
    let wd = graph.add_tensor(weight_d(&[4, 3, 2], DType::F16)).unwrap();
    binds.push((logits, TypedBuffer::from_f32(&[2, 4], &route_vals)));
    binds.push((
        gu,
        TypedBuffer::from_bytes(&[4, 4, 3], DType::F16, &f16_bytes(&gu_vals)),
    ));
    binds.push((
        wd,
        TypedBuffer::from_bytes(&[4, 3, 2], DType::F16, &f16_bytes(&wd_vals)),
    ));
    let op = r9v_ir::MoeFfnOp {
        act: ActivationKind::Silu,
        out_dtype: DType::F32,
        shared_experts: 0,
    };
    let y = add_node(
        &mut graph,
        Op::MoeFfn(op),
        &[x, route[0], route[1], gu, wd],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F32)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, no_setup);
    let out = exec.edge(y).unwrap().to_f32_vec();
    let to64 = |e: EdgeId| {
        exec.edge(e)
            .unwrap()
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>()
    };
    let gu_f16: Vec<f64> = gu_vals
        .iter()
        .map(|&v| r9v_t0::dtype::f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let wd_f16: Vec<f64> = wd_vals
        .iter()
        .map(|&v| r9v_t0::dtype::f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let expected = moe_ffn_f64_reference(
        &to64(x),
        2,
        3,
        &exec.edge(route[0]).unwrap().to_u32_vec(),
        &to64(route[1]),
        2,
        &gu_f16,
        4,
        2,
        &wd_f16,
        ActivationKind::Silu,
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("moe_ffn").unwrap(),
        &out,
        &expected,
        "moe_ffn golden",
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Stateful ops: conv, scan, ngram, attention (Spec 1 §4.A, §4.D–§4.E)
// ---------------------------------------------------------------------------

#[test]
fn executor_causal_conv1d_threads_state_across_steps() {
    // T=2 per step, C=3, Wk=2, Identity: step 2 output depends on step 1.
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let x = embed_front(&mut graph, &mut binds, tok, 2, 8, 3, 40);
    let w_vals = fvec(41, 6, -1.0, 1.0);
    let w = graph.add_tensor(weight_d(&[3, 2], DType::F16)).unwrap();
    binds.push((
        w,
        TypedBuffer::from_bytes(&[3, 2], DType::F16, &f16_bytes(&w_vals)),
    ));
    let handle = r9v_ir::StateHandle::new(0, r9v_ir::StateKind::ConvWindow);
    let op = r9v_ir::CausalConv1dOp {
        kernel: 2,
        act: r9v_ir::ConvActivation::Identity,
        handle,
    };
    let y = add_node(
        &mut graph,
        Op::CausalConv1d(op.clone()),
        &[x, w],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F16)],
    )[0];
    graph.validate().unwrap();

    let x_f64 = |exec: &CpuExecutor| {
        exec.edge(x)
            .unwrap()
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>()
    };
    let w_f16: Vec<f64> = w_vals
        .iter()
        .map(|&v| r9v_t0::dtype::f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    // Step 1 consumes tokens [0,1]; step 2 rebinds tokens [2,3].
    let mut exec = fresh_exec(binds, no_setup);
    step_run(&mut exec, &graph, 2, 0, &[], &mut Vec::new(), None);
    let y1 = exec.edge(y).unwrap().to_f32_vec();
    let (exp_y1, state1) = causal_conv1d_f64_reference(
        &x_f64(&exec),
        2,
        3,
        &w_f16,
        2,
        None,
        r9v_ir::ConvActivation::Identity,
        &[0.0; 3],
        1,
        &[2],
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("causal_conv1d").unwrap(),
        &y1,
        &exp_y1,
        "conv step 1",
    )
    .unwrap();
    exec.bind(tok, TypedBuffer::from_u32(&[2], &[2, 3]));
    step_run(&mut exec, &graph, 2, 2, &[], &mut Vec::new(), None);
    let y2 = exec.edge(y).unwrap().to_f32_vec();
    let (exp_y2, _) = causal_conv1d_f64_reference(
        &x_f64(&exec),
        2,
        3,
        &w_f16,
        2,
        None,
        r9v_ir::ConvActivation::Identity,
        &state1,
        1,
        &[2],
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("causal_conv1d").unwrap(),
        &y2,
        &exp_y2,
        "conv step 2",
    )
    .unwrap();
}

#[test]
fn executor_linear_attn_scan_threads_state_across_steps() {
    // S=1, T=3 then T=3, H=2, D=4, Dv=4, GatedDeltaNet, recurrent (q < 32).
    let mut graph = Graph::new(key(1, 3));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 3, 8);
    let flat = embed_front(&mut graph, &mut binds, tok, 3, 8, 8, 42);
    let q = reshape_to(&mut graph, flat, &[3, 2, 4]);
    let flat2 = embed_front(&mut graph, &mut binds, tok, 3, 8, 8, 43);
    let k = reshape_to(&mut graph, flat2, &[3, 2, 4]);
    let flat3 = embed_front(&mut graph, &mut binds, tok, 3, 8, 8, 44);
    let v = reshape_to(&mut graph, flat3, &[3, 2, 4]);
    let a_vals = fvec(45, 6, -0.5, 0.5);
    let b_vals = a_vals.clone();
    let alpha = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(2)],
            DType::F32,
        ),
    );
    let beta = add_node(
        &mut graph,
        Op::Copy(CopyOp {
            kind: CopyKind::Contiguize,
        }),
        &[alpha],
        vec![act_d(vec![Dim::Concrete(3), Dim::Concrete(2)], DType::F32)],
    )[0];
    binds.push((alpha, TypedBuffer::from_f32(&[3, 2], &a_vals)));
    let handle = r9v_ir::StateHandle::new(0, r9v_ir::StateKind::Recurrent);
    let op = r9v_ir::LinearAttnScanOp {
        kind: r9v_ir::LinearAttnKind::GatedDeltaNet,
        chunk: 64,
        out_dtype: DType::F32,
        handle,
    };
    let y = add_node(
        &mut graph,
        Op::LinearAttnScan(op.clone()),
        &[q, k, v, alpha, beta],
        vec![act_d(
            vec![Dim::Concrete(3), Dim::Concrete(2), Dim::Concrete(4)],
            DType::F32,
        )],
    )[0];
    graph.validate().unwrap();

    let to64 = |exec: &CpuExecutor, e: EdgeId| {
        exec.edge(e)
            .unwrap()
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>()
    };
    let mut exec = fresh_exec(binds, no_setup);
    step_run(&mut exec, &graph, 3, 0, &[], &mut Vec::new(), None);
    let y1 = exec.edge(y).unwrap().to_f32_vec();
    let (exp_y1, state1) = linear_attn_scan_f64_reference(
        &to64(&exec, flat),
        &to64(&exec, flat2),
        &to64(&exec, flat3),
        &a_vals.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &b_vals.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        3,
        2,
        4,
        4,
        &[0.0; 32],
        1,
        &[3],
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("linear_attn_scan").unwrap(),
        &y1,
        &exp_y1,
        "scan step 1",
    )
    .unwrap();
    exec.bind(tok, TypedBuffer::from_u32(&[3], &[3, 4, 5]));
    step_run(&mut exec, &graph, 3, 3, &[], &mut Vec::new(), None);
    let y2 = exec.edge(y).unwrap().to_f32_vec();
    let (exp_y2, _) = linear_attn_scan_f64_reference(
        &to64(&exec, flat),
        &to64(&exec, flat2),
        &to64(&exec, flat3),
        &a_vals.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &b_vals.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        3,
        2,
        4,
        4,
        &state1,
        1,
        &[3],
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("linear_attn_scan").unwrap(),
        &y2,
        &exp_y2,
        "scan step 2",
    )
    .unwrap();
}

#[test]
fn executor_ngram_staged_matches_oracle() {
    let mut graph = Graph::new(key(1, 2));
    let staging_vals =
        r9v_t0::harness::symmetric_i8(&mut rng_for("a1.12-exec", 47, MASTER_SEED), 24);
    let scale_vals = fvec(48, 6, 0.25, 2.0);
    let staging = ext(
        &mut graph,
        ExternalInputKind::GatherStaging,
        r9v_ir::Tensor::new(
            vec![Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(4)],
            DType::I8,
            QuantScheme::Scheme(r9v_format::SchemeId::I8R.to_ir()),
            LayoutId::CONTIGUOUS,
            Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Staging,
        )
        .unwrap(),
    );
    let scales = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(vec![Dim::Concrete(2), Dim::Concrete(3)], DType::F32),
    );
    let op = r9v_ir::NgramGatherOp {
        source: r9v_ir::NgramSource::Staged,
        orders: vec![1, 1, 1].into_boxed_slice(),
        heads: 3,
        hash: r9v_ir::HashId::new(0),
        table_sizes: vec![64, 64, 64].into_boxed_slice(),
        combine: r9v_ir::NgramCombine::Concat,
        out_dtype: DType::F32,
    };
    let y = add_node(
        &mut graph,
        Op::NgramGather(op.clone()),
        &[staging, scales],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(12)], DType::F32)],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(
        &graph,
        vec![
            (
                staging,
                TypedBuffer::from_i8(&[2, 3, 4], &staging_vals)
                    .with_quant(QuantScheme::Scheme(r9v_format::SchemeId::I8R.to_ir())),
            ),
            (scales, TypedBuffer::from_f32(&[2, 3], &scale_vals)),
        ],
        2,
        no_setup,
    );
    let out = exec.edge(y).unwrap().to_f32_vec();
    let expected = ngram_gather_f64_reference_staged(
        &staging_vals,
        &scale_vals.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        2,
        3,
        4,
        r9v_ir::NgramCombine::Concat,
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("ngram_gather").unwrap(),
        &out,
        &expected,
        "ngram staged golden",
    )
    .unwrap();
}

struct ModHash;
impl r9v_t0::ngram_gather::NgramHash for ModHash {
    fn row(&self, tokens: &[u32], pos: usize, order: u32, table_size: u32) -> u32 {
        let arc = tokens
            .iter()
            .fold(0u32, |a, &b| a.wrapping_add(b).wrapping_mul(31));
        arc.wrapping_add(pos as u32).wrapping_add(order) % table_size.max(1)
    }
}

#[test]
fn executor_ngram_device_matches_oracle() {
    let mut graph = Graph::new(key(1, 3));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 3, 16);
    let table_vals = fvec(49, 64, -1.0, 1.0);
    let table = graph.add_tensor(weight_d(&[16, 4], DType::F16)).unwrap();
    binds.push((
        table,
        TypedBuffer::from_bytes(&[16, 4], DType::F16, &f16_bytes(&table_vals)),
    ));
    let op = r9v_ir::NgramGatherOp {
        source: r9v_ir::NgramSource::Device,
        orders: vec![1].into_boxed_slice(),
        heads: 1,
        hash: r9v_ir::HashId::new(0),
        table_sizes: vec![16].into_boxed_slice(),
        combine: r9v_ir::NgramCombine::Concat,
        out_dtype: DType::F32,
    };
    let y = add_node(
        &mut graph,
        Op::NgramGather(op.clone()),
        &[tok, table],
        vec![act_d(vec![Dim::Concrete(3), Dim::Concrete(4)], DType::F32)],
    )[0];
    graph.validate().unwrap();
    let mut exec = fresh_exec(binds, no_setup);
    step_run(
        &mut exec,
        &graph,
        3,
        0,
        &[],
        &mut Vec::new(),
        Some(&ModHash),
    );
    let out = exec.edge(y).unwrap().to_f32_vec();
    let hash = ModHash;
    let tokens = [0u32, 1, 2];
    let row_ids: Vec<u32> = (0..3).map(|p| hash.row(&tokens, p, 1, 16)).collect();
    let table_f16: Vec<f64> = table_vals
        .iter()
        .map(|&v| r9v_t0::dtype::f16_to_f32(f32_to_f16(v)) as f64)
        .collect();
    let expected = ngram_gather_f64_reference_rows(
        &table_f16,
        16,
        4,
        &row_ids,
        3,
        1,
        r9v_ir::NgramCombine::Concat,
    )
    .unwrap();
    check_f32_against_f64(
        tolerance_for("ngram_gather").unwrap(),
        &out,
        &expected,
        "ngram device golden",
    )
    .unwrap();
}

#[test]
fn executor_attention_reads_written_cache() {
    // S=1, T=2, H=2, Hkv=2, D=4, Dv=4, F16 paged cache.
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let flat = embed_front(&mut graph, &mut binds, tok, 2, 8, 8, 50);
    let q = reshape_to(&mut graph, flat, &[2, 2, 4]);
    let flat2 = embed_front(&mut graph, &mut binds, tok, 2, 8, 8, 51);
    let k = reshape_to(&mut graph, flat2, &[2, 2, 4]);
    let flat3 = embed_front(&mut graph, &mut binds, tok, 2, 8, 8, 52);
    let v = reshape_to(&mut graph, flat3, &[2, 2, 4]);
    graph.add_batch_meta_input().unwrap();
    let handle = r9v_ir::StateHandle::new(0, r9v_ir::StateKind::KvPaged);
    add_node(
        &mut graph,
        Op::StateWriteKv(r9v_ir::StateWriteKvOp {
            cache_dtype: DType::F16,
            scale_granularity: r9v_ir::CacheScaleGranularity::PerTokenHead,
            latent: None,
            handle,
        }),
        &[k, v],
        vec![],
    );
    let attn = r9v_ir::AttentionOp {
        softmax_scale: 0.5,
        mask: r9v_ir::AttentionMask::Causal,
        sinks: 0,
        logit_softcap: None,
        mla: None,
        out_dtype: DType::F16,
        handle,
    };
    let o = add_node(
        &mut graph,
        Op::Attention(attn.clone()),
        &[q],
        vec![act_d(
            vec![Dim::Concrete(2), Dim::Concrete(2), Dim::Concrete(4)],
            DType::F16,
        )],
    )[0];
    graph.validate().unwrap();
    let exec = run_step(&graph, binds, 2, |exec: &mut CpuExecutor| {
        exec.register_paged_cache(
            handle,
            r9v_t0::attention::KvPagedCache::new(2, 2, 4, 4, DType::F16).unwrap(),
        );
    });
    let out = exec.edge(o).unwrap().to_f32_vec();
    // Per-query oracle rows over the causal prefix (heads independent).
    let to64 = |e: EdgeId| {
        exec.edge(e)
            .unwrap()
            .to_f32_vec()
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>()
    };
    let (qq, kk, vv) = (to64(flat), to64(flat2), to64(flat3));
    let row = |buf: &[f64], token: usize, head: usize| {
        buf[token * 8 + head * 4..token * 8 + head * 4 + 4].to_vec()
    };
    let mut expected = Vec::with_capacity(16);
    for token in 0..2 {
        for head in 0..2 {
            let k_rows: Vec<Vec<f64>> = (0..=token).map(|t| row(&kk, t, head)).collect();
            let v_rows: Vec<Vec<f64>> = (0..=token).map(|t| row(&vv, t, head)).collect();
            expected.extend(attention_row_f64_reference(
                &row(&qq, token, head),
                &k_rows,
                &v_rows,
                0.5,
                None,
            ));
        }
    }
    check_f32_against_f64(
        tolerance_for("attention").unwrap(),
        &out,
        &expected,
        "attention golden",
    )
    .unwrap();
}

#[test]
fn synthetic_model_decodes_deterministically_on_cpu() {
    let spec = r9v_t0::synthetic::SyntheticSpec::test_default();
    let model = r9v_t0::synthetic::build(&spec).unwrap();
    let prompt = vec![1u32, 2, 3, 4];
    let config = r9v_t0::decode::DecodeConfig {
        max_new_tokens: 8,
        eos: None,
    };

    let mut exec1 = CpuExecutor::new();
    let res1 = r9v_t0::decode::decode_greedy(&mut exec1, &model, &prompt, &config).unwrap();
    assert_eq!(res1.prompt_len, 4);
    assert_eq!(res1.vocab, spec.vocab as usize);
    assert_eq!(res1.generated.len(), 8);
    assert_eq!(res1.step_logits.len(), 8);

    // Determinism gate: second execution from fresh state must be bit-identical.
    let mut exec2 = CpuExecutor::new();
    let res2 = r9v_t0::decode::decode_greedy(&mut exec2, &model, &prompt, &config).unwrap();
    assert_eq!(res1.generated, res2.generated);
    check_bits_equal(
        &f32s_to_bytes(&res1.prompt_logits),
        &f32s_to_bytes(&res2.prompt_logits),
        "prompt logits bit identical",
    )
    .unwrap();
    for (step, (l1, l2)) in res1.step_logits.iter().zip(&res2.step_logits).enumerate() {
        check_bits_equal(
            &f32s_to_bytes(l1),
            &f32s_to_bytes(l2),
            &format!("step {step} logits bit identical"),
        )
        .unwrap();
    }

    // Stop token (eos) early termination.
    let stop_token = res1.generated[2];
    let eos_config = r9v_t0::decode::DecodeConfig {
        max_new_tokens: 8,
        eos: Some(stop_token),
    };
    let mut exec3 = CpuExecutor::new();
    let res3 = r9v_t0::decode::decode_greedy(&mut exec3, &model, &prompt, &eos_config).unwrap();
    assert_eq!(res3.generated.len(), 2);
    assert_eq!(res3.generated, res1.generated[..2]);

    // Typed error: empty prompt.
    let mut exec_err = CpuExecutor::new();
    let err_empty = r9v_t0::decode::decode_greedy(&mut exec_err, &model, &[], &config);
    assert!(matches!(
        err_empty,
        Err(ExecError::T0(r9v_t0::error::T0Error::EmptyInput { .. }))
    ));

    // Typed error: token out of range.
    let err_oob = r9v_t0::decode::decode_greedy(&mut exec_err, &model, &[spec.vocab], &config);
    assert!(matches!(
        err_oob,
        Err(ExecError::T0(
            r9v_t0::error::T0Error::TokenOutOfRange { .. }
        ))
    ));

    // Typed error: prompt + max_new_tokens exceeds max_ctx.
    let overflow_config = r9v_t0::decode::DecodeConfig {
        max_new_tokens: spec.max_ctx + 1,
        eos: None,
    };
    let err_overflow =
        r9v_t0::decode::decode_greedy(&mut exec_err, &model, &prompt, &overflow_config);
    assert!(matches!(
        err_overflow,
        Err(ExecError::T0(
            r9v_t0::error::T0Error::ArithmeticOverflow { .. }
        ))
    ));
}

#[test]
fn executor_logits_postprocess_matches_oracle() {
    let mut graph = Graph::new(key(1, 1));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 1, 8);
    let flat = embed_front(&mut graph, &mut binds, tok, 1, 8, 4, 80);
    let flat_f32 = add_node(
        &mut graph,
        Op::Cast(CastOp { dtype: DType::F32 }),
        &[flat],
        vec![act_d(vec![Dim::Concrete(1), Dim::Concrete(4)], DType::F32)],
    )[0];
    let logits = reshape_to(&mut graph, flat_f32, &[1, 1, 4]);
    let y = add_node(
        &mut graph,
        Op::LogitsPostprocess(LogitsPostprocessOp),
        &[logits],
        vec![act_d(
            vec![
                Dim::Symbolic(ShapeSymbol::S),
                Dim::Concrete(1),
                Dim::Concrete(4),
            ],
            DType::F32,
        )],
    )[0];
    graph.add_sampling_params_input().unwrap();
    graph.validate().unwrap();

    let mut exec = fresh_exec(binds, no_setup);
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
    let mut rng = Vec::new();
    step_run(&mut exec, &graph, 1, 0, &params, &mut rng, None);
    let out = exec.edge(y).unwrap().to_f32_vec();

    let logits_f32 = exec.edge(flat_f32).unwrap().to_f32_vec();
    let logits_f64: Vec<f64> = logits_f32.iter().map(|&v| v as f64).collect();
    let expected = logits_postprocess_f64_reference(&logits_f64, 1, 1, 4, &params, None, None);
    check_f32_against_f64(
        tolerance_for("logits_postprocess").unwrap(),
        &out,
        &expected,
        "logits_postprocess golden",
    )
    .unwrap();
}

#[test]
fn executor_sample_draws_deterministic_token() {
    let mut graph = Graph::new(key(1, 1));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 1, 8);
    let flat = embed_front(&mut graph, &mut binds, tok, 1, 8, 4, 81);
    let flat_f32 = add_node(
        &mut graph,
        Op::Cast(CastOp { dtype: DType::F32 }),
        &[flat],
        vec![act_d(vec![Dim::Concrete(1), Dim::Concrete(4)], DType::F32)],
    )[0];
    let logits = reshape_to(&mut graph, flat_f32, &[1, 1, 4]);
    let probs_3d = add_node(
        &mut graph,
        Op::LogitsPostprocess(LogitsPostprocessOp),
        &[logits],
        vec![act_d(
            vec![
                Dim::Symbolic(ShapeSymbol::S),
                Dim::Concrete(1),
                Dim::Concrete(4),
            ],
            DType::F32,
        )],
    )[0];
    let probs_2d = reshape_to(&mut graph, probs_3d, &[1, 4]);
    let y = add_node(
        &mut graph,
        Op::Sample(SampleOp {
            rng: RngAlgorithm::Philox4x32,
        }),
        &[probs_2d],
        vec![act_d(vec![Dim::Symbolic(ShapeSymbol::S)], DType::U32)],
    )[0];
    graph.add_sampling_params_input().unwrap();
    graph.add_rng_state_input().unwrap();
    graph.add_updated_rng_state_output().unwrap();
    graph.validate().unwrap();

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

    let mut exec1 = fresh_exec(binds.clone(), no_setup);
    let mut rng1 = vec![RngState::new(42, SeqId::new(0), StepId::new(0)).unwrap()];
    step_run(&mut exec1, &graph, 1, 0, &params, &mut rng1, None);
    let tok1 = exec1.edge(y).unwrap().to_u32_vec();

    let mut exec2 = fresh_exec(binds, no_setup);
    let mut rng2 = vec![RngState::new(42, SeqId::new(0), StepId::new(0)).unwrap()];
    step_run(&mut exec2, &graph, 1, 0, &params, &mut rng2, None);
    let tok2 = exec2.edge(y).unwrap().to_u32_vec();

    assert_eq!(tok1, tok2, "sampling is bit-deterministic with same seed");
    assert!(tok1[0] < 4, "token is within vocabulary");
}

#[test]
fn executor_verify_matches_greedy_acceptance() {
    let mut graph = Graph::new(key(2, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    // Draft: tokens reshaped to [2, 1] (S=2, k=1)
    let draft_tok = reshape_to(&mut graph, tok, &[2, 1]);

    // Target: [2, 8] -> [2, 2, 4]
    let flat = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(vec![Dim::Concrete(2), Dim::Concrete(8)], DType::F32),
    );
    let target_probs = reshape_to(&mut graph, flat, &[2, 2, 4]);

    let op = VerifyOp {
        method: VerifyMethod::Greedy,
    };
    let outs = add_node(
        &mut graph,
        Op::Verify(op),
        &[draft_tok, target_probs],
        vec![
            act_d(vec![Dim::Concrete(2), Dim::Concrete(2)], DType::U32),
            act_d(vec![Dim::Concrete(2)], DType::U32),
        ],
    );
    graph.add_rng_state_input().unwrap();
    graph.add_updated_rng_state_output().unwrap();
    graph.validate().unwrap();

    let target_vals = vec![0.25f32; 16];
    binds.push((flat, TypedBuffer::from_f32(&[2, 8], &target_vals)));
    let mut exec = fresh_exec(binds, no_setup);
    let mut rng = vec![
        RngState::new(1, SeqId::new(0), StepId::new(0)).unwrap(),
        RngState::new(2, SeqId::new(1), StepId::new(0)).unwrap(),
    ];
    let batch = BatchMeta::builder(1, 2, 2, 2)
        .seq_ids(vec![0, 1])
        .query_len(vec![1, 1])
        .ctx_len(vec![0, 0])
        .positions(Positions::PerToken(vec![0, 0]))
        .slot_map(vec![0, 1])
        .block_table(vec![0, 1, 0, 1])
        .window_start(vec![0, 0])
        .tree(None)
        .build()
        .unwrap();
    exec.run(
        &graph,
        RunArgs {
            batch: &batch,
            params: &[],
            rng: &mut rng,
            ngram_hash: None,
        },
    )
    .unwrap();

    let accepted = exec.edge(outs[0]).unwrap().to_u32_vec();
    let accept_len = exec.edge(outs[1]).unwrap().to_u32_vec();
    assert_eq!(accept_len.len(), 2);
    assert_eq!(accepted.len(), 4);
}

#[test]
fn executor_collective_all_reduce_and_barrier() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let flat = embed_front(&mut graph, &mut binds, tok, 2, 8, 2, 83);
    let x = add_node(
        &mut graph,
        Op::Cast(CastOp { dtype: DType::F32 }),
        &[flat],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(2)], DType::F32)],
    )[0];
    let barrier = add_node(
        &mut graph,
        Op::Barrier(BarrierOp {
            group: GroupId::new(0),
        }),
        &[],
        vec![],
    );
    assert!(barrier.is_empty());
    let reduce = add_node(
        &mut graph,
        Op::AllReduce(AllReduceOp {
            group: GroupId::new(0),
            op: ReduceOp::Sum,
            dtype: DType::F32,
            reduce_in: DType::F32,
        }),
        &[x],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(2)], DType::F32)],
    )[0];
    graph.validate().unwrap();

    let exec = run_step(&graph, binds, 2, no_setup);
    let x_vals = exec.edge(x).unwrap().to_f32_vec();
    let out = exec.edge(reduce).unwrap().to_f32_vec();
    assert_eq!(out, x_vals, "single-device T0 all_reduce sum is identity");
}

#[test]
fn executor_refuses_unbound_edge() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let w = graph.add_tensor(weight_d(&[8, 4], DType::F16)).unwrap();
    // Intentionally omit binding `w` to `binds`
    let _y = add_node(
        &mut graph,
        Op::EmbedGather(EmbedGatherOp {
            scale: 1.0,
            out_dtype: DType::F16,
        }),
        &[tok, w],
        vec![act_d(vec![Dim::Concrete(2), Dim::Concrete(4)], DType::F16)],
    )[0];
    graph.validate().unwrap();

    let mut exec = fresh_exec(binds, no_setup);
    let batch = step_batch(2, 0);
    let err = exec.run(
        &graph,
        RunArgs {
            batch: &batch,
            params: &[],
            rng: &mut [],
            ngram_hash: None,
        },
    );
    assert!(matches!(err, Err(ExecError::UnboundEdge { .. })));
}

#[test]
fn executor_refuses_unknown_state_and_missing_group() {
    let mut graph = Graph::new(key(1, 2));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 2, 8);
    let flat = embed_front(&mut graph, &mut binds, tok, 2, 8, 4, 84);
    let q = reshape_to(&mut graph, flat, &[2, 1, 4]);
    let handle = r9v_ir::StateHandle::new(0, r9v_ir::StateKind::KvPaged);
    graph.add_batch_meta_input().unwrap();
    let _o = add_node(
        &mut graph,
        Op::Attention(r9v_ir::AttentionOp {
            softmax_scale: 0.5,
            mask: r9v_ir::AttentionMask::Causal,
            sinks: 0,
            logit_softcap: None,
            mla: None,
            out_dtype: DType::F16,
            handle,
        }),
        &[q],
        vec![act_d(
            vec![Dim::Concrete(2), Dim::Concrete(1), Dim::Concrete(4)],
            DType::F16,
        )],
    );
    graph.validate().unwrap();

    let mut exec = fresh_exec(binds, no_setup);
    let batch = step_batch(2, 0);
    let err = exec.run(
        &graph,
        RunArgs {
            batch: &batch,
            params: &[],
            rng: &mut [],
            ngram_hash: None,
        },
    );
    assert!(matches!(err, Err(ExecError::UnknownState { .. })));

    exec.register_paged_cache(
        handle,
        r9v_t0::attention::KvPagedCache::new(2, 1, 4, 4, DType::F16).unwrap(),
    );
    let multi_batch = BatchMeta::builder(2, 1, 2, 2)
        .seq_ids(vec![0])
        .query_len(vec![2])
        .ctx_len(vec![0])
        .positions(Positions::PerToken(vec![0, 1]))
        .slot_map(vec![0, 1, 0, 1])
        .block_table(vec![0, 1, 0, 1])
        .window_start(vec![0, 0])
        .tree(None)
        .build()
        .unwrap();
    let err_grp = exec.run(
        &graph,
        RunArgs {
            batch: &multi_batch,
            params: &[],
            rng: &mut [],
            ngram_hash: None,
        },
    );
    assert!(matches!(err_grp, Err(ExecError::UnknownGroup { .. })));
}

#[test]
fn executor_refuses_sampling_context_mismatch_and_missing_ngram_hash() {
    let mut graph = Graph::new(key(1, 1));
    let mut binds = Vec::new();
    let tok = token_ids(&mut graph, &mut binds, 1, 8);
    let flat = embed_front(&mut graph, &mut binds, tok, 1, 8, 4, 85);
    let flat_f32 = add_node(
        &mut graph,
        Op::Cast(CastOp { dtype: DType::F32 }),
        &[flat],
        vec![act_d(vec![Dim::Concrete(1), Dim::Concrete(4)], DType::F32)],
    )[0];
    let logits = reshape_to(&mut graph, flat_f32, &[1, 1, 4]);
    let _y = add_node(
        &mut graph,
        Op::LogitsPostprocess(LogitsPostprocessOp),
        &[logits],
        vec![act_d(
            vec![
                Dim::Symbolic(ShapeSymbol::S),
                Dim::Concrete(1),
                Dim::Concrete(4),
            ],
            DType::F32,
        )],
    );
    graph.add_sampling_params_input().unwrap();
    graph.validate().unwrap();

    let mut exec = fresh_exec(binds, no_setup);
    let batch = step_batch(1, 0);
    let err = exec.run(
        &graph,
        RunArgs {
            batch: &batch,
            params: &[],
            rng: &mut [],
            ngram_hash: None,
        },
    );
    assert!(matches!(err, Err(ExecError::SamplingContext { .. })));

    let mut graph2 = Graph::new(key(1, 1));
    let tok = ext(
        &mut graph2,
        ExternalInputKind::TokenIds,
        act_d(vec![Dim::Concrete(1)], DType::U32),
    );
    let table = graph2.add_tensor(weight_d(&[4, 2], DType::F16)).unwrap();
    let _y2 = add_node(
        &mut graph2,
        Op::NgramGather(r9v_ir::NgramGatherOp {
            source: r9v_ir::NgramSource::Device,
            orders: vec![1].into_boxed_slice(),
            heads: 1,
            hash: r9v_ir::HashId::new(0),
            table_sizes: vec![4].into_boxed_slice(),
            combine: r9v_ir::NgramCombine::Concat,
            out_dtype: DType::F32,
        }),
        &[tok, table],
        vec![act_d(vec![Dim::Concrete(1), Dim::Concrete(2)], DType::F32)],
    );
    graph2.validate().unwrap();

    let mut exec2 = CpuExecutor::new();
    exec2.bind(tok, TypedBuffer::from_u32(&[1], &[0]));
    exec2.bind(table, TypedBuffer::zeros(&[4, 2], DType::F16));
    let err_hash = exec2.run(
        &graph2,
        RunArgs {
            batch: &batch,
            params: &[],
            rng: &mut [],
            ngram_hash: None,
        },
    );
    assert!(matches!(err_hash, Err(ExecError::MissingNgramHash { .. })));
}

#[test]
fn synthetic_spec_and_build_refuse_arithmetic_overflow() {
    use r9v_t0::synthetic::{build as build_synthetic, SyntheticSpec};

    // 1. heads * head_dim overflow
    let mut spec = SyntheticSpec::test_default();
    spec.heads = 0x8000_0000;
    spec.kv_heads = 1;
    spec.head_dim = 2;
    let err_hd = spec.validate();
    assert!(
        matches!(
            err_hd,
            Err(ExecError::T0(r9v_t0::error::T0Error::ArithmeticOverflow { ref detail, .. }))
            if detail.contains("heads")
        ),
        "expected ArithmeticOverflow on heads * head_dim, got {err_hd:?}"
    );
    let build_err_hd = build_synthetic(&spec);
    assert!(matches!(
        build_err_hd,
        Err(ExecError::T0(
            r9v_t0::error::T0Error::ArithmeticOverflow { .. }
        ))
    ));

    // 2. kv_heads * head_dim overflow
    let mut spec_kv = SyntheticSpec::test_default();
    spec_kv.heads = 0x8000_0000;
    spec_kv.kv_heads = 0x8000_0000;
    spec_kv.head_dim = 2;
    let err_kv = spec_kv.validate();
    let err_kv_str = format!("{err_kv:?}");
    assert!(
        err_kv_str.contains("kv_heads") && err_kv_str.contains("overflows u32"),
        "expected ArithmeticOverflow on kv_heads * head_dim, got {err_kv:?}"
    );

    // 3. Shape product and byte size product overflow
    let mut spec_shape = SyntheticSpec::test_default();
    spec_shape.vocab = u32::MAX;
    spec_shape.dim = u32::MAX;
    let err_shape = spec_shape.validate();
    let err_shape_str = format!("{err_shape:?}");
    assert!(
        err_shape_str.contains("ArithmeticOverflow") && err_shape_str.contains("overflows usize"),
        "expected ArithmeticOverflow on shape product, got {err_shape:?}"
    );

    // 4. Over-sized values reject in build before allocation
    let mut bad_spec = SyntheticSpec::test_default();
    bad_spec.vocab = u32::MAX;
    bad_spec.dim = u32::MAX;
    let err_build = build_synthetic(&bad_spec).err().expect("build must fail");
    let err_build_str = format!("{err_build:?}");
    assert!(
        err_build_str.contains("ArithmeticOverflow") && err_build_str.contains("overflows usize"),
        "expected ArithmeticOverflow on build, got {err_build_str}"
    );
}

#[test]
fn typed_buffer_try_zeros_refuses_overflow() {
    use r9v_t0::buffer::TypedBuffer;
    use r9v_t0::error::T0Error;

    // Shape product overflow
    let err_shape = TypedBuffer::try_zeros(&[usize::MAX, 2], DType::F32);
    assert!(matches!(err_shape, Err(T0Error::ArithmeticOverflow { .. })));

    // Byte size overflow (elements fits in usize, but elements * 4 overflows)
    let err_bytes = TypedBuffer::try_zeros(&[usize::MAX / 2, 2], DType::F32);
    assert!(matches!(err_bytes, Err(T0Error::ArithmeticOverflow { .. })));
}

#[test]
fn executor_output_allocation_refuses_overflow_before_allocating() {
    let mut graph = Graph::new(key(1, 1));
    let x = ext(
        &mut graph,
        ExternalInputKind::EmbedOverride,
        act_d(vec![Dim::Concrete(1), Dim::Concrete(1)], DType::F32),
    );
    // An op node declaring an output shape whose product overflows usize
    let huge_dim = (u32::MAX / 2) + 1;
    let _ = add_node(
        &mut graph,
        Op::ResidualAdd(r9v_ir::ResidualAddOp {
            out_dtype: DType::F32,
            scale: 1.0,
        }),
        &[x, x],
        vec![act_d(
            vec![
                Dim::Concrete(huge_dim),
                Dim::Concrete(huge_dim),
                Dim::Concrete(huge_dim),
            ],
            DType::F32,
        )],
    );

    let mut exec = CpuExecutor::new();
    exec.bind(x, TypedBuffer::from_f32(&[1, 1], &[1.0]));
    let batch = step_batch(1, 0);
    let err = exec.run(
        &graph,
        RunArgs {
            batch: &batch,
            params: &[],
            rng: &mut [],
            ngram_hash: None,
        },
    );
    assert!(
        matches!(
            err,
            Err(ExecError::T0(r9v_t0::error::T0Error::ArithmeticOverflow { ref detail, .. }))
            if detail.contains("overflows usize")
        ),
        "expected ArithmeticOverflow on executor output allocation, got {err:?}"
    );
}
