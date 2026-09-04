// SPDX-License-Identifier: Apache-2.0
//! Tiny seeded dense decoder for CPU execution tests and `r9v eval`
//! (Spec 8 family shape, Spec 4 §2 T0 device, Card A1.12).
//!
//! [`SyntheticSpec`] describes the model; [`build`] emits a symbolic
//! single-group step [`Graph`] plus seeded [`TypedBuffer`] weights. The
//! architecture is one fixed dense decoder (embed → L×{rms-norm, GQA,
//! rope, SwiGLU} → norm → lm_head) with F16 activations/weights and F32
//! norm weights, matching what the T0 matmul/norm/attention contracts
//! accept. It is a test and reference-eval vehicle, not a new model
//! family: no new op, dtype, or scheme.
//!
//! [`TinyModel`] names the per-step edges (token ids, rope positions,
//! logits) and the per-layer KV handles so [`crate::exec::CpuExecutor`]
//! and [`crate::decode`] can drive prefill and single-sequence decode.

use r9v_ir::{
    ActMulOp, ActivationKind, AttentionMask, AttentionOp, CacheScaleGranularity, Class, DType, Dim,
    EdgeId, EmbedGatherOp, Epilogue, ExternalInputKind, Graph, LayoutId, MatmulOp, NormAxis,
    NormKind, NormOp, Op, Placement, PositionsKind, QuantScheme, ResidualAddOp, RopeOp,
    RopeScaling, RopeStyle, ShapeSymbol, ShardLayout, StateHandle, StateKind, StateWriteKvOp,
    StepGraphKey,
};
use serde::{Deserialize, Serialize};

use crate::buffer::TypedBuffer;
use crate::dtype::f32_to_f16;
use crate::exec::ExecError;
use crate::harness::{rng_for, uniform_f32};

/// Fixed block length of the paged KV cache (Spec 3 §3.3).
pub const CACHE_BLOCK_TOKENS: u32 = 32;

/// Serializable description of the tiny dense decoder (Spec 4 §2, Spec 8 §3, Card A1.12).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSpec {
    /// Vocabulary size.
    pub vocab: u32,
    /// Model width.
    pub dim: u32,
    /// Query heads.
    pub heads: u32,
    /// KV heads (GQA grouping requires `heads % kv_heads == 0`).
    pub kv_heads: u32,
    /// Head dim (also the value dim).
    pub head_dim: u32,
    /// Feed-forward width.
    pub ff: u32,
    /// Decoder layers.
    pub layers: u32,
    /// RoPE base frequency.
    pub theta: f32,
    /// Weight seed (the `"a1.12"` harness domain).
    pub seed: u64,
    /// Maximum context positions (cache + block-table sizing).
    pub max_ctx: u32,
}

impl SyntheticSpec {
    /// The default test shape: 2 layers, GQA-2/2, V=64 (Spec 4 §2, Card A1.12).
    pub fn test_default() -> Self {
        Self {
            vocab: 64,
            dim: 32,
            heads: 2,
            kv_heads: 2,
            head_dim: 8,
            ff: 64,
            layers: 2,
            theta: 10_000.0,
            seed: 0xA112,
            max_ctx: 64,
        }
    }

    /// Validates the dims the builder and executor require (Spec 4 §2, Spec 8 §3).
    pub fn validate(&self) -> Result<(), ExecError> {
        let mut problems = Vec::new();
        for (name, value) in [
            ("vocab", self.vocab),
            ("dim", self.dim),
            ("heads", self.heads),
            ("kv_heads", self.kv_heads),
            ("head_dim", self.head_dim),
            ("ff", self.ff),
            ("layers", self.layers),
            ("max_ctx", self.max_ctx),
        ] {
            if value == 0 {
                problems.push(ExecError::T0(crate::error::T0Error::InvalidAttribute {
                    op: "synthetic",
                    attribute: "spec",
                    reason: format!("{name} must be > 0, got 0"),
                }));
            }
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            problems.push(ExecError::T0(crate::error::T0Error::InvalidAttribute {
                op: "synthetic",
                attribute: "theta",
                reason: format!("theta must be finite and > 0, got {}", self.theta),
            }));
        }
        if !self.heads.is_multiple_of(self.kv_heads.max(1)) {
            problems.push(ExecError::T0(crate::error::T0Error::InvalidAttribute {
                op: "synthetic",
                attribute: "heads",
                reason: format!(
                    "heads ({}) must be a multiple of kv_heads ({}) for GQA grouping",
                    self.heads, self.kv_heads
                ),
            }));
        }

        // Audited arithmetic: heads*head_dim and kv_heads*head_dim bounds checks.
        let hd = match self.heads.checked_mul(self.head_dim) {
            Some(v) => Some(v),
            None => {
                problems.push(ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                    op: "synthetic",
                    detail: format!(
                        "heads ({}) * head_dim ({}) overflows u32",
                        self.heads, self.head_dim
                    ),
                }));
                None
            }
        };
        let hkv_d = match self.kv_heads.checked_mul(self.head_dim) {
            Some(v) => Some(v),
            None => {
                problems.push(ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                    op: "synthetic",
                    detail: format!(
                        "kv_heads ({}) * head_dim ({}) overflows u32",
                        self.kv_heads, self.head_dim
                    ),
                }));
                None
            }
        };

        // Audited arithmetic: shape products and byte-size products for untrusted parameters.
        let mut check_tensor_bounds = |name: &'static str, shape: &[usize], elem_bytes: usize| {
            let mut prod = 1usize;
            for &d in shape {
                match prod.checked_mul(d) {
                    Some(next) => prod = next,
                    None => {
                        problems.push(ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                            op: "synthetic",
                            detail: format!(
                                "tensor `{name}` shape {shape:?} element product overflows usize"
                            ),
                        }));
                        return;
                    }
                }
            }
            if prod.checked_mul(elem_bytes).is_none() {
                problems.push(ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                    op: "synthetic",
                    detail: format!(
                        "tensor `{name}` shape {shape:?} byte size (elements * {elem_bytes}) overflows usize"
                    ),
                }));
            }
        };

        let v = self.vocab as usize;
        let d = self.dim as usize;
        let ff = self.ff as usize;

        check_tensor_bounds("embed", &[v, d], std::mem::size_of::<u16>());
        check_tensor_bounds("final_norm", &[d], std::mem::size_of::<f32>());
        check_tensor_bounds("lm_head", &[v, d], std::mem::size_of::<u16>());
        check_tensor_bounds("attn_norm", &[d], std::mem::size_of::<f32>());
        check_tensor_bounds("ffn_norm", &[d], std::mem::size_of::<f32>());
        check_tensor_bounds("wg", &[ff, d], std::mem::size_of::<u16>());
        check_tensor_bounds("wu", &[ff, d], std::mem::size_of::<u16>());
        check_tensor_bounds("wd", &[d, ff], std::mem::size_of::<u16>());

        if let Some(hd_val) = hd {
            let hd_usize = hd_val as usize;
            check_tensor_bounds("wq", &[hd_usize, d], std::mem::size_of::<u16>());
            check_tensor_bounds("wo", &[d, hd_usize], std::mem::size_of::<u16>());
        }
        if let Some(hkv_val) = hkv_d {
            let hkv_usize = hkv_val as usize;
            check_tensor_bounds("wk", &[hkv_usize, d], std::mem::size_of::<u16>());
            check_tensor_bounds("wv", &[hkv_usize, d], std::mem::size_of::<u16>());
        }

        // Cache sizing bounds checks for max_ctx.
        let max_blocks = self.max_ctx.div_ceil(CACHE_BLOCK_TOKENS);
        match (max_blocks as usize).checked_mul(CACHE_BLOCK_TOKENS as usize) {
            Some(slots) => {
                let k_elems = slots
                    .checked_mul(self.kv_heads as usize)
                    .and_then(|val| val.checked_mul(self.head_dim as usize));
                match k_elems {
                    Some(elems) => {
                        if elems.checked_mul(std::mem::size_of::<u16>()).is_none() {
                            problems.push(ExecError::T0(
                                crate::error::T0Error::ArithmeticOverflow {
                                    op: "synthetic",
                                    detail: format!(
                                        "KV cache byte size for max_ctx {} overflows usize",
                                        self.max_ctx
                                    ),
                                },
                            ));
                        }
                    }
                    None => {
                        problems.push(ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                            op: "synthetic",
                            detail: format!(
                                "KV cache element count for max_ctx {} overflows usize",
                                self.max_ctx
                            ),
                        }));
                    }
                }
            }
            None => {
                problems.push(ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                    op: "synthetic",
                    detail: format!(
                        "max_blocks * CACHE_BLOCK_TOKENS for max_ctx {} overflows usize",
                        self.max_ctx
                    ),
                }));
            }
        }

        ExecError::from_problems(problems)
    }
}

/// A built tiny model: step graph, weights, and per-step edge handles (Spec 1 §3.1, Spec 4 §2, Spec 8 §3, Card A1.12).
pub struct TinyModel {
    /// Model description.
    pub spec: SyntheticSpec,
    /// Symbolic single-group step graph (Spec 1 §3.1).
    pub graph: Graph,
    /// `(edge, buffer)` weight and parameter bindings.
    pub weights: Vec<(EdgeId, TypedBuffer)>,
    /// Per-step token-ids edge (`[T]` u32).
    pub token_edge: EdgeId,
    /// Per-step rope-positions edge (`[T]` u32, scalar projection).
    pub positions_edge: EdgeId,
    /// Logits edge (`[T, V]` f16).
    pub logits_edge: EdgeId,
    /// Per-layer paged-KV handles (layer index = position).
    pub handles: Vec<StateHandle>,
}

/// Builds the tiny model: graph, seeded weights, and edge handles (Spec 1 §3.1, Spec 4 §2, Spec 8 §3, Card A1.12).
///
/// Weights draw from the A1.10 harness stream (`"a1.12" | name | seed`);
/// the same spec always builds bit-identical weights.
pub fn build(spec: &SyntheticSpec) -> Result<TinyModel, ExecError> {
    spec.validate()?;
    let key = StepGraphKey::from_unbucketed(r9v_ir::graph::PlanId::new(0xA112), 0, 1, 1, 0, 0)?;
    let mut graph = Graph::new(key);
    let mut weights: Vec<(EdgeId, TypedBuffer)> = Vec::new();
    let mut handles = Vec::new();

    let token_edge = graph.add_external_input(
        ExternalInputKind::TokenIds,
        act_tensor(vec![Dim::Symbolic(ShapeSymbol::T)], DType::U32)?,
    )?;
    graph.add_batch_meta_input()?;
    let positions_edge = graph.bind_positions(PositionsKind::Scalar)?;

    let embed_edge = add_weight(
        &mut graph,
        &mut weights,
        "embed",
        spec.seed,
        vec![spec.vocab as usize, spec.dim as usize],
    )?;
    let x = add_node(
        &mut graph,
        Op::EmbedGather(EmbedGatherOp {
            scale: 1.0,
            out_dtype: DType::F16,
        }),
        &[token_edge, embed_edge],
        vec![act_tensor(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(spec.dim)],
            DType::F16,
        )?],
    )?
    .first()
    .copied();

    let mut x = x.ok_or_else(|| {
        ExecError::T0(crate::error::T0Error::InvalidAttribute {
            op: "synthetic",
            attribute: "graph",
            reason: "embed_gather produced no output edge".to_string(),
        })
    })?;

    for layer in 0..spec.layers {
        x = build_layer(&mut graph, &mut weights, spec, layer, x)?;
        handles.push(StateHandle::new(layer, StateKind::KvPaged));
    }

    let norm_w = add_param(
        &mut graph,
        &mut weights,
        "final_norm",
        spec.seed,
        vec![spec.dim as usize],
    )?;
    x = sole_output(
        &mut graph,
        Op::Norm(NormOp {
            kind: NormKind::Rms,
            eps: 1e-5,
            axis: NormAxis::Last,
            weight_offset: 0.0,
            out_dtype: DType::F16,
        }),
        &[x, norm_w],
        vec![act_tensor(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(spec.dim)],
            DType::F16,
        )?],
    )?;

    let head_w = add_weight(
        &mut graph,
        &mut weights,
        "lm_head",
        spec.seed,
        vec![spec.vocab as usize, spec.dim as usize],
    )?;
    let logits_edge = sole_output(
        &mut graph,
        Op::Matmul(MatmulOp {
            out_dtype: DType::F16,
            epilogue: Epilogue::None,
            transpose_w: false,
        }),
        &[x, head_w],
        vec![act_tensor(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(spec.vocab)],
            DType::F16,
        )?],
    )?;
    // The logits edge is read directly from the executor store; it is not
    // registered as an external output because that contract is rank-3 F32
    // (scheduler logprobs path) while the step graph carries [T, V] F16.
    graph.validate()?;

    Ok(TinyModel {
        spec: *spec,
        graph,
        weights,
        token_edge,
        positions_edge,
        logits_edge,
        handles,
    })
}

/// One decoder layer: rms-norm → GQA (rope, KV write, attention, proj) →
/// residual → rms-norm → SwiGLU → residual (Spec 8 §3.1 order).
fn build_layer(
    graph: &mut Graph,
    weights: &mut Vec<(EdgeId, TypedBuffer)>,
    spec: &SyntheticSpec,
    layer: u32,
    x: EdgeId,
) -> Result<EdgeId, ExecError> {
    let t = Dim::Symbolic(ShapeSymbol::T);
    let hd = spec.heads.checked_mul(spec.head_dim).ok_or_else(|| {
        ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
            op: "synthetic",
            detail: format!(
                "heads ({}) * head_dim ({}) overflows u32",
                spec.heads, spec.head_dim
            ),
        })
    })?;
    let hkv_d = spec.kv_heads.checked_mul(spec.head_dim).ok_or_else(|| {
        ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
            op: "synthetic",
            detail: format!(
                "kv_heads ({}) * head_dim ({}) overflows u32",
                spec.kv_heads, spec.head_dim
            ),
        })
    })?;
    let tag = format!("l{layer}");

    let norm_w = add_param(
        graph,
        weights,
        &format!("{tag}_attn_norm"),
        spec.seed,
        vec![spec.dim as usize],
    )?;
    let xn = sole_output(
        graph,
        Op::Norm(norm_rms()),
        &[x, norm_w],
        vec![act_tensor(vec![t, Dim::Concrete(spec.dim)], DType::F16)?],
    )?;

    // Q/K/V projections ([T, H*D] flat, reshaped to rank 3 for rope/attention).
    let wq = add_weight(
        graph,
        weights,
        &format!("{tag}_wq"),
        spec.seed,
        vec![hd as usize, spec.dim as usize],
    )?;
    let wk = add_weight(
        graph,
        weights,
        &format!("{tag}_wk"),
        spec.seed,
        vec![hkv_d as usize, spec.dim as usize],
    )?;
    let wv = add_weight(
        graph,
        weights,
        &format!("{tag}_wv"),
        spec.seed,
        vec![hkv_d as usize, spec.dim as usize],
    )?;
    let q_flat = sole_output(
        graph,
        Op::Matmul(matmul_none()),
        &[xn, wq],
        vec![act_tensor(vec![t, Dim::Concrete(hd)], DType::F16)?],
    )?;
    let k_flat = sole_output(
        graph,
        Op::Matmul(matmul_none()),
        &[xn, wk],
        vec![act_tensor(vec![t, Dim::Concrete(hkv_d)], DType::F16)?],
    )?;
    let v_flat = sole_output(
        graph,
        Op::Matmul(matmul_none()),
        &[xn, wv],
        vec![act_tensor(vec![t, Dim::Concrete(hkv_d)], DType::F16)?],
    )?;
    let pos = graph
        .positions_binding()
        .map(|(_, edge)| edge)
        .ok_or_else(|| {
            ExecError::T0(crate::error::T0Error::InvalidAttribute {
                op: "synthetic",
                attribute: "positions",
                reason: "positions edge was not bound".to_string(),
            })
        })?;
    let q = reshape(
        graph,
        q_flat,
        vec![t, Dim::Concrete(spec.heads), Dim::Concrete(spec.head_dim)],
    )?;
    let k = reshape(
        graph,
        k_flat,
        vec![
            t,
            Dim::Concrete(spec.kv_heads),
            Dim::Concrete(spec.head_dim),
        ],
    )?;
    let v = reshape(
        graph,
        v_flat,
        vec![
            t,
            Dim::Concrete(spec.kv_heads),
            Dim::Concrete(spec.head_dim),
        ],
    )?;
    let qr = sole_output(
        graph,
        Op::Rope(rope(spec)),
        &[q, pos],
        vec![act_tensor(
            vec![t, Dim::Concrete(spec.heads), Dim::Concrete(spec.head_dim)],
            DType::F16,
        )?],
    )?;
    let kr = sole_output(
        graph,
        Op::Rope(rope(spec)),
        &[k, pos],
        vec![act_tensor(
            vec![
                t,
                Dim::Concrete(spec.kv_heads),
                Dim::Concrete(spec.head_dim),
            ],
            DType::F16,
        )?],
    )?;

    let handle = StateHandle::new(layer, StateKind::KvPaged);
    add_node(
        graph,
        Op::StateWriteKv(StateWriteKvOp {
            cache_dtype: DType::F16,
            scale_granularity: CacheScaleGranularity::PerTokenHead,
            latent: None,
            handle,
        }),
        &[kr, v],
        vec![],
    )?;
    let o = sole_output(
        graph,
        Op::Attention(AttentionOp {
            softmax_scale: 1.0 / (spec.head_dim as f32).sqrt(),
            mask: AttentionMask::Causal,
            sinks: 0,
            logit_softcap: None,
            mla: None,
            out_dtype: DType::F16,
            handle,
        }),
        &[qr],
        vec![act_tensor(
            vec![t, Dim::Concrete(spec.heads), Dim::Concrete(spec.head_dim)],
            DType::F16,
        )?],
    )?;
    let o_flat = reshape(graph, o, vec![t, Dim::Concrete(hd)])?;
    let wo = add_weight(
        graph,
        weights,
        &format!("{tag}_wo"),
        spec.seed,
        vec![spec.dim as usize, hd as usize],
    )?;
    let proj = sole_output(
        graph,
        Op::Matmul(matmul_none()),
        &[o_flat, wo],
        vec![act_tensor(vec![t, Dim::Concrete(spec.dim)], DType::F16)?],
    )?;
    let x = sole_output(
        graph,
        Op::ResidualAdd(ResidualAddOp {
            out_dtype: DType::F16,
            scale: 1.0,
        }),
        &[x, proj],
        vec![act_tensor(vec![t, Dim::Concrete(spec.dim)], DType::F16)?],
    )?;

    let ffn_norm_w = add_param(
        graph,
        weights,
        &format!("{tag}_ffn_norm"),
        spec.seed,
        vec![spec.dim as usize],
    )?;
    let xn2 = sole_output(
        graph,
        Op::Norm(norm_rms()),
        &[x, ffn_norm_w],
        vec![act_tensor(vec![t, Dim::Concrete(spec.dim)], DType::F16)?],
    )?;
    let wg = add_weight(
        graph,
        weights,
        &format!("{tag}_wg"),
        spec.seed,
        vec![spec.ff as usize, spec.dim as usize],
    )?;
    let wu = add_weight(
        graph,
        weights,
        &format!("{tag}_wu"),
        spec.seed,
        vec![spec.ff as usize, spec.dim as usize],
    )?;
    let g = sole_output(
        graph,
        Op::Matmul(matmul_none()),
        &[xn2, wg],
        vec![act_tensor(vec![t, Dim::Concrete(spec.ff)], DType::F16)?],
    )?;
    let u = sole_output(
        graph,
        Op::Matmul(matmul_none()),
        &[xn2, wu],
        vec![act_tensor(vec![t, Dim::Concrete(spec.ff)], DType::F16)?],
    )?;
    let h = sole_output(
        graph,
        Op::ActMul(ActMulOp {
            act: ActivationKind::Silu,
            clamp: None,
        }),
        &[g, u],
        vec![act_tensor(vec![t, Dim::Concrete(spec.ff)], DType::F16)?],
    )?;
    let wd = add_weight(
        graph,
        weights,
        &format!("{tag}_wd"),
        spec.seed,
        vec![spec.dim as usize, spec.ff as usize],
    )?;
    let down = sole_output(
        graph,
        Op::Matmul(matmul_none()),
        &[h, wd],
        vec![act_tensor(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(spec.dim)],
            DType::F16,
        )?],
    )?;
    sole_output(
        graph,
        Op::ResidualAdd(ResidualAddOp {
            out_dtype: DType::F16,
            scale: 1.0,
        }),
        &[x, down],
        vec![act_tensor(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(spec.dim)],
            DType::F16,
        )?],
    )
}

/// RMS norm with the synthetic defaults (`eps 1e-5`, no weight offset).
fn norm_rms() -> NormOp {
    NormOp {
        kind: NormKind::Rms,
        eps: 1e-5,
        axis: NormAxis::Last,
        weight_offset: 0.0,
        out_dtype: DType::F16,
    }
}

/// Plain matmul (`None` epilogue, standard `[N, K]` layout, f16 out).
fn matmul_none() -> MatmulOp {
    MatmulOp {
        out_dtype: DType::F16,
        epilogue: Epilogue::None,
        transpose_w: false,
    }
}

/// Plain RoPE from the model spec (Neox, no scaling).
fn rope(spec: &SyntheticSpec) -> RopeOp {
    RopeOp {
        rot_dim: spec.head_dim,
        theta: spec.theta,
        style: RopeStyle::Neox,
        scaling: RopeScaling::None,
        mrope_sections: None,
        out_dtype: DType::F16,
    }
}

/// Activation tensor descriptor (device rank 0, replicated, contiguous).
fn act_tensor(shape: Vec<Dim>, dtype: DType) -> Result<r9v_ir::Tensor, ExecError> {
    Ok(r9v_ir::Tensor::new(
        shape,
        dtype,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Activation,
    )?)
}

/// Adds one op node, returning its output edges in order.
fn add_node(
    graph: &mut Graph,
    op: Op,
    inputs: &[EdgeId],
    outputs: Vec<r9v_ir::Tensor>,
) -> Result<Vec<EdgeId>, ExecError> {
    let id = graph.add_op(op, inputs, &outputs)?;
    Ok(graph.nodes()[id.0].outputs.clone())
}

/// Adds one op node with exactly one output, returning its edge.
fn sole_output(
    graph: &mut Graph,
    op: Op,
    inputs: &[EdgeId],
    outputs: Vec<r9v_ir::Tensor>,
) -> Result<EdgeId, ExecError> {
    let edges = add_node(graph, op, inputs, outputs)?;
    edges.first().copied().ok_or_else(|| {
        ExecError::T0(crate::error::T0Error::InvalidAttribute {
            op: "synthetic",
            attribute: "graph",
            reason: "op produced no output edge".to_string(),
        })
    })
}

/// Metadata-only contiguous reshape (same bytes, new shape).
fn reshape(graph: &mut Graph, edge: EdgeId, shape: Vec<Dim>) -> Result<EdgeId, ExecError> {
    Ok(graph.reshape_edge(edge, shape)?)
}

/// Adds a seeded F16 weight matrix and returns its edge.
///
/// Tables are byte-backed LE (the T0 table convention: `embed_gather` and
/// quantized GEMM paths read raw bytes, not typed slices).
fn add_weight(
    graph: &mut Graph,
    weights: &mut Vec<(EdgeId, TypedBuffer)>,
    name: &str,
    seed: u64,
    shape: Vec<usize>,
) -> Result<EdgeId, ExecError> {
    let len: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                op: "synthetic",
                detail: format!("weight `{name}` shape {shape:?} element count overflows usize"),
            })
        })?;
    let byte_len = len.checked_mul(std::mem::size_of::<u16>()).ok_or_else(|| {
        ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
            op: "synthetic",
            detail: format!("weight `{name}` shape {shape:?} byte size overflows usize"),
        })
    })?;
    let mut rng = rng_for("a1.12-synthetic", weight_counter(weights.len(), name), seed);
    let values = uniform_f32(&mut rng, len, -1.0, 1.0);
    let mut bytes = Vec::with_capacity(byte_len);
    for value in values {
        bytes.extend_from_slice(&f32_to_f16(value).to_le_bytes());
    }
    let edge = graph.add_tensor(weight_tensor(&shape)?)?;
    weights.push((edge, TypedBuffer::from_bytes(&shape, DType::F16, &bytes)));
    Ok(edge)
}

/// Adds a seeded F32 parameter vector (norm weights around 1.0).
fn add_param(
    graph: &mut Graph,
    weights: &mut Vec<(EdgeId, TypedBuffer)>,
    name: &str,
    seed: u64,
    shape: Vec<usize>,
) -> Result<EdgeId, ExecError> {
    let len: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                op: "synthetic",
                detail: format!("param `{name}` shape {shape:?} element count overflows usize"),
            })
        })?;
    let _byte_len = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
        ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
            op: "synthetic",
            detail: format!("param `{name}` shape {shape:?} byte size overflows usize"),
        })
    })?;
    let mut rng = rng_for("a1.12-synthetic", weight_counter(weights.len(), name), seed);
    let values = uniform_f32(&mut rng, len, 0.5, 1.5);
    let edge = graph.add_tensor(param_tensor(&shape)?)?;
    weights.push((edge, TypedBuffer::from_f32(&shape, &values)));
    Ok(edge)
}

// DECISION(A1.12): per-weight stream independence comes from the harness
// `seed_for` domain (`"a1.12-synthetic" | counter | seed`) mixed with the
// weight index and name length; rejected a single shared stream because
// correlated rows across matrices would hide shared-mode bugs (SI-59
// rationale applied to the synthetic model).
/// Derives an independent case index per weight from its ordinal and name.
fn weight_counter(ordinal: usize, name: &str) -> u64 {
    (ordinal as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(name.len() as u64)
}

/// F16 weight descriptor (device rank 0, replicated, contiguous).
fn weight_tensor(shape: &[usize]) -> Result<r9v_ir::Tensor, ExecError> {
    let mut dims = Vec::with_capacity(shape.len());
    for &d in shape {
        let u = u32::try_from(d).map_err(|_| {
            ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                op: "synthetic",
                detail: format!("shape dimension {d} exceeds u32::MAX"),
            })
        })?;
        dims.push(Dim::Concrete(u));
    }
    Ok(r9v_ir::Tensor::new(
        dims,
        DType::F16,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Weight,
    )?)
}

/// F32 parameter descriptor (norm weights).
fn param_tensor(shape: &[usize]) -> Result<r9v_ir::Tensor, ExecError> {
    let mut dims = Vec::with_capacity(shape.len());
    for &d in shape {
        let u = u32::try_from(d).map_err(|_| {
            ExecError::T0(crate::error::T0Error::ArithmeticOverflow {
                op: "synthetic",
                detail: format!("shape dimension {d} exceeds u32::MAX"),
            })
        })?;
        dims.push(Dim::Concrete(u));
    }
    Ok(r9v_ir::Tensor::new(
        dims,
        DType::F32,
        QuantScheme::None,
        LayoutId::CONTIGUOUS,
        Placement::Device { rank: 0 },
        ShardLayout::Replicated,
        Class::Param,
    )?)
}
