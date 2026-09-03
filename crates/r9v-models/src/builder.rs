// SPDX-License-Identifier: Apache-2.0
//! Sealed model graph builder and model graph representation (Spec 8 §2, §5, §7; card A1.3).
//!
//! A model definition consumes [`GraphBuilder`] to declare weights, state allocations,
//! tensor transformations, fusions, and exports. Sealing the builder guarantees that
//! architecture definitions remain pure functions without side-effects or device access.

use std::collections::BTreeMap;

use r9v_ir::graph::{
    EdgeId, ExternalInputKind, ExternalOutputKind, Graph as IrGraph, PlanId, StepGraphKey,
};
use r9v_ir::op::{
    ActMulOp, ActivationKind, ActivationOp, AttentionMask, AttentionOp, CausalConv1dOp,
    ConvActivation, EmbedGatherOp, Epilogue, HashId, LinearAttnKind, LinearAttnScanOp, MatmulOp,
    MlaAttentionSpec, MlaLatent, MoeFfnOp, MoeGroup, MoeRouteOp, MoeScoring, NgramCombine,
    NgramGatherOp, NgramSource, NormAxis, NormOp, Op, ResidualAddOp, RopeOp, StateWriteKvOp,
};
use r9v_ir::state::StateHandle;
use r9v_ir::tensor::{Class, Dim, ShapeSymbol, ShardLayout, Tensor};
use r9v_ir::version::IrVersion;
use r9v_ir::{DType, LayoutId, QuantScheme};

use crate::error::ModelsError;
use crate::spec::{CacheDtype, NormSpec, PositionEncoding, RopeSpec, StateSpec};
use crate::summary::{ExpertSummary, LayerSummary, MixerKind, ModelSummary, SchemeKey};

mod sealed {
    pub trait Sealed {}
}

/// Marker trait sealing the [`GraphBuilder`] interface (Spec 8 §2; card A1.3).
// DECISION(A1.3): GraphBuilder is sealed using a private Sealed trait so external crates cannot implement it directly, preserving pure graph construction; rejected making GraphBuilder an open trait or unsealed struct. Spec 8 §1, §2.
pub trait SealedGraphBuilder: sealed::Sealed {}

/// Semantic role of a bound weight tensor (Spec 8 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightRole {
    /// General matrix multiplication weight.
    Matmul,
    /// Token embedding lookup table.
    Embed,
    /// Language model output head projection.
    LmHead,
    /// Speculative decoding n-gram hash table.
    NgramTable,
    /// 1D vector parameter (e.g. normalization weight/bias).
    Vector,
}

/// Class of quantization schemes a weight binding can consume (Spec 8 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemeClass {
    /// Matrix multiplication weights: accepts any Spec 2 scheme or unquantized.
    Matmul,
    /// Normalization or bias vectors: must be f32 unquantized.
    Vector,
    /// Embedding lookup table weights.
    Embed,
    /// Speculative n-gram table weights.
    NgramTable,
}

/// Fusion declaration over named weights (Spec 8 §2, §5).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionDecl {
    /// QKV projection fusion (Spec 8 §5: `Qkv { q, k, v }`).
    Qkv {
        /// Q weight name.
        q: String,
        /// K weight name.
        k: String,
        /// V weight name.
        v: String,
    },
    /// FFN Gate/Up projection fusion (Spec 8 §5: `GateUp { gate, up }`).
    GateUp {
        /// Gate weight name.
        gate: String,
        /// Up weight name.
        up: String,
    },
}

/// Tied embedding declaration (Spec 8 §2, §5).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TiedDecl {
    /// Source embedding weight name (e.g. `"token_embd.weight"`).
    pub embed_name: String,
    /// Destination output projection weight name (e.g. `"output.weight"`).
    pub head_name: String,
}

/// Record of a weight bound in a model definition (Spec 8 §2, §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundWeight {
    /// Tensor name following llama.cpp GGUF convention.
    pub name: String,
    /// Semantic role.
    pub role: WeightRole,
    /// Expected logical shape.
    pub shape: Vec<Dim>,
    /// Expected scheme classification.
    pub expected_scheme_class: SchemeClass,
    /// Bound tensor descriptor.
    pub tensor: Tensor,
}

/// Factory entry point for creating a [`GraphBuilder`] (Spec 8 §2).
pub struct Graph;

impl Graph {
    /// Creates a new [`GraphBuilder`] pinned to an IR version and model identifier (Spec 8 §2).
    #[allow(clippy::new_ret_no_self)]
    pub fn new(ir_version: IrVersion, model_id: impl Into<String>) -> GraphBuilder {
        GraphBuilder::new(ir_version, model_id)
    }
}

/// Sealed graph builder for model definitions (Spec 8 §2; card A1.3).
#[derive(Debug)]
pub struct GraphBuilder {
    _sealed: (),
    ir_version: IrVersion,
    model_id: String,
    graph: IrGraph,
    bound_weights: Vec<BoundWeight>,
    fusion_decls: Vec<FusionDecl>,
    tied_decls: Vec<TiedDecl>,
    state_specs: Vec<(u32, StateSpec, StateHandle)>,
    exports: Vec<(String, Tensor)>,
    subgraphs: BTreeMap<String, ModelGraph>,
    edge_map: Vec<(Tensor, EdgeId)>,
    tokens_tensor: Option<Tensor>,
}

impl sealed::Sealed for GraphBuilder {}
impl SealedGraphBuilder for GraphBuilder {}

/// Reads dimension `index` of `tensor`, returning a typed error instead of
/// panicking on a short shape (Spec 8 §2, §6).
pub(crate) fn shape_dim(
    tensor: &Tensor,
    index: usize,
    context: &'static str,
) -> Result<Dim, ModelsError> {
    tensor.shape().get(index).copied().ok_or_else(|| {
        let rank = tensor.rank();
        ModelsError::ShapeAccess {
            context: context.to_string(),
            reason: format!("rank {rank} has no dimension {index}"),
        }
    })
}

/// Extracts the single output tensor of an op emission helper (Spec 8 §2).
pub(crate) fn single_output(
    outputs: Vec<Tensor>,
    context: &'static str,
) -> Result<Tensor, ModelsError> {
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| ModelsError::ShapeAccess {
            context: context.to_string(),
            reason: "op emission produced no output tensor".to_string(),
        })
}

/// Checked `a * b` over untrusted dimensions; reports both operands (Spec 8 §6).
pub(crate) fn checked_mul(a: u32, b: u32, context: &'static str) -> Result<u32, ModelsError> {
    a.checked_mul(b)
        .ok_or_else(|| ModelsError::ArithmeticOverflow {
            context: context.to_string(),
            operation: format!("{a} * {b}"),
        })
}

/// Checked `a + b` over untrusted dimensions; reports both operands (Spec 8 §6).
pub(crate) fn checked_add(a: u32, b: u32, context: &'static str) -> Result<u32, ModelsError> {
    a.checked_add(b)
        .ok_or_else(|| ModelsError::ArithmeticOverflow {
            context: context.to_string(),
            operation: format!("{a} + {b}"),
        })
}

/// Checked `usize -> u32` narrowing for ordinals derived from `Vec` positions
/// (Spec 8 §6).
pub(crate) fn checked_u32(value: usize, context: &'static str) -> Result<u32, ModelsError> {
    u32::try_from(value).map_err(|_| ModelsError::ArithmeticOverflow {
        context: context.to_string(),
        operation: format!("usize {value} exceeds u32 range"),
    })
}

/// Checked `a * b` over byte-accounting totals; reports both operands
/// (Spec 8 §7).
pub(crate) fn checked_mul_u64(a: u64, b: u64, context: &'static str) -> Result<u64, ModelsError> {
    a.checked_mul(b)
        .ok_or_else(|| ModelsError::ArithmeticOverflow {
            context: context.to_string(),
            operation: format!("{a} * {b}"),
        })
}

/// Checked `a + b` over byte-accounting totals; reports both operands
/// (Spec 8 §7).
pub(crate) fn checked_add_u64(a: u64, b: u64, context: &'static str) -> Result<u64, ModelsError> {
    a.checked_add(b)
        .ok_or_else(|| ModelsError::ArithmeticOverflow {
            context: context.to_string(),
            operation: format!("{a} + {b}"),
        })
}

impl GraphBuilder {
    /// Creates a new sealed graph builder (Spec 8 §2).
    pub fn new(ir_version: IrVersion, model_id: impl Into<String>) -> Self {
        let key = StepGraphKey::new(PlanId::new(0), 0, 1, 1, 0, 0)
            .expect("canonical decode bucket key (S=1, T_dec=1, T_pre=0) must be valid");
        let mut graph = IrGraph::new(key);
        // Pre-register global structured metadata inputs required by IR validation (SI-12).
        let _ = graph.add_batch_meta_input();
        let _ = graph.add_sampling_params_input();
        let _ = graph.add_rng_state_input();
        let _ = graph.add_updated_rng_state_output();

        Self {
            _sealed: (),
            ir_version,
            model_id: model_id.into(),
            graph,
            bound_weights: Vec::new(),
            fusion_decls: Vec::new(),
            tied_decls: Vec::new(),
            state_specs: Vec::new(),
            exports: Vec::new(),
            subgraphs: BTreeMap::new(),
            edge_map: Vec::new(),
            tokens_tensor: None,
        }
    }

    /// Pinned IR version.
    pub fn ir_version(&self) -> IrVersion {
        self.ir_version
    }

    /// Model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    fn record_edge(&mut self, tensor: Tensor, edge_id: EdgeId) {
        self.edge_map.push((tensor, edge_id));
    }

    /// Resolves the most recent edge ID for a given tensor.
    pub fn resolve_edge(&self, tensor: &Tensor) -> Result<EdgeId, ModelsError> {
        for (t, edge_id) in self.edge_map.iter().rev() {
            if t == tensor {
                return Ok(*edge_id);
            }
        }
        Err(ModelsError::InvalidModelSpec {
            reason: format!("tensor not registered in builder DAG: {tensor:?}"),
        })
    }

    /// Registers the external token IDs input `[T] u32` (Spec 8 §2; Spec 1 §3.2).
    pub fn input_tokens(&mut self) -> Result<Tensor, ModelsError> {
        if let Some(existing) = &self.tokens_tensor {
            return Ok(existing.clone());
        }

        let shape = vec![Dim::Symbolic(ShapeSymbol::T)];
        let tensor = Tensor::new(
            shape,
            DType::U32,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let edge_id = self
            .graph
            .add_external_input(ExternalInputKind::TokenIds, tensor.clone())?;
        self.record_edge(tensor.clone(), edge_id);
        self.tokens_tensor = Some(tensor.clone());
        Ok(tensor)
    }

    /// Registers the multimodal embedding override tensors `[T, Dm] act`, `[T] bool mask` (Spec 8 §2).
    pub fn input_embed_override(&mut self, dm: u32) -> Result<(Tensor, Tensor), ModelsError> {
        let embed_tensor = Tensor::new(
            vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(dm)],
            DType::F16,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let edge_embed = self
            .graph
            .add_external_input(ExternalInputKind::EmbedOverride, embed_tensor.clone())?;
        self.record_edge(embed_tensor.clone(), edge_embed);

        let mask_tensor = Tensor::new(
            vec![Dim::Symbolic(ShapeSymbol::T)],
            DType::Bool,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let edge_mask = self
            .graph
            .add_external_input(ExternalInputKind::EmbedMask, mask_tensor.clone())?;
        self.record_edge(mask_tensor.clone(), edge_mask);

        Ok((embed_tensor, mask_tensor))
    }

    /// Returns the sequence positions tensor based on position encoding kind (Spec 8 §2).
    pub fn positions(&mut self, kind: PositionEncoding) -> Result<Tensor, ModelsError> {
        let tokens = self.input_tokens()?;
        match kind {
            PositionEncoding::Scalar => Ok(tokens),
            PositionEncoding::MRope(_) => {
                let mrope_tensor = Tensor::new(
                    vec![Dim::Symbolic(ShapeSymbol::T), Dim::Concrete(3)],
                    DType::U32,
                    QuantScheme::None,
                    LayoutId::CONTIGUOUS,
                    r9v_ir::Placement::Device { rank: 0 },
                    ShardLayout::Replicated,
                    Class::Activation,
                )?;
                let tokens_edge = self.resolve_edge(&tokens)?;
                self.record_edge(mrope_tensor.clone(), tokens_edge);
                Ok(mrope_tensor)
            }
        }
    }

    /// Binds a GGUF weight tensor by name, role, expected shape and scheme class (Spec 8 §2, §5).
    pub fn weight(
        &mut self,
        name: impl Into<String>,
        role: WeightRole,
        shape: &[Dim],
        expected: SchemeClass,
    ) -> Result<Tensor, ModelsError> {
        let name = name.into();
        let dtype = match (role, expected) {
            (WeightRole::Vector, _) | (_, SchemeClass::Vector) => DType::F32,
            _ => DType::F16,
        };
        let class = if name.ends_with(".bias") {
            Class::Param
        } else {
            Class::Weight
        };
        let tensor = Tensor::new(
            shape.to_vec(),
            dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            class,
        )?;

        let edge_id = self.graph.add_tensor(tensor.clone())?;
        self.record_edge(tensor.clone(), edge_id);

        self.bound_weights.push(BoundWeight {
            name,
            role,
            shape: shape.to_vec(),
            expected_scheme_class: expected,
            tensor: tensor.clone(),
        });

        Ok(tensor)
    }

    /// Declares per-layer state specification and returns its opaque handle (Spec 8 §2; Spec 3 §2).
    pub fn state(&mut self, layer: u32, spec: StateSpec) -> Result<StateHandle, ModelsError> {
        let handle = StateHandle::new(layer, spec.kind());
        self.state_specs.push((layer, spec, handle));
        Ok(handle)
    }

    /// Lowers an Op IR node into the step graph DAG (Spec 8 §2).
    pub fn op(
        &mut self,
        op: Op,
        inputs: &[Tensor],
        outputs: &[Tensor],
    ) -> Result<Vec<Tensor>, ModelsError> {
        let mut input_edges = Vec::with_capacity(inputs.len());
        for input in inputs {
            input_edges.push(self.resolve_edge(input)?);
        }

        let node_id = self.graph.add_op(op, &input_edges, outputs)?;
        // The node id was just returned by `add_op`, so a missing entry names
        // an internal graph invariant violation rather than untrusted input.
        let node_outputs = self
            .graph
            .nodes()
            .get(node_id.0)
            .ok_or_else(|| ModelsError::ShapeAccess {
                context: "graph.add_op".to_string(),
                reason: format!("node {} missing after successful add_op", node_id.0),
            })?
            .outputs
            .clone();

        let mut result_tensors = Vec::with_capacity(outputs.len());
        for (i, &edge_id) in node_outputs.iter().enumerate() {
            let out_tensor = outputs
                .get(i)
                .ok_or_else(|| ModelsError::ShapeAccess {
                    context: "graph.add_op".to_string(),
                    reason: format!(
                        "node {} reports {} outputs but only {} were declared",
                        node_id.0,
                        node_outputs.len(),
                        outputs.len()
                    ),
                })?
                .clone();
            self.record_edge(out_tensor.clone(), edge_id);
            result_tensors.push(out_tensor);
        }

        Ok(result_tensors)
    }

    /// Exports a tensor as a named graph output (Spec 8 §2).
    pub fn export(&mut self, name: impl Into<String>, tensor: Tensor) -> Result<(), ModelsError> {
        let name = name.into();
        let edge_id = self.resolve_edge(&tensor)?;

        if name == "logits" {
            let _ = self
                .graph
                .add_external_output(ExternalOutputKind::Logits, edge_id);
        } else if name == "hidden" {
            let _ = self
                .graph
                .add_external_output(ExternalOutputKind::Hidden, edge_id);
        }

        self.exports.push((name, tensor));
        Ok(())
    }

    /// Declares an interleave fusion over named weights (Spec 8 §2, §5).
    pub fn declare_fusion(&mut self, fusion: FusionDecl) -> Result<(), ModelsError> {
        self.fusion_decls.push(fusion);
        Ok(())
    }

    /// Declares tied embeddings between token embedding and output head (Spec 8 §2, §5).
    pub fn declare_tied(
        &mut self,
        embed_name: impl Into<String>,
        head_name: impl Into<String>,
    ) -> Result<(), ModelsError> {
        self.tied_decls.push(TiedDecl {
            embed_name: embed_name.into(),
            head_name: head_name.into(),
        });
        Ok(())
    }

    /// Spawns a child builder for a named subgraph (Spec 8 §2; e.g. MTP head, eagle head).
    pub fn subgraph(&mut self, name: &str) -> Result<GraphBuilder, ModelsError> {
        let sub_id = format!("{}.{}", self.model_id, name);
        Ok(GraphBuilder::new(self.ir_version, sub_id))
    }

    /// Adds a finished subgraph to this builder (Spec 8 §2, §5).
    pub fn add_subgraph(
        &mut self,
        name: impl Into<String>,
        sub: ModelGraph,
    ) -> Result<(), ModelsError> {
        let name = name.into();
        if self.subgraphs.contains_key(&name) {
            return Err(ModelsError::SubgraphError {
                name: name.clone(),
                reason: format!("subgraph '{name}' already registered"),
            });
        }
        self.subgraphs.insert(name, sub);
        Ok(())
    }

    /// Reshapes a tensor edge in the graph DAG to a new shape (Spec 1 §2.3).
    pub fn op_reshape(
        &mut self,
        tensor: Tensor,
        new_shape: Vec<Dim>,
    ) -> Result<Tensor, ModelsError> {
        let edge = self.resolve_edge(&tensor)?;
        let new_edge_id = self.graph.reshape_edge(edge, new_shape)?;
        let new_tensor = self
            .graph
            .edges()
            .get(new_edge_id.0)
            .ok_or_else(|| ModelsError::ShapeAccess {
                context: "graph.reshape_edge".to_string(),
                reason: format!("edge {} missing after successful reshape", new_edge_id.0),
            })?
            .tensor
            .clone();
        self.record_edge(new_tensor.clone(), new_edge_id);
        Ok(new_tensor)
    }

    /// Finalizes graph construction and returns the completed [`ModelGraph`] (Spec 8 §2).
    pub fn finish(self) -> Result<ModelGraph, ModelsError> {
        let mut graph = self.graph;
        let _ = graph.materialize_copies()?;
        graph.validate()?;

        Ok(ModelGraph {
            ir_version: self.ir_version,
            model_id: self.model_id,
            graph,
            bound_weights: self.bound_weights,
            fusion_decls: self.fusion_decls,
            tied_decls: self.tied_decls,
            state_specs: self.state_specs,
            exports: self.exports,
            subgraphs: self.subgraphs,
        })
    }

    // -------------------------------------------------------------------------
    // Typed op emission helpers
    // -------------------------------------------------------------------------

    /// Emits a normalization op (`NormOp`).
    pub fn op_norm(
        &mut self,
        x: Tensor,
        weight: Tensor,
        norm_spec: NormSpec,
        axis: NormAxis,
        out_dtype: DType,
    ) -> Result<Tensor, ModelsError> {
        let shape = x.shape().to_vec();
        let out_tensor = Tensor::new(
            shape,
            out_dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::Norm(NormOp {
            kind: norm_spec.kind,
            eps: norm_spec.eps,
            axis,
            weight_offset: norm_spec.weight_offset,
            out_dtype,
        });
        let res = self.op(op, &[x, weight], &[out_tensor])?;
        single_output(res, "op_norm")
    }

    /// Emits a general matrix multiplication (`MatmulOp`).
    pub fn op_matmul(
        &mut self,
        x: Tensor,
        w: Tensor,
        out_dtype: DType,
    ) -> Result<Tensor, ModelsError> {
        let m = shape_dim(&x, 0, "op_matmul x")?;
        let n = shape_dim(&w, 0, "op_matmul w")?;
        let out_tensor = Tensor::new(
            vec![m, n],
            out_dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::Matmul(MatmulOp {
            out_dtype,
            epilogue: Epilogue::None,
            transpose_w: false,
        });
        let res = self.op(op, &[x, w], &[out_tensor])?;
        single_output(res, "op_matmul")
    }

    /// Emits a matrix multiplication with bias epilogue (`MatmulOp`).
    pub fn op_matmul_bias(
        &mut self,
        x: Tensor,
        w: Tensor,
        bias: Tensor,
        out_dtype: DType,
    ) -> Result<Tensor, ModelsError> {
        let m = shape_dim(&x, 0, "op_matmul_bias x")?;
        let n = shape_dim(&w, 0, "op_matmul_bias w")?;
        let out_tensor = Tensor::new(
            vec![m, n],
            out_dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::Matmul(MatmulOp {
            out_dtype,
            epilogue: Epilogue::Bias,
            transpose_w: false,
        });
        let res = self.op(op, &[x, w, bias], &[out_tensor])?;
        single_output(res, "op_matmul_bias")
    }

    /// Emits an elementwise gated activation product (`ActMulOp`).
    pub fn op_act_mul(
        &mut self,
        gate: Tensor,
        up: Tensor,
        act: ActivationKind,
    ) -> Result<Tensor, ModelsError> {
        let out_tensor = Tensor::new(
            gate.shape().to_vec(),
            gate.dtype(),
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::ActMul(ActMulOp { act, clamp: None });
        let res = self.op(op, &[gate, up], &[out_tensor])?;
        single_output(res, "op_act_mul")
    }

    /// Emits an activation function pass (`ActivationOp`).
    pub fn op_activation(&mut self, x: Tensor, act: ActivationKind) -> Result<Tensor, ModelsError> {
        let out_tensor = Tensor::new(
            x.shape().to_vec(),
            x.dtype(),
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::Activation(ActivationOp { act, clamp: None });
        let res = self.op(op, &[x], &[out_tensor])?;
        single_output(res, "op_activation")
    }

    /// Emits a residual addition (`ResidualAddOp`).
    pub fn op_residual_add(
        &mut self,
        a: Tensor,
        b: Tensor,
        out_dtype: DType,
    ) -> Result<Tensor, ModelsError> {
        let out_tensor = Tensor::new(
            a.shape().to_vec(),
            out_dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::ResidualAdd(ResidualAddOp { out_dtype });
        let res = self.op(op, &[a, b], &[out_tensor])?;
        single_output(res, "op_residual_add")
    }

    /// Emits a Rotary Position Embedding op (`RopeOp`).
    pub fn op_rope(
        &mut self,
        x: Tensor,
        pos: Tensor,
        rope: &RopeSpec,
    ) -> Result<Tensor, ModelsError> {
        let out_tensor = Tensor::new(
            x.shape().to_vec(),
            x.dtype(),
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::Rope(RopeOp {
            rot_dim: rope.rot_dim,
            theta: rope.theta,
            style: rope.style,
            scaling: rope.scaling,
            mrope_sections: rope.mrope_sections,
            out_dtype: x.dtype(),
        });
        let res = self.op(op, &[x, pos], &[out_tensor])?;
        single_output(res, "op_rope")
    }

    /// Emits a KV cache state write op (`StateWriteKvOp`).
    pub fn op_state_write_kv(
        &mut self,
        k: Tensor,
        v: Tensor,
        handle: StateHandle,
        cache: CacheDtype,
        latent: Option<MlaLatent>,
    ) -> Result<(), ModelsError> {
        let cache_dtype = match cache {
            CacheDtype::E4m3 => DType::E4m3,
            CacheDtype::I8 => DType::I8,
            CacheDtype::F16 => DType::F16,
        };
        let op = Op::StateWriteKv(StateWriteKvOp {
            cache_dtype,
            scale_granularity: r9v_ir::op::CacheScaleGranularity::PerTokenHead,
            latent,
            handle,
        });
        self.op(op, &[k, v], &[])?;
        Ok(())
    }

    /// Emits a multi-head attention op (`AttentionOp`).
    ///
    /// A plain attention output repeats the query shape `[T, H, D]`; an MLA
    /// output is `[T, H, v_dim]` (Spec 1 §4.D), where `v_dim` may differ from
    /// `qk_nope_dim + qk_rope_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn op_attention(
        &mut self,
        q: Tensor,
        handle: StateHandle,
        softmax_scale: f32,
        mask: AttentionMask,
        sinks: u32,
        logit_softcap: Option<f32>,
        mla: Option<MlaAttentionSpec>,
        out_dtype: DType,
    ) -> Result<Tensor, ModelsError> {
        let out_shape = match &mla {
            Some(spec) => vec![
                shape_dim(&q, 0, "op_attention q")?,
                shape_dim(&q, 1, "op_attention q")?,
                Dim::Concrete(spec.v_dim),
            ],
            None => q.shape().to_vec(),
        };
        let out_tensor = Tensor::new(
            out_shape,
            out_dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::Attention(AttentionOp {
            softmax_scale,
            mask,
            sinks,
            logit_softcap,
            mla,
            out_dtype,
            handle,
        });
        let res = self.op(op, &[q], &[out_tensor])?;
        single_output(res, "op_attention")
    }

    /// Emits an MoE gating / router op (`MoeRouteOp`).
    #[allow(clippy::too_many_arguments)]
    pub fn op_moe_route(
        &mut self,
        logits: Tensor,
        bias: Option<Tensor>,
        top_k: u32,
        scoring: MoeScoring,
        renormalize: bool,
        group: Option<MoeGroup>,
        scale: f32,
    ) -> Result<(Tensor, Tensor), ModelsError> {
        let t = shape_dim(&logits, 0, "op_moe_route logits")?;
        let weights_tensor = Tensor::new(
            vec![t, Dim::Concrete(top_k)],
            DType::F32,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let indices_tensor = Tensor::new(
            vec![t, Dim::Concrete(top_k)],
            DType::U32,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::MoeRoute(MoeRouteOp {
            top_k,
            scoring,
            renormalize,
            group,
            scale,
        });

        let mut inputs = vec![logits];
        if let Some(b) = bias {
            inputs.push(b);
        }
        let res = self.op(op, &inputs, &[indices_tensor, weights_tensor])?;
        let indices = res
            .first()
            .ok_or_else(|| ModelsError::ShapeAccess {
                context: "op_moe_route".to_string(),
                reason: "moe_route produced no expert_ids tensor".to_string(),
            })?
            .clone();
        let weights = res
            .get(1)
            .ok_or_else(|| ModelsError::ShapeAccess {
                context: "op_moe_route".to_string(),
                reason: "moe_route produced no weights tensor".to_string(),
            })?
            .clone();
        Ok((indices, weights))
    }

    /// Emits an MoE expert execution op (`MoeFfnOp`).
    #[allow(clippy::too_many_arguments)]
    pub fn op_moe_ffn(
        &mut self,
        x: Tensor,
        expert_ids: Tensor,
        weights: Tensor,
        w_gate_up: Tensor,
        w_down: Tensor,
        act: ActivationKind,
        out_dtype: DType,
        shared_experts: u32,
    ) -> Result<Tensor, ModelsError> {
        let out_tensor = Tensor::new(
            x.shape().to_vec(),
            out_dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::MoeFfn(MoeFfnOp {
            act,
            out_dtype,
            shared_experts,
        });
        let res = self.op(
            op,
            &[x, expert_ids, weights, w_gate_up, w_down],
            &[out_tensor],
        )?;
        single_output(res, "op_moe_ffn")
    }

    /// Emits a causal 1D convolution op (`CausalConv1dOp`).
    pub fn op_causal_conv1d(
        &mut self,
        x: Tensor,
        w: Tensor,
        bias: Option<Tensor>,
        kernel: u32,
        act: ConvActivation,
        handle: StateHandle,
    ) -> Result<Tensor, ModelsError> {
        let out_tensor = Tensor::new(
            x.shape().to_vec(),
            x.dtype(),
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::CausalConv1d(CausalConv1dOp {
            kernel,
            act,
            handle,
        });
        let mut inputs = vec![x, w];
        if let Some(b) = bias {
            inputs.push(b);
        }
        let res = self.op(op, &inputs, &[out_tensor])?;
        single_output(res, "op_causal_conv1d")
    }

    /// Emits a linear attention / SSM recurrence scan op (`LinearAttnScanOp`).
    #[allow(clippy::too_many_arguments)]
    pub fn op_linear_attn_scan(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        kind: LinearAttnKind,
        chunk: u32,
        out_dtype: DType,
        handle: StateHandle,
    ) -> Result<Tensor, ModelsError> {
        let out_tensor = Tensor::new(
            v.shape().to_vec(),
            out_dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::LinearAttnScan(LinearAttnScanOp {
            kind,
            chunk,
            out_dtype,
            handle,
        });
        let res = self.op(op, &[q, k, v, alpha, beta], &[out_tensor])?;
        single_output(res, "op_linear_attn_scan")
    }

    /// Emits an n-gram speculative feature gather op (`NgramGatherOp`).
    #[allow(clippy::too_many_arguments)]
    pub fn op_ngram_gather(
        &mut self,
        source: NgramSource,
        inputs: &[Tensor],
        orders: &[u32],
        heads: u32,
        hash: HashId,
        table_sizes: &[u32],
        combine: NgramCombine,
        out_dim: u32,
        out_dtype: DType,
    ) -> Result<Tensor, ModelsError> {
        let first = inputs.first().ok_or_else(|| ModelsError::ShapeAccess {
            context: "op_ngram_gather".to_string(),
            reason: "ngram_gather requires at least one input tensor".to_string(),
        })?;
        let t = shape_dim(first, 0, "op_ngram_gather inputs")?;
        let out_tensor = Tensor::new(
            vec![t, Dim::Concrete(out_dim)],
            out_dtype,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::NgramGather(NgramGatherOp {
            source,
            orders: orders.to_vec().into_boxed_slice(),
            heads,
            hash,
            table_sizes: table_sizes.to_vec().into_boxed_slice(),
            combine,
            out_dtype,
        });
        let res = self.op(op, inputs, &[out_tensor])?;
        single_output(res, "op_ngram_gather")
    }

    /// Emits a token embedding lookup op (`EmbedGatherOp`).
    pub fn op_embed_gather(
        &mut self,
        tokens: Tensor,
        embed: Tensor,
        scale: f32,
    ) -> Result<Tensor, ModelsError> {
        let t = shape_dim(&tokens, 0, "op_embed_gather tokens")?;
        let dm = shape_dim(&embed, 1, "op_embed_gather embed")?;
        let out_tensor = Tensor::new(
            vec![t, dm],
            DType::F16,
            QuantScheme::None,
            LayoutId::CONTIGUOUS,
            r9v_ir::Placement::Device { rank: 0 },
            ShardLayout::Replicated,
            Class::Activation,
        )?;
        let op = Op::EmbedGather(EmbedGatherOp {
            scale,
            out_dtype: DType::F16,
        });
        let res = self.op(op, &[tokens, embed], &[out_tensor])?;
        single_output(res, "op_embed_gather")
    }
}

/// Representation of a built model graph DAG and structural declarations (Spec 8 §2).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelGraph {
    ir_version: IrVersion,
    model_id: String,
    graph: IrGraph,
    bound_weights: Vec<BoundWeight>,
    fusion_decls: Vec<FusionDecl>,
    tied_decls: Vec<TiedDecl>,
    state_specs: Vec<(u32, StateSpec, StateHandle)>,
    exports: Vec<(String, Tensor)>,
    subgraphs: BTreeMap<String, ModelGraph>,
}

impl ModelGraph {
    /// Pinned IR version.
    pub fn ir_version(&self) -> IrVersion {
        self.ir_version
    }

    /// Model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Reference to the underlying captured step graph DAG.
    pub fn graph(&self) -> &IrGraph {
        &self.graph
    }

    /// Slices of all bound weight declarations.
    pub fn bound_weights(&self) -> &[BoundWeight] {
        &self.bound_weights
    }

    /// Slices of all weight fusion declarations.
    pub fn fusion_decls(&self) -> &[FusionDecl] {
        &self.fusion_decls
    }

    /// Slices of all tied embedding declarations.
    pub fn tied_decls(&self) -> &[TiedDecl] {
        &self.tied_decls
    }

    /// Slices of all state specifications and handles.
    pub fn state_specs(&self) -> &[(u32, StateSpec, StateHandle)] {
        &self.state_specs
    }

    /// Slices of all exported graph output tensors.
    pub fn exports(&self) -> &[(String, Tensor)] {
        &self.exports
    }

    /// Registered nested subgraphs (e.g. MTP head, eagle head).
    pub fn subgraphs(&self) -> &BTreeMap<String, ModelGraph> {
        &self.subgraphs
    }

    /// Computes the comprehensive [`ModelSummary`] for partitioner planning (Spec 8 §7; Spec 5 §5.1).
    ///
    /// Checked: every byte total uses checked `u64` arithmetic and reports
    /// [`ModelsError::ArithmeticOverflow`] instead of clamping, wrapping, or
    /// panicking. Allocation-bound counts (layer count, expert count, `hkv`
    /// divisor enumeration) are rejected with typed validation before any
    /// allocation, so adversarial graphs fail without OOM.
    pub fn summary(&self) -> Result<ModelSummary, ModelsError> {
        const CTX: &str = "ModelGraph::summary";
        // Compute embedding weight bytes
        let mut embed_bytes = 0u64;
        let mut head_bytes = 0u64;
        let mut vocab = 0u32;
        let mut dm = 0u32;
        let mut ngram_table_bytes = 0u64;

        for w in &self.bound_weights {
            if w.role == WeightRole::Embed || w.name == "token_embd.weight" {
                let bytes = compute_tensor_bytes(&w.tensor, CTX)?;
                embed_bytes = checked_add_u64(embed_bytes, bytes, CTX)?;
                if let (Some(&Dim::Concrete(v)), Some(&Dim::Concrete(d))) =
                    (w.shape.first(), w.shape.get(1))
                {
                    vocab = v;
                    dm = d;
                }
            } else if w.role == WeightRole::LmHead || w.name == "output.weight" {
                // If tied to embedding table, storage is shared (Spec 2 §4; Spec 8 §5)
                let is_tied = self.tied_decls.iter().any(|t| t.head_name == w.name);
                if !is_tied {
                    head_bytes =
                        checked_add_u64(head_bytes, compute_tensor_bytes(&w.tensor, CTX)?, CTX)?;
                }
            } else if w.role == WeightRole::NgramTable {
                ngram_table_bytes = checked_add_u64(
                    ngram_table_bytes,
                    compute_tensor_bytes(&w.tensor, CTX)?,
                    CTX,
                )?;
            }
        }

        // Find maximum layer index across bound weights and state specs
        // (`None` when only graph-global weights such as the embedding table
        // are bound, so no phantom layer is summarized).
        let mut max_layer: Option<u32> = None;
        let mut observe_layer = |layer: u32| {
            max_layer = Some(max_layer.map_or(layer, |m| m.max(layer)));
        };
        for (layer, _, _) in &self.state_specs {
            observe_layer(*layer);
        }
        for w in &self.bound_weights {
            // MTP head-output tensors (`blk.N.mtp.output.weight`) name a
            // prediction head, not a transformer layer; counting them appends
            // phantom layers to the summary.
            if w.name.ends_with(".mtp.output.weight") {
                continue;
            }
            if let Some(layer_idx) = parse_layer_index(&w.name) {
                observe_layer(layer_idx);
            }
        }

        let num_layers = match max_layer {
            None => 0u32,
            Some(m) => m
                .checked_add(1)
                .ok_or_else(|| ModelsError::ArithmeticOverflow {
                    context: CTX.to_string(),
                    operation: format!("layer index {m} + 1"),
                })?,
        };
        // Reject absurd layer ordinals before `Vec::with_capacity` below.
        if num_layers > crate::spec::MAX_MODEL_LAYERS {
            return Err(ModelsError::InvalidModelSpec {
                reason: format!(
                    "summary layer count {num_layers} exceeds implementation limit {}",
                    crate::spec::MAX_MODEL_LAYERS
                ),
            });
        }

        let mut layers = Vec::with_capacity(num_layers as usize);
        let mut global_hkv = 0u32;
        let mut has_latent = false;

        for l in 0..num_layers {
            let mut weight_bytes_by_scheme: BTreeMap<SchemeKey, u64> = BTreeMap::new();
            let mut state_per_token_bytes = 0u64;
            let mut state_per_seq_bytes = 0u64;
            let mut experts: Option<ExpertSummary> = None;
            let mut mixer_kind = None;

            for (layer, spec, _) in &self.state_specs {
                if *layer == l {
                    state_per_token_bytes =
                        checked_add_u64(state_per_token_bytes, spec.state_per_token_bytes()?, CTX)?;
                    state_per_seq_bytes =
                        checked_add_u64(state_per_seq_bytes, spec.state_per_seq_bytes()?, CTX)?;
                    match spec {
                        StateSpec::KvPaged { hkv, .. } => {
                            if *hkv > global_hkv {
                                global_hkv = *hkv;
                            }
                            mixer_kind = Some(MixerKind::Attention);
                        }
                        StateSpec::KvLatent { .. } => {
                            has_latent = true;
                            mixer_kind = Some(MixerKind::Attention);
                        }
                        StateSpec::Recurrent { .. } | StateSpec::ConvWindow { .. } => {
                            mixer_kind = Some(MixerKind::LinearAttention);
                        }
                    }
                }
            }

            for w in &self.bound_weights {
                if parse_layer_index(&w.name) == Some(l) {
                    let bytes = compute_tensor_bytes(&w.tensor, CTX)?;
                    let scheme_key = SchemeKey::from(w.tensor.quant());
                    let entry = weight_bytes_by_scheme.entry(scheme_key).or_insert(0u64);
                    *entry = checked_add_u64(*entry, bytes, CTX)?;

                    if w.name.contains("ffn_gate_up_exps") || w.name.contains("ffn_down_exps") {
                        if let Some(&Dim::Concrete(e)) = w.shape.first() {
                            // Bound the expert count before sizing `hot_hint`
                            // below; weight shapes are untrusted input here.
                            if e > crate::spec::MAX_EXPERTS {
                                return Err(ModelsError::InvalidModelSpec {
                                    reason: format!(
                                        "summary expert count {e} exceeds implementation limit {}",
                                        crate::spec::MAX_EXPERTS
                                    ),
                                });
                            }
                            let added_bytes = if e > 0 {
                                bytes.checked_div(u64::from(e)).ok_or_else(|| {
                                    ModelsError::ArithmeticOverflow {
                                        context: CTX.to_string(),
                                        operation: format!("{bytes} / {e}"),
                                    }
                                })?
                            } else {
                                0
                            };
                            match &mut experts {
                                Some(exp) => {
                                    exp.bytes_each =
                                        checked_add_u64(exp.bytes_each, added_bytes, CTX)?;
                                }
                                None => {
                                    let hot_hint = if e > 0 {
                                        vec![1.0 / (e as f32); e as usize]
                                    } else {
                                        Vec::new()
                                    };
                                    experts = Some(ExpertSummary {
                                        e,
                                        bytes_each: added_bytes,
                                        hot_hint,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            layers.push(LayerSummary {
                weight_bytes_by_scheme,
                state_per_token_bytes,
                state_per_seq_bytes,
                experts,
                mixer_kind,
            });
        }

        // Tensor Parallel divisors of hkv (Spec 8 §7; Spec 5 §5.2)
        // DECISION(A1.3): a pure-MLA model reports hkv 1 (one latent stream per
        // layer) instead of 0; StateSpec::KvLatent carries no head count to
        // derive TP degrees from. Rejected 0 (meaningless downstream) and the
        // query-head count (not stored on the state spec). Spec 8 §7.
        let summary_hkv = if global_hkv > 0 {
            global_hkv
        } else if has_latent {
            1
        } else {
            0
        };
        // Reject absurd head counts before enumerating divisors below.
        if summary_hkv > crate::spec::MAX_KV_HEADS {
            return Err(ModelsError::InvalidModelSpec {
                reason: format!(
                    "summary hkv {summary_hkv} exceeds implementation limit {}",
                    crate::spec::MAX_KV_HEADS
                ),
            });
        }
        let tp_divisors = if summary_hkv > 0 {
            (1..=summary_hkv)
                .filter(|d| summary_hkv.is_multiple_of(*d))
                .collect()
        } else {
            vec![1]
        };

        let mtp = self.subgraphs.contains_key("mtp")
            || self.subgraphs.keys().any(|k| k.starts_with("mtp"));
        let export_hidden = self.exports.iter().any(|(name, _)| name == "hidden");

        Ok(ModelSummary {
            layers,
            embed_bytes,
            head_bytes,
            vocab,
            dm,
            hkv: summary_hkv,
            tp_divisors,
            ngram_table_bytes,
            mtp,
            export_hidden,
        })
    }
}

fn parse_layer_index(name: &str) -> Option<u32> {
    let blk = name.strip_prefix("blk.")?;
    let dot = blk.find('.')?;
    blk[..dot].parse::<u32>().ok()
}

/// Byte size of a bound tensor from shape metadata and dtype width
/// (Spec 8 §7).
///
/// Checked: dimension products that overflow `u64` report
/// [`ModelsError::ArithmeticOverflow`]. Symbolic extents (batch tokens)
/// contribute a single element each.
fn compute_tensor_bytes(tensor: &Tensor, context: &'static str) -> Result<u64, ModelsError> {
    let mut elements = 1u64;
    for &d in tensor.shape() {
        if let Dim::Concrete(c) = d {
            elements = checked_mul_u64(elements, u64::from(c), context)?;
        }
    }

    match tensor.dtype() {
        DType::F32 | DType::I32 | DType::U32 => checked_mul_u64(elements, 4, context),
        DType::F16 | DType::Bf16 => checked_mul_u64(elements, 2, context),
        DType::E4m3 | DType::E5m2 | DType::I8 | DType::Bool => Ok(elements),
        DType::I4 => Ok(elements.div_ceil(2)),
    }
}
