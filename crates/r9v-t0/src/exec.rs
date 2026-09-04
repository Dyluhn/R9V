// SPDX-License-Identifier: Apache-2.0
//! CPU graph interpreter over T0 ops (Spec 4 §2 T0 device, Card A1.12).
//!
//! [`CpuExecutor`] runs an [`r9v_ir::Graph`] step graph on the CPU device by
//! dispatching every node to the scalar T0 reference implementation for its
//! op. All 32 closed-set [`r9v_ir::Op`] variants dispatch; anything the
//! executor cannot supply (unbound edges, unregistered state, unresolvable
//! symbolic dims, missing sampling context) fails closed with a typed
//! [`ExecError`].
//!
//! Execution model (Spec 1 §3.1, Spec 4 §2):
//!
//! - Nodes run in [`r9v_ir::Graph::topological_order`]; insertion order is
//!   never assumed.
//! - Tensor inputs come from edges bound with [`CpuExecutor::bind`] (weights,
//!   external inputs) or produced by earlier nodes. Outputs are allocated
//!   from the producing edge's tensor descriptor with symbolic `S`/`T`
//!   resolved against the step [`BatchMeta`]; any other symbolic dim fails
//!   closed.
//! - Structured execution context ([`RunArgs`]) carries the one shared
//!   [`BatchMeta`], per-sequence [`SamplingParams`], mutable Philox
//!   [`RngState`]s, and the optional device-`ngram_gather` hash.
//! - Op state lives in the executor across steps: paged/latent KV caches
//!   registered per [`StateHandle`], and recurrent/conv-window double
//!   buffers auto-created per handle from op geometry (A/B swap, Spec 3 §4).
//! - The executor is deterministic: fixed topological order, ascending
//!   index reductions inside T0, no unordered iteration over anything that
//!   reaches an output. [`BTreeMap`] is used for every keyed store.

use std::collections::BTreeMap;

use r9v_ir::{
    AttentionOp, BatchMeta, DType, Dim, EdgeId, Graph, GraphNode, NgramSource, NodeId, Op,
    SampleOp, SamplingParams, ShapeSymbol, StateHandle, StateKind, StateWriteKvOp, VerifyOp,
};

/// State-map key: `(layer, kind tag)`.
///
/// `StateHandle` exposes `layer()`/`kind()` but no ordering; the tag keeps
/// the map keyed and deterministic without depending on enum discriminants.
fn state_key(handle: StateHandle) -> (u32, u8) {
    let tag = match handle.kind() {
        StateKind::KvPaged => 0,
        StateKind::KvLatent => 1,
        StateKind::Recurrent => 2,
        StateKind::ConvWindow => 3,
    };
    (handle.layer(), tag)
}

use crate::attention::{KvCache, KvLatentCache, KvPagedCache};
use crate::buffer::{TensorView, TensorViewMut, TypedBuffer};
use crate::error::T0Error;
use crate::ngram_gather::{ngram_gather_device, NgramHash};
use crate::philox::RngState;
use crate::sampling::{sample, verify};
use crate::segments::SeqLayout;

/// Typed executor failure (Spec 4 §2, CONVENTIONS.md §1.1).
///
/// Every variant carries the numbers needed to fix the problem: node and
/// edge indices, expected vs observed shapes, and the full accumulation of
/// independent failures where collection is possible.
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    /// Wrapping a T0 reference failure from the dispatched op.
    #[error(transparent)]
    T0(#[from] T0Error),

    /// Wrapping an IR failure (e.g. topological sort over a bad graph).
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),

    /// A node read an edge with no bound or produced buffer.
    #[error(
        "node {node}: input edge {edge} is not bound (bind weights and external inputs before run)"
    )]
    UnboundEdge {
        /// Index of the consuming node.
        node: usize,
        /// Missing edge id.
        edge: usize,
    },

    /// A node produced no buffer for one of its declared outputs.
    #[error("node {node}: output edge {edge} was not produced by its op")]
    MissingOutput {
        /// Index of the producing node.
        node: usize,
        /// Missing edge id.
        edge: usize,
    },

    /// An output edge carries a symbolic dim the executor cannot resolve.
    ///
    /// Only `S` and `T` resolve (against [`BatchMeta`]); model dims must be
    /// concrete after capture (Spec 1 §2.4: kernels see concrete integers).
    #[error("node {node}: output edge {edge} has unresolvable symbolic dim {symbol:?} (only S/T resolve against BatchMeta)")]
    SymbolicDim {
        /// Index of the producing node.
        node: usize,
        /// Output edge id.
        edge: usize,
        /// The unresolvable symbol.
        symbol: ShapeSymbol,
    },

    /// A stateful op named a cache or state slot that was never registered.
    #[error(
        "node {node}: op `{op}` needs state for layer {layer} ({kind:?}), but none is registered"
    )]
    UnknownState {
        /// Index of the node.
        node: usize,
        /// Op name.
        op: &'static str,
        /// State-handle layer.
        layer: u32,
        /// State kind.
        kind: StateKind,
    },

    /// A state entry exists but has the wrong backing for the op.
    #[error("node {node}: op `{op}` needs {expected} state for layer {layer} ({kind:?}), but the registered state is {got}")]
    StateKindMismatch {
        /// Index of the node.
        node: usize,
        /// Op name.
        op: &'static str,
        /// State-handle layer.
        layer: u32,
        /// State kind.
        kind: StateKind,
        /// Required backing.
        expected: &'static str,
        /// Observed backing.
        got: &'static str,
    },

    /// A multi-group batch named a layer with no group assignment.
    #[error("node {node}: batch has {g} groups but layer {layer} has no group assignment (call set_layer_group)")]
    UnknownGroup {
        /// Index of the node.
        node: usize,
        /// Layer index.
        layer: u32,
        /// Group count of the batch.
        g: u32,
    },

    /// Sampling context is missing or has the wrong length for the batch.
    #[error("node {node}: op `{op}` needs {expected} `{what}` for S={s}, got {got}")]
    SamplingContext {
        /// Index of the node.
        node: usize,
        /// Op name.
        op: &'static str,
        /// Which context item is wrong.
        what: &'static str,
        /// Required count.
        expected: usize,
        /// Observed count.
        got: usize,
        /// Batch sequence count.
        s: usize,
    },

    /// A device-mode `ngram_gather` ran with no hash carrier.
    #[error("node {node}: device ngram_gather needs an NgramHash carrier in RunArgs (staged mode needs none)")]
    MissingNgramHash {
        /// Index of the node.
        node: usize,
    },

    /// A `logits_postprocess` auxiliary input is neither history nor grammar.
    #[error("node {node}: logits_postprocess input {input} has rank {rank} dtype {dtype:?}; expected history [S,V] u32 or grammar [S,q,V] bool")]
    BadPostprocessInput {
        /// Index of the node.
        node: usize,
        /// Position of the input among the node's inputs.
        input: usize,
        /// Observed rank.
        rank: usize,
        /// Observed dtype.
        dtype: r9v_ir::DType,
    },

    /// Multiple independent per-node failures collected across one step.
    #[error("multiple executor failures ({} failures): {problems:?}", problems.len())]
    Multiple {
        /// Accumulated typed problems.
        problems: Box<[ExecError]>,
    },

    /// A metadata-only view edge (reshape/transpose) cannot be realized.
    #[error("node {node}: view edge {edge} (from source edge {source_edge}) is not a contiguous reshape and cannot be materialized")]
    NonContiguousView {
        /// Index of the consuming node.
        node: usize,
        /// The view edge id.
        edge: usize,
        /// The source edge id.
        source_edge: usize,
    },
}

impl ExecError {
    /// Aggregates typed problems per CONVENTIONS.md §1.4.
    pub fn from_problems(mut problems: Vec<ExecError>) -> Result<(), Self> {
        match problems.len() {
            0 => Ok(()),
            1 => {
                if let Some(problem) = problems.pop() {
                    Err(problem)
                } else {
                    Ok(())
                }
            }
            _ => Err(Self::Multiple {
                problems: problems.into_boxed_slice(),
            }),
        }
    }
}

/// Per-step structured execution context (Spec 1 §2.5, §3.2, §4.D.1, §4.F).
///
/// Tensor-backed externals (token ids, positions, staging) travel as bound
/// graph edges; everything here is the non-tensor side: the one shared
/// [`BatchMeta`], per-sequence sampling parameters, mutable Philox states,
/// and the optional device-`ngram_gather` hash carrier.
pub struct RunArgs<'a> {
    /// The step's batch metadata (Spec 1 §2.5).
    pub batch: &'a BatchMeta,
    /// Per-sequence sampling parameters, `[S]` (Spec 1 §4.F).
    pub params: &'a [SamplingParams],
    /// Mutable Philox states, `[S]` (Spec 1 §4.F).
    pub rng: &'a mut [RngState],
    /// Hash carrier for device-mode `ngram_gather` (Spec 1 §4.A).
    ///
    /// Staged mode never consults this; device mode fails closed without it.
    pub ngram_hash: Option<&'a dyn NgramHash>,
}

// Ping-pong recurrent state lives in two maps plus a flip bit per handle
// (Spec 3 §4 A/B swap) so the input slot borrows immutably while the
// output slot borrows mutably.

/// Shared per-step context split out of [`RunArgs`] so the mutable Philox
/// states travel as a separate borrow.
struct StepCtx<'a> {
    batch: &'a BatchMeta,
    params: &'a [SamplingParams],
    ngram_hash: Option<&'a dyn NgramHash>,
}

/// Mutable op state split out of [`CpuExecutor`] so input views borrowing
/// the edge store and state mutation coexist without aliasing.
struct State<'s> {
    caches: &'s mut BTreeMap<(u32, u8), KvCache>,
    scan_a: &'s mut BTreeMap<(u32, u8), TypedBuffer>,
    scan_b: &'s mut BTreeMap<(u32, u8), TypedBuffer>,
    scan_flip: &'s mut BTreeMap<(u32, u8), bool>,
    groups: &'s BTreeMap<u32, u32>,
}

/// CPU graph interpreter over T0 ops (Spec 4 §2, Card A1.12).
///
/// Owns edge buffers, KV caches, and scan double buffers across steps; a
/// single [`CpuExecutor::run`] executes one step graph. Weights and external
/// inputs are bound with [`CpuExecutor::bind`] before the first run and
/// rebound per step only where values change (token ids, positions).
pub struct CpuExecutor {
    store: Vec<Option<TypedBuffer>>,
    caches: BTreeMap<(u32, u8), KvCache>,
    scan_a: BTreeMap<(u32, u8), TypedBuffer>,
    scan_b: BTreeMap<(u32, u8), TypedBuffer>,
    scan_flip: BTreeMap<(u32, u8), bool>,
    scales: BTreeMap<usize, usize>,
    groups: BTreeMap<u32, u32>,
}

impl CpuExecutor {
    /// Creates an empty executor with no bindings or state (Spec 4 §2).
    pub fn new() -> Self {
        Self {
            store: Vec::new(),
            caches: BTreeMap::new(),
            scan_a: BTreeMap::new(),
            scan_b: BTreeMap::new(),
            scan_flip: BTreeMap::new(),
            scales: BTreeMap::new(),
            groups: BTreeMap::new(),
        }
    }

    /// Binds an owned buffer to an edge, replacing any previous binding (Spec 1 §3.1, Spec 4 §2).
    ///
    /// Used for weights, external inputs, and per-step values (token ids,
    /// positions). Bindings persist across [`CpuExecutor::run`] calls.
    pub fn bind(&mut self, edge: EdgeId, buffer: TypedBuffer) {
        let idx = edge.0;
        if self.store.len() <= idx {
            self.store.resize_with(idx + 1, || None);
        }
        self.store[idx] = Some(buffer);
    }

    /// Drops every edge binding. Caches, scan slots, scales, and group
    /// assignments are retained (Spec 4 §2).
    pub fn clear_edges(&mut self) {
        for slot in &mut self.store {
            *slot = None;
        }
    }

    /// Registers a paged KV cache for `handle` (Spec 1 §4.D, Spec 3 §3.3).
    pub fn register_paged_cache(&mut self, handle: StateHandle, cache: KvPagedCache) {
        self.caches.insert(state_key(handle), KvCache::Paged(cache));
    }

    /// Registers a latent MLA cache for `handle` (Spec 1 §4.D, Spec 3 §2).
    pub fn register_latent_cache(&mut self, handle: StateHandle, cache: KvLatentCache) {
        self.caches
            .insert(state_key(handle), KvCache::Latent(cache));
    }

    /// Maps a data edge to its out-of-band scale edge (Spec 1 §2.2, Spec 2 §3.4, SI-52 carrier rule).
    ///
    /// Quantized operands whose scales travel attached to views resolve
    /// through this table; operands with no entry rely on inline scales and
    /// fail closed inside T0 when scales are required but absent.
    pub fn set_scale_edge(&mut self, data: EdgeId, scale: EdgeId) {
        self.scales.insert(data.0, scale.0);
    }

    /// Assigns a layer to a batch group for multi-group batches (Spec 3 §3.5).
    ///
    /// Single-group batches (`G == 1`) need no assignment and always use
    /// group 0; multi-group batches fail closed on unassigned layers.
    pub fn set_layer_group(&mut self, layer: u32, group: u32) {
        self.groups.insert(layer, group);
    }

    /// Returns the buffer bound to or produced for `edge`, if any (Spec 4 §2).
    pub fn edge(&self, edge: EdgeId) -> Option<&TypedBuffer> {
        self.store.get(edge.0).and_then(|slot| slot.as_ref())
    }

    /// Executes one step graph in topological order (Spec 1 §3.1, Spec 4 §2).
    ///
    /// Reads bound/produced inputs, allocates each node's outputs from its
    /// edge descriptors, dispatches to T0, and stores outputs for downstream
    /// nodes. Stateful ops consult [`RunArgs`] and executor-owned state.
    pub fn run(&mut self, graph: &Graph, args: RunArgs<'_>) -> Result<(), ExecError> {
        let RunArgs {
            batch,
            params,
            rng,
            ngram_hash,
        } = args;
        let ctx = StepCtx {
            batch,
            params,
            ngram_hash,
        };
        let Self {
            store,
            scales,
            groups,
            caches,
            scan_a,
            scan_b,
            scan_flip,
        } = self;
        let mut state = State {
            caches,
            scan_a,
            scan_b,
            scan_flip,
            groups,
        };
        let order = state_ordered(graph)?;
        for node_id in order {
            let node = &graph.nodes()[node_id.0];
            let produced = run_node(
                graph, node_id.0, node, &ctx, &mut *rng, store, scales, &mut state,
            )?;
            // Publish outputs after successful execution only.
            for (idx, buffer) in produced {
                if store.len() <= idx {
                    store.resize_with(idx + 1, || None);
                }
                store[idx] = Some(buffer);
            }
        }
        Ok(())
    }
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Orders nodes for execution: tensor DAG order plus handle program order.
///
/// [`Graph::topological_order`] only sees tensor edges, but `state_write_kv`
/// has no tensor outputs, so nothing orders it before the `attention` that
/// reads its slots. For every [`StateHandle`], nodes touching it (writers,
/// readers, read-write scans) are additionally chained in graph insertion
/// order, which is program order. A tensor path contradicting a state chain
/// fails closed as a cycle.
// DECISION(A1.12): state handle program order is insertion order; rejected
// tensor-DAG-only scheduling because side-effecting writes would race
// their readers, and rejected executor heuristics (e.g. "writes first")
// because insertion order is the only program order the graph carries.
// Spec 1 §3.1, Spec 3 §4.
fn state_ordered(graph: &Graph) -> Result<Vec<NodeId>, ExecError> {
    use std::collections::VecDeque;

    let nodes = graph.nodes();
    let count = nodes.len();
    // Tensor producer map, following metadata-only views to their source
    // (mirrors `Graph::topological_order` over public fields).
    let mut producers: BTreeMap<usize, usize> = BTreeMap::new();
    for node in nodes {
        for output in &node.outputs {
            producers.insert(output.0, node.id.0);
        }
    }
    let resolve = |mut edge: usize| -> Option<usize> {
        for _ in 0..=graph.edges().len() {
            if let Some(&producer) = producers.get(&edge) {
                return Some(producer);
            }
            edge = graph.edges().get(edge)?.source_edge?.0;
        }
        None
    };
    let mut in_degree = vec![0usize; count];
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut link = |from: usize, to: usize| {
        adjacency[from].push(to);
        in_degree[to] += 1;
    };
    for node in nodes {
        for input in &node.inputs {
            if let Some(producer) = resolve(input.0) {
                link(producer, node.id.0);
            }
        }
    }
    // State chains per handle in insertion order.
    let mut touching: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    for node in nodes {
        let handle = match &node.op {
            Op::StateWriteKv(op) => Some(op.handle),
            Op::Attention(op) => Some(op.handle),
            Op::CausalConv1d(op) => Some(op.handle),
            Op::LinearAttnScan(op) => Some(op.handle),
            _ => None,
        };
        if let Some(handle) = handle {
            touching
                .entry(state_key(handle))
                .or_default()
                .push(node.id.0);
        }
    }
    // Deterministic chain insertion: BTreeMap provides ascending key order,
    // links in insertion order (nodes iterate ascending already).
    for chain in touching.values() {
        for pair in chain.windows(2) {
            link(pair[0], pair[1]);
        }
    }
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (index, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(index);
        }
    }
    let mut order = Vec::with_capacity(count);
    while let Some(node) = queue.pop_front() {
        order.push(NodeId(node));
        for &consumer in &adjacency[node] {
            in_degree[consumer] -= 1;
            if in_degree[consumer] == 0 {
                queue.push_back(consumer);
            }
        }
    }
    if order.len() != count {
        let node = in_degree.iter().position(|&degree| degree > 0).unwrap_or(0);
        return Err(ExecError::T0(T0Error::Ir(r9v_ir::IrError::GraphCycle {
            node,
        })));
    }
    Ok(order)
}

/// Returns the group index for `handle` under `batch`.
///
/// Single-group batches always use group 0; multi-group batches fail
/// closed on unassigned layers.
fn group_for(
    assigned: Option<u32>,
    node: usize,
    handle: StateHandle,
    batch: &BatchMeta,
) -> Result<u32, ExecError> {
    if batch.g() == 1 {
        return Ok(0);
    }
    assigned.ok_or(ExecError::UnknownGroup {
        node,
        layer: handle.layer(),
        g: batch.g(),
    })
}

/// Executes one node: resolves outputs, builds input views, dispatches.
///
/// Returns `(edge, buffer)` pairs for the caller to publish; nothing is
/// stored on failure.
#[allow(clippy::too_many_arguments)]
fn run_node(
    graph: &Graph,
    node_idx: usize,
    node: &GraphNode,
    ctx: &StepCtx<'_>,
    rng: &mut [RngState],
    store: &[Option<TypedBuffer>],
    scales: &BTreeMap<usize, usize>,
    state: &mut State<'_>,
) -> Result<Vec<(usize, TypedBuffer)>, ExecError> {
    let batch = ctx.batch;
    // Resolve and allocate every output up front so arity and shape
    // failures surface before any state mutation.
    let mut outputs: Vec<TypedBuffer> = Vec::with_capacity(node.outputs.len());
    for &edge in &node.outputs {
        let descriptor = &graph.edges()[edge.0].tensor;
        let shape = resolve_shape(node_idx, edge.0, descriptor, batch)?;
        let expected: usize = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                ExecError::T0(T0Error::ArithmeticOverflow {
                    op: "executor",
                    detail: format!(
                        "node {node_idx}: output edge {} shape {shape:?} element product overflows usize",
                        edge.0
                    ),
                })
            })?;
        let buffer = TypedBuffer::try_zeros(&shape, descriptor.dtype())
            .map_err(ExecError::T0)?
            .with_quant(descriptor.quant())
            .with_layout(descriptor.layout());
        // Fail closed when the descriptor's own element count disagrees
        // with the resolved shape (catches I4 half-byte miscounts early).
        if buffer.num_elements() != expected {
            return Err(ExecError::T0(T0Error::BufferLengthMismatch {
                tensor: "executor_output",
                buffer_len: buffer.num_elements(),
                expected_len: expected,
                shape,
            }));
        }
        outputs.push(buffer);
    }
    // Gather input views (plus attached scales) before dispatch.
    // DECISION(A1.12): contiguous metadata-only views (reshape) materialize
    // by copying source bytes under the resolved shape; non-contiguous
    // views (transpose) fail closed because permuting packed or quantized
    // backings needs layout-aware code the T0 device does not own.
    // Rejected aliasing views: T0 views borrow owned buffers. Spec 1 §3.3.
    let mut materialized: Vec<TypedBuffer> = Vec::new();
    let mut realized_for: Vec<Option<usize>> = vec![None; node.inputs.len()];
    for (position, &edge) in node.inputs.iter().enumerate() {
        if store.get(edge.0).and_then(|slot| slot.as_ref()).is_some() {
            continue;
        }
        let view_edge = &graph.edges()[edge.0];
        let source = view_edge.source_edge.ok_or(ExecError::UnboundEdge {
            node: node_idx,
            edge: edge.0,
        })?;
        if !view_edge.is_contiguous() {
            return Err(ExecError::NonContiguousView {
                node: node_idx,
                edge: edge.0,
                source_edge: source.0,
            });
        }
        let source_buf =
            store
                .get(source.0)
                .and_then(|slot| slot.as_ref())
                .ok_or(ExecError::UnboundEdge {
                    node: node_idx,
                    edge: source.0,
                })?;
        let resolved = resolve_shape(node_idx, edge.0, &view_edge.tensor, batch)?;
        let realized =
            source_buf
                .copy_with_shape(&resolved)
                .ok_or(ExecError::NonContiguousView {
                    node: node_idx,
                    edge: edge.0,
                    source_edge: source.0,
                })?;
        materialized.push(realized);
        realized_for[position] = Some(materialized.len() - 1);
    }
    let mut inputs: Vec<TensorView<'_>> = Vec::with_capacity(node.inputs.len());
    for (position, &edge) in node.inputs.iter().enumerate() {
        let stored =
            match realized_for[position] {
                Some(index) => &materialized[index],
                None => store.get(edge.0).and_then(|slot| slot.as_ref()).ok_or(
                    ExecError::UnboundEdge {
                        node: node_idx,
                        edge: edge.0,
                    },
                )?,
            };
        let mut view = stored.as_view();
        if let Some(&scale_edge) = scales.get(&edge.0) {
            let scale_buf = store.get(scale_edge).and_then(|slot| slot.as_ref()).ok_or(
                ExecError::UnboundEdge {
                    node: node_idx,
                    edge: scale_edge,
                },
            )?;
            view = view.with_scale(scale_buf.as_view());
        }
        inputs.push(view);
    }
    dispatch(node_idx, &node.op, &inputs, &mut outputs, ctx, rng, state)?;
    Ok(node
        .outputs
        .iter()
        .zip(outputs)
        .map(|(edge, buffer)| (edge.0, buffer))
        .collect())
}

/// Dispatches one node to its T0 implementation (Spec 1 §4, Spec 4 §2).
///
/// Exhaustive over the closed [`Op`] set: no wildcard arm, so adding an
/// op fails compilation here until its dispatch is written.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn dispatch(
    node: usize,
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TypedBuffer],
    ctx: &StepCtx<'_>,
    rng: &mut [RngState],
    state: &mut State<'_>,
) -> Result<(), ExecError> {
    match op {
        Op::Norm(_)
        | Op::ResidualAdd(_)
        | Op::ActMul(_)
        | Op::Activation(_)
        | Op::Rope(_)
        | Op::Cast(_)
        | Op::Copy(_)
        | Op::QuantAct(_)
        | Op::Split(_)
        | Op::Concat(_)
        | Op::LogitSoftcap(_) => {
            let mut views = views_mut(outputs, node, op.op_name())?;
            crate::execute_elementwise_op(op, inputs, &mut views)?;
            Ok(())
        }
        Op::Matmul(matmul_op) => {
            let mut views = views_mut(outputs, node, op.op_name())?;
            crate::execute_matmul_op(matmul_op, inputs, &mut views)?;
            Ok(())
        }
        Op::EmbedGather(_) | Op::GatherRows(_) | Op::ScatterAddRows(_) => {
            let mut views = views_mut(outputs, node, op.op_name())?;
            crate::execute_lookup_op(op, inputs, &mut views)?;
            Ok(())
        }
        Op::MoeRoute(_) | Op::MoeFfn(_) => {
            let mut views = views_mut(outputs, node, op.op_name())?;
            crate::execute_moe_op(op, inputs, &mut views)?;
            Ok(())
        }
        Op::CausalConv1d(_) | Op::LinearAttnScan(_) => {
            dispatch_scan(node, op, inputs, outputs, ctx.batch, state)
        }
        Op::NgramGather(ngram_op) => match ngram_op.source {
            NgramSource::Staged => {
                let mut views = views_mut(outputs, node, op.op_name())?;
                crate::execute_ngram_op(op, inputs, &mut views)?;
                Ok(())
            }
            NgramSource::Device => dispatch_ngram_device(node, op, inputs, outputs, ctx),
        },
        Op::StateWriteKv(write_op) => dispatch_state_write(node, write_op, inputs, ctx, state),
        Op::Attention(attn_op) => dispatch_attention(node, attn_op, inputs, outputs, ctx, state),
        Op::LogitsPostprocess(_) => dispatch_postprocess(node, op, inputs, outputs, ctx),
        Op::Sample(sample_op) => dispatch_sample(node, sample_op, inputs, outputs, ctx, rng),
        Op::Verify(verify_op) => dispatch_verify(node, verify_op, inputs, outputs, ctx, rng),
        Op::AllReduce(_)
        | Op::AllGather(_)
        | Op::ReduceScatter(_)
        | Op::AllToAll(_)
        | Op::Send(_)
        | Op::Recv(_)
        | Op::Barrier(_) => {
            let mut views = views_mut(outputs, node, op.op_name())?;
            crate::execute_collective_op(op, inputs, &mut views)?;
            Ok(())
        }
    }
}

/// Borrows every output buffer mutably for T0 dispatch.
///
/// Slice patterns bind disjoint elements, so the simultaneous borrows are
/// safe in purely safe Rust without raw pointers (rejected in this crate outside SIMD modules).
/// No closed-set op has more than two outputs; more fails closed.
fn views_mut<'a>(
    outputs: &'a mut [TypedBuffer],
    node: usize,
    op: &'static str,
) -> Result<Vec<TensorViewMut<'a>>, ExecError> {
    match outputs {
        [] => Ok(Vec::new()),
        [a] => Ok(vec![a.as_view_mut()]),
        [a, b] => Ok(vec![a.as_view_mut(), b.as_view_mut()]),
        _ => Err(ExecError::T0(T0Error::InvalidAttribute {
            op,
            attribute: "outputs",
            reason: format!(
                "node {node}: executor supports at most 2 outputs, got {}",
                outputs.len()
            ),
        })),
    }
}

/// Resolves one output tensor descriptor to a concrete shape.
fn resolve_shape(
    node: usize,
    edge: usize,
    tensor: &r9v_ir::Tensor,
    batch: &BatchMeta,
) -> Result<Vec<usize>, ExecError> {
    let mut shape = Vec::with_capacity(tensor.shape().len());
    for dim in tensor.shape() {
        let extent = match dim {
            Dim::Concrete(v) => *v as usize,
            Dim::Symbolic(ShapeSymbol::S) => batch.num_seqs(),
            Dim::Symbolic(ShapeSymbol::T) => batch.total_tokens(),
            Dim::Symbolic(symbol) => {
                return Err(ExecError::SymbolicDim {
                    node,
                    edge,
                    symbol: *symbol,
                });
            }
        };
        shape.push(extent);
    }
    Ok(shape)
}

/// Materializes a view as f32, converting each element (Spec 1 §6: the T0
/// conversion path, not a second implementation).
fn view_f32(view: &TensorView<'_>) -> Vec<f32> {
    if let Some(slice) = view.as_f32_slice() {
        return slice.to_vec();
    }
    (0..view.num_elements()).map(|i| view.read_f32(i)).collect()
}

/// Materializes a u32 view, borrowing the backing when it is u32.
fn view_u32(view: &TensorView<'_>) -> Vec<u32> {
    if let Some(slice) = view.as_u32_slice() {
        return slice.to_vec();
    }
    (0..view.num_elements())
        .map(|i| view.read_f32(i) as u32)
        .collect()
}

/// Borrows a byte-backed view (bool, fp8, packed ints); `None` otherwise.
fn view_bytes(view: &TensorView<'_>) -> Option<Vec<u8>> {
    view.as_bytes().map(|slice| slice.to_vec())
}

/// `state_write_kv`: writes K/V rows into the registered cache (Spec 1 §4.D).
fn dispatch_state_write(
    node: usize,
    op: &StateWriteKvOp,
    inputs: &[TensorView<'_>],
    ctx: &StepCtx<'_>,
    state: &mut State<'_>,
) -> Result<(), ExecError> {
    if inputs.len() != 2 {
        return Err(ExecError::T0(T0Error::InvalidAttribute {
            op: "state_write_kv",
            attribute: "inputs",
            reason: format!(
                "state_write_kv requires 2 inputs (k, v), got {}",
                inputs.len()
            ),
        }));
    }
    let group = group_for(
        state.groups.get(&op.handle.layer()).copied(),
        node,
        op.handle,
        ctx.batch,
    )?;
    let cache = state
        .caches
        .get_mut(&state_key(op.handle))
        .ok_or(ExecError::UnknownState {
            node,
            op: "state_write_kv",
            layer: op.handle.layer(),
            kind: op.handle.kind(),
        })?;
    crate::attention::state_write_kv(op, &inputs[0], &inputs[1], ctx.batch, group, cache)?;
    Ok(())
}

/// `attention`: reads the registered cache through `BatchMeta` (Spec 1 §4.D).
fn dispatch_attention(
    node: usize,
    op: &AttentionOp,
    inputs: &[TensorView<'_>],
    outputs: &mut [TypedBuffer],
    ctx: &StepCtx<'_>,
    state: &mut State<'_>,
) -> Result<(), ExecError> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(ExecError::T0(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "inputs/outputs",
            reason: format!(
                "attention requires 1 input and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            ),
        }));
    }
    let group = group_for(
        state.groups.get(&op.handle.layer()).copied(),
        node,
        op.handle,
        ctx.batch,
    )?;
    let cache = state
        .caches
        .get(&state_key(op.handle))
        .ok_or(ExecError::UnknownState {
            node,
            op: "attention",
            layer: op.handle.layer(),
            kind: op.handle.kind(),
        })?;
    let out = outputs[0].as_view_mut();
    let mut out_hold = out;
    crate::attention::attention(op, &inputs[0], ctx.batch, group, cache, &mut out_hold)?;
    Ok(())
}

/// `causal_conv1d` / `linear_attn_scan` over auto-created A/B slots (Spec 1 §4.E).
fn dispatch_scan(
    node: usize,
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TypedBuffer],
    batch: &BatchMeta,
    state: &mut State<'_>,
) -> Result<(), ExecError> {
    if outputs.len() != 1 {
        return Err(ExecError::T0(T0Error::InvalidAttribute {
            op: op.op_name(),
            attribute: "outputs",
            reason: format!("scan ops require 1 output, got {}", outputs.len()),
        }));
    }
    let s = batch.num_seqs();
    let (handle, slot_shape, slot_dtype) = scan_geometry(node, op, inputs, s)?;
    let key = state_key(handle);
    ensure_scan_slots(key, &slot_shape, slot_dtype, state)?;
    let flip = state.scan_flip.get(&key).copied().unwrap_or(false);
    // Split A/B across two maps so the input view and the output
    // view_mut borrow disjoint state (no aliasing, purely safe Rust).
    let chunked = batch.query_len().iter().any(|&q| q >= 32);
    let seq = SeqLayout::new(batch.query_len())?;
    if flip {
        let state_in = state
            .scan_b
            .get(&key)
            .expect("slot created above")
            .as_view();
        let mut state_out = state
            .scan_a
            .get_mut(&key)
            .expect("slot created above")
            .as_view_mut();
        let out = outputs[0].as_view_mut();
        let mut out_slice = [out];
        crate::execute_state_scan_op(
            op,
            inputs,
            &state_in,
            &seq,
            &mut out_slice,
            &mut state_out,
            chunked,
        )?;
    } else {
        let state_in = state
            .scan_a
            .get(&key)
            .expect("slot created above")
            .as_view();
        let mut state_out = state
            .scan_b
            .get_mut(&key)
            .expect("slot created above")
            .as_view_mut();
        let out = outputs[0].as_view_mut();
        let mut out_slice = [out];
        crate::execute_state_scan_op(
            op,
            inputs,
            &state_in,
            &seq,
            &mut out_slice,
            &mut state_out,
            chunked,
        )?;
    }
    state.scan_flip.insert(key, !flip);
    Ok(())
}

/// Creates zeroed A/B scan slots on first use; refuses geometry drift.
fn ensure_scan_slots(
    key: (u32, u8),
    shape: &[usize],
    dtype: r9v_ir::DType,
    state: &mut State<'_>,
) -> Result<(), ExecError> {
    let expected: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            ExecError::T0(T0Error::ArithmeticOverflow {
                op: "scan_state",
                detail: format!("state shape {shape:?} element product overflows usize"),
            })
        })?;
    for (name, map) in [("A", &mut state.scan_a), ("B", &mut state.scan_b)] {
        if let Some(slot) = map.get(&key) {
            if slot.shape() != shape {
                return Err(ExecError::T0(T0Error::DimensionMismatch {
                    dim_name: "scan_state",
                    expected_from: name,
                    expected,
                    tensor: "scan_state",
                    got: slot.num_elements(),
                }));
            }
        } else {
            let buffer = TypedBuffer::try_zeros(shape, dtype).map_err(ExecError::T0)?;
            map.insert(key, buffer);
        }
    }
    Ok(())
}

/// Derives a scan/conv state's slot shape, dtype, and handle from the op and inputs.
fn scan_geometry(
    node: usize,
    op: &Op,
    inputs: &[TensorView<'_>],
    s: usize,
) -> Result<(StateHandle, Vec<usize>, DType), ExecError> {
    match op {
        Op::CausalConv1d(conv_op) => {
            let x = inputs.first().ok_or_else(|| {
                ExecError::T0(T0Error::InvalidAttribute {
                    op: "causal_conv1d",
                    attribute: "inputs",
                    reason: format!("node {node}: causal_conv1d requires 2 or 3 inputs"),
                })
            })?;
            let channels = *x.shape().get(1).unwrap_or(&0);
            let window = (conv_op.kernel as usize).saturating_sub(1);
            Ok((conv_op.handle, vec![s, window, channels], DType::F16))
        }
        Op::LinearAttnScan(scan_op) => {
            let q = inputs.first().ok_or_else(|| {
                ExecError::T0(T0Error::InvalidAttribute {
                    op: "linear_attn_scan",
                    attribute: "inputs",
                    reason: format!("node {node}: linear_attn_scan requires 5 inputs"),
                })
            })?;
            let v = inputs.get(2).ok_or_else(|| {
                ExecError::T0(T0Error::InvalidAttribute {
                    op: "linear_attn_scan",
                    attribute: "inputs",
                    reason: format!("node {node}: linear_attn_scan requires 5 inputs"),
                })
            })?;
            let (h, d) = (
                *q.shape().get(1).unwrap_or(&0),
                *q.shape().get(2).unwrap_or(&0),
            );
            let dv = *v.shape().get(2).unwrap_or(&0);
            Ok((scan_op.handle, vec![s, h, d, dv], DType::F32))
        }
        _ => Err(ExecError::T0(T0Error::InvalidAttribute {
            op: op.op_name(),
            attribute: "op",
            reason: format!("node {node}: not a scan op"),
        })),
    }
}

/// Device-mode `ngram_gather` with the caller-supplied hash (Spec 1 §4.A).
fn dispatch_ngram_device(
    node: usize,
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TypedBuffer],
    ctx: &StepCtx<'_>,
) -> Result<(), ExecError> {
    let Op::NgramGather(ngram_op) = op else {
        return Err(ExecError::T0(T0Error::InvalidAttribute {
            op: op.op_name(),
            attribute: "op",
            reason: "ngram device dispatch called on a non-ngram op".to_string(),
        }));
    };
    let hash = ctx.ngram_hash.ok_or(ExecError::MissingNgramHash { node })?;
    if outputs.len() != 1 || inputs.len() < 2 || inputs.len() > 3 {
        return Err(ExecError::T0(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "inputs/outputs",
            reason: format!(
                "device ngram_gather requires 2 or 3 inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            ),
        }));
    }
    let scale = inputs.get(2);
    let out = outputs[0].as_view_mut();
    let mut out_hold = out;
    ngram_gather_device(ngram_op, &inputs[0], &inputs[1], scale, hash, &mut out_hold)?;
    Ok(())
}

/// `logits_postprocess` over `[S, q, V]` with classified auxiliaries (Spec 1 §4.F).
fn dispatch_postprocess(
    node: usize,
    op: &Op,
    inputs: &[TensorView<'_>],
    outputs: &mut [TypedBuffer],
    ctx: &StepCtx<'_>,
) -> Result<(), ExecError> {
    let s = check_sampling_len(node, op.op_name(), "params", ctx.params.len(), ctx.batch)?;
    if inputs.is_empty() || outputs.len() != 1 {
        return Err(ExecError::T0(T0Error::InvalidAttribute {
            op: "logits_postprocess",
            attribute: "inputs/outputs",
            reason: format!(
                "logits_postprocess requires 1-3 inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            ),
        }));
    }
    let logits = &inputs[0];
    if logits.rank() != 3 || logits.shape()[0] != s {
        return Err(ExecError::T0(T0Error::DimensionMismatch {
            dim_name: "S",
            expected_from: "batch",
            expected: s,
            tensor: "logits",
            got: logits.shape().first().copied().unwrap_or(0),
        }));
    }
    let (q, v) = (logits.shape()[1], logits.shape()[2]);
    let mut history: Option<Vec<u32>> = None;
    let mut grammar: Option<Vec<bool>> = None;
    for (position, aux) in inputs.iter().enumerate().skip(1) {
        match (aux.rank(), aux.dtype()) {
            (2, DType::U32) => {
                if history.is_some() {
                    return Err(duplicate_aux(node, position, "history_counts"));
                }
                history = Some(view_u32(aux));
            }
            (3, DType::Bool) => {
                if grammar.is_some() {
                    return Err(duplicate_aux(node, position, "grammar_mask"));
                }
                let bytes = view_bytes(aux).ok_or(ExecError::BadPostprocessInput {
                    node,
                    input: position,
                    rank: aux.rank(),
                    dtype: aux.dtype(),
                })?;
                grammar = Some(bytes.iter().map(|&b| b != 0).collect());
            }
            (rank, dtype) => {
                return Err(ExecError::BadPostprocessInput {
                    node,
                    input: position,
                    rank,
                    dtype,
                });
            }
        }
    }
    let logits_f32 = view_f32(logits);
    if outputs[0].dtype() != DType::F32 {
        return Err(ExecError::T0(T0Error::DTypeMismatch {
            tensor: "probs",
            expected: vec![DType::F32],
            got: outputs[0].dtype(),
        }));
    }
    let out_shape = outputs[0].shape().to_vec();
    let out = outputs[0].as_f32_slice_mut().ok_or_else(|| {
        ExecError::T0(T0Error::BufferLengthMismatch {
            tensor: "probs",
            buffer_len: 0,
            expected_len: logits_f32.len(),
            shape: out_shape.clone(),
        })
    })?;
    crate::logits_postprocess(
        &logits_f32,
        s,
        q,
        v,
        ctx.params,
        history.as_deref(),
        grammar.as_deref(),
        out,
    )?;
    Ok(())
}

/// `sample`: inverse-CDF draw per sequence with Philox states (Spec 1 §4.F).
fn dispatch_sample(
    node: usize,
    _op: &SampleOp,
    inputs: &[TensorView<'_>],
    outputs: &mut [TypedBuffer],
    ctx: &StepCtx<'_>,
    rng: &mut [RngState],
) -> Result<(), ExecError> {
    let s = check_sampling_len(node, "sample", "rng", rng.len(), ctx.batch)?;
    if inputs.len() != 1 || outputs.len() != 1 {
        return Err(ExecError::T0(T0Error::InvalidAttribute {
            op: "sample",
            attribute: "inputs/outputs",
            reason: format!(
                "sample requires 1 input and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            ),
        }));
    }
    let probs = &inputs[0];
    if probs.rank() != 2 || probs.shape()[0] != s {
        return Err(ExecError::T0(T0Error::DimensionMismatch {
            dim_name: "S",
            expected_from: "batch",
            expected: s,
            tensor: "probs",
            got: probs.shape().first().copied().unwrap_or(0),
        }));
    }
    let v = probs.shape()[1];
    let tokens = sample(&view_f32(probs), s, v, rng)?;
    replace_u32_output(node, "sample", outputs, 0, &[s], &tokens)
}

/// `verify`: speculative acceptance over drafts (Spec 1 §4.F, Spec 7 §4).
fn dispatch_verify(
    node: usize,
    op: &VerifyOp,
    inputs: &[TensorView<'_>],
    outputs: &mut [TypedBuffer],
    ctx: &StepCtx<'_>,
    rng: &mut [RngState],
) -> Result<(), ExecError> {
    let s = check_sampling_len(node, "verify", "rng", rng.len(), ctx.batch)?;
    if inputs.len() < 2 || inputs.len() > 3 || outputs.len() != 2 {
        return Err(ExecError::T0(T0Error::InvalidAttribute {
            op: "verify",
            attribute: "inputs/outputs",
            reason: format!(
                "verify requires 2 or 3 inputs and 2 outputs, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            ),
        }));
    }
    let draft = &inputs[0];
    let target = &inputs[1];
    if draft.rank() != 2 || draft.shape()[0] != s {
        return Err(ExecError::T0(T0Error::DimensionMismatch {
            dim_name: "S",
            expected_from: "batch",
            expected: s,
            tensor: "draft_tokens",
            got: draft.shape().first().copied().unwrap_or(0),
        }));
    }
    let k = draft.shape()[1];
    let draft_probs = if inputs.len() == 3 {
        Some(view_f32(&inputs[2]))
    } else {
        None
    };
    let out = verify(
        &view_u32(draft),
        draft_probs.as_deref(),
        &view_f32(target),
        s,
        k,
        target.shape().last().copied().unwrap_or(0),
        &op.method,
        rng,
        ctx.batch.tree(),
    )?;
    replace_u32_output(node, "verify", outputs, 0, &[s, k + 1], &out.accepted)?;
    replace_u32_output(node, "verify", outputs, 1, &[s], &out.accept_len)
}

/// Checks a sampling-context length against the batch (`[S]` rule, Spec 1 §4.F).
fn check_sampling_len(
    node: usize,
    op: &'static str,
    what: &'static str,
    got: usize,
    batch: &BatchMeta,
) -> Result<usize, ExecError> {
    let s = batch.num_seqs();
    if got != s {
        return Err(ExecError::SamplingContext {
            node,
            op,
            what,
            expected: s,
            got,
            s,
        });
    }
    Ok(s)
}

/// Typed refusal for a doubled `logits_postprocess` auxiliary.
fn duplicate_aux(node: usize, input: usize, what: &'static str) -> ExecError {
    ExecError::T0(T0Error::InvalidAttribute {
        op: "logits_postprocess",
        attribute: "inputs",
        reason: format!("duplicate {what} at input {input} in node {node}"),
    })
}

/// Replaces a fresh output buffer with u32 results, checking the length first
/// so [`TypedBuffer::from_u32`]'s internal assert is unreachable on bad input.
fn replace_u32_output(
    node: usize,
    op: &'static str,
    outputs: &mut [TypedBuffer],
    position: usize,
    shape: &[usize],
    values: &[u32],
) -> Result<(), ExecError> {
    let expected: usize = shape.iter().product();
    if values.len() != expected {
        return Err(ExecError::T0(T0Error::ShapeLengthMismatch {
            op,
            tensor: "output",
            expected,
            got: values.len(),
            detail: format!("node {node} output {position} length mismatch"),
        }));
    }
    if outputs[position].shape() != shape {
        return Err(ExecError::T0(T0Error::DimensionMismatch {
            dim_name: "output",
            expected_from: "graph",
            expected,
            tensor: "output",
            got: outputs[position].num_elements(),
        }));
    }
    outputs[position] = TypedBuffer::from_u32(shape, values);
    Ok(())
}
