// SPDX-License-Identifier: Apache-2.0
//! Step 2 (binding half) — resolve every required tensor (Spec 9 §2 step 2,
//! Spec 8 §6 items 1–2, Spec 1 §2.3).
//!
//! Every `weight()` call in the lowered graph — root and nested subgraphs
//! (MTP heads live in `subgraph("mtp")`, Spec 8 §5) — must resolve to a
//! checkpoint tensor with the expected logical shape and a consumable
//! scheme class. All failures are collected and reported together;
//! checkpoint tensors no `weight()` call named are returned as unused
//! warnings, never errors. A tied head whose alias is absent from the
//! checkpoint resolves to its declared source with undiminished shape and
//! scheme checks, occupying no additional storage.

use std::collections::{BTreeMap, BTreeSet};

use r9v_format::TensorType;
use r9v_ir::{Dim, Placement, PlanStrategy};
use r9v_models::{BoundWeight, ModelGraph, SchemeClass, WeightRole};

use crate::error::{LoaderError, TensorProblem, TensorProblemKind};
use crate::open::OpenedCheckpoint;

/// One resolved tensor binding (Spec 8 §2, Spec 9 §2 step 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTensor {
    /// Tensor name following llama.cpp GGUF convention.
    pub name: String,
    /// Semantic role from the model definition.
    pub role: WeightRole,
    /// Planned single-device placement (always legal; see
    /// [`intended_placement`]).
    pub placement: Placement,
    /// Expected logical shape, outer-last order.
    pub expected_shape: Vec<u64>,
    /// Actual logical shape from the container, outer-last order.
    pub actual_shape: Vec<u64>,
    /// Container tensor type name.
    pub tensor_type: String,
    /// Exact additional destination bytes. Zero for a tied alias, which
    /// shares its source's storage (Spec 8 §5, Spec 2 §4).
    pub bytes: u64,
    /// Tied source this binding shares storage with (`None` for real
    /// storage). Set exactly when `bytes` is zero via a tied declaration.
    pub aliased_to: Option<String>,
}

/// Binding result (Spec 8 §6 items 1–2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindReport {
    /// Resolved tensors in model binding order (root weights, then nested
    /// subgraphs in name order, recursively).
    pub bound: Vec<BoundTensor>,
    /// Checkpoint tensors no `weight()` call named, in merged table order
    /// (warnings, not errors).
    pub unused: Vec<String>,
}

/// Intended single-device placement for a strategy (Spec 1 §2.3, Spec 5 §5.2).
///
/// Every strategy maps to `Device(0)` here: rank 0 denotes the single
/// execution device, whether that is a GPU or the host CPU under a `Cpu`
/// plan. `Host`/`Tiered` assignments belong to multi-device planning with
/// hot/cold expert sets (Spec 5 §3.4), which `r9v-part` owns.
// DECISION(A2.6): rank 0 denotes the single execution device (GPU or CPU),
// so CPU-plan weights stay `Device(0)` and budgets charge them to host;
// rejected `Host` placements for CPU plans because Spec 1 §2.3 restricts
// `Host`/`Tiered` to expert, n-gram, and embedding weights. Multi-device
// placement is `r9v-part`'s planner, not this constructor. Spec 1 §2.3,
// Spec 5 §5.2.
pub fn intended_placement(strategy: PlanStrategy) -> Placement {
    match strategy {
        PlanStrategy::Cpu
        | PlanStrategy::Single
        | PlanStrategy::Pp
        | PlanStrategy::Tp
        | PlanStrategy::Ep
        | PlanStrategy::PpTp => Placement::Device { rank: 0 },
    }
}

/// Whether `placement` is legal for a weight of `role` that is (or is not)
/// a stacked-expert tensor (Spec 1 §2.3).
///
/// `Device` is always legal. `Host`/`Tiered` are legal only for stacked
/// experts, n-gram tables, and embeddings. Expert identity arrives as a
/// carried build-time fact ([`ModelGraph::is_stacked_expert`]), never a
/// re-parsed name segment.
pub fn placement_is_legal(role: WeightRole, is_stacked_expert: bool, placement: Placement) -> bool {
    match placement {
        Placement::Device { .. } => true,
        Placement::Host | Placement::Tiered => {
            is_stacked_expert || matches!(role, WeightRole::Embed | WeightRole::NgramTable)
        }
    }
}

/// Whether `name` is a stacked-expert tensor in `graph` or any nested
/// subgraph (Spec 8 §5).
///
/// Reads the build-time fact recorded by the generic MoE lowering
/// ([`ModelGraph::is_stacked_expert`]) at every level; the loader's budget
/// rule uses this to isolate expert bytes for the `experts.hot_set_vram`
/// suggestion (Spec 9 §4.3).
pub fn is_stacked_expert_weight(graph: &ModelGraph, name: &str) -> bool {
    if graph.is_stacked_expert(name) {
        return true;
    }
    graph
        .subgraphs()
        .values()
        .any(|sub| is_stacked_expert_weight(sub, name))
}

/// Binds every required tensor across the graph and its nested subgraphs,
/// collecting all failures (Spec 9 §2 step 2, Spec 8 §6 items 1–2).
pub fn bind(
    graph: &ModelGraph,
    ckpt: &OpenedCheckpoint,
    strategy: PlanStrategy,
) -> Result<BindReport, LoaderError> {
    // Tied heads by alias name, deterministic: first declaration wins.
    let mut tied: BTreeMap<&str, &str> = BTreeMap::new();
    collect_tied(graph, &mut tied);

    let mut problems = Vec::new();
    let mut bound = Vec::new();
    let mut seen: BTreeSet<&str> = BTreeSet::new();

    // Root weights, then nested subgraphs in name order, recursively, so
    // binding order is deterministic for every graph shape.
    let mut queue = Vec::new();
    collect_weights(graph, &mut queue);
    for weight in &queue {
        seen.insert(weight.name.as_str());
        match bind_one(weight, ckpt, strategy, tied.get(weight.name.as_str())) {
            Ok(resolved) => bound.push(resolved),
            Err(problem) => problems.push(problem),
        }
    }

    if !problems.is_empty() {
        return Err(LoaderError::Tensors { details: problems });
    }

    // Unused warnings in merged table order (shard order, then table
    // order), so split checkpoints report deterministically.
    let mut unused = Vec::new();
    let set = ckpt.shard_set();
    for i in 0..set.len() {
        if let Some((_, info)) = set.tensor_at(i) {
            if !seen.contains(info.name.as_str()) {
                unused.push(info.name.clone());
            }
        }
    }

    Ok(BindReport { bound, unused })
}

/// Collects weights: root first, then subgraphs in name order,
/// recursively.
fn collect_weights<'a>(graph: &'a ModelGraph, out: &mut Vec<&'a BoundWeight>) {
    for weight in graph.bound_weights() {
        out.push(weight);
    }
    for sub in graph.subgraphs().values() {
        collect_weights(sub, out);
    }
}

/// Collects tied declarations across the graph and its subgraphs.
fn collect_tied<'a>(graph: &'a ModelGraph, out: &mut BTreeMap<&'a str, &'a str>) {
    for decl in graph.tied_decls() {
        out.entry(decl.head_name.as_str())
            .or_insert(decl.embed_name.as_str());
    }
    for sub in graph.subgraphs().values() {
        collect_tied(sub, out);
    }
}

/// Resolves one bound weight against the merged checkpoint tables.
fn bind_one(
    weight: &BoundWeight,
    ckpt: &OpenedCheckpoint,
    strategy: PlanStrategy,
    tied_source: Option<&&str>,
) -> Result<BoundTensor, TensorProblem> {
    let missing = |kind: TensorProblemKind| TensorProblem {
        name: weight.name.clone(),
        kind,
    };

    match ckpt.tensor(&weight.name) {
        Some((shard_index, info)) => {
            let expected_shape = expected_shape_of(weight).map_err(&missing)?;
            let actual_shape = info.shape();
            if expected_shape != actual_shape {
                return Err(missing(TensorProblemKind::ShapeMismatch {
                    expected: expected_shape,
                    actual: actual_shape,
                }));
            }
            check_scheme(weight, info.dtype).map_err(&missing)?;
            let bytes = ckpt.tensor_nbytes(shard_index, info).map_err(|e| {
                missing(TensorProblemKind::Unmeasurable {
                    reason: e.to_string(),
                })
            })?;
            Ok(BoundTensor {
                name: weight.name.clone(),
                role: weight.role,
                // DECISION(A2.6): no per-tensor placement check here:
                // `intended_placement` yields `Device` by construction and
                // `Device` is always legal, so a check could never fire.
                // The Spec 1 §2.3 rule itself lives in
                // `placement_is_legal`, covered by its own tests, for the
                // multi-device planner that assigns `Host`/`Tiered`.
                placement: intended_placement(strategy),
                expected_shape,
                actual_shape,
                tensor_type: info.dtype.name(),
                bytes,
                aliased_to: None,
            })
        }
        None => {
            // Spec 8 §5: an absent tied alias resolves to its declared
            // source with undiminished checks and shared storage.
            let source = tied_source
                .copied()
                .ok_or_else(|| missing(TensorProblemKind::Missing))?;
            let (source_shard, source_info) = ckpt
                .tensor(source)
                .ok_or_else(|| missing(TensorProblemKind::Missing))?;
            let expected_shape = expected_shape_of(weight).map_err(&missing)?;
            let actual_shape = source_info.shape();
            if expected_shape != actual_shape {
                return Err(missing(TensorProblemKind::ShapeMismatch {
                    expected: expected_shape,
                    actual: actual_shape,
                }));
            }
            check_scheme(weight, source_info.dtype).map_err(&missing)?;
            let _ = source_shard;
            Ok(BoundTensor {
                name: weight.name.clone(),
                role: weight.role,
                placement: intended_placement(strategy),
                expected_shape,
                actual_shape,
                tensor_type: source_info.dtype.name(),
                // Shared storage budgets once: the alias occupies no
                // additional destination bytes (Spec 2 §4).
                bytes: 0,
                aliased_to: Some(source.to_string()),
            })
        }
    }
}

/// Expected logical shape of a bound weight (outer-last order).
fn expected_shape_of(weight: &BoundWeight) -> Result<Vec<u64>, TensorProblemKind> {
    let mut expected_shape = Vec::with_capacity(weight.shape.len());
    for dim in &weight.shape {
        match dim {
            Dim::Concrete(extent) => expected_shape.push(u64::from(*extent)),
            Dim::Symbolic(symbol) => {
                return Err(TensorProblemKind::Unmeasurable {
                    reason: format!(
                        "expected shape of '{}' contains unresolved symbol {symbol:?}",
                        weight.name,
                    ),
                });
            }
        }
    }
    Ok(expected_shape)
}

/// Scheme-class consumability of a container type (Spec 8 §2).
fn check_scheme(weight: &BoundWeight, dtype: TensorType) -> Result<(), TensorProblemKind> {
    if scheme_is_consumable(weight.expected_scheme_class, dtype) {
        Ok(())
    } else {
        Err(TensorProblemKind::Scheme {
            expected_class: scheme_class_name(weight.expected_scheme_class).to_string(),
            actual: dtype.name(),
        })
    }
}

/// Whether a container type is consumable under a scheme class (Spec 8 §2:
/// all matmul weights accept any spec 2 scheme; vectors must be f32).
fn scheme_is_consumable(expected: SchemeClass, dtype: TensorType) -> bool {
    match expected {
        SchemeClass::Vector => dtype == TensorType::F32,
        // DECISION(A2.6): embeddings accept unquantized types only, n-gram
        // tables accept any known type (tables may be quantized i4/i8 per
        // Spec 1 §4.A); rejected wider embedding acceptance because no spec
        // admits quantized embeddings. Spec 8 §2 names only the matmul and
        // vector cases.
        SchemeClass::Embed => is_unquantized(dtype),
        SchemeClass::NgramTable => is_unquantized(dtype) || dtype.scheme().is_some(),
        SchemeClass::Matmul => is_unquantized(dtype) || dtype.scheme().is_some(),
    }
}

/// Whether a container type is an unquantized element type.
fn is_unquantized(dtype: TensorType) -> bool {
    matches!(dtype, TensorType::F32 | TensorType::F16 | TensorType::BF16)
}

/// Stable scheme-class name for reports.
fn scheme_class_name(class: SchemeClass) -> &'static str {
    match class {
        SchemeClass::Matmul => "matmul",
        SchemeClass::Vector => "vector",
        SchemeClass::Embed => "embed",
        SchemeClass::NgramTable => "ngram_table",
    }
}
