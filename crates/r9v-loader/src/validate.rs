// SPDX-License-Identifier: Apache-2.0
//! Step 2 — family resolution and model validation (Spec 9 §2 step 2,
//! Spec 8 §6).
//!
//! Resolves the model family from `general.architecture` (Spec 8 §4),
//! downgrades a wholly-absent optional MTP head (Spec 8 §5), builds the
//! [`ModelSpec`], lowers the [`ModelGraph`], and checks the file-level
//! structural constraints of Spec 8 §6 against the merged checkpoint
//! tables. Independent failures — file-level validation, tensor binding,
//! and fusion declarations — are all collected and reported together
//! (`CONVENTIONS.md` §1.4); checkpoint tensors the model never names are
//! reported as unused warnings, never errors (Spec 8 §6 item 2).

use r9v_format::{Interleave, KvValue};
use r9v_ir::IrVersion;
use r9v_models::{
    build_from_meta, build_model, FusionDecl, GgufMeta, Graph, ModelGraph, ModelSpec, ModelSummary,
};

use crate::bind::bind;
use crate::error::{LoaderError, TensorProblem, TensorProblemKind};
use crate::open::OpenedCheckpoint;

/// Resolved and validated model (Spec 9 §2 step 2 output).
#[derive(Debug)]
pub struct ValidatedModel {
    /// Model specification built from checkpoint metadata (Spec 8 §4–§5).
    pub spec: ModelSpec,
    /// Lowered model graph with bound weights and state specs (Spec 8 §2).
    pub graph: ModelGraph,
    /// Planner summary computed from bound weights and state specs
    /// (Spec 8 §7).
    pub summary: ModelSummary,
    /// Tensor bindings with unused-tensor warnings (Spec 8 §6).
    pub bind: crate::bind::BindReport,
}

/// Resolves the family, builds and validates the model, binds every
/// required tensor, and checks file-level structural constraints,
/// collecting every independent failure (Spec 9 §2 step 2, Spec 8 §6).
///
/// `model_id` names the lowered graph; pass the checkpoint path or
/// `general.name`. Unknown architectures fail naming the architecture and
/// its nearest family (Spec 9 §12). `strategy` records the intended
/// placement on each binding.
pub fn resolve_and_validate(
    meta: &(impl GgufMeta + ?Sized),
    ckpt: &OpenedCheckpoint,
    model_id: &str,
    strategy: r9v_ir::PlanStrategy,
) -> Result<ValidatedModel, LoaderError> {
    // Spec 8 §4: unknown `general.architecture` errors naming it and the
    // nearest family; `families::build` already reports both. Without a
    // spec nothing else is computable, so this returns immediately: it
    // hides no independent check, since every later check needs the spec.
    let spec = build_from_meta(meta)?;
    // The spec's own validity gates graph construction (`build_model`
    // re-validates identically), so an invalid spec also returns here as
    // the typed models error rather than an aggregated string.
    spec.validate()?;

    let (spec, graph) = downgrade_absent_mtp(spec, ckpt, model_id)?;
    let summary = graph.summary()?;

    // Spec 8 §6 item 4: the spec is checked against the IR version it pins.
    if graph.ir_version() != IrVersion::CURRENT {
        return Err(LoaderError::IrVersionMismatch {
            pinned: graph.ir_version(),
            current: IrVersion::CURRENT,
        });
    }

    // K-family granularity (`K % 256`, Spec 8 §6 item 3) is enforced by
    // container table validation during open (card A2.5 block-divisibility
    // checks run over every tensor before binding); `d % 16` and `dff_e`
    // consistency are enforced by `ModelSpec::validate` above. What remains
    // loader-owned runs below to completion in every class — file-level
    // validation, tensor binding, and fusion declarations are independent,
    // so a missing tensor never hides a vocab or fusion problem behind it.
    let mut validation_problems = Vec::new();
    check_vocab_tokenizer_match(&spec, ckpt, &mut validation_problems);

    let mut tensor_problems: Vec<TensorProblem> = check_fusion_hits(&graph, ckpt)
        .into_iter()
        .map(|hit| TensorProblem {
            name: hit.name,
            kind: TensorProblemKind::Fusion { detail: hit.detail },
        })
        .collect();
    let bind_report = match bind(&graph, ckpt, strategy) {
        Ok(report) => Some(report),
        Err(LoaderError::Tensors { details }) => {
            tensor_problems.extend(details);
            None
        }
        Err(e) => return Err(e),
    };

    // Every class ran; report the union with no class hiding another.
    tensor_problems.sort_by(|a, b| a.name.cmp(&b.name));
    match (tensor_problems.is_empty(), validation_problems.is_empty()) {
        (true, true) => {
            // Internal invariant: an empty tensor list means `bind`
            // succeeded, so the report is present.
            let report = bind_report.expect("empty tensor problems imply a bind report");
            Ok(ValidatedModel {
                spec,
                graph,
                summary,
                bind: report,
            })
        }
        (true, false) => Err(LoaderError::Validation {
            problems: validation_problems,
        }),
        (false, true) => Err(LoaderError::Tensors {
            details: tensor_problems,
        }),
        (false, false) => Err(LoaderError::Step2 {
            details: tensor_problems,
            problems: validation_problems,
        }),
    }
}

/// Downgrades a wholly-absent optional MTP head (Spec 8 §5).
///
/// The `mtp` subgraph's weights bind inside `subgraph("mtp")`. When the
/// merged checkpoint holds none of them, the head is absent: the spec is
/// rebuilt with `mtp = None` and the graph re-lowered, so planning,
/// budgets, and the load report proceed without the head and proposer
/// resolution skips it (Spec 7 §6). When at least one MTP tensor is
/// present, the head is required and kept whole: binding later lists every
/// missing member exactly, never silently ignoring a partially-present
/// head. Only the subgraph named exactly `mtp` downgrades; any other
/// subgraph's weights are unconditionally required.
// DECISION(A2.6): "absent" means zero `mtp`-subgraph weights present in the
// merged tables; rejected metadata-key probing (the checkpoint's tensors
// are the binding truth) and partial tolerance (a half-present head is a
// corrupt checkpoint, not an absent feature). Spec 8 §5 states the
// downgrade without defining its trigger.
pub fn downgrade_absent_mtp(
    spec: ModelSpec,
    ckpt: &OpenedCheckpoint,
    model_id: &str,
) -> Result<(ModelSpec, ModelGraph), LoaderError> {
    if spec.mtp.is_none() {
        let builder = Graph::new(IrVersion::CURRENT, model_id);
        let graph = build_model(builder, &spec)?;
        return Ok((spec, graph));
    }
    let builder = Graph::new(IrVersion::CURRENT, model_id);
    let graph = build_model(builder, &spec)?;
    let Some(mtp) = graph.subgraphs().get("mtp") else {
        return Ok((spec, graph));
    };
    let present = mtp
        .bound_weights()
        .iter()
        .filter(|w| ckpt.tensor(&w.name).is_some())
        .count();
    if present > 0 {
        return Ok((spec, graph));
    }
    let mut downgraded = spec;
    downgraded.mtp = None;
    downgraded.validate()?;
    let builder = Graph::new(IrVersion::CURRENT, model_id);
    let graph = build_model(builder, &downgraded)?;
    Ok((downgraded, graph))
}

/// Checks every fusion declaration of the graph and its nested subgraphs
/// against the checkpoint, returning one problem string per violation
/// (Spec 8 §5, Spec 2 §4).
///
/// Each declaration's members must be bound weights with compatible
/// geometry (`GateUp` members share their full shape; `Qkv` members are
/// rank-2 projections over the same model dimension), and — for native
/// checkpoints, which carry the interleave already — each member's
/// declared `r9v.tensor.*.interleave` must name the declaration's kind, or
/// load fails. Standard GGUF carries no interleave metadata; its fusion
/// is applied on repack (card A2.8), so only membership and geometry are
/// checked there.
pub fn check_fusion_decls(graph: &ModelGraph, ckpt: &OpenedCheckpoint) -> Vec<String> {
    check_fusion_hits(graph, ckpt)
        .into_iter()
        .map(|hit| hit.detail)
        .collect()
}

/// Derives the graph model id from metadata: `general.name` when present,
/// else `general.architecture` (Spec 8 §4).
// DECISION(A2.6): `general.name`-else-architecture for the graph model id;
// rejected the checkpoint path because the id must be stable across cache
// copies of the same model. Spec 8 §4 is silent on the id source.
pub fn model_id_from_meta(meta: &(impl GgufMeta + ?Sized)) -> Result<String, LoaderError> {
    if let Ok(Some(name)) = meta.get_str("general.name") {
        return Ok(name.to_string());
    }
    Ok(meta.str("general.architecture")?.to_string())
}

/// Spec 8 §6 item 3: `vocab` matches the tokenizer.
fn check_vocab_tokenizer_match(
    spec: &ModelSpec,
    ckpt: &OpenedCheckpoint,
    problems: &mut Vec<String>,
) {
    let tokens = match ckpt.file().kv("tokenizer.ggml.tokens") {
        Some(KvValue::Array { items, .. }) => items,
        Some(_) | None => return,
    };
    match u32::try_from(tokens.len()) {
        Ok(n) if n == spec.vocab => {}
        Ok(n) => problems.push(format!(
            "tokenizer token count {n} does not match model vocab {} (Spec 8 §6)",
            spec.vocab,
        )),
        Err(_) => problems.push(format!(
            "tokenizer token count {} exceeds u32 range (Spec 8 §6)",
            tokens.len(),
        )),
    }
}

/// One fusion violation: the tensor it attaches to plus the report text.
struct FusionHit {
    /// Member (or first member) the violation attaches to, for tensor
    /// aggregation and deterministic ordering.
    name: String,
    /// Human-readable violation, naming the declaration and member.
    detail: String,
}

/// One problem string per fusion violation, in declaration order.
fn check_fusion_hits(graph: &ModelGraph, ckpt: &OpenedCheckpoint) -> Vec<FusionHit> {
    let mut out = Vec::new();
    check_graph_fusions(graph, ckpt, &mut out);
    for sub in graph.subgraphs().values() {
        check_graph_fusions_recursive(sub, ckpt, &mut out);
    }
    out
}

/// Recurses nested subgraphs in name order for fusion checks.
fn check_graph_fusions_recursive(
    graph: &ModelGraph,
    ckpt: &OpenedCheckpoint,
    out: &mut Vec<FusionHit>,
) {
    check_graph_fusions(graph, ckpt, out);
    for sub in graph.subgraphs().values() {
        check_graph_fusions_recursive(sub, ckpt, out);
    }
}

/// Checks one graph level's fusion declarations.
fn check_graph_fusions(graph: &ModelGraph, ckpt: &OpenedCheckpoint, out: &mut Vec<FusionHit>) {
    use std::collections::BTreeMap;
    let mut shapes: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    for w in graph.bound_weights() {
        let mut shape = Vec::with_capacity(w.shape.len());
        for dim in &w.shape {
            match dim {
                r9v_ir::Dim::Concrete(n) => shape.push(u64::from(*n)),
                r9v_ir::Dim::Symbolic(_) => break,
            }
        }
        if shape.len() == w.shape.len() {
            shapes.insert(w.name.as_str(), shape);
        }
    }
    for decl in graph.fusion_decls() {
        match decl {
            FusionDecl::Qkv { q, k, v } => {
                check_fusion_members(ckpt, "qkv", &[q, k, v], &shapes, out);
                check_qkv_geometry(&shapes, q, k, v, out);
            }
            FusionDecl::GateUp { gate, up } => {
                check_fusion_members(ckpt, "gate_up", &[gate, up], &shapes, out);
                check_gate_up_geometry(&shapes, gate, up, out);
            }
        }
    }
}

/// Pushes one hit.
fn hit(out: &mut Vec<FusionHit>, name: &str, detail: String) {
    out.push(FusionHit {
        name: name.to_string(),
        detail,
    });
}

/// Membership: every fused name is a bound weight of this graph level.
fn check_fusion_members(
    ckpt: &OpenedCheckpoint,
    kind: &str,
    members: &[&String],
    shapes: &std::collections::BTreeMap<&str, Vec<u64>>,
    out: &mut Vec<FusionHit>,
) {
    let expected = if kind == "qkv" {
        Interleave::Qkv
    } else {
        Interleave::GateUp
    };
    for member in members {
        if !shapes.contains_key(member.as_str()) {
            hit(
                out,
                member,
                format!("fusion {kind} names unbound weight '{member}' (Spec 8 §5)"),
            );
            continue;
        }
        // Native checkpoints carry the interleave already; the declaration
        // must match or load fails (Spec 8 §5). Standard GGUF skips this:
        // fusion is applied on repack (card A2.8).
        if let Some((shard_index, _)) = ckpt.tensor(member.as_str()) {
            let is_native = ckpt
                .shard_file(shard_index)
                .map(|f| f.is_native())
                .unwrap_or(false);
            let declared = ckpt
                .shard_r9v_meta(shard_index)
                .and_then(|m| m.tensor(member.as_str()))
                .map(|t| t.interleave)
                .unwrap_or(Interleave::None);
            if is_native && declared != expected {
                hit(
                    out,
                    member,
                    format!(
                        "fusion {kind} member '{member}' declares interleave '{}' in the native checkpoint, expected '{kind}' (Spec 8 §5)",
                        interleave_name(declared),
                    ),
                );
            }
        }
    }
}

/// Geometry: `GateUp` members share their full logical shape.
fn check_gate_up_geometry(
    shapes: &std::collections::BTreeMap<&str, Vec<u64>>,
    gate: &str,
    up: &str,
    out: &mut Vec<FusionHit>,
) {
    match (shapes.get(gate), shapes.get(up)) {
        (Some(g), Some(u)) if g != u => hit(
            out,
            gate,
            format!(
                "fusion gate_up members have different shapes: '{gate}' {g:?} vs '{up}' {u:?} (Spec 8 §5)"
            ),
        ),
        _ => {}
    }
}

/// Geometry: `Qkv` members are rank-2 projections over one model dimension.
fn check_qkv_geometry(
    shapes: &std::collections::BTreeMap<&str, Vec<u64>>,
    q: &str,
    k: &str,
    v: &str,
    out: &mut Vec<FusionHit>,
) {
    let dims: Vec<(&str, &Vec<u64>)> = [q, k, v]
        .into_iter()
        .filter_map(|name| shapes.get(name).map(|s| (name, s)))
        .collect();
    for (name, shape) in &dims {
        if shape.len() != 2 {
            hit(
                out,
                name,
                format!(
                    "fusion qkv member '{name}' has rank {}, expected a rank-2 projection (Spec 8 §5)",
                    shape.len(),
                ),
            );
        }
    }
    let mut k_dims = dims.iter().filter(|(_, s)| s.len() == 2).map(|(_, s)| s[1]);
    if let Some(first) = k_dims.next() {
        for other in k_dims {
            if other != first {
                hit(
                    out,
                    q,
                    format!(
                        "fusion qkv members project over different model dimensions: '{q}', '{k}', '{v}' (Spec 8 §5)"
                    ),
                );
                break;
            }
        }
    }
}

/// Stable interleave name for reports.
fn interleave_name(interleave: Interleave) -> &'static str {
    match interleave {
        Interleave::None => "none",
        Interleave::GateUp => "gate_up",
        Interleave::Qkv => "qkv",
    }
}
