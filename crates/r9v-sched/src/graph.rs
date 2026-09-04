// SPDX-License-Identifier: Apache-2.0
//! Step graph capture against the kernel registry and graph cache (Spec 1 §3.1, Spec 6 §5.1, §5.2).

use std::collections::BTreeMap;
use std::sync::Arc;

use r9v_ir::{PlanId, StepGraphKey};
use r9v_registry::{ArchName, LaunchEntry, LaunchList, OpId, OpStatic, Registry, Tier};

use crate::arena::WorkspaceArena;
use crate::error::{SchedError, SchedResult};

/// Warm bucket lists captured eagerly at load (Spec 6 §5.1, §10).
pub const WARM_S: [u32; 3] = [1, 2, 4];
pub const WARM_T_DEC: [u32; 6] = [1, 2, 4, 8, 16, 32];
pub const WARM_T_PRE: [u32; 4] = [0, 128, 512, 2048];

/// An immutable captured step graph replayed during device step execution (Spec 6 §5.1, §5.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedGraph {
    /// Distinct key identifying the plan, rank, bucket, and segment (Spec 6 §5.1).
    pub key: StepGraphKey,
    /// Ordered launch operations to be dispatched to the device executor (Spec 4 §7, Spec 6 §5.2).
    pub launches: LaunchList,
    /// Total scratch workspace required by launches in bytes (Spec 4 §7, Spec 6 §5.3).
    pub required_workspace_bytes: u64,
    /// Distinct checked 256-byte aligned fixed workspace offsets bound to each launch entry (Spec 6 §5.3).
    pub workspace_offsets: Vec<u64>,
    /// List of resolved operations and their selected tiers (Spec 4 §9.2, Spec 6 §5.1).
    pub resolved_tiers: Vec<(OpId, Tier)>,
}

/// Type alias for op static resolver function.
pub type OpResolver = Arc<dyn Fn(&StepGraphKey) -> Option<OpStatic> + Send + Sync>;

/// Type alias for the argument blob template builder function (Spec 4 §7).
pub type ArgsTemplateBuilder = Arc<dyn Fn(&StepGraphKey) -> Option<Vec<u8>> + Send + Sync>;

/// Operation registered in a step graph program.
#[derive(Clone)]
pub struct StepProgramOp {
    /// Operation identifier to resolve against the registry (Spec 4 §9).
    pub op_id: OpId,
    /// Diagnostic description of this operation.
    pub desc: String,
    /// Function producing the `OpStatic` descriptor for a given `StepGraphKey`,
    /// or `None` if this operation does not participate for this key (e.g. prefill-only when `t_pre == 0`).
    ///
    /// For pure prefill step graphs, `key.t_dec = 1` serves as a sentinel bucket per Spec 1 §3.1.
    pub resolver: OpResolver,
    /// Function supplying the argument blob template for this operation (Spec 4 §7).
    pub args_template_builder: ArgsTemplateBuilder,
    /// Declared byte offset of the 8-byte workspace pointer/offset slot within the argument blob (Spec 4 §7, Spec 6 §5.3).
    pub workspace_slot: Option<usize>,
}

impl std::fmt::Debug for StepProgramOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepProgramOp")
            .field("op_id", &self.op_id)
            .field("desc", &self.desc)
            .field("workspace_slot", &self.workspace_slot)
            .finish()
    }
}

impl StepProgramOp {
    // DECISION(A3.9): construction requires an explicit argument blob template and an
    // explicit optional workspace slot; rejected a silent generic 64-byte zero blob with
    // slot 0 because a real program would bind the wrong ABI slot without noticing.
    // Callers that truly want zeros write `vec![0u8; 64]` themselves. Spec 4 §7.
    /// Constructs a new step program operation with an explicit argument blob
    /// template and an explicit optional workspace slot (Spec 4 §7).
    ///
    /// The template supplies every argument byte; when the resolved variant needs
    /// workspace, `workspace_slot` must be `Some` 8-byte-aligned offset naming the
    /// 8-byte slot the capture binds, otherwise capture fails instead of guessing.
    pub fn new(
        op_id: OpId,
        desc: impl Into<String>,
        resolver: impl Fn(&StepGraphKey) -> Option<OpStatic> + Send + Sync + 'static,
        args_template: Vec<u8>,
        workspace_slot: Option<usize>,
    ) -> Self {
        Self {
            op_id,
            desc: desc.into(),
            resolver: Arc::new(resolver),
            args_template_builder: Arc::new(move |_| Some(args_template.clone())),
            workspace_slot,
        }
    }

    /// Sets an explicit argument blob template and declared workspace slot (Spec 4 §7).
    pub fn with_args_template(mut self, template: Vec<u8>, workspace_slot: Option<usize>) -> Self {
        self.args_template_builder = Arc::new(move |_| Some(template.clone()));
        self.workspace_slot = workspace_slot;
        self
    }

    /// Sets an argument blob template builder and declared workspace slot (Spec 4 §7).
    pub fn with_args_builder(
        mut self,
        builder: impl Fn(&StepGraphKey) -> Option<Vec<u8>> + Send + Sync + 'static,
        workspace_slot: Option<usize>,
    ) -> Self {
        self.args_template_builder = Arc::new(builder);
        self.workspace_slot = workspace_slot;
        self
    }

    /// Sets the declared workspace slot for this operation.
    pub fn with_workspace_slot(mut self, slot: Option<usize>) -> Self {
        self.workspace_slot = slot;
        self
    }
}

/// Injected concrete step graph program specifying the operations that compose
/// a step graph for a model and plan (Spec 1 §3.1, Spec 6 §5.1).
///
/// Dispatched per `StepGraphKey` (participating plan, rank, S, T_dec, T_pre, segment).
/// Note: for pure prefill step graphs, `t_dec = 1` is used as a sentinel bucket per Spec 1 §3.1.
#[derive(Clone, Debug, Default)]
pub struct StepGraphProgram {
    ops: Vec<StepProgramOp>,
}

impl StepGraphProgram {
    /// Constructs an empty step graph program.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Appends an operation to the program.
    pub fn add_op(&mut self, op: StepProgramOp) -> &mut Self {
        self.ops.push(op);
        self
    }

    /// Appends an operation to the program and returns `self` (fluent builder pattern).
    pub fn with_op(mut self, op: StepProgramOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Returns the operations in the program.
    pub fn ops(&self) -> &[StepProgramOp] {
        &self.ops
    }
}

/// Builder that captures a step graph by resolving op instances against the registry (Spec 6 §5.1).
pub struct StepGraphBuilder;

impl StepGraphBuilder {
    /// Captures a step graph for the given `key` against the registry and workspace arena (Spec 1 §3.1, Spec 6 §5.1, §5.3).
    ///
    /// Every variant workspace is allocated at a distinct checked 256-byte aligned fixed offset.
    /// The offset is bound into the declared workspace slot while preserving all other argument blob bytes.
    /// The capture verifies non-overlapping offsets and arena capacity bounds without mutating runtime arena state.
    pub fn capture(
        registry: &Registry,
        arch: &ArchName,
        key: StepGraphKey,
        program: &StepGraphProgram,
        arena: &WorkspaceArena,
    ) -> SchedResult<CapturedGraph> {
        if program.ops().is_empty() {
            return Err(SchedError::GraphCaptureFailed {
                key,
                reason: "step graph program is empty (no operations declared)".to_owned(),
            });
        }

        if !program.ops().iter().any(|op| op.op_id == OpId::Sample) {
            return Err(SchedError::GraphCaptureFailed {
                key,
                reason: "step graph program lacks required OpId::Sample operation".to_owned(),
            });
        }

        let mut launches = LaunchList::new();
        let mut resolved_tiers = Vec::new();
        let mut workspace_offsets = Vec::new();
        let mut current_offset: u64 = 0;

        for op_def in program.ops() {
            let op_static = match (op_def.resolver)(&key) {
                Some(s) => s,
                None => continue,
            };

            let variant = registry
                .resolve(op_def.op_id, arch, &op_static)
                .map_err(|e| SchedError::GraphCaptureFailed {
                    key,
                    reason: format!(
                        "failed to resolve op {:?} for {}: {e}",
                        op_def.op_id, op_def.desc
                    ),
                })?;

            resolved_tiers.push((op_def.op_id, variant.tier));

            let mut args_blob = match (op_def.args_template_builder)(&key) {
                Some(b) => b,
                None => {
                    return Err(SchedError::GraphCaptureFailed {
                        key,
                        reason: format!(
                            "missing argument blob template for op {:?} ({})",
                            op_def.op_id, op_def.desc
                        ),
                    });
                }
            };

            if variant.workspace_bytes > 0 {
                let slot = match op_def.workspace_slot {
                    Some(s) => s,
                    None => {
                        return Err(SchedError::GraphCaptureFailed {
                            key,
                            reason: format!(
                                "op {:?} requires {} B workspace but declared no workspace slot (missing binding)",
                                op_def.op_id, variant.workspace_bytes
                            ),
                        });
                    }
                };

                if slot % 8 != 0 {
                    return Err(SchedError::GraphCaptureFailed {
                        key,
                        reason: format!(
                            "op {:?} workspace slot {slot} is not 8-byte aligned (ambiguous binding)",
                            op_def.op_id
                        ),
                    });
                }

                let required_len = slot.checked_add(8).ok_or_else(|| {
                    SchedError::overflow("workspace_slot", "workspace slot offset overflow")
                })?;
                if args_blob.len() < required_len {
                    return Err(SchedError::GraphCaptureFailed {
                        key,
                        reason: format!(
                            "op {:?} args blob length {} is undersized for workspace slot at {slot} (requires {required_len} bytes)",
                            op_def.op_id, args_blob.len()
                        ),
                    });
                }

                // Align offset to 256-byte alignment (WORKSPACE_ALIGNMENT)
                let aligned_offset = if current_offset.is_multiple_of(256) {
                    current_offset
                } else {
                    let remainder = current_offset % 256;
                    let pad = 256 - remainder;
                    current_offset.checked_add(pad).ok_or_else(|| {
                        SchedError::overflow("workspace_alignment", "alignment overflow")
                    })?
                };

                // Bind checked 256-byte-aligned workspace pointer/offset into correct declared slot,
                // preserving EVERY other argument byte!
                let slot_range_end = slot.checked_add(8).ok_or_else(|| {
                    SchedError::overflow("workspace_slot", "workspace slot range overflow")
                })?;
                let slot_bytes = args_blob.get_mut(slot..slot_range_end).ok_or_else(|| {
                    SchedError::Internal("workspace slot range out of args blob".to_owned())
                })?;
                slot_bytes.copy_from_slice(&aligned_offset.to_le_bytes());
                workspace_offsets.push(aligned_offset);

                let next_offset = aligned_offset
                    .checked_add(variant.workspace_bytes)
                    .ok_or_else(|| {
                        SchedError::overflow("workspace_offset", "workspace offset overflow")
                    })?;
                current_offset = next_offset;
            } else {
                workspace_offsets.push(0);
            }

            launches.record(LaunchEntry::new(
                variant.variant_hash,
                variant.entry_symbol,
                variant.launch_geometry,
                variant.workspace_bytes,
                variant.static_bytes,
                variant.static_flops,
                args_blob,
            ));
        }

        if launches.is_empty() {
            return Err(SchedError::GraphCaptureFailed {
                key,
                reason: "step graph capture produced zero launches for key".to_owned(),
            });
        }

        let total_required_workspace = current_offset;

        // Verify capacity bounds against arena without mutating runtime state
        arena.check_requirement(total_required_workspace)?;

        // Verify no overlap among allocated non-zero workspace slices
        let mut bounds: Vec<(u64, u64)> = Vec::new();
        for (start, entry) in workspace_offsets.iter().zip(launches.entries()) {
            if entry.workspace_bytes == 0 {
                continue;
            }
            let end = start
                .checked_add(entry.workspace_bytes)
                .ok_or_else(|| SchedError::overflow("workspace_bounds", "upper bound overflow"))?;
            bounds.push((*start, end));
        }
        for (i, (start_i, end_i)) in bounds.iter().enumerate() {
            for (j, (start_j, end_j)) in bounds.iter().enumerate().skip(i + 1) {
                if !(end_i <= start_j || end_j <= start_i) {
                    return Err(SchedError::Internal(format!(
                        "detected overlapping workspace slices between launch {i} [{start_i}..{end_i}) and launch {j} [{start_j}..{end_j})"
                    )));
                }
            }
        }

        Ok(CapturedGraph {
            key,
            launches,
            required_workspace_bytes: total_required_workspace,
            workspace_offsets,
            resolved_tiers,
        })
    }
}

/// Cache of captured step graphs managing eager warm capture and lazy runtime capture (Spec 6 §5.1).
#[derive(Debug, Clone)]
pub struct GraphCache {
    program: StepGraphProgram,
    graphs: BTreeMap<StepGraphKey, CapturedGraph>,
    arena_generation: u64,
}

impl GraphCache {
    /// Creates a new graph cache for the given program.
    pub fn new(program: StepGraphProgram) -> Self {
        Self {
            program,
            graphs: BTreeMap::new(),
            arena_generation: 1,
        }
    }

    /// Captures all warm buckets eagerly at initialization (Spec 6 §5.1).
    pub fn eager_capture_warm_buckets(
        registry: &Registry,
        arch: &ArchName,
        plan_id: PlanId,
        rank: u32,
        program: StepGraphProgram,
        arena: &WorkspaceArena,
    ) -> SchedResult<Self> {
        let mut cache = Self {
            program,
            graphs: BTreeMap::new(),
            arena_generation: arena.generation(),
        };
        for &s in &[1u32] {
            for &t_dec in &WARM_T_DEC {
                if t_dec < s {
                    continue;
                }
                for &t_pre in &WARM_T_PRE {
                    let key = StepGraphKey {
                        plan_id,
                        rank,
                        s,
                        t_dec,
                        t_pre,
                        segment: 0,
                    };
                    let graph =
                        StepGraphBuilder::capture(registry, arch, key, &cache.program, arena)?;
                    cache.graphs.insert(key, graph);
                }
            }
        }
        Ok(cache)
    }

    /// Retrieves an existing graph or captures it lazily on first use (Spec 6 §5.1).
    ///
    /// If a larger bucket genuinely requires growing the arena, grows the arena and
    /// recaptures every graph on that rank with the new arena generation (Spec 6 §5.3).
    /// Returns `(graph, was_lazy_captured)`.
    pub fn get_or_capture<'a>(
        &'a mut self,
        registry: &Registry,
        arch: &ArchName,
        key: StepGraphKey,
        arena: &mut WorkspaceArena,
    ) -> SchedResult<(&'a CapturedGraph, bool)> {
        if self.graphs.contains_key(&key) {
            let graph = self
                .graphs
                .get(&key)
                .ok_or_else(|| SchedError::Internal("verified graph exists in cache".to_owned()))?;
            return Ok((graph, false));
        }

        let captured = match StepGraphBuilder::capture(registry, arch, key, &self.program, arena) {
            Ok(g) => g,
            Err(SchedError::ArenaOverflow { required, .. }) => {
                // Spec 6 §5.3: Growing the arena is the only runtime allocation,
                // and it forces recapture of every graph on that rank.
                arena.grow(required)?;
                let new_gen = arena.generation();
                let existing_keys: Vec<StepGraphKey> = self.graphs.keys().cloned().collect();
                for existing_key in existing_keys {
                    let recaptured = StepGraphBuilder::capture(
                        registry,
                        arch,
                        existing_key,
                        &self.program,
                        arena,
                    )?;
                    self.graphs.insert(existing_key, recaptured);
                }
                self.arena_generation = new_gen;

                // Capture the new graph with the newly grown arena
                StepGraphBuilder::capture(registry, arch, key, &self.program, arena)?
            }
            Err(e) => return Err(e),
        };

        self.graphs.insert(key, captured);
        let graph = self.graphs.get(&key).ok_or_else(|| {
            SchedError::Internal("just inserted graph exists in cache".to_owned())
        })?;
        Ok((graph, true))
    }

    /// Returns the active arena generation of the cached graphs (Spec 6 §5.3).
    pub fn arena_generation(&self) -> u64 {
        self.arena_generation
    }

    /// Returns the number of cached graphs.
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Returns a reference to the program.
    pub fn program(&self) -> &StepGraphProgram {
        &self.program
    }
}
