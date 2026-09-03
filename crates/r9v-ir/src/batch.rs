// SPDX-License-Identifier: Apache-2.0
//! Batch metadata shared by ops (Spec 1 §2.5, §4.D.1).
//!
//! One [`BatchMeta`] is an external input shared by ops (Spec 1 §2.5). `G`
//! (layer groups, one block table per group per Spec 3 §6.1) is fixed per
//! model, which keeps `BatchMeta` fixed-shape (Spec 3 §3.3). Build with
//! [`BatchMeta::builder`]; the builder checks every `[G, T]` / `[G, S,
//! max_blocks]` / `[G, S]` length against the declared dims and reports every
//! mismatch at once (CONVENTIONS.md §1.4).

use crate::IrError;

/// Sentinel padding value for unused `block_table` entries.
///
/// Tables are `[G, S, max_blocks]` with `max_blocks = ceil(max_ctx / 32)`
/// (Spec 3 §3.3); rows shorter than `max_blocks` are padded with a value that
/// can never be a real block id.
// DECISION(A1.1): u32::MAX. Spec 3 §3.3 requires "a sentinel" without naming
// it; rejected 0 because block 0 is a valid allocation.
pub const BLOCK_TABLE_SENTINEL: u32 = u32::MAX;

/// Token positions: per-token absolute positions, or triplets under mrope
/// (`positions [T] u32 | [T,3] u32`, Spec 1 §2.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Positions {
    /// One absolute position per token (Spec 1 §2.5).
    PerToken(Vec<u32>),
    /// One `(t, h, w)`-style triplet per token under mrope (Spec 1 §2.5).
    Mrope(Vec<[u32; 3]>),
}

impl Positions {
    /// Number of tokens described; must equal batch `T` (Spec 1 §2.5).
    pub fn len(&self) -> usize {
        match self {
            Positions::PerToken(v) => v.len(),
            Positions::Mrope(v) => v.len(),
        }
    }

    /// True when no token is described.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Speculative tree mask (Spec 1 §4.D.1).
///
/// `parents [T] i32` (−1 = root of its sequence) plus the derived
/// `[T, T_max] bool` ancestor mask, built by the scheduler. Kernels may
/// consume either form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeMask {
    parents: Vec<i32>,
    t_max: u32,
    ancestors: Vec<bool>,
}

impl TreeMask {
    /// Builds a tree mask, checking `parents` ids (−1 or `< T`, never self)
    /// and the `T * t_max` ancestor length (Spec 1 §4.D.1).
    ///
    /// `t_max` is the column count per token row; producers (scheduler
    /// pre-step, Spec 6) size it to cover the longest query in the batch.
    // DECISION(A1.1): T_max sizing rule. Spec 1 §4.D.1 gives the shape
    // `[T, T_max]` but no sizing rule; rejected fixing it to a bucket
    // constant because bucket sizes vary (Spec 1 §3.5). The IR carries the
    // mask the scheduler built; it does not derive ancestor sets itself.
    pub fn new(parents: Vec<i32>, t_max: u32, ancestors: Vec<bool>) -> Result<Self, IrError> {
        let t = parents.len();
        let mut problems = Vec::new();
        if t > 0 && t_max == 0 {
            problems.push(IrError::ZeroTreeMax { t });
        }
        match t.checked_mul(t_max as usize) {
            Some(expected) if ancestors.len() != expected => {
                problems.push(IrError::AncestorLength {
                    t,
                    t_max,
                    expected,
                    actual: ancestors.len(),
                });
            }
            Some(_) => {}
            None => problems.push(IrError::AncestorShapeOverflow { t, t_max }),
        }
        for (token, parent) in parents.iter().enumerate() {
            if *parent < -1 || *parent >= t as i32 {
                problems.push(IrError::BadParent {
                    token,
                    parent: *parent,
                    t,
                });
            } else if *parent == token as i32 {
                problems.push(IrError::SelfParent { token });
            }
        }
        collect_parent_cycles(&parents, &mut problems);
        if problems.is_empty() {
            Ok(Self {
                parents,
                t_max,
                ancestors,
            })
        } else if problems.len() == 1 {
            Err(problems
                // Internal invariant: this branch runs only when len == 1.
                .pop()
                .expect("problems holds exactly one entry"))
        } else {
            Err(IrError::Multiple {
                problems: problems.into_boxed_slice(),
            })
        }
    }

    /// Token count `T` (`parents.len()`).
    pub fn t(&self) -> usize {
        self.parents.len()
    }

    /// Columns per row (`T_max`).
    pub fn t_max(&self) -> u32 {
        self.t_max
    }

    /// Parent ids, −1 for roots (Spec 1 §4.D.1).
    pub fn parents(&self) -> &[i32] {
        &self.parents
    }

    /// Flattened `[T, T_max]` row-major ancestor mask (Spec 1 §4.D.1).
    pub fn ancestors(&self) -> &[bool] {
        &self.ancestors
    }

    /// Ancestor bit for token `tok` at column `pos`.
    ///
    /// Panics on out-of-bounds indices: dimensions are fixed at construction,
    /// so a bad index is a caller bug, not input data.
    pub fn is_ancestor(&self, tok: usize, pos: usize) -> bool {
        assert!(
            tok < self.parents.len(),
            "TreeMask::is_ancestor token {tok} out of bounds for T={}",
            self.parents.len(),
        );
        assert!(
            (pos as u32) < self.t_max,
            "TreeMask::is_ancestor pos {pos} out of bounds for t_max={}",
            self.t_max,
        );
        self.ancestors[tok * self.t_max as usize + pos]
    }
}

/// Batch metadata: one external input shared by ops (Spec 1 §2.5).
///
/// Explicit shape semantics: `G` layer groups (fixed per model, Spec 3 §3.3),
/// `S` sequences, `T = T_dec + T_pre` tokens (Spec 1 §3.1), `max_blocks =
/// ceil(max_ctx / 32)` blocks per sequence per group (Spec 3 §3.3).
/// `slot_map` is `[G, T]` flattened slots, `block_table` is
/// `[G, S, max_blocks]`, `window_start` is `[G, S]` (first retained position
/// for Window groups, 0 otherwise; Spec 1 §2.5, Spec 3 §3.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchMeta {
    g: u32,
    s: u32,
    t: u32,
    max_blocks: u32,
    seq_ids: Vec<u32>,
    query_len: Vec<u32>,
    ctx_len: Vec<u32>,
    positions: Positions,
    slot_map: Vec<u32>,
    block_table: Vec<u32>,
    window_start: Vec<u32>,
    tree: Option<TreeMask>,
}

impl BatchMeta {
    /// Starts a builder for the given dims (Spec 1 §2.5 shapes).
    pub fn builder(g: u32, s: u32, t: u32, max_blocks: u32) -> BatchMetaBuilder {
        BatchMetaBuilder {
            g,
            s,
            t,
            max_blocks,
            seq_ids: None,
            query_len: None,
            ctx_len: None,
            positions: None,
            slot_map: None,
            block_table: None,
            window_start: None,
            tree: None,
        }
    }

    /// Layer-group count `G`, fixed per model (Spec 3 §3.3).
    pub fn g(&self) -> u32 {
        self.g
    }

    /// Sequence count `S` (Spec 1 §2.5).
    pub fn s(&self) -> u32 {
        self.s
    }

    /// Token count `T = T_dec + T_pre` (Spec 1 §2.5, §3.1).
    pub fn t(&self) -> u32 {
        self.t
    }

    /// Blocks per sequence per group, `ceil(max_ctx / 32)` (Spec 3 §3.3).
    pub fn max_blocks(&self) -> u32 {
        self.max_blocks
    }

    /// Sequence ids `[S]` (Spec 1 §2.5).
    pub fn seq_ids(&self) -> &[u32] {
        &self.seq_ids
    }

    /// Query lengths `[S]`: 1 for plain decode, `k+1` for spec verify, chunk
    /// size for prefill (Spec 1 §2.5).
    pub fn query_len(&self) -> &[u32] {
        &self.query_len
    }

    /// Tokens already in state before this step, `[S]` (Spec 1 §2.5).
    pub fn ctx_len(&self) -> &[u32] {
        &self.ctx_len
    }

    /// Token positions, `[T]` or `[T,3]` (Spec 1 §2.5).
    pub fn positions(&self) -> &Positions {
        &self.positions
    }

    /// Flattened KV destination slot per new token per group, `[G, T]`
    /// row-major (Spec 1 §2.5, Spec 3 §3.3).
    pub fn slot_map(&self) -> &[u32] {
        &self.slot_map
    }

    /// Block tables, `[G, S, max_blocks]` row-major, padded with
    /// [`BLOCK_TABLE_SENTINEL`] (Spec 1 §2.5, Spec 3 §3.3).
    pub fn block_table(&self) -> &[u32] {
        &self.block_table
    }

    /// First retained position per group per sequence, `[G, S]` row-major; 0
    /// for non-window groups (Spec 1 §2.5, Spec 3 §3.5).
    pub fn window_start(&self) -> &[u32] {
        &self.window_start
    }

    /// Speculative tree mask, if the step verifies drafts (Spec 1 §4.D.1).
    pub fn tree(&self) -> Option<&TreeMask> {
        self.tree.as_ref()
    }

    /// Slot for new token `tok` in group `group` (row-major `[G, T]`).
    ///
    /// Panics on out-of-bounds indices: dims are fixed at construction, so a
    /// bad index is a caller bug, not input data.
    pub fn slot(&self, group: u32, tok: u32) -> u32 {
        assert!(
            group < self.g,
            "BatchMeta::slot group {group} >= G={}",
            self.g
        );
        assert!(tok < self.t, "BatchMeta::slot tok {tok} >= T={}", self.t);
        self.slot_map[group as usize * self.t as usize + tok as usize]
    }

    /// Block id for sequence `seq`, slot-entry `b`, in group `group`
    /// (row-major `[G, S, max_blocks]`).
    ///
    /// Panics on out-of-bounds indices: dims are fixed at construction, so a
    /// bad index is a caller bug, not input data.
    pub fn block(&self, group: u32, seq: u32, b: u32) -> u32 {
        assert!(
            group < self.g,
            "BatchMeta::block group {group} >= G={}",
            self.g
        );
        assert!(seq < self.s, "BatchMeta::block seq {seq} >= S={}", self.s);
        assert!(
            b < self.max_blocks,
            "BatchMeta::block b {b} >= max_blocks={}",
            self.max_blocks,
        );
        self.block_table[(group as usize * self.s as usize + seq as usize)
            * self.max_blocks as usize
            + b as usize]
    }

    /// First retained position for sequence `seq` in group `group`
    /// (row-major `[G, S]`).
    ///
    /// Panics on out-of-bounds indices: dims are fixed at construction, so a
    /// bad index is a caller bug, not input data.
    pub fn window(&self, group: u32, seq: u32) -> u32 {
        assert!(
            group < self.g,
            "BatchMeta::window group {group} >= G={}",
            self.g
        );
        assert!(seq < self.s, "BatchMeta::window seq {seq} >= S={}", self.s);
        self.window_start[group as usize * self.s as usize + seq as usize]
    }
}

/// Builder for [`BatchMeta`] (Spec 1 §2.5).
///
/// `G/S/T/max_blocks` are fixed up front; every field is set by name and
/// [`BatchMetaBuilder::build`] checks presence plus all lengths at once.
#[derive(Debug, Clone)]
pub struct BatchMetaBuilder {
    g: u32,
    s: u32,
    t: u32,
    max_blocks: u32,
    seq_ids: Option<Vec<u32>>,
    query_len: Option<Vec<u32>>,
    ctx_len: Option<Vec<u32>>,
    positions: Option<Positions>,
    slot_map: Option<Vec<u32>>,
    block_table: Option<Vec<u32>>,
    window_start: Option<Vec<u32>>,
    tree: Option<TreeMask>,
}

impl BatchMetaBuilder {
    /// Sequence ids `[S]` (Spec 1 §2.5).
    pub fn seq_ids(mut self, v: Vec<u32>) -> Self {
        self.seq_ids = Some(v);
        self
    }

    /// Query lengths `[S]` (Spec 1 §2.5).
    pub fn query_len(mut self, v: Vec<u32>) -> Self {
        self.query_len = Some(v);
        self
    }

    /// Context lengths `[S]` (Spec 1 §2.5).
    pub fn ctx_len(mut self, v: Vec<u32>) -> Self {
        self.ctx_len = Some(v);
        self
    }

    /// Token positions, `[T]` or `[T,3]` (Spec 1 §2.5).
    pub fn positions(mut self, v: Positions) -> Self {
        self.positions = Some(v);
        self
    }

    /// Flattened slots `[G, T]` row-major (Spec 1 §2.5).
    pub fn slot_map(mut self, v: Vec<u32>) -> Self {
        self.slot_map = Some(v);
        self
    }

    /// Block tables `[G, S, max_blocks]` row-major (Spec 1 §2.5).
    pub fn block_table(mut self, v: Vec<u32>) -> Self {
        self.block_table = Some(v);
        self
    }

    /// First retained positions `[G, S]` row-major (Spec 1 §2.5).
    pub fn window_start(mut self, v: Vec<u32>) -> Self {
        self.window_start = Some(v);
        self
    }

    /// Speculative tree mask (Spec 1 §4.D.1). Optional; `None` when the step
    /// verifies no drafts.
    pub fn tree(mut self, v: Option<TreeMask>) -> Self {
        self.tree = v;
        self
    }

    /// Validates presence and every `[G, T]` / `[G, S, max_blocks]` / `[G, S]`
    /// length, reporting all failures at once (CONVENTIONS.md §1.4).
    pub fn build(self) -> Result<BatchMeta, IrError> {
        let mut problems = Vec::new();
        if self.g == 0 || self.s == 0 || self.t == 0 || self.max_blocks == 0 {
            problems.push(IrError::ZeroBatchDim {
                g: self.g,
                s: self.s,
                t: self.t,
                max_blocks: self.max_blocks,
            });
        }
        let s = self.s as usize;
        let t = self.t as usize;

        let seq_ids = check_len(&mut problems, "seq_ids", self.seq_ids, Some(s));
        let query_len = check_len(&mut problems, "query_len", self.query_len, Some(s));
        let ctx_len = check_len(&mut problems, "ctx_len", self.ctx_len, Some(s));
        let slot_map_expected = checked_product(&mut problems, "slot_map", &[self.g, self.t]);
        let block_table_expected = checked_product(
            &mut problems,
            "block_table",
            &[self.g, self.s, self.max_blocks],
        );
        let window_start_expected =
            checked_product(&mut problems, "window_start", &[self.g, self.s]);
        let slot_map = check_len(&mut problems, "slot_map", self.slot_map, slot_map_expected);
        let block_table = check_len(
            &mut problems,
            "block_table",
            self.block_table,
            block_table_expected,
        );
        let window_start = check_len(
            &mut problems,
            "window_start",
            self.window_start,
            window_start_expected,
        );

        let positions = match self.positions {
            Some(p) if p.len() == t => Some(p),
            Some(p) => {
                problems.push(IrError::BatchLength {
                    field: "positions",
                    expected: t,
                    actual: p.len(),
                });
                None
            }
            None => {
                problems.push(IrError::MissingField { field: "positions" });
                None
            }
        };

        if let Some(q) = query_len.as_ref() {
            for (seq, len) in q.iter().enumerate() {
                if *len == 0 {
                    problems.push(IrError::EmptyQuery { seq });
                }
            }
            let actual = q.iter().map(|&len| u64::from(len)).sum();
            if actual != u64::from(self.t) {
                problems.push(IrError::QueryTokenCount {
                    expected: self.t,
                    actual,
                });
            }
        }

        if let Some(tree) = self.tree.as_ref() {
            if tree.t() != t {
                problems.push(IrError::TreeBatchMismatch {
                    tree_t: tree.t(),
                    batch_t: self.t,
                });
            }
            if let Some(query_len) = query_len.as_ref() {
                let required = query_len.iter().copied().max().unwrap_or(0);
                if tree.t_max() < required {
                    problems.push(IrError::TreeMaxTooSmall {
                        required,
                        actual: tree.t_max(),
                    });
                }
                if tree.t() == t {
                    let mut seq_start = 0usize;
                    for (seq, &seq_len) in query_len.iter().enumerate() {
                        let seq_end = seq_start + seq_len as usize;
                        for token in seq_start..seq_end {
                            let parent = tree.parents()[token];
                            if parent >= 0
                                && ((parent as usize) < seq_start || (parent as usize) >= seq_end)
                            {
                                problems.push(IrError::TreeParentCrossesSequence {
                                    token,
                                    parent,
                                    seq,
                                    seq_start,
                                    seq_end,
                                });
                            }
                        }
                        seq_start = seq_end;
                    }
                }
            }
        }

        if problems.is_empty() {
            Ok(BatchMeta {
                g: self.g,
                s: self.s,
                t: self.t,
                max_blocks: self.max_blocks,
                seq_ids: seq_ids.expect("checked present"),
                query_len: query_len.expect("checked present"),
                ctx_len: ctx_len.expect("checked present"),
                positions: positions.expect("checked present"),
                slot_map: slot_map.expect("checked present"),
                block_table: block_table.expect("checked present"),
                window_start: window_start.expect("checked present"),
                tree: self.tree,
            })
        } else if problems.len() == 1 {
            Err(problems
                // Internal invariant: this branch runs only when len == 1.
                .pop()
                .expect("problems holds exactly one entry"))
        } else {
            Err(IrError::Multiple {
                problems: problems.into_boxed_slice(),
            })
        }
    }
}

fn check_len(
    problems: &mut Vec<IrError>,
    field: &'static str,
    value: Option<Vec<u32>>,
    expected: Option<usize>,
) -> Option<Vec<u32>> {
    match value {
        Some(v) if expected.is_some_and(|expected| v.len() == expected) => Some(v),
        Some(v) if expected.is_some() => {
            problems.push(IrError::BatchLength {
                field,
                expected: expected.expect("guard proves expected is present"),
                actual: v.len(),
            });
            None
        }
        Some(_) => None,
        None => {
            problems.push(IrError::MissingField { field });
            None
        }
    }
}

fn checked_product(
    problems: &mut Vec<IrError>,
    field: &'static str,
    factors: &[u32],
) -> Option<usize> {
    let product = factors.iter().try_fold(1usize, |product, &factor| {
        product.checked_mul(factor as usize)
    });
    if product.is_none() {
        problems.push(IrError::BatchShapeOverflow {
            field,
            factors: factors.to_vec().into_boxed_slice(),
        });
    }
    product
}

fn collect_parent_cycles(parents: &[i32], problems: &mut Vec<IrError>) {
    let mut state = vec![0u8; parents.len()];
    for start in 0..parents.len() {
        if state[start] == 2 {
            continue;
        }
        let mut path = Vec::new();
        let mut token = start;
        loop {
            match state[token] {
                2 => break,
                1 => {
                    let first = path.iter().position(|&seen| seen == token).unwrap_or(0);
                    let cycle_token = path[first..].iter().copied().min().unwrap_or(token);
                    problems.push(IrError::TreeCycle { token: cycle_token });
                    break;
                }
                _ => {}
            }
            state[token] = 1;
            path.push(token);
            let parent = parents[token];
            if parent < 0 || parent as usize >= parents.len() || parent as usize == token {
                break;
            }
            token = parent as usize;
        }
        for token in path {
            state[token] = 2;
        }
    }
}
