// SPDX-License-Identifier: Apache-2.0
//! Loader error type (Spec 9 §12; `CONVENTIONS.md` §1).
//!
//! Every failure of pipeline steps 1–4 reports what was required, what was
//! available, and every failing item — never just the first one. Lower-crate
//! errors compose down the dependency graph via typed `#[from]` variants.

use r9v_ir::IrVersion;

/// Where a budget applies (Spec 9 §4.3: per device and for host).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BudgetScope {
    /// Device arena on the given rank (Spec 9 §4.1).
    Device {
        /// Plan rank owning the arena.
        rank: u32,
    },
    /// Pinned host memory (Spec 9 §4.2).
    Host,
}

impl std::fmt::Display for BudgetScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetScope::Device { rank } => write!(f, "device {rank}"),
            BudgetScope::Host => write!(f, "host"),
        }
    }
}

/// One tensor binding problem (Spec 8 §6, Spec 9 §12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorProblem {
    /// Tensor name following llama.cpp GGUF convention.
    pub name: String,
    /// What failed for this tensor.
    pub kind: TensorProblemKind,
}

/// The failure class for one tensor (Spec 8 §6, Spec 9 §12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorProblemKind {
    /// No tensor of this name exists in the checkpoint.
    Missing,
    /// File tensor has a different logical shape than the model expects
    /// (Spec 8 §6 item 1, after spec 2 padding rules).
    ShapeMismatch {
        /// Expected logical shape, outer-last order.
        expected: Vec<u64>,
        /// Actual logical shape from the container, outer-last order.
        actual: Vec<u64>,
    },
    /// File tensor's type is not consumable under the weight's expected
    /// scheme class (Spec 8 §2).
    Scheme {
        /// Expected scheme class name.
        expected_class: String,
        /// Actual container tensor type name.
        actual: String,
    },
    /// The planned placement is illegal for the tensor's semantic role
    /// (Spec 1 §2.3).
    Placement {
        /// Semantic role name.
        role: String,
        /// Planned placement rendering.
        placement: String,
    },
    /// A fusion declaration is violated: a member is missing, mis-shaped,
    /// or (for native checkpoints) carries a non-matching interleave
    /// (Spec 8 §5, Spec 2 §4).
    Fusion {
        /// Human-readable declaration violation, naming the declaration
        /// and the offending member.
        detail: String,
    },
    /// The tensor's byte size cannot be determined exactly (unknown type
    /// code or unrepresentable geometry); budgets must be exact, so this
    /// fails closed rather than estimating.
    Unmeasurable {
        /// Why no exact byte count exists.
        reason: String,
    },
}

/// Loader pipeline error (Spec 9 §2, §4, §12; `CONVENTIONS.md` §1).
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    /// Container parse or table validation failure (card A2.5).
    #[error(transparent)]
    Format(#[from] r9v_format::FormatError),

    /// Model resolution, spec validation, or graph build failure
    /// (cards A1.3/A1.4).
    #[error(transparent)]
    Models(#[from] r9v_models::ModelsError),

    /// State grouping or pool-sizing failure (card A1.11).
    #[error(transparent)]
    State(#[from] r9v_state::StateError),

    /// Shared infrastructure failure.
    #[error(transparent)]
    Common(#[from] r9v_common::R9vError),

    /// Checkpoint file could not be read (Spec 9 §12 short read / I/O).
    #[error("cannot read checkpoint '{path}': {message}")]
    Io {
        /// Path that failed.
        path: String,
        /// Underlying I/O error rendering.
        message: String,
    },

    /// Budget refusal with exact numbers and an actionable suggestion
    /// (Spec 9 §4.3). Never silently lowers a setting.
    #[error("{scope} budget: required {required} B, available {available} B, shortfall {shortfall} B; largest: {largest:?}; suggestion: {suggestion}")]
    Budget {
        /// Device or host scope.
        scope: BudgetScope,
        /// Total bytes required.
        required: u64,
        /// Total bytes available.
        available: u64,
        /// `required - available`.
        shortfall: u64,
        /// Top contributors by bytes, descending.
        largest: Vec<(String, u64)>,
        /// Smallest single config change that would fit, with numbers.
        suggestion: String,
    },

    /// Every missing or mis-shaped tensor, reported together
    /// (Spec 8 §6 item 1, Spec 9 §12).
    #[error("{} tensor(s) missing or mis-shaped: {details:?}", details.len())]
    Tensors {
        /// All tensor problems, in model binding order.
        details: Vec<TensorProblem>,
    },

    /// Every non-tensor validation failure, reported together
    /// (Spec 8 §6 items 3–4).
    #[error("{} validation problem(s): {problems:?}", problems.len())]
    Validation {
        /// All problems, in check order.
        problems: Vec<String>,
    },

    /// Step 2 failed in both classes at once: tensor bindings and
    /// file-level validation each found problems, and both are reported
    /// together so no round trip hides the other (Spec 8 §6,
    /// `CONVENTIONS.md` §1.4).
    #[error("{} tensor problem(s) and {} validation problem(s): tensors {details:?}; validation {problems:?}", details.len(), problems.len())]
    Step2 {
        /// All tensor problems, in model binding order.
        details: Vec<TensorProblem>,
        /// All non-tensor problems, in check order.
        problems: Vec<String>,
    },

    /// A declared split sibling shard has no file at its derived path
    /// (Spec 9 §2 step 1).
    #[error("split shard {shard_index} of {shard_count} missing: expected file '{expected_path}'")]
    MissingShard {
        /// Zero-based shard index (`split.no`).
        shard_index: u32,
        /// Declared total shards (`split.count`).
        shard_count: u32,
        /// Derived filesystem path the shard was expected at.
        expected_path: String,
    },

    /// A single path declares a split set but its file name carries no
    /// derivable `-NNNNN-of-MMMMM` shard pattern (Spec 9 §2 step 1).
    #[error("cannot derive split siblings of '{path}': {detail}")]
    ShardPattern {
        /// Path that declared the split.
        path: String,
        /// Why no sibling set could be derived.
        detail: String,
    },

    /// The model pins an IR version the engine does not implement
    /// (Spec 8 §6 item 4).
    #[error("IR version mismatch: model pins {pinned}, engine implements {current} (Spec 8 §6)")]
    IrVersionMismatch {
        /// Version pinned by the model definition.
        pinned: IrVersion,
        /// Engine's current version.
        current: IrVersion,
    },

    /// Checked size arithmetic overflowed instead of wrapping
    /// (`CONVENTIONS.md` §1.5: untrusted input must not wrap or saturate).
    #[error("size computation for {what} overflowed: {detail}")]
    Overflow {
        /// What was being computed.
        what: String,
        /// Operands involved.
        detail: String,
    },

    /// The metadata prefix demanded more bytes than the allocator would
    /// provide (Spec 9 §2 step 1). Reports the demanded length, the
    /// current prefix length, and the allocator refusal; no byte is read
    /// and the prefix is unchanged. This is a refusal, not a ceiling:
    /// any demand the allocator honors still opens.
    #[error(
        "cannot grow open prefix to {required} bytes (current {current} bytes): {detail} (Spec 9 §2)"
    )]
    PrefixAlloc {
        /// Demanded prefix length in bytes.
        required: u64,
        /// Current prefix length in bytes.
        current: u64,
        /// Allocator refusal rendering.
        detail: String,
    },

    /// `tokenizer.ggml.*` metadata is missing, mistyped, or inconsistent.
    /// All problems are collected before returning (CONVENTIONS.md §1.4).
    #[error("tokenizer metadata invalid: {details:?}")]
    TokenizerMeta {
        /// Every problem found, not just the first.
        details: Vec<String>,
    },

    /// `tokenizer.ggml.model` names a tokenizer family this build does not
    /// implement. Fail closed: never guess (Spec 9 §7).
    #[error("unsupported tokenizer model {model:?}; supported: {supported:?}")]
    UnsupportedTokenizer {
        /// The `tokenizer.ggml.model` value found.
        model: String,
        /// Families this build implements.
        supported: Vec<String>,
    },

    /// `tokenizer.ggml.pre` names a pre-tokenizer this build does not
    /// implement for the tokenizer family. Fail closed (Spec 9 §7).
    #[error(
        "unsupported pre-tokenizer {pre:?} for tokenizer model {model:?}; supported: {supported:?}"
    )]
    UnsupportedPreTokenizer {
        /// The `tokenizer.ggml.pre` value found.
        pre: String,
        /// The tokenizer model it was requested for.
        model: String,
        /// Pre-tokenizers supported for that model.
        supported: Vec<String>,
    },

    /// A token id is outside `[0, vocab_size)`.
    #[error("token id {id} out of range for vocab size {vocab_size}")]
    TokenIdOutOfRange {
        /// The offending id.
        id: u32,
        /// Vocabulary length.
        vocab_size: usize,
    },

    /// A merge line is not exactly `"left right"` with both sides non-empty.
    #[error("tokenizer.ggml.merges[{index}] is malformed: {line:?}")]
    MalformedMerge {
        /// Merge-table index.
        index: usize,
        /// The offending line (truncated to 64 chars).
        line: String,
    },

    /// A resource bound was exceeded (fail closed; Spec 9 §12).
    #[error("tokenizer limit exceeded: {what} (limit {limit}, got {got})")]
    Limit {
        /// Which bound tripped.
        what: &'static str,
        /// The bound.
        limit: usize,
        /// The observed value.
        got: usize,
    },

    /// The chat template failed to parse.
    #[error("chat template parse error at byte {offset}: {detail}")]
    TemplateParse {
        /// Byte offset of the failure.
        offset: usize,
        /// What went wrong.
        detail: String,
    },

    /// The chat template failed to render (unknown name, bad type,
    /// unknown filter/test, budget exhausted, or a `raise_exception` call).
    #[error("chat template render error: {detail}")]
    TemplateRender {
        /// What went wrong, with the failing construct named.
        detail: String,
    },

    /// The chat template uses a construct outside the sandboxed subset
    /// (filesystem, network, time, or code execution). Fail closed
    /// (Spec 10 §3.1).
    #[error("chat template uses unsupported construct: {detail}")]
    TemplateUnsupported {
        /// The rejected construct.
        detail: String,
    },
}
