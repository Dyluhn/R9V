// SPDX-License-Identifier: Apache-2.0
//! Domain error types for the sequence scheduler (Spec 6, CONVENTIONS.md §1).

use r9v_ir::StepGraphKey;

/// Result type alias for scheduler operations (CONVENTIONS.md §1.1).
pub type SchedResult<T> = std::result::Result<T, SchedError>;

/// Domain-specific error enum for `r9v-sched` (Spec 6, CONVENTIONS.md §1.1).
#[derive(Debug, thiserror::Error)]
pub enum SchedError {
    /// Transparent wrapper for underlying sequence-state manager errors (Spec 3, CONVENTIONS.md §1.1).
    #[error(transparent)]
    State(#[from] r9v_state::StateError),

    /// Transparent wrapper for underlying kernel registry errors (Spec 4, CONVENTIONS.md §1.1).
    #[error(transparent)]
    Registry(#[from] r9v_registry::RegistryError),

    /// Transparent wrapper for Op IR errors (Spec 1, CONVENTIONS.md §1.1).
    #[error(transparent)]
    Ir(#[from] r9v_ir::IrError),

    /// Invalid request parameters detected at admission boundary with collect-all reporting (CONVENTIONS.md §1.4).
    #[error("invalid request: {problems:?}")]
    InvalidRequest {
        /// All collected validation failure reasons (CONVENTIONS.md §1.4).
        problems: Vec<String>,
    },

    /// Checked arithmetic overflow on input-derived quantities (Spec 6 §1, §4).
    #[error("arithmetic overflow in {what}: {details}")]
    ArithmeticOverflow {
        /// Context or operation where the overflow occurred.
        what: String,
        /// Numerical details of the operands.
        details: String,
    },

    /// Workspace arena capacity exceeded with full numeric accounting (Spec 6 §5.3, CONVENTIONS.md §1.3).
    #[error("workspace arena overflow: required {required} B, available {available} B, shortfall {shortfall} B")]
    ArenaOverflow {
        /// Total workspace memory required in bytes.
        required: u64,
        /// Total workspace memory available in bytes.
        available: u64,
        /// Shortfall in bytes.
        shortfall: u64,
    },

    /// Referenced sequence identifier not found in active scheduler state.
    #[error("sequence {seq_id} not found in scheduler")]
    SequenceNotFound {
        /// Sequence ID that was not found.
        seq_id: u64,
    },

    /// Step budget exceeded during admission or scheduling (Spec 6 §4.1).
    #[error("step budget refusal: required {required_ms:.3} ms, available {available_ms:.3} ms, shortfall {shortfall_ms:.3} ms")]
    BudgetRefusal {
        /// Step time required in milliseconds.
        required_ms: f32,
        /// Step time available in budget in milliseconds.
        available_ms: f32,
        /// Shortfall in milliseconds.
        shortfall_ms: f32,
    },

    /// Graph capture failure against the kernel registry (Spec 6 §5.1).
    #[error("step graph capture failed for {key:?}: {reason}")]
    GraphCaptureFailed {
        /// Capture key identifying the step graph.
        key: StepGraphKey,
        /// Underlying failure reason.
        reason: String,
    },

    /// Incremental detokenization failure during stop string evaluation (Spec 6 §7).
    #[error("detokenization failure: {detail}")]
    DetokenizeError {
        /// Specific detokenizer failure detail.
        detail: String,
    },

    /// Step execution or device launch failure (Spec 6 §8).
    #[error("device execution failure: {detail}")]
    ExecutionFailed {
        /// Details of the device execution failure.
        detail: String,
    },

    /// Non-finite or negative step cost or budget observed at a scheduler
    /// consumption boundary (Spec 6 §4.1).
    ///
    /// Costs fail closed: NaN, infinite, or negative values are rejected with
    /// this typed error instead of being clamped to zero.
    #[error("invalid cost value at {context}: got {value}")]
    InvalidCost {
        /// Scheduler boundary where the invalid value was consumed.
        context: &'static str,
        /// The offending cost or budget value in milliseconds.
        value: f32,
    },

    /// Explicit admission backpressure: total outstanding requests
    /// (finished retained + queued + active + paused) reached the configured
    /// maximum (Spec 6 §2).
    ///
    /// Taking a finished result reopens capacity; nothing is evicted or lost.
    #[error("scheduler at capacity: outstanding {outstanding} >= maximum {maximum}")]
    CapacityExceeded {
        /// Total outstanding requests observed at refusal time.
        outstanding: usize,
        /// Configured maximum total outstanding requests.
        maximum: usize,
    },

    /// Internal scheduler invariant error.
    #[error("internal scheduler error: {0}")]
    Internal(String),
}

impl SchedError {
    /// Constructs an `InvalidRequest` error accumulating multiple validation problems (CONVENTIONS.md §1.4).
    pub fn invalid_request(problems: Vec<String>) -> Self {
        Self::InvalidRequest { problems }
    }

    /// Constructs an `ArithmeticOverflow` error.
    pub fn overflow(what: impl Into<String>, details: impl Into<String>) -> Self {
        Self::ArithmeticOverflow {
            what: what.into(),
            details: details.into(),
        }
    }
}
