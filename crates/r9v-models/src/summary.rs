// SPDX-License-Identifier: Apache-2.0
//! Model architecture summary for the partitioner and planner (Spec 8 §7; Spec 5 §5; card A1.3).

use std::collections::BTreeMap;

use r9v_ir::QuantScheme;

use crate::builder::checked_add_u64;
use crate::error::ModelsError;

/// Quantization scheme key with canonical ordering for deterministic summaries (Spec 8 §7; CONVENTIONS.md §3.2).
// DECISION(A1.3): SchemeKey implements Ord to enable deterministic BTreeMap ordering of weight bytes by scheme; rejected HashMap or unordered iteration to satisfy determinism rules. Spec 8 §7, CONVENTIONS.md §3.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SchemeKey {
    /// Unquantized weight tensors.
    None,
    /// Per-row scale quantization.
    PerRow,
    /// Spec 2 native scheme by opaque scheme ID.
    Scheme(u64),
    /// Per-token activation scheme.
    PerToken,
    /// Per-block-32 activation scheme.
    PerBlock32,
}

impl From<QuantScheme> for SchemeKey {
    fn from(scheme: QuantScheme) -> Self {
        match scheme {
            QuantScheme::None => Self::None,
            QuantScheme::PerRow => Self::PerRow,
            QuantScheme::Scheme(id) => Self::Scheme(id.as_u64()),
            QuantScheme::PerToken => Self::PerToken,
            QuantScheme::PerBlock32 => Self::PerBlock32,
        }
    }
}

/// Token mixing sublayer classification for resource planning (Spec 8 §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MixerKind {
    /// Multi-Head Attention or variant.
    Attention,
    /// Linear Attention or SSM scan.
    LinearAttention,
}

/// MoE expert memory summary for a layer (Spec 8 §7).
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertSummary {
    /// Total routed experts count `E`.
    pub e: u32,
    /// Memory weight bytes per individual expert.
    pub bytes_each: u64,
    /// Expert activation frequency hint (uniform if unrecorded by quant tool).
    pub hot_hint: Vec<f32>,
}

/// Memory and state summary for an individual transformer layer (Spec 8 §7).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSummary {
    /// Weight bytes bucketed by quantization scheme in deterministic order.
    pub weight_bytes_by_scheme: BTreeMap<SchemeKey, u64>,
    /// State memory cost in bytes per token (KV cache).
    pub state_per_token_bytes: u64,
    /// State memory cost in bytes per sequence (Recurrent/Conv window).
    pub state_per_seq_bytes: u64,
    /// MoE expert parameters, if present.
    pub experts: Option<ExpertSummary>,
    /// Mixer classification.
    pub mixer_kind: Option<MixerKind>,
}

impl LayerSummary {
    /// Total weight memory bytes in this layer.
    ///
    /// Checked: bucket totals that overflow `u64` report
    /// [`ModelsError::ArithmeticOverflow`] instead of clamping, wrapping, or
    /// panicking.
    pub fn total_weight_bytes(&self) -> Result<u64, ModelsError> {
        const CTX: &str = "LayerSummary::total_weight_bytes";
        let mut total = 0u64;
        for bytes in self.weight_bytes_by_scheme.values() {
            total = checked_add_u64(total, *bytes, CTX)?;
        }
        Ok(total)
    }
}

/// Comprehensive model summary consumed by the partitioner and planner (Spec 8 §7; Spec 5 §5.1).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSummary {
    /// Per-layer summaries.
    pub layers: Vec<LayerSummary>,
    /// Token embedding table weight bytes.
    pub embed_bytes: u64,
    /// Language model output head weight bytes (0 if tied to embeddings).
    pub head_bytes: u64,
    /// Vocabulary size `V`.
    pub vocab: u32,
    /// Hidden dimension `Dm`.
    pub dm: u32,
    /// Key/Value head count `Hkv`.
    pub hkv: u32,
    /// Divisors of `Hkv` defining valid Tensor Parallel degrees (Spec 8 §7; Spec 5 §5.2).
    pub tp_divisors: Vec<u32>,
    /// Speculative n-gram table weight bytes, if present.
    pub ngram_table_bytes: u64,
    /// Whether Multi-Token Prediction (MTP) heads are present.
    pub mtp: bool,
    /// Whether pre-lm_head final hidden states are exported.
    pub export_hidden: bool,
}

impl ModelSummary {
    /// Total model weight memory bytes across all layers, embeddings, head, and n-gram tables.
    ///
    /// Checked: see [`LayerSummary::total_weight_bytes`].
    pub fn total_weight_bytes(&self) -> Result<u64, ModelsError> {
        const CTX: &str = "ModelSummary::total_weight_bytes";
        let mut total = 0u64;
        for layer in &self.layers {
            total = checked_add_u64(total, layer.total_weight_bytes()?, CTX)?;
        }
        total = checked_add_u64(total, self.embed_bytes, CTX)?;
        total = checked_add_u64(total, self.head_bytes, CTX)?;
        total = checked_add_u64(total, self.ngram_table_bytes, CTX)?;
        Ok(total)
    }

    /// Total state memory cost in bytes per token across all layers.
    ///
    /// Checked: see [`LayerSummary::total_weight_bytes`].
    pub fn total_state_per_token_bytes(&self) -> Result<u64, ModelsError> {
        const CTX: &str = "ModelSummary::total_state_per_token_bytes";
        let mut total = 0u64;
        for layer in &self.layers {
            total = checked_add_u64(total, layer.state_per_token_bytes, CTX)?;
        }
        Ok(total)
    }

    /// Total state memory cost in bytes per sequence across all layers.
    ///
    /// Checked: see [`LayerSummary::total_weight_bytes`].
    pub fn total_state_per_seq_bytes(&self) -> Result<u64, ModelsError> {
        const CTX: &str = "ModelSummary::total_state_per_seq_bytes";
        let mut total = 0u64;
        for layer in &self.layers {
            total = checked_add_u64(total, layer.state_per_seq_bytes, CTX)?;
        }
        Ok(total)
    }
}
