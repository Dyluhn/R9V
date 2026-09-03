// SPDX-License-Identifier: Apache-2.0
//! Tensors and their structural metadata (Spec 1 §2.3–§2.4, §5.1).
//!
//! [`Tensor::new`] enforces the structural rules spec 1 states once, at
//! construction, so the type carries the guarantee afterwards (engineering
//! standards §2.1): bounded non-empty shapes, legal quantization and layout
//! combinations, and `Host`/`Tiered` placement only on `Weight` class.

use std::fmt;

use crate::{IrError, LayoutId, QuantScheme};

/// Symbolic shape vocabulary (Spec 1 §2.4, plus weight dims).
///
/// `T` tokens in the batch, `S` sequences, `Dm` model dim, `Dff` FFN dim, `H`
/// query heads, `Hkv` KV heads, `D` head dim, `E` experts, `K` top-k, `V`
/// vocab, `Np` n-gram hash heads, and `L` layers (Spec 1 §2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeSymbol {
    /// `T`: tokens in the batch, sum of query lengths (Spec 1 §2.4).
    T,
    /// `S`: sequences (Spec 1 §2.4).
    S,
    /// `Dm`: model dim (Spec 1 §2.4).
    Dm,
    /// `Dff`: FFN dim (Spec 1 §2.4).
    Dff,
    /// `H`: query heads (Spec 1 §2.4).
    H,
    /// `Hkv`: KV heads (Spec 1 §2.4).
    Hkv,
    /// `D`: head dim (Spec 1 §2.4).
    D,
    /// `E`: experts (Spec 1 §2.4).
    E,
    /// `K`: top-k (Spec 1 §2.4).
    K,
    /// `V`: vocab (Spec 1 §2.4).
    V,
    /// `Np`: n-gram hash heads (Spec 1 §2.4).
    Np,
    /// `L`: layers (Spec 1 §2.4).
    L,
}

/// One tensor dimension: concrete after capture, symbolic before.
///
/// Kernels see concrete integers per bucket (Spec 1 §2.4); model definitions
/// build graphs over the symbolic form (Spec 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Resolved extent (Spec 1 §2.4: what kernels see per bucket).
    Concrete(u32),
    /// Unresolved shape symbol (Spec 1 §2.4).
    Symbolic(ShapeSymbol),
}

/// Tensor placement (Spec 1 §2.3).
///
/// Placement is resolved at load time by the planner, never baked into model
/// artifacts (D-001). A quant tool's `tiered` hint (Spec 2 §4) is resolved
/// into per-unit placements at load (Spec 5 §3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Device memory on the given rank (Spec 1 §2.3).
    Device {
        /// Rank owning the memory.
        rank: u32,
    },
    /// Pinned host memory: host-computed or host-gathered (Spec 1 §2.3).
    /// Legal only for `Weight` class tensors (Spec 1 §2.3).
    Host,
    /// Slab-backed, fetched by unit (Spec 9 §6). Legal only for `Weight`
    /// class tensors (Spec 1 §2.3).
    Tiered,
}

impl fmt::Display for Placement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Placement::Device { rank } => write!(f, "device({rank})"),
            Placement::Host => write!(f, "host"),
            Placement::Tiered => write!(f, "tiered"),
        }
    }
}

/// Sharding assignment of a tensor (Spec 1 §5.1).
///
/// Each op's legal-layouts table lists legal `(inputs) → output` tuples over
/// these values (Spec 1 §5.2); the partitioner only ever applies that table
/// (Spec 1 §1 principle 4). `Partial` must be resolved by `all_reduce` before
/// any op that requires `Replicated` (Spec 1 §5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardLayout {
    /// Whole tensor on every rank (Spec 1 §5.1).
    Replicated,
    /// Split along output features, Megatron column-parallel (Spec 1 §5.1).
    ColShard {
        /// Sharded axis.
        axis: u32,
    },
    /// Split along input features, row-parallel; consumer output is `Partial`
    /// (Spec 1 §5.1).
    RowShard {
        /// Sharded axis.
        axis: u32,
    },
    /// Attention heads split; KV heads and state split identically
    /// (Spec 1 §5.1).
    HeadShard {
        /// Head count `H`.
        heads: u32,
    },
    /// Experts distributed across ranks (Spec 1 §5.1).
    ExpertShard {
        /// Expert count `E`.
        experts: u32,
    },
    /// Sum across ranks pending; must be resolved by `all_reduce` before any
    /// op that requires `Replicated` (Spec 1 §5.1–§5.2).
    Partial,
}

/// Tensor class (Spec 1 §2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// Model weights (Spec 1 §2.3).
    Weight,
    /// Per-step activations (Spec 1 §2.3).
    Activation,
    /// Sequence state, named via `StateHandle` (Spec 1 §2.3, §2.6).
    State,
    /// Temporary staging, e.g. `gather_staging` (Spec 1 §2.3, §3.2).
    Staging,
    /// Compile-time parameters, e.g. `moe_route` bias (Spec 1 §2.3, §4.C).
    Param,
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Class::Weight => "weight",
            Class::Activation => "activation",
            Class::State => "state",
            Class::Staging => "staging",
            Class::Param => "param",
        };
        write!(f, "{name}")
    }
}

/// An IR tensor: shape, dtype, scheme, layout, placement, sharding, class
/// (Spec 1 §2.3).
///
/// Fields are private: use [`Tensor::new`], which enforces the structural
/// rules, then read through the accessors. Reshapes and transposes are
/// metadata on the edge, not ops (Spec 1 §2.3); that edge metadata is owned
/// by card A1.2, not by this type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tensor {
    shape: Vec<Dim>,
    dtype: crate::DType,
    quant: QuantScheme,
    layout: LayoutId,
    placement: Placement,
    sharding: ShardLayout,
    class: Class,
}

impl Tensor {
    /// Builds a tensor, enforcing the spec 1 §2.3 structural rules:
    /// rank `1..=4`, nonzero concrete extents, class-appropriate quantization
    /// and layouts, and `Host`/`Tiered` only on `Weight` class. Every failure
    /// is reported, not just the first (CONVENTIONS.md §1.4).
    // DECISION(A1.1): rank 0 rejected (every §4 signature has rank ≥ 1) and
    // zero concrete extents rejected; spec 1 §2.3 states only the ≤4 bound.
    // Rejected: accepting scalars/empty dims for later ops to trip over.
    pub fn new(
        shape: Vec<Dim>,
        dtype: crate::DType,
        quant: QuantScheme,
        layout: LayoutId,
        placement: Placement,
        sharding: ShardLayout,
        class: Class,
    ) -> Result<Self, IrError> {
        let mut problems = Vec::new();
        if shape.is_empty() || shape.len() > 4 {
            problems.push(IrError::InvalidRank { got: shape.len() });
        }
        for (axis, dim) in shape.iter().enumerate() {
            if matches!(dim, Dim::Concrete(0)) {
                problems.push(IrError::ZeroExtent { axis });
            }
        }
        match (placement, class) {
            // DECISION(A1.1): enforce the class-level placement rule here and
            // defer semantic weight-role legality to loader/planner binding,
            // where those roles exist; rejected adding an unspecified Tensor
            // field solely for constructor validation. See SI-5.
            (Placement::Host | Placement::Tiered, Class::Weight) => {}
            (Placement::Host | Placement::Tiered, _) => {
                problems.push(IrError::PlacementForClass { placement, class });
            }
            (Placement::Device { .. }, _) => {}
        }
        match (quant, class) {
            (QuantScheme::None, _) => {}
            // DECISION(A1.1): weight-side quantization also applies to
            // Staging because ngram_gather consumes quantized i4/i8 table
            // rows through gather_staging; rejected making that closed op
            // unrepresentable. See SI-6 and Spec 1 §4.A.
            (QuantScheme::PerRow | QuantScheme::Scheme(_), Class::Weight | Class::Staging) => {}
            (QuantScheme::PerToken | QuantScheme::PerBlock32, Class::Activation) => {}
            _ => problems.push(IrError::QuantForClass { quant, class }),
        }
        let invalid_layout = (class == Class::Activation && layout != LayoutId::CONTIGUOUS)
            || (matches!(layout, LayoutId::L1 | LayoutId::L1S) && class != Class::Weight);
        if invalid_layout {
            problems.push(IrError::LayoutForClass { layout, class });
        }
        if dtype == crate::DType::I4 && !matches!(class, Class::Weight | Class::Staging) {
            problems.push(IrError::DTypeForClass { dtype, class });
        }
        let invalid_quant_dtype = match quant {
            QuantScheme::PerToken => !matches!(dtype, crate::DType::I8 | crate::DType::E4m3),
            QuantScheme::PerBlock32 => dtype != crate::DType::I8,
            QuantScheme::PerRow | QuantScheme::Scheme(_) => matches!(
                dtype,
                crate::DType::F32
                    | crate::DType::Bf16
                    | crate::DType::E5m2
                    | crate::DType::I32
                    | crate::DType::U32
                    | crate::DType::Bool
            ),
            QuantScheme::None => dtype == crate::DType::I4,
        };
        if invalid_quant_dtype {
            problems.push(IrError::QuantDType { quant, dtype });
        }
        if problems.is_empty() {
            Ok(Self {
                shape,
                dtype,
                quant,
                layout,
                placement,
                sharding,
                class,
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

    /// Shape dims, symbolic or concrete (Spec 1 §2.3–§2.4).
    pub fn shape(&self) -> &[Dim] {
        &self.shape
    }

    /// Element dtype (Spec 1 §2.1).
    pub fn dtype(&self) -> crate::DType {
        self.dtype
    }

    /// Quantization scheme (Spec 1 §2.2).
    pub fn quant(&self) -> QuantScheme {
        self.quant
    }

    /// Logical layout version (Spec 2 §2; activations use `Contiguous`).
    pub fn layout(&self) -> LayoutId {
        self.layout
    }

    /// Where the tensor lives (Spec 1 §2.3).
    pub fn placement(&self) -> Placement {
        self.placement
    }

    /// Sharding assignment (Spec 1 §5.1).
    pub fn sharding(&self) -> ShardLayout {
        self.sharding
    }

    /// Tensor class (Spec 1 §2.3).
    pub fn class(&self) -> Class {
        self.class
    }

    /// Rank (number of shape dims), always `1..=4` by construction.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}
