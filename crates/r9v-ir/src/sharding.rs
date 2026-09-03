// SPDX-License-Identifier: Apache-2.0
//! Declared sharding rules and layout compatibility tables (Spec 1 §5; card A1.2).
//!
//! Sharding is declared as data, not discovered (Spec 1 §1 Principle 4).
//! Every op carries a legal-layouts table. The partitioner only ever applies
//! this table (Spec 1 §5.2).

use crate::{Op, ShardLayout};

/// Symbolic head count representation in sharding patterns (Spec 1 §5.1; card A1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadCount {
    /// Concrete number of heads.
    Concrete(u32),
    /// Symbolic head count `H` (or `Hkv`), resolved per-model at instantiation.
    Symbolic,
}

/// Symbolic expert count representation in sharding patterns (Spec 1 §5.1; card A1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertCount {
    /// Concrete number of experts.
    Concrete(u32),
    /// Symbolic expert count `E`, resolved per-model at instantiation.
    Symbolic,
}

/// Sharding layout pattern for static op sharding declarations (Spec 1 §5.1; card A1.2).
// DECISION(A1.2): Spec 1 §5.1 defines concrete `ShardLayout::HeadShard { heads: u32 }`
// and `ShardLayout::ExpertShard { experts: u32 }` for materialized tensors, where
// zero extents are forbidden (CONVENTIONS.md §2.1, SI-5). Rather than using magic
// zero (`heads: 0` or `experts: 0`) in static op tables, `ShardLayoutPattern` provides
// typed symbolic variants (`HeadCount::Symbolic`, `ExpertCount::Symbolic`) for static
// op rules. Rejected: magic zeros in concrete ShardLayout which misrepresent invalid
// layouts as valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardLayoutPattern {
    /// Whole tensor on every rank (Spec 1 §5.1).
    Replicated,
    /// Split along output features, Megatron column-parallel (Spec 1 §5.1).
    ColShard {
        /// Sharded axis.
        axis: u32,
    },
    /// Split along input features, row-parallel; consumer output is `Partial` (Spec 1 §5.1).
    RowShard {
        /// Sharded axis.
        axis: u32,
    },
    /// Attention heads split; KV heads and state split identically (Spec 1 §5.1).
    HeadShard {
        /// Head count.
        heads: HeadCount,
    },
    /// Experts distributed across ranks (Spec 1 §5.1).
    ExpertShard {
        /// Expert count.
        experts: ExpertCount,
    },
    /// Sum across ranks pending; must be resolved by `all_reduce` before Replicated (Spec 1 §5.1).
    Partial,
}

impl ShardLayoutPattern {
    /// Matches this pattern against a concrete [`ShardLayout`].
    ///
    /// Per Spec 1 §5.1 and CONVENTIONS.md §2.1 / SI-5, valid concrete layouts
    /// never contain magic zeros (`heads == 0` or `experts == 0`). A concrete
    /// layout with zero extent is rejected and returns `false`.
    pub fn matches(&self, layout: &ShardLayout) -> bool {
        self.matches_layout(layout)
    }

    /// Matches this pattern against a concrete [`ShardLayout`].
    ///
    /// Per Spec 1 §5.1 and CONVENTIONS.md §2.1 / SI-5, valid concrete layouts
    /// never contain magic zeros (`heads == 0` or `experts == 0`). A concrete
    /// layout with zero extent is rejected and returns `false`.
    pub fn matches_layout(&self, layout: &ShardLayout) -> bool {
        match self {
            Self::Replicated => match layout {
                ShardLayout::Replicated => true,
                ShardLayout::ColShard { .. }
                | ShardLayout::RowShard { .. }
                | ShardLayout::HeadShard { .. }
                | ShardLayout::ExpertShard { .. }
                | ShardLayout::Partial => false,
            },
            Self::ColShard { axis: a1 } => match layout {
                ShardLayout::ColShard { axis: a2 } => a1 == a2,
                ShardLayout::Replicated
                | ShardLayout::RowShard { .. }
                | ShardLayout::HeadShard { .. }
                | ShardLayout::ExpertShard { .. }
                | ShardLayout::Partial => false,
            },
            Self::RowShard { axis: a1 } => match layout {
                ShardLayout::RowShard { axis: a2 } => a1 == a2,
                ShardLayout::Replicated
                | ShardLayout::ColShard { .. }
                | ShardLayout::HeadShard { .. }
                | ShardLayout::ExpertShard { .. }
                | ShardLayout::Partial => false,
            },
            Self::HeadShard { heads } => match layout {
                ShardLayout::HeadShard { heads: concrete } => match heads {
                    HeadCount::Concrete(expected) => *concrete > 0 && concrete == expected,
                    HeadCount::Symbolic => *concrete > 0,
                },
                ShardLayout::Replicated
                | ShardLayout::ColShard { .. }
                | ShardLayout::RowShard { .. }
                | ShardLayout::ExpertShard { .. }
                | ShardLayout::Partial => false,
            },
            Self::ExpertShard { experts } => match layout {
                ShardLayout::ExpertShard { experts: concrete } => match experts {
                    ExpertCount::Concrete(expected) => *concrete > 0 && concrete == expected,
                    ExpertCount::Symbolic => *concrete > 0,
                },
                ShardLayout::Replicated
                | ShardLayout::ColShard { .. }
                | ShardLayout::RowShard { .. }
                | ShardLayout::HeadShard { .. }
                | ShardLayout::Partial => false,
            },
            Self::Partial => match layout {
                ShardLayout::Partial => true,
                ShardLayout::Replicated
                | ShardLayout::ColShard { .. }
                | ShardLayout::RowShard { .. }
                | ShardLayout::HeadShard { .. }
                | ShardLayout::ExpertShard { .. } => false,
            },
        }
    }

    /// Instantiates a concrete [`ShardLayout`] from this pattern using resolved symbolic dimensions.
    ///
    /// Returns `None` if zero heads or zero experts are provided, rejecting magic zero per Spec 1 §5.1.
    pub fn to_concrete(&self, heads: u32, experts: u32) -> Option<ShardLayout> {
        match self {
            Self::Replicated => Some(ShardLayout::Replicated),
            Self::ColShard { axis } => Some(ShardLayout::ColShard { axis: *axis }),
            Self::RowShard { axis } => Some(ShardLayout::RowShard { axis: *axis }),
            Self::HeadShard {
                heads: HeadCount::Symbolic,
            } => {
                if heads > 0 {
                    Some(ShardLayout::HeadShard { heads })
                } else {
                    None
                }
            }
            Self::HeadShard {
                heads: HeadCount::Concrete(c),
            } => {
                if *c > 0 {
                    Some(ShardLayout::HeadShard { heads: *c })
                } else {
                    None
                }
            }
            Self::ExpertShard {
                experts: ExpertCount::Symbolic,
            } => {
                if experts > 0 {
                    Some(ShardLayout::ExpertShard { experts })
                } else {
                    None
                }
            }
            Self::ExpertShard {
                experts: ExpertCount::Concrete(c),
            } => {
                if *c > 0 {
                    Some(ShardLayout::ExpertShard { experts: *c })
                } else {
                    None
                }
            }
            Self::Partial => Some(ShardLayout::Partial),
        }
    }
}

/// A legal sharding rule: input layouts and corresponding output layout(s) (Spec 1 §5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShardingRule {
    /// Required sharding layout for each input tensor in signature order.
    pub inputs: &'static [ShardLayoutPattern],
    /// Resulting sharding layout for each output tensor in signature order.
    pub outputs: &'static [ShardLayoutPattern],
}

impl ShardingRule {
    /// Creates a new sharding rule with the given input and output layout slices.
    pub const fn new(
        inputs: &'static [ShardLayoutPattern],
        outputs: &'static [ShardLayoutPattern],
    ) -> Self {
        Self { inputs, outputs }
    }

    /// Input layout requirements.
    pub const fn inputs(&self) -> &'static [ShardLayoutPattern] {
        self.inputs
    }

    /// Output layout results.
    pub const fn outputs(&self) -> &'static [ShardLayoutPattern] {
        self.outputs
    }

    /// Primary output layout, or `None` if the op has no output tensors.
    pub const fn output(&self) -> Option<ShardLayoutPattern> {
        if self.outputs.is_empty() {
            None
        } else {
            Some(self.outputs[0])
        }
    }

    /// Returns the rule as an `(inputs, outputs)` tuple.
    pub const fn as_tuple(&self) -> (&'static [ShardLayoutPattern], &'static [ShardLayoutPattern]) {
        (self.inputs, self.outputs)
    }

    /// Returns whether the provided concrete input and output layouts match this rule.
    ///
    /// Validates exact input/output arity and binds symbolic head and expert
    /// cardinalities consistently across all inputs and outputs (Spec 1 §5.1–§5.2).
    pub fn matches(&self, inputs: &[ShardLayout], outputs: &[ShardLayout]) -> bool {
        if self.inputs.len() != inputs.len() || self.outputs.len() != outputs.len() {
            return false;
        }
        let mut bound_heads: Option<u32> = None;
        let mut bound_experts: Option<u32> = None;

        for (pat, layout) in self.inputs.iter().zip(inputs) {
            if !match_pattern_with_bindings(pat, layout, &mut bound_heads, &mut bound_experts) {
                return false;
            }
        }
        for (pat, layout) in self.outputs.iter().zip(outputs) {
            if !match_pattern_with_bindings(pat, layout, &mut bound_heads, &mut bound_experts) {
                return false;
            }
        }
        true
    }

    /// Returns whether the provided input and output tensors match this rule's layouts.
    ///
    /// Validates exact arity, verifies that actual tensor sharding is structurally valid
    /// (sharded axis within tensor rank, non-zero head/expert counts), and ensures that
    /// symbolic head and expert cardinalities match consistently across all tensors (Spec 1 §5.1–§5.2).
    // DECISION(A1.2): matches_tensors validates exact rule arity, concrete sharded axis within tensor rank, non-zero head/expert extents, and binds symbolic head and expert cardinalities consistently across all rule tensors per Spec 1 §5.1-§5.2; rejected independent unverified wildcard matching across tensors.
    pub fn matches_tensors(&self, inputs: &[crate::Tensor], outputs: &[crate::Tensor]) -> bool {
        if self.inputs.len() != inputs.len() || self.outputs.len() != outputs.len() {
            return false;
        }
        for t in inputs.iter().chain(outputs.iter()) {
            match t.sharding() {
                ShardLayout::ColShard { axis } | ShardLayout::RowShard { axis } => {
                    if axis as usize >= t.rank() {
                        return false;
                    }
                }
                ShardLayout::HeadShard { heads } => {
                    if heads == 0 {
                        return false;
                    }
                }
                ShardLayout::ExpertShard { experts } => {
                    if experts == 0 {
                        return false;
                    }
                }
                ShardLayout::Replicated | ShardLayout::Partial => {}
            }
        }
        let mut bound_heads: Option<u32> = None;
        let mut bound_experts: Option<u32> = None;

        for (pat, t) in self.inputs.iter().zip(inputs) {
            if !match_pattern_with_bindings(
                pat,
                &t.sharding(),
                &mut bound_heads,
                &mut bound_experts,
            ) {
                return false;
            }
        }
        for (pat, t) in self.outputs.iter().zip(outputs) {
            if !match_pattern_with_bindings(
                pat,
                &t.sharding(),
                &mut bound_heads,
                &mut bound_experts,
            ) {
                return false;
            }
        }
        true
    }
}

/// Helper matching a layout pattern against a concrete layout, binding symbolic counts.
///
/// Exhaustive matching over all variants without wildcards (CONVENTIONS.md §3.2).
fn match_pattern_with_bindings(
    pattern: &ShardLayoutPattern,
    layout: &ShardLayout,
    bound_heads: &mut Option<u32>,
    bound_experts: &mut Option<u32>,
) -> bool {
    match pattern {
        ShardLayoutPattern::Replicated => match layout {
            ShardLayout::Replicated => true,
            ShardLayout::ColShard { .. }
            | ShardLayout::RowShard { .. }
            | ShardLayout::HeadShard { .. }
            | ShardLayout::ExpertShard { .. }
            | ShardLayout::Partial => false,
        },
        ShardLayoutPattern::ColShard { axis: a1 } => match layout {
            ShardLayout::ColShard { axis: a2 } => a1 == a2,
            ShardLayout::Replicated
            | ShardLayout::RowShard { .. }
            | ShardLayout::HeadShard { .. }
            | ShardLayout::ExpertShard { .. }
            | ShardLayout::Partial => false,
        },
        ShardLayoutPattern::RowShard { axis: a1 } => match layout {
            ShardLayout::RowShard { axis: a2 } => a1 == a2,
            ShardLayout::Replicated
            | ShardLayout::ColShard { .. }
            | ShardLayout::HeadShard { .. }
            | ShardLayout::ExpertShard { .. }
            | ShardLayout::Partial => false,
        },
        ShardLayoutPattern::HeadShard { heads } => match layout {
            ShardLayout::HeadShard { heads: concrete } => {
                if *concrete == 0 {
                    return false;
                }
                match heads {
                    HeadCount::Concrete(expected) => concrete == expected,
                    HeadCount::Symbolic => match bound_heads {
                        Some(existing) => *existing == *concrete,
                        None => {
                            *bound_heads = Some(*concrete);
                            true
                        }
                    },
                }
            }
            ShardLayout::Replicated
            | ShardLayout::ColShard { .. }
            | ShardLayout::RowShard { .. }
            | ShardLayout::ExpertShard { .. }
            | ShardLayout::Partial => false,
        },
        ShardLayoutPattern::ExpertShard { experts } => match layout {
            ShardLayout::ExpertShard { experts: concrete } => {
                if *concrete == 0 {
                    return false;
                }
                match experts {
                    ExpertCount::Concrete(expected) => concrete == expected,
                    ExpertCount::Symbolic => match bound_experts {
                        Some(existing) => *existing == *concrete,
                        None => {
                            *bound_experts = Some(*concrete);
                            true
                        }
                    },
                }
            }
            ShardLayout::Replicated
            | ShardLayout::ColShard { .. }
            | ShardLayout::RowShard { .. }
            | ShardLayout::HeadShard { .. }
            | ShardLayout::Partial => false,
        },
        ShardLayoutPattern::Partial => match layout {
            ShardLayout::Partial => true,
            ShardLayout::Replicated
            | ShardLayout::ColShard { .. }
            | ShardLayout::RowShard { .. }
            | ShardLayout::HeadShard { .. }
            | ShardLayout::ExpertShard { .. } => false,
        },
    }
}

// -----------------------------------------------------------------------------
// Static layout rules per op (Spec 1 §4, §5)
// -----------------------------------------------------------------------------

// DECISION(A1.2): EMBED_GATHER_RULES covers Replicated and RowShard(0) vocab sharding per Spec 1 §4.A; rejected invented ColShard(1) embed rule which has no spec basis.
pub static EMBED_GATHER_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::RowShard { axis: 0 },
        ],
        &[ShardLayoutPattern::Partial],
    ),
];

// DECISION(A1.2): NGRAM_GATHER_RULES matches op tensor validation arity (2 inputs: staging/token_ids, row_scales/table; 1 output: x) per Spec 1 §4.A & SI-8; rejected stale 1-input rules.
pub static NGRAM_GATHER_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::RowShard { axis: 1 },
            ShardLayoutPattern::RowShard { axis: 1 },
        ],
        &[ShardLayoutPattern::RowShard { axis: 1 }],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::RowShard { axis: 1 },
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::RowShard { axis: 1 }],
    ),
];

// DECISION(A1.2): QUANT_ACT_RULES matches op tensor validation arity (1 input: x; 2 outputs: xq, scale) per Spec 1 §4.A & SI-7; rejected RowShard(0) batch-axis invention.
pub static QUANT_ACT_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[ShardLayoutPattern::Replicated],
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::ColShard { axis: 1 }],
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::Replicated,
        ],
    ),
];

// DECISION(A1.2): PASSTHROUGH_RULES excludes Partial per Spec 1 §5.2 (Partial may flow only through residual_add and matmul epilogues); rejected generic passthrough of unresolved partial sums.
pub static PASSTHROUGH_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[ShardLayoutPattern::Replicated],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::ColShard { axis: 0 }],
        &[ShardLayoutPattern::ColShard { axis: 0 }],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::ColShard { axis: 1 }],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::RowShard { axis: 0 }],
        &[ShardLayoutPattern::RowShard { axis: 0 }],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::RowShard { axis: 1 }],
        &[ShardLayoutPattern::RowShard { axis: 1 }],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Symbolic,
        }],
        &[ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Symbolic,
        }],
    ),
];

pub static GATHER_ROWS_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
];

pub static SCATTER_ADD_ROWS_RULES: &[ShardingRule] = &[
    // x, indices -> y
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
    // x, indices, dest -> y
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::ColShard { axis: 1 },
        ],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
];

// DECISION(A1.2): NORM_RULES enforces replicated normalized axis per Spec 1 §4.B; rejected RowShard(0) batch-axis invention.
pub static NORM_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
];

// DECISION(A1.2): RESIDUAL_ADD_RULES covers Spec 1 §5.2 Partial sum flow and feature sharding; rejected RowShard(0) batch-axis invention.
pub static RESIDUAL_ADD_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::Partial, ShardLayoutPattern::Replicated],
        &[ShardLayoutPattern::Partial],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::Replicated, ShardLayoutPattern::Partial],
        &[ShardLayoutPattern::Partial],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::Partial, ShardLayoutPattern::Partial],
        &[ShardLayoutPattern::Partial],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::ColShard { axis: 1 },
        ],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::RowShard { axis: 1 },
            ShardLayoutPattern::RowShard { axis: 1 },
        ],
        &[ShardLayoutPattern::RowShard { axis: 1 }],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
        ],
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
    ),
];

// DECISION(A1.2): ACT_MUL_RULES matches feature-axis sharding per Spec 1 §4.B; rejected RowShard(0) batch-axis invention.
pub static ACT_MUL_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::ColShard { axis: 1 },
        ],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
];

// DECISION(A1.2): ACTIVATION_RULES matches elementwise feature-axis sharding per Spec 1 §4.B; rejected RowShard(0) batch-axis invention.
pub static ACTIVATION_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[ShardLayoutPattern::Replicated],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::ColShard { axis: 1 }],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
];

// DECISION(A1.2): ROPE_RULES covers Replicated and HeadShard per Spec 1 §4.B; rejected RowShard(0) batch-axis invention.
pub static ROPE_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
    ),
];

// DECISION(A1.2): MATMUL_RULES contains exactly the three §4.C principal x/w/y rows; rejected invented data-parallel RowShard(0), bias 3-input rules, and residual 3-input variants.
pub static MATMUL_RULES: &[ShardingRule] = &[
    // 1. Column parallel: x: Replicated, w: ColShard(0) -> y: ColShard(1)
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::ColShard { axis: 0 },
        ],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
    // 2. Row parallel: x: ColShard(1), w: RowShard(1) -> y: Partial
    ShardingRule::new(
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::RowShard { axis: 1 },
        ],
        &[ShardLayoutPattern::Partial],
    ),
    // 3. Fully replicated
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
];

pub static MOE_ROUTE_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[ShardLayoutPattern::Replicated],
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
    ),
];

pub static MOE_FFN_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::ExpertShard {
                experts: ExpertCount::Symbolic,
            },
            ShardLayoutPattern::ExpertShard {
                experts: ExpertCount::Symbolic,
            },
        ],
        &[ShardLayoutPattern::Partial],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
];

pub static STATE_WRITE_KV_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
        ],
        &[],
    ),
];

pub static ATTENTION_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[ShardLayoutPattern::Replicated],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
    ),
];

pub static CAUSAL_CONV1D_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::ColShard { axis: 0 },
        ],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::ColShard { axis: 1 },
            ShardLayoutPattern::ColShard { axis: 0 },
            ShardLayoutPattern::ColShard { axis: 0 },
        ],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
];

pub static LINEAR_ATTN_SCAN_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
            ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            },
        ],
        &[ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        }],
    ),
];

pub static LOGITS_POSTPROCESS_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[ShardLayoutPattern::Replicated],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
];

// DECISION(A1.2): SAMPLE_RULES matches op tensor validation arity (1 input: probs; 1 output: token) per Spec 1 §4.F & SI-12; rejected stale 2/2 rule.
pub static SAMPLE_RULES: &[ShardingRule] = &[ShardingRule::new(
    &[ShardLayoutPattern::Replicated],
    &[ShardLayoutPattern::Replicated],
)];

// DECISION(A1.2): VERIFY_RULES matches op tensor validation arity (2 or 3 inputs; 2 outputs: accepted, accept_len) per Spec 1 §4.F & SI-12; rejected stale 4/3 rule.
pub static VERIFY_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
    ),
];

pub static ALL_REDUCE_RULES: &[ShardingRule] = &[ShardingRule::new(
    &[ShardLayoutPattern::Partial],
    &[ShardLayoutPattern::Replicated],
)];

pub static ALL_GATHER_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[ShardLayoutPattern::ColShard { axis: 0 }],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::ColShard { axis: 1 }],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::RowShard { axis: 0 }],
        &[ShardLayoutPattern::Replicated],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::RowShard { axis: 1 }],
        &[ShardLayoutPattern::Replicated],
    ),
];

pub static REDUCE_SCATTER_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[ShardLayoutPattern::Partial],
        &[ShardLayoutPattern::RowShard { axis: 0 }],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::Partial],
        &[ShardLayoutPattern::RowShard { axis: 1 }],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::Partial],
        &[ShardLayoutPattern::ColShard { axis: 0 }],
    ),
    ShardingRule::new(
        &[ShardLayoutPattern::Partial],
        &[ShardLayoutPattern::ColShard { axis: 1 }],
    ),
];

// DECISION(A1.2): ALL_TO_ALL_RULES covers exactly inputs (x, counts) to y per Spec 1 §4.G and SI-11; rejected stale 1-input or 2-output rows.
pub static ALL_TO_ALL_RULES: &[ShardingRule] = &[
    ShardingRule::new(
        &[
            ShardLayoutPattern::ExpertShard {
                experts: ExpertCount::Symbolic,
            },
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Symbolic,
        }],
    ),
    ShardingRule::new(
        &[
            ShardLayoutPattern::Replicated,
            ShardLayoutPattern::Replicated,
        ],
        &[ShardLayoutPattern::Replicated],
    ),
];

// DECISION(A1.2): SEND_RULES and RECV_RULES exclude Partial per Spec 1 §5.2 (Partial resolved before pipeline boundaries); rejected sending unresolved partial sums across stages.
pub static SEND_RULES: &[ShardingRule] = &[
    ShardingRule::new(&[ShardLayoutPattern::Replicated], &[]),
    ShardingRule::new(&[ShardLayoutPattern::ColShard { axis: 0 }], &[]),
    ShardingRule::new(&[ShardLayoutPattern::ColShard { axis: 1 }], &[]),
];

pub static RECV_RULES: &[ShardingRule] = &[
    ShardingRule::new(&[], &[ShardLayoutPattern::Replicated]),
    ShardingRule::new(&[], &[ShardLayoutPattern::ColShard { axis: 0 }]),
    ShardingRule::new(&[], &[ShardLayoutPattern::ColShard { axis: 1 }]),
];

pub static BARRIER_RULES: &[ShardingRule] = &[ShardingRule::new(&[], &[])];

// DECISION(A1.14): split/concat/softcap ship a single Replicated rule each:
// every A1.14 producer binds Replicated tensors, and head/expert sharding of
// MLA channel ranges is a future partitioner card's table entry, not this
// card's. Rejected inventing HeadShard channel-range rules the partitioner
// has no lowering for. SI-20.
pub static SPLIT_RULES: &[ShardingRule] = &[ShardingRule::new(
    &[ShardLayoutPattern::Replicated],
    &[
        ShardLayoutPattern::Replicated,
        ShardLayoutPattern::Replicated,
    ],
)];

pub static CONCAT_RULES: &[ShardingRule] = &[ShardingRule::new(
    &[
        ShardLayoutPattern::Replicated,
        ShardLayoutPattern::Replicated,
    ],
    &[ShardLayoutPattern::Replicated],
)];

pub static LOGIT_SOFTCAP_RULES: &[ShardingRule] = &[ShardingRule::new(
    &[ShardLayoutPattern::Replicated],
    &[ShardLayoutPattern::Replicated],
)];

/// Returns the legal input/output sharding layout rules for the given op (Spec 1 §5.2).
///
/// Exhaustive matching over all 32 closed ops without wildcards (CONVENTIONS.md §3.2).
pub fn legal_layouts(op: &Op) -> &'static [ShardingRule] {
    match op {
        Op::EmbedGather(_) => EMBED_GATHER_RULES,
        Op::NgramGather(_) => NGRAM_GATHER_RULES,
        Op::QuantAct(_) => QUANT_ACT_RULES,
        Op::Cast(_) => PASSTHROUGH_RULES,
        Op::Copy(_) => PASSTHROUGH_RULES,
        Op::GatherRows(_) => GATHER_ROWS_RULES,
        Op::ScatterAddRows(_) => SCATTER_ADD_ROWS_RULES,
        Op::Split(_) => SPLIT_RULES,
        Op::Concat(_) => CONCAT_RULES,
        Op::Norm(_) => NORM_RULES,
        Op::ResidualAdd(_) => RESIDUAL_ADD_RULES,
        Op::ActMul(_) => ACT_MUL_RULES,
        Op::Activation(_) => ACTIVATION_RULES,
        Op::LogitSoftcap(_) => LOGIT_SOFTCAP_RULES,
        Op::Rope(_) => ROPE_RULES,
        Op::Matmul(_) => MATMUL_RULES,
        Op::MoeRoute(_) => MOE_ROUTE_RULES,
        Op::MoeFfn(_) => MOE_FFN_RULES,
        Op::StateWriteKv(_) => STATE_WRITE_KV_RULES,
        Op::Attention(_) => ATTENTION_RULES,
        Op::CausalConv1d(_) => CAUSAL_CONV1D_RULES,
        Op::LinearAttnScan(_) => LINEAR_ATTN_SCAN_RULES,
        Op::LogitsPostprocess(_) => LOGITS_POSTPROCESS_RULES,
        Op::Sample(_) => SAMPLE_RULES,
        Op::Verify(_) => VERIFY_RULES,
        Op::AllReduce(_) => ALL_REDUCE_RULES,
        Op::AllGather(_) => ALL_GATHER_RULES,
        Op::ReduceScatter(_) => REDUCE_SCATTER_RULES,
        Op::AllToAll(_) => ALL_TO_ALL_RULES,
        Op::Send(_) => SEND_RULES,
        Op::Recv(_) => RECV_RULES,
        Op::Barrier(_) => BARRIER_RULES,
    }
}

/// Returns legal sharding rules formatted as tuples `(inputs, outputs)`.
pub fn legal_layout_tuples(
    op: &Op,
) -> Vec<(&'static [ShardLayoutPattern], &'static [ShardLayoutPattern])> {
    legal_layouts(op).iter().map(|r| r.as_tuple()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_matches_concrete_without_magic_zero() {
        // Replicated
        assert!(ShardLayoutPattern::Replicated.matches(&ShardLayout::Replicated));
        assert!(!ShardLayoutPattern::Replicated.matches(&ShardLayout::Partial));

        // ColShard
        let pat_col = ShardLayoutPattern::ColShard { axis: 1 };
        assert!(pat_col.matches(&ShardLayout::ColShard { axis: 1 }));
        assert!(!pat_col.matches(&ShardLayout::ColShard { axis: 0 }));

        // RowShard
        let pat_row = ShardLayoutPattern::RowShard { axis: 0 };
        assert!(pat_row.matches(&ShardLayout::RowShard { axis: 0 }));
        assert!(!pat_row.matches(&ShardLayout::RowShard { axis: 1 }));

        // HeadShard concrete
        let pat_head_c = ShardLayoutPattern::HeadShard {
            heads: HeadCount::Concrete(8),
        };
        assert!(pat_head_c.matches(&ShardLayout::HeadShard { heads: 8 }));
        assert!(!pat_head_c.matches(&ShardLayout::HeadShard { heads: 4 }));
        // Magic zero rejected
        assert!(!pat_head_c.matches(&ShardLayout::HeadShard { heads: 0 }));

        // HeadShard symbolic
        let pat_head_s = ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        };
        assert!(pat_head_s.matches(&ShardLayout::HeadShard { heads: 8 }));
        assert!(pat_head_s.matches(&ShardLayout::HeadShard { heads: 32 }));
        // Magic zero rejected
        assert!(!pat_head_s.matches(&ShardLayout::HeadShard { heads: 0 }));

        // ExpertShard concrete
        let pat_exp_c = ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Concrete(16),
        };
        assert!(pat_exp_c.matches(&ShardLayout::ExpertShard { experts: 16 }));
        assert!(!pat_exp_c.matches(&ShardLayout::ExpertShard { experts: 8 }));
        // Magic zero rejected
        assert!(!pat_exp_c.matches(&ShardLayout::ExpertShard { experts: 0 }));

        // ExpertShard symbolic
        let pat_exp_s = ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Symbolic,
        };
        assert!(pat_exp_s.matches(&ShardLayout::ExpertShard { experts: 16 }));
        assert!(pat_exp_s.matches(&ShardLayout::ExpertShard { experts: 64 }));
        // Magic zero rejected
        assert!(!pat_exp_s.matches(&ShardLayout::ExpertShard { experts: 0 }));

        // Partial
        assert!(ShardLayoutPattern::Partial.matches(&ShardLayout::Partial));
        assert!(!ShardLayoutPattern::Partial.matches(&ShardLayout::Replicated));
    }

    #[test]
    fn pattern_to_concrete_instantiation_rejects_zero() {
        let pat_head_s = ShardLayoutPattern::HeadShard {
            heads: HeadCount::Symbolic,
        };
        assert_eq!(
            pat_head_s.to_concrete(8, 16),
            Some(ShardLayout::HeadShard { heads: 8 })
        );
        assert_eq!(pat_head_s.to_concrete(0, 16), None);

        let pat_exp_s = ShardLayoutPattern::ExpertShard {
            experts: ExpertCount::Symbolic,
        };
        assert_eq!(
            pat_exp_s.to_concrete(8, 16),
            Some(ShardLayout::ExpertShard { experts: 16 })
        );
        assert_eq!(pat_exp_s.to_concrete(8, 0), None);
    }

    #[test]
    fn matmul_rules_has_exactly_three_principal_rows() {
        assert_eq!(MATMUL_RULES.len(), 3);
        assert_eq!(
            MATMUL_RULES[0].inputs,
            &[
                ShardLayoutPattern::Replicated,
                ShardLayoutPattern::ColShard { axis: 0 }
            ]
        );
        assert_eq!(
            MATMUL_RULES[0].outputs,
            &[ShardLayoutPattern::ColShard { axis: 1 }]
        );

        assert_eq!(
            MATMUL_RULES[1].inputs,
            &[
                ShardLayoutPattern::ColShard { axis: 1 },
                ShardLayoutPattern::RowShard { axis: 1 }
            ]
        );
        assert_eq!(MATMUL_RULES[1].outputs, &[ShardLayoutPattern::Partial]);

        assert_eq!(
            MATMUL_RULES[2].inputs,
            &[
                ShardLayoutPattern::Replicated,
                ShardLayoutPattern::Replicated
            ]
        );
        assert_eq!(MATMUL_RULES[2].outputs, &[ShardLayoutPattern::Replicated]);
    }

    #[test]
    fn embed_gather_rules_has_no_invented_colshard() {
        assert_eq!(EMBED_GATHER_RULES.len(), 2);
        assert!(!EMBED_GATHER_RULES.iter().any(|r| {
            r.inputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::ColShard { .. }))
        }));
    }

    #[test]
    fn ngram_gather_rules_has_no_one_input_rules() {
        assert_eq!(NGRAM_GATHER_RULES.len(), 3);
        for rule in NGRAM_GATHER_RULES {
            assert_eq!(rule.inputs.len(), 2);
            assert_eq!(rule.outputs.len(), 1);
        }
    }

    #[test]
    fn quant_act_rules_has_no_batch_axis_rowshard() {
        assert_eq!(QUANT_ACT_RULES.len(), 2);
        for rule in QUANT_ACT_RULES {
            assert!(!rule
                .inputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::RowShard { axis: 0 })));
            assert_eq!(rule.inputs.len(), 1);
            assert_eq!(rule.outputs.len(), 2);
        }
    }

    #[test]
    fn sample_rules_matches_op_tensor_arity() {
        assert_eq!(SAMPLE_RULES.len(), 1);
        assert_eq!(SAMPLE_RULES[0].inputs.len(), 1);
        assert_eq!(SAMPLE_RULES[0].outputs.len(), 1);
    }

    #[test]
    fn verify_rules_has_no_stale_four_input_rules() {
        assert_eq!(VERIFY_RULES.len(), 2);
        for rule in VERIFY_RULES {
            assert!(rule.inputs.len() == 2 || rule.inputs.len() == 3);
            assert_eq!(rule.outputs.len(), 2);
        }
    }

    #[test]
    fn all_to_all_rules_has_exact_two_inputs_one_output() {
        assert_eq!(ALL_TO_ALL_RULES.len(), 2);
        for rule in ALL_TO_ALL_RULES {
            assert_eq!(rule.inputs.len(), 2);
            assert_eq!(rule.outputs.len(), 1);
        }
        assert_eq!(
            ALL_TO_ALL_RULES[0].inputs,
            &[
                ShardLayoutPattern::ExpertShard {
                    experts: ExpertCount::Symbolic,
                },
                ShardLayoutPattern::Replicated,
            ]
        );
        assert_eq!(
            ALL_TO_ALL_RULES[0].outputs,
            &[ShardLayoutPattern::ExpertShard {
                experts: ExpertCount::Symbolic,
            }]
        );
        assert_eq!(
            ALL_TO_ALL_RULES[1].inputs,
            &[
                ShardLayoutPattern::Replicated,
                ShardLayoutPattern::Replicated,
            ]
        );
        assert_eq!(
            ALL_TO_ALL_RULES[1].outputs,
            &[ShardLayoutPattern::Replicated]
        );
    }

    #[test]
    fn passthrough_and_send_recv_rules_exclude_partial() {
        for rule in PASSTHROUGH_RULES {
            assert!(!rule
                .inputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::Partial)));
            assert!(!rule
                .outputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::Partial)));
        }
        for rule in SEND_RULES {
            assert!(!rule
                .inputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::Partial)));
        }
        for rule in RECV_RULES {
            assert!(!rule
                .outputs
                .iter()
                .any(|p| matches!(p, ShardLayoutPattern::Partial)));
        }
    }

    #[test]
    fn rule_matches_binds_symbolic_head_and_expert_cardinalities() {
        // HeadShard symbolic binding
        let head_rule = ShardingRule::new(
            &[ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            }],
            &[ShardLayoutPattern::HeadShard {
                heads: HeadCount::Symbolic,
            }],
        );
        // Matching head counts: 8 -> 8 matches
        assert!(head_rule.matches(
            &[ShardLayout::HeadShard { heads: 8 }],
            &[ShardLayout::HeadShard { heads: 8 }]
        ));
        // Mismatched head counts: 8 -> 4 rejected
        assert!(!head_rule.matches(
            &[ShardLayout::HeadShard { heads: 8 }],
            &[ShardLayout::HeadShard { heads: 4 }]
        ));

        // ExpertShard symbolic binding
        let exp_rule = ALL_TO_ALL_RULES[0];
        // Matching expert counts: 16 -> 16 matches
        assert!(exp_rule.matches(
            &[
                ShardLayout::ExpertShard { experts: 16 },
                ShardLayout::Replicated
            ],
            &[ShardLayout::ExpertShard { experts: 16 }]
        ));
        // Mismatched expert counts: 16 -> 8 rejected
        assert!(!exp_rule.matches(
            &[
                ShardLayout::ExpertShard { experts: 16 },
                ShardLayout::Replicated
            ],
            &[ShardLayout::ExpertShard { experts: 8 }]
        ));
    }
}
