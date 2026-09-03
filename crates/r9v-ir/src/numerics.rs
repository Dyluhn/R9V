// SPDX-License-Identifier: Apache-2.0
//! Op numerics contract (Spec 1 §6; card A1.2).
//!
//! Every op defines its numerics contract: accumulation dtype (`f32` or `i32`,
//! never `f16`/`bf16`, Spec 1 §6.1) and a deterministic reduction order tag.

use crate::{DType, IrError, QuantScheme};

/// Deterministic reduction order tag (Spec 1 §6.1).
///
/// Batch invariance and cross-tier reproducibility require reductions to follow
/// a fixed, documented order (Spec 1 §6.1, engineering standards §2.5–§2.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOrder {
    /// No reduction performed by this op.
    None,
    /// Ascending K feature dimension order (e.g. full-K matmul accumulate).
    AscendingK,
    /// Ascending block index order (e.g. PerBlock32 matmul, attention KV blocks).
    AscendingBlock,
    /// Ascending feature axis order (e.g. norm mean and variance reductions).
    AscendingAxis,
    /// Ascending device rank order (e.g. all-reduce, reduce-scatter collectives).
    AscendingRank,
    /// Ascending token or expert index order (e.g. sampling CDF, MoE combine,
    /// scatter-add rows).
    AscendingIndex,
}

/// Numerics contract per op (Spec 1 §6; card A1.2).
///
/// Encapsulates the accumulator dtype (must be `f32` or `i32` if an accumulator
/// is used, Spec 1 §6.1) and the reduction order tag. Used by op-level test
/// harnesses (card A1.10) to verify determinism and precision guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Numerics {
    /// Accumulator dtype, or `None` if the op performs no accumulation.
    pub accumulator: Option<DType>,
    /// Tag indicating the deterministic reduction order.
    pub reduction_order: ReductionOrder,
}

impl Numerics {
    /// Creates a numerics contract, validating that any declared accumulator
    /// is either `f32` or `i32` per Spec 1 §6.1.
    pub fn new(
        accumulator: Option<DType>,
        reduction_order: ReductionOrder,
    ) -> Result<Self, IrError> {
        if let Some(acc) = accumulator {
            if acc != DType::F32 && acc != DType::I32 {
                return Err(IrError::InvalidAccumulator { got: acc });
            }
        }
        Ok(Self {
            accumulator,
            reduction_order,
        })
    }

    /// Helper for `f32` accumulation with the given reduction order.
    pub const fn f32(reduction_order: ReductionOrder) -> Self {
        Self {
            accumulator: Some(DType::F32),
            reduction_order,
        }
    }

    /// Helper for `i32` accumulation with the given reduction order.
    pub const fn i32(reduction_order: ReductionOrder) -> Self {
        Self {
            accumulator: Some(DType::I32),
            reduction_order,
        }
    }

    /// Helper for ops that perform no reduction or accumulation.
    pub const fn none() -> Self {
        Self {
            accumulator: None,
            reduction_order: ReductionOrder::None,
        }
    }

    /// Computes the input-dependent numerics contract for a matmul operation (Spec 1 §4.C, §6.2).
    pub fn for_matmul(
        x_dtype: DType,
        w_dtype: DType,
        x_quant: QuantScheme,
        w_quant: QuantScheme,
    ) -> Result<Self, IrError> {
        matmul_numerics(x_dtype, w_dtype, x_quant, w_quant)
    }

    /// Computes the input-dependent numerics contract for MoE FFN GEMM (Spec 1 §4.C, §6.2).
    pub fn for_moe_ffn(
        x_dtype: DType,
        w_dtype: DType,
        x_quant: QuantScheme,
        w_quant: QuantScheme,
    ) -> Result<Self, IrError> {
        moe_ffn_gemm_numerics(x_dtype, w_dtype, x_quant, w_quant)
    }
}

/// Returns the input-dependent numerics contract for matmul operations (Spec 1 §4.C, §6.1, §6.2).
///
/// Integer activation/weight pairs use `i32`; mixed or floating-point pairs
/// dequantize/convert before `f32` accumulation. True block-scaled operands use
/// ascending block order; folded per-row scales use full-K ascending order.
// DECISION(A1.2): matmul_numerics accepts both x_quant and w_quant per Spec 1 §6.2; rejected omitting w_quant which makes distinguishing folded per-row vs true per-block scales impossible.
pub fn matmul_numerics(
    x_dtype: DType,
    w_dtype: DType,
    x_quant: QuantScheme,
    w_quant: QuantScheme,
) -> Result<Numerics, IrError> {
    let valid_x_quant = match x_dtype {
        DType::F16 | DType::Bf16 => x_quant == QuantScheme::None,
        DType::I8 => matches!(x_quant, QuantScheme::PerToken | QuantScheme::PerBlock32),
        DType::E4m3 => x_quant == QuantScheme::PerToken,
        DType::F32 | DType::E5m2 | DType::I4 | DType::I32 | DType::U32 | DType::Bool => {
            return Err(IrError::OpDTypeMismatch {
                op: "matmul",
                tensor: "x",
                expected: vec![DType::F16, DType::Bf16, DType::I8, DType::E4m3].into_boxed_slice(),
                got: x_dtype,
            });
        }
    };
    if !valid_x_quant {
        return Err(IrError::OpQuantMismatch {
            op: "matmul",
            tensor: "x",
            quant: x_quant,
        });
    }

    let valid_w_quant = match w_dtype {
        DType::I4 | DType::I8 | DType::E4m3 => {
            matches!(w_quant, QuantScheme::PerRow | QuantScheme::Scheme(_))
        }
        DType::F16 => w_quant == QuantScheme::None,
        DType::F32 | DType::Bf16 | DType::E5m2 | DType::I32 | DType::U32 | DType::Bool => {
            return Err(IrError::OpDTypeMismatch {
                op: "matmul",
                tensor: "w",
                expected: vec![DType::I4, DType::I8, DType::E4m3, DType::F16].into_boxed_slice(),
                got: w_dtype,
            });
        }
    };
    if !valid_w_quant {
        return Err(IrError::OpQuantMismatch {
            op: "matmul",
            tensor: "w",
            quant: w_quant,
        });
    }

    let reduction_order =
        if x_quant == QuantScheme::PerBlock32 || matches!(w_quant, QuantScheme::Scheme(_)) {
            ReductionOrder::AscendingBlock
        } else {
            ReductionOrder::AscendingK
        };
    if x_dtype == DType::I8 && matches!(w_dtype, DType::I4 | DType::I8) {
        Ok(Numerics::i32(reduction_order))
    } else {
        Ok(Numerics::f32(reduction_order))
    }
}

/// Computes the input-dependent numerics contract for MoE FFN GEMM (Spec 1 §4.C, §6.2).
pub fn moe_ffn_gemm_numerics(
    x_dtype: DType,
    w_dtype: DType,
    x_quant: QuantScheme,
    w_quant: QuantScheme,
) -> Result<Numerics, IrError> {
    matmul_numerics(x_dtype, w_dtype, x_quant, w_quant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SchemeId;

    #[test]
    fn matmul_numerics_expresses_exact_spec6_rules() {
        // i8 PerToken x PerRow -> i32 AscendingK (full-K)
        let n_i8_token = matmul_numerics(
            DType::I8,
            DType::I8,
            QuantScheme::PerToken,
            QuantScheme::PerRow,
        )
        .unwrap();
        assert_eq!(n_i8_token.accumulator, Some(DType::I32));
        assert_eq!(n_i8_token.reduction_order, ReductionOrder::AscendingK);

        // i8 PerToken x Scheme(true per-block) -> i32 AscendingBlock
        let n_i8_true_block = matmul_numerics(
            DType::I8,
            DType::I8,
            QuantScheme::PerToken,
            QuantScheme::Scheme(SchemeId::new(1)),
        )
        .unwrap();
        assert_eq!(n_i8_true_block.accumulator, Some(DType::I32));
        assert_eq!(
            n_i8_true_block.reduction_order,
            ReductionOrder::AscendingBlock
        );

        // i8 PerBlock32 -> i32 AscendingBlock
        let n_i8_block = matmul_numerics(
            DType::I8,
            DType::I8,
            QuantScheme::PerBlock32,
            QuantScheme::PerRow,
        )
        .unwrap();
        assert_eq!(n_i8_block.accumulator, Some(DType::I32));
        assert_eq!(n_i8_block.reduction_order, ReductionOrder::AscendingBlock);

        // i8 x i4 with PerRow -> i32 AscendingK
        let n_i4_perrow = matmul_numerics(
            DType::I8,
            DType::I4,
            QuantScheme::PerToken,
            QuantScheme::PerRow,
        )
        .unwrap();
        assert_eq!(n_i4_perrow.accumulator, Some(DType::I32));
        assert_eq!(n_i4_perrow.reduction_order, ReductionOrder::AscendingK);

        // i8 x i4 with Scheme (true per-block) -> i32 AscendingBlock
        let n_i4_block = matmul_numerics(
            DType::I8,
            DType::I4,
            QuantScheme::PerToken,
            QuantScheme::Scheme(SchemeId::new(2)),
        )
        .unwrap();
        assert_eq!(n_i4_block.accumulator, Some(DType::I32));
        assert_eq!(n_i4_block.reduction_order, ReductionOrder::AscendingBlock);

        // e4m3 x e4m3 -> f32 AscendingK
        let n_fp8 = matmul_numerics(
            DType::E4m3,
            DType::E4m3,
            QuantScheme::PerToken,
            QuantScheme::PerRow,
        )
        .unwrap();
        assert_eq!(n_fp8.accumulator, Some(DType::F32));
        assert_eq!(n_fp8.reduction_order, ReductionOrder::AscendingK);

        // f16 x f16 -> f32 AscendingK
        let n_f16 =
            matmul_numerics(DType::F16, DType::F16, QuantScheme::None, QuantScheme::None).unwrap();
        assert_eq!(n_f16.accumulator, Some(DType::F32));
        assert_eq!(n_f16.reduction_order, ReductionOrder::AscendingK);

        // bf16 activations x f16 weights -> f32 AscendingK
        let n_bf16 = matmul_numerics(
            DType::Bf16,
            DType::F16,
            QuantScheme::None,
            QuantScheme::None,
        )
        .unwrap();
        assert_eq!(n_bf16.accumulator, Some(DType::F32));
        assert_eq!(n_bf16.reduction_order, ReductionOrder::AscendingK);

        // Typed error for unsupported x dtype
        assert!(matches!(
            matmul_numerics(DType::F32, DType::F16, QuantScheme::None, QuantScheme::None),
            Err(IrError::OpDTypeMismatch { .. })
        ));

        // Mixed integer activation and f16 weight dequantizes to f32.
        let n_i8_f16 = matmul_numerics(
            DType::I8,
            DType::F16,
            QuantScheme::PerToken,
            QuantScheme::None,
        )
        .unwrap();
        assert_eq!(n_i8_f16.accumulator, Some(DType::F32));

        // Typed error for unsupported weight dtype.
        assert!(matches!(
            matmul_numerics(
                DType::I8,
                DType::Bf16,
                QuantScheme::PerToken,
                QuantScheme::None
            ),
            Err(IrError::OpDTypeMismatch { .. })
        ));

        // Typed error for unsupported quant scheme on x
        assert!(matches!(
            matmul_numerics(DType::I8, DType::I8, QuantScheme::None, QuantScheme::PerRow),
            Err(IrError::OpQuantMismatch { .. })
        ));
    }

    #[test]
    fn moe_ffn_gemm_helper_matches_matmul() {
        let n_matmul = matmul_numerics(
            DType::I8,
            DType::I8,
            QuantScheme::PerToken,
            QuantScheme::PerRow,
        )
        .unwrap();
        let n_moe_ffn = moe_ffn_gemm_numerics(
            DType::I8,
            DType::I8,
            QuantScheme::PerToken,
            QuantScheme::PerRow,
        )
        .unwrap();
        assert_eq!(n_matmul, n_moe_ffn);
    }

    #[test]
    fn invalid_accumulators_are_rejected() {
        assert!(Numerics::new(Some(DType::F16), ReductionOrder::AscendingK).is_err());
        assert!(Numerics::new(Some(DType::Bf16), ReductionOrder::AscendingK).is_err());
        assert!(Numerics::new(Some(DType::F32), ReductionOrder::AscendingK).is_ok());
        assert!(Numerics::new(Some(DType::I32), ReductionOrder::AscendingK).is_ok());
        assert!(Numerics::new(None, ReductionOrder::None).is_ok());
    }
}
