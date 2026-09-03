// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::needless_range_loop)]
//! Scalar T0 implementations of `ngram_gather` (Spec 1 §4.A, Card A1.9).
//!
//! Staged mode dequantizes host-gathered rows and lays them out; device mode
//! hashes on the fly through a caller-supplied [`NgramHash`] and gathers from
//! the device table. Both modes stage outputs and commit once.

use r9v_format::{FormatError, SchemeId};
use r9v_ir::{DType, LayoutId, NgramCombine, NgramGatherOp, NgramSource, QuantScheme};

use crate::buffer::{TensorData, TensorView, TensorViewMut};
use crate::error::T0Error;

/// N-gram row hash execution parameter (Spec 1 §4.A, SI-53).
///
/// The spec names no `HashId` enumeration, so T0 never hard-codes a hash
/// family: the caller (scheduler/models scope, where `NgramSpec.hash` lives)
/// supplies the hash. `tokens` is the full `[T]` id slice, `pos` the querying
/// position, `order` the head's n-gram order, `table_size` the head's table
/// extent; the returned row must satisfy `row < table_size`.
pub trait NgramHash {
    /// Resolves the table row for `(pos, order)` in a head table of `table_size` rows.
    fn row(&self, tokens: &[u32], pos: usize, order: u32, table_size: u32) -> u32;
}

/// Executes scalar T0 staged `ngram_gather` (Spec 1 §4.A, Card A1.9).
///
/// Inputs `(gather_staging [T, Np, Dn] i4|i8, row_scales)`; the host already
/// gathered rows into the pinned staging buffer and T0 only dequantizes and
/// lays out: Concat → `y [T, Np·Dn]`, Sum → `y [T, Dn]` (elementwise `f32`
/// sum across heads, ascending), cast once to `out_dtype`.
///
/// Row contract: `Scheme(I8R)` rows carry one scalar scale per `(t, h)` in
/// `row_scales` for any `Dn`; `Scheme(I8B128)` rows require `Dn == 128` so the
/// single row scale is the block scale. Multi-block rows under one scalar
/// have no specified scale application and fail closed, as do `I4K` staged
/// rows (a superblock record cannot be named by a scalar) — SI-53.
///
/// DECISION(A1.9): staged tables are row-major (`CONTIGUOUS`/`L0`) with a
/// separate scalar `row_scales` carrier; rejected inline scale records because
/// the `[T, Np, Dn]` shape leaves no room for them. Per SI-53.
pub fn ngram_gather(
    op: &NgramGatherOp,
    staging: &TensorView<'_>,
    row_scales: &TensorView<'_>,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    staging.validate_backing("gather_staging")?;
    row_scales.validate_backing("row_scales")?;
    y.validate_backing("y")?;

    if op.source != NgramSource::Staged {
        return Err(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "source",
            reason: "staged entry point requires NgramSource::Staged".to_string(),
        });
    }

    let mut problems = Vec::new();
    if staging.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "gather_staging",
            expected: 3,
            got: staging.rank(),
            shape: staging.shape().to_vec(),
        });
    }
    if staging.layout() != LayoutId::CONTIGUOUS && staging.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "gather_staging",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: staging.layout(),
        });
    }
    if !matches!(staging.dtype(), DType::I4 | DType::I8) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "gather_staging",
            expected: vec![DType::I4, DType::I8],
            got: staging.dtype(),
        });
    }
    if row_scales.rank() != 1 && row_scales.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "row_scales",
            expected: 2,
            got: row_scales.rank(),
            shape: row_scales.shape().to_vec(),
        });
    }
    if !matches!(row_scales.dtype(), DType::F32 | DType::F16) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "row_scales",
            expected: vec![DType::F32, DType::F16],
            got: row_scales.dtype(),
        });
    }
    if y.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "y",
            expected: 2,
            got: y.rank(),
            shape: y.shape().to_vec(),
        });
    }
    if !matches!(op.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "out_dtype",
            reason: format!("must be f16, bf16, or f32, got {:?}", op.out_dtype),
        });
    }
    if y.dtype() != op.out_dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.out_dtype],
            got: y.dtype(),
        });
    }
    if op.heads == 0 {
        problems.push(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "heads",
            reason: "heads must be > 0".to_string(),
        });
    }
    match op.combine {
        NgramCombine::Concat | NgramCombine::Sum => {}
    }
    // Scheme gate: native schemes only; I4K and exotic schemes fail closed.
    let block_only = match staging.quant() {
        QuantScheme::Scheme(ir_s) => {
            let sid = SchemeId::from_ir(ir_s)?;
            if !sid.is_native() {
                return Err(T0Error::Format(FormatError::ReservedScheme {
                    scheme: sid.name(),
                    owner: sid.owner_card(),
                }));
            }
            match sid {
                SchemeId::I8R => false,
                SchemeId::I8B128 => true,
                _ => {
                    problems.push(T0Error::QuantMismatch {
                        tensor: "gather_staging",
                        expected: vec![
                            QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                            QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                        ],
                        got: staging.quant(),
                    });
                    false
                }
            }
        }
        _ => {
            problems.push(T0Error::QuantMismatch {
                tensor: "gather_staging",
                expected: vec![
                    QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                    QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                ],
                got: staging.quant(),
            });
            false
        }
    };
    T0Error::from_problems(problems)?;

    let t = staging.shape()[0];
    let np = staging.shape()[1];
    let dn = staging.shape()[2];
    if t == 0 || dn == 0 {
        return Err(T0Error::EmptyInput {
            op: "ngram_gather",
            tensor: "gather_staging",
        });
    }
    if np == 0 {
        return Err(T0Error::EmptyInput {
            op: "ngram_gather",
            tensor: "gather_staging",
        });
    }
    if np != op.heads as usize {
        return Err(T0Error::DimensionMismatch {
            dim_name: "Np",
            expected_from: "heads",
            expected: op.heads as usize,
            tensor: "gather_staging",
            got: np,
        });
    }
    if block_only && dn != 128 {
        return Err(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "row_scales",
            reason: format!(
                "I8B128 staged rows need Dn == 128 for the scalar row scale, got Dn={dn} (SI-53)"
            ),
        });
    }
    for dim in [t, np, dn] {
        u32::try_from(dim).map_err(|_| T0Error::ArithmeticOverflow {
            op: "ngram_gather",
            detail: format!("dimension exceeds u32: {dim}"),
        })?;
    }

    let mut problems = Vec::new();
    if row_scales.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "gather_staging",
            expected: t,
            tensor: "row_scales",
            got: row_scales.shape()[0],
        });
    }
    if row_scales.rank() == 2 {
        let h = row_scales.shape()[1];
        if h != 1 && h != np {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Np",
                expected_from: "gather_staging",
                expected: np,
                tensor: "row_scales",
                got: h,
            });
        }
    }
    let expected_y1 = match op.combine {
        NgramCombine::Concat => np
            .checked_mul(dn)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "ngram_gather",
                detail: format!("Np * Dn overflows usize for Np={np}, Dn={dn}"),
            })?,
        NgramCombine::Sum => dn,
    };
    if y.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "gather_staging",
            expected: t,
            tensor: "y",
            got: y.shape()[0],
        });
    }
    if y.shape()[1] != expected_y1 {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dn",
            expected_from: "gather_staging",
            expected: expected_y1,
            tensor: "y",
            got: y.shape()[1],
        });
    }
    // Row scales must be finite; every offender is collected.
    for row in 0..t {
        for head in 0..np {
            let s = read_row_scale(row_scales, row, head);
            if !s.is_finite() {
                problems.push(T0Error::InvalidAttribute {
                    op: "ngram_gather",
                    attribute: "row_scales",
                    reason: format!("non-finite scale at (t={row}, h={head}): {s}"),
                });
            }
        }
    }
    T0Error::from_problems(problems)?;

    let y_len = t
        .checked_mul(expected_y1)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "ngram_gather",
            detail: "output buffer size overflows usize".to_string(),
        })?;
    let mut y_tmp = vec![0.0f32; y_len];
    for row in 0..t {
        for head in 0..np {
            let scale = read_row_scale(row_scales, row, head);
            for d in 0..dn {
                let q = read_staging_q(staging, row, head, d, np, dn)?;
                let v = (q as f32) * scale;
                match op.combine {
                    NgramCombine::Concat => {
                        y_tmp[row * expected_y1 + head * dn + d] = v;
                    }
                    NgramCombine::Sum => {
                        y_tmp[row * expected_y1 + d] += v;
                    }
                }
            }
        }
    }
    for (idx, &val) in y_tmp.iter().enumerate() {
        y.write_f32(idx, val);
    }
    Ok(())
}

/// Reads one `i8` staging element (typed or raw-byte backing).
fn read_staging_q(
    staging: &TensorView<'_>,
    row: usize,
    head: usize,
    d: usize,
    np: usize,
    dn: usize,
) -> Result<i8, T0Error> {
    let idx = row
        .checked_mul(np)
        .and_then(|v| v.checked_add(head))
        .and_then(|v| v.checked_mul(dn))
        .and_then(|v| v.checked_add(d))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "ngram_gather",
            detail: "staging index overflows usize".to_string(),
        })?;
    match &staging.data {
        TensorData::I8(slice) => {
            slice
                .get(idx)
                .copied()
                .ok_or_else(|| T0Error::BufferLengthMismatch {
                    tensor: "gather_staging",
                    buffer_len: slice.len(),
                    expected_len: idx + 1,
                    shape: staging.shape().to_vec(),
                })
        }
        TensorData::Bytes(_, slice) => {
            slice
                .get(idx)
                .copied()
                .map(|b| b as i8)
                .ok_or_else(|| T0Error::BufferLengthMismatch {
                    tensor: "gather_staging",
                    buffer_len: slice.len(),
                    expected_len: idx + 1,
                    shape: staging.shape().to_vec(),
                })
        }
        _ => Err(T0Error::DTypeMismatch {
            tensor: "gather_staging",
            expected: vec![DType::I8],
            got: staging.dtype(),
        }),
    }
}

/// Reads one row scale for `(row, head)` from `[T]` or `[T, 1|Np]` scales.
fn read_row_scale(scales: &TensorView<'_>, row: usize, head: usize) -> f32 {
    if scales.rank() == 2 {
        if scales.shape()[1] == 1 {
            scales.read_f32(row)
        } else {
            scales.read_f32(row * scales.shape()[1] + head)
        }
    } else {
        scales.read_f32(row)
    }
}

/// Executes scalar T0 device-table `ngram_gather` (Spec 1 §4.A, Card A1.9).
///
/// Inputs `(token_ids [T] u32, table [Σ table_sizes, Dn])`; per `(t, head)`,
/// `row = hash.row(token_ids, t, orders[head], table_sizes[head])` selects a
/// row of head `head`'s table segment (segments concatenate in head order),
/// which is dequantized and laid out exactly like the staged mode.
///
/// `table_scale` carries the table's scale records when the table is
/// quantized (falling back to `table.scale()` when `None`, mirroring
/// `matmul`); unquantized tables take no scale. Only row-major
/// (`CONTIGUOUS`/`L0`) tables are accepted.
///
/// Fail-closed (SI-53): any `orders[head] > 1` is rejected — order-n context
/// rows are not in the signature, so the hash input for them cannot be named.
/// Unknown `HashId` values never reach T0: the hash arrives as `&dyn
/// NgramHash`, so T0 maps no hash family itself.
///
/// DECISION(A1.9): quantized device tables require the separate scale carrier
/// (`[entries]`/`[entries, Dn/128]` `F16` bytes, `[entries, Dn/256, 4]` `U32`
/// bytes); rejected inline scale rows because hashed row addressing needs
/// fixed row strides. Per SI-53.
pub fn ngram_gather_device(
    op: &NgramGatherOp,
    token_ids: &TensorView<'_>,
    table: &TensorView<'_>,
    table_scale: Option<&TensorView<'_>>,
    hash: &dyn NgramHash,
    y: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    token_ids.validate_backing("token_ids")?;
    table.validate_backing("table")?;
    if let Some(s) = table_scale {
        s.validate_backing("table_scale")?;
    }
    y.validate_backing("y")?;
    let table_scale = table_scale.or_else(|| table.scale());
    if let Some(s) = table_scale {
        s.validate_backing("table_scale")?;
    }

    if op.source != NgramSource::Device {
        return Err(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "source",
            reason: "device entry point requires NgramSource::Device".to_string(),
        });
    }

    let mut problems = Vec::new();
    if token_ids.rank() != 1 {
        problems.push(T0Error::RankMismatch {
            tensor: "token_ids",
            expected: 1,
            got: token_ids.rank(),
            shape: token_ids.shape().to_vec(),
        });
    }
    if token_ids.dtype() != DType::U32 {
        problems.push(T0Error::DTypeMismatch {
            tensor: "token_ids",
            expected: vec![DType::U32],
            got: token_ids.dtype(),
        });
    }
    if table.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "table",
            expected: 2,
            got: table.rank(),
            shape: table.shape().to_vec(),
        });
    }
    if table.layout() != LayoutId::CONTIGUOUS && table.layout() != LayoutId::L0 {
        problems.push(T0Error::LayoutMismatch {
            tensor: "table",
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: table.layout(),
        });
    }
    if !matches!(
        table.dtype(),
        DType::I4 | DType::I8 | DType::F16 | DType::Bf16 | DType::F32
    ) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "table",
            expected: vec![DType::I4, DType::I8, DType::F16, DType::Bf16, DType::F32],
            got: table.dtype(),
        });
    }
    if y.rank() != 2 {
        problems.push(T0Error::RankMismatch {
            tensor: "y",
            expected: 2,
            got: y.rank(),
            shape: y.shape().to_vec(),
        });
    }
    if !matches!(op.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "out_dtype",
            reason: format!("must be f16, bf16, or f32, got {:?}", op.out_dtype),
        });
    }
    if y.dtype() != op.out_dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "y",
            expected: vec![op.out_dtype],
            got: y.dtype(),
        });
    }
    if op.heads == 0 {
        problems.push(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "heads",
            reason: "heads must be > 0".to_string(),
        });
    }
    match op.combine {
        NgramCombine::Concat | NgramCombine::Sum => {}
    }
    if op.orders.len() != op.heads as usize {
        problems.push(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "orders",
            reason: format!(
                "orders length {} must equal heads {}",
                op.orders.len(),
                op.heads
            ),
        });
    }
    if op.table_sizes.len() != op.heads as usize {
        problems.push(T0Error::InvalidAttribute {
            op: "ngram_gather",
            attribute: "table_sizes",
            reason: format!(
                "table_sizes length {} must equal heads {}",
                op.table_sizes.len(),
                op.heads
            ),
        });
    }
    for (head, &order) in op.orders.iter().enumerate() {
        if order != 1 {
            problems.push(T0Error::InvalidAttribute {
                op: "ngram_gather",
                attribute: "orders",
                reason: format!(
                    "device mode supports order 1 only, got order {order} at head {head} (SI-53)"
                ),
            });
        }
    }
    // Table quant gate: unquantized float tables, PerRow I8, or native block
    // schemes with a separate scale carrier.
    let table_scheme = match (table.dtype(), table.quant()) {
        (DType::F16 | DType::Bf16 | DType::F32, QuantScheme::None) => None,
        (DType::I8, QuantScheme::PerRow) => Some(SchemeId::I8R),
        (DType::I8 | DType::I4, QuantScheme::Scheme(ir_s)) => {
            let sid = SchemeId::from_ir(ir_s)?;
            if !sid.is_native() {
                return Err(T0Error::Format(FormatError::ReservedScheme {
                    scheme: sid.name(),
                    owner: sid.owner_card(),
                }));
            }
            match sid {
                SchemeId::I8R | SchemeId::I8B128 if table.dtype() == DType::I8 => Some(sid),
                SchemeId::I4K if table.dtype() == DType::I4 => Some(sid),
                _ => {
                    problems.push(T0Error::QuantMismatch {
                        tensor: "table",
                        expected: vec![
                            QuantScheme::None,
                            QuantScheme::PerRow,
                            QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                            QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                            QuantScheme::Scheme(SchemeId::I4K.to_ir()),
                        ],
                        got: table.quant(),
                    });
                    None
                }
            }
        }
        _ => {
            problems.push(T0Error::QuantMismatch {
                tensor: "table",
                expected: vec![
                    QuantScheme::None,
                    QuantScheme::PerRow,
                    QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                    QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                    QuantScheme::Scheme(SchemeId::I4K.to_ir()),
                ],
                got: table.quant(),
            });
            None
        }
    };
    T0Error::from_problems(problems)?;

    let t = token_ids.shape()[0];
    let entries = table.shape()[0];
    let dn = table.shape()[1];
    if t == 0 || entries == 0 || dn == 0 {
        return Err(T0Error::EmptyInput {
            op: "ngram_gather",
            tensor: "table",
        });
    }
    let np = op.heads as usize;
    // Head segment bases with checked arithmetic.
    let mut bases = vec![0usize; np];
    let mut total = 0usize;
    for (head, &size) in op.table_sizes.iter().enumerate() {
        bases[head] = total;
        total = total
            .checked_add(size as usize)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "ngram_gather",
                detail: format!("sum of table_sizes overflows usize at head {head}"),
            })?;
    }
    if total != entries {
        return Err(T0Error::DimensionMismatch {
            dim_name: "entries",
            expected_from: "table_sizes_sum",
            expected: total,
            tensor: "table",
            got: entries,
        });
    }
    // Block-scheme row widths must hold whole blocks.
    match table_scheme {
        Some(SchemeId::I8B128) if !dn.is_multiple_of(128) => {
            return Err(T0Error::DimensionMismatch {
                dim_name: "Dn",
                expected_from: "block_size_128",
                expected: 128,
                tensor: "table",
                got: dn,
            });
        }
        Some(SchemeId::I4K) if !dn.is_multiple_of(256) => {
            return Err(T0Error::DimensionMismatch {
                dim_name: "Dn",
                expected_from: "superblock_256",
                expected: 256,
                tensor: "table",
                got: dn,
            });
        }
        _ => {}
    }
    validate_device_table_scales(table, table_scale, table_scheme, entries, dn)?;
    // Resolve token ids once (validated U32 above).
    let mut tokens = vec![0u32; t];
    for (idx, slot) in tokens.iter_mut().enumerate() {
        *slot = token_ids.try_read_u32(idx, "token_ids").map_err(|_| {
            T0Error::BufferLengthMismatch {
                tensor: "token_ids",
                buffer_len: token_ids.backing_len(),
                expected_len: idx + 1,
                shape: token_ids.shape().to_vec(),
            }
        })?;
    }
    // Resolve and bounds-check every hashed row before touching `y`.
    let row_count = t
        .checked_mul(np)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "ngram_gather",
            detail: "T * Np overflows usize".to_string(),
        })?;
    let mut row_ids = vec![0usize; row_count];
    let mut problems = Vec::new();
    for row in 0..t {
        for head in 0..np {
            let size = op.table_sizes[head];
            let prow = hash.row(&tokens, row, op.orders[head], size);
            if prow >= size {
                problems.push(T0Error::RowIndexOutOfRange {
                    op: "ngram_gather",
                    tensor: "table",
                    position: row * np + head,
                    index: prow,
                    upper_bound: size as usize,
                });
            } else {
                row_ids[row * np + head] = bases[head] + prow as usize;
            }
        }
    }
    let expected_y1 = match op.combine {
        NgramCombine::Concat => np
            .checked_mul(dn)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "ngram_gather",
                detail: format!("Np * Dn overflows usize for Np={np}, Dn={dn}"),
            })?,
        NgramCombine::Sum => dn,
    };
    if y.shape()[0] != t {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "T",
            expected_from: "token_ids",
            expected: t,
            tensor: "y",
            got: y.shape()[0],
        });
    }
    if y.shape()[1] != expected_y1 {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dn",
            expected_from: "table",
            expected: expected_y1,
            tensor: "y",
            got: y.shape()[1],
        });
    }
    T0Error::from_problems(problems)?;

    let y_len = t
        .checked_mul(expected_y1)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "ngram_gather",
            detail: "output buffer size overflows usize".to_string(),
        })?;
    let mut y_tmp = vec![0.0f32; y_len];
    for row in 0..t {
        for head in 0..np {
            let entry = row_ids[row * np + head];
            for d in 0..dn {
                let val = decode_device_entry(table, table_scale, table_scheme, entry, d, dn)?;
                match op.combine {
                    NgramCombine::Concat => {
                        y_tmp[row * expected_y1 + head * dn + d] = val;
                    }
                    NgramCombine::Sum => {
                        y_tmp[row * expected_y1 + d] += val;
                    }
                }
            }
        }
    }
    for (idx, &val) in y_tmp.iter().enumerate() {
        y.write_f32(idx, val);
    }
    Ok(())
}

/// Validates the device-table scale carrier (Spec 1 §4.A).
///
/// Quantized tables require a separate `CONTIGUOUS` raw-byte carrier:
/// `[entries]`/`[entries, Dn/128]` `F16` bytes for `I8R`/`I8B128`,
/// `[entries, Dn/256, 4]` `U32` bytes for `I4K`. Unquantized tables take no
/// scale. Quantized table values must be raw bytes (mirrors matmul).
fn validate_device_table_scales(
    table: &TensorView<'_>,
    table_scale: Option<&TensorView<'_>>,
    scheme: Option<SchemeId>,
    entries: usize,
    dn: usize,
) -> Result<(), T0Error> {
    const OP: &str = "ngram_gather";
    if scheme.is_some() {
        match &table.data {
            TensorData::Bytes(_, _) => {}
            _ => {
                return Err(T0Error::BackingRepresentationMismatch {
                    op: OP,
                    dtype: table.dtype(),
                });
            }
        }
    }
    let Some(sid) = scheme else {
        if table_scale.is_some() {
            return Err(T0Error::InvalidAttribute {
                op: OP,
                attribute: "table_scale",
                reason: "table_scale provided for unquantized table".to_string(),
            });
        }
        return Ok(());
    };
    let Some(ws) = table_scale else {
        return Err(T0Error::MissingOperand {
            op: OP,
            operand: "table_scale",
            detail: "table_scale required for quantized device tables".to_string(),
        });
    };
    if ws.layout() != LayoutId::CONTIGUOUS {
        return Err(T0Error::LayoutMismatch {
            tensor: "table_scale",
            expected: vec![LayoutId::CONTIGUOUS],
            got: ws.layout(),
        });
    }
    let ws_bytes: &[u8] = match &ws.data {
        TensorData::Bytes(_, slice) => slice,
        _ => {
            return Err(T0Error::BackingRepresentationMismatch {
                op: OP,
                dtype: ws.dtype(),
            });
        }
    };
    match sid {
        SchemeId::I8R | SchemeId::I8B128 => {
            if ws.dtype() != DType::F16 {
                return Err(T0Error::DTypeMismatch {
                    tensor: "table_scale",
                    expected: vec![DType::F16],
                    got: ws.dtype(),
                });
            }
            let blocks = match sid {
                SchemeId::I8R => 1,
                _ => dn / 128,
            };
            let req_elems =
                entries
                    .checked_mul(blocks)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: OP,
                        detail: "table scale element count overflows usize".to_string(),
                    })?;
            let shapes_match = if blocks == 1 {
                ws.shape() == [entries].as_slice()
            } else {
                ws.shape() == [entries, blocks].as_slice()
            };
            if !shapes_match {
                return Err(T0Error::DimensionMismatch {
                    dim_name: "scale_shape",
                    expected_from: "table",
                    expected: req_elems,
                    tensor: "table_scale",
                    got: ws.num_elements(),
                });
            }
            if ws_bytes.len() != req_elems * 2 {
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "table_scale",
                    buffer_len: ws_bytes.len(),
                    expected_len: req_elems * 2,
                    shape: ws.shape().to_vec(),
                });
            }
        }
        SchemeId::I4K => {
            if ws.dtype() != DType::U32 {
                return Err(T0Error::DTypeMismatch {
                    tensor: "table_scale",
                    expected: vec![DType::U32],
                    got: ws.dtype(),
                });
            }
            let sbs = dn / 256;
            if ws.shape() != [entries, sbs, 4].as_slice() {
                return Err(T0Error::DimensionMismatch {
                    dim_name: "scale_shape",
                    expected_from: "table",
                    expected: entries * sbs * 4,
                    tensor: "table_scale",
                    got: ws.num_elements(),
                });
            }
            let req_bytes = entries * sbs * 16;
            if ws_bytes.len() != req_bytes {
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "table_scale",
                    buffer_len: ws_bytes.len(),
                    expected_len: req_bytes,
                    shape: ws.shape().to_vec(),
                });
            }
        }
        _ => {
            return Err(T0Error::QuantMismatch {
                tensor: "table",
                expected: vec![
                    QuantScheme::None,
                    QuantScheme::PerRow,
                    QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                    QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                    QuantScheme::Scheme(SchemeId::I4K.to_ir()),
                ],
                got: table.quant(),
            });
        }
    }
    Ok(())
}

/// Reads one `F16`-backed scale as `f32` with bounds reporting.
fn scale_f16_at(scale_bytes: &[u8], idx: usize) -> Result<f32, T0Error> {
    let off = idx
        .checked_mul(2)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "ngram_gather",
            detail: format!("table scale index overflows usize at {idx}"),
        })?;
    let raw: [u8; 2] = scale_bytes
        .get(off..off + 2)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| T0Error::BufferLengthMismatch {
            tensor: "table_scale",
            buffer_len: scale_bytes.len(),
            expected_len: off + 2,
            shape: vec![off + 2],
        })?;
    Ok(crate::dtype::f16_to_f32(u16::from_le_bytes(raw)))
}

/// Decodes one device-table element to `f32` (Spec 1 §4.A).
fn decode_device_entry(
    table: &TensorView<'_>,
    table_scale: Option<&TensorView<'_>>,
    scheme: Option<SchemeId>,
    entry: usize,
    d: usize,
    dn: usize,
) -> Result<f32, T0Error> {
    const OP: &str = "ngram_gather";
    let scale_bytes_of = || -> Result<&[u8], T0Error> {
        match &table_scale {
            Some(ws) => match &ws.data {
                TensorData::Bytes(_, slice) => Ok(slice),
                _ => Err(T0Error::BackingRepresentationMismatch {
                    op: OP,
                    dtype: ws.dtype(),
                }),
            },
            None => Err(T0Error::MissingOperand {
                op: OP,
                operand: "table_scale",
                detail: "table_scale required for quantized device tables".to_string(),
            }),
        }
    };
    let table_bytes_of = || -> Result<&[u8], T0Error> {
        match &table.data {
            TensorData::Bytes(_, slice) => Ok(slice),
            _ => Err(T0Error::BackingRepresentationMismatch {
                op: OP,
                dtype: table.dtype(),
            }),
        }
    };
    match scheme {
        None => {
            let idx = entry
                .checked_mul(dn)
                .and_then(|v| v.checked_add(d))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: OP,
                    detail: "table index overflows usize".to_string(),
                })?;
            if idx >= table.backing_len() {
                return Err(T0Error::BufferLengthMismatch {
                    tensor: "table",
                    buffer_len: table.backing_len(),
                    expected_len: idx + 1,
                    shape: table.shape().to_vec(),
                });
            }
            Ok(table.read_f32(idx))
        }
        Some(SchemeId::I8R) => {
            let bytes = table_bytes_of()?;
            let idx = entry
                .checked_mul(dn)
                .and_then(|v| v.checked_add(d))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: OP,
                    detail: "table index overflows usize".to_string(),
                })?;
            let q = bytes
                .get(idx)
                .copied()
                .ok_or_else(|| T0Error::BufferLengthMismatch {
                    tensor: "table",
                    buffer_len: bytes.len(),
                    expected_len: idx + 1,
                    shape: table.shape().to_vec(),
                })? as i8;
            Ok((q as f32) * scale_f16_at(scale_bytes_of()?, entry)?)
        }
        Some(SchemeId::I8B128) => {
            let bytes = table_bytes_of()?;
            let idx = entry
                .checked_mul(dn)
                .and_then(|v| v.checked_add(d))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: OP,
                    detail: "table index overflows usize".to_string(),
                })?;
            let q = bytes
                .get(idx)
                .copied()
                .ok_or_else(|| T0Error::BufferLengthMismatch {
                    tensor: "table",
                    buffer_len: bytes.len(),
                    expected_len: idx + 1,
                    shape: table.shape().to_vec(),
                })? as i8;
            let blocks = dn / 128;
            let scale_idx = entry
                .checked_mul(blocks)
                .and_then(|v| v.checked_add(d / 128))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: OP,
                    detail: "table scale index overflows usize".to_string(),
                })?;
            Ok((q as f32) * scale_f16_at(scale_bytes_of()?, scale_idx)?)
        }
        Some(SchemeId::I4K) => {
            let bytes = table_bytes_of()?;
            let sb = d / 256;
            let sub = (d % 256) / 32;
            let sbs = dn / 256;
            let ws_bytes = scale_bytes_of()?;
            let header_off = entry
                .checked_mul(sbs)
                .and_then(|v| v.checked_add(sb))
                .and_then(|v| v.checked_mul(16))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: OP,
                    detail: "table header index overflows usize".to_string(),
                })?;
            let raw: [u8; 16] = ws_bytes
                .get(header_off..header_off + 16)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| T0Error::BufferLengthMismatch {
                    tensor: "table_scale",
                    buffer_len: ws_bytes.len(),
                    expected_len: header_off + 16,
                    shape: table.shape().to_vec(),
                })?;
            let header = r9v_format::records::I4KSuperblock::from_bytes(&raw);
            let sc = header.scales();
            let mn = header.mins();
            let s_block = header.d_value(sb as u64)? * (sc[sub] as f32);
            let m_block = header.dmin_value(sb as u64)? * (mn[sub] as f32);
            let flat = entry
                .checked_mul(dn)
                .and_then(|v| v.checked_add(d))
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: OP,
                    detail: "table nibble index overflows usize".to_string(),
                })?;
            let byte =
                bytes
                    .get(flat / 2)
                    .copied()
                    .ok_or_else(|| T0Error::BufferLengthMismatch {
                        tensor: "table",
                        buffer_len: bytes.len(),
                        expected_len: flat / 2 + 1,
                        shape: table.shape().to_vec(),
                    })?;
            // Nibble parity follows the flat element index (even packs low).
            let q = if flat % 2 == 0 {
                (byte & 0x0F) as i32
            } else {
                ((byte >> 4) & 0x0F) as i32
            };
            Ok(s_block * (q as f32) - m_block)
        }
        Some(other) => Err(T0Error::QuantMismatch {
            tensor: "table",
            expected: vec![
                QuantScheme::None,
                QuantScheme::PerRow,
                QuantScheme::Scheme(SchemeId::I8R.to_ir()),
                QuantScheme::Scheme(SchemeId::I8B128.to_ir()),
                QuantScheme::Scheme(SchemeId::I4K.to_ir()),
            ],
            got: QuantScheme::Scheme(other.to_ir()),
        }),
    }
}

/// 64-bit reference staged n-gram gather for testing (Spec 1 §4.A).
///
/// Independent `f64` path: plain nested loops over `&[i8]` rows and `f64`
/// scales, never calling [`ngram_gather`]. `scales` carries one `f64` scale
/// per `(t, head)` in row-major order. Slice lengths and every extent
/// product are validated with typed errors; there is no silent empty
/// fallback.
pub fn ngram_gather_f64_reference_staged(
    staging_i8: &[i8],
    scales: &[f64],
    t: usize,
    np: usize,
    dn: usize,
    combine: NgramCombine,
) -> Result<Vec<f64>, T0Error> {
    const OP: &str = "ngram_gather";
    let staging_len = t
        .checked_mul(np)
        .and_then(|v| v.checked_mul(dn))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "T * Np * Dn overflows usize".to_string(),
        })?;
    let scales_len = t
        .checked_mul(np)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("T * Np overflows usize for T={t}, Np={np}"),
        })?;
    let mut problems = Vec::new();
    if staging_i8.len() != staging_len {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "staging_i8",
            expected: staging_len,
            got: staging_i8.len(),
            detail: "staging length must equal T * Np * Dn".to_string(),
        });
    }
    if scales.len() != scales_len {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "scales",
            expected: scales_len,
            got: scales.len(),
            detail: "scales length must equal T * Np".to_string(),
        });
    }
    T0Error::from_problems(problems)?;
    let out1 = match combine {
        NgramCombine::Concat => np
            .checked_mul(dn)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: OP,
                detail: format!("Np * Dn overflows usize for Np={np}, Dn={dn}"),
            })?,
        NgramCombine::Sum => dn,
    };
    let y_len = t
        .checked_mul(out1)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "output buffer size overflows usize".to_string(),
        })?;
    let mut y = vec![0.0f64; y_len];
    for row in 0..t {
        for head in 0..np {
            let scale = scales[row * np + head];
            for d in 0..dn {
                let v = staging_i8[(row * np + head) * dn + d] as f64 * scale;
                match combine {
                    NgramCombine::Concat => {
                        y[row * out1 + head * dn + d] = v;
                    }
                    NgramCombine::Sum => {
                        y[row * out1 + d] += v;
                    }
                }
            }
        }
    }
    Ok(y)
}

/// 64-bit reference device-table n-gram layout for testing (Spec 1 §4.A).
///
/// Independent `f64` path over already-resolved `row_ids [T·Np]` and an `f64`
/// table: covers layout/combine only, while hash resolution and bounds
/// rejection are tested directly against [`ngram_gather_device`]. Slice
/// lengths, row bounds, and every extent product are validated with typed
/// errors; there is no silent empty fallback.
pub fn ngram_gather_f64_reference_rows(
    table_f64: &[f64],
    entries: usize,
    dn: usize,
    row_ids: &[u32],
    t: usize,
    np: usize,
    combine: NgramCombine,
) -> Result<Vec<f64>, T0Error> {
    const OP: &str = "ngram_gather";
    let table_len = entries
        .checked_mul(dn)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "entries * Dn overflows usize".to_string(),
        })?;
    let ids_len = t
        .checked_mul(np)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: format!("T * Np overflows usize for T={t}, Np={np}"),
        })?;
    let mut problems = Vec::new();
    if table_f64.len() != table_len {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "table_f64",
            expected: table_len,
            got: table_f64.len(),
            detail: "table length must equal entries * Dn".to_string(),
        });
    }
    if row_ids.len() != ids_len {
        problems.push(T0Error::ShapeLengthMismatch {
            op: OP,
            tensor: "row_ids",
            expected: ids_len,
            got: row_ids.len(),
            detail: "row_ids length must equal T * Np".to_string(),
        });
    }
    for (pos, &id) in row_ids.iter().enumerate() {
        if (id as usize) >= entries {
            problems.push(T0Error::RowIndexOutOfRange {
                op: OP,
                tensor: "row_ids",
                position: pos,
                index: id,
                upper_bound: entries,
            });
        }
    }
    T0Error::from_problems(problems)?;
    let out1 = match combine {
        NgramCombine::Concat => np
            .checked_mul(dn)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: OP,
                detail: format!("Np * Dn overflows usize for Np={np}, Dn={dn}"),
            })?,
        NgramCombine::Sum => dn,
    };
    let y_len = t
        .checked_mul(out1)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: OP,
            detail: "output buffer size overflows usize".to_string(),
        })?;
    let mut y = vec![0.0f64; y_len];
    for row in 0..t {
        for head in 0..np {
            let entry = row_ids[row * np + head] as usize;
            if entry >= entries {
                return Err(T0Error::RowIndexOutOfRange {
                    op: OP,
                    tensor: "row_ids",
                    position: row * np + head,
                    index: row_ids[row * np + head],
                    upper_bound: entries,
                });
            }
            for d in 0..dn {
                let v = table_f64[entry * dn + d];
                match combine {
                    NgramCombine::Concat => {
                        y[row * out1 + head * dn + d] = v;
                    }
                    NgramCombine::Sum => {
                        y[row * out1 + d] += v;
                    }
                }
            }
        }
    }
    Ok(y)
}
