// SPDX-License-Identifier: Apache-2.0
//! Scalar deterministic T0 attention group: `state_write_kv` + `attention`
//! (Spec 1 §4.D, §6.3, Spec 3 §2, §3, Card A1.7).
//!
//! T0 is the ground-truth oracle for the attention group. Both ops are scalar,
//! strictly deterministic (ascending logical block/position order everywhere),
//! accumulate QK/PV and the online-softmax max/sum in f32, and perform no
//! mutation until every validation problem has been collected and reported
//! (CONVENTIONS.md §1.4).
//!
//! Cache storage models the Spec 3 §3.2 physical regions exactly: per-block
//! per-head K/V values in the cache dtype plus one f16 scale per token per
//! head for `i8`/`e4m3` (`KvPaged`); one `[32, latent]` region in the cache
//! dtype with `[32]` f16 scales plus one `[32, rope]` f16 region for
//! `KvLatent` (rope is always f16 per Spec 3 §2).

use r9v_format::scales::{f16_to_f32 as format_f16_to_f32, f32_to_f16_bits, E4m3};
use r9v_ir::{
    AttentionMask, AttentionOp, BatchMeta, CacheScaleGranularity, DType, LayoutId, StateWriteKvOp,
    BLOCK_TABLE_SENTINEL,
};

use crate::buffer::{TensorView, TensorViewMut};
use crate::error::{u64_to_usize, T0Error};

/// Tokens per KV block (Spec 3 §3.1).
const BLOCK_TOKENS: usize = 32;

/// Slot-map value marking a token that writes no state (pad token or a
/// non-paged group row). Equals `r9v_state::SLOT_NONE` (`u32::MAX`); `r9v-t0`
/// must not depend on `r9v-state` (Spec 14 §2 layering), so the value is
/// restated here with its provenance instead of adding a dependency.
const SLOT_NONE: u32 = u32::MAX;

// DECISION(A1.7): q/k/v/o tensor views must be CONTIGUOUS or L0; tiled
// (L1/L1S) layouts are rejected with a typed LayoutMismatch. Spec 1 §4.D is
// silent on tiled attention inputs and the T0 views index logical row-major
// order; rejected silently accepting tiles because that would misread the
// physical lane order as logical rows.
fn check_row_major_layout(view_layout: LayoutId, tensor: &'static str) -> Result<(), T0Error> {
    if view_layout != LayoutId::CONTIGUOUS && view_layout != LayoutId::L0 {
        return Err(T0Error::LayoutMismatch {
            tensor,
            expected: vec![LayoutId::CONTIGUOUS, LayoutId::L0],
            got: view_layout,
        });
    }
    Ok(())
}

/// Checked `a * b + c` for flat index math; every address in this module goes
/// through here or an equivalent checked chain (no unchecked products/ranges).
fn checked_addr(a: usize, b: usize, c: usize, what: &str) -> Result<usize, T0Error> {
    a.checked_mul(b)
        .and_then(|v| v.checked_add(c))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "attention",
            detail: format!("{what}: {a} * {b} + {c} overflows usize"),
        })
}

/// Total guards over the asserting `BatchMeta` accessors (`slot`/`block`/
/// `window` panic on out-of-bounds indices per `r9v-ir`). Every call site in
/// this module routes through these wrappers, so a malformed `BatchMeta`
/// relationship surfaces as a typed error and never reaches the `assert!`
/// inside `r9v-ir` (Spec 1 §2.5, CONVENTIONS.md §1.5).
fn checked_slot(meta: &BatchMeta, group: u32, tok: u32) -> Result<u32, T0Error> {
    if (group as usize) >= meta.num_groups() {
        return Err(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "group",
            reason: format!(
                "group {group} out of range for BatchMeta with {} groups",
                meta.num_groups()
            ),
        });
    }
    if (tok as usize) >= meta.total_tokens() {
        return Err(T0Error::RowIndexOutOfRange {
            op: "attention",
            tensor: "slot_map",
            position: tok as usize,
            index: tok,
            upper_bound: meta.total_tokens(),
        });
    }
    Ok(meta.slot(group, tok))
}

/// Guarded `BatchMeta::block`: proves `group < G`, `seq < S`, `b < max_blocks`
/// before touching the asserting accessor.
fn checked_block(meta: &BatchMeta, group: u32, seq: u32, b: u32) -> Result<u32, T0Error> {
    if (group as usize) >= meta.num_groups() {
        return Err(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "group",
            reason: format!(
                "group {group} out of range for BatchMeta with {} groups",
                meta.num_groups()
            ),
        });
    }
    if (seq as usize) >= meta.num_seqs() {
        return Err(T0Error::RowIndexOutOfRange {
            op: "attention",
            tensor: "block_table",
            position: seq as usize,
            index: seq,
            upper_bound: meta.num_seqs(),
        });
    }
    if b >= meta.max_blocks() {
        return Err(T0Error::RowIndexOutOfRange {
            op: "attention",
            tensor: "block_table",
            position: b as usize,
            index: b,
            upper_bound: meta.max_blocks() as usize,
        });
    }
    Ok(meta.block(group, seq, b))
}

/// Guarded `BatchMeta::window`: proves `group < G`, `seq < S` before touching
/// the asserting accessor.
fn checked_window(meta: &BatchMeta, group: u32, seq: u32) -> Result<u32, T0Error> {
    if (group as usize) >= meta.num_groups() {
        return Err(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "group",
            reason: format!(
                "group {group} out of range for BatchMeta with {} groups",
                meta.num_groups()
            ),
        });
    }
    if (seq as usize) >= meta.num_seqs() {
        return Err(T0Error::RowIndexOutOfRange {
            op: "attention",
            tensor: "window_start",
            position: seq as usize,
            index: seq,
            upper_bound: meta.num_seqs(),
        });
    }
    Ok(meta.window(group, seq))
}

/// Checked store into a flat cache region: the typed backstop behind every
/// cache write, so a validation-dominance hole fails with
/// `BufferLengthMismatch` instead of an index panic.
fn checked_store<T: Copy>(
    slice: &mut [T],
    idx: usize,
    value: T,
    tensor: &'static str,
) -> Result<(), T0Error> {
    let len = slice.len();
    match slice.get_mut(idx) {
        Some(slot) => {
            *slot = value;
            Ok(())
        }
        None => Err(T0Error::BufferLengthMismatch {
            tensor,
            buffer_len: len,
            expected_len: idx.saturating_add(1),
            shape: vec![len],
        }),
    }
}

/// Checked read of one element from a planned flat row (same backstop as
/// [`checked_store`]; dimension bounds were proven by validation).
fn flat_at(flat: &[f32], head: usize, row_len: usize, d: usize) -> Result<f32, T0Error> {
    let idx = head
        .checked_mul(row_len)
        .and_then(|v| v.checked_add(d))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "state_write_kv",
            detail: "planned row index overflows usize".to_string(),
        })?;
    flat.get(idx)
        .copied()
        .ok_or_else(|| T0Error::BufferLengthMismatch {
            tensor: "plan",
            buffer_len: flat.len(),
            expected_len: idx.saturating_add(1),
            shape: vec![flat.len()],
        })
}

/// Checked head-row slice of a planned flat row (same backstop as
/// [`checked_store`]).
fn flat_row(flat: &[f32], head: usize, row_len: usize) -> Result<&[f32], T0Error> {
    let start = head
        .checked_mul(row_len)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "state_write_kv",
            detail: "planned row start overflows usize".to_string(),
        })?;
    let end = start
        .checked_add(row_len)
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "state_write_kv",
            detail: "planned row end overflows usize".to_string(),
        })?;
    flat.get(start..end)
        .ok_or_else(|| T0Error::BufferLengthMismatch {
            tensor: "plan",
            buffer_len: flat.len(),
            expected_len: end,
            shape: vec![flat.len()],
        })
}

/// Paged KV cache for one layer group (Spec 3 §3.2, `KvPaged`).
///
/// Holds all local KV heads for `num_blocks * 32` slots. Quantized (`i8`,
/// `e4m3`) values carry one f16 scale per token per head for K and for V;
/// `f16` carries no scales. A parallel `written` bitmap marks slots filled by
/// [`state_write_kv_paged`]; reads of retained but unwritten slots fail with a
/// typed error instead of silently returning zero.
#[derive(Debug, Clone)]
pub struct KvPagedCache {
    num_blocks: usize,
    hkv: usize,
    d: usize,
    dv: usize,
    dtype: DType,
    k_f16: Vec<u16>,
    v_f16: Vec<u16>,
    k_i8: Vec<i8>,
    v_i8: Vec<i8>,
    k_e4m3: Vec<u8>,
    v_e4m3: Vec<u8>,
    k_scales: Vec<u16>,
    v_scales: Vec<u16>,
    written: Vec<bool>,
}

impl KvPagedCache {
    /// Creates a zeroed (unwritten) paged cache (Spec 3 §3.2).
    pub fn new(
        num_blocks: usize,
        hkv: usize,
        d: usize,
        dv: usize,
        dtype: DType,
    ) -> Result<Self, T0Error> {
        let mut problems = Vec::new();
        if !matches!(dtype, DType::F16 | DType::I8 | DType::E4m3) {
            problems.push(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "cache_dtype",
                reason: format!("paged cache dtype must be f16, i8, or e4m3; got {dtype:?}"),
            });
        }
        for (name, value) in [
            ("num_blocks", num_blocks),
            ("hkv", hkv),
            ("d", d),
            ("dv", dv),
        ] {
            if value == 0 {
                problems.push(T0Error::InvalidAttribute {
                    op: "state_write_kv",
                    attribute: "cache_geometry",
                    reason: format!("{name} must be > 0, got 0"),
                });
            }
        }
        T0Error::from_problems(problems)?;
        let slots = checked_addr(num_blocks, BLOCK_TOKENS, 0, "paged cache slots")?;
        let k_elems = slots
            .checked_mul(hkv)
            .and_then(|v| v.checked_mul(d))
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "attention",
                detail: format!("paged K elements {slots} * {hkv} * {d} overflows usize"),
            })?;
        let v_elems = slots
            .checked_mul(hkv)
            .and_then(|v| v.checked_mul(dv))
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "attention",
                detail: format!("paged V elements {slots} * {hkv} * {dv} overflows usize"),
            })?;
        let scale_elems = slots
            .checked_mul(hkv)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "attention",
                detail: format!("paged scale elements {slots} * {hkv} overflows usize"),
            })?;
        let (k_f16, v_f16, k_i8, v_i8, k_e4m3, v_e4m3, k_scales, v_scales) = match dtype {
            DType::F16 => (
                vec![0u16; k_elems],
                vec![0u16; v_elems],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
            DType::I8 => (
                Vec::new(),
                Vec::new(),
                vec![0i8; k_elems],
                vec![0i8; v_elems],
                Vec::new(),
                Vec::new(),
                vec![0u16; scale_elems],
                vec![0u16; scale_elems],
            ),
            DType::E4m3 => (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![0u8; k_elems],
                vec![0u8; v_elems],
                vec![0u16; scale_elems],
                vec![0u16; scale_elems],
            ),
            _ => {
                return Err(T0Error::InvalidAttribute {
                    op: "state_write_kv",
                    attribute: "cache_dtype",
                    reason: "paged/latent cache dtype must be f16, i8, or e4m3".to_string(),
                });
            }
        };
        Ok(Self {
            num_blocks,
            hkv,
            d,
            dv,
            dtype,
            k_f16,
            v_f16,
            k_i8,
            v_i8,
            k_e4m3,
            v_e4m3,
            k_scales,
            v_scales,
            written: vec![false; slots],
        })
    }

    /// Block count backing this cache.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Local KV head count.
    pub fn hkv(&self) -> usize {
        self.hkv
    }

    /// K head dimension.
    pub fn d(&self) -> usize {
        self.d
    }

    /// V head dimension.
    pub fn dv(&self) -> usize {
        self.dv
    }

    /// Cache storage dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Flat slot count (`num_blocks * 32`).
    pub fn num_slots(&self) -> usize {
        self.written.len()
    }

    /// Whether `slot` was written by [`state_write_kv_paged`].
    pub fn is_written(&self, slot: usize) -> bool {
        self.written.get(slot).copied().unwrap_or(false)
    }

    fn k_offset(&self, slot: usize, head: usize, dim: usize) -> Result<usize, T0Error> {
        checked_addr(slot, self.hkv, head, "paged K slot/head")
            .and_then(|v| checked_addr(v, self.d, dim, "paged K head/dim"))
    }

    fn v_offset(&self, slot: usize, head: usize, dim: usize) -> Result<usize, T0Error> {
        checked_addr(slot, self.hkv, head, "paged V slot/head")
            .and_then(|v| checked_addr(v, self.dv, dim, "paged V head/dim"))
    }

    fn scale_offset(&self, slot: usize, head: usize) -> Result<usize, T0Error> {
        checked_addr(slot, self.hkv, head, "paged scale slot/head")
    }

    fn check_slot(&self, slot: usize) -> Result<(), T0Error> {
        if slot >= self.written.len() {
            return Err(T0Error::RowIndexOutOfRange {
                op: "attention",
                tensor: "cache",
                position: slot,
                // Error report only: saturate rather than wrap on 16-bit targets.
                index: u32::try_from(slot).unwrap_or(u32::MAX),
                upper_bound: self.written.len(),
            });
        }
        Ok(())
    }

    /// Reads one dequantized K value in f32 (Spec 1 §6.3: cache dequant to f32).
    pub fn read_k_f32(&self, slot: usize, head: usize, dim: usize) -> Result<f32, T0Error> {
        self.check_slot(slot)?;
        let off = self.k_offset(slot, head, dim)?;
        match self.dtype {
            DType::F16 => self
                .k_f16
                .get(off)
                .map(|&b| format_f16_to_f32(b))
                .ok_or_else(|| T0Error::BufferLengthMismatch {
                    tensor: "cache.k",
                    buffer_len: self.k_f16.len(),
                    expected_len: off + 1,
                    shape: vec![self.num_slots(), self.hkv, self.d],
                }),
            DType::I8 => {
                let q = *self
                    .k_i8
                    .get(off)
                    .ok_or_else(|| T0Error::BufferLengthMismatch {
                        tensor: "cache.k",
                        buffer_len: self.k_i8.len(),
                        expected_len: off + 1,
                        shape: vec![self.num_slots(), self.hkv, self.d],
                    })?;
                let s = self.scale_offset(slot, head).and_then(|so| {
                    self.k_scales
                        .get(so)
                        .copied()
                        .ok_or_else(|| T0Error::BufferLengthMismatch {
                            tensor: "cache.k_scales",
                            buffer_len: self.k_scales.len(),
                            expected_len: so + 1,
                            shape: vec![self.num_slots(), self.hkv],
                        })
                })?;
                Ok(format_f16_to_f32(s) * f32::from(q))
            }
            DType::E4m3 => {
                let b = *self
                    .k_e4m3
                    .get(off)
                    .ok_or_else(|| T0Error::BufferLengthMismatch {
                        tensor: "cache.k",
                        buffer_len: self.k_e4m3.len(),
                        expected_len: off + 1,
                        shape: vec![self.num_slots(), self.hkv, self.d],
                    })?;
                let s = self.scale_offset(slot, head).and_then(|so| {
                    self.k_scales
                        .get(so)
                        .copied()
                        .ok_or_else(|| T0Error::BufferLengthMismatch {
                            tensor: "cache.k_scales",
                            buffer_len: self.k_scales.len(),
                            expected_len: so + 1,
                            shape: vec![self.num_slots(), self.hkv],
                        })
                })?;
                Ok(format_f16_to_f32(s) * E4m3::new(b).to_f32())
            }
            _ => Err(T0Error::InvalidAttribute {
                op: "attention",
                attribute: "cache_dtype",
                reason: "cache dtype must be f16, i8, or e4m3".to_string(),
            }),
        }
    }

    /// Reads one dequantized V value in f32 (Spec 1 §6.3: cache dequant to f32).
    pub fn read_v_f32(&self, slot: usize, head: usize, dim: usize) -> Result<f32, T0Error> {
        self.check_slot(slot)?;
        let off = self.v_offset(slot, head, dim)?;
        match self.dtype {
            DType::F16 => self
                .v_f16
                .get(off)
                .map(|&b| format_f16_to_f32(b))
                .ok_or_else(|| T0Error::BufferLengthMismatch {
                    tensor: "cache.v",
                    buffer_len: self.v_f16.len(),
                    expected_len: off + 1,
                    shape: vec![self.num_slots(), self.hkv, self.dv],
                }),
            DType::I8 => {
                let q = *self
                    .v_i8
                    .get(off)
                    .ok_or_else(|| T0Error::BufferLengthMismatch {
                        tensor: "cache.v",
                        buffer_len: self.v_i8.len(),
                        expected_len: off + 1,
                        shape: vec![self.num_slots(), self.hkv, self.dv],
                    })?;
                let s = self.scale_offset(slot, head).and_then(|so| {
                    self.v_scales
                        .get(so)
                        .copied()
                        .ok_or_else(|| T0Error::BufferLengthMismatch {
                            tensor: "cache.v_scales",
                            buffer_len: self.v_scales.len(),
                            expected_len: so + 1,
                            shape: vec![self.num_slots(), self.hkv],
                        })
                })?;
                Ok(format_f16_to_f32(s) * f32::from(q))
            }
            DType::E4m3 => {
                let b = *self
                    .v_e4m3
                    .get(off)
                    .ok_or_else(|| T0Error::BufferLengthMismatch {
                        tensor: "cache.v",
                        buffer_len: self.v_e4m3.len(),
                        expected_len: off + 1,
                        shape: vec![self.num_slots(), self.hkv, self.dv],
                    })?;
                let s = self.scale_offset(slot, head).and_then(|so| {
                    self.v_scales
                        .get(so)
                        .copied()
                        .ok_or_else(|| T0Error::BufferLengthMismatch {
                            tensor: "cache.v_scales",
                            buffer_len: self.v_scales.len(),
                            expected_len: so + 1,
                            shape: vec![self.num_slots(), self.hkv],
                        })
                })?;
                Ok(format_f16_to_f32(s) * E4m3::new(b).to_f32())
            }
            _ => Err(T0Error::InvalidAttribute {
                op: "attention",
                attribute: "cache_dtype",
                reason: "cache dtype must be f16, i8, or e4m3".to_string(),
            }),
        }
    }
}

/// MLA latent cache for one layer group (Spec 3 §2, §3.2, `KvLatent`).
///
/// One `[32, latent]` region in the cache dtype with `[32]` f16 scales plus
/// one `[32, rope]` f16 region per block. There is no head dimension: the
/// producing builder writes `[T, 1, latent]` / `[T, 1, rope]` (Spec 8 §3.1,
/// SI-29), so every slot holds a single compressed vector plus its rope part.
#[derive(Debug, Clone)]
pub struct KvLatentCache {
    num_blocks: usize,
    latent: usize,
    rope: usize,
    dtype: DType,
    latent_f16: Vec<u16>,
    latent_i8: Vec<i8>,
    latent_e4m3: Vec<u8>,
    latent_scales: Vec<u16>,
    rope_f16: Vec<u16>,
    written: Vec<bool>,
}

impl KvLatentCache {
    /// Creates a zeroed (unwritten) latent cache (Spec 3 §2, §3.2).
    pub fn new(
        num_blocks: usize,
        latent: usize,
        rope: usize,
        dtype: DType,
    ) -> Result<Self, T0Error> {
        let mut problems = Vec::new();
        if !matches!(dtype, DType::F16 | DType::I8 | DType::E4m3) {
            problems.push(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "cache_dtype",
                reason: format!("latent cache dtype must be f16, i8, or e4m3; got {dtype:?}"),
            });
        }
        for (name, value) in [
            ("num_blocks", num_blocks),
            ("latent", latent),
            ("rope", rope),
        ] {
            if value == 0 {
                problems.push(T0Error::InvalidAttribute {
                    op: "state_write_kv",
                    attribute: "cache_geometry",
                    reason: format!("{name} must be > 0, got 0"),
                });
            }
        }
        T0Error::from_problems(problems)?;
        let slots = checked_addr(num_blocks, BLOCK_TOKENS, 0, "latent cache slots")?;
        let latent_elems =
            slots
                .checked_mul(latent)
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "attention",
                    detail: format!("latent elements {slots} * {latent} overflows usize"),
                })?;
        let rope_elems = slots
            .checked_mul(rope)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "attention",
                detail: format!("rope elements {slots} * {rope} overflows usize"),
            })?;
        let (latent_f16, latent_i8, latent_e4m3, latent_scales) = match dtype {
            DType::F16 => (vec![0u16; latent_elems], Vec::new(), Vec::new(), Vec::new()),
            DType::I8 => (
                Vec::new(),
                vec![0i8; latent_elems],
                Vec::new(),
                vec![0u16; slots],
            ),
            DType::E4m3 => (
                Vec::new(),
                Vec::new(),
                vec![0u8; latent_elems],
                vec![0u16; slots],
            ),
            _ => {
                return Err(T0Error::InvalidAttribute {
                    op: "state_write_kv",
                    attribute: "cache_dtype",
                    reason: "paged/latent cache dtype must be f16, i8, or e4m3".to_string(),
                });
            }
        };
        Ok(Self {
            num_blocks,
            latent,
            rope,
            dtype,
            latent_f16,
            latent_i8,
            latent_e4m3,
            latent_scales,
            rope_f16: vec![0u16; rope_elems],
            written: vec![false; slots],
        })
    }

    /// Block count backing this cache.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Compressed latent width.
    pub fn latent(&self) -> usize {
        self.latent
    }

    /// Rope part width (always f16 storage per Spec 3 §2).
    pub fn rope(&self) -> usize {
        self.rope
    }

    /// Cache storage dtype of the latent part.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Flat slot count (`num_blocks * 32`).
    pub fn num_slots(&self) -> usize {
        self.written.len()
    }

    /// Whether `slot` was written.
    pub fn is_written(&self, slot: usize) -> bool {
        self.written.get(slot).copied().unwrap_or(false)
    }

    fn check_slot(&self, slot: usize) -> Result<(), T0Error> {
        if slot >= self.written.len() {
            return Err(T0Error::RowIndexOutOfRange {
                op: "attention",
                tensor: "cache",
                position: slot,
                // Error report only: saturate rather than wrap on 16-bit targets.
                index: u32::try_from(slot).unwrap_or(u32::MAX),
                upper_bound: self.written.len(),
            });
        }
        Ok(())
    }

    /// Reads one dequantized latent value in f32.
    pub fn read_latent_f32(&self, slot: usize, dim: usize) -> Result<f32, T0Error> {
        self.check_slot(slot)?;
        let off = checked_addr(slot, self.latent, dim, "latent slot/dim")?;
        match self.dtype {
            DType::F16 => self
                .latent_f16
                .get(off)
                .map(|&b| format_f16_to_f32(b))
                .ok_or_else(|| T0Error::BufferLengthMismatch {
                    tensor: "cache.latent",
                    buffer_len: self.latent_f16.len(),
                    expected_len: off + 1,
                    shape: vec![self.num_slots(), self.latent],
                }),
            DType::I8 => {
                let q = *self
                    .latent_i8
                    .get(off)
                    .ok_or_else(|| T0Error::BufferLengthMismatch {
                        tensor: "cache.latent",
                        buffer_len: self.latent_i8.len(),
                        expected_len: off + 1,
                        shape: vec![self.num_slots(), self.latent],
                    })?;
                let s = self.latent_scales.get(slot).copied().ok_or_else(|| {
                    T0Error::BufferLengthMismatch {
                        tensor: "cache.latent_scales",
                        buffer_len: self.latent_scales.len(),
                        expected_len: slot + 1,
                        shape: vec![self.num_slots()],
                    }
                })?;
                Ok(format_f16_to_f32(s) * f32::from(q))
            }
            DType::E4m3 => {
                let b =
                    *self
                        .latent_e4m3
                        .get(off)
                        .ok_or_else(|| T0Error::BufferLengthMismatch {
                            tensor: "cache.latent",
                            buffer_len: self.latent_e4m3.len(),
                            expected_len: off + 1,
                            shape: vec![self.num_slots(), self.latent],
                        })?;
                let s = self.latent_scales.get(slot).copied().ok_or_else(|| {
                    T0Error::BufferLengthMismatch {
                        tensor: "cache.latent_scales",
                        buffer_len: self.latent_scales.len(),
                        expected_len: slot + 1,
                        shape: vec![self.num_slots()],
                    }
                })?;
                Ok(format_f16_to_f32(s) * E4m3::new(b).to_f32())
            }
            _ => Err(T0Error::InvalidAttribute {
                op: "attention",
                attribute: "cache_dtype",
                reason: "cache dtype must be f16, i8, or e4m3".to_string(),
            }),
        }
    }

    /// Reads one rope value in f32 (rope storage is always f16 per Spec 3 §2).
    pub fn read_rope_f32(&self, slot: usize, dim: usize) -> Result<f32, T0Error> {
        self.check_slot(slot)?;
        let off = checked_addr(slot, self.rope, dim, "rope slot/dim")?;
        self.rope_f16
            .get(off)
            .map(|&b| format_f16_to_f32(b))
            .ok_or_else(|| T0Error::BufferLengthMismatch {
                tensor: "cache.rope",
                buffer_len: self.rope_f16.len(),
                expected_len: off + 1,
                shape: vec![self.num_slots(), self.rope],
            })
    }
}

/// Either paged-KV or latent cache backing one layer group (Spec 1 §2.6).
#[derive(Debug, Clone)]
pub enum KvCache {
    /// Paged K/V cache (`StateKind::KvPaged`).
    Paged(KvPagedCache),
    /// MLA compressed latent + rope cache (`StateKind::KvLatent`).
    Latent(KvLatentCache),
}

// ----------------------------------------------------------------------------
// state_write_kv (Spec 1 §4.D, Spec 3 §3.2)
// ----------------------------------------------------------------------------

// DECISION(A1.7): quantized cache rows use symmetric per-token-head scales
// with the quant_act convention (scale = absmax / 127 for i8,
// absmax / 448 for e4m3; i8 rounded ties-even clamped to [-127, 127];
// all-zero rows emit scale 0 with zero codes); rejected asymmetric or
// [-128, 127] bounds because Spec 1 §6.2 assumes symmetric bounds and the
// cache must agree with activation quantization bit-for-bit.
fn quant_row_i8(values: &[f32]) -> (Vec<i8>, f32) {
    let mut absmax = 0.0f32;
    for &v in values {
        let a = v.abs();
        if a > absmax {
            absmax = a;
        }
    }
    let scale = absmax / 127.0f32;
    if scale == 0.0f32 {
        return (vec![0i8; values.len()], 0.0f32);
    }
    let q = values
        .iter()
        .map(|&v| (v / scale).round_ties_even().clamp(-127.0f32, 127.0f32) as i8)
        .collect();
    (q, scale)
}

/// Quantizes one row to e4m3, failing closed with a typed error when a scaled
/// value is not encodable (Spec 1 §6.3, CONVENTIONS.md §1.5).
///
/// `E4m3::from_f32` rejects only non-finite inputs, which the write-path
/// pre-validation already collects as typed problems before mutation, so this
/// error fires only if that validation ever has a hole. There is no silent
/// zero-byte fallback: an unencodable value refuses instead of corrupting the
/// cache (rejected `debug_assert!` + `0x00`, which vanished in release).
fn quant_row_e4m3(
    values: &[f32],
    tensor: &'static str,
    tok: usize,
    head: usize,
) -> Result<(Vec<u8>, f32), T0Error> {
    let mut absmax = 0.0f32;
    for &v in values {
        let a = v.abs();
        if a > absmax {
            absmax = a;
        }
    }
    let scale = absmax / 448.0f32;
    if scale == 0.0f32 {
        return Ok((vec![0u8; values.len()], 0.0f32));
    }
    let mut q = Vec::with_capacity(values.len());
    for &v in values {
        match E4m3::from_f32(v / scale) {
            Some(c) => q.push(c.bits()),
            None => {
                return Err(T0Error::InvalidAttribute {
                    op: "state_write_kv",
                    attribute: "k/v",
                    reason: format!(
                        "token {tok} head {head}: scaled value {} / scale {scale} is not encodable in e4m3 {tensor} cache",
                        v / scale,
                    ),
                });
            }
        }
    }
    Ok((q, scale))
}

/// Validates the shared `state_write_kv` preconditions (Spec 1 §4.D).
///
/// Collects every problem before returning; performs no mutation.
#[allow(clippy::too_many_arguments)]
fn validate_write_common(
    op: &StateWriteKvOp,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    meta: &BatchMeta,
    group: u32,
    problems: &mut Vec<T0Error>,
) {
    if k.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "k",
            expected: 3,
            got: k.rank(),
            shape: k.shape().to_vec(),
        });
    }
    if v.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "v",
            expected: 3,
            got: v.rank(),
            shape: v.shape().to_vec(),
        });
    }
    for (name, view) in [("k", k), ("v", v)] {
        if !matches!(view.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
            problems.push(T0Error::DTypeMismatch {
                tensor: name,
                expected: vec![DType::F16, DType::Bf16, DType::F32],
                got: view.dtype(),
            });
        }
        if let Err(e) = check_row_major_layout(view.layout(), name) {
            problems.push(e);
        }
        if let Err(e) = view.validate_backing(name) {
            problems.push(e);
        }
    }
    if !matches!(op.cache_dtype, DType::F16 | DType::I8 | DType::E4m3) {
        problems.push(T0Error::InvalidAttribute {
            op: "state_write_kv",
            attribute: "cache_dtype",
            reason: format!("must be f16, i8, or e4m3; got {:?}", op.cache_dtype),
        });
    }
    // SI-45: PerBlock scale geometry is undefined (no record shape, block
    // size, or placement anywhere in Spec 1 §4.D / Spec 3 §2–§3), so T0 fails
    // it closed with a typed error rather than guessing a layout.
    if !matches!(op.scale_granularity, CacheScaleGranularity::PerTokenHead) {
        problems.push(T0Error::InvalidAttribute {
            op: "state_write_kv",
            attribute: "scale_granularity",
            reason: "PerBlock cache scale geometry is undefined (see SI-45); only PerTokenHead is supported".to_string(),
        });
    }
    if (group as usize) >= meta.num_groups() {
        problems.push(T0Error::InvalidAttribute {
            op: "state_write_kv",
            attribute: "group",
            reason: format!(
                "group {group} out of range for BatchMeta with {} groups",
                meta.num_groups()
            ),
        });
    }
    if k.rank() == 3 && v.rank() == 3 {
        let (kt, kh, kd) = (k.shape()[0], k.shape()[1], k.shape()[2]);
        let (vt, vh, _vd) = (v.shape()[0], v.shape()[1], v.shape()[2]);
        if kt != meta.total_tokens() {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "T",
                expected_from: "BatchMeta",
                expected: meta.total_tokens(),
                tensor: "k",
                got: kt,
            });
        }
        if vt != meta.total_tokens() {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "T",
                expected_from: "BatchMeta",
                expected: meta.total_tokens(),
                tensor: "v",
                got: vt,
            });
        }
        if kh != vh {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Hkv",
                expected_from: "k",
                expected: kh,
                tensor: "v",
                got: vh,
            });
        }
        let _ = kd;
    }
}

/// Scans destination slots for one write without mutating (fail-before-mutation).
///
/// Returns the per-token slot (`None` for `SLOT_NONE` pad skips) after
/// checking every slot is either skippable or inside the cache.
fn plan_write_slots(
    meta: &BatchMeta,
    group: u32,
    num_slots: usize,
    problems: &mut Vec<T0Error>,
) -> Option<Vec<Option<usize>>> {
    let t = meta.total_tokens();
    let mut slots = Vec::with_capacity(t);
    for tok in 0..t {
        // Total guard: `checked_slot` proves `group < G` and `tok < T`
        // before touching the asserting `BatchMeta::slot` accessor, so a
        // malformed group or token index is a typed problem, never a panic.
        let tok_u32 = match u32::try_from(tok) {
            Ok(v) => v,
            Err(_) => {
                problems.push(T0Error::ArithmeticOverflow {
                    op: "state_write_kv",
                    detail: format!("token index {tok} overflows u32"),
                });
                slots.push(None);
                continue;
            }
        };
        let raw = match checked_slot(meta, group, tok_u32) {
            Ok(v) => v,
            Err(e) => {
                problems.push(e);
                slots.push(None);
                continue;
            }
        };
        if raw == SLOT_NONE {
            slots.push(None);
            continue;
        }
        let slot = match u64_to_usize(u64::from(raw), "slot_map slot") {
            Ok(s) => s,
            Err(e) => {
                problems.push(e);
                slots.push(None);
                continue;
            }
        };
        if slot >= num_slots {
            problems.push(T0Error::RowIndexOutOfRange {
                op: "state_write_kv",
                tensor: "slot_map",
                position: tok,
                index: raw,
                upper_bound: num_slots,
            });
            slots.push(None);
        } else {
            slots.push(Some(slot));
        }
    }
    if problems.is_empty() {
        Some(slots)
    } else {
        None
    }
}

/// Writes K/V rows into a paged cache (Spec 1 §4.D, Spec 3 §3.2).
///
/// Signature: `k [T, Hkv, D], v [T, Hkv, Dv]` plus `slot_map` via the
/// structured `BatchMeta` input (SI-12) → (). `SLOT_NONE` slots are skipped
/// without mutation.
pub fn state_write_kv_paged(
    op: &StateWriteKvOp,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    meta: &BatchMeta,
    group: u32,
    cache: &mut KvPagedCache,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();
    validate_write_common(op, k, v, meta, group, &mut problems);
    if op.latent.is_some() {
        problems.push(T0Error::InvalidAttribute {
            op: "state_write_kv",
            attribute: "latent",
            reason: "paged write takes no latent spec; use a KvLatent cache".to_string(),
        });
    }
    if k.rank() == 3 && v.rank() == 3 {
        let (kh, kd) = (k.shape()[1], k.shape()[2]);
        let (vd, vt) = (v.shape()[2], v.shape()[0]);
        let _ = vt;
        if kh != cache.hkv {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Hkv",
                expected_from: "cache",
                expected: cache.hkv,
                tensor: "k",
                got: kh,
            });
        }
        if kd != cache.d {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "D",
                expected_from: "cache",
                expected: cache.d,
                tensor: "k",
                got: kd,
            });
        }
        if vd != cache.dv {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Dv",
                expected_from: "cache",
                expected: cache.dv,
                tensor: "v",
                got: vd,
            });
        }
    }
    // Non-finite inputs cannot be represented in quantized caches; collect
    // every offending (token, head) before mutating. F16 passes bit patterns
    // through, including infinities.
    if matches!(op.cache_dtype, DType::I8 | DType::E4m3) && k.rank() == 3 && v.rank() == 3 {
        let (t, h) = (k.shape()[0], k.shape()[1]);
        let (kd, vd) = (k.shape()[2], v.shape()[2]);
        for tok in 0..t {
            for head in 0..h {
                let row = match checked_addr(tok, h, head, "paged write validation row") {
                    Ok(v) => v,
                    Err(e) => {
                        problems.push(e);
                        continue;
                    }
                };
                let (Some(k_off), Some(v_off)) = (row.checked_mul(kd), row.checked_mul(vd)) else {
                    problems.push(T0Error::ArithmeticOverflow {
                        op: "state_write_kv",
                        detail: format!("paged write validation row {row} * dims overflows usize"),
                    });
                    continue;
                };
                let mut bad = 0usize;
                for d in 0..kd {
                    match k_off.checked_add(d) {
                        Some(i) if k.read_f32(i).is_finite() => {}
                        _ => bad += 1,
                    }
                }
                for d in 0..vd {
                    match v_off.checked_add(d) {
                        Some(i) if v.read_f32(i).is_finite() => {}
                        _ => bad += 1,
                    }
                }
                if bad > 0 {
                    problems.push(T0Error::InvalidAttribute {
                        op: "state_write_kv",
                        attribute: "k/v",
                        reason: format!(
                            "token {tok} head {head}: {bad} non-finite value(s) cannot be stored in {:?} cache",
                            op.cache_dtype
                        ),
                    });
                }
            }
        }
    }
    // The group bound was validated above; skip slot planning when it failed.
    // `plan_write_slots` additionally guards every access through
    // `checked_slot`, so a bad group is a typed problem either way.
    let slots = if (group as usize) < meta.num_groups() {
        plan_write_slots(meta, group, cache.num_slots(), &mut problems)
    } else {
        None
    };
    T0Error::from_problems(problems)?;
    // Total backstop: `plan_write_slots` returns `Some` exactly when its scan
    // found no problems, so reaching here with `None` means a logic bug, not
    // input. Refuse with a typed error rather than panicking.
    let slots = match slots {
        Some(s) => s,
        None => {
            return Err(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "slot_map",
                reason: "slot plan missing after successful validation".to_string(),
            });
        }
    };

    let h = k.shape()[1];
    let (kd, vd) = (k.shape()[2], v.shape()[2]);
    // Plan-then-commit: read every row (fallible address math, including a
    // possible e4m3 encode later) BEFORE the first cache byte changes, so any
    // failure refuses without partial mutation (CONVENTIONS.md §1.4). Row
    // bounds (tok < T, head < H, d < D) were proven by the shape validation
    // above; every address below is still built with checked math.
    let mut plan: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::new();
    for (tok, slot) in slots.iter().enumerate() {
        let Some(slot) = *slot else { continue };
        let mut k_rows = Vec::with_capacity(h * kd);
        let mut v_rows = Vec::with_capacity(h * vd);
        for head in 0..h {
            let row = checked_addr(tok, h, head, "paged write plan row")?;
            let k_off = row
                .checked_mul(kd)
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "state_write_kv",
                    detail: format!("paged write plan row {row} * {kd} overflows usize"),
                })?;
            let v_off = row
                .checked_mul(vd)
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "state_write_kv",
                    detail: format!("paged write plan row {row} * {vd} overflows usize"),
                })?;
            for d in 0..kd {
                let i = k_off
                    .checked_add(d)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "state_write_kv",
                        detail: "paged write plan k index overflows usize".to_string(),
                    })?;
                k_rows.push(k.read_f32(i));
            }
            for d in 0..vd {
                let i = v_off
                    .checked_add(d)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "state_write_kv",
                        detail: "paged write plan v index overflows usize".to_string(),
                    })?;
                v_rows.push(v.read_f32(i));
            }
        }
        plan.push((slot, k_rows, v_rows));
    }
    for (tok, (slot, k_flat, v_flat)) in plan.iter().enumerate() {
        let (slot, k_flat, v_flat) = (*slot, k_flat.as_slice(), v_flat.as_slice());
        match op.cache_dtype {
            DType::F16 => {
                for head in 0..h {
                    for d in 0..kd {
                        let dst = cache.k_offset(slot, head, d)?;
                        checked_store(
                            &mut cache.k_f16,
                            dst,
                            f32_to_f16_bits(flat_at(k_flat, head, kd, d)?),
                            "cache.k",
                        )?;
                    }
                    for d in 0..vd {
                        let dst = cache.v_offset(slot, head, d)?;
                        checked_store(
                            &mut cache.v_f16,
                            dst,
                            f32_to_f16_bits(flat_at(v_flat, head, vd, d)?),
                            "cache.v",
                        )?;
                    }
                }
            }
            DType::I8 => {
                for head in 0..h {
                    let k_row = flat_row(k_flat, head, kd)?;
                    let v_row = flat_row(v_flat, head, vd)?;
                    let (kq, ks) = quant_row_i8(k_row);
                    let (vq, vs) = quant_row_i8(v_row);
                    for (d, &q) in kq.iter().enumerate() {
                        let dst = cache.k_offset(slot, head, d)?;
                        checked_store(&mut cache.k_i8, dst, q, "cache.k")?;
                    }
                    for (d, &q) in vq.iter().enumerate() {
                        let dst = cache.v_offset(slot, head, d)?;
                        checked_store(&mut cache.v_i8, dst, q, "cache.v")?;
                    }
                    let kso = cache.scale_offset(slot, head)?;
                    let vso = cache.scale_offset(slot, head)?;
                    checked_store(
                        &mut cache.k_scales,
                        kso,
                        f32_to_f16_bits(ks),
                        "cache.k_scales",
                    )?;
                    checked_store(
                        &mut cache.v_scales,
                        vso,
                        f32_to_f16_bits(vs),
                        "cache.v_scales",
                    )?;
                }
            }
            DType::E4m3 => {
                for head in 0..h {
                    let k_row = flat_row(k_flat, head, kd)?;
                    let v_row = flat_row(v_flat, head, vd)?;
                    let (kq, ks) = quant_row_e4m3(k_row, "k", tok, head)?;
                    let (vq, vs) = quant_row_e4m3(v_row, "v", tok, head)?;
                    for (d, &q) in kq.iter().enumerate() {
                        let dst = cache.k_offset(slot, head, d)?;
                        checked_store(&mut cache.k_e4m3, dst, q, "cache.k")?;
                    }
                    for (d, &q) in vq.iter().enumerate() {
                        let dst = cache.v_offset(slot, head, d)?;
                        checked_store(&mut cache.v_e4m3, dst, q, "cache.v")?;
                    }
                    let kso = cache.scale_offset(slot, head)?;
                    let vso = cache.scale_offset(slot, head)?;
                    checked_store(
                        &mut cache.k_scales,
                        kso,
                        f32_to_f16_bits(ks),
                        "cache.k_scales",
                    )?;
                    checked_store(
                        &mut cache.v_scales,
                        vso,
                        f32_to_f16_bits(vs),
                        "cache.v_scales",
                    )?;
                }
            }
            _ => {
                return Err(T0Error::InvalidAttribute {
                    op: "attention",
                    attribute: "cache_dtype",
                    reason: "cache dtype must be f16, i8, or e4m3".to_string(),
                });
            }
        }
        checked_store(&mut cache.written, slot, true, "cache.written")?;
    }
    Ok(())
}

// DECISION(A1.7): latent writes require H == 1 on both operands and store
// head 0's rows; the producing A1.14 builder emits [T, 1, latent] /
// [T, 1, rope] while Spec 3 §3.2 states the regions have no head dimension,
// so a multi-head latent write has no defined destaggering. Rejected storing
// per-head copies (contradicts the no-head-dimension storage) and averaging
// heads (invents numerics). See SI-44.
fn check_latent_heads(shape: &[usize], tensor: &'static str, problems: &mut Vec<T0Error>) {
    if shape.len() == 3 && shape[1] != 1 {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "H",
            expected_from: "KvLatent (no head dimension)",
            expected: 1,
            tensor,
            got: shape[1],
        });
    }
}

/// Writes compressed latent + rope rows into a latent cache (Spec 1 §4.D,
/// Spec 3 §2, §3.2, SI-29).
///
/// Accepts the landed A1.14 canonical forms: the exact split pair
/// (`c_kv [T, 1, kv_lora_rank]`, `k_rope [T, 1, rope_dim]`) or the combined
/// form (operand 0 `[T, 1, kv_lora_rank + rope_dim]`, split at
/// `kv_lora_rank`; operand 1 must still match on T/H but its values are not
/// stored, see SI-44). The rope part is always stored as f16 per Spec 3 §2.
pub fn state_write_kv_latent(
    op: &StateWriteKvOp,
    c_kv: &TensorView<'_>,
    k_rope: &TensorView<'_>,
    meta: &BatchMeta,
    group: u32,
    cache: &mut KvLatentCache,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();
    validate_write_common(op, c_kv, k_rope, meta, group, &mut problems);
    let latent = match op.latent {
        Some(l) => l,
        None => {
            problems.push(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "latent",
                reason: "latent write requires a latent spec; use a KvPaged cache otherwise"
                    .to_string(),
            });
            // Total backstop: the pushed problem guarantees `from_problems`
            // below errors; the typed return keeps this path total even if a
            // future edit breaks that coupling.
            T0Error::from_problems(problems)?;
            return Err(T0Error::MissingOperand {
                op: "state_write_kv",
                operand: "latent",
                detail: "latent write requires a latent spec".to_string(),
            });
        }
    };
    if latent.kv_lora_rank == 0 || latent.rope_dim == 0 {
        problems.push(T0Error::InvalidAttribute {
            op: "state_write_kv",
            attribute: "latent",
            reason: format!(
                "kv_lora_rank ({}) and rope_dim ({}) must both be > 0",
                latent.kv_lora_rank, latent.rope_dim
            ),
        });
    }
    let rank = latent.kv_lora_rank as usize;
    let rope_dim = latent.rope_dim as usize;
    let combined = rank.checked_add(rope_dim);
    // Form classification mirrors the IR validation (SI-29): exact split or
    // combined operand 0; anything else is a typed shape failure.
    let mut is_split = false;
    let mut is_combined = false;
    if c_kv.rank() == 3 && k_rope.rank() == 3 {
        let (d0, d1) = (c_kv.shape()[2], k_rope.shape()[2]);
        is_split = d0 == rank && d1 == rope_dim;
        is_combined = Some(d0) == combined;
        if !is_split && !is_combined {
            problems.push(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "k/v",
                reason: format!(
                    "MLA latent dim {d0} / rotary dim {d1} must be the exact split pair (latent {rank} / rotary {rope_dim}) or the combined latent (rank {rank} + rope {rope_dim})"
                ),
            });
        }
        check_latent_heads(c_kv.shape(), "c_kv", &mut problems);
        check_latent_heads(k_rope.shape(), "k_rope", &mut problems);
        if rank != cache.latent {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "latent",
                expected_from: "cache",
                expected: cache.latent,
                tensor: "c_kv",
                got: rank,
            });
        }
        if rope_dim != cache.rope {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "rope",
                expected_from: "cache",
                expected: cache.rope,
                tensor: "k_rope",
                got: rope_dim,
            });
        }
    }
    if matches!(op.cache_dtype, DType::I8 | DType::E4m3) && c_kv.rank() == 3 && k_rope.rank() == 3 {
        let (t, h) = (c_kv.shape()[0], c_kv.shape()[1]);
        let d0 = c_kv.shape()[2];
        for tok in 0..t {
            for head in 0..h {
                let off = match checked_addr(tok, h, head, "latent write validation row")
                    .ok()
                    .and_then(|v| v.checked_mul(d0))
                {
                    Some(v) => v,
                    None => {
                        problems.push(T0Error::ArithmeticOverflow {
                            op: "state_write_kv",
                            detail: format!(
                                "latent write validation row ({tok} * {h} + {head}) * {d0} overflows usize"
                            ),
                        });
                        continue;
                    }
                };
                let mut bad = 0usize;
                for d in 0..d0 {
                    match off.checked_add(d) {
                        Some(i) if c_kv.read_f32(i).is_finite() => {}
                        _ => bad += 1,
                    }
                }
                if bad > 0 {
                    problems.push(T0Error::InvalidAttribute {
                        op: "state_write_kv",
                        attribute: "c_kv",
                        reason: format!(
                            "token {tok} head {head}: {bad} non-finite value(s) cannot be stored in {:?} cache",
                            op.cache_dtype
                        ),
                    });
                }
            }
        }
    }
    let slots = if (group as usize) < meta.num_groups() {
        plan_write_slots(meta, group, cache.num_slots(), &mut problems)
    } else {
        None
    };
    T0Error::from_problems(problems)?;
    // Total backstop, as in `state_write_kv_paged`: `None` here means a logic
    // bug, refused with a typed error rather than a panic.
    let slots = match slots {
        Some(s) => s,
        None => {
            return Err(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "slot_map",
                reason: "slot plan missing after successful validation".to_string(),
            });
        }
    };

    let d0 = c_kv.shape()[2];
    let d1 = k_rope.shape()[2];
    // Plan-then-commit: build every row (fallible address math and e4m3
    // encode) BEFORE the first cache byte changes (CONVENTIONS.md §1.4).
    // Latent rows ignore the head stride because H == 1 was proven above.
    let mut plan: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::new();
    for (tok, slot) in slots.iter().enumerate() {
        let Some(slot) = *slot else { continue };
        let latent_base = tok
            .checked_mul(d0)
            .ok_or_else(|| T0Error::ArithmeticOverflow {
                op: "state_write_kv",
                detail: format!("latent write plan row {tok} * {d0} overflows usize"),
            })?;
        // Latent row: split form reads operand 0 directly; combined form
        // reads the head of operand 0. Rope row: split form reads operand 1;
        // combined form reads the tail of operand 0 (SI-44). A neither-form
        // row cannot reach here (validation refused it), and stays a typed
        // refusal rather than a debug-only tripwire or a silent combined read.
        let latent_row: Vec<f32> = if is_split || is_combined {
            let mut row = Vec::with_capacity(rank);
            for d in 0..rank {
                let i = latent_base
                    .checked_add(d)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "state_write_kv",
                        detail: "latent write plan index overflows usize".to_string(),
                    })?;
                row.push(c_kv.read_f32(i));
            }
            row
        } else {
            return Err(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "k/v",
                reason: "latent form is neither the split pair nor the combined form".to_string(),
            });
        };
        let rope_row: Vec<f32> = if is_split {
            let base = tok
                .checked_mul(d1)
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "state_write_kv",
                    detail: format!("rope write plan row {tok} * {d1} overflows usize"),
                })?;
            let mut row = Vec::with_capacity(rope_dim);
            for d in 0..rope_dim {
                let i = base
                    .checked_add(d)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "state_write_kv",
                        detail: "rope write plan index overflows usize".to_string(),
                    })?;
                row.push(k_rope.read_f32(i));
            }
            row
        } else if is_combined {
            let base =
                latent_base
                    .checked_add(rank)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "state_write_kv",
                        detail: "combined rope write plan base overflows usize".to_string(),
                    })?;
            let mut row = Vec::with_capacity(rope_dim);
            for d in 0..rope_dim {
                let i = base
                    .checked_add(d)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "state_write_kv",
                        detail: "rope write plan index overflows usize".to_string(),
                    })?;
                row.push(c_kv.read_f32(i));
            }
            row
        } else {
            return Err(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "k/v",
                reason: "latent form is neither the split pair nor the combined form".to_string(),
            });
        };
        plan.push((slot, latent_row, rope_row));
    }
    for (tok, (slot, latent_row, rope_row)) in plan.iter().enumerate() {
        let (slot, latent_row, rope_row) = (*slot, latent_row.as_slice(), rope_row.as_slice());
        match op.cache_dtype {
            DType::F16 => {
                for (d, &val) in latent_row.iter().enumerate() {
                    let dst = checked_addr(slot, cache.latent, d, "latent write")?;
                    checked_store(
                        &mut cache.latent_f16,
                        dst,
                        f32_to_f16_bits(val),
                        "cache.latent",
                    )?;
                }
            }
            DType::I8 => {
                let (q, s) = quant_row_i8(latent_row);
                for (d, &qv) in q.iter().enumerate() {
                    let dst = checked_addr(slot, cache.latent, d, "latent write")?;
                    checked_store(&mut cache.latent_i8, dst, qv, "cache.latent")?;
                }
                checked_store(
                    &mut cache.latent_scales,
                    slot,
                    f32_to_f16_bits(s),
                    "cache.latent_scales",
                )?;
            }
            DType::E4m3 => {
                let (q, s) = quant_row_e4m3(latent_row, "c_kv", tok, 0)?;
                for (d, &qv) in q.iter().enumerate() {
                    let dst = checked_addr(slot, cache.latent, d, "latent write")?;
                    checked_store(&mut cache.latent_e4m3, dst, qv, "cache.latent")?;
                }
                checked_store(
                    &mut cache.latent_scales,
                    slot,
                    f32_to_f16_bits(s),
                    "cache.latent_scales",
                )?;
            }
            _ => {
                return Err(T0Error::InvalidAttribute {
                    op: "attention",
                    attribute: "cache_dtype",
                    reason: "cache dtype must be f16, i8, or e4m3".to_string(),
                });
            }
        }
        for (d, &val) in rope_row.iter().enumerate() {
            let dst = checked_addr(slot, cache.rope, d, "rope write")?;
            checked_store(&mut cache.rope_f16, dst, f32_to_f16_bits(val), "cache.rope")?;
        }
        checked_store(&mut cache.written, slot, true, "cache.written")?;
    }
    Ok(())
}

/// Dispatches `state_write_kv` to the cache matching the handle kind
/// (Spec 1 §2.6, §4.D).
pub fn state_write_kv(
    op: &StateWriteKvOp,
    k: &TensorView<'_>,
    v: &TensorView<'_>,
    meta: &BatchMeta,
    group: u32,
    cache: &mut KvCache,
) -> Result<(), T0Error> {
    use r9v_ir::StateKind;
    match (op.handle.kind(), cache) {
        (StateKind::KvPaged, KvCache::Paged(paged)) => {
            state_write_kv_paged(op, k, v, meta, group, paged)
        }
        (StateKind::KvLatent, KvCache::Latent(latent)) => {
            state_write_kv_latent(op, k, v, meta, group, latent)
        }
        (expected, actual) => {
            let got = match actual {
                KvCache::Paged(_) => StateKind::KvPaged,
                KvCache::Latent(_) => StateKind::KvLatent,
            };
            Err(T0Error::InvalidAttribute {
                op: "state_write_kv",
                attribute: "handle",
                reason: format!(
                    "handle kind {expected:?} does not match cache {got:?}; paged writes need KvPaged, latent writes need KvLatent"
                ),
            })
        }
    }
}

// ----------------------------------------------------------------------------
// attention (Spec 1 §4.D, §6.3)
// ----------------------------------------------------------------------------

/// One flat query token and its owning sequence (Spec 1 §2.5, §4.D).
struct QueryInfo {
    seq: usize,
    /// Absolute position (`ctx_len + query offset`).
    abs_pos: u64,
    /// Tokens already in state before this step.
    ctx: u64,
    /// Query length of this sequence.
    qlen: usize,
}

/// Walks `query_len` in ascending sequence order (Spec 1 §2.5).
///
/// Fails with a typed error when the lengths do not cover exactly `T` tokens.
fn plan_queries(meta: &BatchMeta) -> Result<Vec<QueryInfo>, T0Error> {
    let mut out = Vec::with_capacity(meta.total_tokens());
    let mut flat = 0usize;
    for (seq, (&ctx, &qlen)) in meta
        .ctx_len()
        .iter()
        .zip(meta.query_len().iter())
        .enumerate()
    {
        // BatchMeta stores lengths as u32; the narrowing to usize is lossless
        // on every supported target, and stays a typed refusal otherwise.
        let qlen_usize = usize::try_from(qlen).map_err(|_| T0Error::ArithmeticOverflow {
            op: "attention",
            detail: format!("query length {qlen} overflows usize"),
        })?;
        for i in 0..qlen {
            let abs_pos = u64::from(ctx).checked_add(u64::from(i)).ok_or_else(|| {
                T0Error::ArithmeticOverflow {
                    op: "attention",
                    detail: format!("absolute position {ctx} + {i} overflows u64"),
                }
            })?;
            out.push(QueryInfo {
                seq,
                abs_pos,
                ctx: u64::from(ctx),
                qlen: qlen_usize,
            });
            flat += 1;
        }
    }
    if flat != meta.total_tokens() {
        return Err(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "query_len",
            reason: format!(
                "query lengths cover {flat} tokens but BatchMeta T is {}",
                meta.total_tokens()
            ),
        });
    }
    Ok(out)
}

// DECISION(A1.7): retention for every mask kind is the union of the pinned
// prefix (p < sinks) and the window (p >= window_start). Under Causal with
// All retention both conditions are trivially true, so sinks > 0 needs no
// special case and never fails closed; under CausalWindow it selects exactly
// the Spec 3 Sink(n)+Window(w) survivor set. Rejected learned-sink logits:
// Spec 1 §6.3 names them but no sink-value operand exists in the op
// signature, so there is nothing to add into the softmax (SI-43).
fn is_retained(sinks: u64, window_start: u64, pos: u64) -> bool {
    pos < sinks || pos >= window_start
}

// DECISION(A1.7): Tree visibility is context-always (subject to retention)
// plus query-query governed solely by the ancestors bit; columns index
// within-sequence query offsets (column j of token tok covers the query
// token at offset j of the same sequence). Rejected re-applying absolute
// causal order to tree pairs because the ancestors matrix already is the
// causal structure for drafts ("uses BatchMeta.tree for masking, not
// positions", Spec 1 §4.D).
fn tree_query_visible(
    tree: &r9v_ir::TreeMask,
    tok: usize,
    query_col: usize,
) -> Result<bool, T0Error> {
    if query_col >= tree.t_max() as usize {
        return Err(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "tree",
            reason: format!(
                "query offset {query_col} exceeds tree t_max {} for token {tok}",
                tree.t_max()
            ),
        });
    }
    Ok(tree.is_ancestor(tok, query_col))
}

/// Resolves a logical position to a cache slot (`None` = sentinel-evicted hole).
///
/// Never reads evicted data and never synthesizes zeros: holes stay holes.
fn slot_for_position(
    meta: &BatchMeta,
    group: u32,
    seq: usize,
    pos: u64,
    num_slots: usize,
) -> Result<Option<usize>, T0Error> {
    let block_idx = pos / u64::from(BLOCK_TOKENS as u32);
    let lane = (pos % u64::from(BLOCK_TOKENS as u32)) as u32;
    if block_idx >= u64::from(meta.max_blocks()) {
        return Err(T0Error::RowIndexOutOfRange {
            op: "attention",
            tensor: "block_table",
            // Error report only: saturate rather than wrap.
            position: usize::try_from(pos).unwrap_or(usize::MAX),
            index: u32::try_from(block_idx).unwrap_or(u32::MAX),
            upper_bound: meta.max_blocks() as usize,
        });
    }
    let seq_u32 = u32::try_from(seq).map_err(|_| T0Error::ArithmeticOverflow {
        op: "attention",
        detail: format!("sequence index {seq} overflows u32"),
    })?;
    // Total guard: `block_idx < max_blocks (u32)` was proven above so the
    // narrowing is lossless, and `checked_block` proves `group < G` and
    // `seq < S` before touching the asserting accessor.
    let block_id = checked_block(meta, group, seq_u32, block_idx as u32)?;
    if block_id == BLOCK_TABLE_SENTINEL {
        return Ok(None);
    }
    let slot = u64::from(block_id)
        .checked_mul(u64::from(BLOCK_TOKENS as u32))
        .and_then(|v| v.checked_add(u64::from(lane)))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "attention",
            detail: format!("slot {block_id} * 32 + {lane} overflows u64"),
        })?;
    let slot = u64_to_usize(slot, "cache slot")?;
    if slot >= num_slots {
        return Err(T0Error::RowIndexOutOfRange {
            op: "attention",
            tensor: "cache",
            // Error report only: saturate rather than wrap.
            position: usize::try_from(pos).unwrap_or(usize::MAX),
            index: block_id,
            upper_bound: num_slots,
        });
    }
    Ok(Some(slot))
}

/// One visible key position: its cache slot (the ancestors bit for tree
/// query pairs was already applied when the plan was built).
#[derive(Debug, Clone, Copy)]
struct VisiblePos {
    slot: usize,
}

/// Visibility pre-pass shared by paged and MLA attention.
///
/// Resolves every key position in ascending order, checks writtenness of
/// retained slots, and proves a non-empty visible set per query — all before
/// any compute (fail-before-mutation, never silent zeros).
#[allow(clippy::too_many_arguments)]
fn plan_visible(
    meta: &BatchMeta,
    group: u32,
    queries: &[QueryInfo],
    sinks: u64,
    is_tree: bool,
    num_slots: usize,
    is_written: &dyn Fn(usize) -> bool,
    problems: &mut Vec<T0Error>,
) -> Vec<Vec<VisiblePos>> {
    let tree = meta.tree();
    let mut plans: Vec<Vec<VisiblePos>> = Vec::with_capacity(queries.len());
    for (tok, query) in queries.iter().enumerate() {
        // Total guard: `checked_window` proves `group < G` and `seq < S`
        // before touching the asserting `BatchMeta::window` accessor, so a
        // malformed (group, seq) pair is a typed problem, never a panic.
        let seq_u32 = match u32::try_from(query.seq) {
            Ok(v) => v,
            Err(_) => {
                problems.push(T0Error::ArithmeticOverflow {
                    op: "attention",
                    detail: format!("sequence index {} overflows u32", query.seq),
                });
                plans.push(Vec::new());
                continue;
            }
        };
        let window_start = match checked_window(meta, group, seq_u32) {
            Ok(w) => u64::from(w),
            Err(e) => {
                problems.push(e);
                plans.push(Vec::new());
                continue;
            }
        };
        let mut visible = Vec::new();
        // Causal masks see p <= abs_pos; tree masks see context plus
        // ancestors (the ancestors matrix is the causal structure there).
        let p_end = if is_tree {
            let qlen_u64 = match u64::try_from(query.qlen) {
                Ok(v) => v,
                Err(_) => {
                    problems.push(T0Error::ArithmeticOverflow {
                        op: "attention",
                        detail: format!("query length {} overflows u64", query.qlen),
                    });
                    plans.push(Vec::new());
                    continue;
                }
            };
            match query.ctx.checked_add(qlen_u64) {
                Some(v) => v,
                None => {
                    problems.push(T0Error::ArithmeticOverflow {
                        op: "attention",
                        detail: format!(
                            "tree query end {} + {} overflows u64",
                            query.ctx, qlen_u64
                        ),
                    });
                    plans.push(Vec::new());
                    continue;
                }
            }
        } else {
            match query.abs_pos.checked_add(1) {
                Some(v) => v,
                None => {
                    problems.push(T0Error::ArithmeticOverflow {
                        op: "attention",
                        detail: format!("absolute position {} + 1 overflows u64", query.abs_pos),
                    });
                    plans.push(Vec::new());
                    continue;
                }
            }
        };
        let mut p = 0u64;
        while p < p_end {
            if !is_retained(sinks, window_start, p) {
                p += 1;
                continue;
            }
            let in_context = p < query.ctx;
            if !in_context && !is_tree && p > query.abs_pos {
                // Non-tree query-query pairs obey absolute causal order.
                p += 1;
                continue;
            }
            if !in_context && is_tree {
                let Some(tree) = tree else {
                    // Missing-tree error already recorded by validation;
                    // skip without touching the asserting accessor.
                    p += 1;
                    continue;
                };
                // Internal invariant: query tokens satisfy p >= ctx, so the
                // subtraction cannot underflow.
                match tree_query_visible(tree, tok, (p - query.ctx) as usize) {
                    Ok(true) => {}
                    Ok(false) => {
                        p += 1;
                        continue;
                    }
                    Err(e) => {
                        problems.push(e);
                        p += 1;
                        continue;
                    }
                }
            }
            match slot_for_position(meta, group, query.seq, p, num_slots) {
                Ok(None) => {
                    // Sentinel-evicted hole: skipped, never read as zero.
                }
                Ok(Some(slot)) => {
                    if !is_written(slot) {
                        problems.push(T0Error::InvalidAttribute {
                            op: "attention",
                            attribute: "cache",
                            reason: format!(
                                "sequence {} position {p} is retained but its cache slot {slot} was never written; run state_write_kv first",
                                query.seq
                            ),
                        });
                    } else {
                        visible.push(VisiblePos { slot });
                    }
                }
                Err(e) => problems.push(e),
            }
            p += 1;
        }
        if visible.is_empty() && problems.is_empty() {
            problems.push(T0Error::InvalidDistribution {
                op: "attention",
                seq: query.seq,
                // Error report only: saturate rather than wrap.
                pos: usize::try_from(query.abs_pos).unwrap_or(usize::MAX),
                sum: 0.0,
            });
        }
        plans.push(visible);
    }
    plans
}

/// Validates the shared `attention` preconditions (Spec 1 §4.D).
///
/// Collects every problem; performs no reads of cache data and no mutation.
#[allow(clippy::too_many_arguments)]
fn validate_attention_common(
    op: &AttentionOp,
    q: &TensorView<'_>,
    o: &TensorViewMut<'_>,
    meta: &BatchMeta,
    group: u32,
    problems: &mut Vec<T0Error>,
) {
    if let Err(e) = q.validate_backing("q") {
        problems.push(e);
    }
    if let Err(e) = o.validate_backing("o") {
        problems.push(e);
    }
    if let Err(e) = check_row_major_layout(q.layout(), "q") {
        problems.push(e);
    }
    if let Err(e) = check_row_major_layout(o.layout(), "o") {
        problems.push(e);
    }
    if q.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "q",
            expected: 3,
            got: q.rank(),
            shape: q.shape().to_vec(),
        });
    }
    if o.rank() != 3 {
        problems.push(T0Error::RankMismatch {
            tensor: "o",
            expected: 3,
            got: o.rank(),
            shape: o.shape().to_vec(),
        });
    }
    if !matches!(q.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::DTypeMismatch {
            tensor: "q",
            expected: vec![DType::F16, DType::Bf16, DType::F32],
            got: q.dtype(),
        });
    }
    if !matches!(op.out_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "out_dtype",
            reason: format!("must be f16, bf16, or f32; got {:?}", op.out_dtype),
        });
    }
    if o.dtype() != op.out_dtype {
        problems.push(T0Error::DTypeMismatch {
            tensor: "o",
            expected: vec![op.out_dtype],
            got: o.dtype(),
        });
    }
    if !op.softmax_scale.is_finite() || op.softmax_scale <= 0.0 {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "softmax_scale",
            reason: format!("must be finite and > 0, got {}", op.softmax_scale),
        });
    }
    if let AttentionMask::CausalWindow(w) = op.mask {
        if w == 0 {
            problems.push(T0Error::InvalidAttribute {
                op: "attention",
                attribute: "mask",
                reason: "causal window must be > 0".to_string(),
            });
        }
    }
    if let Some(c) = op.logit_softcap.filter(|c| !c.is_finite() || *c <= 0.0) {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "logit_softcap",
            reason: format!("must be finite and > 0, got {c}"),
        });
    }
    if matches!(op.mask, AttentionMask::Tree) && meta.tree().is_none() {
        problems.push(T0Error::MissingOperand {
            op: "attention",
            operand: "BatchMeta.tree",
            detail: "Tree mask requires a TreeMask in BatchMeta".to_string(),
        });
    }
    if (group as usize) >= meta.num_groups() {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "group",
            reason: format!(
                "group {group} out of range for BatchMeta with {} groups",
                meta.num_groups()
            ),
        });
    }
    if q.rank() == 3 && o.rank() == 3 {
        let (qt, qh, qd) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let (ot, oh, od) = (o.shape()[0], o.shape()[1], o.shape()[2]);
        if qt != meta.total_tokens() {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "T",
                expected_from: "BatchMeta",
                expected: meta.total_tokens(),
                tensor: "q",
                got: qt,
            });
        }
        if ot != meta.total_tokens() {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "T",
                expected_from: "BatchMeta",
                expected: meta.total_tokens(),
                tensor: "o",
                got: ot,
            });
        }
        if qh != oh {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "H",
                expected_from: "q",
                expected: qh,
                tensor: "o",
                got: oh,
            });
        }
        if qh == 0 {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "H",
                expected_from: "q",
                expected: 1,
                tensor: "q",
                got: 0,
            });
        }
        let _ = (qd, od);
    }
}

/// Paged-attention read and compute over block tables (Spec 1 §4.D, §6.3).
///
/// Covers decode (`query_len == 1`), spec verify (`1 < query_len <= 16`),
/// and prefill chunks through one op: for each sequence, its query rows
/// attend to `ctx_len + query_len` positions through `block_table`.
/// GQA/MQA group query head `h` onto KV head `h / (H / Hkv)`; `H % Hkv == 0`
/// is required with a typed failure. QK/PV accumulate in f32 and the softmax
/// runs online (f32 max/sum) in ascending logical block/position order.
///
/// The cache must already hold every retained position (callers run
/// [`state_write_kv`] first for both the fused decode launch and the
/// separate prefill write). Results stage into scratch and commit to `o`
/// only after every row is computed.
pub fn attention_paged(
    op: &AttentionOp,
    q: &TensorView<'_>,
    meta: &BatchMeta,
    group: u32,
    cache: &KvPagedCache,
    o: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();
    validate_attention_common(op, q, o, meta, group, &mut problems);
    use r9v_ir::StateKind;
    if op.handle.kind() != StateKind::KvPaged {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "handle",
            reason: format!(
                "paged attention requires a KvPaged handle, got {:?}",
                op.handle.kind()
            ),
        });
    }
    if op.mla.is_some() {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "mla",
            reason: "paged attention takes no MLA spec; use attention_mla with a KvLatent cache"
                .to_string(),
        });
    }
    let hkv = cache.hkv;
    if hkv == 0 {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "cache",
            reason: "paged cache Hkv must be > 0".to_string(),
        });
    }
    if q.rank() == 3 && o.rank() == 3 {
        let (qh, qd) = (q.shape()[1], q.shape()[2]);
        let od = o.shape()[2];
        if hkv == 0 || qh % hkv != 0 {
            problems.push(T0Error::InvalidAttribute {
                op: "attention",
                attribute: "qkv_heads",
                reason: format!("H ({qh}) must be a multiple of Hkv ({hkv}) for GQA/MQA grouping"),
            });
        }
        if qd != cache.d {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "D",
                expected_from: "cache",
                expected: cache.d,
                tensor: "q",
                got: qd,
            });
        }
        if od != cache.dv {
            problems.push(T0Error::DimensionMismatch {
                dim_name: "Dv",
                expected_from: "cache",
                expected: cache.dv,
                tensor: "o",
                got: od,
            });
        }
    }
    // Group-gated accessors below route through `checked_*` wrappers, but a
    // bad group is still refused up front with the collected problems.
    if (group as usize) >= meta.num_groups() {
        T0Error::from_problems(problems)?;
        // Total backstop: validation pushed the group problem above, so
        // `from_problems` errors; the typed return keeps this path total.
        return Err(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "group",
            reason: format!(
                "group {group} out of range for BatchMeta with {} groups",
                meta.num_groups()
            ),
        });
    }
    let queries = match plan_queries(meta) {
        Ok(qs) => qs,
        Err(e) => {
            problems.push(e);
            T0Error::from_problems(std::mem::take(&mut problems))?;
            // Total backstop: the pushed problem guarantees `from_problems`
            // errors; the typed return keeps this path total.
            return Err(T0Error::InvalidAttribute {
                op: "attention",
                attribute: "query_len",
                reason: "query plan failed (see aggregated problems)".to_string(),
            });
        }
    };
    let sinks = u64::from(op.sinks);
    let is_tree = matches!(op.mask, AttentionMask::Tree);
    let scale = op.softmax_scale;
    let softcap = op.logit_softcap;

    let plans = plan_visible(
        meta,
        group,
        &queries,
        sinks,
        is_tree,
        cache.num_slots(),
        &|slot| cache.is_written(slot),
        &mut problems,
    );
    T0Error::from_problems(problems)?;

    let (t, h) = (q.shape()[0], q.shape()[1]);
    let (d, dv) = (cache.d, cache.dv);
    // Proven by validation above: H > 0, Hkv > 0, H % Hkv == 0, so
    // `group_size >= 1` and the `head / group_size` grouping below holds.
    let group_size = h / hkv;
    let out_elems = t
        .checked_mul(h)
        .and_then(|v| v.checked_mul(dv))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "attention",
            detail: format!("output elements {t} * {h} * {dv} overflows usize"),
        })?;
    let mut scratch = vec![0.0f32; out_elems];
    let mut acc = vec![0.0f32; dv];
    for (tok, (query, visible)) in queries.iter().zip(plans.iter()).enumerate() {
        let _ = query;
        for head in 0..h {
            let kv_head = head / group_size;
            let q_base = checked_addr(
                checked_addr(tok, h, head, "paged attention query row")?,
                d,
                0,
                "paged attention query base",
            )?;
            // Online softmax in f32, ascending position order (Spec 1 §6.3).
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for v in acc.iter_mut() {
                *v = 0.0;
            }
            for pos in visible.iter() {
                let mut s = 0.0f32;
                for dim in 0..d {
                    // Reads were proven written in the visibility pass; any
                    // residual failure is a typed refusal, never a panic.
                    let kv = cache.read_k_f32(pos.slot, kv_head, dim)?;
                    let qi =
                        q_base
                            .checked_add(dim)
                            .ok_or_else(|| T0Error::ArithmeticOverflow {
                                op: "attention",
                                detail: "paged attention query index overflows usize".to_string(),
                            })?;
                    s += q.read_f32(qi) * kv;
                }
                s *= scale;
                if let Some(cap) = softcap {
                    s = cap * (s / cap).tanh();
                }
                let m_new = m.max(s);
                let a = (m - m_new).exp();
                let b = (s - m_new).exp();
                l = l * a + b;
                for (j, slot) in acc.iter_mut().enumerate() {
                    let vv = cache.read_v_f32(pos.slot, kv_head, j)?;
                    *slot = *slot * a + b * vv;
                }
                m = m_new;
            }
            let o_base = checked_addr(
                checked_addr(tok, h, head, "paged attention output row")?,
                dv,
                0,
                "paged attention output base",
            )?;
            for j in 0..dv {
                let idx = o_base
                    .checked_add(j)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "attention",
                        detail: "paged attention output index overflows usize".to_string(),
                    })?;
                let a = acc
                    .get(j)
                    .copied()
                    .ok_or_else(|| T0Error::BufferLengthMismatch {
                        tensor: "attention.acc",
                        buffer_len: acc.len(),
                        expected_len: idx.saturating_add(1),
                        shape: vec![acc.len()],
                    })?;
                checked_store(&mut scratch, idx, a / l, "attention.scratch")?;
            }
        }
    }
    for (i, &val) in scratch.iter().enumerate() {
        o.write_f32(i, val);
    }
    Ok(())
}

// DECISION(A1.7): T0 implements the absorbed MLA form: it requires
// qk_nope_dim == kv_lora_rank, qk_rope_dim == rope_dim, and
// v_dim == kv_lora_rank, failing closed with typed errors otherwise. The IR
// admits non-absorbed dims (e.g. nope 128 against rank 512) but the op
// carries no projection operands to map between them, so any T0 lowering of
// those graphs would invent numerics; rejected truncation/padding of the
// latent (SI-46). Scores are scale * (q_nope . c_kv + q_rope . k_rope) with
// values c_kv, all in f32 with the same online softmax as paged attention.
fn validate_mla_dims(
    op: &AttentionOp,
    qd: usize,
    od: usize,
    cache: &KvLatentCache,
    problems: &mut Vec<T0Error>,
) {
    let Some(mla) = op.mla else {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "mla",
            reason: "latent attention requires an MLA spec".to_string(),
        });
        return;
    };
    let nope = mla.qk_nope_dim as usize;
    let rope = mla.qk_rope_dim as usize;
    let vdim = mla.v_dim as usize;
    let rank = mla.kv_lora_rank as usize;
    if nope.checked_add(rope) != Some(qd) {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "D",
            expected_from: "mla.nope+rope",
            expected: nope.saturating_add(rope),
            tensor: "q",
            got: qd,
        });
    }
    if od != vdim {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "Dv",
            expected_from: "mla.v_dim",
            expected: vdim,
            tensor: "o",
            got: od,
        });
    }
    if nope != rank {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "mla.qk_nope_dim",
            reason: format!(
                "absorbed MLA requires qk_nope_dim ({nope}) == kv_lora_rank ({rank}); see SI-46"
            ),
        });
    }
    if rope != cache.rope {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "rope",
            expected_from: "cache",
            expected: cache.rope,
            tensor: "mla.qk_rope_dim",
            got: rope,
        });
    }
    if rank != cache.latent {
        problems.push(T0Error::DimensionMismatch {
            dim_name: "latent",
            expected_from: "cache",
            expected: cache.latent,
            tensor: "mla.kv_lora_rank",
            got: rank,
        });
    }
    if vdim != rank {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "mla.v_dim",
            reason: format!(
                "absorbed MLA requires v_dim ({vdim}) == kv_lora_rank ({rank}); see SI-46"
            ),
        });
    }
}

/// MLA attention over a latent cache (Spec 1 §4.D, §6.3, Spec 8 §3.1).
///
/// Same masking/retention/ordering contract as [`attention_paged`]; the per
/// position key is `(c_kv, k_rope)` shared by all query heads (the latent
/// cache has no head dimension, so `H % 1 == 0` always holds).
pub fn attention_mla(
    op: &AttentionOp,
    q: &TensorView<'_>,
    meta: &BatchMeta,
    group: u32,
    cache: &KvLatentCache,
    o: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    let mut problems = Vec::new();
    validate_attention_common(op, q, o, meta, group, &mut problems);
    use r9v_ir::StateKind;
    if op.handle.kind() != StateKind::KvLatent {
        problems.push(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "handle",
            reason: format!(
                "MLA attention requires a KvLatent handle, got {:?}",
                op.handle.kind()
            ),
        });
    }
    if q.rank() == 3 && o.rank() == 3 {
        validate_mla_dims(op, q.shape()[2], o.shape()[2], cache, &mut problems);
    }
    // Group-gated accessors below route through `checked_*` wrappers, but a
    // bad group is still refused up front with the collected problems.
    if (group as usize) >= meta.num_groups() {
        T0Error::from_problems(problems)?;
        // Total backstop: validation pushed the group problem above, so
        // `from_problems` errors; the typed return keeps this path total.
        return Err(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "group",
            reason: format!(
                "group {group} out of range for BatchMeta with {} groups",
                meta.num_groups()
            ),
        });
    }
    let queries = match plan_queries(meta) {
        Ok(qs) => qs,
        Err(e) => {
            problems.push(e);
            T0Error::from_problems(std::mem::take(&mut problems))?;
            // Total backstop: the pushed problem guarantees `from_problems`
            // errors; the typed return keeps this path total.
            return Err(T0Error::InvalidAttribute {
                op: "attention",
                attribute: "query_len",
                reason: "query plan failed (see aggregated problems)".to_string(),
            });
        }
    };
    let Some(mla) = op.mla else {
        T0Error::from_problems(problems)?;
        // Total backstop: validation recorded the missing MLA spec above, so
        // `from_problems` errors; the typed return keeps this path total.
        return Err(T0Error::MissingOperand {
            op: "attention",
            operand: "mla",
            detail: "latent attention requires an MLA spec".to_string(),
        });
    };
    let nope = mla.qk_nope_dim as usize;
    let rope_dim = mla.qk_rope_dim as usize;
    let rank = mla.kv_lora_rank as usize;
    let sinks = u64::from(op.sinks);
    let is_tree = matches!(op.mask, AttentionMask::Tree);
    let scale = op.softmax_scale;
    let softcap = op.logit_softcap;

    let plans = plan_visible(
        meta,
        group,
        &queries,
        sinks,
        is_tree,
        cache.num_slots(),
        &|slot| cache.is_written(slot),
        &mut problems,
    );
    T0Error::from_problems(problems)?;

    let (t, h) = (q.shape()[0], q.shape()[1]);
    let out_elems = t
        .checked_mul(h)
        .and_then(|v| v.checked_mul(rank))
        .ok_or_else(|| T0Error::ArithmeticOverflow {
            op: "attention",
            detail: format!("output elements {t} * {h} * {rank} overflows usize"),
        })?;
    let mut scratch = vec![0.0f32; out_elems];
    let mut acc = vec![0.0f32; rank];
    for (tok, visible) in plans.iter().enumerate() {
        for head in 0..h {
            let qd = nope
                .checked_add(rope_dim)
                .ok_or_else(|| T0Error::ArithmeticOverflow {
                    op: "attention",
                    detail: format!("MLA query width {nope} + {rope_dim} overflows usize"),
                })?;
            let q_base = checked_addr(
                checked_addr(tok, h, head, "MLA attention query row")?,
                qd,
                0,
                "MLA attention query base",
            )?;
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            for v in acc.iter_mut() {
                *v = 0.0;
            }
            for pos in visible.iter() {
                let mut s = 0.0f32;
                for dim in 0..nope {
                    // Reads were proven written in the visibility pass; any
                    // residual failure is a typed refusal, never a panic.
                    let c = cache.read_latent_f32(pos.slot, dim)?;
                    let qi =
                        q_base
                            .checked_add(dim)
                            .ok_or_else(|| T0Error::ArithmeticOverflow {
                                op: "attention",
                                detail: "MLA attention query index overflows usize".to_string(),
                            })?;
                    s += q.read_f32(qi) * c;
                }
                for dim in 0..rope_dim {
                    let r = cache.read_rope_f32(pos.slot, dim)?;
                    let qi = q_base
                        .checked_add(nope)
                        .and_then(|v| v.checked_add(dim))
                        .ok_or_else(|| T0Error::ArithmeticOverflow {
                            op: "attention",
                            detail: "MLA attention rope index overflows usize".to_string(),
                        })?;
                    s += q.read_f32(qi) * r;
                }
                s *= scale;
                if let Some(cap) = softcap {
                    s = cap * (s / cap).tanh();
                }
                let m_new = m.max(s);
                let a = (m - m_new).exp();
                let b = (s - m_new).exp();
                l = l * a + b;
                for (j, slot) in acc.iter_mut().enumerate() {
                    let c = cache.read_latent_f32(pos.slot, j)?;
                    *slot = *slot * a + b * c;
                }
                m = m_new;
            }
            let o_base = checked_addr(
                checked_addr(tok, h, head, "MLA attention output row")?,
                rank,
                0,
                "MLA attention output base",
            )?;
            for j in 0..rank {
                let idx = o_base
                    .checked_add(j)
                    .ok_or_else(|| T0Error::ArithmeticOverflow {
                        op: "attention",
                        detail: "MLA attention output index overflows usize".to_string(),
                    })?;
                let a = acc
                    .get(j)
                    .copied()
                    .ok_or_else(|| T0Error::BufferLengthMismatch {
                        tensor: "attention.acc",
                        buffer_len: acc.len(),
                        expected_len: idx.saturating_add(1),
                        shape: vec![acc.len()],
                    })?;
                checked_store(&mut scratch, idx, a / l, "attention.scratch")?;
            }
        }
    }
    for (i, &val) in scratch.iter().enumerate() {
        o.write_f32(i, val);
    }
    Ok(())
}

/// Dispatches `attention` to the cache matching the handle kind
/// (Spec 1 §2.6, §4.D).
pub fn attention(
    op: &AttentionOp,
    q: &TensorView<'_>,
    meta: &BatchMeta,
    group: u32,
    cache: &KvCache,
    o: &mut TensorViewMut<'_>,
) -> Result<(), T0Error> {
    use r9v_ir::StateKind;
    match (op.handle.kind(), op.mla.is_some(), cache) {
        (StateKind::KvPaged, false, KvCache::Paged(paged)) => {
            attention_paged(op, q, meta, group, paged, o)
        }
        (StateKind::KvLatent, true, KvCache::Latent(latent)) => {
            attention_mla(op, q, meta, group, latent, o)
        }
        _ => Err(T0Error::InvalidAttribute {
            op: "attention",
            attribute: "handle",
            reason: format!(
                "handle kind {:?} with mla={} does not match cache {}; paged attention needs KvPaged without MLA, MLA attention needs KvLatent with an MLA spec",
                op.handle.kind(),
                op.mla.is_some(),
                match cache {
                    KvCache::Paged(_) => "KvPaged",
                    KvCache::Latent(_) => "KvLatent",
                }
            ),
        }),
    }
}

// ----------------------------------------------------------------------------
// Dense independent f64 oracles (Spec 1 §6.1, §6.3)
// ----------------------------------------------------------------------------

// DECISION(A1.7): the f64 oracles below are dense and two-pass (explicit max,
// then normalized weights) while the T0 implementation is the online
// single-pass f32 form; the algorithmic difference is deliberate so the
// oracle cannot encode the same accumulation-order assumption as the code
// under test.

/// Dense two-pass f64 reference for one standard-attention query row
/// (Spec 1 §6.3).
///
/// Independent of [`attention_paged`]: full score vector, explicit max, then
/// normalized weights, all in f64. Total on all inputs: key/value row pairs
/// truncate to the shorter side (callers always pass equal lengths; a
/// mismatch is refused upstream, never a panic here).
pub fn attention_row_f64_reference(
    q_row: &[f64],
    k_rows: &[Vec<f64>],
    v_rows: &[Vec<f64>],
    scale: f64,
    softcap: Option<f64>,
) -> Vec<f64> {
    let n = k_rows.len().min(v_rows.len());
    if n == 0 {
        return vec![0.0; v_rows.first().map_or(0, Vec::len)];
    }
    let (k_rows, v_rows) = (&k_rows[..n], &v_rows[..n]);
    let dv = v_rows[0].len();
    let mut scores: Vec<f64> = k_rows
        .iter()
        .map(|k| {
            let mut s: f64 = q_row.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            s *= scale;
            if let Some(cap) = softcap {
                s = cap * (s / cap).tanh();
            }
            s
        })
        .collect();
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0f64;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    let mut out = vec![0.0f64; dv];
    for (w, v) in scores.iter().zip(v_rows.iter()) {
        for (o, &x) in out.iter_mut().zip(v.iter()) {
            *o += (*w / sum) * x;
        }
    }
    out
}

/// Dense two-pass f64 reference for one absorbed-MLA query row (Spec 1 §6.3).
///
/// Scores are `scale * (q_nope . c + q_rope . r)` with values `c`, in f64.
/// Total on all inputs: latent/rope row pairs truncate to the shorter side
/// (callers always pass equal lengths; a mismatch is refused upstream, never
/// a panic here).
pub fn mla_row_f64_reference(
    q_nope: &[f64],
    q_rope: &[f64],
    c_rows: &[Vec<f64>],
    r_rows: &[Vec<f64>],
    scale: f64,
    softcap: Option<f64>,
) -> Vec<f64> {
    let n = c_rows.len().min(r_rows.len());
    if n == 0 {
        return vec![0.0; c_rows.first().map_or(0, Vec::len)];
    }
    let (c_rows, r_rows) = (&c_rows[..n], &r_rows[..n]);
    let rank = c_rows[0].len();
    let mut scores: Vec<f64> = c_rows
        .iter()
        .zip(r_rows.iter())
        .map(|(c, r)| {
            let mut s: f64 = q_nope.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            s += q_rope.iter().zip(r.iter()).map(|(a, b)| a * b).sum::<f64>();
            s *= scale;
            if let Some(cap) = softcap {
                s = cap * (s / cap).tanh();
            }
            s
        })
        .collect();
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0f64;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    let mut out = vec![0.0f64; rank];
    for (w, c) in scores.iter().zip(c_rows.iter()) {
        for (o, &x) in out.iter_mut().zip(c.iter()) {
            *o += (*w / sum) * x;
        }
    }
    out
}

// DECISION(A1.7): query/output f16/bf16/f32 element conversion uses the
// established T0 view helpers (`read_f32`/`write_f32`, the same path every
// A1.5 op uses) while all cache value/scale codecs use the canonical
// `r9v-format` helpers (`f32_to_f16_bits`, `f16_to_f32`, `E4m3`); rejected
// bypassing the views because `r9v-format` ships no bf16 codec and the T0
// views are the codebase-canonical q/o path.
