// SPDX-License-Identifier: Apache-2.0
//! GGUF v3 container reader and writer (Spec 2 §6, §9; Spec 9 §3; card A2.5).
//!
//! [`GgufFile`] parses one GGUF shard's header, metadata KV table
//! (all thirteen [`KvType`] value types), and tensor-info table from
//! a byte buffer, with no I/O inside the parser. [`GgufWriter`]
//! emits byte-identical-layout files (verified against gguf-py
//! 0.19.0's writer). [`TensorType`] names every upstream
//! `GGMLQuantizationType` code of gguf-py 0.19.0 plus the R9V native
//! range 1000–1099 ([`R9vTensorType`]); the closed [`crate::GgmlType`]
//! set of cards A2.3/A2.4 is untouched. [`entry_regions`] derives the
//! exact §6 value/scale/index region offsets, [`GgufFile::file_fp`]/
//! [`model_fp`]
//! implement the Spec 9 §3 fingerprints, and
//! [`accept_format_version`] enforces the Spec 2 §9 version rule.
//! Typed `r9v.*` metadata lives in [`mod@crate::meta`].
//!
//! Reference: gguf-py 0.19.0 (`constants.py` type codes and
//! `GGML_QUANT_SIZES`, `gguf_reader.py` field order and duplicate
//! rules, `gguf_writer.py` offset and padding rules).

use std::collections::BTreeMap;

use crate::FormatError;

/// GGUF magic `"GGUF"` as a little-endian `u32` (gguf-py `GGUF_MAGIC`).
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// GGUF version this crate writes (gguf-py `GGUF_VERSION`).
pub const GGUF_VERSION: u32 = 3;
/// GGUF versions this reader parses (matches gguf-py
/// `READER_SUPPORTED_VERSIONS`).
// DECISION(A2.5): accept GGUF v2 as well as v3; rejected v3-only
// because the v2 header layout is identical and real files in the
// wild still carry v2. Spec 2 §6 names v3 only.
pub const GGUF_VERSIONS_ACCEPTED: [u32; 2] = [2, 3];
/// Alignment used when `general.alignment` is absent (gguf-py
/// `GGUF_DEFAULT_ALIGNMENT`).
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
/// Current `r9v.format_version` (Spec 2 §6: native files write 1).
pub const R9V_FORMAT_VERSION: u32 = 1;
/// Base of the R9V tensor-type range (Spec 2 §6: type IDs 1000–1099).
pub const R9V_TENSOR_TYPE_BASE: u32 = 1000;
/// R9V id of `scheme`: `1000 + scheme.code()` (Spec 2 §6).
// DECISION(A2.5): 1000 + SchemeId code (1001–1022 today); rejected
// hashing names or reusing GGUF ids because the mapping must be
// invertible without a table and the 1000–1099 range stays spare.
// Spec 2 §6 fixes only the range.
pub const fn r9v_tensor_type_id(scheme: crate::SchemeId) -> u32 {
    R9V_TENSOR_TYPE_BASE + scheme.code() as u32
}
/// Native-file tensor-region alignment (Spec 2 §6: 4 KiB).
pub const NATIVE_ALIGNMENT: u64 = 4096;
/// In-entry scale-region alignment (Spec 2 §6: 256 bytes).
pub const SCALE_ALIGN: u64 = 256;

/// GGUF metadata value type (gguf-py `GGUFValueType`; card A2.5).
///
/// Closed enum over the thirteen upstream codes; every `match` stays
/// exhaustive with no wildcard arm (CONVENTIONS.md §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KvType {
    /// `u8` (code 0).
    U8,
    /// `i8` (code 1).
    I8,
    /// `u16` (code 2).
    U16,
    /// `i16` (code 3).
    I16,
    /// `u32` (code 4).
    U32,
    /// `i32` (code 5).
    I32,
    /// `f32` (code 6).
    F32,
    /// `bool`, one byte `0x00`/`0x01` (code 7; gguf-py packs `?`).
    Bool,
    /// UTF-8 string with `u64` length prefix (code 8).
    Str,
    /// Homogeneous array: element-type `u32`, length `u64`, then
    /// elements (code 9; nesting arrays is malformed).
    Array,
    /// `u64` (code 10).
    U64,
    /// `i64` (code 11).
    I64,
    /// `f64` (code 12).
    F64,
}

impl KvType {
    /// All thirteen types in code order.
    pub const ALL: [KvType; 13] = [
        KvType::U8,
        KvType::I8,
        KvType::U16,
        KvType::I16,
        KvType::U32,
        KvType::I32,
        KvType::F32,
        KvType::Bool,
        KvType::Str,
        KvType::Array,
        KvType::U64,
        KvType::I64,
        KvType::F64,
    ];

    /// Returns the upstream `GGUFValueType` code.
    pub const fn code(self) -> u32 {
        match self {
            KvType::U8 => 0,
            KvType::I8 => 1,
            KvType::U16 => 2,
            KvType::I16 => 3,
            KvType::U32 => 4,
            KvType::I32 => 5,
            KvType::F32 => 6,
            KvType::Bool => 7,
            KvType::Str => 8,
            KvType::Array => 9,
            KvType::U64 => 10,
            KvType::I64 => 11,
            KvType::F64 => 12,
        }
    }

    /// Decodes an upstream code; anything else is malformed input,
    /// never a guess (Spec 2 §6; card A2.5).
    pub fn from_code(code: u32, offset: u64) -> Result<Self, FormatError> {
        match code {
            0 => Ok(KvType::U8),
            1 => Ok(KvType::I8),
            2 => Ok(KvType::U16),
            3 => Ok(KvType::I16),
            4 => Ok(KvType::U32),
            5 => Ok(KvType::I32),
            6 => Ok(KvType::F32),
            7 => Ok(KvType::Bool),
            8 => Ok(KvType::Str),
            9 => Ok(KvType::Array),
            10 => Ok(KvType::U64),
            11 => Ok(KvType::I64),
            12 => Ok(KvType::F64),
            _ => Err(FormatError::Malformed {
                offset,
                detail: format!("unknown GGUF metadata type code {code}"),
            }),
        }
    }

    /// Returns the stable uppercase name matching gguf-py.
    pub const fn name(self) -> &'static str {
        match self {
            KvType::U8 => "UINT8",
            KvType::I8 => "INT8",
            KvType::U16 => "UINT16",
            KvType::I16 => "INT16",
            KvType::U32 => "UINT32",
            KvType::I32 => "INT32",
            KvType::F32 => "FLOAT32",
            KvType::Bool => "BOOL",
            KvType::Str => "STRING",
            KvType::Array => "ARRAY",
            KvType::U64 => "UINT64",
            KvType::I64 => "INT64",
            KvType::F64 => "FLOAT64",
        }
    }
}

/// A decoded GGUF metadata value (Spec 2 §6; card A2.5).
///
/// Scalar variants hold native Rust values; [`KvValue::Array`]
/// holds the element type plus one-level scalar items (nested
/// arrays are rejected at parse as malformed, matching gguf-py,
/// which never writes them).
#[derive(Debug, Clone, PartialEq)]
pub enum KvValue {
    /// `UINT8` scalar.
    U8(u8),
    /// `INT8` scalar.
    I8(i8),
    /// `UINT16` scalar.
    U16(u16),
    /// `INT16` scalar.
    I16(i16),
    /// `UINT32` scalar.
    U32(u32),
    /// `INT32` scalar.
    I32(i32),
    /// `FLOAT32` scalar.
    F32(f32),
    /// `BOOL` scalar.
    Bool(bool),
    /// `STRING` scalar.
    Str(String),
    /// `ARRAY` with element type and scalar items.
    Array {
        /// Element type of every item.
        elem: KvType,
        /// One-level scalar items (never nested arrays).
        items: Vec<KvValue>,
    },
    /// `UINT64` scalar.
    U64(u64),
    /// `INT64` scalar.
    I64(i64),
    /// `FLOAT64` scalar.
    F64(f64),
}

impl KvValue {
    /// Returns the value's GGUF type.
    pub fn kv_type(&self) -> KvType {
        match self {
            KvValue::U8(_) => KvType::U8,
            KvValue::I8(_) => KvType::I8,
            KvValue::U16(_) => KvType::U16,
            KvValue::I16(_) => KvType::I16,
            KvValue::U32(_) => KvType::U32,
            KvValue::I32(_) => KvType::I32,
            KvValue::F32(_) => KvType::F32,
            KvValue::Bool(_) => KvType::Bool,
            KvValue::Str(_) => KvType::Str,
            KvValue::Array { .. } => KvType::Array,
            KvValue::U64(_) => KvType::U64,
            KvValue::I64(_) => KvType::I64,
            KvValue::F64(_) => KvType::F64,
        }
    }

    /// Byte length of this value as encoded (without the key or the
    /// outer type tag), for writer-side accounting (card A2.5).
    pub fn encoded_len(&self) -> u64 {
        fn str_len(s: &str) -> u64 {
            8 + s.len() as u64
        }
        match self {
            KvValue::U8(_) | KvValue::I8(_) | KvValue::Bool(_) => 1,
            KvValue::U16(_) | KvValue::I16(_) => 2,
            KvValue::U32(_) | KvValue::I32(_) | KvValue::F32(_) => 4,
            KvValue::U64(_) | KvValue::I64(_) | KvValue::F64(_) => 8,
            KvValue::Str(s) => str_len(s),
            KvValue::Array { items, .. } => {
                let mut len = 4 + 8;
                for item in items {
                    len += item.encoded_len();
                }
                len
            }
        }
    }
}

/// One metadata key/value pair in file order (Spec 2 §6; card A2.5).
#[derive(Debug, Clone, PartialEq)]
pub struct KvEntry {
    /// Metadata key (`general.*`, `tokenizer.*`, `r9v.*`, ...).
    pub key: String,
    /// Decoded value.
    pub value: KvValue,
}

/// An R9V native tensor type id, 1000–1099 (Spec 2 §6; card A2.5).
///
/// A 1:1 wrapper over the closed [`crate::SchemeId`] set: id `1000 +
/// scheme.code()`, so 1001–1022 today and the range stays spare for
/// future schemes. A parallel 22-arm enum was rejected: it would
/// duplicate the §3 closed set and could drift from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct R9vTensorType {
    scheme: crate::SchemeId,
}

impl R9vTensorType {
    /// Wraps the scheme whose repacked bytes this id names.
    pub const fn new(scheme: crate::SchemeId) -> Self {
        Self { scheme }
    }

    /// Returns the wrapped scheme.
    pub const fn scheme(self) -> crate::SchemeId {
        self.scheme
    }

    /// Returns the tensor-info `type` id (`1000 + scheme.code()`).
    pub const fn code(self) -> u32 {
        R9V_TENSOR_TYPE_BASE + self.scheme.code() as u32
    }

    /// Decodes an R9V type id; returns `None` for codes outside
    /// 1001–1022 (the caller maps those to standard or unknown
    /// types; Spec 2 §6, §7 step 1).
    pub fn from_code(code: u32) -> Option<Self> {
        if code <= R9V_TENSOR_TYPE_BASE {
            return None;
        }
        let scheme_code = (code - R9V_TENSOR_TYPE_BASE) as u64;
        crate::SchemeId::from_code(scheme_code).ok().map(Self::new)
    }
}

/// Tensor-info `type` code (Spec 2 §6, §7 step 1; card A2.5).
///
/// Covers every upstream `GGMLQuantizationType` code of gguf-py
/// 0.19.0 (numeric codes and `(block_len, block_bytes)` transcribed
/// from its `constants.py`), plus [`R9vTensorType`] for native ids.
/// Codes with no upstream meaning stay representable as
/// [`TensorType::Unknown`] so the table still parses; anything that
/// needs their bytes fails closed naming the code and tensor (Spec 2
/// §7 step 1). The closed [`crate::GgmlType`] repack set is
/// untouched: [`TensorType::ggml`] maps into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum TensorType {
    /// Unquantized `f32` (upstream id 0).
    F32,
    /// Unquantized `f16` (upstream id 1).
    F16,
    /// `Q4_0` (upstream id 2).
    Q4_0,
    /// `Q4_1` (upstream id 3).
    Q4_1,
    /// `Q5_0` (upstream id 6).
    Q5_0,
    /// `Q5_1` (upstream id 7).
    Q5_1,
    /// `Q8_0` (upstream id 8).
    Q8_0,
    /// `Q8_1` (upstream id 9).
    Q8_1,
    /// `Q2_K` (upstream id 10).
    Q2_K,
    /// `Q3_K` (upstream id 11).
    Q3_K,
    /// `Q4_K` (upstream id 12).
    Q4_K,
    /// `Q5_K` (upstream id 13).
    Q5_K,
    /// `Q6_K` (upstream id 14).
    Q6_K,
    /// `Q8_K` (upstream id 15).
    Q8_K,
    /// `IQ2_XXS` (upstream id 16).
    IQ2_XXS,
    /// `IQ2_XS` (upstream id 17).
    IQ2_XS,
    /// `IQ3_XXS` (upstream id 18).
    IQ3_XXS,
    /// `IQ1_S` (upstream id 19).
    IQ1_S,
    /// `IQ4_NL` (upstream id 20).
    IQ4_NL,
    /// `IQ3_S` (upstream id 21).
    IQ3_S,
    /// `IQ2_S` (upstream id 22).
    IQ2_S,
    /// `IQ4_XS` (upstream id 23).
    IQ4_XS,
    /// `I8` (upstream id 24).
    I8,
    /// `I16` (upstream id 25).
    I16,
    /// `I32` (upstream id 26).
    I32,
    /// `I64` (upstream id 27).
    I64,
    /// `F64` (upstream id 28).
    F64,
    /// `IQ1_M` (upstream id 29).
    IQ1_M,
    /// `BF16` (upstream id 30).
    BF16,
    /// `TQ1_0` (upstream id 34).
    TQ1_0,
    /// `TQ2_0` (upstream id 35).
    TQ2_0,
    /// `MXFP4` (upstream id 39).
    MXFP4,
    /// `NVFP4` (upstream id 40).
    NVFP4,
    /// `Q1_0` (upstream id 41).
    Q1_0,
    /// R9V native type (ids 1001–1022).
    R9v(R9vTensorType),
    /// A `type` code with no upstream meaning (Spec 2 §7 step 1:
    /// reading the table succeeds; sizing or mapping it is a hard
    /// error naming the code).
    Unknown(u32),
}

// DECISION(A2.5): TensorType carries the full upstream code table
// (all 35 gguf-py 0.19.0 ids with GGML_QUANT_SIZES block sizes) so a
// real llama.cpp file's tensor table always parses, including F32
// biases and integer tensors outside the A2.3/A2.4 repack set;
// rejected extending GgmlType because that closed set owns repack
// behavior and this table owns only wire sizes. Spec 2 §7 step 1.
impl TensorType {
    /// All 34 upstream-named types in code order (excludes `R9v`
    /// and `Unknown`, which are code-dependent).
    pub const ALL: [TensorType; 34] = [
        TensorType::F32,
        TensorType::F16,
        TensorType::Q4_0,
        TensorType::Q4_1,
        TensorType::Q5_0,
        TensorType::Q5_1,
        TensorType::Q8_0,
        TensorType::Q8_1,
        TensorType::Q2_K,
        TensorType::Q3_K,
        TensorType::Q4_K,
        TensorType::Q5_K,
        TensorType::Q6_K,
        TensorType::Q8_K,
        TensorType::IQ2_XXS,
        TensorType::IQ2_XS,
        TensorType::IQ3_XXS,
        TensorType::IQ1_S,
        TensorType::IQ4_NL,
        TensorType::IQ3_S,
        TensorType::IQ2_S,
        TensorType::IQ4_XS,
        TensorType::I8,
        TensorType::I16,
        TensorType::I32,
        TensorType::I64,
        TensorType::F64,
        TensorType::IQ1_M,
        TensorType::BF16,
        TensorType::TQ1_0,
        TensorType::TQ2_0,
        TensorType::MXFP4,
        TensorType::NVFP4,
        TensorType::Q1_0,
    ];

    /// Returns the tensor-info `type` code.
    pub const fn code(self) -> u32 {
        match self {
            TensorType::F32 => 0,
            TensorType::F16 => 1,
            TensorType::Q4_0 => 2,
            TensorType::Q4_1 => 3,
            TensorType::Q5_0 => 6,
            TensorType::Q5_1 => 7,
            TensorType::Q8_0 => 8,
            TensorType::Q8_1 => 9,
            TensorType::Q2_K => 10,
            TensorType::Q3_K => 11,
            TensorType::Q4_K => 12,
            TensorType::Q5_K => 13,
            TensorType::Q6_K => 14,
            TensorType::Q8_K => 15,
            TensorType::IQ2_XXS => 16,
            TensorType::IQ2_XS => 17,
            TensorType::IQ3_XXS => 18,
            TensorType::IQ1_S => 19,
            TensorType::IQ4_NL => 20,
            TensorType::IQ3_S => 21,
            TensorType::IQ2_S => 22,
            TensorType::IQ4_XS => 23,
            TensorType::I8 => 24,
            TensorType::I16 => 25,
            TensorType::I32 => 26,
            TensorType::I64 => 27,
            TensorType::F64 => 28,
            TensorType::IQ1_M => 29,
            TensorType::BF16 => 30,
            TensorType::TQ1_0 => 34,
            TensorType::TQ2_0 => 35,
            TensorType::MXFP4 => 39,
            TensorType::NVFP4 => 40,
            TensorType::Q1_0 => 41,
            TensorType::R9v(t) => t.code(),
            TensorType::Unknown(code) => code,
        }
    }

    /// Decodes a `type` code. Total function: R9V ids map to
    /// [`TensorType::R9v`], unlisted codes to [`TensorType::Unknown`]
    /// (Spec 2 §7 step 1: the table parses; sizing fails closed).
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => TensorType::F32,
            1 => TensorType::F16,
            2 => TensorType::Q4_0,
            3 => TensorType::Q4_1,
            6 => TensorType::Q5_0,
            7 => TensorType::Q5_1,
            8 => TensorType::Q8_0,
            9 => TensorType::Q8_1,
            10 => TensorType::Q2_K,
            11 => TensorType::Q3_K,
            12 => TensorType::Q4_K,
            13 => TensorType::Q5_K,
            14 => TensorType::Q6_K,
            15 => TensorType::Q8_K,
            16 => TensorType::IQ2_XXS,
            17 => TensorType::IQ2_XS,
            18 => TensorType::IQ3_XXS,
            19 => TensorType::IQ1_S,
            20 => TensorType::IQ4_NL,
            21 => TensorType::IQ3_S,
            22 => TensorType::IQ2_S,
            23 => TensorType::IQ4_XS,
            24 => TensorType::I8,
            25 => TensorType::I16,
            26 => TensorType::I32,
            27 => TensorType::I64,
            28 => TensorType::F64,
            29 => TensorType::IQ1_M,
            30 => TensorType::BF16,
            34 => TensorType::TQ1_0,
            35 => TensorType::TQ2_0,
            39 => TensorType::MXFP4,
            40 => TensorType::NVFP4,
            41 => TensorType::Q1_0,
            _ => match R9vTensorType::from_code(code) {
                Some(t) => TensorType::R9v(t),
                None => TensorType::Unknown(code),
            },
        }
    }

    /// Returns the GGUF spelling of the upstream name, `R9V_<scheme>`
    /// for native ids, or `UNKNOWN` for unlisted codes.
    pub fn name(self) -> String {
        match self {
            TensorType::F32 => "F32".to_owned(),
            TensorType::F16 => "F16".to_owned(),
            TensorType::Q4_0 => "Q4_0".to_owned(),
            TensorType::Q4_1 => "Q4_1".to_owned(),
            TensorType::Q5_0 => "Q5_0".to_owned(),
            TensorType::Q5_1 => "Q5_1".to_owned(),
            TensorType::Q8_0 => "Q8_0".to_owned(),
            TensorType::Q8_1 => "Q8_1".to_owned(),
            TensorType::Q2_K => "Q2_K".to_owned(),
            TensorType::Q3_K => "Q3_K".to_owned(),
            TensorType::Q4_K => "Q4_K".to_owned(),
            TensorType::Q5_K => "Q5_K".to_owned(),
            TensorType::Q6_K => "Q6_K".to_owned(),
            TensorType::Q8_K => "Q8_K".to_owned(),
            TensorType::IQ2_XXS => "IQ2_XXS".to_owned(),
            TensorType::IQ2_XS => "IQ2_XS".to_owned(),
            TensorType::IQ3_XXS => "IQ3_XXS".to_owned(),
            TensorType::IQ1_S => "IQ1_S".to_owned(),
            TensorType::IQ4_NL => "IQ4_NL".to_owned(),
            TensorType::IQ3_S => "IQ3_S".to_owned(),
            TensorType::IQ2_S => "IQ2_S".to_owned(),
            TensorType::IQ4_XS => "IQ4_XS".to_owned(),
            TensorType::I8 => "I8".to_owned(),
            TensorType::I16 => "I16".to_owned(),
            TensorType::I32 => "I32".to_owned(),
            TensorType::I64 => "I64".to_owned(),
            TensorType::F64 => "F64".to_owned(),
            TensorType::IQ1_M => "IQ1_M".to_owned(),
            TensorType::BF16 => "BF16".to_owned(),
            TensorType::TQ1_0 => "TQ1_0".to_owned(),
            TensorType::TQ2_0 => "TQ2_0".to_owned(),
            TensorType::MXFP4 => "MXFP4".to_owned(),
            TensorType::NVFP4 => "NVFP4".to_owned(),
            TensorType::Q1_0 => "Q1_0".to_owned(),
            TensorType::R9v(t) => format!("R9V_{}", t.scheme().name()),
            TensorType::Unknown(code) => format!("UNKNOWN_{code}"),
        }
    }

    /// Wire `(block_len, block_bytes)` from gguf-py 0.19.0
    /// `GGML_QUANT_SIZES` (R9V ids reuse their scheme's repacked
    /// geometry via [`mod@crate::repack`]; unlisted codes have none).
    pub fn quant_size(self) -> Option<(u32, u64)> {
        match self {
            TensorType::F32 => Some((1, 4)),
            TensorType::F16 => Some((1, 2)),
            TensorType::Q4_0 => Some((32, 18)),
            TensorType::Q4_1 => Some((32, 20)),
            TensorType::Q5_0 => Some((32, 22)),
            TensorType::Q5_1 => Some((32, 24)),
            TensorType::Q8_0 => Some((32, 34)),
            TensorType::Q8_1 => Some((32, 40)),
            TensorType::Q2_K => Some((256, 84)),
            TensorType::Q3_K => Some((256, 110)),
            TensorType::Q4_K => Some((256, 144)),
            TensorType::Q5_K => Some((256, 176)),
            TensorType::Q6_K => Some((256, 210)),
            TensorType::Q8_K => Some((256, 292)),
            TensorType::IQ2_XXS => Some((256, 66)),
            TensorType::IQ2_XS => Some((256, 74)),
            TensorType::IQ3_XXS => Some((256, 98)),
            TensorType::IQ1_S => Some((256, 50)),
            TensorType::IQ4_NL => Some((32, 18)),
            TensorType::IQ3_S => Some((256, 110)),
            TensorType::IQ2_S => Some((256, 82)),
            TensorType::IQ4_XS => Some((256, 136)),
            TensorType::I8 => Some((1, 1)),
            TensorType::I16 => Some((1, 2)),
            TensorType::I32 => Some((1, 4)),
            TensorType::I64 => Some((1, 8)),
            TensorType::F64 => Some((1, 8)),
            TensorType::IQ1_M => Some((256, 56)),
            TensorType::BF16 => Some((1, 2)),
            TensorType::TQ1_0 => Some((256, 54)),
            TensorType::TQ2_0 => Some((256, 66)),
            TensorType::MXFP4 => Some((32, 17)),
            TensorType::NVFP4 => Some((64, 36)),
            TensorType::Q1_0 => Some((128, 18)),
            TensorType::R9v(_) | TensorType::Unknown(_) => None,
        }
    }

    /// Maps into the closed [`crate::GgmlType`] repack set
    /// (`None` for types outside cards A2.3/A2.4: halves, integers,
    /// and exotic quants have no repack behavior).
    pub fn ggml(self) -> Option<crate::GgmlType> {
        match self {
            TensorType::F16 => Some(crate::GgmlType::F16),
            TensorType::Q4_0 => Some(crate::GgmlType::Q4_0),
            TensorType::Q4_1 => Some(crate::GgmlType::Q4_1),
            TensorType::Q5_0 => Some(crate::GgmlType::Q5_0),
            TensorType::Q5_1 => Some(crate::GgmlType::Q5_1),
            TensorType::Q8_0 => Some(crate::GgmlType::Q8_0),
            TensorType::Q2_K => Some(crate::GgmlType::Q2_K),
            TensorType::Q3_K => Some(crate::GgmlType::Q3_K),
            TensorType::Q4_K => Some(crate::GgmlType::Q4_K),
            TensorType::Q5_K => Some(crate::GgmlType::Q5_K),
            TensorType::Q6_K => Some(crate::GgmlType::Q6_K),
            TensorType::IQ2_XXS => Some(crate::GgmlType::IQ2_XXS),
            TensorType::IQ2_XS => Some(crate::GgmlType::IQ2_XS),
            TensorType::IQ3_XXS => Some(crate::GgmlType::IQ3_XXS),
            TensorType::IQ1_S => Some(crate::GgmlType::IQ1_S),
            TensorType::IQ4_NL => Some(crate::GgmlType::IQ4_NL),
            TensorType::IQ3_S => Some(crate::GgmlType::IQ3_S),
            TensorType::IQ2_S => Some(crate::GgmlType::IQ2_S),
            TensorType::IQ4_XS => Some(crate::GgmlType::IQ4_XS),
            TensorType::IQ1_M => Some(crate::GgmlType::IQ1_M),
            TensorType::BF16 => Some(crate::GgmlType::BF16),
            TensorType::F32
            | TensorType::Q8_1
            | TensorType::Q8_K
            | TensorType::I8
            | TensorType::I16
            | TensorType::I32
            | TensorType::I64
            | TensorType::F64
            | TensorType::TQ1_0
            | TensorType::TQ2_0
            | TensorType::MXFP4
            | TensorType::NVFP4
            | TensorType::Q1_0
            | TensorType::R9v(_)
            | TensorType::Unknown(_) => None,
        }
    }

    /// Maps to the repack [`crate::SchemeId`] where one exists:
    /// R9V ids unwrap directly, standard quant types go through
    /// [`crate::GgmlType::scheme`], unquantized dtypes and exotic
    /// types map to `None` (SI-26: halves are dtypes, not schemes).
    pub fn scheme(self) -> Option<crate::SchemeId> {
        match self {
            TensorType::R9v(t) => Some(t.scheme()),
            TensorType::Unknown(_) => None,
            _ => self.ggml().and_then(|g| g.scheme()),
        }
    }

    /// Data bytes for file-order `dims` (product over dims in blocks
    /// times block bytes; `None` for R9V ids — whose entry sizes come
    /// from [`entry_regions`] — and unlisted codes).
    pub fn data_nbytes(self, dims: &[u64]) -> Option<u64> {
        let (block_len, block_bytes) = self.quant_size()?;
        let mut elems: u64 = 1;
        for d in dims {
            elems = elems.checked_mul(*d)?;
        }
        if block_len as u64 == 0 || !elems.is_multiple_of(block_len as u64) {
            return None;
        }
        elems
            .checked_div(block_len as u64)?
            .checked_mul(block_bytes)
    }
}

/// One tensor-info row in table order (Spec 2 §6; card A2.5).
///
/// `dims` are in file order (innermost dimension first, as written
/// by gguf-py: `ti.shape` reversed). [`TensorInfo::shape`] returns
/// the logical outer-last order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Dimensions in file order (innermost first).
    pub dims: Vec<u64>,
    /// Tensor `type` code as a [`TensorType`].
    pub dtype: TensorType,
    /// Byte offset of this tensor's data relative to the data
    /// section start ([`GgufFile::data_start`]).
    pub offset: u64,
}

impl TensorInfo {
    /// Logical shape in outer-last order (reversed file order).
    pub fn shape(&self) -> Vec<u64> {
        let mut shape = self.dims.clone();
        shape.reverse();
        shape
    }

    /// Element count (product of dims; `None` on overflow).
    pub fn n_elems(&self) -> Option<u64> {
        let mut elems: u64 = 1;
        for d in &self.dims {
            elems = elems.checked_mul(*d)?;
        }
        Some(elems)
    }
}

/// Checked little-endian cursor over the file bytes (card A2.5).
///
/// Every read is bounds-checked against the buffer and reports
/// [`FormatError::Truncated`] with the offset, need, and field name;
/// structural violations report [`FormatError::Malformed`]. No
/// panics, no wrapping, no saturating arithmetic on input.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: u64,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> u64 {
        self.bytes.len() as u64 - self.pos
    }

    fn take(&mut self, len: u64, what: &'static str) -> Result<&'a [u8], FormatError> {
        if len > self.remaining() {
            return Err(FormatError::Truncated {
                offset: self.pos,
                need: len,
                what,
            });
        }
        let start = self.pos as usize;
        let end = start + len as usize;
        self.pos += len;
        Ok(&self.bytes[start..end])
    }

    fn read_u8(&mut self, what: &'static str) -> Result<u8, FormatError> {
        let b = self.take(1, what)?;
        Ok(b[0])
    }

    fn read_u16(&mut self, what: &'static str) -> Result<u16, FormatError> {
        let b = self.take(2, what)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self, what: &'static str) -> Result<u32, FormatError> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self, what: &'static str) -> Result<u64, FormatError> {
        let b = self.take(8, what)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i8(&mut self, what: &'static str) -> Result<i8, FormatError> {
        Ok(self.read_u8(what)? as i8)
    }

    fn read_i16(&mut self, what: &'static str) -> Result<i16, FormatError> {
        Ok(self.read_u16(what)? as i16)
    }

    fn read_i32(&mut self, what: &'static str) -> Result<i32, FormatError> {
        Ok(self.read_u32(what)? as i32)
    }

    fn read_i64(&mut self, what: &'static str) -> Result<i64, FormatError> {
        Ok(self.read_u64(what)? as i64)
    }

    fn read_f32(&mut self, what: &'static str) -> Result<f32, FormatError> {
        Ok(f32::from_bits(self.read_u32(what)?))
    }

    fn read_f64(&mut self, what: &'static str) -> Result<f64, FormatError> {
        Ok(f64::from_bits(self.read_u64(what)?))
    }

    // DECISION(A2.5): BOOL bytes outside 0/1 are malformed;
    // rejected gguf-py's nonzero-is-True view because a stray byte
    // is corruption, not a boolean. Spec 2 §6 is silent.
    fn read_bool(&mut self, what: &'static str) -> Result<bool, FormatError> {
        let offset = self.pos;
        match self.read_u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(FormatError::Malformed {
                offset,
                detail: format!("BOOL value {v} is not 0 or 1"),
            }),
        }
    }

    fn read_str(&mut self, what: &'static str) -> Result<String, FormatError> {
        let offset = self.pos;
        let len = self.read_u64("string length")?;
        let raw = self.take(len, what)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|e| FormatError::Malformed {
                offset,
                detail: format!("invalid UTF-8 in {what}: {e}"),
            })
    }

    fn read_scalar(&mut self, ty: KvType) -> Result<KvValue, FormatError> {
        match ty {
            KvType::U8 => Ok(KvValue::U8(self.read_u8("u8 value")?)),
            KvType::I8 => Ok(KvValue::I8(self.read_i8("i8 value")?)),
            KvType::U16 => Ok(KvValue::U16(self.read_u16("u16 value")?)),
            KvType::I16 => Ok(KvValue::I16(self.read_i16("i16 value")?)),
            KvType::U32 => Ok(KvValue::U32(self.read_u32("u32 value")?)),
            KvType::I32 => Ok(KvValue::I32(self.read_i32("i32 value")?)),
            KvType::F32 => Ok(KvValue::F32(self.read_f32("f32 value")?)),
            KvType::Bool => Ok(KvValue::Bool(self.read_bool("bool value")?)),
            KvType::Str => Ok(KvValue::Str(self.read_str("string value")?)),
            KvType::U64 => Ok(KvValue::U64(self.read_u64("u64 value")?)),
            KvType::I64 => Ok(KvValue::I64(self.read_i64("i64 value")?)),
            KvType::F64 => Ok(KvValue::F64(self.read_f64("f64 value")?)),
            KvType::Array => {
                let offset = self.pos;
                Err(FormatError::Malformed {
                    offset,
                    detail: "nested arrays are not valid GGUF metadata".to_owned(),
                })
            }
        }
    }

    fn read_value(&mut self, ty: KvType) -> Result<KvValue, FormatError> {
        match ty {
            KvType::Array => {
                let elem_off = self.pos;
                let elem = KvType::from_code(self.read_u32("array element type")?, elem_off)?;
                let len = self.read_u64("array length")?;
                let mut items = Vec::new();
                let mut i: u64 = 0;
                while i < len {
                    // Internal invariant: the loop appends exactly one
                    // item per iteration, so `i` counts items read.
                    items.push(self.read_scalar(elem)?);
                    i += 1;
                }
                Ok(KvValue::Array { elem, items })
            }
            KvType::U8
            | KvType::I8
            | KvType::U16
            | KvType::I16
            | KvType::U32
            | KvType::I32
            | KvType::F32
            | KvType::Bool
            | KvType::Str
            | KvType::U64
            | KvType::I64
            | KvType::F64 => self.read_scalar(ty),
        }
    }
}

/// One parsed GGUF shard (Spec 2 §6, Spec 9 §3 step 1; card A2.5).
///
/// Owns the decoded header, metadata table, and tensor-info table.
/// Tensor *data* is not copied: [`GgufFile::tensor_bytes`] slices
/// the caller's buffer with checked ranges. Fingerprint inputs are
/// the exact raw byte ranges recorded at parse
/// ([`GgufFile::file_fp`]).
#[derive(Debug, Clone)]
pub struct GgufFile {
    version: u32,
    alignment: u64,
    kvs: Vec<KvEntry>,
    kv_index: BTreeMap<String, usize>,
    tensors: Vec<TensorInfo>,
    tensor_index: BTreeMap<String, usize>,
    data_start: u64,
    file_size: u64,
    header_range: (u64, u64),
    kv_range: (u64, u64),
    ti_range: (u64, u64),
}

/// Rounds `value` up to `align` (a nonzero power of two) in checked
/// arithmetic; `None` on overflow (card A2.5).
fn align_up(value: u64, align: u64) -> Option<u64> {
    value.checked_add(align - 1).map(|v| v / align * align)
}

impl GgufFile {
    /// Parses one GGUF shard from `bytes` (Spec 2 §6; card A2.5).
    ///
    /// Structural decode errors (magic, version, truncation,
    /// malformed fields) return immediately with their offset;
    /// table-level validation (duplicates, alignment, tensor ranges,
    /// unknown types) collects every problem before returning
    /// (CONVENTIONS.md §1.4).
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cursor = Cursor::new(bytes);
        let magic = cursor.read_u32("magic")?;
        if magic != GGUF_MAGIC {
            return Err(FormatError::BadMagic { found: magic });
        }
        let version = cursor.read_u32("version")?;
        if !GGUF_VERSIONS_ACCEPTED.contains(&version) {
            return Err(FormatError::UnsupportedVersion {
                found: version,
                accepted: GGUF_VERSIONS_ACCEPTED.to_vec(),
            });
        }
        let n_tensors = cursor.read_u64("tensor count")?;
        let n_kv = cursor.read_u64("metadata count")?;
        let header_range = (0, cursor.pos);

        let kv_start = cursor.pos;
        let mut kvs = Vec::new();
        let mut k: u64 = 0;
        while k < n_kv {
            let key = cursor.read_str("metadata key")?;
            let type_off = cursor.pos;
            let ty = KvType::from_code(cursor.read_u32("metadata type")?, type_off)?;
            let value = cursor.read_value(ty)?;
            kvs.push(KvEntry { key, value });
            k += 1;
        }
        let kv_range = (kv_start, cursor.pos);

        let ti_start = cursor.pos;
        let mut tensors = Vec::new();
        let mut t: u64 = 0;
        while t < n_tensors {
            let name = cursor.read_str("tensor name")?;
            let n_dims = cursor.read_u32("tensor dimension count")?;
            let mut dims = Vec::new();
            let mut d: u32 = 0;
            while d < n_dims {
                dims.push(cursor.read_u64("tensor dimension")?);
                d += 1;
            }
            let dtype = TensorType::from_code(cursor.read_u32("tensor type")?);
            let offset = cursor.read_u64("tensor offset")?;
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
            t += 1;
        }
        let ti_range = (ti_start, cursor.pos);

        let file_size = bytes.len() as u64;
        let mut file = Self {
            version,
            alignment: GGUF_DEFAULT_ALIGNMENT,
            kvs,
            kv_index: BTreeMap::new(),
            tensors,
            tensor_index: BTreeMap::new(),
            data_start: cursor.pos,
            file_size,
            header_range,
            kv_range,
            ti_range,
        };
        file.validate()?;
        Ok(file)
    }

    /// Reads and parses the file at `path` (card A2.5). I/O failure
    /// is a typed [`FormatError::Io`], never a panic.
    pub fn parse_file(path: &std::path::Path) -> Result<Self, FormatError> {
        let bytes = std::fs::read(path).map_err(|e| FormatError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        Self::parse(&bytes)
    }

    /// Table-level validation: duplicate keys/tensors, alignment,
    /// and every tensor's data range (CONVENTIONS.md §1.4: all
    /// problems are collected before returning).
    fn validate(&mut self) -> Result<(), FormatError> {
        let mut problems = Vec::new();

        for (index, kv) in self.kvs.iter().enumerate() {
            if self.kv_index.contains_key(&kv.key) {
                problems.push(FormatError::DuplicateKey {
                    key: kv.key.clone(),
                });
            } else {
                self.kv_index.insert(kv.key.clone(), index);
            }
        }
        for (index, tensor) in self.tensors.iter().enumerate() {
            if self.tensor_index.contains_key(&tensor.name) {
                problems.push(FormatError::DuplicateTensor {
                    name: tensor.name.clone(),
                });
            } else {
                self.tensor_index.insert(tensor.name.clone(), index);
            }
        }

        // DECISION(A2.5): general.alignment must be a nonzero power
        // of two, matching the gguf-py reader; rejected accepting
        // any multiple because align_up assumes power-of-two. Spec 2
        // §6 fixes 4096 for native files only.
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        if let Some(index) = self.kv_index.get("general.alignment") {
            // Internal invariant: `kv_index` values are positions in
            // `kvs` inserted above, so indexing stays in bounds.
            let value = &self.kvs[*index].value;
            match value {
                KvValue::U32(v) => {
                    let v = *v as u64;
                    if v == 0 || !v.is_power_of_two() {
                        problems.push(FormatError::InvalidAlignment { value: v });
                    } else {
                        alignment = v;
                    }
                }
                _ => problems.push(FormatError::KvTypeMismatch {
                    key: "general.alignment".to_owned(),
                    found: value.kv_type().name(),
                    expected: KvType::U32.name(),
                }),
            }
        }
        self.alignment = alignment;

        match align_up(self.ti_range.1, alignment) {
            Some(start) => self.data_start = start,
            None => problems.push(FormatError::Overflow {
                what: "data_start",
                detail: format!("ti_end={} align={alignment}", self.ti_range.1),
            }),
        }

        // Range checks run against the aligned start even when the
        // alignment itself failed, so one bad file reports everything.
        let mut spans: Vec<(u64, u64, usize)> = Vec::new();
        for (index, tensor) in self.tensors.iter().enumerate() {
            let nbytes = match self.tensor_nbytes(tensor) {
                Ok(n) => n,
                Err(e) => {
                    problems.push(e);
                    continue;
                }
            };
            let start = match self.data_start.checked_add(tensor.offset) {
                Some(s) => s,
                None => {
                    problems.push(FormatError::BadTensorRange {
                        name: tensor.name.clone(),
                        start: u64::MAX,
                        end: u64::MAX,
                        reason: format!(
                            "data_start {} + offset {} overflows",
                            self.data_start, tensor.offset
                        ),
                    });
                    continue;
                }
            };
            let end = match start.checked_add(nbytes) {
                Some(e) => e,
                None => {
                    problems.push(FormatError::BadTensorRange {
                        name: tensor.name.clone(),
                        start,
                        end: u64::MAX,
                        reason: format!("range end {start} + {nbytes} bytes overflows"),
                    });
                    continue;
                }
            };
            if end > self.file_size {
                problems.push(FormatError::BadTensorRange {
                    name: tensor.name.clone(),
                    start,
                    end,
                    reason: format!("range end {end} is beyond file size {}", self.file_size),
                });
                continue;
            }
            spans.push((start, end, index));
        }
        spans.sort();
        let mut i = 1;
        while i < spans.len() {
            // Internal invariant: `spans` is sorted, so every pair of
            // neighbors is the only possible overlap at this step.
            let (prev_start, prev_end, prev_index) = spans[i - 1];
            let (start, end, index) = spans[i];
            if start < prev_end {
                problems.push(FormatError::BadTensorRange {
                    name: self.tensors[index].name.clone(),
                    start,
                    end,
                    reason: format!(
                        "overlaps {:?} [{prev_start}, {prev_end})",
                        self.tensors[prev_index].name
                    ),
                });
            }
            i += 1;
        }

        FormatError::collect(problems)
    }

    /// Returns the GGUF version found in the file.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the effective alignment (`general.alignment` or 32).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Returns the aligned data-section start (Spec 2 §6).
    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// Returns the parsed file size in bytes.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Returns the metadata entries in file order.
    pub fn kvs(&self) -> &[KvEntry] {
        &self.kvs
    }

    /// Returns the tensor-info rows in table order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Looks up one metadata value by key (`None` when absent).
    pub fn kv(&self, key: &str) -> Option<&KvValue> {
        self.kv_index.get(key).map(|i| &self.kvs[*i].value)
    }

    /// Looks up one tensor-info row by name (`None` when absent).
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|i| &self.tensors[*i])
    }

    /// Returns the raw header byte range (for [`GgufFile::file_fp`]).
    pub fn header_range(&self) -> (u64, u64) {
        self.header_range
    }

    /// Returns the raw metadata-KV byte range.
    pub fn kv_range(&self) -> (u64, u64) {
        self.kv_range
    }

    /// Returns the raw tensor-info byte range.
    pub fn ti_range(&self) -> (u64, u64) {
        self.ti_range
    }

    // DECISION(A2.5): native R9V tensor entry sizes in the container are
    // derived via entry_regions from the tensor's scheme, layout, and dims;
    // rejected leaving R9V tensors unsized because per-entry xxh3 validation
    // (Spec 2 §10) and container bounds/overlap checks require entry slicing.
    // Spec 2 §6, §10; card A2.5.
    /// Returns the byte length of one tensor entry (Spec 2 §6, §10; card A2.5).
    ///
    /// For standard wire types, size is derived from wire block size
    /// and dimensions. For native R9V types, size is derived from
    /// [`entry_regions`] using the tensor's scheme, layout, and shape.
    pub fn tensor_nbytes(&self, info: &TensorInfo) -> Result<u64, FormatError> {
        match info.dtype {
            TensorType::Unknown(code) => Err(FormatError::UnknownTensorType {
                code,
                tensor: info.name.clone(),
            }),
            TensorType::R9v(r9v_type) => {
                if info.dims.is_empty() {
                    return Err(FormatError::Malformed {
                        offset: 0,
                        detail: format!("tensor {:?} has zero dimensions", info.name),
                    });
                }
                let k = u32::try_from(info.dims[0]).map_err(|_| FormatError::Overflow {
                    what: "tensor k dim",
                    detail: format!("tensor {:?} dim[0]={}", info.name, info.dims[0]),
                })?;
                let n = if info.dims.len() == 1 {
                    1u32
                } else {
                    let mut prod: u64 = 1;
                    for d in &info.dims[1..] {
                        prod = prod.checked_mul(*d).ok_or_else(|| FormatError::Overflow {
                            what: "tensor n dims product",
                            detail: format!("tensor {:?}", info.name),
                        })?;
                    }
                    u32::try_from(prod).map_err(|_| FormatError::Overflow {
                        what: "tensor n dim",
                        detail: format!("tensor {:?} product={prod}", info.name),
                    })?
                };

                let sparse_key = format!("r9v.tensor.{}.sparse", info.name);
                let is_sparse = match self.kv(&sparse_key) {
                    Some(KvValue::Str(s)) => match s.as_str() {
                        "none" => false,
                        "s24" => true,
                        _ => {
                            return Err(FormatError::Malformed {
                                offset: 0,
                                detail: format!(
                                    "tensor {:?} sparse value {s:?}: expected none or s24 (Spec 2 §4)",
                                    info.name
                                ),
                            });
                        }
                    },
                    Some(other) => {
                        return Err(FormatError::KvTypeMismatch {
                            key: sparse_key,
                            found: other.kv_type().name(),
                            expected: "STRING",
                        });
                    }
                    None => false,
                };

                let roles_key = format!("r9v.tensor.{}.roles", info.name);
                let parsed_roles = match self.kv(&roles_key) {
                    Some(KvValue::Array { elem, items }) => {
                        if *elem != KvType::Str {
                            return Err(FormatError::KvTypeMismatch {
                                key: roles_key,
                                found: elem.name(),
                                expected: "STRING",
                            });
                        }
                        if items.is_empty() {
                            return Err(FormatError::Malformed {
                                offset: 0,
                                detail: format!(
                                    "tensor {:?} roles array is empty (Spec 2 §4)",
                                    info.name
                                ),
                            });
                        }
                        let mut parsed = Vec::with_capacity(items.len());
                        for item in items {
                            match item {
                                KvValue::Str(s) => match crate::meta::Role::parse(s) {
                                    Ok(role) => parsed.push(role),
                                    Err(_) => {
                                        return Err(FormatError::Malformed {
                                            offset: 0,
                                            detail: format!(
                                                "tensor {:?} has unknown role {s:?} (Spec 2 §4)",
                                                info.name
                                            ),
                                        });
                                    }
                                },
                                other => {
                                    return Err(FormatError::KvTypeMismatch {
                                        key: roles_key,
                                        found: other.kv_type().name(),
                                        expected: "STRING",
                                    });
                                }
                            }
                        }
                        // Spec 2 §4 closed set: [matmul] | [embed] | [lm_head] | [embed, lm_head] | [ngram_table] | [vector]
                        let is_valid_combo = matches!(
                            parsed.as_slice(),
                            [crate::meta::Role::Matmul]
                                | [crate::meta::Role::Embed]
                                | [crate::meta::Role::LmHead]
                                | [crate::meta::Role::NgramTable]
                                | [crate::meta::Role::Vector]
                                | [crate::meta::Role::Embed, crate::meta::Role::LmHead]
                                | [crate::meta::Role::LmHead, crate::meta::Role::Embed]
                        );
                        if !is_valid_combo {
                            return Err(FormatError::Malformed {
                                offset: 0,
                                detail: format!(
                                    "tensor {:?} has invalid roles combination {parsed:?} (Spec 2 §4)",
                                    info.name
                                ),
                            });
                        }
                        Some(parsed)
                    }
                    Some(other) => {
                        return Err(FormatError::KvTypeMismatch {
                            key: roles_key,
                            found: other.kv_type().name(),
                            expected: "ARRAY",
                        });
                    }
                    None => None,
                };

                // Closed-set validation on 1D vectors and sparsity
                if info.dims.len() == 1 {
                    if is_sparse {
                        return Err(FormatError::Malformed {
                            offset: 0,
                            detail: format!(
                                "1D vector {:?} cannot be sparse (Spec 2 §4, §5)",
                                info.name
                            ),
                        });
                    }
                    if let Some(roles) = &parsed_roles {
                        if roles != &[crate::meta::Role::Vector] {
                            return Err(FormatError::Malformed {
                                offset: 0,
                                detail: format!(
                                    "1D tensor {:?} has non-vector role {roles:?} (Spec 2 §4, §5)",
                                    info.name
                                ),
                            });
                        }
                    }
                }

                if is_sparse {
                    if let Some(roles) = &parsed_roles {
                        if roles != &[crate::meta::Role::Matmul] {
                            return Err(FormatError::Malformed {
                                offset: 0,
                                detail: format!(
                                    "sparse tensor {:?} has non-matmul role {roles:?} (Spec 2 §4, §5)",
                                    info.name
                                ),
                            });
                        }
                    }
                    if !matches!(
                        r9v_type.scheme(),
                        crate::SchemeId::I8R
                            | crate::SchemeId::I8B128
                            | crate::SchemeId::I4K
                            | crate::SchemeId::E4M3B128
                    ) {
                        return Err(FormatError::UnsupportedLayout {
                            scheme: r9v_type.scheme().name(),
                            layout: crate::Layout::L1S.name(),
                        });
                    }
                }

                // Layout determination:
                // 1. Sparse is L1S
                // 2. 1D tensors are vectors -> L0 (Spec 2 §5, §7)
                // 3. Roles:
                //    - Tied [embed, lm_head] is stored once, in L1 (Spec 2 §4)
                //    - Standalone embed / ngram_table / vector -> L0 (Spec 2 §5, §7)
                //    - matmul / standalone lm_head -> L1
                // 4. Missing roles:
                //    - file-level r9v.layout_id (L0 / L1 / L1S)
                //    - default L1
                let layout = if is_sparse {
                    crate::Layout::L1S
                } else if info.dims.len() == 1 {
                    crate::Layout::L0
                } else if let Some(roles) = &parsed_roles {
                    let has_embed = roles.contains(&crate::meta::Role::Embed);
                    let has_lm_head = roles.contains(&crate::meta::Role::LmHead);
                    if has_embed && has_lm_head {
                        crate::Layout::L1
                    } else if has_embed
                        || roles.contains(&crate::meta::Role::NgramTable)
                        || roles.contains(&crate::meta::Role::Vector)
                    {
                        crate::Layout::L0
                    } else {
                        crate::Layout::L1
                    }
                } else if let Some(val) = self.kv("r9v.layout_id") {
                    match val {
                        KvValue::Str(s) => match s.to_ascii_lowercase().as_str() {
                            "l0" => crate::Layout::L0,
                            "l1s" => crate::Layout::L1S,
                            "l1" => crate::Layout::L1,
                            _ => return Err(FormatError::UnknownLayout { value: s.clone() }),
                        },
                        other => {
                            return Err(FormatError::KvTypeMismatch {
                                key: "r9v.layout_id".to_owned(),
                                found: other.kv_type().name(),
                                expected: "STRING",
                            });
                        }
                    }
                } else {
                    crate::Layout::L1
                };

                let regions = entry_regions(r9v_type.scheme(), layout, n, k)?;

                // If explicit r9v.tensor.<name>.regions metadata is present,
                // strictly validate typing, offset consistency, and agreement
                // with derived entry geometry (Spec 2 §6; card A2.5).
                let regions_key = format!("r9v.tensor.{}.regions", info.name);
                if let Some(val) = self.kv(&regions_key) {
                    match val {
                        KvValue::Array { elem, items } => {
                            if *elem != KvType::U64 {
                                return Err(FormatError::KvTypeMismatch {
                                    key: regions_key,
                                    found: elem.name(),
                                    expected: "UINT64",
                                });
                            }
                            if items.len() != 3 {
                                return Err(FormatError::Malformed {
                                    offset: 0,
                                    detail: format!(
                                        "tensor {:?} regions has {} item(s), expected 3 (Spec 2 §6)",
                                        info.name,
                                        items.len()
                                    ),
                                });
                            }
                            let mut explicit = [0u64; 3];
                            for (i, item) in items.iter().enumerate() {
                                match item {
                                    KvValue::U64(v) => explicit[i] = *v,
                                    other => {
                                        return Err(FormatError::KvTypeMismatch {
                                            key: regions_key,
                                            found: other.kv_type().name(),
                                            expected: "UINT64",
                                        });
                                    }
                                }
                            }
                            if explicit[0] != 0
                                || explicit[0] > explicit[1]
                                || explicit[1] > explicit[2]
                                || explicit[1] % 256 != 0
                            {
                                return Err(FormatError::Malformed {
                                    offset: 0,
                                    detail: format!(
                                        "tensor {:?} has invalid explicit region offsets {explicit:?} (Spec 2 §6)",
                                        info.name
                                    ),
                                });
                            }
                            let expected = regions.offsets();
                            if explicit != expected {
                                return Err(FormatError::Malformed {
                                    offset: 0,
                                    detail: format!(
                                        "tensor {:?} explicit regions {explicit:?} do not match derived regions {expected:?} (Spec 2 §6)",
                                        info.name
                                    ),
                                });
                            }
                        }
                        other => {
                            return Err(FormatError::KvTypeMismatch {
                                key: regions_key,
                                found: other.kv_type().name(),
                                expected: "ARRAY",
                            });
                        }
                    }
                }

                Ok(regions.entry_bytes)
            }
            _ => {
                if let Some(nbytes) = info.dtype.data_nbytes(&info.dims) {
                    Ok(nbytes)
                } else if info.dtype.quant_size().is_none() {
                    Err(FormatError::Malformed {
                        offset: 0,
                        detail: format!(
                            "tensor {:?} ({}): no wire block size is known",
                            info.name,
                            info.dtype.name(),
                        ),
                    })
                } else {
                    let elems = info.n_elems().unwrap_or(u64::MAX);
                    Err(FormatError::Malformed {
                        offset: 0,
                        detail: format!(
                            "tensor {:?} ({}): {} element(s) are not a whole number of blocks",
                            info.name,
                            info.dtype.name(),
                            elems,
                        ),
                    })
                }
            }
        }
    }

    /// Slices one tensor's data from `bytes` (the same buffer that
    /// was parsed) with checked bounds (Spec 2 §6; card A2.5).
    pub fn tensor_bytes<'a>(&self, name: &str, bytes: &'a [u8]) -> Result<&'a [u8], FormatError> {
        let info = self.tensor(name).ok_or_else(|| FormatError::Malformed {
            offset: 0,
            detail: format!("no tensor named {name:?} in this shard"),
        })?;
        let nbytes = self.tensor_nbytes(info)?;
        let start = self.data_start.checked_add(info.offset).ok_or_else(|| {
            FormatError::BadTensorRange {
                name: info.name.clone(),
                start: u64::MAX,
                end: u64::MAX,
                reason: "data_start + offset overflows".to_owned(),
            }
        })?;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| FormatError::BadTensorRange {
                name: info.name.clone(),
                start,
                end: u64::MAX,
                reason: "range end overflows".to_owned(),
            })?;
        if end > bytes.len() as u64 {
            return Err(FormatError::BadTensorRange {
                name: info.name.clone(),
                start,
                end,
                reason: format!("range end {end} is beyond buffer size {}", bytes.len()),
            });
        }
        Ok(&bytes[start as usize..end as usize])
    }

    /// `true` when every tensor has a standard upstream id and no
    /// `r9v.*` key is present: a standard GGUF that loads through
    /// repack (Spec 2 §6; card A2.5).
    pub fn is_standard_gguf(&self) -> bool {
        let no_r9v = !self.kvs.iter().any(|kv| kv.key.starts_with("r9v."));
        let all_standard = self
            .tensors
            .iter()
            .all(|t| !matches!(t.dtype, TensorType::R9v(_) | TensorType::Unknown(_)));
        no_r9v && all_standard
    }

    /// `xxh3` of one tensor entry's bytes (Spec 2 §7 step 6, §10:
    /// per-entry checksum; card A2.5).
    pub fn entry_xxh3(&self, name: &str, bytes: &[u8]) -> Result<u64, FormatError> {
        Ok(r9v_common::xxh3_64(self.tensor_bytes(name, bytes)?))
    }

    /// `file_fp` over this shard (Spec 9 §3; card A2.5):
    /// `xxh3(header ‖ tensor-info table ‖ metadata KV bytes ‖ file
    /// size ‖ shard count)`. The hashed slices are the exact raw
    /// ranges recorded at parse, so the fingerprint is stable across
    /// runs and machines for identical files.
    pub fn file_fp(&self, bytes: &[u8], shard_count: u64) -> u128 {
        let slice = |range: (u64, u64)| -> &[u8] {
            let (start, end) = range;
            let len = bytes.len() as u64;
            let start = start.min(len) as usize;
            let end = end.min(len).max(start as u64) as usize;
            &bytes[start..end]
        };
        // Internal invariant: the hasher accepts any byte string, so
        // these writes cannot fail; lengths are fixed below.
        let mut input = Vec::new();
        input.extend_from_slice(slice(self.header_range));
        input.extend_from_slice(slice(self.ti_range));
        input.extend_from_slice(slice(self.kv_range));
        input.extend_from_slice(&self.file_size.to_le_bytes());
        input.extend_from_slice(&shard_count.to_le_bytes());
        r9v_common::xxh3_128(&input)
    }
}

/// Encodes one metadata value in gguf-py `_pack_val` order (card
/// A2.5): scalars little-endian, `BOOL` as one byte, strings as
/// `u64` length plus UTF-8, arrays as `u32` element type plus `u64`
/// length plus items.
fn encode_value(out: &mut Vec<u8>, value: &KvValue) {
    match value {
        KvValue::U8(v) => out.push(*v),
        KvValue::I8(v) => out.push(*v as u8),
        KvValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        KvValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        KvValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        KvValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        KvValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        KvValue::Bool(v) => out.push(u8::from(*v)),
        KvValue::Str(s) => {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        KvValue::Array { elem, items } => {
            out.extend_from_slice(&elem.code().to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                encode_value(out, item);
            }
        }
        KvValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        KvValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        KvValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
    }
}

fn encode_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// One writer-side tensor: logical shape, type, and raw data bytes
/// (Spec 2 §6; card A2.5).
#[derive(Debug, Clone)]
pub struct OutTensor {
    /// Tensor name.
    pub name: String,
    /// Logical shape in outer-last order (written reversed, matching
    /// gguf-py's `ti.shape[n_dims - 1 - j]`).
    pub shape: Vec<u64>,
    /// Tensor `type` code.
    pub dtype: TensorType,
    /// Raw tensor data bytes (`dtype.data_nbytes(file_dims)` long
    /// for wire types, `entry_regions` long for R9V ids).
    pub data: Vec<u8>,
}

/// GGUF writer (Spec 2 §6; card A2.5).
///
/// Emits header, metadata KVs in insertion order, the tensor-info
/// table with gguf-py-identical offsets (each tensor's offset
/// advances by its alignment-padded size), alignment padding, then
/// tensor data with inter-tensor padding. Native files set
/// alignment 4096 via [`GgufWriter::with_alignment`].
#[derive(Debug, Clone, Default)]
pub struct GgufWriter {
    kvs: Vec<KvEntry>,
    tensors: Vec<OutTensor>,
    alignment: u64,
}

impl GgufWriter {
    /// Creates a writer with the standard default alignment (32).
    pub fn new() -> Self {
        Self {
            kvs: Vec::new(),
            tensors: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
        }
    }

    /// Sets the data alignment (native files use 4096 per Spec 2
    /// §6; must be a nonzero power of two).
    pub fn with_alignment(mut self, alignment: u64) -> Result<Self, FormatError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(FormatError::InvalidAlignment { value: alignment });
        }
        self.alignment = alignment;
        Ok(self)
    }

    /// Appends one metadata KV (duplicate keys are rejected so
    /// output order is exactly insertion order; card A2.5).
    pub fn add_kv(&mut self, key: &str, value: KvValue) -> Result<(), FormatError> {
        if self.kvs.iter().any(|kv| kv.key == key) {
            return Err(FormatError::DuplicateKey {
                key: key.to_owned(),
            });
        }
        if let KvValue::Array { items, .. } = &value {
            for item in items {
                if matches!(item, KvValue::Array { .. }) {
                    return Err(FormatError::Malformed {
                        offset: 0,
                        detail: format!(
                            "metadata key {key:?}: nested arrays are not valid GGUF metadata"
                        ),
                    });
                }
            }
        }
        self.kvs.push(KvEntry {
            key: key.to_owned(),
            value,
        });
        Ok(())
    }

    /// Appends one tensor. `data` must be exactly the byte length
    /// the table implies (`dtype.data_nbytes` over reversed
    /// `shape`); R9V-typed tensors carry opaque entry bytes and pass
    /// through unchecked.
    pub fn add_tensor(
        &mut self,
        name: &str,
        shape: &[u64],
        dtype: TensorType,
        data: Vec<u8>,
    ) -> Result<(), FormatError> {
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(FormatError::DuplicateTensor {
                name: name.to_owned(),
            });
        }
        let mut file_dims = shape.to_vec();
        file_dims.reverse();
        if let Some(expected) = dtype.data_nbytes(&file_dims) {
            if expected != data.len() as u64 {
                return Err(FormatError::LengthMismatch {
                    what: "tensor data",
                    expected,
                    got: data.len() as u64,
                });
            }
        }
        self.tensors.push(OutTensor {
            name: name.to_owned(),
            shape: shape.to_vec(),
            dtype,
            data,
        });
        Ok(())
    }

    /// Returns the queued metadata entries in order.
    pub fn kvs(&self) -> &[KvEntry] {
        &self.kvs
    }

    /// Returns the queued tensors in order.
    pub fn tensors(&self) -> &[OutTensor] {
        &self.tensors
    }

    /// Returns the data alignment.
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Emits the complete file (Spec 2 §6; card A2.5). All
    /// arithmetic is checked; sizes that overflow fail instead of
    /// wrapping.
    pub fn emit(&self) -> Result<Vec<u8>, FormatError> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for kv in &self.kvs {
            encode_str(&mut out, &kv.key);
            out.extend_from_slice(&kv.value.kv_type().code().to_le_bytes());
            encode_value(&mut out, &kv.value);
        }
        // Offsets advance by alignment-padded sizes, matching
        // gguf-py's write_ti_data_to_file exactly.
        let mut offset: u64 = 0;
        let mut sizes: Vec<u64> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            encode_str(&mut out, &t.name);
            out.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            let mut d = t.shape.len();
            while d > 0 {
                d -= 1;
                // Internal invariant: `d` counts down from the shape
                // length, so indexing stays in bounds.
                out.extend_from_slice(&t.shape[d].to_le_bytes());
            }
            out.extend_from_slice(&t.dtype.code().to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            let size = t.data.len() as u64;
            sizes.push(size);
            offset = align_up(size, self.alignment)
                .and_then(|padded| offset.checked_add(padded))
                .ok_or_else(|| FormatError::Overflow {
                    what: "tensor table offset",
                    detail: format!("tensor {:?} size={size}", t.name),
                })?;
        }
        let pad =
            align_up(out.len() as u64, self.alignment).ok_or_else(|| FormatError::Overflow {
                what: "data section padding",
                detail: format!("header_len={}", out.len()),
            })? - out.len() as u64;
        out.resize(out.len() + pad as usize, 0);
        for (t, size) in self.tensors.iter().zip(sizes.iter()) {
            out.extend_from_slice(&t.data);
            let pad = align_up(*size, self.alignment).ok_or_else(|| FormatError::Overflow {
                what: "tensor data padding",
                detail: format!("tensor {:?} size={size}", t.name),
            })? - size;
            out.resize(out.len() + pad as usize, 0);
        }
        Ok(out)
    }
}

/// Exact in-entry region offsets for one native tensor (Spec 2 §6;
/// card A2.5): `values` → `scales` → (if `L1S`) `indices`, with
/// scales 256-byte aligned within the entry and the entry padded to
/// a whole number of 4 KiB tensor regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryRegions {
    /// Byte offset of the value region (always 0).
    pub values_offset: u64,
    /// Byte length of the value region.
    pub values_bytes: u64,
    /// Byte offset of the scale region (256-aligned).
    pub scales_offset: u64,
    /// Byte length of the scale region.
    pub scales_bytes: u64,
    /// Byte offset of the index region (`entry_bytes` when absent).
    pub indices_offset: u64,
    /// Byte length of the index region (0 unless `L1S`).
    pub indices_bytes: u64,
    /// Total entry bytes (4096-aligned).
    pub entry_bytes: u64,
}

impl EntryRegions {
    /// The three `r9v.tensor.<name>.regions` offsets
    /// `[values, scales, indices]` (Spec 2 §6; card A2.5). Region
    /// sizes follow by differencing against `entry_bytes`.
    // DECISION(A2.5): always three offsets with absent regions
    // pointing at entry_bytes; rejected variable-length regions
    // because fixed arity keeps the metadata schema closed. Spec 2
    // §6 lists the three regions without fixing the arity.
    pub fn offsets(&self) -> [u64; 3] {
        [self.values_offset, self.scales_offset, self.indices_offset]
    }
}

// DECISION(A2.5): scales_offset pads values_bytes up to 256 and the
// entry pads to 4096, even when the value region already aligns;
// rejected reusing sparse::L1Regions offsets verbatim because those
// pack scales immediately after values while §6 requires the 256
// alignment. Spec 2 §6.
//
// DECISION(A2.5): L0 tensors get a values-only entry (scales and
// indices offsets equal entry_bytes); rejected composing L0's
// per-row trailing scales into the triple because §2.1 stores them
// inline per row, not as a separate SoA region. Spec 2 §2.1, §6.
//
/// Derives the exact §6 region offsets for `(scheme, layout)` over
/// logical `(n, k)` (Spec 2 §6; card A2.5).
pub fn entry_regions(
    scheme: crate::SchemeId,
    layout: crate::Layout,
    n: u32,
    k: u32,
) -> Result<EntryRegions, FormatError> {
    use crate::layout::{l0_region_bytes, l0_row_stride_bytes, PaddedDims};
    use crate::{l1s_index_region_bytes, l1s_value_dims};
    match layout {
        crate::Layout::L0 => {
            let packing = crate::repack_packing(scheme)?;
            let elem_bytes: u32 = match packing {
                crate::Packing::Byte => 1,
                crate::Packing::Half16 => 2,
                crate::Packing::Nibble4 | crate::Packing::BitPlanes { .. } => {
                    return Err(FormatError::UnsupportedLayout {
                        scheme: scheme.name(),
                        layout: layout.name(),
                    });
                }
            };
            // DECISION(A2.5): L0 stride carries one f16 record slot
            // per row, the minimal stride under the A2.1 law that
            // every L0 row carries trailing scale records; rejected
            // scaleless rows because l0_row_stride_bytes requires a
            // record slot. Spec 2 §2.1, §6.
            let values_bytes = l0_region_bytes(n, l0_row_stride_bytes(k, elem_bytes, 1, 2)?)?;
            let entry_bytes =
                align_up(values_bytes, NATIVE_ALIGNMENT).ok_or_else(|| FormatError::Overflow {
                    what: "l0 entry_bytes",
                    detail: format!("values_bytes={values_bytes}"),
                })?;
            Ok(EntryRegions {
                values_offset: 0,
                values_bytes,
                scales_offset: entry_bytes,
                scales_bytes: 0,
                indices_offset: entry_bytes,
                indices_bytes: 0,
                entry_bytes,
            })
        }
        crate::Layout::L1 | crate::Layout::L1S => {
            if layout == crate::Layout::L1S
                && !matches!(
                    scheme,
                    crate::SchemeId::I8R
                        | crate::SchemeId::I8B128
                        | crate::SchemeId::I4K
                        | crate::SchemeId::E4M3B128
                )
            {
                return Err(FormatError::UnsupportedLayout {
                    scheme: scheme.name(),
                    layout: layout.name(),
                });
            }
            // Row-wise I8_R reports no outer block and pads plain to
            // tiles; every other scheme pads K to its wire block.
            let superblock = crate::repack_outer_block(scheme)?;
            let dims = PaddedDims::new(n, k, superblock)?;
            let packing = crate::repack_packing(scheme)?;
            let record_bytes = crate::repack_record_bytes(scheme)?;
            // DECISION(A2.5): L1S scale records group over the dense
            // K (same n_blocks/k_blocks as L1); rejected grouping
            // over compressed K because §3.1 fixes the grouping as
            // [N/16][K/B][16] over the tensor's K. Spec 2 §2.3, §3.1.
            let block = superblock.unwrap_or_else(|| dims.k_padded());
            let (n_blocks, k_blocks) = crate::layout::scale_block_counts(&dims, block)?;
            let records = crate::layout::scale_record_count(n_blocks, k_blocks)?;
            let scales_bytes = crate::layout::scale_region_bytes(records, record_bytes)?;
            // The outer match already handled L0, so this arm is L1
            // by elimination; it stays spelled out so a new Layout
            // variant breaks the build instead of inheriting L1 math.
            let (values_bytes, indices_bytes) = match layout {
                crate::Layout::L1S => {
                    let value_dims = l1s_value_dims(&dims, superblock)?;
                    let values_bytes = value_dims
                        .tile_count()
                        .checked_mul(packing.tile_bytes())
                        .ok_or_else(|| FormatError::Overflow {
                            what: "l1s values_bytes",
                            detail: format!("tiles={}", value_dims.tile_count()),
                        })?;
                    let indices_bytes = l1s_index_region_bytes(value_dims.tile_count())?;
                    (values_bytes, indices_bytes)
                }
                crate::Layout::L0 | crate::Layout::L1 => (dims.value_region_bytes(packing)?, 0),
            };
            let scales_offset =
                align_up(values_bytes, SCALE_ALIGN).ok_or_else(|| FormatError::Overflow {
                    what: "scales_offset",
                    detail: format!("values_bytes={values_bytes}"),
                })?;
            let indices_offset =
                scales_offset
                    .checked_add(scales_bytes)
                    .ok_or_else(|| FormatError::Overflow {
                        what: "indices_offset",
                        detail: format!(
                            "scales_offset={scales_offset} scales_bytes={scales_bytes}"
                        ),
                    })?;
            let entry_bytes = align_up(
                indices_offset
                    .checked_add(indices_bytes)
                    .ok_or_else(|| FormatError::Overflow {
                        what: "entry_bytes",
                        detail: format!(
                            "indices_offset={indices_offset} indices_bytes={indices_bytes}"
                        ),
                    })?,
                NATIVE_ALIGNMENT,
            )
            .ok_or_else(|| FormatError::Overflow {
                what: "entry_bytes",
                detail: format!(
                    "regions_end={indices_offset}+{indices_bytes} align={NATIVE_ALIGNMENT}"
                ),
            })?;
            Ok(EntryRegions {
                values_offset: 0,
                values_bytes,
                scales_offset,
                scales_bytes,
                indices_offset: if layout == crate::Layout::L1S {
                    indices_offset
                } else {
                    entry_bytes
                },
                indices_bytes,
                entry_bytes,
            })
        }
    }
}

/// `model_fp` over one shard set (Spec 9 §3; card A2.5):
/// `xxh3(file_fp ‖ every per-tensor xxh3 in table order)`.
/// `tensor_hashes` are the `r9v.tensor.*.xxh3` values — or, for a
/// standard GGUF, the per-tensor xxh3 of the raw entry bytes
/// computed during repack — in tensor-table order (shard order,
/// then table order for splits).
pub fn model_fp(file_fp: u128, tensor_hashes: &[u64]) -> u128 {
    let mut input = Vec::with_capacity(16 + 8 * tensor_hashes.len());
    input.extend_from_slice(&file_fp.to_le_bytes());
    for h in tensor_hashes {
        input.extend_from_slice(&h.to_le_bytes());
    }
    r9v_common::xxh3_128(&input)
}

/// Enforces the Spec 2 §9 version rule (card A2.5): files without
/// `r9v.format_version` are standard GGUF and always accepted;
/// `format_version ≤ R9V_FORMAT_VERSION` is accepted; anything
/// newer is a [`FormatError::FormatVersion`].
pub fn accept_format_version(version: Option<u32>) -> Result<(), FormatError> {
    match version {
        None => Ok(()),
        Some(v) if v <= R9V_FORMAT_VERSION => Ok(()),
        Some(v) => Err(FormatError::FormatVersion {
            found: v,
            max: R9V_FORMAT_VERSION,
        }),
    }
}

/// Merged view over one GGUF split shard set (Spec 9 §2 step 1;
/// card A2.5).
///
/// Shard `i` carries `split.no = i`, `split.count = N`,
/// `split.tensors.count = total` (gguf-py `add_shard_kv_data`).
/// Metadata is taken from shard 0; tensor tables concatenate in
/// shard order with per-tensor shard addressing. Duplicate tensor
/// names across shards are rejected, collecting every duplicate
/// (CONVENTIONS.md §1.4).
// DECISION(A2.5): metadata from shard 0 with split.no/count
// cross-checked, not deep-compared; rejected requiring byte-equal
// metadata because writers duplicate most but not all keys across
// shards. Spec 9 §2 is silent on merge rules.
#[derive(Debug, Clone)]
pub struct ShardSet {
    shards: Vec<GgufFile>,
    order: Vec<(usize, usize)>,
}

impl ShardSet {
    /// Opens an ordered shard set from already-parsed shards.
    /// `bytes[i]` is the raw buffer shard `i` was parsed from and
    /// is only used for fingerprint cross-checks (it may be empty
    /// to skip them).
    pub fn open(shards: Vec<GgufFile>) -> Result<Self, FormatError> {
        if shards.is_empty() {
            return Err(FormatError::Malformed {
                offset: 0,
                detail: "split shard set is empty".to_owned(),
            });
        }
        let mut problems = Vec::new();
        let mut seen: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut order = Vec::new();
        for (shard_index, shard) in shards.iter().enumerate() {
            for (tensor_index, tensor) in shard.tensors().iter().enumerate() {
                if let Some((first_shard, _)) = seen.get(&tensor.name) {
                    problems.push(FormatError::DuplicateTensor {
                        name: format!("{} (shards {first_shard} and {shard_index})", tensor.name),
                    });
                } else {
                    seen.insert(tensor.name.clone(), (shard_index, tensor_index));
                    order.push((shard_index, tensor_index));
                }
            }
        }
        // Shard KV cross-checks: every shard's split.count must agree
        // with the set size, and split.no must sequence 0..N.
        for (shard_index, shard) in shards.iter().enumerate() {
            match shard.kv("split.count") {
                Some(KvValue::U16(count)) => {
                    if *count as usize != shards.len() {
                        problems.push(FormatError::Malformed {
                            offset: 0,
                            detail: format!(
                                "shard {shard_index}: split.count is {count}, set holds {} shards",
                                shards.len(),
                            ),
                        });
                    }
                }
                Some(other) => problems.push(FormatError::KvTypeMismatch {
                    key: "split.count".to_owned(),
                    found: other.kv_type().name(),
                    expected: KvType::U16.name(),
                }),
                None => {}
            }
            match shard.kv("split.no") {
                Some(KvValue::U16(no)) => {
                    if *no as usize != shard_index {
                        problems.push(FormatError::Malformed {
                            offset: 0,
                            detail: format!(
                                "shard {shard_index}: split.no is {no}, expected {shard_index}"
                            ),
                        });
                    }
                }
                Some(other) => problems.push(FormatError::KvTypeMismatch {
                    key: "split.no".to_owned(),
                    found: other.kv_type().name(),
                    expected: KvType::U16.name(),
                }),
                None => {}
            }
        }
        FormatError::collect(problems)?;
        Ok(Self { shards, order })
    }

    /// Returns the parsed shards in order.
    pub fn shards(&self) -> &[GgufFile] {
        &self.shards
    }

    /// Returns the merged tensor count.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns `true` when the merged set holds no tensors.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Returns `(shard_index, tensor)` for merged position `i`
    /// (`None` when out of range).
    pub fn tensor_at(&self, i: usize) -> Option<(usize, &TensorInfo)> {
        let (shard_index, tensor_index) = *self.order.get(i)?;
        // Internal invariant: `order` entries are built from the
        // shards' own tables above, so both indexes stay in bounds.
        Some((
            shard_index,
            &self.shards[shard_index].tensors()[tensor_index],
        ))
    }

    /// Finds a tensor by name across shards (`None` when absent).
    pub fn tensor(&self, name: &str) -> Option<(usize, &TensorInfo)> {
        for (shard_index, shard) in self.shards.iter().enumerate() {
            if let Some(info) = shard.tensor(name) {
                return Some((shard_index, info));
            }
        }
        None
    }
}
