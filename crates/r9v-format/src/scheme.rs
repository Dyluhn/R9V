// SPDX-License-Identifier: Apache-2.0
//! Canonical quantization-scheme ids and per-scheme metadata (Spec 2 §3; card A2.2).
//!
//! [`SchemeId`] is the closed set of §3.2 native ids plus the §3.3
//! repack-only ids as reserved variants. Native scale-record contents,
//! SoA placement ([`crate::geometry::ScaleGeometry`]), exact rational
//! bits-per-weight ([`bits_per_weight`]) and reference decode live
//! here; repack rules and repack-only reference dequant fail closed
//! with their owning card (A2.3/A2.4) via
//! [`FormatError::ReservedScheme`].
//!
//! Code ownership: this crate assigns the stable codes, mirroring how
//! card A2.1 owns layout codes. [`r9v_ir::SchemeId`] stays an opaque
//! `u64` handle that only transports them (card A1.1), so there is one
//! code table, not two. Conversions are explicit ([`SchemeId::to_ir`],
//! [`SchemeId::from_ir`]) and total on the known set.

use std::fmt;
use std::str::FromStr;

use crate::FormatError;

/// Canonical quantization-scheme id (Spec 2 §3; card A2.2).
///
/// Closed enum: adding a scheme is a spec change (Spec 2 §9), and every
/// `match` stays exhaustive with no wildcard arm (CONVENTIONS.md §3.2).
/// Stable codes are part of the contract (§9: ids are immutable); see
/// [`SchemeId::code`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemeId {
    /// `i8` per row, `s: f16` per row stored `[N]` (Spec 2 §3.2).
    I8R,
    /// `i8` per 128, `s: f16` per block (Spec 2 §3.2).
    I8B128,
    /// `u4` 32-in-256, Q4_K-identical 16 B record per superblock
    /// (Spec 2 §3.2; DECISIONS.md D-004).
    I4K,
    /// `e4m3` per 128, `s: f16` per block (Spec 2 §3.2).
    E4M3B128,
    /// `Q8_0` repack: `i8` per 32, `s: f16` (Spec 2 §3.3; card A2.3).
    I8B32F,
    /// `Q4_0` repack: `u4` per 32, `s: f16` (Spec 2 §3.3; card A2.3).
    I4B32F,
    /// `Q4_1` repack: `u4` per 32, `s: f16, m: f16`
    /// (Spec 2 §3.3; card A2.3).
    I4B32FM,
    /// `Q5_0` repack: `u5` per 32 (Spec 2 §3.3; card A2.3).
    I5B32F,
    /// `Q5_1` repack: `u5` per 32 (Spec 2 §3.3; card A2.3).
    I5B32FM,
    /// `Q5_K` repack: `u5` K-family (Spec 2 §3.3; card A2.3).
    I5K,
    /// `Q6_K` repack: `i6` K-family (Spec 2 §3.3; card A2.3).
    I6K,
    /// `Q3_K` repack: `u3` K-family (Spec 2 §3.3; card A2.3).
    I3K,
    /// `Q2_K` repack: `u2` K-family (Spec 2 §3.3; card A2.3).
    I2K,
    /// `IQ4_NL` repack: 16-entry LUT (Spec 2 §3.3; card A2.4).
    I4Nl,
    /// `IQ4_XS` repack: 16-entry LUT (Spec 2 §3.3; card A2.4).
    I4Xs,
    /// `IQ3_XXS` repack: codebook (Spec 2 §3.3; card A2.4).
    Iq3Xxs,
    /// `IQ3_S` repack: codebook (Spec 2 §3.3; card A2.4).
    Iq3S,
    /// `IQ2_XXS` repack: codebook (Spec 2 §3.3; card A2.4).
    Iq2Xxs,
    /// `IQ2_XS` repack: codebook (Spec 2 §3.3; card A2.4).
    Iq2Xs,
    /// `IQ2_S` repack: codebook (Spec 2 §3.3; card A2.4).
    Iq2S,
    /// `IQ1_S` repack: codebook (Spec 2 §3.3; card A2.4).
    Iq1S,
    /// `IQ1_M` repack: codebook (Spec 2 §3.3; card A2.4).
    Iq1M,
}
// DECISION(A2.2): stable codes are contiguous in spec-table order
// (§3.2 natives 1-4, then §3.3 repacks 5-22); rejected reusing GGUF
// upstream type ids because repack-only ids share GGUF's space while
// native ids need R9V-owned codes, and one contiguous table keeps the
// IR-handle mapping total. Spec 2 §3 assigns no numeric codes.
impl SchemeId {
    /// All ids in code order (Spec 2 §3.2 then §3.3).
    pub const ALL: [SchemeId; 22] = [
        SchemeId::I8R,
        SchemeId::I8B128,
        SchemeId::I4K,
        SchemeId::E4M3B128,
        SchemeId::I8B32F,
        SchemeId::I4B32F,
        SchemeId::I4B32FM,
        SchemeId::I5B32F,
        SchemeId::I5B32FM,
        SchemeId::I5K,
        SchemeId::I6K,
        SchemeId::I3K,
        SchemeId::I2K,
        SchemeId::I4Nl,
        SchemeId::I4Xs,
        SchemeId::Iq3Xxs,
        SchemeId::Iq3S,
        SchemeId::Iq2Xxs,
        SchemeId::Iq2Xs,
        SchemeId::Iq2S,
        SchemeId::Iq1S,
        SchemeId::Iq1M,
    ];

    /// Returns the stable code (Spec 2 §9: ids are immutable).
    pub const fn code(self) -> u64 {
        match self {
            SchemeId::I8R => 1,
            SchemeId::I8B128 => 2,
            SchemeId::I4K => 3,
            SchemeId::E4M3B128 => 4,
            SchemeId::I8B32F => 5,
            SchemeId::I4B32F => 6,
            SchemeId::I4B32FM => 7,
            SchemeId::I5B32F => 8,
            SchemeId::I5B32FM => 9,
            SchemeId::I5K => 10,
            SchemeId::I6K => 11,
            SchemeId::I3K => 12,
            SchemeId::I2K => 13,
            SchemeId::I4Nl => 14,
            SchemeId::I4Xs => 15,
            SchemeId::Iq3Xxs => 16,
            SchemeId::Iq3S => 17,
            SchemeId::Iq2Xxs => 18,
            SchemeId::Iq2Xs => 19,
            SchemeId::Iq2S => 20,
            SchemeId::Iq1S => 21,
            SchemeId::Iq1M => 22,
        }
    }

    /// Decodes a stable code; unknown codes are errors, never guesses
    /// (Spec 2 §9: a changed scale record is a new scheme id).
    pub fn from_code(code: u64) -> Result<Self, FormatError> {
        match code {
            1 => Ok(SchemeId::I8R),
            2 => Ok(SchemeId::I8B128),
            3 => Ok(SchemeId::I4K),
            4 => Ok(SchemeId::E4M3B128),
            5 => Ok(SchemeId::I8B32F),
            6 => Ok(SchemeId::I4B32F),
            7 => Ok(SchemeId::I4B32FM),
            8 => Ok(SchemeId::I5B32F),
            9 => Ok(SchemeId::I5B32FM),
            10 => Ok(SchemeId::I5K),
            11 => Ok(SchemeId::I6K),
            12 => Ok(SchemeId::I3K),
            13 => Ok(SchemeId::I2K),
            14 => Ok(SchemeId::I4Nl),
            15 => Ok(SchemeId::I4Xs),
            16 => Ok(SchemeId::Iq3Xxs),
            17 => Ok(SchemeId::Iq3S),
            18 => Ok(SchemeId::Iq2Xxs),
            19 => Ok(SchemeId::Iq2Xs),
            20 => Ok(SchemeId::Iq2S),
            21 => Ok(SchemeId::Iq1S),
            22 => Ok(SchemeId::Iq1M),
            _ => Err(FormatError::UnknownScheme {
                value: code.to_string(),
            }),
        }
    }

    /// Returns the stable lowercase name (CONVENTIONS.md §3.2:
    /// serialization uses names, never discriminants).
    pub const fn name(self) -> &'static str {
        match self {
            SchemeId::I8R => "i8_r",
            SchemeId::I8B128 => "i8_b128",
            SchemeId::I4K => "i4_k",
            SchemeId::E4M3B128 => "e4m3_b128",
            SchemeId::I8B32F => "i8_b32f",
            SchemeId::I4B32F => "i4_b32f",
            SchemeId::I4B32FM => "i4_b32fm",
            SchemeId::I5B32F => "i5_b32f",
            SchemeId::I5B32FM => "i5_b32fm",
            SchemeId::I5K => "i5_k",
            SchemeId::I6K => "i6_k",
            SchemeId::I3K => "i3_k",
            SchemeId::I2K => "i2_k",
            SchemeId::I4Nl => "i4_nl",
            SchemeId::I4Xs => "i4_xs",
            SchemeId::Iq3Xxs => "iq3_xxs",
            SchemeId::Iq3S => "iq3_s",
            SchemeId::Iq2Xxs => "iq2_xxs",
            SchemeId::Iq2Xs => "iq2_xs",
            SchemeId::Iq2S => "iq2_s",
            SchemeId::Iq1S => "iq1_s",
            SchemeId::Iq1M => "iq1_m",
        }
    }

    /// Parses a stable name; anything else is an error (Spec 2 §3).
    pub fn from_name(name: &str) -> Result<Self, FormatError> {
        match name {
            "i8_r" => Ok(SchemeId::I8R),
            "i8_b128" => Ok(SchemeId::I8B128),
            "i4_k" => Ok(SchemeId::I4K),
            "e4m3_b128" => Ok(SchemeId::E4M3B128),
            "i8_b32f" => Ok(SchemeId::I8B32F),
            "i4_b32f" => Ok(SchemeId::I4B32F),
            "i4_b32fm" => Ok(SchemeId::I4B32FM),
            "i5_b32f" => Ok(SchemeId::I5B32F),
            "i5_b32fm" => Ok(SchemeId::I5B32FM),
            "i5_k" => Ok(SchemeId::I5K),
            "i6_k" => Ok(SchemeId::I6K),
            "i3_k" => Ok(SchemeId::I3K),
            "i2_k" => Ok(SchemeId::I2K),
            "i4_nl" => Ok(SchemeId::I4Nl),
            "i4_xs" => Ok(SchemeId::I4Xs),
            "iq3_xxs" => Ok(SchemeId::Iq3Xxs),
            "iq3_s" => Ok(SchemeId::Iq3S),
            "iq2_xxs" => Ok(SchemeId::Iq2Xxs),
            "iq2_xs" => Ok(SchemeId::Iq2Xs),
            "iq2_s" => Ok(SchemeId::Iq2S),
            "iq1_s" => Ok(SchemeId::Iq1S),
            "iq1_m" => Ok(SchemeId::Iq1M),
            _ => Err(FormatError::UnknownScheme {
                value: name.to_owned(),
            }),
        }
    }

    /// Converts to the opaque IR handle (Spec 1 §2.2, Spec 2 §3).
    /// `r9v-ir` transports the code; this crate owns its meaning.
    pub fn to_ir(self) -> r9v_ir::SchemeId {
        r9v_ir::SchemeId::new(self.code())
    }

    /// Converts from the opaque IR handle (Spec 1 §2.2, Spec 2 §3).
    /// IR codes outside the closed set (future schemes the IR must
    /// decode per card A1.1) are errors here.
    pub fn from_ir(id: r9v_ir::SchemeId) -> Result<Self, FormatError> {
        Self::from_code(id.as_u64())
    }

    /// Whether the quant tool emits this scheme (Spec 2 §3.2 natives)
    /// as opposed to the loader producing it by repack (§3.3).
    pub const fn is_native(self) -> bool {
        match self {
            SchemeId::I8R | SchemeId::I8B128 | SchemeId::I4K | SchemeId::E4M3B128 => true,
            SchemeId::I8B32F
            | SchemeId::I4B32F
            | SchemeId::I4B32FM
            | SchemeId::I5B32F
            | SchemeId::I5B32FM
            | SchemeId::I5K
            | SchemeId::I6K
            | SchemeId::I3K
            | SchemeId::I2K
            | SchemeId::I4Nl
            | SchemeId::I4Xs
            | SchemeId::Iq3Xxs
            | SchemeId::Iq3S
            | SchemeId::Iq2Xxs
            | SchemeId::Iq2Xs
            | SchemeId::Iq2S
            | SchemeId::Iq1S
            | SchemeId::Iq1M => false,
        }
    }

    /// The card that owns this scheme's record and decode behavior:
    /// `A2.2` for natives, `A2.3`/`A2.4` for repack-only ids
    /// (phase-a-agent-breakdown §A2.3, §A2.4).
    pub const fn owner_card(self) -> &'static str {
        match self {
            SchemeId::I8R | SchemeId::I8B128 | SchemeId::I4K | SchemeId::E4M3B128 => "A2.2",
            SchemeId::I8B32F
            | SchemeId::I4B32F
            | SchemeId::I4B32FM
            | SchemeId::I5B32F
            | SchemeId::I5B32FM
            | SchemeId::I5K
            | SchemeId::I6K
            | SchemeId::I3K
            | SchemeId::I2K => "A2.3",
            SchemeId::I4Nl
            | SchemeId::I4Xs
            | SchemeId::Iq3Xxs
            | SchemeId::Iq3S
            | SchemeId::Iq2Xxs
            | SchemeId::Iq2Xs
            | SchemeId::Iq2S
            | SchemeId::Iq1S
            | SchemeId::Iq1M => "A2.4",
        }
    }
}

impl fmt::Display for SchemeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for SchemeId {
    type Err = FormatError;
    /// Parses a stable scheme name (see [`SchemeId::from_name`]).
    fn from_str(s: &str) -> Result<Self, FormatError> {
        Self::from_name(s)
    }
}
/// Native schemes with A2.2-owned records and decode (Spec 2 §3.2).
///
/// One 22-arm classification point ([`SchemeId::as_native`]) so every
/// other function matches four arms instead of repeating the reserved
/// list. Crate-private: the public surface stays [`SchemeId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum NativeScheme {
    I8R,
    I8B128,
    I4K,
    E4M3B128,
}

impl SchemeId {
    /// Classifies `self`, failing closed for repack-only ids with
    /// their owning card (Spec 2 §3.3; that behavior is A2.3/A2.4's).
    pub(crate) fn as_native(self) -> Result<NativeScheme, FormatError> {
        match self {
            SchemeId::I8R => Ok(NativeScheme::I8R),
            SchemeId::I8B128 => Ok(NativeScheme::I8B128),
            SchemeId::I4K => Ok(NativeScheme::I4K),
            SchemeId::E4M3B128 => Ok(NativeScheme::E4M3B128),
            SchemeId::I8B32F
            | SchemeId::I4B32F
            | SchemeId::I4B32FM
            | SchemeId::I5B32F
            | SchemeId::I5B32FM
            | SchemeId::I5K
            | SchemeId::I6K
            | SchemeId::I3K
            | SchemeId::I2K
            | SchemeId::I4Nl
            | SchemeId::I4Xs
            | SchemeId::Iq3Xxs
            | SchemeId::Iq3S
            | SchemeId::Iq2Xxs
            | SchemeId::Iq2Xs
            | SchemeId::Iq2S
            | SchemeId::Iq1S
            | SchemeId::Iq1M => Err(FormatError::ReservedScheme {
                scheme: self.name(),
                owner: self.owner_card(),
            }),
        }
    }
}

/// Exact bits-per-weight as `(bits, weights)` for `k` weights of one
/// row (Spec 2 §8 "including all scale overhead"; card A2.2).
///
/// The pair is unreduced so the bit counts stay readable: `I4_K` over a
/// 256-superblock is `(1152, 256)` = 4.5 (`2+2+12+128` bytes per
/// Q4_K-identical block), `I8_B128`/`E4M3_B128` over a 128-block are
/// `(1040, 128)` = 8.125, and `I8_R` over a row of `k` is `(8*k+16, k)`
/// (one `f16` per row; §8 prints this as 8.0). `k` carries the scheme's
/// block-divisibility requirement (128, 256); repack-only ids fail
/// closed with their owning card. The `s24` sparse multiplier is not
/// applied here: index geometry is owned by card A2.1 (SI-15).
pub fn bits_per_weight(scheme: SchemeId, k: u32) -> Result<(u64, u64), FormatError> {
    match scheme.as_native()? {
        NativeScheme::I8R => {
            if k == 0 {
                return Err(FormatError::InvalidDim {
                    name: "k",
                    value: 0,
                    reason: "must be at least 1",
                });
            }
            let bits = (k as u64)
                .checked_mul(8)
                .and_then(|v| v.checked_add(16))
                .ok_or_else(|| FormatError::Overflow {
                    what: "i8_r bits_per_weight",
                    detail: format!("k={k}"),
                })?;
            Ok((bits, k as u64))
        }
        NativeScheme::I8B128 | NativeScheme::E4M3B128 => {
            if k == 0 || !k.is_multiple_of(128) {
                return Err(FormatError::InvalidBlock {
                    name: "k",
                    value: k as u64,
                    reason: "must be a nonzero multiple of 128",
                });
            }
            let bits = (k as u64 / 128)
                .checked_mul(1040)
                .ok_or_else(|| FormatError::Overflow {
                    what: "b128 bits_per_weight",
                    detail: format!("k={k}"),
                })?;
            Ok((bits, k as u64))
        }
        NativeScheme::I4K => {
            if k == 0 || !k.is_multiple_of(256) {
                return Err(FormatError::InvalidBlock {
                    name: "k",
                    value: k as u64,
                    reason: "must be a nonzero multiple of 256",
                });
            }
            let bits = (k as u64 / 256)
                .checked_mul(1152)
                .ok_or_else(|| FormatError::Overflow {
                    what: "i4_k bits_per_weight",
                    detail: format!("k={k}"),
                })?;
            Ok((bits, k as u64))
        }
    }
}
