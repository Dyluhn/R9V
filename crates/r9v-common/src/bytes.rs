// SPDX-License-Identifier: Apache-2.0
//! Strict byte-size parsing and formatting (Spec 12 §3, Spec 14 §2, CONVENTIONS.md §1.3).

use std::fmt;
use std::str::FromStr;

// DECISION(A0.4): byte-size parsing treats B, K/KB/KiB, M/MB/MiB, G/GB/GiB, T/TB/TiB as binary powers of 1024; rejected decimal 1000-based multipliers because memory/VRAM budgets in Spec 12 §3 and Spec 9 use standard binary allocation sizes.

/// Multiplier for binary kilo-units (2^10) (Spec 14 §2).
pub const KIB: u64 = 1024;
/// Multiplier for binary mega-units (2^20) (Spec 14 §2).
pub const MIB: u64 = 1024 * KIB;
/// Multiplier for binary giga-units (2^30) (Spec 14 §2).
pub const GIB: u64 = 1024 * MIB;
/// Multiplier for binary tera-units (2^40) (Spec 14 §2).
pub const TIB: u64 = 1024 * GIB;
/// Multiplier for binary peta-units (2^50) (Spec 14 §2).
pub const PIB: u64 = 1024 * TIB;

/// Errors arising from strict byte-size parsing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ByteSizeError {
    /// The input string was empty or contained only whitespace.
    #[error("byte size string cannot be empty")]
    Empty,

    /// The input string did not start with a valid numeric component.
    #[error("byte size string must start with a digit: '{input}'")]
    MissingNumber {
        /// The original input string.
        input: String,
    },

    /// The numeric component was malformed.
    #[error("invalid byte size number in '{input}': {reason}")]
    InvalidNumber {
        /// The original input string.
        input: String,
        /// Detail on why the number could not be parsed.
        reason: String,
    },

    /// Negative values are disallowed.
    #[error("negative byte size is not allowed: '{input}'")]
    Negative {
        /// The original input string.
        input: String,
    },

    /// The byte size resolved to a fractional (non-integral) number of bytes.
    #[error("byte size must resolve to an exact integral number of bytes: '{input}'")]
    FractionalByte {
        /// The original input string.
        input: String,
    },

    /// An unrecognized or unsupported unit suffix was encountered.
    #[error("unknown byte size unit '{unit}' in '{input}'")]
    UnknownUnit {
        /// The unrecognized unit suffix.
        unit: String,
        /// The original input string.
        input: String,
    },

    /// Calculating the byte size resulted in integer overflow.
    #[error("byte size calculation overflowed in '{input}'")]
    Overflow {
        /// The original input string.
        input: String,
    },

    /// Unexpected trailing characters remained after the unit.
    #[error("unexpected trailing characters '{trailing}' in '{input}'")]
    TrailingCharacters {
        /// The unexpected trailing characters.
        trailing: String,
        /// The original input string.
        input: String,
    },
}

/// Parses a byte size string into bytes as a `u64` (Spec 12 §3, Spec 14 §2).
///
/// Parsing is strictly checked with exact decimal arithmetic:
/// - Leading and trailing whitespace is trimmed.
/// - Negative values are rejected.
/// - Non-integral byte amounts are rejected with [`ByteSizeError::FractionalByte`].
/// - Units supported (case-insensitive):
///   - `B` or omitted: bytes
///   - `K`, `KB`, `KiB`: 1024 B
///   - `M`, `MB`, `MiB`: 1024^2 B (1,048,576 B)
///   - `G`, `GB`, `GiB`: 1024^3 B (1,073,741,824 B)
///   - `T`, `TB`, `TiB`: 1024^4 B (1,099,511,627,776 B)
///   - `P`, `PB`, `PiB`: 1024^5 B (1,125,899,906,842,624 B)
/// - Any trailing garbage or integer overflow produces an error.
pub fn parse_byte_size(input: &str) -> Result<u64, ByteSizeError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ByteSizeError::Empty);
    }

    if trimmed.starts_with('-') {
        return Err(ByteSizeError::Negative {
            input: input.to_owned(),
        });
    }

    let mut split_idx = 0;
    let mut has_digits = false;
    let mut has_dot = false;

    for (i, c) in trimmed.char_indices() {
        if c.is_ascii_digit() {
            has_digits = true;
            split_idx = i + c.len_utf8();
        } else if c == '.' {
            if has_dot {
                return Err(ByteSizeError::InvalidNumber {
                    input: input.to_owned(),
                    reason: "multiple decimal points encountered".to_owned(),
                });
            }
            has_dot = true;
            split_idx = i + c.len_utf8();
        } else {
            break;
        }
    }

    if !has_digits {
        return Err(ByteSizeError::MissingNumber {
            input: input.to_owned(),
        });
    }

    let num_str = &trimmed[..split_idx];
    let remainder = trimmed[split_idx..].trim_start();

    // Check for trailing characters after unit
    let unit_str = match remainder.find(char::is_whitespace) {
        Some(idx) => {
            let (u, rest) = remainder.split_at(idx);
            let rest_trimmed = rest.trim();
            if !rest_trimmed.is_empty() {
                return Err(ByteSizeError::TrailingCharacters {
                    trailing: rest_trimmed.to_owned(),
                    input: input.to_owned(),
                });
            }
            u
        }
        None => remainder,
    };

    let multiplier = unit_multiplier(unit_str).ok_or_else(|| ByteSizeError::UnknownUnit {
        unit: unit_str.to_owned(),
        input: input.to_owned(),
    })?;

    // Exact checked decimal arithmetic without float rounding
    if let Some((int_part, frac_raw)) = num_str.split_once('.') {
        if int_part.is_empty() {
            return Err(ByteSizeError::MissingNumber {
                input: input.to_owned(),
            });
        }
        let int_val: u128 = int_part.parse().map_err(|e: std::num::ParseIntError| {
            if e.kind() == &std::num::IntErrorKind::PosOverflow {
                ByteSizeError::Overflow {
                    input: input.to_owned(),
                }
            } else {
                ByteSizeError::InvalidNumber {
                    input: input.to_owned(),
                    reason: e.to_string(),
                }
            }
        })?;

        let frac_trimmed = frac_raw.trim_end_matches('0');
        let frac_bytes = if frac_trimmed.is_empty() {
            0u128
        } else {
            let frac_len = frac_trimmed.len();
            if frac_len > 28 {
                return Err(ByteSizeError::FractionalByte {
                    input: input.to_owned(),
                });
            }
            let frac_val: u128 = frac_trimmed.parse().map_err(|e: std::num::ParseIntError| {
                if e.kind() == &std::num::IntErrorKind::PosOverflow {
                    ByteSizeError::Overflow {
                        input: input.to_owned(),
                    }
                } else {
                    ByteSizeError::InvalidNumber {
                        input: input.to_owned(),
                        reason: e.to_string(),
                    }
                }
            })?;

            let pow10 =
                10u128
                    .checked_pow(frac_len as u32)
                    .ok_or_else(|| ByteSizeError::Overflow {
                        input: input.to_owned(),
                    })?;

            let numerator = frac_val.checked_mul(multiplier as u128).ok_or_else(|| {
                ByteSizeError::Overflow {
                    input: input.to_owned(),
                }
            })?;

            if !numerator.is_multiple_of(pow10) {
                return Err(ByteSizeError::FractionalByte {
                    input: input.to_owned(),
                });
            }

            numerator / pow10
        };

        let int_bytes =
            int_val
                .checked_mul(multiplier as u128)
                .ok_or_else(|| ByteSizeError::Overflow {
                    input: input.to_owned(),
                })?;

        let total_bytes =
            int_bytes
                .checked_add(frac_bytes)
                .ok_or_else(|| ByteSizeError::Overflow {
                    input: input.to_owned(),
                })?;

        if total_bytes > (u64::MAX as u128) {
            return Err(ByteSizeError::Overflow {
                input: input.to_owned(),
            });
        }

        Ok(total_bytes as u64)
    } else {
        let val: u128 = num_str.parse().map_err(|e: std::num::ParseIntError| {
            if e.kind() == &std::num::IntErrorKind::PosOverflow {
                ByteSizeError::Overflow {
                    input: input.to_owned(),
                }
            } else {
                ByteSizeError::InvalidNumber {
                    input: input.to_owned(),
                    reason: e.to_string(),
                }
            }
        })?;

        let total = val
            .checked_mul(multiplier as u128)
            .ok_or_else(|| ByteSizeError::Overflow {
                input: input.to_owned(),
            })?;

        if total > (u64::MAX as u128) {
            return Err(ByteSizeError::Overflow {
                input: input.to_owned(),
            });
        }

        Ok(total as u64)
    }
}

fn unit_multiplier(unit: &str) -> Option<u64> {
    let lower = unit.to_ascii_lowercase();
    match lower.as_str() {
        "" | "b" | "bytes" | "byte" => Some(1),
        "k" | "kb" | "kib" => Some(KIB),
        "m" | "mb" | "mib" => Some(MIB),
        "g" | "gb" | "gib" => Some(GIB),
        "t" | "tb" | "tib" => Some(TIB),
        "p" | "pb" | "pib" => Some(PIB),
        _ => None,
    }
}

/// Formats a byte count into a human-readable string with binary units (Spec 12 §3).
pub fn format_byte_size(bytes: u64) -> String {
    if bytes >= PIB && bytes.is_multiple_of(PIB) {
        format!("{} PiB", bytes / PIB)
    } else if bytes >= PIB {
        format!("{:.2} PiB", bytes as f64 / PIB as f64)
    } else if bytes >= TIB && bytes.is_multiple_of(TIB) {
        format!("{} TiB", bytes / TIB)
    } else if bytes >= TIB {
        format!("{:.2} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB && bytes.is_multiple_of(GIB) {
        format!("{} GiB", bytes / GIB)
    } else if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB && bytes.is_multiple_of(MIB) {
        format!("{} MiB", bytes / MIB)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB && bytes.is_multiple_of(KIB) {
        format!("{} KiB", bytes / KIB)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Strongly-typed byte-size wrapper (Spec 12 §3, Spec 14 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteSize(u64);

impl ByteSize {
    /// Creates a new [`ByteSize`] (Spec 14 §2).
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// Returns the number of bytes as a `u64` (Spec 14 §2).
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_byte_size(self.0))
    }
}

impl FromStr for ByteSize {
    type Err = ByteSizeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_byte_size(s).map(Self)
    }
}
