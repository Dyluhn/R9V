// SPDX-License-Identifier: Apache-2.0
//! Format error type (Spec 2 §2, CONVENTIONS.md §1).
//!
//! Every function that touches untrusted dimensions or buffers returns
//! [`FormatError`]; validation collects every problem before returning
//! (CONVENTIONS.md §1.4). Nothing here panics or saturates: all arithmetic
//! on untrusted sizes is checked and reported.

/// Format-layer failure (Spec 2 §2; CONVENTIONS.md §1.1, §1.3).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FormatError {
    /// A dimension was zero, misaligned, or out of range where the spec
    /// requires tiled extents (Spec 2 §2.2: weights are `[N, K]` tiles).
    #[error("invalid dimension {name} = {value}: {reason} (Spec 2 §2)")]
    InvalidDim {
        /// Which dimension failed (`n`, `k`, `lane`, ...).
        name: &'static str,
        /// The offending value.
        value: u64,
        /// Why it is rejected.
        reason: &'static str,
    },
    /// A superblock or block size was zero, misaligned, or did not
    /// divide padded K (Spec 2 §2.2: K pads to the scheme superblock
    /// where one exists; the superblock must itself be tile-aligned).
    #[error("invalid block size {name} = {value}: {reason} (Spec 2 §2.2)")]
    InvalidBlock {
        /// Which block parameter failed (`superblock_k`, `block_b`, ...).
        name: &'static str,
        /// The offending value.
        value: u64,
        /// Why it is rejected.
        reason: &'static str,
    },
    /// Checked size arithmetic overflowed instead of saturating
    /// (CONVENTIONS.md §1.5: untrusted input must not wrap or saturate).
    #[error("size computation for {what} overflowed: {detail} (Spec 2 §2)")]
    Overflow {
        /// What was being computed (`padded_k`, `region_bytes`, ...).
        what: &'static str,
        /// Operands involved.
        detail: String,
    },
    /// An unknown layout code or name was supplied (Spec 2 §2.4, §9:
    /// layout ids are immutable; decoding an unknown id is an error,
    /// never a guess).
    #[error("unknown layout {value} (Spec 2 §2)")]
    UnknownLayout {
        /// The unrecognized code or name.
        value: String,
    },
    /// An unsupported element bit width was supplied for the bit-plane
    /// packing (Spec 2 §2.2 table, §3.3: planes exist for 2/3/5/6-bit
    /// types).
    #[error("unsupported bit-plane width {bits} (Spec 2 §2.2)")]
    InvalidBitWidth {
        /// The offending width in bits.
        bits: u8,
    },
    /// A buffer did not have the byte length the geometry requires
    /// (Spec 2 §2, §7).
    #[error("length mismatch for {what}: expected {expected} bytes, got {got} (Spec 2 §2)")]
    LengthMismatch {
        /// What the buffer should hold (`l1 tile region`, ...).
        what: &'static str,
        /// Required byte length.
        expected: u64,
        /// Actual byte length.
        got: u64,
    },
    /// An unknown scheme code or name was supplied (Spec 2 §3, §9:
    /// scheme ids are immutable; decoding an unknown id is an error,
    /// never a guess).
    #[error("unknown scheme {value} (Spec 2 §3)")]
    UnknownScheme {
        /// The unrecognized code or name.
        value: String,
    },
    /// A repack-only scheme was used where only native behavior exists
    /// (Spec 2 §3.3: repack rules and reference dequant for these ids
    /// are owned by cards A2.3/A2.4, not A2.2).
    #[error("scheme {scheme} is reserved for {owner} (Spec 2 §3.3)")]
    ReservedScheme {
        /// The repack-only scheme name (`i8_b32f`, ...).
        scheme: &'static str,
        /// The card that owns this scheme's behavior (`A2.3`, `A2.4`).
        owner: &'static str,
    },
    /// A value or scale record of the wrong kind was passed for the
    /// scheme (Spec 2 §3.2: each native scheme fixes its value bits
    /// and scale record format).
    #[error("scheme {scheme} expects {expected}, got {got} (Spec 2 §3.2)")]
    SchemeMismatch {
        /// The scheme that was requested.
        scheme: &'static str,
        /// The value/scale kind the scheme requires.
        expected: &'static str,
        /// The value/scale kind that was supplied.
        got: &'static str,
    },
    /// A scale record failed validity: NaN or infinite, negative, or
    /// not representable in its stored dtype (Spec 2 §3.2: scales are
    /// non-negative finite multipliers; super-scales come from maxima).
    #[error("invalid scale for {scheme} record {record}: {reason} (Spec 2 §3.2)")]
    InvalidScale {
        /// The scheme owning the record.
        scheme: &'static str,
        /// Index of the offending record in input order.
        record: u64,
        /// `f32::to_bits` of the offending scale value.
        bits: u32,
        /// Why it is rejected (`nan`, `infinite`, `negative`, ...).
        reason: &'static str,
    },
    /// SoA scale geometry was requested for a layout that does not use
    /// it (Spec 2 §2.1 vs §3.1: `L0` rows carry trailing scale records
    /// composed with the `l0_*` helpers, not the SoA region).
    #[error("scheme {scheme} has no SoA scale region on layout {layout} (Spec 2 §3.1)")]
    UnsupportedLayout {
        /// The scheme that was requested.
        scheme: &'static str,
        /// The layout that was supplied.
        layout: &'static str,
    },
    /// A logical element value did not fit its packing (Spec 2 §2.2
    /// table: nibbles hold 0..16, L1S indices hold 0..4).
    #[error("value {value} at position {position} does not fit {what} (Spec 2 §2)")]
    ValueOutOfRange {
        /// What the value should fit (`nibble`, `l1s index`, ...).
        what: &'static str,
        /// Position of the offending element in row-major order.
        position: u64,
        /// The offending value.
        value: u64,
    },
    /// Tile padding that must read back as zero did not (Spec 2 §2.2:
    /// padding rows and columns are zero).
    #[error("nonzero padding at row {row}, col {col}: value {value} (Spec 2 §2.2)")]
    PaddingNonzero {
        /// Padded row holding the nonzero value.
        row: u32,
        /// Padded column holding the nonzero value.
        col: u32,
        /// The offending value (widened to u64 for uniform reporting).
        value: u64,
    },
    /// Multiple validation problems found; every problem is reported,
    /// never just the first (CONVENTIONS.md §1.3, §1.4).
    #[error("{problems:?}")]
    Multiple {
        /// Every problem found, in deterministic input order.
        problems: Box<[FormatError]>,
    },
}

impl FormatError {
    /// Collects per-item results into one [`FormatError`], preserving
    /// input order (CONVENTIONS.md §1.4). Returns `Ok` when empty, the
    /// single problem when there is one, and [`FormatError::Multiple`]
    /// otherwise.
    pub fn collect<I>(problems: I) -> Result<(), FormatError>
    where
        I: IntoIterator<Item = FormatError>,
    {
        let problems: Box<[FormatError]> = problems.into_iter().collect();
        if problems.is_empty() {
            Ok(())
        } else if problems.len() == 1 {
            let mut problems = problems.into_vec();
            // Internal invariant: this branch runs only when len == 1.
            Err(problems.pop().expect("problems holds exactly one entry"))
        } else {
            Err(FormatError::Multiple { problems })
        }
    }
}
