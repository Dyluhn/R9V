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
    /// An unknown GGUF `ggml_type` code was supplied (Spec 2 §7 step
    /// 1: unknown type → hard error naming the type).
    #[error("unknown ggml type {code} (Spec 2 §7)")]
    UnknownGgmlType {
        /// The unrecognized numeric `ggml_type` code.
        code: u32,
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
    /// The first four bytes were not the GGUF magic (Spec 2 §6: the
    /// container is GGUF v3; card A2.5).
    #[error("bad GGUF magic {found:#010x}, expected 0x46554747 (Spec 2 §6)")]
    BadMagic {
        /// The little-endian `u32` actually present at offset 0.
        found: u32,
    },
    /// The GGUF version is not one this reader accepts (Spec 2 §6:
    /// GGUF v3; gguf-py also reads v2, so v2 is named, not guessed).
    #[error("unsupported GGUF version {found}, accepts {accepted:?} (Spec 2 §6)")]
    UnsupportedVersion {
        /// The version found in the file.
        found: u32,
        /// The versions this reader parses.
        accepted: Vec<u32>,
    },
    /// The buffer ends before a complete field (Spec 2 §6; card A2.5).
    /// Carries the byte offset where decoding stopped and how many
    /// bytes the field needed.
    #[error("truncated GGUF at offset {offset}: needed {need} byte(s) for {what} (Spec 2 §6)")]
    Truncated {
        /// Byte offset where decoding stopped.
        offset: u64,
        /// Bytes the field required from that offset.
        need: u64,
        /// Which field was being decoded (`magic`, `kv key`, ...).
        what: &'static str,
    },
    /// A field decoded but is structurally invalid (Spec 2 §6; card
    /// A2.5). Lengths that would overflow, impossible dimension
    /// counts, and invalid UTF-8 all land here with their offset.
    #[error("malformed GGUF at offset {offset}: {detail} (Spec 2 §6)")]
    Malformed {
        /// Byte offset of the offending field.
        offset: u64,
        /// What is wrong, with the numbers.
        detail: String,
    },
    /// A metadata key appears twice (matches the gguf-py reader,
    /// which rejects duplicates; card A2.5).
    #[error("duplicate metadata key {key:?} (Spec 2 §6)")]
    DuplicateKey {
        /// The repeated key.
        key: String,
    },
    /// A tensor name appears twice in one shard (matches the gguf-py
    /// reader, which rejects duplicates; card A2.5).
    #[error("duplicate tensor name {name:?} (Spec 2 §6)")]
    DuplicateTensor {
        /// The repeated tensor name.
        name: String,
    },
    /// `general.alignment` is zero or not a power of two (matches the
    /// gguf-py reader rule; Spec 2 §6 fixes native files at 4096).
    #[error("invalid alignment {value}: must be a nonzero power of two (Spec 2 §6)")]
    InvalidAlignment {
        /// The offending alignment value.
        value: u64,
    },
    /// A metadata value had a different GGUF type than the accessor
    /// required (Spec 2 §6; card A2.5).
    #[error("metadata key {key:?} has type {found}, expected {expected} (Spec 2 §6)")]
    KvTypeMismatch {
        /// The key that was read.
        key: String,
        /// The GGUF value type actually stored.
        found: &'static str,
        /// The type the accessor required.
        expected: &'static str,
    },
    /// A required metadata key is absent (Spec 2 §6; card A2.5).
    #[error("missing metadata key {key:?} (Spec 2 §6)")]
    MissingKey {
        /// The absent key.
        key: String,
    },
    /// A tensor-info `type` code names no known GGUF or R9V type
    /// (Spec 2 §7 step 1: unknown type → hard error naming the type;
    /// card A2.5).
    #[error("unknown tensor type code {code} for tensor {tensor:?} (Spec 2 §7)")]
    UnknownTensorType {
        /// The unrecognized numeric `type` code.
        code: u32,
        /// The tensor carrying it.
        tensor: String,
    },
    /// A tensor's data range is not inside the file or overlaps
    /// another entry (Spec 2 §6; card A2.5). All offending tensors
    /// are collected before returning.
    #[error("tensor {name:?} data range [{start}, {end}) is invalid: {reason} (Spec 2 §6)")]
    BadTensorRange {
        /// The tensor whose range failed.
        name: String,
        /// Range start as a file offset.
        start: u64,
        /// Range end as a file offset.
        end: u64,
        /// Why it is rejected (`outside file`, `overlaps ...`, ...).
        reason: String,
    },
    /// `r9v.format_version` is newer than this reader (Spec 2 §9:
    /// the loader accepts any `format_version ≤ current`; card A2.5).
    #[error("unsupported r9v.format_version {found}, newest accepted is {max} (Spec 2 §9)")]
    FormatVersion {
        /// The version found in the file.
        found: u32,
        /// The newest version this reader accepts.
        max: u32,
    },
    /// File-system read of a container path failed (card A2.5). The
    /// message carries the path and the OS error, never a panic.
    #[error("cannot read container file {path:?}: {message} (Spec 2 §6)")]
    Io {
        /// The path that was opened.
        path: String,
        /// The underlying OS error text.
        message: String,
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
        let mut iter = problems.into_iter();
        match (iter.next(), iter.next()) {
            (None, _) => Ok(()),
            (Some(single), None) => Err(single),
            (Some(first), Some(second)) => {
                let mut all = vec![first, second];
                all.extend(iter);
                Err(FormatError::Multiple {
                    problems: all.into_boxed_slice(),
                })
            }
        }
    }
}
