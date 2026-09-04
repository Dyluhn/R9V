// SPDX-License-Identifier: Apache-2.0
//! Step 1 — metadata-only checkpoint open over a split shard set
//! (Spec 9 §2 step 1, §3).
//!
//! Reads only each shard's GGUF header, metadata KV table, and tensor-info
//! table, plus the Spec 9 §3 fingerprints. Tensor payload bytes are never
//! required: opening a file whose payload is truncated (with its true
//! logical size supplied separately) succeeds, which is how the tests prove
//! steps 1–4 never touch weight data.
//!
//! A checkpoint is one shard (the common case) or a GGUF split set merged
//! through [`r9v_format::ShardSet`]: shard `i` carries `split.no = i`,
//! `split.count = N`, `split.tensors.count = total`. A single path that
//! declares a split deterministically derives its sibling paths from the
//! `-NNNNN-of-MMMMM` file-name pattern and checks every sibling; an
//! explicit path set ([`open_shard_set`]) is merged in declared `split.no`
//! order. Metadata comes from shard 0 and tensor tables concatenate in
//! shard order (card A2.5); binding and fingerprints below cover the merged
//! set, never just the first shard.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use r9v_format::{parse_r9v_meta, GgufFile, KvValue, R9vMeta, ShardSet, TensorInfo};
use r9v_models::{GgufMeta, ModelsError};

use crate::error::LoaderError;

/// Exact fixed GGUF header length: magic `u32` + version `u32` +
/// tensor-count `u64` + metadata-count `u64` (Spec 2 §6).
///
/// The metadata-only open starts here, so the first read covers the header
/// alone and no payload byte is ever in the initial prefix.
const GGUF_FIXED_HEADER_LEN: u64 = 4 + 4 + 8 + 8;

/// Model fingerprint lifecycle (Spec 9 §3).
///
/// A native checkpoint whose merged tensor table carries a complete set of
/// per-tensor `r9v.tensor.*.xxh3` checksums reports [`ModelFingerprint::Ready`]
/// computed metadata-only. Anything else — standard GGUF, or a native file
/// with incomplete checksums — reports
/// [`ModelFingerprint::PendingUntilRepack`]: the per-tensor hashes are first
/// observed during the repack pass (card A2.8), so no value is fabricated
/// here and no payload byte is read to mint one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFingerprint {
    /// Complete `model_fp` over the merged set (Spec 9 §3).
    Ready(u128),
    /// Per-tensor hashes are not all present in metadata; `model_fp` is
    /// first computable during repack (Spec 9 §3, card A2.8).
    PendingUntilRepack,
}

impl ModelFingerprint {
    /// The fingerprint value, when ready.
    pub fn as_u128(self) -> Option<u128> {
        match self {
            ModelFingerprint::Ready(fp) => Some(fp),
            ModelFingerprint::PendingUntilRepack => None,
        }
    }
}

/// One opened shard: parsed tables plus its own fingerprint (Spec 9 §3).
#[derive(Debug)]
struct OpenedShard {
    /// Filesystem path this shard was opened from.
    path: String,
    /// Parsed header, metadata, and tensor-info tables (card A2.5).
    file: GgufFile,
    /// Spec 9 §3 per-shard file fingerprint.
    file_fp: u128,
    /// Typed native metadata, when the shard carries `r9v.*` keys.
    r9v_meta: Option<R9vMeta>,
    /// Distinct disk bytes read while opening this shard.
    bytes_read: u64,
}

/// Opened checkpoint: parsed shard set plus merged fingerprints
/// (Spec 9 §2 step 1).
///
/// Holds no tensor payload bytes: each [`GgufFile`] owns only the parsed
/// header, KV, and tensor-info tables.
#[derive(Debug)]
pub struct OpenedCheckpoint {
    /// Per-shard state in shard order.
    shards: Vec<OpenedShard>,
    /// Merged view over the set (card A2.5): metadata from shard 0,
    /// tensor tables concatenated in shard order.
    set: ShardSet,
    /// Merged file fingerprint over the set (Spec 9 §3; see
    /// [`merged_file_fp`]).
    merged_file_fp: u128,
    /// Model fingerprint lifecycle state (Spec 9 §3).
    model_fp: ModelFingerprint,
    /// Total distinct disk bytes read while opening all shards.
    bytes_read: u64,
}

impl OpenedCheckpoint {
    /// Path of shard 0 (the path [`open`] was given).
    pub fn path(&self) -> &str {
        &self.shards[0].path
    }

    /// Paths of every shard in shard order.
    pub fn shard_paths(&self) -> Vec<String> {
        self.shards.iter().map(|s| s.path.clone()).collect()
    }

    /// Number of shards in the set.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Parsed container tables of shard 0 (carries the set metadata).
    pub fn file(&self) -> &GgufFile {
        &self.shards[0].file
    }

    /// Parsed container tables of shard `index` (`None` when out of range).
    pub fn shard_file(&self, index: usize) -> Option<&GgufFile> {
        self.shards.get(index).map(|s| &s.file)
    }

    /// Typed native metadata of shard `index` (`None` for standard GGUF
    /// shards or when out of range).
    pub fn shard_r9v_meta(&self, index: usize) -> Option<&R9vMeta> {
        self.shards.get(index).and_then(|s| s.r9v_meta.as_ref())
    }

    /// Merged shard-set view: metadata from shard 0, tensor tables
    /// concatenated in shard order (card A2.5).
    pub fn shard_set(&self) -> &ShardSet {
        &self.set
    }

    /// Finds a tensor by name across shards: `(shard index, table row)`.
    pub fn tensor(&self, name: &str) -> Option<(usize, &TensorInfo)> {
        self.set.tensor(name)
    }

    /// Exact payload bytes of `info` from its owning shard's tables.
    pub fn tensor_nbytes(&self, shard_index: usize, info: &TensorInfo) -> Result<u64, LoaderError> {
        let shard = self
            .shards
            .get(shard_index)
            .ok_or_else(|| LoaderError::Validation {
                problems: vec![format!(
                    "tensor '{}' addresses out-of-range shard {shard_index} of {} (Spec 9 §2)",
                    info.name,
                    self.shards.len(),
                )],
            })?;
        shard.file.tensor_nbytes(info).map_err(LoaderError::Format)
    }

    /// Spec 9 §3 merged file fingerprint over the shard set.
    pub fn file_fp(&self) -> u128 {
        self.merged_file_fp
    }

    /// Spec 9 §3 model fingerprint lifecycle state.
    pub fn model_fp(&self) -> ModelFingerprint {
        self.model_fp
    }

    /// Total distinct bytes read from disk while opening (metadata prefixes
    /// only).
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }
}

/// Merged file fingerprint over a shard set (Spec 9 §3).
///
/// A single shard reports its own `file_fp` unchanged. For a split set,
/// `xxh3_128` over the per-shard `file_fp` values concatenated in shard
/// order. Each per-shard value already binds its header, tables, size, and
/// the declared shard count, so the merge binds every shard's tables
/// exactly once in a deterministic order.
// DECISION(A2.6): identity for one shard, concatenated per-shard
// fingerprints in shard order for splits; rejected re-hashing raw table
// bytes (duplicates per-shard work and reimplements the card A2.5 rule)
// and hashing only shard 0 (ignores the set). Spec 9 §3 defines `file_fp`
// per file and is silent on the merge; see SI-75. Spec 9 §3.
fn merged_file_fp(shard_fps: &[u128]) -> u128 {
    if shard_fps.len() == 1 {
        return shard_fps[0];
    }
    let mut input = Vec::with_capacity(16 * shard_fps.len());
    for fp in shard_fps {
        input.extend_from_slice(&fp.to_le_bytes());
    }
    r9v_common::xxh3_128(&input)
}

/// Opens `path` reading only metadata prefixes (Spec 9 §2 step 1).
///
/// The logical file size of each shard is its on-disk size. When the shard
/// declares `split.count = N > 1`, sibling paths are derived from the
/// `-NNNNN-of-MMMMM` file-name pattern and every sibling is opened and
/// checked; a missing sibling fails as [`LoaderError::MissingShard`] naming
/// the exact expected file. Tensor payload bytes are never required in
/// memory; reads stop once each header and table set parses.
pub fn open(path: &Path) -> Result<OpenedCheckpoint, LoaderError> {
    open_one_or_set(path, None)
}

/// Opens `path` as a metadata prefix of a logically `file_size`-byte first
/// shard (Spec 9 §2 step 1, Spec 9 §3).
///
/// Mirrors [`GgufFile::parse_metadata_only`]: tensor ranges validate against
/// `file_size`, so a payload-truncated prefix opens successfully when its
/// tables are intact. Production callers pass the on-disk size (see
/// [`open`]); tests pass the pre-truncation size to prove steps 1–4 never
/// touch payload data. Declared split siblings open at their on-disk sizes.
pub fn open_with_file_size(path: &Path, file_size: u64) -> Result<OpenedCheckpoint, LoaderError> {
    open_one_or_set(path, Some(file_size))
}

/// Opens an explicit shard path set in any order, merging in declared
/// `split.no` order (Spec 9 §2 step 1).
///
/// Every path opens metadata-only at its on-disk size. When all shards
/// declare `split.no`, the set is ordered by it, so input order never
/// affects the merged tables or fingerprints. [`ShardSet::open`] then
/// rejects duplicate tensors and inconsistent split declarations,
/// collecting every problem.
pub fn open_shard_set(paths: &[PathBuf]) -> Result<OpenedCheckpoint, LoaderError> {
    open_shard_set_with_file_sizes(paths, &vec![None; paths.len()])
}

/// Opens an explicit shard path set with per-shard logical sizes
/// (Spec 9 §2 step 1).
///
/// `sizes[i]` is the logical size of `paths[i]` (`None` selects the on-disk
/// size). Tests truncate shard payloads and pass pre-truncation sizes to
/// prove steps 1–4 never touch weight data. A length mismatch between
/// `paths` and `sizes` fails closed.
pub fn open_shard_set_with_file_sizes(
    paths: &[PathBuf],
    sizes: &[Option<u64>],
) -> Result<OpenedCheckpoint, LoaderError> {
    if paths.is_empty() {
        return Err(LoaderError::Validation {
            problems: vec!["shard path set is empty (Spec 9 §2)".to_string()],
        });
    }
    if paths.len() != sizes.len() {
        return Err(LoaderError::Validation {
            problems: vec![format!(
                "shard path/size count mismatch: {} paths vs {} sizes (Spec 9 §2)",
                paths.len(),
                sizes.len(),
            )],
        });
    }
    let mut shards = Vec::with_capacity(paths.len());
    for (path, size) in paths.iter().zip(sizes.iter()) {
        let logical = match size {
            Some(n) => *n,
            None => disk_len(path)?,
        };
        shards.push(open_one(path, logical)?);
    }
    // DECISION(A2.6): order an explicit set by declared `split.no` when
    // every shard declares it, so caller order never affects merged tables
    // or fingerprints; rejected sorting by path (unrelated to shard
    // identity) and keeping caller order (nondeterministic merge). shards
    // without declarations keep caller order and `ShardSet::open` enforces
    // declaration consistency. Spec 9 §2 is silent on set ordering.
    if shards.iter().all(|s| split_no(&s.file).is_some()) {
        shards.sort_by_key(|s| split_no(&s.file).unwrap_or(u32::MAX));
    }
    assemble(shards)
}

/// Single-path open with split discovery.
fn open_one_or_set(path: &Path, file_size: Option<u64>) -> Result<OpenedCheckpoint, LoaderError> {
    let logical = match file_size {
        Some(n) => n,
        None => disk_len(path)?,
    };
    let first = open_one(path, logical)?;
    let Some((_no, count)) = split_decl(&first.file) else {
        return assemble(vec![first]);
    };
    if count <= 1 {
        return assemble(vec![first]);
    }
    let siblings = sibling_paths(path, &first.file)?;
    let own = first_no(&first.file) as usize;
    // Internal invariant: derivation round-trips, so the derived own-shard
    // path is the path already opened above.
    if siblings.get(own) != Some(&path.to_path_buf()) {
        return Err(LoaderError::ShardPattern {
            path: path.display().to_string(),
            detail: "derived shard paths do not include the opened file".to_string(),
        });
    }
    let mut shards = Vec::with_capacity(siblings.len());
    let mut first = Some(first);
    for (index, sibling) in siblings.iter().enumerate() {
        if index == own {
            // Internal invariant: the derivation visits every index
            // exactly once, so the own shard is taken exactly once.
            shards.push(
                first
                    .take()
                    .expect("split derivation visits the own shard exactly once"),
            );
        } else {
            shards.push(open_sibling(index, count as usize, sibling)?);
        }
    }
    assemble(shards)
}

/// Opens one derived sibling, mapping a missing file to the typed error.
fn open_sibling(index: usize, count: usize, path: &Path) -> Result<OpenedShard, LoaderError> {
    if !path.exists() {
        return Err(LoaderError::MissingShard {
            shard_index: index as u32,
            shard_count: count as u32,
            expected_path: path.display().to_string(),
        });
    }
    let logical = disk_len(path)?;
    open_one(path, logical)
}

/// On-disk length of `path`.
fn disk_len(path: &Path) -> Result<u64, LoaderError> {
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| LoaderError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })
}

/// Opens one shard file as a metadata prefix of a logical `file_size`-byte
/// file, returning the shard plus the exact prefix bytes the successful
/// parse consumed (for the per-shard fingerprint).
///
/// Every read stays within the metadata tables: the first read covers only
/// the fixed header, and each later read extends the prefix to exactly the
/// `offset + need` the [`r9v_format::FormatError::Truncated`] response
/// demands. On a full untruncated file the final prefix is exactly the
/// tensor-info table end, so `bytes_read` never crosses into payload.
fn open_one(path: &Path, file_size: u64) -> Result<OpenedShard, LoaderError> {
    // DECISION(A2.6): start at the exact fixed GGUF header size and grow
    // only to the checked `offset + need` of each `Truncated` response;
    // rejected the former 32 KiB doubling prefix because a normal file
    // whose data section starts below 32 KiB would page payload bytes, and
    // rejected a single full-file read because steps 1-4 must not page
    // payload at all. Only `Truncated` grows the prefix; any other error
    // returns immediately so malformed inputs fail fast. Spec 9 §2 steps
    // 1-4.
    let path_str = path.display().to_string();
    let mut disk = File::open(path).map_err(|e| LoaderError::Io {
        path: path_str.clone(),
        message: e.to_string(),
    })?;
    let disk_len = disk
        .metadata()
        .map(|m| m.len())
        .map_err(|e| LoaderError::Io {
            path: path_str.clone(),
            message: e.to_string(),
        })?;

    let io_err = |e: std::io::Error| LoaderError::Io {
        path: path_str.clone(),
        message: e.to_string(),
    };
    // The first read covers the header alone (or the whole file when the
    // file itself is shorter than the header); every later read appends
    // exactly the demanded tail, so each byte is read once and no read
    // ranges past the final table end.
    let mut buf = vec![0u8; GGUF_FIXED_HEADER_LEN.min(disk_len) as usize];
    if !buf.is_empty() {
        disk.read_exact(&mut buf).map_err(io_err)?;
    }
    loop {
        match GgufFile::parse_metadata_only(&buf, file_size) {
            Ok(file) => {
                let bytes_read = buf.len() as u64;
                let fp = file_fp_of(&file, &buf)?;
                return Ok(OpenedShard {
                    path: path_str,
                    file,
                    file_fp: fp,
                    r9v_meta: None,
                    bytes_read,
                });
            }
            Err(r9v_format::FormatError::Truncated { offset, need, what })
                if (buf.len() as u64) < disk_len =>
            {
                let next = offset
                    .checked_add(need)
                    .ok_or_else(|| LoaderError::Overflow {
                        what: "open prefix end".to_string(),
                        detail: format!("truncated {what} at offset {offset} needs {need} bytes"),
                    })?;
                if next <= buf.len() as u64 {
                    return Err(LoaderError::Format(r9v_format::FormatError::Truncated {
                        offset,
                        need,
                        what,
                    }));
                }
                if next > disk_len {
                    return Err(LoaderError::Format(r9v_format::FormatError::Truncated {
                        offset,
                        need,
                        what,
                    }));
                }
                let next_len = usize::try_from(next).map_err(|_| LoaderError::Overflow {
                    what: "open prefix end".to_string(),
                    detail: format!("prefix length {next} does not fit this platform"),
                })?;
                // Fallible reservation first: a hostile demand that fits
                // `disk_len` (e.g. a sparse file's enormous metadata field)
                // must fail typed here, before any byte is read, rather
                // than aborting inside infallible allocation. The resize
                // after a successful reservation cannot allocate.
                let tail = reserve_prefix_growth(&mut buf, next_len)?;
                buf.resize(next_len, 0);
                disk.read_exact(&mut buf[next_len - tail..])
                    .map_err(io_err)?;
            }
            Err(e) => return Err(LoaderError::Format(e)),
        }
    }
}

/// Fallibly reserves the checked growth of the metadata prefix from its
/// current length to `next_len` (Spec 9 §2 step 1).
///
/// `Vec::try_reserve_exact` takes `additional` beyond `len`, not the total,
/// so the helper derives `additional = next_len - len` checked and reserves
/// exactly that before any byte is read. Allocator refusal maps to the typed
/// [`LoaderError::PrefixAlloc`] carrying the demanded and current lengths,
/// leaving `buf` (length and prior bytes) unchanged, so the open fails
/// closed with no partial read. After success the following `resize` cannot
/// allocate. A stale demand (`next_len <= len`) fails as
/// [`LoaderError::Overflow`]; the caller already rejects it as a stale
/// `Truncated` before reaching here.
///
/// Returns `additional` for the caller to slice the tail read.
///
/// Buffer-allocation audit for the open loop: the initial prefix is the fixed
/// 24-byte header (`GGUF_FIXED_HEADER_LEN`, never metadata-sized); this
/// growth is the only metadata-driven (`Truncated` `offset + need`)
/// allocation in the loop; the metadata parser pushes row by row behind the
/// cursor and reports `Truncated` when bytes run out instead of
/// pre-allocating an untrusted count (card A2.5). Outside the loop,
/// capacities are caller- or parse-bounded: shard counts come from
/// `split.count` (`u16`, at most 65_535 entries) and tensor/KV counts are
/// bounded by the bytes already parsed.
// DECISION(A2.6): fallible `try_reserve_exact(additional)` mapped to a typed
// `PrefixAlloc` refusal with no byte read; rejected infallible `resize`
// (aborts the process on a hostile demand that fits `disk_len`) and rejected
// an arbitrary metadata ceiling (no spec or config defines one, so any
// honored demand still opens). Spec 9 §2 steps 1-4.
fn reserve_prefix_growth(buf: &mut Vec<u8>, next_len: usize) -> Result<usize, LoaderError> {
    let additional = next_len
        .checked_sub(buf.len())
        .filter(|&a| a > 0)
        .ok_or_else(|| LoaderError::Overflow {
            what: "open prefix growth".to_string(),
            detail: format!(
                "demanded prefix {next_len} does not extend current prefix {}",
                buf.len(),
            ),
        })?;
    buf.try_reserve_exact(additional)
        .map_err(|e| LoaderError::PrefixAlloc {
            required: next_len as u64,
            current: buf.len() as u64,
            detail: e.to_string(),
        })?;
    Ok(additional)
}

/// Assembles opened shards into the merged checkpoint: split cross-checks,
/// per-shard native metadata, merged fingerprints.
fn assemble(mut shards: Vec<OpenedShard>) -> Result<OpenedCheckpoint, LoaderError> {
    let files: Vec<GgufFile> = shards.iter().map(|s| s.file.clone()).collect();
    let set = ShardSet::open(files).map_err(LoaderError::Format)?;

    let mut bytes_read: u64 = 0;
    let mut fps = Vec::with_capacity(shards.len());
    for shard in &mut shards {
        bytes_read =
            bytes_read
                .checked_add(shard.bytes_read)
                .ok_or_else(|| LoaderError::Overflow {
                    what: "open bytes_read total".to_string(),
                    detail: format!("{} + {}", bytes_read, shard.bytes_read),
                })?;
        // Native metadata parses (and fails) here, at the boundary, so
        // malformed `r9v.*` keys fail the open rather than surfacing
        // mid-pipeline; standard GGUF shards skip the parse entirely.
        if shard.file.is_native() {
            shard.r9v_meta = parse_r9v_meta(&shard.file).map_err(LoaderError::Format)?;
        }
        fps.push(shard.file_fp);
    }
    let merged_file_fp = merged_file_fp(&fps);
    let model_fp = model_fp_of(&set, &shards, merged_file_fp);
    Ok(OpenedCheckpoint {
        shards,
        set,
        merged_file_fp,
        model_fp,
        bytes_read,
    })
}

/// Spec 9 §3 `file_fp` of one shard over the bytes the open consumed.
fn file_fp_of(file: &GgufFile, bytes: &[u8]) -> Result<u128, LoaderError> {
    // DECISION(A2.6): shard count comes from `split.count` when the file
    // declares a split shard, else 1; rejected hard-coding 1 because Spec 9
    // §3 fingerprints the shard count. Container validation already rejects
    // a mistyped `split.count`, so any other variant here is unreachable for
    // a parsed file and falls back to 1. Spec 9 §3.
    let shards = match file.kv("split.count") {
        Some(KvValue::U16(n)) => u64::from(*n),
        _ => 1,
    };
    file.file_fp(bytes, shards).map_err(LoaderError::Format)
}

/// Spec 9 §3 `model_fp` lifecycle over the merged set.
///
/// Complete per-tensor checksums in merged table order hash metadata-only
/// into [`ModelFingerprint::Ready`]; any gap — standard GGUF, or a native
/// shard with missing entries — yields
/// [`ModelFingerprint::PendingUntilRepack`] without reading payload.
fn model_fp_of(set: &ShardSet, shards: &[OpenedShard], merged_file_fp: u128) -> ModelFingerprint {
    let mut hashes = Vec::with_capacity(set.len());
    for i in 0..set.len() {
        let Some((shard_index, info)) = set.tensor_at(i) else {
            return ModelFingerprint::PendingUntilRepack;
        };
        let hash = shards
            .get(shard_index)
            .and_then(|s| s.r9v_meta.as_ref())
            .and_then(|m| m.tensor(&info.name))
            .and_then(|t| t.xxh3);
        match hash {
            Some(h) => hashes.push(h),
            None => return ModelFingerprint::PendingUntilRepack,
        }
    }
    ModelFingerprint::Ready(r9v_format::model_fp(merged_file_fp, &hashes))
}

/// Declared `(split.no, split.count)` of a parsed shard (`None` when the
/// shard carries no split declaration).
fn split_decl(file: &GgufFile) -> Option<(u32, u32)> {
    let no = split_no(file)?;
    let count = match file.kv("split.count") {
        Some(KvValue::U16(n)) => u32::from(*n),
        _ => return None,
    };
    Some((no, count))
}

/// Declared `split.no` of a parsed shard.
fn split_no(file: &GgufFile) -> Option<u32> {
    match file.kv("split.no") {
        Some(KvValue::U16(n)) => Some(u32::from(*n)),
        _ => None,
    }
}

/// Declared `split.no` of a parsed shard file (0 when undeclared).
fn first_no(file: &GgufFile) -> u32 {
    split_no(file).unwrap_or(0)
}

/// Derives every sibling path `1..=count` from the opened shard's file
/// name, verifying the `-NNNNN-of-MMMMM` pattern against the declaration.
///
/// gguf split files are named `{base}-{i:05}-of-{N:05}.gguf` with 1-based
/// `i`; the width is whatever the opened file uses, preserved for every
/// sibling so derivation round-trips exactly.
fn sibling_paths(path: &Path, file: &GgufFile) -> Result<Vec<PathBuf>, LoaderError> {
    let err = |detail: String| LoaderError::ShardPattern {
        path: path.display().to_string(),
        detail,
    };
    let (no, count) = split_decl(file).ok_or_else(|| {
        err("shard declares a split set but carries no split.no/split.count".to_string())
    })?;
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| err("file name is not valid UTF-8".to_string()))?;
    // Split stem from extension: `model-00001-of-00002.gguf` ->
    // stem `model-00001-of-00002`, extension `.gguf`.
    let (stem, ext) = match file_name.rfind('.') {
        Some(dot) => (&file_name[..dot], &file_name[dot..]),
        None => (file_name, ""),
    };
    let (prefix, own_field, total_field) = match stem.rfind("-of-") {
        Some(of) => {
            let (head, total) = (&stem[..of], &stem[of + 4..]);
            match head.rfind('-') {
                Some(dash) => (&head[..dash + 1], &head[dash + 1..], total),
                None => {
                    return Err(err(format!(
                        "file name '{file_name}' has no '-NNNNN-of-MMMMM' shard pattern"
                    )));
                }
            }
        }
        None => {
            return Err(err(format!(
                "file name '{file_name}' has no '-NNNNN-of-MMMMM' shard pattern"
            )));
        }
    };
    if !own_field.bytes().all(|b| b.is_ascii_digit())
        || !total_field.bytes().all(|b| b.is_ascii_digit())
        || own_field.is_empty()
        || total_field.is_empty()
    {
        return Err(err(format!(
            "file name '{file_name}' has no '-NNNNN-of-MMMMM' shard pattern"
        )));
    }
    let width = own_field.len();
    let own: u32 = own_field
        .parse()
        .map_err(|_| err(format!("shard number '{own_field}' does not parse")))?;
    let total: u32 = total_field
        .parse()
        .map_err(|_| err(format!("shard total '{total_field}' does not parse")))?;
    if total != count {
        return Err(err(format!(
            "file name declares {total} shards but split.count is {count}"
        )));
    }
    if own != no + 1 {
        return Err(err(format!(
            "file name declares shard {own} of {total} but split.no is {no}"
        )));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let mut out = Vec::with_capacity(count as usize);
    for i in 1..=count {
        // Width-preserving zero pad; numbers wider than the observed
        // width extend rather than truncate (format width is a minimum).
        let name = format!("{prefix}{i:0>width$}-of-{total_field}{ext}");
        out.push(parent.join(name));
    }
    Ok(out)
}

/// Typed metadata view over a parsed container (Spec 8 §4, §10).
///
/// Lets model definitions consume checkpoint metadata through the
/// [`GgufMeta`] trait without `r9v-models` depending on the container crate
/// (card A1.3). Built over shard 0, which carries the set metadata.
#[derive(Debug)]
pub struct GgufFileMeta<'a> {
    /// Parsed container backing the lookup.
    file: &'a GgufFile,
}

impl<'a> GgufFileMeta<'a> {
    /// Borrows a parsed container as typed metadata.
    pub fn new(file: &'a GgufFile) -> Self {
        Self { file }
    }

    /// Raw value lookup.
    fn lookup(&self, key: &str, expected_type: &'static str) -> Result<&KvValue, ModelsError> {
        self.file
            .kv(key)
            .ok_or_else(|| ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type,
            })
    }

    /// Type-mismatch error for `key`.
    fn mismatch(key: &str, expected: &'static str, found: &KvValue) -> ModelsError {
        ModelsError::MetaTypeMismatch {
            key: key.to_string(),
            expected,
            found: found.kv_type().name().to_string(),
        }
    }
}

impl GgufMeta for GgufFileMeta<'_> {
    fn has(&self, key: &str) -> bool {
        self.file.kv(key).is_some()
    }

    fn str(&self, key: &str) -> Result<&str, ModelsError> {
        match self.lookup(key, "string")? {
            KvValue::Str(s) => Ok(s.as_str()),
            found => Err(Self::mismatch(key, "string", found)),
        }
    }

    fn u32(&self, key: &str) -> Result<u32, ModelsError> {
        // DECISION(A2.6): accept U8/U16/U32 exactly and U64 with a checked
        // narrowing conversion; rejected exact-width-only because the Spec 8
        // §4 key table lists `u32 / u64` for several counts and gguf-py
        // varies widths. Overflow fails closed as a type mismatch. Spec 8 §4.
        let found = self.lookup(key, "u32")?;
        match found {
            KvValue::U8(v) => Ok(u32::from(*v)),
            KvValue::U16(v) => Ok(u32::from(*v)),
            KvValue::U32(v) => Ok(*v),
            KvValue::U64(v) => u32::try_from(*v).map_err(|_| Self::mismatch(key, "u32", found)),
            _ => Err(Self::mismatch(key, "u32", found)),
        }
    }

    fn u64(&self, key: &str) -> Result<u64, ModelsError> {
        match self.lookup(key, "u64")? {
            KvValue::U8(v) => Ok(u64::from(*v)),
            KvValue::U16(v) => Ok(u64::from(*v)),
            KvValue::U32(v) => Ok(u64::from(*v)),
            KvValue::U64(v) => Ok(*v),
            found => Err(Self::mismatch(key, "u64", found)),
        }
    }

    fn i32(&self, key: &str) -> Result<i32, ModelsError> {
        let found = self.lookup(key, "i32")?;
        match found {
            KvValue::I8(v) => Ok(i32::from(*v)),
            KvValue::I16(v) => Ok(i32::from(*v)),
            KvValue::I32(v) => Ok(*v),
            KvValue::I64(v) => i32::try_from(*v).map_err(|_| Self::mismatch(key, "i32", found)),
            _ => Err(Self::mismatch(key, "i32", found)),
        }
    }

    fn f32(&self, key: &str) -> Result<f32, ModelsError> {
        match self.lookup(key, "f32")? {
            KvValue::F32(v) => Ok(*v),
            found => Err(Self::mismatch(key, "f32", found)),
        }
    }

    fn bool(&self, key: &str) -> Result<bool, ModelsError> {
        match self.lookup(key, "bool")? {
            KvValue::Bool(v) => Ok(*v),
            found => Err(Self::mismatch(key, "bool", found)),
        }
    }

    fn str_array(&self, key: &str) -> Result<Vec<String>, ModelsError> {
        match self.lookup(key, "array of string")? {
            KvValue::Array { items, .. } => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        KvValue::Str(s) => out.push(s.clone()),
                        found => return Err(Self::mismatch(key, "array of string", found)),
                    }
                }
                Ok(out)
            }
            found => Err(Self::mismatch(key, "array of string", found)),
        }
    }

    fn u32_array(&self, key: &str) -> Result<Vec<u32>, ModelsError> {
        match self.lookup(key, "array of u32")? {
            KvValue::Array { items, .. } => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        KvValue::U8(v) => out.push(u32::from(*v)),
                        KvValue::U16(v) => out.push(u32::from(*v)),
                        KvValue::U32(v) => out.push(*v),
                        found => return Err(Self::mismatch(key, "array of u32", found)),
                    }
                }
                Ok(out)
            }
            found => Err(Self::mismatch(key, "array of u32", found)),
        }
    }

    fn bool_array(&self, key: &str) -> Result<Vec<bool>, ModelsError> {
        match self.lookup(key, "array of bool")? {
            KvValue::Array { items, .. } => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        KvValue::Bool(v) => out.push(*v),
                        found => return Err(Self::mismatch(key, "array of bool", found)),
                    }
                }
                Ok(out)
            }
            found => Err(Self::mismatch(key, "array of bool", found)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_growth_reserves_exact_additional_beyond_len() {
        let mut buf = vec![0u8; 24];
        let additional = reserve_prefix_growth(&mut buf, 100).expect("small growth reserves");
        // `additional` is relative to `len` (76), never the total (100):
        // passing the total would over-reserve the prefix.
        assert_eq!(additional, 76);
        assert_eq!(buf.len(), 24);
        assert!(buf.capacity() >= 100);
    }

    #[test]
    fn prefix_growth_stale_demand_fails_overflow_unchanged() {
        let mut buf = vec![0u8; 100];
        let (len_before, cap_before) = (buf.len(), buf.capacity());
        assert!(
            matches!(
                reserve_prefix_growth(&mut buf, 100),
                Err(LoaderError::Overflow { .. })
            ),
            "equal demand must fail typed"
        );
        assert!(
            matches!(
                reserve_prefix_growth(&mut buf, 50),
                Err(LoaderError::Overflow { .. })
            ),
            "shorter demand must fail typed"
        );
        assert_eq!(buf.len(), len_before);
        assert_eq!(buf.capacity(), cap_before);
    }

    #[test]
    fn prefix_growth_huge_demand_fails_prefix_alloc_without_huge_alloc() {
        let mut buf = vec![0u8; 24];
        // `usize::MAX` exceeds `isize::MAX`, so the allocator refuses with
        // `CapacityOverflow` before touching memory: no huge allocation,
        // deterministic on every platform and tier.
        let err = reserve_prefix_growth(&mut buf, usize::MAX).expect_err("must refuse");
        let LoaderError::PrefixAlloc {
            required, current, ..
        } = err
        else {
            panic!("expected PrefixAlloc, got {err:?}");
        };
        assert_eq!(required, usize::MAX as u64);
        assert_eq!(current, 24);
        assert_eq!(buf.len(), 24);
    }
}
