//! `r9v eval --logits`: CPU reference logits for a token file (Card A1.12).
//!
//! Reads a [`SyntheticSpec`] model file (JSON) and a whitespace-separated
//! token file, runs one prefill step on the CPU T0 device, and writes the
//! `[T, V]` f32 logits as a NumPy `.npy` file (v1.0, C order, `<f4`).
//!
//! The JSON model file is the interim A1.12 vehicle: the GGUF loader path
//! (cards A2.5–A2.6) does not exist yet, so `eval` cannot consume real
//! checkpoints. Recorded as SI-63; see module docs on the file format.

use std::fs;
use std::path::{Path, PathBuf};

use r9v_t0::decode::{prepare, run_step};
use r9v_t0::exec::CpuExecutor;
use r9v_t0::synthetic::{build, SyntheticSpec};

/// Runs `r9v eval --logits`, writing logits to `out` (default:
/// `<tokens>.logits.npy`) and printing the path and shape (Spec 14 §10, Card A1.12).
pub fn eval_logits(model: &Path, tokens: &Path, out: Option<&Path>) -> Result<(), String> {
    let spec = read_model(model)?;
    let ids = read_tokens(tokens, spec.vocab)?;
    if ids.is_empty() {
        return Err(format!(
            "r9v eval: token file {} holds no tokens",
            tokens.display()
        ));
    }
    if ids.len() as u64 > spec.max_ctx as u64 {
        return Err(format!(
            "r9v eval: {} tokens exceed model max_ctx {}",
            ids.len(),
            spec.max_ctx
        ));
    }
    let built = build(&spec).map_err(|error| format!("r9v eval: cannot build model: {error}"))?;
    let mut exec = CpuExecutor::new();
    let max_blocks =
        prepare(&mut exec, &built).map_err(|error| format!("r9v eval: cannot prepare: {error}"))?;
    let mut rng = Vec::new();
    let logits = run_step(&mut exec, &built, &ids, 0, max_blocks, &mut rng)
        .map_err(|error| format!("r9v eval: prefill failed: {error}"))?;
    let out_path: PathBuf = match out {
        Some(path) => path.to_path_buf(),
        None => tokens.with_extension("logits.npy"),
    };
    // DECISION(A1.12): default output replaces the tokens file extension
    // (`tokens.txt` -> `tokens.logits.npy`); rejected stdout because a
    // 2 MB float tensor is not terminal output. Spec 14 §10 is silent.
    write_npy_f32(&out_path, &[ids.len(), spec.vocab as usize], &logits)?;
    println!(
        "wrote {} shape [{}, {}]",
        out_path.display(),
        ids.len(),
        spec.vocab
    );
    Ok(())
}

/// Reads and validates the JSON [`SyntheticSpec`] model file.
fn read_model(path: &Path) -> Result<SyntheticSpec, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("r9v eval: cannot read {}: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("r9v eval: bad model file {}: {error}", path.display()))
}

/// Reads whitespace-separated token ids, checking the vocabulary bound.
fn read_tokens(path: &Path, vocab: u32) -> Result<Vec<u32>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("r9v eval: cannot read {}: {error}", path.display()))?;
    let mut ids = Vec::new();
    for piece in text.split_whitespace() {
        let id: u32 = piece.parse().map_err(|_| {
            format!(
                "r9v eval: token file {} holds non-u32 token {piece:?}",
                path.display()
            )
        })?;
        if id >= vocab {
            return Err(format!(
                "r9v eval: token {id} in {} is outside vocabulary 0..{vocab}",
                path.display()
            ));
        }
        ids.push(id);
    }
    Ok(ids)
}

/// Writes f32 row-major data as NumPy `.npy` v1.0 (`<f4`, C order) (Spec 14 §10, Card A1.12).
///
/// Hand-rolled (no new dependency): magic, version 1.0, dict header padded
/// to 64-byte alignment, then little-endian floats.
pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            format!(
                "r9v eval: cannot write {}: shape {shape:?} overflows usize",
                path.display()
            )
        })?;
    if data.len() != expected {
        return Err(format!(
            "r9v eval: cannot write {}: data length {} != shape {shape:?} ({expected})",
            path.display(),
            data.len()
        ));
    }
    let dims = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let trailing = if shape.len() == 1 { "," } else { "" };
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}{trailing}), }}");
    let mut header = dict.into_bytes();
    // Pad with spaces BEFORE the final newline so magic(6) + version(2) +
    // len(2) + header.len() is a multiple of 64, terminating with '\n'.
    // NumPy parsers evaluate the header with Python literal_eval and reject
    // whitespace after the newline.
    let unpadded_len = 10usize
        .checked_add(header.len())
        .and_then(|l| l.checked_add(1))
        .ok_or_else(|| {
            format!(
                "r9v eval: cannot write {}: header length overflows usize",
                path.display()
            )
        })?;
    let pad = (64 - (unpadded_len % 64)) % 64;
    header.extend(std::iter::repeat_n(b' ', pad));
    header.push(b'\n');

    let header_len = u16::try_from(header.len()).map_err(|_| {
        format!(
            "r9v eval: cannot write {}: header length {} exceeds u16::MAX for .npy v1.0",
            path.display(),
            header.len()
        )
    })?;
    let data_bytes_len = data
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            format!(
                "r9v eval: cannot write {}: data byte size overflows usize",
                path.display()
            )
        })?;
    let total_bytes = 10usize
        .checked_add(header.len())
        .and_then(|l| l.checked_add(data_bytes_len))
        .ok_or_else(|| {
            format!(
                "r9v eval: cannot write {}: total file size overflows usize",
                path.display()
            )
        })?;
    let mut bytes = Vec::with_capacity(total_bytes);
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, &bytes)
        .map_err(|error| format!("r9v eval: cannot write {}: {error}", path.display()))
}

/// Reads a `.npy` v1.0 `<f4` C-order file written by [`write_npy_f32`].
///
/// Test-only consumer (round-trip proof); fails closed on any deviation
/// from exactly what the writer emits.
#[cfg(test)]
pub fn read_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("r9v eval: cannot read {}: {error}", path.display()))?;
    let fail = |detail: &str| format!("r9v eval: bad npy {}: {detail}", path.display());
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(fail("missing magic"));
    }
    if bytes[6] != 1 || bytes[7] != 0 {
        return Err(fail("only v1.0 is supported"));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_end = 10usize
        .checked_add(header_len)
        .ok_or_else(|| fail("header length overflows usize"))?;
    if bytes.len() < header_end {
        return Err(fail("truncated header"));
    }
    let header = std::str::from_utf8(&bytes[10..header_end]).map_err(|_| fail("bad header"))?;
    if !header.contains("'<f4'") || !header.contains("False") {
        return Err(fail("only <f4 C-order is supported"));
    }
    let shape = parse_npy_shape(header).ok_or_else(|| fail("bad shape"))?;
    let count: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| fail("shape element count overflows usize"))?;
    let body = &bytes[header_end..];
    let expected_body_len = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| fail("body byte size overflows usize"))?;
    if body.len() != expected_body_len {
        return Err(fail("body length disagrees with shape"));
    }
    let mut values = Vec::with_capacity(count);
    for chunk in body.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, values))
}

/// Parses the `(d0, d1, ...)` shape tuple from a `.npy` header.
#[cfg(test)]
fn parse_npy_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape':")?;
    let open = header[start..].find('(')? + start;
    let close = header[open..].find(')')? + open;
    let inner = header[open + 1..close].trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(str::trim)
        .filter(|piece| !piece.is_empty())
        .map(|piece| piece.parse::<usize>().ok())
        .collect()
}
