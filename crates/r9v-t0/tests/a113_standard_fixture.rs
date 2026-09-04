// SPDX-License-Identifier: Apache-2.0
//! A1.13 fixture identity: the generated F16 checkpoint is a standard GGUF
//! whose 75 tensors are byte-identical to the Rust synthetic generation
//! (card A1.13 repair; spec 2 §6, spec 8 §8, spec 13 §12).
//!
//! [`GgufFile::parse`] must accept the file with no native-format
//! validation errors and [`GgufFile::is_standard_gguf`] must hold (no
//! `r9v.*` keys, no R9V type ids). Every tensor is then tied by
//! name, logical shape, dtype, and exact value bits to the corresponding
//! [`synthetic::build`] weight, so the "bit-identical weights" claim is a
//! test, not a docstring.
//!
//! Weight identity comes from production: [`TinyModel::weight_names`]
//! records each canonical synthetic name against its edge atomically inside
//! `add_weight`/`add_param`. This test asserts that edge/name association
//! first, then checks each named edge against its canonical llama tensor
//! name, shape, dtype, and exact GGUF bytes. It never zips the weight list
//! against a locally mirrored call order.
//!
//! The GGUF is generated at pytest time (gitignored) and located through
//! `R9V_A113_GGUF` / `R9V_A113_MODEL_JSON`. The test is `#[ignore]`d by
//! default so a plain `cargo test` reports it as ignored rather than a
//! false pass; the A1.13 acceptance pytest drives it with
//! `cargo test -- --ignored --exact <name>` and the required environment.
//!
//! DECISION(A1.13): fixture paths arrive via environment, not a checked-in
//! path; rejected committing the ~60 MB GGUF because generated fixtures
//! stay out of the tree per CONVENTIONS.md §4. Spec 2 §6 is silent on how
//! tests locate files.

use std::collections::{BTreeMap, BTreeSet};

use r9v_format::{GgufFile, TensorType};
use r9v_ir::DType;
use r9v_t0::synthetic::{build, SyntheticSpec, TinyModel};

/// Expected logical shape and norm flag for one canonical synthetic name.
///
/// Pure by-name lookup from the spec dims (no call-order table): norm
/// vectors are F32 `[dim]`, everything else is F16 with the shapes from
/// `synthetic.rs` build order.
fn expected_shape(synth: &str, spec: &SyntheticSpec) -> (Vec<usize>, bool) {
    let vocab = spec.vocab as usize;
    let dim = spec.dim as usize;
    let hd = spec.heads as usize * spec.head_dim as usize;
    let hkv = spec.kv_heads as usize * spec.head_dim as usize;
    let ff = spec.ff as usize;
    if synth == "embed" {
        return (vec![vocab, dim], false);
    }
    if synth == "lm_head" {
        return (vec![vocab, dim], false);
    }
    if synth == "final_norm" {
        return (vec![dim], true);
    }
    let (layer, rest) = synth
        .split_once('_')
        .unwrap_or_else(|| panic!("layer weight has a tag prefix: {synth:?}"));
    assert!(
        layer.starts_with('l') && layer.len() > 1 && layer[1..].parse::<u32>().is_ok(),
        "layer weight has a numeric layer tag: {synth:?}"
    );
    let shape = match rest {
        "attn_norm" => vec![dim],
        "wq" => vec![hd, dim],
        "wk" => vec![hkv, dim],
        "wv" => vec![hkv, dim],
        "wo" => vec![dim, hd],
        "ffn_norm" => vec![dim],
        "wg" => vec![ff, dim],
        "wu" => vec![ff, dim],
        "wd" => vec![dim, ff],
        _ => panic!("unknown synthetic weight {synth:?}"),
    };
    let is_norm = rest == "attn_norm" || rest == "ffn_norm";
    (shape, is_norm)
}

/// Full canonical synthetic name set for the spec dims (order-free).
fn expected_names(spec: &SyntheticSpec) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert("embed".to_owned());
    for layer in 0..spec.layers {
        let tag = format!("l{layer}");
        for suffix in [
            "attn_norm",
            "wq",
            "wk",
            "wv",
            "wo",
            "ffn_norm",
            "wg",
            "wu",
            "wd",
        ] {
            names.insert(format!("{tag}_{suffix}"));
        }
    }
    names.insert("final_norm".to_owned());
    names.insert("lm_head".to_owned());
    names
}

/// Synthetic name to llama tensor name (spec 8 §4 weight binding; mirrors
/// `llama_name` in `tools/r9v-quant/tests/gen_fixture.py`).
fn llama_name(synth: &str) -> String {
    if synth == "embed" {
        return "token_embd.weight".to_owned();
    }
    if synth == "lm_head" {
        return "output.weight".to_owned();
    }
    if synth == "final_norm" {
        return "output_norm.weight".to_owned();
    }
    let (layer, rest) = synth
        .split_once('_')
        .expect("layer weight has a tag prefix");
    let idx = &layer[1..];
    match rest {
        "attn_norm" => format!("blk.{idx}.attn_norm.weight"),
        "wq" => format!("blk.{idx}.attn_q.weight"),
        "wk" => format!("blk.{idx}.attn_k.weight"),
        "wv" => format!("blk.{idx}.attn_v.weight"),
        "wo" => format!("blk.{idx}.attn_output.weight"),
        "ffn_norm" => format!("blk.{idx}.ffn_norm.weight"),
        "wg" => format!("blk.{idx}.ffn_gate.weight"),
        "wu" => format!("blk.{idx}.ffn_up.weight"),
        "wd" => format!("blk.{idx}.ffn_down.weight"),
        _ => panic!("unknown synthetic weight {synth:?}"),
    }
}

/// Decodes raw little-endian F32 GGUF bytes to f32 bit patterns (exact).
fn gguf_f32_to_bits(bytes: &[u8], name: &str) -> Vec<u32> {
    assert!(
        bytes.len().is_multiple_of(4),
        "{name}: F32 byte length {} is not a multiple of 4",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).to_bits())
        .collect()
}

/// Production edge/name binding for one weight, resolved before any GGUF check.
fn production_bindings(model: &TinyModel) -> BTreeMap<String, (r9v_ir::EdgeId, usize)> {
    assert_eq!(
        model.weights.len(),
        model.weight_names.len(),
        "production weight/name binding length mismatch: {} weights vs {} names",
        model.weights.len(),
        model.weight_names.len()
    );
    let mut bindings = BTreeMap::new();
    for (index, ((weight_edge, _), (name_edge, name))) in model
        .weights
        .iter()
        .zip(model.weight_names.iter())
        .enumerate()
    {
        assert_eq!(
            weight_edge, name_edge,
            "production edge/name association broken at index {index}: \
             weight edge {weight_edge:?} != name edge {name_edge:?} ({name})"
        );
        assert!(
            bindings
                .insert(name.clone(), (*weight_edge, index))
                .is_none(),
            "production weight name {name:?} is bound twice"
        );
    }
    bindings
}

#[test]
#[ignore = "A1.13 fixture-driven: needs R9V_A113_GGUF/R9V_A113_MODEL_JSON from the pytest fixture generation; driven by tests/test_a113_torch_match.py via `cargo test -- --ignored --exact <name>`"]
fn a113_fixture_parses_as_standard_gguf_with_all_75_synthetic_tensors_byte_identical() {
    let gguf_path = std::env::var("R9V_A113_GGUF").unwrap_or_default();
    let model_path = std::env::var("R9V_A113_MODEL_JSON").unwrap_or_default();
    assert!(
        !gguf_path.is_empty() && !model_path.is_empty(),
        "R9V_A113_GGUF / R9V_A113_MODEL_JSON are unset; \
         run the A1.13 pytest to generate the fixture and drive this test"
    );
    let bytes = std::fs::read(&gguf_path).expect("reads the A1.13 fixture GGUF");
    let file = GgufFile::parse(&bytes).expect("fixture parses with no format errors");
    assert!(
        file.is_standard_gguf(),
        "fixture must be a standard GGUF (no r9v.* keys, no R9V type ids)"
    );
    assert!(
        !file.kvs().iter().any(|kv| kv.key.starts_with("r9v.")),
        "fixture carries reserved r9v.* metadata"
    );

    let model_text = std::fs::read_to_string(&model_path).expect("reads model.json");
    let spec: SyntheticSpec =
        serde_json::from_str(&model_text).expect("model.json is a SyntheticSpec");
    let model = build(&spec).expect("synthetic model builds from model.json");

    // Identity first, from production: every (edge, name) binding, with the
    // full canonical set covered exactly once.
    let bindings = production_bindings(&model);
    let want_names = expected_names(&spec);
    let got_names: BTreeSet<String> = bindings.keys().cloned().collect();
    assert_eq!(
        got_names, want_names,
        "production synthetic name set disagrees with the canonical set"
    );
    assert_eq!(
        model.weights.len(),
        3 + 9 * spec.layers as usize,
        "synthetic weight count disagrees with the spec layer count"
    );
    assert_eq!(
        file.tensors().len(),
        model.weights.len(),
        "GGUF tensor count disagrees with the synthetic weight count"
    );

    let mut problems = Vec::new();
    for (synth, _) in bindings.iter() {
        let (shape, is_norm) = expected_shape(synth, &spec);
        let name = llama_name(synth);
        let Some((_, index)) = bindings.get(synth) else {
            problems.push(format!("{name}: production binding missing for {synth:?}"));
            continue;
        };
        let (_, buffer) = &model.weights[*index];
        if buffer.shape() != shape.as_slice() {
            problems.push(format!(
                "{name}: rust shape {:?} != {shape:?} (synthetic {synth:?})",
                buffer.shape()
            ));
            continue;
        }
        let want_dtype = if is_norm { DType::F32 } else { DType::F16 };
        if buffer.dtype() != want_dtype {
            problems.push(format!(
                "{name}: rust dtype {:?} != {want_dtype:?} (synthetic {synth:?})",
                buffer.dtype()
            ));
            continue;
        }
        let Some(info) = file.tensor(&name) else {
            problems.push(format!("{name}: missing from the GGUF tensor table"));
            continue;
        };
        let want_shape: Vec<u64> = shape.iter().map(|d| *d as u64).collect();
        if info.shape() != want_shape {
            problems.push(format!(
                "{name}: gguf shape {:?} != {want_shape:?}",
                info.shape()
            ));
            continue;
        }
        let want_type = if is_norm {
            TensorType::F32
        } else {
            TensorType::F16
        };
        if info.dtype != want_type {
            problems.push(format!(
                "{name}: gguf type {:?} != {want_type:?}",
                info.dtype
            ));
            continue;
        }
        match file.tensor_bytes(&name, &bytes) {
            Ok(raw) => {
                let want_len = buffer.num_elements() * if is_norm { 4 } else { 2 };
                if raw.len() != want_len {
                    problems.push(format!(
                        "{name}: gguf byte length {} != {want_len}",
                        raw.len()
                    ));
                    continue;
                }
                // F16 weights are byte-backed LE half bytes on both sides:
                // direct byte equality. F32 norms compare by exact bits
                // (the conversion is bijective, so this is byte equality).
                let mismatch = if is_norm {
                    let actual = gguf_f32_to_bits(raw, &name);
                    let expected: Vec<u32> =
                        buffer.to_f32_vec().iter().map(|v| v.to_bits()).collect();
                    if actual == expected {
                        None
                    } else {
                        let bad = actual
                            .iter()
                            .zip(expected.iter())
                            .filter(|(a, e)| a != e)
                            .count();
                        Some(format!(
                            "{bad}/{} values differ from the synthetic stream",
                            actual.len()
                        ))
                    }
                } else if raw == buffer.byte_data() {
                    None
                } else {
                    Some(format!(
                        "gguf byte length {} != rust byte length {}",
                        raw.len(),
                        buffer.byte_data().len()
                    ))
                };
                if let Some(detail) = mismatch {
                    problems.push(format!("{name}: {detail}"));
                }
            }
            Err(error) => problems.push(format!("{name}: tensor bytes unreadable: {error}")),
        }
    }
    // Reverse coverage: every GGUF tensor must be one of the canonical llama
    // names bound above (no extras hiding outside the fingerprint).
    let bound_llama: BTreeSet<String> = bindings.keys().map(|s| llama_name(s)).collect();
    for info in file.tensors() {
        if !bound_llama.contains(&info.name) {
            problems.push(format!(
                "{}: GGUF tensor has no production synthetic binding",
                info.name
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "{} tensor fingerprint mismatch(es):\n{}",
        problems.len(),
        problems.join("\n")
    );
}
