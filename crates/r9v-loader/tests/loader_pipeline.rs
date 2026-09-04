// SPDX-License-Identifier: Apache-2.0
//! Steps 1–4 integration tests (card A2.6; Spec 9 §2 steps 1–4, §4, §12).
//!
//! Fixtures are synthetic F32 llama checkpoints written with `GgufWriter`
//! from the model's own bound weights, so tensor names and shapes always
//! match the definition under test. Scratch files live in the process temp
//! dir under unique names and are removed at the end of each test.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use r9v_format::{FormatError, GgufFile, GgufWriter, KvValue, TensorType, GGUF_MAGIC};
use r9v_ir::{ActivationKind, IrVersion, MoeScoring, PlanStrategy};
use r9v_loader::{
    arena_layout, bind, check_device_budget, check_fusion_decls, downgrade_absent_mtp,
    is_stacked_expert_weight, open, plan_single_device, prepare, prepare_shard_set,
    prepare_with_file_size, BudgetScope, DeviceBudgetInput, LoaderError, ModelFingerprint,
    PlannedDevice, PrepareOptions, TensorProblemKind, DEFAULT_CHUNK_BYTES, DEFAULT_QUEUE_DEPTH,
    DEFAULT_RESERVE_BYTES, TENSOR_ALIGN_BYTES,
};
use r9v_models::{
    build_from_meta, build_model, Ffn, Graph, ModelGraph, MtpSource, MtpSpec, SyntheticGgufMeta,
};

static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

fn scratch_path(tag: &str) -> PathBuf {
    let n = SCRATCH_SEQ.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!("r9v-a26-{tag}-{}-{n}.gguf", std::process::id()))
}

fn remove_quiet(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
}

/// Minimal llama metadata satisfying `families::build` (Spec 8 §4).
fn tiny_meta() -> SyntheticGgufMeta {
    let mut meta = SyntheticGgufMeta::new();
    meta.insert_str("general.architecture", "llama");
    meta.insert_u32("llama.block_count", 2);
    meta.insert_u32("llama.embedding_length", 32);
    meta.insert_u32("llama.feed_forward_length", 64);
    meta.insert_u32("llama.attention.head_count", 2);
    meta.insert_u32("llama.attention.head_count_kv", 2);
    meta.insert_f32("llama.attention.layer_norm_rms_epsilon", 0.00001);
    meta.insert_u32("llama.vocab_size", 64);
    meta
}

/// Writer KVs mirroring [`tiny_meta`] in container types.
fn tiny_kvs(extra: Vec<(String, KvValue)>) -> Vec<(String, KvValue)> {
    let mut kvs = vec![
        (
            "general.architecture".to_string(),
            KvValue::Str("llama".to_string()),
        ),
        ("llama.block_count".to_string(), KvValue::U32(2)),
        ("llama.embedding_length".to_string(), KvValue::U32(32)),
        ("llama.feed_forward_length".to_string(), KvValue::U32(64)),
        ("llama.attention.head_count".to_string(), KvValue::U32(2)),
        ("llama.attention.head_count_kv".to_string(), KvValue::U32(2)),
        (
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            KvValue::F32(0.00001),
        ),
        ("llama.vocab_size".to_string(), KvValue::U32(64)),
    ];
    kvs.extend(extra);
    kvs
}

/// Bound (name, outer-last shape) pairs from the model definition itself.
fn tiny_bound_tensors() -> Vec<(String, Vec<u64>)> {
    let meta = tiny_meta();
    let spec = build_from_meta(&meta).expect("tiny llama spec builds");
    let graph = build_model(Graph::new(IrVersion::CURRENT, "tiny"), &spec)
        .expect("tiny llama graph builds");
    graph
        .bound_weights()
        .iter()
        .map(|w| {
            let shape = w
                .shape
                .iter()
                .map(|d| match d {
                    r9v_ir::Dim::Concrete(n) => u64::from(*n),
                    r9v_ir::Dim::Symbolic(s) => panic!("unexpected symbolic dim {s:?}"),
                })
                .collect();
            (w.name.clone(), shape)
        })
        .collect()
}

fn zeros_for(shape: &[u64], dtype: TensorType) -> Vec<u8> {
    let elems: u64 = shape.iter().product();
    let bpv: u64 = match dtype {
        TensorType::F32 => 4,
        _ => panic!("fixture helper only sizes F32, got {dtype:?}"),
    };
    vec![0u8; (elems * bpv) as usize]
}

/// Emits a checkpoint file; returns its full bytes.
fn emit_checkpoint(
    path: &PathBuf,
    kvs: Vec<(String, KvValue)>,
    tensors: &[(String, Vec<u64>, TensorType)],
) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    for (key, value) in kvs {
        writer.add_kv(&key, value).expect("kv inserts");
    }
    for (name, shape, dtype) in tensors {
        let data = zeros_for(shape, *dtype);
        writer
            .add_tensor(name, shape, *dtype, data)
            .expect("tensor inserts");
    }
    let bytes = writer.emit().expect("emit succeeds");
    std::fs::write(path, &bytes).expect("scratch write");
    bytes
}

/// Full tiny checkpoint: every bound tensor as F32.
fn write_tiny(path: &PathBuf) -> Vec<u8> {
    let tensors: Vec<(String, Vec<u64>, TensorType)> = tiny_bound_tensors()
        .into_iter()
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    emit_checkpoint(path, tiny_kvs(Vec::new()), &tensors)
}

fn test_options(host_pinned: u64, devices: Vec<PlannedDevice>) -> PrepareOptions {
    PrepareOptions {
        devices,
        max_ctx: 64,
        max_seqs: 1,
        reserve_bytes: 1 << 20,
        workspace_bytes: 1 << 20,
        host_pinned_bytes: host_pinned,
        chunk_bytes: DEFAULT_CHUNK_BYTES,
        queue_depth: DEFAULT_QUEUE_DEPTH,
        slab_bytes: 0,
        per_step_bytes: 0,
    }
}

fn single_gpu(vram: u64) -> Vec<PlannedDevice> {
    vec![PlannedDevice {
        rank: 0,
        vram_bytes: vram,
    }]
}

const GIB: u64 = 1 << 30;

#[test]
fn steps_1_to_4_succeed_metadata_only() {
    let path = scratch_path("happy");
    let full = write_tiny(&path);

    let prepared = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect("prepare succeeds");

    // Every bound tensor resolved; plan is Single on rank 0.
    assert!(!prepared.bind.bound.is_empty());
    assert!(prepared.bind.unused.is_empty());
    assert_eq!(prepared.plan.strategy, PlanStrategy::Single);
    assert_eq!(prepared.plan.tp_degree, 1);
    // Fingerprint matches the container oracle over the same bytes.
    let file = GgufFile::parse(&full).expect("oracle parse");
    assert_eq!(prepared.file_fp, file.file_fp(&full, 1).expect("oracle fp"));
    // Single shard: one path, and a standard GGUF never fabricates a model
    // fingerprint.
    assert_eq!(prepared.shard_paths.len(), 1);
    assert_eq!(prepared.model_fp, ModelFingerprint::PendingUntilRepack);
    // Only the metadata prefix was read from disk.
    assert!(prepared.bytes_read < full.len() as u64);
    assert!(prepared.device_budget.is_some());
    remove_quiet(&path);
}

#[test]
fn truncated_payload_file_completes_steps_1_to_4() {
    let path = scratch_path("trunc");
    let full = write_tiny(&path);
    let file = GgufFile::parse(&full).expect("oracle parse");
    let data_start = file.data_start();

    // Deliberately destroy the payload: the file ends where it starts.
    let prefix = full[..data_start as usize].to_vec();
    std::fs::write(&path, &prefix).expect("truncate write");

    let prepared = prepare_with_file_size(
        &path,
        full.len() as u64,
        &test_options(GIB, single_gpu(GIB)),
    )
    .expect("steps 1-4 succeed without payload bytes");
    assert_eq!(prepared.bind.bound.len(), tiny_bound_tensors().len());
    // No read covered a payload offset.
    assert!(prepared.bytes_read <= data_start);

    // The same truncated file with its on-disk size fails closed: tensor
    // ranges genuinely exceed the file, and that must still refuse.
    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    assert!(matches!(err, LoaderError::Format(_)), "unexpected: {err:?}");
    remove_quiet(&path);
}

#[test]
fn open_full_file_never_reads_payload() {
    let path = scratch_path("nopayload");
    let full = write_tiny(&path);
    let file = GgufFile::parse(&full).expect("oracle parse");
    let data_start = file.data_start();
    let table_end = file.ti_range().1;
    // The fixture's data section starts below the old fixed prefix, so this
    // test would fail against a reader that pages a flat 32 KiB up front.
    assert!(
        data_start < 32 * 1024,
        "fixture needs data_start {data_start} < 32 KiB"
    );
    assert!(table_end <= data_start);

    // The full untruncated file opens at its on-disk size.
    let prepared = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect("prepare succeeds");
    // No read crossed into payload: distinct disk coverage ends at the
    // tensor-info table end, at or before the data section start.
    assert_eq!(prepared.bytes_read, table_end);
    assert!(prepared.bytes_read <= data_start);
    assert!(prepared.bytes_read < full.len() as u64);
    // The fingerprint over the same prefix matches the container oracle.
    assert_eq!(prepared.file_fp, file.file_fp(&full, 1).expect("oracle fp"));
    remove_quiet(&path);
}

#[test]
fn all_missing_tensors_reported_together() {
    let path = scratch_path("missing");
    let all = tiny_bound_tensors();
    let dropped: Vec<String> = all
        .iter()
        .filter(|(name, _)| name.starts_with("blk.1.") || name == "output.weight")
        .map(|(name, _)| name.clone())
        .collect();
    assert!(dropped.len() >= 2, "fixture needs several dropped tensors");
    let kept: Vec<(String, Vec<u64>, TensorType)> = all
        .into_iter()
        .filter(|(name, _)| !dropped.contains(name))
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    emit_checkpoint(&path, tiny_kvs(Vec::new()), &kept);

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::Tensors { details } = err else {
        panic!("expected Tensors, got {err:?}");
    };
    // Every missing name reported, nothing else.
    assert_eq!(details.len(), dropped.len(), "details: {details:?}");
    for name in &dropped {
        assert!(
            details
                .iter()
                .any(|d| &d.name == name && matches!(d.kind, TensorProblemKind::Missing)),
            "missing report for {name}; details: {details:?}",
        );
    }
    remove_quiet(&path);
}

#[test]
fn missing_and_misshaped_tensors_collected_together() {
    let path = scratch_path("mixed");
    let all = tiny_bound_tensors();
    let drop_name = all
        .iter()
        .find(|(name, _)| name.starts_with("blk.1."))
        .map(|(name, _)| name.clone())
        .expect("blk.1 tensor exists");
    let skew_name = "token_embd.weight".to_string();
    let skew_shape = all
        .iter()
        .find(|(name, _)| name == &skew_name)
        .map(|(_, shape)| vec![shape[1], shape[0]])
        .expect("embed tensor exists");
    let kept: Vec<(String, Vec<u64>, TensorType)> = all
        .into_iter()
        .filter(|(name, _)| name != &drop_name)
        .map(|(name, shape)| {
            let shape = if name == skew_name {
                skew_shape.clone()
            } else {
                shape
            };
            (name, shape, TensorType::F32)
        })
        .collect();
    emit_checkpoint(&path, tiny_kvs(Vec::new()), &kept);

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::Tensors { details } = err else {
        panic!("expected Tensors, got {err:?}");
    };
    assert_eq!(details.len(), 2, "details: {details:?}");
    assert!(details
        .iter()
        .any(|d| d.name == drop_name && matches!(d.kind, TensorProblemKind::Missing)));
    let skew = details
        .iter()
        .find(|d| d.name == skew_name)
        .expect("skew report");
    let TensorProblemKind::ShapeMismatch { expected, actual } = &skew.kind else {
        panic!("expected ShapeMismatch, got {:?}", skew.kind);
    };
    assert_eq!(expected, &vec![64u64, 32u64]);
    assert_eq!(actual, &skew_shape);
    remove_quiet(&path);
}

#[test]
fn unused_tensors_warn_without_blocking() {
    let path = scratch_path("unused");
    let mut tensors: Vec<(String, Vec<u64>, TensorType)> = tiny_bound_tensors()
        .into_iter()
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    tensors.push((
        "vision.unused".to_string(),
        vec![8u64, 8u64],
        TensorType::F32,
    ));
    emit_checkpoint(&path, tiny_kvs(Vec::new()), &tensors);

    let prepared = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect("prepare succeeds");
    assert_eq!(prepared.bind.unused, vec!["vision.unused".to_string()]);
    remove_quiet(&path);
}

#[test]
fn unknown_architecture_names_arch_and_nearest() {
    let path = scratch_path("arch");
    let mut kvs = tiny_kvs(Vec::new());
    for (key, value) in &mut kvs {
        if key == "general.architecture" {
            *value = KvValue::Str("klingon".to_string());
        }
    }
    let tensors: Vec<(String, Vec<u64>, TensorType)> = tiny_bound_tensors()
        .into_iter()
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    emit_checkpoint(&path, kvs, &tensors);

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::Models(r9v_models::ModelsError::UnknownArchitecture { arch, nearest }) = err
    else {
        panic!("expected UnknownArchitecture, got {err:?}");
    };
    assert_eq!(arch, "klingon");
    assert!(!nearest.is_empty());
    remove_quiet(&path);
}

#[test]
fn malformed_inputs_fail_closed() {
    // Bad magic.
    let bad = scratch_path("badmagic");
    std::fs::write(&bad, vec![0u8; 64]).expect("scratch write");
    let err = prepare(&bad, &test_options(GIB, single_gpu(GIB))).expect_err("bad magic refuses");
    assert!(matches!(err, LoaderError::Format(_)), "unexpected: {err:?}");

    // Truncated header: fewer bytes than any valid table.
    let short = scratch_path("short");
    std::fs::write(&short, vec![b'G'; 10]).expect("scratch write");
    let err = prepare(&short, &test_options(GIB, single_gpu(GIB))).expect_err("short file refuses");
    assert!(matches!(err, LoaderError::Format(_)), "unexpected: {err:?}");

    // Missing architecture key fails closed, not defaulted.
    let nokey = scratch_path("nokey");
    let kvs: Vec<(String, KvValue)> = tiny_kvs(Vec::new())
        .into_iter()
        .filter(|(key, _)| key != "general.architecture")
        .collect();
    let tensors: Vec<(String, Vec<u64>, TensorType)> = tiny_bound_tensors()
        .into_iter()
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    emit_checkpoint(&nokey, kvs, &tensors);
    let err = prepare(&nokey, &test_options(GIB, single_gpu(GIB))).expect_err("no arch refuses");
    assert!(matches!(err, LoaderError::Models(_)), "unexpected: {err:?}");

    // Wrong metadata value type fails closed, not coerced.
    let badtype = scratch_path("badtype");
    let kvs: Vec<(String, KvValue)> = tiny_kvs(Vec::new())
        .into_iter()
        .map(|(key, value)| {
            if key == "llama.block_count" {
                (key, KvValue::Str("two".to_string()))
            } else {
                (key, value)
            }
        })
        .collect();
    emit_checkpoint(&badtype, kvs, &tensors);
    let err = prepare(&badtype, &test_options(GIB, single_gpu(GIB))).expect_err("bad type refuses");
    assert!(matches!(err, LoaderError::Models(_)), "unexpected: {err:?}");

    for path in [&bad, &short, &nokey, &badtype] {
        remove_quiet(path);
    }
}

/// Raw GGUF header: magic + version 3 + zero tensors + one metadata entry.
fn raw_header_one_kv() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    out
}

/// Appends one `Str` metadata entry with key `key`, a declared value length
/// of `len`, no value bytes, and `pad` trailing zero bytes on disk.
fn raw_str_kv_bomb(key: &str, len: u64, pad: usize) -> Vec<u8> {
    let mut out = raw_header_one_kv();
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(&8u32.to_le_bytes()); // KvType::Str
    out.extend_from_slice(&len.to_le_bytes());
    out.extend(std::iter::repeat_n(0u8, pad));
    out
}

#[test]
fn open_huge_string_length_fails_truncated_without_giant_read() {
    let path = scratch_path("strbomb");
    // Declares a 1 GiB string value on a ~100-byte disk: the exact-growth
    // reader must refuse as truncated without allocating the gigabyte.
    let raw = raw_str_kv_bomb("m", 1 << 30, 64);
    std::fs::write(&path, &raw).expect("scratch write");

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::Format(FormatError::Truncated { offset, need, .. }) = err else {
        panic!("expected Truncated, got {err:?}");
    };
    assert_eq!((offset, need), (45, 1 << 30));
    remove_quiet(&path);
}

#[test]
fn open_overflowing_string_length_fails_overflow() {
    let path = scratch_path("stroverflow");
    // `offset + need` overflows u64: checked growth must fail typed instead
    // of wrapping to a small prefix. Trailing bytes keep the disk longer
    // than the parsed prefix so growth is actually attempted.
    let raw = raw_str_kv_bomb("m", u64::MAX - 10, 64);
    std::fs::write(&path, &raw).expect("scratch write");

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    assert!(
        matches!(err, LoaderError::Overflow { .. }),
        "unexpected: {err:?}"
    );
    remove_quiet(&path);
}

#[test]
fn open_mid_table_truncation_fails_as_truncated() {
    let path = scratch_path("midtable");
    let full = write_tiny(&path);
    let file = GgufFile::parse(&full).expect("oracle parse");
    let table_end = file.ti_range().1;
    // Cut inside the tables, well past the fixed header.
    let cut = table_end / 2;
    assert!(cut > 24, "fixture tables must extend past the header");
    std::fs::write(&path, &full[..cut as usize]).expect("truncate write");

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::Format(FormatError::Truncated { offset, need, .. }) = err else {
        panic!("expected Truncated, got {err:?}");
    };
    // The demand runs past the bytes on disk: the file is genuinely short.
    assert!(offset.checked_add(need).is_some());
    assert!(offset + need > cut);
    remove_quiet(&path);
}

/// Overwrites the tensor-info `dims[0]` u64 of `tensor_name` in `bytes`.
fn patch_dims0(bytes: &mut [u8], tensor_name: &str, value: u64) {
    let needle = tensor_name.as_bytes();
    let pos = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("tensor name present");
    let dims0_at = pos + needle.len() + 4; // + u32 n_dims
    bytes[dims0_at..dims0_at + 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn k_family_misalignment_and_overflow_fail_closed() {
    // Valid K-quant tensor first (positive control via unused warning path).
    let path = scratch_path("kq");
    let mut tensors: Vec<(String, Vec<u64>, TensorType)> = tiny_bound_tensors()
        .into_iter()
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    // Q4_K block is 256-wide; [64, 256] outer-last needs 64*144 payload bytes.
    tensors.push((
        "extra.quant".to_string(),
        vec![64u64, 256u64],
        TensorType::Q4_K,
    ));
    let kvs = tiny_kvs(Vec::new());
    let mut writer = GgufWriter::new();
    for (key, value) in &kvs {
        writer.add_kv(key, value.clone()).expect("kv");
    }
    for (name, shape, dtype) in &tensors {
        let data = if *dtype == TensorType::Q4_K {
            vec![0u8; 64 * 144]
        } else {
            zeros_for(shape, *dtype)
        };
        writer
            .add_tensor(name, shape, *dtype, data)
            .expect("tensor");
    }
    let mut full = writer.emit().expect("emit");
    std::fs::write(&path, &full).expect("scratch write");
    let ok = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect("K-quant file loads");
    assert!(ok.bind.unused.contains(&"extra.quant".to_string()));

    // K = 128 violates the 256-block geometry: open refuses, no panic.
    patch_dims0(&mut full, "extra.quant", 128);
    std::fs::write(&path, &full).expect("scratch write");
    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("K violation refuses");
    assert!(matches!(err, LoaderError::Format(_)), "unexpected: {err:?}");

    // K = u64::MAX overflows every size computation: refuses, never wraps.
    let mut full2 = {
        let mut w = GgufWriter::new();
        for (key, value) in &kvs {
            w.add_kv(key, value.clone()).expect("kv");
        }
        for (name, shape) in tiny_bound_tensors() {
            w.add_tensor(
                &name,
                &shape,
                TensorType::F32,
                zeros_for(&shape, TensorType::F32),
            )
            .expect("tensor");
        }
        w.emit().expect("emit")
    };
    patch_dims0(&mut full2, "token_embd.weight", u64::MAX);
    std::fs::write(&path, &full2).expect("scratch write");
    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("overflow refuses");
    assert!(matches!(err, LoaderError::Format(_)), "unexpected: {err:?}");
    remove_quiet(&path);
}

#[test]
fn device_refusal_carries_exact_numbers() {
    let path = scratch_path("refuse");
    write_tiny(&path);

    // Success run establishes the exact requirement R.
    let big = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect("fits in 1 GiB");
    let budget = big.device_budget.expect("device budget");
    let required = budget.required_bytes;

    // One byte less refuses with required == R, shortfall == 1.
    let err =
        prepare(&path, &test_options(GIB, single_gpu(required - 1))).expect_err("must refuse");
    let LoaderError::Budget {
        scope,
        required: got_required,
        available,
        shortfall,
        largest,
        suggestion,
    } = err
    else {
        panic!("expected Budget, got {err:?}");
    };
    assert_eq!(scope, BudgetScope::Device { rank: 0 });
    assert_eq!(got_required, required);
    assert_eq!(available, required - 1);
    assert_eq!(shortfall, 1);
    assert!(!largest.is_empty());
    assert!(largest.len() <= 5);
    assert!(
        largest.windows(2).all(|w| w[0].1 >= w[1].1),
        "sorted: {largest:?}"
    );
    assert!(!suggestion.is_empty(), "actionable suggestion required");
    remove_quiet(&path);
}

#[test]
fn device_refusal_suggests_fitting_max_ctx() {
    let path = scratch_path("ctxsuggest");
    write_tiny(&path);

    // Fit at max_ctx = 64 to learn the requirement, then demand far more
    // context against that same envelope.
    let mut feasible = test_options(GIB, single_gpu(GIB));
    feasible.max_ctx = 64;
    let small = prepare(&path, &feasible).expect("fits");
    let available = small.device_budget.expect("budget").required_bytes;

    let mut hungry = test_options(GIB, single_gpu(available));
    hungry.max_ctx = 1 << 16;
    let err = prepare(&path, &hungry).expect_err("must refuse");
    let LoaderError::Budget { suggestion, .. } = err else {
        panic!("expected Budget, got {err:?}");
    };
    assert!(
        suggestion.starts_with("state.max_ctx = "),
        "got: {suggestion}"
    );
    let suggested: u32 = suggestion
        .split_whitespace()
        .nth(2)
        .expect("number present")
        .parse()
        .expect("number parses");
    assert!(suggested < 1 << 16 && suggested.is_multiple_of(32));

    // The suggestion is actionable: it loads.
    let mut fixed = test_options(GIB, single_gpu(available));
    fixed.max_ctx = suggested;
    prepare(&path, &fixed).expect("suggested ctx fits");
    remove_quiet(&path);
}

#[test]
fn hopeless_device_refusal_suggests_smaller_quant() {
    let path = scratch_path("quant");
    write_tiny(&path);

    let err = prepare(&path, &test_options(GIB, single_gpu(1024))).expect_err("must refuse");
    let LoaderError::Budget {
        suggestion,
        shortfall,
        required,
        available,
        ..
    } = err
    else {
        panic!("expected Budget, got {err:?}");
    };
    assert_eq!(available, 1024);
    assert_eq!(shortfall, required - 1024);
    assert!(suggestion.contains("smaller quant"), "got: {suggestion}");
    remove_quiet(&path);
}

#[test]
fn host_refusal_names_pinned_budget_line() {
    let path = scratch_path("hostref");
    write_tiny(&path);

    // Staging alone (16 MiB x 8) exceeds 1 MiB of pinned budget.
    let starved = test_options(1 << 20, single_gpu(GIB));
    let err = prepare(&path, &starved).expect_err("must refuse");
    let LoaderError::Budget {
        scope, suggestion, ..
    } = err
    else {
        panic!("expected Budget, got {err:?}");
    };
    assert_eq!(scope, BudgetScope::Host);
    assert!(
        suggestion.contains("host.pinned_budget"),
        "got: {suggestion}"
    );
    remove_quiet(&path);
}

#[test]
fn zero_one_and_many_devices_plan_without_hardware() {
    let path = scratch_path("devices");
    write_tiny(&path);
    // Junk in the environment must not matter: planning reads no env, no HIP.
    std::env::set_var("CUDA_VISIBLE_DEVICES", "junk");
    std::env::set_var("HIP_VISIBLE_DEVICES", "junk");

    // Zero GPUs: Cpu plan, no device budget, weights charged to host.
    let cpu = prepare(&path, &test_options(GIB, Vec::new())).expect("cpu prepares");
    assert_eq!(cpu.plan.strategy, PlanStrategy::Cpu);
    assert!(cpu.device_budget.is_none());
    assert!(cpu.host_budget.tensor_bytes > 0);

    // One GPU: Single plan on rank 0.
    let one = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect("single prepares");
    assert_eq!(one.plan.strategy, PlanStrategy::Single);
    assert_eq!(one.device_budget.expect("budget").rank, 0);
    assert_eq!(one.host_budget.tensor_bytes, 0);

    // Many GPUs: the single-device plan still targets the lowest rank.
    let many = prepare(
        &path,
        &test_options(
            GIB,
            vec![
                PlannedDevice {
                    rank: 2,
                    vram_bytes: GIB,
                },
                PlannedDevice {
                    rank: 7,
                    vram_bytes: GIB,
                },
                PlannedDevice {
                    rank: 1,
                    vram_bytes: GIB,
                },
            ],
        ),
    )
    .expect("multi prepares");
    assert_eq!(many.plan.strategy, PlanStrategy::Single);
    assert_eq!(many.device_budget.expect("budget").rank, 1);

    std::env::remove_var("CUDA_VISIBLE_DEVICES");
    std::env::remove_var("HIP_VISIBLE_DEVICES");
    remove_quiet(&path);
}

#[test]
fn placement_legality_matrix() {
    use r9v_ir::Placement;
    use r9v_models::WeightRole;

    let roles = [
        WeightRole::Matmul,
        WeightRole::Embed,
        WeightRole::LmHead,
        WeightRole::NgramTable,
        WeightRole::Vector,
    ];
    for role in roles {
        for is_expert in [false, true] {
            // Device placement is always legal.
            assert!(
                r9v_loader::placement_is_legal(role, is_expert, Placement::Device { rank: 0 }),
                "{role:?} expert={is_expert} on device"
            );
            // Host/Tiered legality follows the semantic role and the
            // carried stacked-expert fact (Spec 1 §2.3).
            let ok = is_expert || matches!(role, WeightRole::Embed | WeightRole::NgramTable);
            for placement in [Placement::Host, Placement::Tiered] {
                assert_eq!(
                    r9v_loader::placement_is_legal(role, is_expert, placement),
                    ok,
                    "{role:?} expert={is_expert} at {placement}"
                );
            }
        }
        // LmHead is not an embedding for this rule even when tied-able.
        if role == WeightRole::LmHead {
            assert!(!r9v_loader::placement_is_legal(
                role,
                false,
                Placement::Host
            ));
        }
    }
    // Embeddings and n-gram tables are host-legal by role.
    assert!(r9v_loader::placement_is_legal(
        WeightRole::Embed,
        false,
        Placement::Host
    ));
    assert!(r9v_loader::placement_is_legal(
        WeightRole::NgramTable,
        false,
        Placement::Tiered
    ));
    // A dense matmul is host-illegal; the same role with the carried
    // expert fact is host-legal.
    assert!(!r9v_loader::placement_is_legal(
        WeightRole::Matmul,
        false,
        Placement::Host
    ));
    assert!(r9v_loader::placement_is_legal(
        WeightRole::Matmul,
        true,
        Placement::Host
    ));
}

#[test]
fn prepare_is_deterministic() {
    let path = scratch_path("det");
    write_tiny(&path);
    let opts = test_options(GIB, single_gpu(GIB));

    let first = prepare(&path, &opts).expect("first prepares");
    let second = prepare(&path, &opts).expect("second prepares");
    let snapshot = |p: &r9v_loader::PreparedLoad| {
        format!(
            "{:?}{:?}{:?}{:?}{:?}{:?}",
            p.file_fp, p.bind, p.plan, p.device_budget, p.host_budget, p.spec
        )
    };
    assert_eq!(snapshot(&first), snapshot(&second));
    remove_quiet(&path);
}

#[test]
fn spec_sizing_constants_match_spec_text() {
    // Spec 9 §4.1/§5.1 numbers are data, not tunable: pin them.
    assert_eq!(TENSOR_ALIGN_BYTES, 256);
    assert_eq!(DEFAULT_RESERVE_BYTES, 512 * 1024 * 1024);
    assert_eq!(DEFAULT_CHUNK_BYTES, 16 * 1024 * 1024);
    assert_eq!(DEFAULT_QUEUE_DEPTH, 8);
}

#[test]
fn vocab_tokenizer_mismatch_refuses() {
    let path = scratch_path("vocab");
    let tensors: Vec<(String, Vec<u64>, TensorType)> = tiny_bound_tensors()
        .into_iter()
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    // Tokenizer claims 63 tokens; the model needs 64.
    let tokens = KvValue::Array {
        elem: r9v_format::KvType::Str,
        items: (0..63).map(|i| KvValue::Str(format!("tok{i}"))).collect(),
    };
    let mut kvs = tiny_kvs(Vec::new());
    kvs.push(("tokenizer.ggml.tokens".to_string(), tokens));
    emit_checkpoint(&path, kvs, &tensors);

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    assert!(
        matches!(err, LoaderError::Validation { .. }),
        "unexpected: {err:?}"
    );
    remove_quiet(&path);
}

// ---------------------------------------------------------------------------
// A2.6 repair: split shards, tied aliases, MoE/MTP, fusion, model_fp,
// budget arithmetic, and metadata-only proofs.
// ---------------------------------------------------------------------------

/// Split-shard KVs for shard `no` of `count` holding `total` tensors.
fn split_kvs(no: u16, count: u16, total: i32) -> Vec<(String, KvValue)> {
    vec![
        ("split.no".to_string(), KvValue::U16(no)),
        ("split.count".to_string(), KvValue::U16(count)),
        ("split.tensors.count".to_string(), KvValue::I32(total)),
    ]
}

/// Writes one split shard; returns its full bytes. `total` is the merged
/// tensor count every shard declares in `split.tensors.count`.
fn write_shard(
    dir: &std::path::Path,
    stem: &str,
    no: u16,
    count: u16,
    total: i32,
    kvs: Vec<(String, KvValue)>,
    tensors: &[(String, Vec<u64>, TensorType)],
) -> (PathBuf, Vec<u8>) {
    let name = format!("{stem}-{:05}-of-{:05}.gguf", u32::from(no) + 1, count);
    let path = dir.join(name);
    let mut all_kvs = tiny_kvs(Vec::new());
    all_kvs.extend(split_kvs(no, count, total));
    all_kvs.extend(kvs);
    let bytes = emit_checkpoint(&path, all_kvs, tensors);
    (path, bytes)
}

fn tensors_total(tensors: &[(String, Vec<u64>, TensorType)]) -> i32 {
    i32::try_from(tensors.len()).expect("fixture tensor count fits i32")
}

/// All bound weights of a graph (root, then nested subgraphs in name
/// order) as F32 checkpoint rows.
fn graph_f32_tensors(graph: &ModelGraph) -> Vec<(String, Vec<u64>, TensorType)> {
    let mut out = Vec::new();
    collect_graph_tensors(graph, &mut out);
    out
}

fn collect_graph_tensors(graph: &ModelGraph, out: &mut Vec<(String, Vec<u64>, TensorType)>) {
    for w in graph.bound_weights() {
        let shape = w
            .shape
            .iter()
            .map(|d| match d {
                r9v_ir::Dim::Concrete(n) => u64::from(*n),
                r9v_ir::Dim::Symbolic(s) => panic!("unexpected symbolic dim {s:?}"),
            })
            .collect();
        out.push((w.name.clone(), shape, TensorType::F32));
    }
    for sub in graph.subgraphs().values() {
        collect_graph_tensors(sub, out);
    }
}

/// Tiny spec with layer 0 replaced by a small MoE.
fn moe_spec() -> r9v_models::ModelSpec {
    let meta = tiny_meta();
    let mut spec = build_from_meta(&meta).expect("tiny llama spec builds");
    spec.layers[0].ffn = Ffn::Moe {
        e: 4,
        k: 2,
        dff_e: 16,
        act: ActivationKind::Silu,
        scoring: MoeScoring::Softmax,
        renormalize: true,
        group: None,
        route_bias: false,
        route_scale: 1.0,
        shared: None,
        shared_gate: false,
    };
    spec
}

/// Tiny spec with one MTP head over a copy of layer 1.
fn mtp_spec() -> r9v_models::ModelSpec {
    let meta = tiny_meta();
    let mut spec = build_from_meta(&meta).expect("tiny llama spec builds");
    spec.mtp = Some(MtpSpec {
        heads: 1,
        layers_per_head: vec![spec.layers[1].clone()],
        takes_hidden_from: MtpSource::Last,
    });
    spec
}

fn build_graph_for(spec: &r9v_models::ModelSpec) -> ModelGraph {
    build_model(Graph::new(IrVersion::CURRENT, "repair-fixture"), spec).expect("graph builds")
}

#[test]
fn split_shards_discover_bind_and_cover_set() {
    let dir = std::env::temp_dir().join(format!(
        "r9v-a26-split-{}-{}",
        std::process::id(),
        SCRATCH_SEQ.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let all = tiny_bound_tensors();
    let total = i32::try_from(all.len()).expect("count fits");
    let mid = all.len() / 2;
    let first_half: Vec<(String, Vec<u64>, TensorType)> = all[..mid]
        .iter()
        .map(|(n, s)| (n.clone(), s.clone(), TensorType::F32))
        .collect();
    let second_half: Vec<(String, Vec<u64>, TensorType)> = all[mid..]
        .iter()
        .map(|(n, s)| (n.clone(), s.clone(), TensorType::F32))
        .collect();

    let (p1, bytes1) = write_shard(&dir, "split", 0, 2, total, Vec::new(), &first_half);
    let (p2, bytes2) = write_shard(&dir, "split", 1, 2, total, Vec::new(), &second_half);

    // Single-path open from the *second* shard discovers its sibling.
    let prepared = prepare(&p2, &test_options(GIB, single_gpu(GIB))).expect("split prepares");
    assert_eq!(prepared.shard_paths.len(), 2);
    assert_eq!(prepared.bind.bound.len(), all.len());
    assert!(prepared.bind.unused.is_empty());
    assert_eq!(prepared.model_fp, ModelFingerprint::PendingUntilRepack);

    // Merged fingerprint covers the set: reassembled independently from
    // per-shard oracle values in shard order.
    let f1 = GgufFile::parse(&bytes1).expect("oracle parse 1");
    let f2 = GgufFile::parse(&bytes2).expect("oracle parse 2");
    let fp1 = f1.file_fp(&bytes1, 2).expect("oracle fp 1");
    let fp2 = f2.file_fp(&bytes2, 2).expect("oracle fp 2");
    let mut input = Vec::with_capacity(32);
    input.extend_from_slice(&fp1.to_le_bytes());
    input.extend_from_slice(&fp2.to_le_bytes());
    assert_eq!(prepared.file_fp, r9v_common::xxh3_128(&input));

    // Explicit path set in reverse order merges identically.
    let reversed = prepare_shard_set(
        &[p2.clone(), p1.clone()],
        &[None, None],
        &test_options(GIB, single_gpu(GIB)),
    )
    .expect("reversed set prepares");
    assert_eq!(reversed.file_fp, prepared.file_fp);
    assert_eq!(reversed.bind.bound, prepared.bind.bound);

    // Every declared tensor bound, in model order across the shard split.
    let names: Vec<&str> = prepared
        .bind
        .bound
        .iter()
        .map(|b| b.name.as_str())
        .collect();
    for (name, _) in &all {
        assert!(names.contains(&name.as_str()), "unbound {name}");
    }

    remove_quiet(&p1);
    remove_quiet(&p2);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn split_missing_sibling_fails_with_exact_path() {
    let dir = std::env::temp_dir().join(format!(
        "r9v-a26-missingshard-{}-{}",
        std::process::id(),
        SCRATCH_SEQ.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let all = tiny_bound_tensors();
    let mid = all.len() / 2;
    let first_half: Vec<(String, Vec<u64>, TensorType)> = all[..mid]
        .iter()
        .map(|(n, s)| (n.clone(), s.clone(), TensorType::F32))
        .collect();
    // Only shard 1 exists; shard 2 is missing.
    let total = tensors_total(&first_half) + 1;
    let (p1, _) = write_shard(&dir, "orphan", 0, 2, total, Vec::new(), &first_half);

    let err = prepare(&p1, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::MissingShard {
        shard_index,
        shard_count,
        expected_path,
    } = err
    else {
        panic!("expected MissingShard, got {err:?}");
    };
    assert_eq!(shard_index, 1);
    assert_eq!(shard_count, 2);
    assert!(
        expected_path.ends_with("orphan-00002-of-00002.gguf"),
        "got: {expected_path}"
    );

    remove_quiet(&p1);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn split_name_without_pattern_fails_typed() {
    let path = scratch_path("nopattern");
    let all = tiny_bound_tensors();
    let tensors: Vec<(String, Vec<u64>, TensorType)> = all
        .into_iter()
        .map(|(n, s)| (n, s, TensorType::F32))
        .collect();
    // Declares a 2-shard set but the file name carries no shard pattern.
    let mut kvs = tiny_kvs(Vec::new());
    kvs.extend(split_kvs(0, 2, tensors_total(&tensors)));
    emit_checkpoint(&path, kvs, &tensors);

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    assert!(
        matches!(err, LoaderError::ShardPattern { .. }),
        "unexpected: {err:?}"
    );
    remove_quiet(&path);
}

#[test]
fn truncated_split_proves_metadata_only() {
    let dir = std::env::temp_dir().join(format!(
        "r9v-a26-truncsplit-{}-{}",
        std::process::id(),
        SCRATCH_SEQ.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let all = tiny_bound_tensors();
    let mid = all.len() / 2;
    let mk = |slice: &[(String, Vec<u64>)]| {
        slice
            .iter()
            .map(|(n, s)| (n.clone(), s.clone(), TensorType::F32))
            .collect::<Vec<_>>()
    };
    let total = i32::try_from(all.len()).expect("count fits");
    let (p1, full1) = write_shard(&dir, "trunc", 0, 2, total, Vec::new(), &mk(&all[..mid]));
    let (p2, full2) = write_shard(&dir, "trunc", 1, 2, total, Vec::new(), &mk(&all[mid..]));
    let f1 = GgufFile::parse(&full1).expect("oracle 1");
    let f2 = GgufFile::parse(&full2).expect("oracle 2");
    // Destroy both payloads: each file ends where its payload starts.
    std::fs::write(&p1, &full1[..f1.data_start() as usize]).expect("truncate 1");
    std::fs::write(&p2, &full2[..f2.data_start() as usize]).expect("truncate 2");

    let prepared = prepare_shard_set(
        &[p1.clone(), p2.clone()],
        &[Some(full1.len() as u64), Some(full2.len() as u64)],
        &test_options(GIB, single_gpu(GIB)),
    )
    .expect("split steps 1-4 succeed without payload bytes");
    assert_eq!(prepared.bind.bound.len(), all.len());

    remove_quiet(&p1);
    remove_quiet(&p2);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn open_full_split_shards_never_read_payload() {
    let dir = std::env::temp_dir().join(format!(
        "r9v-a26-fullsplit-{}-{}",
        std::process::id(),
        SCRATCH_SEQ.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let all = tiny_bound_tensors();
    let mid = all.len() / 2;
    let mk = |slice: &[(String, Vec<u64>)]| {
        slice
            .iter()
            .map(|(n, s)| (n.clone(), s.clone(), TensorType::F32))
            .collect::<Vec<_>>()
    };
    let total = i32::try_from(all.len()).expect("count fits");
    let (p1, full1) = write_shard(&dir, "full", 0, 2, total, Vec::new(), &mk(&all[..mid]));
    let (p2, full2) = write_shard(&dir, "full", 1, 2, total, Vec::new(), &mk(&all[mid..]));
    let f1 = GgufFile::parse(&full1).expect("oracle 1");
    let f2 = GgufFile::parse(&full2).expect("oracle 2");

    // Full files open at their on-disk sizes; neither shard's reads may
    // cross its data section start.
    let prepared = prepare_shard_set(
        &[p1.clone(), p2.clone()],
        &[None, None],
        &test_options(GIB, single_gpu(GIB)),
    )
    .expect("split prepares");
    assert_eq!(prepared.bind.bound.len(), all.len());
    assert!(prepared.bytes_read <= f1.data_start() + f2.data_start());
    assert!(prepared.bytes_read < (full1.len() + full2.len()) as u64);

    remove_quiet(&p1);
    remove_quiet(&p2);
    let _ = std::fs::remove_dir(&dir);
}

/// Tied metadata: llama with word embeddings tied.
fn tied_kvs() -> Vec<(String, KvValue)> {
    vec![("llama.tie_word_embeddings".to_string(), KvValue::Bool(true))]
}

#[test]
fn tied_alias_resolves_and_budgets_once() {
    let path = scratch_path("tied");
    let all = tiny_bound_tensors();
    // The tied head is absent from the checkpoint by contract.
    let tensors: Vec<(String, Vec<u64>, TensorType)> = all
        .iter()
        .filter(|(name, _)| name != "output.weight")
        .map(|(name, shape)| (name.clone(), shape.clone(), TensorType::F32))
        .collect();
    let kvs = tiny_kvs(tied_kvs());
    emit_checkpoint(&path, kvs, &tensors);

    let prepared = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect("tied prepares");
    assert!(prepared.spec.tied_embeddings);
    let alias = prepared
        .bind
        .bound
        .iter()
        .find(|b| b.name == "output.weight")
        .expect("tied head bound");
    assert_eq!(alias.bytes, 0);
    assert_eq!(alias.aliased_to.as_deref(), Some("token_embd.weight"));
    // Shape and scheme report the source exactly.
    let source = prepared
        .bind
        .bound
        .iter()
        .find(|b| b.name == "token_embd.weight")
        .expect("source bound");
    assert_eq!(alias.actual_shape, source.actual_shape);
    assert_eq!(alias.tensor_type, source.tensor_type);
    assert!(alias.expected_shape == vec![64u64, 32u64]);

    // Shared storage budgets once: the device arena covers the resident
    // tensors exactly, with no second copy for the alias.
    let resident: Vec<(String, u64)> = prepared
        .bind
        .bound
        .iter()
        .filter(|b| b.bytes > 0)
        .map(|b| (b.name.clone(), b.bytes))
        .collect();
    assert!(!resident.iter().any(|(n, _)| n == "output.weight"));
    let budget = prepared.device_budget.expect("device budget");
    assert_eq!(
        budget.weights_bytes,
        arena_layout(&resident).expect("layout").1
    );
    remove_quiet(&path);
}

#[test]
fn tied_alias_validates_source_exactly() {
    let path = scratch_path("tiedskew");
    let all = tiny_bound_tensors();
    // Skew the tied source: every other tensor keeps its shape, the
    // embedding loses a row, and the head stays absent.
    let tensors: Vec<(String, Vec<u64>, TensorType)> = all
        .iter()
        .filter(|(name, _)| name != "output.weight")
        .map(|(name, shape)| {
            let shape = if name == "token_embd.weight" {
                vec![63u64, 32u64]
            } else {
                shape.clone()
            };
            (name.clone(), shape, TensorType::F32)
        })
        .collect();
    let kvs = tiny_kvs(tied_kvs());
    emit_checkpoint(&path, kvs, &tensors);

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::Tensors { details } = err else {
        panic!("expected Tensors, got {err:?}");
    };
    // Both the source and the alias report the exact mismatch.
    for name in ["token_embd.weight", "output.weight"] {
        let problem = details
            .iter()
            .find(|d| d.name == name)
            .unwrap_or_else(|| panic!("no report for {name}; details: {details:?}"));
        let TensorProblemKind::ShapeMismatch { expected, actual } = &problem.kind else {
            panic!("expected ShapeMismatch for {name}, got {:?}", problem.kind);
        };
        assert_eq!(expected, &vec![64u64, 32u64]);
        assert_eq!(actual, &vec![63u64, 32u64]);
    }
    remove_quiet(&path);
}

#[test]
fn moe_binds_stacked_experts_with_carried_facts() {
    let path = scratch_path("moe");
    let spec = moe_spec();
    let graph = build_graph_for(&spec);
    // The MoE lowering marks exactly the two stacked tensors per MoE layer.
    assert!(graph.is_stacked_expert("blk.0.ffn_gate_up_exps.weight"));
    assert!(graph.is_stacked_expert("blk.0.ffn_down_exps.weight"));
    assert!(!graph.is_stacked_expert("blk.0.attn_q.weight"));
    assert!(!graph.is_stacked_expert("blk.1.ffn_gate.weight"));
    assert!(is_stacked_expert_weight(
        &graph,
        "blk.0.ffn_down_exps.weight"
    ));
    assert!(!is_stacked_expert_weight(&graph, "token_embd.weight"));

    let tensors = graph_f32_tensors(&graph);
    emit_checkpoint(&path, tiny_kvs(Vec::new()), &tensors);
    let ckpt = open(&path).expect("moe opens");
    let report = bind(&graph, &ckpt, PlanStrategy::Single).expect("moe binds");
    assert_eq!(report.bound.len(), tensors.len());
    assert!(report.unused.is_empty());

    // Expert bytes are exactly the two stacked tensors (F32).
    let expert_bytes: u64 = report
        .bound
        .iter()
        .filter(|b| is_stacked_expert_weight(&graph, &b.name))
        .map(|b| b.bytes)
        .sum();
    // gate_up [4, 32, 32] + down [4, 32, 16] at 4 B/elem.
    assert_eq!(expert_bytes, (4 * 32 * 32 + 4 * 32 * 16) * 4);

    // The plan carries the summary's expert facts onto the device.
    let summary = graph.summary().expect("summary builds");
    let plan = plan_single_device(&summary, &single_gpu(GIB)).expect("plans");
    assert_eq!(plan.expert_map.len(), 4);
    for assign in &plan.expert_map {
        assert_eq!(assign.layer, 0);
        assert_eq!(assign.rank, 0);
        assert_eq!(assign.placement, r9v_ir::ExpertPlacement::Device);
    }
    remove_quiet(&path);
}

#[test]
fn mtp_absent_downgrades_to_none() {
    let path = scratch_path("mtpabsent");
    let spec = mtp_spec();
    assert!(spec.mtp.is_some());
    let graph = build_graph_for(&spec);
    assert!(graph.subgraphs().contains_key("mtp"));
    // Checkpoint holds the base model only: no MTP tensor at all.
    let mut tensors = Vec::new();
    for w in graph.bound_weights() {
        let shape = w
            .shape
            .iter()
            .map(|d| match d {
                r9v_ir::Dim::Concrete(n) => u64::from(*n),
                r9v_ir::Dim::Symbolic(s) => panic!("unexpected symbolic dim {s:?}"),
            })
            .collect();
        tensors.push((w.name.clone(), shape, TensorType::F32));
    }
    emit_checkpoint(&path, tiny_kvs(Vec::new()), &tensors);

    let ckpt = open(&path).expect("opens");
    let (spec, graph) = downgrade_absent_mtp(spec, &ckpt, "mtp-absent").expect("downgrades");
    assert!(spec.mtp.is_none());
    assert!(!graph.subgraphs().contains_key("mtp"));
    // The downgraded graph binds cleanly against the same file.
    let report = bind(&graph, &ckpt, PlanStrategy::Single).expect("downgraded binds");
    assert_eq!(report.bound.len(), tensors.len());
    remove_quiet(&path);
}

#[test]
fn mtp_partial_lists_every_missing_member() {
    let path = scratch_path("mtppartial");
    let spec = mtp_spec();
    let graph = build_graph_for(&spec);
    let mtp = graph.subgraphs().get("mtp").expect("mtp subgraph built");
    let mtp_names: Vec<String> = mtp.bound_weights().iter().map(|w| w.name.clone()).collect();
    assert!(mtp_names.len() > 1, "mtp head needs several weights");

    // Base weights plus exactly one MTP tensor: partially present head.
    let mut tensors = graph_f32_tensors(&graph)
        .into_iter()
        .filter(|(name, _, _)| !mtp_names.contains(name) || *name == "blk.0.mtp.output.weight")
        .collect::<Vec<_>>();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    emit_checkpoint(&path, tiny_kvs(Vec::new()), &tensors);

    // Present-but-incomplete keeps the head: no downgrade.
    let ckpt = open(&path).expect("opens");
    let (kept, _) = downgrade_absent_mtp(spec, &ckpt, "mtp-partial").expect("kept");
    assert!(kept.mtp.is_some());

    // Binding lists every missing member exactly — never a silent skip.
    let err = bind(&graph, &ckpt, PlanStrategy::Single).expect_err("must refuse");
    let LoaderError::Tensors { details } = err else {
        panic!("expected Tensors, got {err:?}");
    };
    let expected: Vec<&str> = mtp_names
        .iter()
        .filter(|n| *n != "blk.0.mtp.output.weight")
        .map(String::as_str)
        .collect();
    assert_eq!(details.len(), expected.len(), "details: {details:?}");
    for name in expected {
        assert!(
            details
                .iter()
                .any(|d| d.name == name && matches!(d.kind, TensorProblemKind::Missing)),
            "missing report for {name}; details: {details:?}"
        );
    }
    remove_quiet(&path);
}

#[test]
fn mtp_complete_binds_subgraph_weights() {
    let path = scratch_path("mtpfull");
    let spec = mtp_spec();
    let graph = build_graph_for(&spec);
    let tensors = graph_f32_tensors(&graph);
    emit_checkpoint(&path, tiny_kvs(Vec::new()), &tensors);

    let ckpt = open(&path).expect("opens");
    let (kept, kept_graph) = downgrade_absent_mtp(spec, &ckpt, "mtp-full").expect("kept");
    assert!(kept.mtp.is_some());
    let report = bind(&kept_graph, &ckpt, PlanStrategy::Single).expect("mtp binds");
    assert_eq!(report.bound.len(), tensors.len());
    assert!(
        report
            .bound
            .iter()
            .any(|b| b.name == "blk.0.mtp.output.weight"),
        "mtp head weights bound"
    );
    let summary = kept_graph.summary().expect("summary builds");
    assert!(summary.mtp);
    remove_quiet(&path);
}

/// One native checkpoint row with its payload.
struct NativeRow {
    name: String,
    shape: Vec<u64>,
    dtype: TensorType,
    payload: Vec<u8>,
}

/// Native dtypes: multi-dim weights as F16, 1-D vectors as F32 (Spec 2
/// §3.3: native F32 is 1-D only and must declare role [vector]).
/// Payloads are zero-padded to the 4096-byte native entry size, and the
/// emitted `xxh3` checksums hash the padded entries the container slices.
fn native_rows(tensors: &[(String, Vec<u64>, TensorType)]) -> Vec<NativeRow> {
    tensors
        .iter()
        .map(|(name, shape, _)| {
            let elems: u64 = shape.iter().product();
            let wire = if shape.len() == 1 {
                elems * 4
            } else {
                elems * 2
            };
            let padded = wire.div_ceil(4096) * 4096;
            let dtype = if shape.len() == 1 {
                TensorType::F32
            } else {
                TensorType::F16
            };
            NativeRow {
                name: name.clone(),
                shape: shape.clone(),
                dtype,
                payload: vec![0u8; padded as usize],
            }
        })
        .collect()
}

/// Expected interleave per fused member from the graph's own declarations.
fn fusion_interleaves(graph: &ModelGraph) -> Vec<(String, String)> {
    let mut out = Vec::new();
    collect_fusion_interleaves(graph, &mut out);
    out
}

fn collect_fusion_interleaves(graph: &ModelGraph, out: &mut Vec<(String, String)>) {
    for decl in graph.fusion_decls() {
        match decl {
            r9v_models::FusionDecl::Qkv { q, k, v } => {
                for m in [q, k, v] {
                    out.push((m.clone(), "qkv".to_string()));
                }
            }
            r9v_models::FusionDecl::GateUp { gate, up } => {
                for m in [gate, up] {
                    out.push((m.clone(), "gate_up".to_string()));
                }
            }
        }
    }
    for sub in graph.subgraphs().values() {
        collect_fusion_interleaves(sub, out);
    }
}

/// Emits a native checkpoint: 4096 alignment, format version, layout,
/// per-tensor roles for vectors, real `xxh3` checksums over each payload
/// (unless `skip_xxh3` names it), and interleave declarations for every
/// fused member (unless `skip_interleave` names it).
fn emit_native(
    path: &PathBuf,
    rows: &[NativeRow],
    skip_xxh3: &[&str],
    skip_interleave: &[&str],
    interleaves: &[(String, String)],
) -> Vec<u8> {
    let mut writer = GgufWriter::new()
        .with_alignment(4096)
        .expect("4096 is a valid alignment");
    for (key, value) in tiny_kvs(Vec::new()) {
        writer.add_kv(&key, value).expect("kv inserts");
    }
    // The reader takes alignment from `general.alignment` (default 32);
    // native files fix 4096 (Spec 2 §6).
    writer
        .add_kv("general.alignment", KvValue::U32(4096))
        .expect("alignment kv");
    writer
        .add_kv("r9v.format_version", KvValue::U32(1))
        .expect("version kv");
    writer
        .add_kv("r9v.layout_id", KvValue::Str("l1".to_string()))
        .expect("layout kv");
    for row in rows {
        if row.dtype == TensorType::F32 {
            writer
                .add_kv(
                    &format!("r9v.tensor.{}.roles", row.name),
                    KvValue::Array {
                        elem: r9v_format::KvType::Str,
                        items: vec![KvValue::Str("vector".to_string())],
                    },
                )
                .expect("roles kv");
        }
        if !skip_xxh3.contains(&row.name.as_str()) {
            writer
                .add_kv(
                    &format!("r9v.tensor.{}.xxh3", row.name),
                    KvValue::U64(r9v_common::xxh3_64(&row.payload)),
                )
                .expect("xxh3 kv");
        }
    }
    for (member, kind) in interleaves {
        if skip_interleave.contains(&member.as_str()) {
            continue;
        }
        writer
            .add_kv(
                &format!("r9v.tensor.{member}.interleave"),
                KvValue::Str(kind.clone()),
            )
            .expect("interleave kv");
    }
    for row in rows {
        writer
            .add_tensor(&row.name, &row.shape, row.dtype, row.payload.clone())
            .expect("tensor inserts");
    }
    let bytes = writer.emit().expect("emit succeeds");
    std::fs::write(path, &bytes).expect("scratch write");
    bytes
}

#[test]
fn fusion_interleave_mismatch_refuses_on_native() {
    let path = scratch_path("fusionbad");
    let meta = tiny_meta();
    let spec = build_from_meta(&meta).expect("tiny spec builds");
    let graph = build_graph_for(&spec);
    let rows = native_rows(
        &tiny_bound_tensors()
            .into_iter()
            .map(|(n, s)| (n, s, TensorType::F32))
            .collect::<Vec<_>>(),
    );
    let interleaves = fusion_interleaves(&graph);
    assert!(!interleaves.is_empty(), "tiny model declares fusions");
    // One gate member declares nothing: native default is `none`.
    let skewed = interleaves[0].0.clone();
    emit_native(&path, &rows, &[], &[&skewed], &interleaves);

    let ckpt = open(&path).expect("native opens");
    let problems = check_fusion_decls(&graph, &ckpt);
    assert_eq!(problems.len(), 1, "problems: {problems:?}");
    assert!(
        problems[0].contains(&skewed),
        "problem names {skewed}: {}",
        problems[0]
    );

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::Tensors { details } = err else {
        panic!("expected Tensors, got {err:?}");
    };
    assert_eq!(details.len(), 1, "details: {details:?}");
    assert_eq!(details[0].name, skewed);
    assert!(
        matches!(details[0].kind, TensorProblemKind::Fusion { .. }),
        "got {:?}",
        details[0].kind
    );
    remove_quiet(&path);
}

#[test]
fn fusion_decls_pass_on_matching_native_and_standard() {
    let meta = tiny_meta();
    let spec = build_from_meta(&meta).expect("tiny spec builds");
    let graph = build_graph_for(&spec);
    let interleaves = fusion_interleaves(&graph);

    // Fully declared native checkpoint prepares cleanly.
    let native = scratch_path("fusiongood");
    let rows = native_rows(
        &tiny_bound_tensors()
            .into_iter()
            .map(|(n, s)| (n, s, TensorType::F32))
            .collect::<Vec<_>>(),
    );
    emit_native(&native, &rows, &[], &[], &interleaves);
    let ckpt = open(&native).expect("native opens");
    assert!(check_fusion_decls(&graph, &ckpt).is_empty());
    let prepared =
        prepare(&native, &test_options(GIB, single_gpu(GIB))).expect("matching native prepares");
    assert!(matches!(prepared.model_fp, ModelFingerprint::Ready(_)));
    remove_quiet(&native);

    // Standard GGUF carries no interleave metadata: membership and
    // geometry still check, with nothing to mismatch.
    let standard = scratch_path("fusionstd");
    write_tiny(&standard);
    let ckpt = open(&standard).expect("standard opens");
    assert!(check_fusion_decls(&graph, &ckpt).is_empty());
    remove_quiet(&standard);
}

#[test]
fn model_fp_ready_for_complete_native_pending_otherwise() {
    let meta = tiny_meta();
    let spec = build_from_meta(&meta).expect("tiny spec builds");
    let graph = build_graph_for(&spec);
    let interleaves = fusion_interleaves(&graph);
    let rows = native_rows(
        &tiny_bound_tensors()
            .into_iter()
            .map(|(n, s)| (n, s, TensorType::F32))
            .collect::<Vec<_>>(),
    );

    // Complete checksums: Ready, matching the reassembled oracle over the
    // merged table order.
    let full = scratch_path("fpready");
    let bytes = emit_native(&full, &rows, &[], &[], &interleaves);
    let ckpt = open(&full).expect("native opens");
    let ModelFingerprint::Ready(fp) = ckpt.model_fp() else {
        panic!("complete native must be Ready");
    };
    let file = GgufFile::parse(&bytes).expect("oracle parse");
    let mut hashes = Vec::new();
    for info in file.tensors() {
        hashes.push(r9v_common::xxh3_64(
            file.tensor_bytes(&info.name, &bytes).expect("entry bytes"),
        ));
    }
    assert_eq!(fp, r9v_format::model_fp(ckpt.file_fp(), &hashes));

    // Payload bytes cannot move it: flip payload, same Ready value.
    let mut flipped = bytes.clone();
    let start = file.data_start() as usize;
    flipped[start] ^= 0xFF;
    std::fs::write(&full, &flipped).expect("flip write");
    let reopened = open(&full).expect("reopens after flip");
    assert_eq!(reopened.model_fp(), ModelFingerprint::Ready(fp));
    remove_quiet(&full);

    // One missing checksum: Pending, never fabricated.
    let partial = scratch_path("fppartial");
    emit_native(&partial, &rows, &["token_embd.weight"], &[], &interleaves);
    let ckpt = open(&partial).expect("partial native opens");
    assert_eq!(ckpt.model_fp(), ModelFingerprint::PendingUntilRepack);
    remove_quiet(&partial);

    // Standard GGUF: Pending.
    let standard = scratch_path("fpstd");
    write_tiny(&standard);
    let ckpt = open(&standard).expect("standard opens");
    assert_eq!(ckpt.model_fp(), ModelFingerprint::PendingUntilRepack);
    remove_quiet(&standard);
}

#[test]
fn file_fp_matches_independent_reassembly() {
    let path = scratch_path("fporacle");
    let full = write_tiny(&path);
    let ckpt = open(&path).expect("opens");

    // Reassemble the hashed input from the recorded ranges, in the
    // documented order: header ‖ tensor-info ‖ KV ‖ size ‖ shard count.
    let file = GgufFile::parse(&full).expect("oracle parse");
    let slice = |range: (u64, u64)| &full[range.0 as usize..range.1 as usize];
    let mut input = Vec::new();
    input.extend_from_slice(slice(file.header_range()));
    input.extend_from_slice(slice(file.ti_range()));
    input.extend_from_slice(slice(file.kv_range()));
    input.extend_from_slice(&(full.len() as u64).to_le_bytes());
    input.extend_from_slice(&1u64.to_le_bytes());
    assert_eq!(ckpt.file_fp(), r9v_common::xxh3_128(&input));

    // Frozen golden for this exact deterministic fixture.
    assert_eq!(format!("{:032x}", ckpt.file_fp()), TINY_FILE_FP_HEX);
    remove_quiet(&path);
}

// Set from the first green run of `file_fp_matches_independent_reassembly`
// (deterministic writer output); the reassembly above guards it.
const TINY_FILE_FP_HEX: &str = "c58fd9981cef94ba47f882ba5afcd7ca";

#[test]
fn open_ignores_payload_content() {
    let path = scratch_path("flip");
    let full = write_tiny(&path);
    let before = open(&path).expect("opens");
    let names_before: Vec<String> = merged_tensor_names(&before);

    // Corrupt 64 payload bytes: tables and fingerprint must not move.
    let file = GgufFile::parse(&full).expect("oracle parse");
    let mut corrupted = full.clone();
    let start = file.data_start() as usize;
    for b in &mut corrupted[start..start + 64] {
        *b ^= 0xFF;
    }
    std::fs::write(&path, &corrupted).expect("corrupt write");

    let after = open(&path).expect("opens despite corrupt payload");
    assert_eq!(after.file_fp(), before.file_fp());
    let names_after: Vec<String> = merged_tensor_names(&after);
    assert_eq!(names_after, names_before);
    remove_quiet(&path);
}

/// Tensor names in merged table order.
fn merged_tensor_names(ckpt: &r9v_loader::OpenedCheckpoint) -> Vec<String> {
    let set = ckpt.shard_set();
    (0..set.len())
        .filter_map(|i| set.tensor_at(i).map(|(_, info)| info.name.clone()))
        .collect()
}

#[test]
fn large_metadata_opens_with_prefix_growth() {
    let path = scratch_path("largemeta");
    let tensors: Vec<(String, Vec<u64>, TensorType)> = tiny_bound_tensors()
        .into_iter()
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    // ~110 KiB of keys forces the prefix far past the fixed header while
    // staying well under the full file, so exact growth is exercised and
    // the final read still proves payload is never paged.
    let mut extra = Vec::new();
    for i in 0..1500u32 {
        extra.push((format!("bulk.pad.{i:05}"), KvValue::Str("x".repeat(40))));
    }
    let full = emit_checkpoint(&path, tiny_kvs(extra), &tensors);
    assert!(
        full.len() as u64 > 160 * 1024,
        "fixture must exceed prefixes"
    );
    let file = GgufFile::parse(&full).expect("oracle parse");

    let prepared = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect("prepares");
    assert_eq!(prepared.bind.bound.len(), tensors.len());
    // The prefix grew past the fixed header to exactly the table end, yet
    // never paged payload.
    assert!(prepared.bytes_read > 24);
    assert_eq!(prepared.bytes_read, file.ti_range().1);
    assert!(prepared.bytes_read <= file.data_start());
    assert!(prepared.bytes_read < full.len() as u64);
    assert_eq!(prepared.file_fp, file.file_fp(&full, 1).expect("oracle fp"));
    remove_quiet(&path);
}

#[test]
fn step2_aggregates_tensor_and_validation_problems() {
    let path = scratch_path("step2");
    let all = tiny_bound_tensors();
    // One missing tensor plus a tokenizer/vocab mismatch in one file.
    let tensors: Vec<(String, Vec<u64>, TensorType)> = all
        .into_iter()
        .filter(|(name, _)| name != "blk.1.ffn_down.weight")
        .map(|(name, shape)| (name, shape, TensorType::F32))
        .collect();
    let tokens = KvValue::Array {
        elem: r9v_format::KvType::Str,
        items: (0..63).map(|i| KvValue::Str(format!("tok{i}"))).collect(),
    };
    let mut kvs = tiny_kvs(Vec::new());
    kvs.push(("tokenizer.ggml.tokens".to_string(), tokens));
    emit_checkpoint(&path, kvs, &tensors);

    let err = prepare(&path, &test_options(GIB, single_gpu(GIB))).expect_err("must refuse");
    let LoaderError::Step2 { details, problems } = err else {
        panic!("expected Step2, got {err:?}");
    };
    assert!(
        details
            .iter()
            .any(|d| d.name == "blk.1.ffn_down.weight"
                && matches!(d.kind, TensorProblemKind::Missing)),
        "tensor class: {details:?}"
    );
    assert!(
        problems.iter().any(|p| p.contains("tokenizer")),
        "validation class: {problems:?}"
    );
    remove_quiet(&path);
}

#[test]
fn arena_layout_aligns_starts_not_sizes() {
    // Each tensor starts 256-aligned; sizes are never rounded. Two
    // 1-byte tensors occupy [0, 1) and [256, 257): total 257, not 512.
    let (offsets, total) =
        arena_layout(&[("a".to_string(), 1), ("b".to_string(), 1)]).expect("layouts");
    assert_eq!(offsets, vec![("a".to_string(), 0), ("b".to_string(), 256)]);
    assert_eq!(total, 257);

    // An already-aligned tensor adds no pad.
    let (offsets, total) = arena_layout(&[("c".to_string(), 256)]).expect("layouts");
    assert_eq!(offsets, vec![("c".to_string(), 0)]);
    assert_eq!(total, 256);

    // Empty arena is empty.
    let (offsets, total) = arena_layout(&[]).expect("layouts");
    assert!(offsets.is_empty());
    assert_eq!(total, 0);

    // Overflow fails closed instead of wrapping: the second start
    // rounds past the end of the address space.
    let err = arena_layout(&[("x".to_string(), u64::MAX), ("y".to_string(), 1)])
        .expect_err("must refuse");
    assert!(
        matches!(err, LoaderError::Overflow { .. }),
        "unexpected: {err:?}"
    );
}

#[test]
fn device_budget_suggests_hot_set_vram_with_numbers() {
    use r9v_state::LayerGroup;
    let groups: Vec<LayerGroup> = Vec::new();
    // 10 MiB dense + 40 MiB experts, 1 MiB workspace + 1 MiB reserve.
    let tensors = vec![
        ("dense.weight".to_string(), 10 << 20),
        ("blk.0.ffn_gate_up_exps.weight".to_string(), 40 << 20),
    ];
    let base = |available: u64| DeviceBudgetInput {
        rank: 0,
        tensors: &tensors,
        expert_bytes: 40 << 20,
        groups: &groups,
        max_ctx: 64,
        max_seqs: 1,
        workspace_bytes: 1 << 20,
        comms_bytes: 0,
        reserve_bytes: 1 << 20,
        available_bytes: available,
    };
    // Fits at 64 MiB.
    let ok = check_device_budget(&base(64 << 20)).expect("fits");
    assert_eq!(ok.required_bytes, ok.weights_bytes + 2 * (1 << 20));

    // At 20 MiB, context knobs cannot help (no state pools here), so the
    // expert knob fires with a numeric resulting requirement.
    let err = check_device_budget(&base(20 << 20)).expect_err("must refuse");
    let LoaderError::Budget { suggestion, .. } = err else {
        panic!("expected Budget, got {err:?}");
    };
    assert!(
        suggestion.starts_with("experts.hot_set_vram = "),
        "got: {suggestion}"
    );
    // Largest fitting hot set: H + 12 MiB <= 20 MiB.
    assert_eq!(
        suggestion,
        "experts.hot_set_vram = 8388608 (requires 20971520 B of 20971520 B available)"
    );
}

#[test]
fn expert_bytes_beyond_total_fails_closed() {
    use r9v_state::LayerGroup;
    let groups: Vec<LayerGroup> = Vec::new();
    let tensors = vec![("w".to_string(), 1024)];
    let err = check_device_budget(&DeviceBudgetInput {
        rank: 0,
        tensors: &tensors,
        expert_bytes: 1025,
        groups: &groups,
        max_ctx: 64,
        max_seqs: 1,
        workspace_bytes: 0,
        comms_bytes: 0,
        reserve_bytes: 0,
        available_bytes: GIB,
    })
    .expect_err("must refuse");
    assert!(
        matches!(err, LoaderError::Validation { .. }),
        "unexpected: {err:?}"
    );
}

#[test]
fn duplicate_ranks_refuse_deterministically() {
    let meta = tiny_meta();
    let spec = build_from_meta(&meta).expect("tiny spec builds");
    let graph = build_graph_for(&spec);
    let summary = graph.summary().expect("summary builds");

    let dupes_a = vec![
        PlannedDevice {
            rank: 1,
            vram_bytes: GIB,
        },
        PlannedDevice {
            rank: 0,
            vram_bytes: GIB,
        },
        PlannedDevice {
            rank: 1,
            vram_bytes: GIB,
        },
    ];
    let dupes_b = vec![
        PlannedDevice {
            rank: 0,
            vram_bytes: GIB,
        },
        PlannedDevice {
            rank: 1,
            vram_bytes: GIB,
        },
        PlannedDevice {
            rank: 1,
            vram_bytes: GIB,
        },
    ];
    let err_a = plan_single_device(&summary, &dupes_a).expect_err("must refuse");
    let err_b = plan_single_device(&summary, &dupes_b).expect_err("must refuse");
    // Input order cannot change the refusal.
    assert_eq!(format!("{err_a:?}"), format!("{err_b:?}"));
    let LoaderError::Validation { problems } = err_a else {
        panic!("expected Validation, got {err_a:?}");
    };
    assert_eq!(problems.len(), 1);
    assert!(
        problems[0].contains("duplicate device rank 1"),
        "got: {problems:?}"
    );

    // prepare() refuses the same way before any I/O budgeting.
    let path = scratch_path("dupes");
    write_tiny(&path);
    let err = prepare(&path, &test_options(GIB, dupes_a)).expect_err("must refuse");
    assert!(
        matches!(err, LoaderError::Validation { .. }),
        "unexpected: {err:?}"
    );
    remove_quiet(&path);
}
