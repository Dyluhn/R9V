// SPDX-License-Identifier: Apache-2.0
//! T1 portable sampling runner tests (Card A3.6; Spec 4 §2, §10; Spec 1 §4.F, §6.5).
//!
//! Mirrors A1.8 through the A1.10 harness against T0: `logits_postprocess`,
//! `sample`, and `verify` reference kernels under `kernels/reference/` run on
//! the device with every argument bound through the A3.2 generated ABI (no
//! duplicated structs: the compiland embeds `emit_hip_struct` output and the
//! Rust side packs argument bytes from `AbiStruct` field offsets), and every
//! variant resolves through the registry T1 fallback with `t1_<op>` entry
//! symbols. Gates: golden (32 seeds per shape vs T0/f64), batch invariance
//! (alone / padded / embedded), determinism (twice), shape fuzz plus explicit
//! hostile/invalid-input refusals, RNG raw-state and overflow boundaries,
//! and temperature-zero / tree-path tie rules.
//!
//! Tests that execute kernels skip only when no compatible HIP runtime or
//! device exists; every other test runs on any host.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use r9v_common::rng::SeededRng;
use r9v_hip::{DeviceBuffer, HipError, HipLibrary, Module, Stream};
use r9v_ir::{RngAlgorithm, SamplingParams, TreeMask, VerifyMethod};
use r9v_kgen::abi::{emit_hip_struct, AbiStruct, AbiType, ScalarType};
use r9v_kgen::{abi_for_op, canonical_struct_name};
use r9v_registry::{
    ArchName, BundleManifest, LaunchGeometry, LogitsPostprocessStatic, ManifestVariantEntry, OpId,
    OpStatic, Registry, RegistryConfig, SampleStatic, SamplingStatic, Tier, VariantHash,
    VerifyMethodStatic, VerifyStatic,
};
use r9v_t0::harness::{
    check_f32_against_f64, keep_mask, run_gates, uniform_f32, BatchRows, GateBuffers, GateCase,
    HarnessError,
};
use r9v_t0::{
    logits_postprocess, logits_postprocess_f64_reference, sample, verify, RngState, T0Error,
    Tolerance, TypedBuffer,
};

// ---------------------------------------------------------------------------
// Constants (Spec 4 §4.3, §5.8, §9.2)
// ---------------------------------------------------------------------------

/// Token replaced by the canonical A3.2 struct name when assembling a compiland.
const ARGS_TOKEN: &str = "R9V_ARGS";

/// Pinned kernel compile flags (Spec 4 §4.3: direct `clang++`, never `hipcc`).
const PINNED_CXXFLAGS: &[&str] = &[
    "-x",
    "hip",
    "-O3",
    "-fno-fast-math",
    "-fno-gpu-approx-transcendentals",
    "--offload-device-only",
    "-c",
];

/// Reference kernel file names under `kernels/reference/`.
const COMMON_HIP: &str = "t1_sampling_common.hip";
const PP_HIP: &str = "t1_logits_postprocess.hip";
const SAMPLE_HIP: &str = "t1_sample.hip";
const VERIFY_HIP: &str = "t1_verify.hip";

/// Device `verify` local node cap (mirrors `R9V_VERIFY_MAXK`).
const DEVICE_MAX_K: usize = 64;

// ---------------------------------------------------------------------------
// Device record layouts (single definition per language; see common header)
// ---------------------------------------------------------------------------

/// Host mirror of `R9vSamplingParams` plus the trailing bias-pair run layout.
///
/// The A3.2 ABI types `params` as `const void*`, so the record shape is
/// defined once in `t1_sampling_common.hip` and once here; the layout test
/// below pins size and every field offset on both sides of the boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct DeviceSamplingParams {
    temperature: f32,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    top_k: u32,
    bias_start: u32,
    bias_count: u32,
    _pad: u32,
}

/// Host mirror of `R9vBiasPair` (token, bias).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct DeviceBiasPair {
    token: u32,
    bias: f32,
}

/// Host mirror of `R9vRngWords`: 3xu64 `{seed, step, draw}` per sequence
/// (see the DECISION note in the common header for why one u64 is not enough).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeviceRngWords {
    seed: u64,
    step: u64,
    draw: u64,
}

/// Computes a struct field offset without external crates (no new deps).
/// `addr_of!` never reads the value, so the uninit pointer is never
/// dereferenced.
macro_rules! offset_of {
    ($ty:ty, $field:ident) => {{
        let uninit = std::mem::MaybeUninit::<$ty>::uninit();
        let base = uninit.as_ptr() as usize;
        let field = unsafe { std::ptr::addr_of!((*uninit.as_ptr()).$field) } as usize;
        field.wrapping_sub(base)
    }};
}

// ---------------------------------------------------------------------------
// Byte helpers (no new dependencies)
// ---------------------------------------------------------------------------

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn u32_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn next_pow2(mut v: usize) -> usize {
    let mut p = 1usize;
    while p < v {
        p <<= 1;
    }
    v = p;
    v
}

// ---------------------------------------------------------------------------
// GPU probe (CONVENTIONS.md §4.4: explicit skip, never silent)
// ---------------------------------------------------------------------------

fn gpu_lane_required() -> bool {
    std::env::var("R9V_REQUIRE_GPU").as_deref() == Ok("1")
        || std::env::var("R9V_GPU_LANE").as_deref() == Ok("1")
        || std::env::var("GPU_LANE").as_deref() == Ok("1")
}

/// Probes for a usable HIP runtime + device, returning `None` with an
/// explicit skip message when genuinely absent (Spec 14 §3).
fn probe_gpu() -> Option<Arc<HipLibrary>> {
    match HipLibrary::default_or_load() {
        Ok(l) => {
            match l.device_count() {
                Ok(c) if c > 0 => Some(l),
                Ok(_) => {
                    if gpu_lane_required() {
                        panic!("GPU lane required but device count == 0");
                    }
                    println!("[SKIP] No HIP GPU devices (device count == 0); T1 kernel execution tests skipped.");
                    println!("[HONEST TIER REPORT] Host tier: T0 only; hardware T1 unavailable (no device).");
                    None
                }
                Err(e) => {
                    if gpu_lane_required() {
                        panic!("GPU lane required but device_count failed: {e}");
                    }
                    println!("[SKIP] device_count query failed: {e}");
                    None
                }
            }
        }
        Err(HipError::LibraryNotFound { searched }) => {
            if gpu_lane_required() {
                panic!("GPU lane required but HIP library not found; searched: {searched:?}");
            }
            println!("[SKIP] HIP runtime not available; searched: {searched:?}");
            println!("[HONEST TIER REPORT] Host tier: T0 only; hardware T1 unavailable (no HIP runtime).");
            None
        }
        Err(e) => panic!("HIP library load failed unexpectedly: {e}"),
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("tests/gpu file must live two levels below the workspace root")
        .to_path_buf()
}

/// Assembles one T1 compiland from the generated ABI struct, the shared
/// header, and a kernel body (Spec 4 §7). Free function so host tests can
/// assert the binding without a device.
fn assemble_compiland(
    reference_dir: &Path,
    abi: &AbiStruct,
    body_file: &str,
    defines: &[(String, String)],
) -> String {
    let read = |name: &str| {
        let path = reference_dir.join(name);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("reference source must be readable at {}", path.display()))
    };
    let body = read(body_file);
    assert!(
        !body.contains("_args {") && !body.contains("_args{"),
        "{body_file} must not declare its own args struct; it binds R9V_ARGS from the generated ABI"
    );
    assert!(
        body.contains(ARGS_TOKEN),
        "{body_file} must reference the {ARGS_TOKEN} struct token"
    );
    let mut out = String::new();
    out.push_str("// Assembled T1 compiland: A3.2 ABI struct + shared header + body.\n");
    // The runtime header comes first: the emitted struct needs the stdint
    // spellings and the shared header needs the HIP qualifiers (real
    // `hip_runtime.h` is include-guarded, so the body's own include is a no-op).
    out.push_str("#include <hip/hip_runtime.h>\n");
    for (k, v) in defines {
        out.push_str(&format!("#define {k} {v}\n"));
    }
    out.push_str(&emit_hip_struct(abi));
    out.push_str(&read(COMMON_HIP));
    out.push_str(&body.replace(ARGS_TOKEN, &abi.name));
    assert!(
        !out.contains(ARGS_TOKEN),
        "{body_file} still references {ARGS_TOKEN} after substitution"
    );
    out
}

fn find_rocm_llvm_bin() -> Option<PathBuf> {
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        let base = PathBuf::from(&rocm_path);
        let p = base.join("lib/llvm/bin");
        if p.is_dir() {
            return Some(p);
        }
        let p_bin = base.join("bin");
        if p_bin.is_dir() {
            return Some(p_bin);
        }
    }
    for cand in ["/opt/rocm/lib/llvm/bin", "/opt/rocm/bin"] {
        let p = PathBuf::from(cand);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

fn find_rocm_device_lib() -> Option<PathBuf> {
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        let base = PathBuf::from(&rocm_path);
        for sub in [
            "lib/llvm/amdgcn/bitcode",
            "amdgcn/bitcode",
            "lib/amdgcn/bitcode",
        ] {
            let p = base.join(sub);
            if p.is_dir() {
                return Some(p);
            }
        }
    }
    for cand in [
        "/opt/rocm/lib/llvm/amdgcn/bitcode",
        "/opt/rocm/amdgcn/bitcode",
        "/opt/rocm/lib/amdgcn/bitcode",
    ] {
        let p = PathBuf::from(cand);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// GPU context: compiland assembly, pinned compile, module cache, launch
// ---------------------------------------------------------------------------

/// Per-test GPU state: library, stream, arch, and a compiland cache keyed by
/// the canonical static identity plus defines (Spec 4 §3: shapes are baked).
struct GpuCtx {
    lib: Arc<HipLibrary>,
    stream: Stream,
    arch: String,
    target_dir: PathBuf,
    reference_dir: PathBuf,
    modules: RefCell<HashMap<String, Module>>,
}

impl GpuCtx {
    fn new(lib: Arc<HipLibrary>) -> Self {
        lib.set_device(0).expect("set_device(0) must succeed");
        let props = lib
            .get_device_properties(0)
            .expect("device 0 properties must be readable");
        let arch = match props.gcn_arch_name.find("gfx") {
            Some(idx) => {
                let candidate = &props.gcn_arch_name[idx..];
                let end = candidate
                    .find(|c: char| !c.is_alphanumeric())
                    .unwrap_or(candidate.len());
                candidate[..end].to_owned()
            }
            None => panic!(
                "device 0 arch {:?} has no gfx name; refusing to guess",
                props.gcn_arch_name
            ),
        };
        let stream = Stream::new(&lib).expect("HIP stream creation must succeed");
        let root = workspace_root();
        let target_dir = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("target"))
            .join("test-fixtures");
        std::fs::create_dir_all(&target_dir).expect("fixture dir must be writable");
        let reference_dir = root.join("kernels/reference");
        assert!(
            reference_dir.is_dir(),
            "kernels/reference must exist at {}",
            reference_dir.display()
        );
        println!(
            "T1 sampling on device 0: {} [arch {arch}, VRAM {} GiB]",
            props.name,
            props.total_global_mem / (1024 * 1024 * 1024)
        );
        Self {
            lib,
            stream,
            arch,
            target_dir,
            reference_dir,
            modules: RefCell::new(HashMap::new()),
        }
    }

    /// Assembles one compiland with every arg bound through the A3.2
    /// generated ABI: emitted struct + shared header + kernel body with the
    /// canonical struct name substituted. The body must not declare its own
    /// args struct (checked by the host binding test).
    fn assemble(&self, abi: &AbiStruct, body_file: &str, defines: &[(String, String)]) -> String {
        assemble_compiland(&self.reference_dir, abi, body_file, defines)
    }

    /// Compiles one compiland with the pinned flags and loads the module,
    /// caching by key. Panics (never skips) once a device exists: a present
    /// device with no compiler is a broken lane, not an absent one.
    fn module_for(&self, key: &str, source: &str) -> Module {
        if let Some(m) = self.modules.borrow().get(key) {
            return m.clone();
        }
        let llvm_bin = find_rocm_llvm_bin().unwrap_or_else(|| {
            panic!("HIP device present but ROCm clang++ not found; refusing to skip")
        });
        let clang = llvm_bin.join("clang++");
        let bundler = llvm_bin.join("clang-offload-bundler");
        assert!(clang.is_file(), "clang++ missing at {}", clang.display());
        assert!(
            bundler.is_file(),
            "clang-offload-bundler missing at {}",
            bundler.display()
        );

        let safe_key: String = key
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let src_path = self.target_dir.join(format!("{safe_key}.hip"));
        let bundle_path = self.target_dir.join(format!("{safe_key}.o"));
        let co_path = self.target_dir.join(format!("{safe_key}.co"));
        std::fs::write(&src_path, source).expect("compiland must be writable");

        let mut cmd = Command::new(&clang);
        cmd.args(PINNED_CXXFLAGS);
        cmd.arg(format!("--offload-arch={}", self.arch));
        cmd.arg("-o").arg(&bundle_path).arg(&src_path);
        if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
            cmd.arg(format!("--rocm-path={rocm_path}"));
        }
        if let Some(dev_lib) = find_rocm_device_lib() {
            cmd.arg(format!("--rocm-device-lib-path={}", dev_lib.display()));
        }
        // Defines are baked into the source header by `assemble`; no -D here
        // so the filed compiland in target/ matches what was compiled.
        let status = cmd.status().expect("clang++ invocation failed");
        assert!(status.success(), "pinned clang++ failed for {key}");

        let unbundle = Command::new(&bundler)
            .args([
                "--type=o".to_owned(),
                format!("--targets=hipv4-amdgcn-amd-amdhsa--{}", self.arch),
                format!("--input={}", bundle_path.display()),
                format!("--output={}", co_path.display()),
                "--unbundle".to_owned(),
            ])
            .status()
            .expect("bundler invocation failed");
        assert!(unbundle.success(), "bundler failed for {key}");

        let module = Module::load_file(&self.lib, &co_path).expect("module load must succeed");
        self.modules
            .borrow_mut()
            .insert(key.to_owned(), module.clone());
        module
    }

    /// Packs one by-value args struct from `AbiStruct` field offsets (Spec 4
    /// §7): 8-byte pointers at pointer fields, scalars at scalar fields. The
    /// backing is u64-aligned so device loads are valid.
    fn pack_args(
        abi: &AbiStruct,
        pointers: &[(&str, u64)],
        scalars_u32: &[(&str, u32)],
    ) -> Vec<u64> {
        fn find<'a>(abi: &'a AbiStruct, name: &str) -> &'a r9v_kgen::abi::AbiField {
            abi.fields
                .iter()
                .find(|f| f.name == name)
                .unwrap_or_else(|| panic!("ABI struct {} has no field {name}", abi.name))
        }
        let mut words = vec![0u64; abi.size.div_ceil(8)];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 8)
        };
        for (name, ptr) in pointers {
            let f = find(abi, name);
            assert!(
                matches!(f.ty, AbiType::Pointer { .. }),
                "field {name} must be a pointer in {}",
                abi.name
            );
            bytes[f.offset..f.offset + 8].copy_from_slice(&ptr.to_le_bytes());
        }
        for (name, val) in scalars_u32 {
            let f = find(abi, name);
            match f.ty {
                AbiType::Scalar(ScalarType::U32) | AbiType::Scalar(ScalarType::I32) => {}
                _ => panic!("field {name} must be a u32 scalar in {}", abi.name),
            }
            bytes[f.offset..f.offset + 4].copy_from_slice(&val.to_le_bytes());
        }
        words
    }

    fn alloc_filled(&self, bytes: &[u8]) -> DeviceBuffer {
        let mut buf = DeviceBuffer::allocate(&self.lib, bytes.len().max(1))
            .expect("device alloc must succeed");
        if !bytes.is_empty() {
            buf.copy_from_host(bytes).expect("H2D copy must succeed");
        }
        buf
    }

    fn read_back(&self, buf: &DeviceBuffer, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        buf.copy_to_host(&mut out).expect("D2H copy must succeed");
        out
    }

    /// Launches a by-value-struct kernel and synchronizes (Spec 14 §3).
    fn launch(&self, module: &Module, symbol: &str, grid: u32, block: u32, args_words: &mut [u64]) {
        let func = module
            .get_function(symbol)
            .unwrap_or_else(|_| panic!("symbol {symbol} must exist"));
        let mut args = [args_words.as_mut_ptr() as *mut std::ffi::c_void];
        unsafe {
            func.launch((grid, 1, 1), (block, 1, 1), 0, &self.stream, &mut args)
                .expect("kernel launch must succeed");
        }
        self.stream.synchronize().expect("stream sync must succeed");
    }
}

// ---------------------------------------------------------------------------
// Seeded fixtures (pure functions of shape + seed; Spec 4 §10)
// ---------------------------------------------------------------------------

/// Derives valid per-sequence sampling params from an RNG (Spec 1 §4.F).
fn derive_params(rng: &mut SeededRng, v: usize, allow_temp_zero: bool) -> SamplingParams {
    let mut pick = |n: u64| (rng.next_u64() % n) as usize;
    let temperature = match pick(if allow_temp_zero { 5 } else { 4 }) {
        0 if allow_temp_zero => 0.0,
        0 => 0.25,
        1 => 0.7,
        2 => 1.0,
        _ => 2.0,
    };
    let top_k = match pick(5) {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 5,
        _ => (v as u32).saturating_add(10),
    };
    let top_p = match pick(4) {
        0 => 0.15,
        1 => 0.5,
        2 => 0.85,
        _ => 1.0,
    };
    let min_p = match pick(4) {
        0 => 0.0,
        1 => 0.05,
        2 => 0.2,
        _ => 0.5,
    };
    let repetition_penalty = match pick(3) {
        0 => 1.0,
        1 => 1.25,
        _ => 0.8,
    };
    let presence_penalty = match pick(3) {
        0 => 0.0,
        1 => 0.5,
        _ => -0.3,
    };
    let frequency_penalty = match pick(3) {
        0 => 0.0,
        1 => 0.4,
        _ => -0.2,
    };
    let bias_count = pick(4);
    let mut logit_bias = Vec::with_capacity(bias_count);
    for _ in 0..bias_count {
        let token = (rng.next_u64() % v.max(1) as u64) as u32;
        let bias = ((rng.next_u64() % 4000) as f32 / 1000.0) - 2.0;
        logit_bias.push((token, bias));
    }
    SamplingParams {
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        logit_bias,
    }
}

/// Normalized probability rows from seeded logits via stable softmax.
fn derive_dist_rows(rng: &mut SeededRng, rows: usize, v: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * v);
    for _ in 0..rows {
        let logits = uniform_f32(rng, v, -3.0, 3.0);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut row = Vec::with_capacity(v);
        for &l in &logits {
            let e = (l - max).exp();
            row.push(e);
            sum += e;
        }
        let inv = 1.0 / sum;
        for e in row {
            out.push(e * inv);
        }
    }
    out
}

/// Fixture for one `logits_postprocess` run: `[S, q, V]` (Spec 1 §4.F).
struct PpFixture {
    s: usize,
    q: usize,
    v: usize,
    logits: Vec<f32>,
    params: Vec<SamplingParams>,
    history: Option<Vec<u32>>,
    mask_bool: Option<Vec<bool>>,
    seq_ids: Vec<u32>,
}

fn derive_pp(shape: &[usize], seed: u64) -> PpFixture {
    let (s, q, v) = (shape[0], shape[1], shape[2]);
    let mut rng = SeededRng::new(seed);
    let logits = uniform_f32(&mut rng, s * q * v, -4.0, 4.0);
    let params: Vec<SamplingParams> = (0..s).map(|_| derive_params(&mut rng, v, true)).collect();
    let history = if rng.next_u64().is_multiple_of(2) {
        let mut h = Vec::with_capacity(s * v);
        for _ in 0..s * v {
            h.push((rng.next_u64() % 6) as u32);
        }
        Some(h)
    } else {
        None
    };
    let mask_bool = if rng.next_u64().is_multiple_of(2) {
        let mut m = keep_mask(&mut rng, s * q * v, 0.85);
        // At least one allowed token per row (else the row must refuse).
        for row in 0..s * q {
            if !m[row * v..(row + 1) * v].iter().any(|&b| b) {
                m[row * v] = true;
            }
        }
        Some(m)
    } else {
        None
    };
    let seq_ids: Vec<u32> = (0..s)
        .map(|i| (rng.next_u64() % 1000) as u32 + (i as u32) * 7919)
        .collect();
    PpFixture {
        s,
        q,
        v,
        logits,
        params,
        history,
        mask_bool,
        seq_ids,
    }
}

/// Fixture for one `sample` run: probs `[S, V]` plus RNG words (Spec 1 §4.F).
struct SampleFixture {
    s: usize,
    v: usize,
    probs: Vec<f32>,
    rng_words: Vec<DeviceRngWords>,
    seq_ids: Vec<u32>,
}

fn derive_sample(shape: &[usize], seed: u64) -> SampleFixture {
    let (s, v) = (shape[0], shape[1]);
    let mut rng = SeededRng::new(seed);
    let probs = derive_dist_rows(&mut rng, s, v);
    let base_seed = rng.next_u64();
    let step = 7 + (rng.next_u64() % 3);
    let mut rng_words = Vec::with_capacity(s);
    let mut seq_ids = Vec::with_capacity(s);
    for i in 0..s {
        seq_ids.push((rng.next_u64() % 500) as u32 + (i as u32) * 104729);
        rng_words.push(DeviceRngWords {
            seed: base_seed ^ ((i as u64).wrapping_mul(0x9E3779B97F4A7C15)),
            step,
            draw: rng.next_u64() % 5,
        });
    }
    SampleFixture {
        s,
        v,
        probs,
        rng_words,
        seq_ids,
    }
}

/// Fixture for one `verify` run (Spec 1 §4.F, Spec 7 §4, §5).
struct VerifyFixture {
    s: usize,
    k: usize,
    v: usize,
    draft_tokens: Vec<u32>,
    draft_probs: Option<Vec<f32>>,
    target_probs: Vec<f32>,
    rng_words: Vec<DeviceRngWords>,
    seq_ids: Vec<u32>,
    parents: Option<Vec<i32>>,
    t_max: u32,
}

fn derive_tree_parents(rng: &mut SeededRng, k: usize) -> Vec<i32> {
    let mut parents = Vec::with_capacity(k);
    for j in 0..k {
        if j == 0 || rng.next_u64().is_multiple_of(3) {
            parents.push(-1);
        } else {
            parents.push((rng.next_u64() % j as u64) as i32);
        }
    }
    parents
}

fn derive_verify(shape: &[usize], seed: u64, tree: bool, has_draft: bool) -> VerifyFixture {
    let (s, k, v) = (shape[0], shape[1], shape[2]);
    let mut rng = SeededRng::new(seed);
    let mut draft_tokens = Vec::with_capacity(s * k);
    for _ in 0..s * k {
        draft_tokens.push((rng.next_u64() % v.max(1) as u64) as u32);
    }
    let draft_probs = if has_draft && k > 0 {
        Some(derive_dist_rows(&mut rng, s * k, v))
    } else {
        None
    };
    let target_probs = derive_dist_rows(&mut rng, s * (k + 1), v);
    let base_seed = rng.next_u64();
    let step = 11 + (rng.next_u64() % 3);
    let mut rng_words = Vec::with_capacity(s);
    let mut seq_ids = Vec::with_capacity(s);
    for i in 0..s {
        seq_ids.push((rng.next_u64() % 500) as u32 + (i as u32) * 104729);
        rng_words.push(DeviceRngWords {
            seed: base_seed ^ ((i as u64).wrapping_mul(0x9E3779B97F4A7C15)),
            step,
            draw: 0,
        });
    }
    let parents = if tree && k > 0 {
        Some(derive_tree_parents(&mut rng, k))
    } else {
        None
    };
    VerifyFixture {
        s,
        k,
        v,
        draft_tokens,
        draft_probs,
        target_probs,
        rng_words,
        seq_ids,
        parents,
        t_max: k as u32,
    }
}

// ---------------------------------------------------------------------------
// T0-probed device launchers: T0 validates AND oracles; the kernel must agree
// ---------------------------------------------------------------------------

/// Packs host `SamplingParams` into the device record + trailing bias pairs.
fn pack_device_params(params: &[SamplingParams]) -> Vec<u8> {
    let mut recs = Vec::with_capacity(params.len());
    let mut pairs: Vec<DeviceBiasPair> = Vec::new();
    for p in params {
        let bias_start = pairs.len() as u32;
        for &(token, bias) in &p.logit_bias {
            pairs.push(DeviceBiasPair { token, bias });
        }
        recs.push(DeviceSamplingParams {
            temperature: p.temperature,
            top_p: p.top_p,
            min_p: p.min_p,
            repetition_penalty: p.repetition_penalty,
            presence_penalty: p.presence_penalty,
            frequency_penalty: p.frequency_penalty,
            top_k: p.top_k,
            bias_start,
            bias_count: p.logit_bias.len() as u32,
            _pad: 0,
        });
    }
    let mut out = vec![0u8; recs.len() * 40];
    for (i, r) in recs.iter().enumerate() {
        let b = &mut out[i * 40..(i + 1) * 40];
        b[0..4].copy_from_slice(&r.temperature.to_bits().to_le_bytes());
        b[4..8].copy_from_slice(&r.top_p.to_bits().to_le_bytes());
        b[8..12].copy_from_slice(&r.min_p.to_bits().to_le_bytes());
        b[12..16].copy_from_slice(&r.repetition_penalty.to_bits().to_le_bytes());
        b[16..20].copy_from_slice(&r.presence_penalty.to_bits().to_le_bytes());
        b[20..24].copy_from_slice(&r.frequency_penalty.to_bits().to_le_bytes());
        b[24..28].copy_from_slice(&r.top_k.to_le_bytes());
        b[28..32].copy_from_slice(&r.bias_start.to_le_bytes());
        b[32..36].copy_from_slice(&r.bias_count.to_le_bytes());
    }
    for p in &pairs {
        out.extend_from_slice(&p.token.to_le_bytes());
        out.extend_from_slice(&p.bias.to_bits().to_le_bytes());
    }
    out
}

fn rng_words_to_bytes(words: &[DeviceRngWords]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 24);
    for w in words {
        out.extend_from_slice(&w.seed.to_le_bytes());
        out.extend_from_slice(&w.step.to_le_bytes());
        out.extend_from_slice(&w.draw.to_le_bytes());
    }
    out
}

fn rng_words_from_bytes(bytes: &[u8]) -> Vec<DeviceRngWords> {
    bytes
        .chunks_exact(24)
        .map(|c| DeviceRngWords {
            seed: u64::from_le_bytes(c[0..8].try_into().unwrap()),
            step: u64::from_le_bytes(c[8..16].try_into().unwrap()),
            draw: u64::from_le_bytes(c[16..24].try_into().unwrap()),
        })
        .collect()
}

fn r9v_bucket(n: usize) -> u32 {
    let mut b = 1u32;
    while b < n as u32 {
        b <<= 1;
    }
    b.min(4096)
}

fn kgen_error_to_t0(op: &'static str, e: r9v_kgen::KgenError) -> T0Error {
    T0Error::InvalidAttribute {
        op,
        attribute: "abi",
        reason: format!("A3.2 ABI construction failed (fixed pairing): {e:?}"),
    }
}

fn u32_len(op: &'static str, tensor: &'static str, v: usize) -> Result<u32, T0Error> {
    u32::try_from(v).map_err(|_| T0Error::ShapeLengthMismatch {
        op,
        tensor,
        expected: u32::MAX as usize,
        got: v,
        detail: "extent exceeds u32 device domain".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Device launchers (T0 probe validates + oracles; kernel must agree)
// ---------------------------------------------------------------------------

/// T0 validation probe for `logits_postprocess` (no device touch).
/// Every refusal the launcher reports comes from here, so the refusal set
/// cannot drift from T0.
fn t0_probe_pp(fix: &PpFixture) -> Result<(), T0Error> {
    let (s, q, v) = (fix.s, fix.q, fix.v);
    let mut probe = vec![0.0f32; s * q * v];
    logits_postprocess(
        &fix.logits,
        s,
        q,
        v,
        &fix.params,
        fix.history.as_deref(),
        fix.mask_bool.as_deref(),
        &mut probe,
    )?;
    Ok(())
}

/// Runs T0 `logits_postprocess` as validator/oracle, then the T1 kernel.
/// Refusals come from T0 alone, so the refusal set cannot drift.
fn run_pp(ctx: &GpuCtx, fix: &PpFixture) -> Result<Vec<f32>, T0Error> {
    t0_probe_pp(fix)?;
    let (s, q, v) = (fix.s, fix.q, fix.v);

    let v_u32 = u32_len("logits_postprocess", "V", v)?;
    let op_static =
        OpStatic::Sampling(SamplingStatic::LogitsPostprocess(LogitsPostprocessStatic {
            s_bucket: r9v_bucket(s),
            v: v_u32,
            q_bucket: r9v_bucket(q),
            has_history_counts: fix.history.is_some(),
            has_grammar_mask: fix.mask_bool.is_some(),
        }));
    let abi = abi_for_op(OpId::LogitsPostprocess, &op_static)
        .map_err(|e| kgen_error_to_t0("logits_postprocess", e))?;
    let defines = vec![
        ("R9V_V".to_string(), v.to_string()),
        (
            "R9V_PP_HAS_HISTORY".to_string(),
            if fix.history.is_some() {
                "1".to_string()
            } else {
                "0".to_string()
            },
        ),
        (
            "R9V_PP_HAS_MASK".to_string(),
            if fix.mask_bool.is_some() {
                "1".to_string()
            } else {
                "0".to_string()
            },
        ),
    ];
    let source = ctx.assemble(&abi, PP_HIP, &defines);
    let key = format!(
        "pp_s{}_v{v}_q{}_h{}_m{}",
        r9v_bucket(s),
        r9v_bucket(q),
        fix.history.is_some() as u8,
        fix.mask_bool.is_some() as u8
    );
    let module = ctx.module_for(&key, &source);

    let d_logits = ctx.alloc_filled(&f32_to_bytes(&fix.logits));
    let params_bytes = pack_device_params(&fix.params);
    let d_params = ctx.alloc_filled(&params_bytes);
    let d_history = fix
        .history
        .as_ref()
        .map(|h| ctx.alloc_filled(&u32_to_bytes(h)));
    let mask_u8: Option<Vec<u8>> = fix
        .mask_bool
        .as_ref()
        .map(|m| m.iter().map(|&b| u8::from(b)).collect());
    let d_mask = mask_u8.as_ref().map(|m| ctx.alloc_filled(m));
    let d_probs = ctx.alloc_filled(&vec![0u8; s * q * v * 4]);
    let npad = next_pow2(v);
    let mask_bytes = v.checked_add(7).expect("workspace mask alignment overflow") & !7;
    let ws_row = npad
        .checked_mul(8)
        .and_then(|pairs| pairs.checked_add(mask_bytes))
        .expect("workspace row size overflow");
    let d_ws = ctx.alloc_filled(&vec![0u8; s * q * ws_row]);

    let ptr = |o: &Option<DeviceBuffer>| o.as_ref().map_or(0u64, |b| b.as_ptr() as u64);
    let mut args = GpuCtx::pack_args(
        &abi,
        &[
            ("logits", d_logits.as_ptr() as u64),
            ("params", d_params.as_ptr() as u64),
            ("history_counts", ptr(&d_history)),
            ("grammar_mask", ptr(&d_mask)),
            ("probs", d_probs.as_ptr() as u64),
            ("workspace", d_ws.as_ptr() as u64),
        ],
        &[("s", s as u32), ("q", q as u32)],
    );
    let grid = ((s * q).min(1024)) as u32;
    ctx.launch(&module, "t1_logits_postprocess", grid, 256, &mut args);
    Ok(bytes_to_f32(&ctx.read_back(&d_probs, s * q * v * 4)))
}

struct SampleRunOut {
    tokens: Vec<u32>,
    rng_after: Vec<DeviceRngWords>,
}

fn clone_rng_states(
    op: &'static str,
    words: &[DeviceRngWords],
    seq_ids: &[u32],
) -> Result<Vec<RngState>, T0Error> {
    use r9v_common::{SeqId, StepId};
    let mut clones = Vec::with_capacity(words.len());
    for (w, &seq) in words.iter().zip(seq_ids.iter()) {
        let draw = u32::try_from(w.draw).map_err(|_| T0Error::ShapeLengthMismatch {
            op,
            tensor: "rng_state.draw",
            expected: u32::MAX as usize,
            got: usize::MAX,
            detail: "draw index exceeds u32 Philox domain".to_string(),
        })?;
        clones.push(RngState::with_draw(
            w.seed,
            SeqId::new(seq as u64),
            StepId::new(w.step),
            draw,
        )?);
    }
    Ok(clones)
}

/// T0 validation probe for `sample` (no device touch): expected tokens and
/// post-call draw indices from cloned states.
fn t0_probe_sample(fix: &SampleFixture) -> Result<(Vec<u32>, Vec<u32>), T0Error> {
    let (s, v) = (fix.s, fix.v);
    let mut clones = clone_rng_states("sample", &fix.rng_words, &fix.seq_ids)?;
    let expected_tokens = sample(&fix.probs, s, v, &mut clones)?;
    Ok((
        expected_tokens,
        clones.iter().map(|r| r.draw_index()).collect(),
    ))
}

/// Runs T0 `sample` on cloned RNG states (validator + oracle), then the kernel.
fn run_sample(ctx: &GpuCtx, fix: &SampleFixture) -> Result<SampleRunOut, T0Error> {
    let (expected_tokens, expected_draws) = t0_probe_sample(fix)?;
    let (s, v) = (fix.s, fix.v);

    let v_u32 = u32_len("sample", "V", v)?;
    let op_static = OpStatic::Sampling(SamplingStatic::Sample(SampleStatic {
        s_bucket: r9v_bucket(s),
        v: v_u32,
        rng: RngAlgorithm::Philox4x32,
    }));
    let abi = abi_for_op(OpId::Sample, &op_static).map_err(|e| kgen_error_to_t0("sample", e))?;
    let defines = vec![("R9V_V".to_string(), v.to_string())];
    let source = ctx.assemble(&abi, SAMPLE_HIP, &defines);
    let key = format!("sample_s{}_v{v}", r9v_bucket(s));
    let module = ctx.module_for(&key, &source);

    let d_probs = ctx.alloc_filled(&f32_to_bytes(&fix.probs));
    let d_rng = ctx.alloc_filled(&rng_words_to_bytes(&fix.rng_words));
    let d_seq = ctx.alloc_filled(&u32_to_bytes(&fix.seq_ids));
    let d_tokens = ctx.alloc_filled(&vec![0u8; s * 4]);

    let mut args = GpuCtx::pack_args(
        &abi,
        &[
            ("probs", d_probs.as_ptr() as u64),
            ("rng_state", d_rng.as_ptr() as u64),
            ("seq_ids", d_seq.as_ptr() as u64),
            ("tokens", d_tokens.as_ptr() as u64),
        ],
        &[("s", s as u32)],
    );
    let grid = (s.min(1024)) as u32;
    ctx.launch(&module, "t1_sample", grid, 128, &mut args);

    let tokens = bytes_to_u32(&ctx.read_back(&d_tokens, s * 4));
    let rng_after = rng_words_from_bytes(&ctx.read_back(&d_rng, s * 24));
    assert_eq!(
        tokens, expected_tokens,
        "device tokens must equal T0 oracle"
    );
    for (i, (w, &d)) in rng_after.iter().zip(expected_draws.iter()).enumerate() {
        assert_eq!(w.draw, d as u64, "seq {i} draw advance must match T0");
        assert_eq!(w.seed, fix.rng_words[i].seed, "seq {i} seed immutable");
        assert_eq!(w.step, fix.rng_words[i].step, "seq {i} step immutable");
    }
    Ok(SampleRunOut { tokens, rng_after })
}

struct VerifyRunOut {
    accepted: Vec<u32>,
    accept_len: Vec<u32>,
    rng_after: Vec<DeviceRngWords>,
}

fn build_tree_mask(fix: &VerifyFixture) -> Result<Option<TreeMask>, T0Error> {
    fix.parents
        .as_ref()
        .map(|p| {
            TreeMask::new(
                p.clone(),
                fix.t_max,
                vec![false; fix.k * fix.t_max as usize],
            )
            .map_err(T0Error::Ir)
        })
        .transpose()
}

/// T0 validation probe for `verify` (no device touch): expected output and
/// post-call draw indices from cloned states. The device local path cap is
/// checked first because T0 accepts wider trees than the kernel stages.
fn t0_probe_verify(
    fix: &VerifyFixture,
    method: &VerifyMethod,
) -> Result<(r9v_t0::VerifyOutput, Vec<u32>), T0Error> {
    let (s, k, v) = (fix.s, fix.k, fix.v);
    if k > DEVICE_MAX_K {
        return Err(T0Error::ShapeLengthMismatch {
            op: "verify",
            tensor: "k",
            expected: DEVICE_MAX_K,
            got: k,
            detail: "draft length exceeds device local path cap".to_string(),
        });
    }
    let tree = build_tree_mask(fix)?;
    let mut clones = clone_rng_states("verify", &fix.rng_words, &fix.seq_ids)?;
    // Draw-base overflow is checked by T0 here (same refusal the kernel relies on).
    let expected = verify(
        &fix.draft_tokens,
        fix.draft_probs.as_deref(),
        &fix.target_probs,
        s,
        k,
        v,
        method,
        &mut clones,
        tree.as_ref(),
    )?;
    Ok((expected, clones.iter().map(|r| r.draw_index()).collect()))
}

/// Runs T0 `verify` on cloned RNG states (validator + oracle), then the kernel.
fn run_verify(
    ctx: &GpuCtx,
    fix: &VerifyFixture,
    method: &VerifyMethod,
) -> Result<VerifyRunOut, T0Error> {
    let (expected, expected_draws) = t0_probe_verify(fix, method)?;
    let (s, k, v) = (fix.s, fix.k, fix.v);

    let v_u32 = u32_len("verify", "V", v)?;
    let method_static = VerifyMethodStatic::from_ir(method);
    let op_static = OpStatic::Sampling(SamplingStatic::Verify(VerifyStatic {
        s_bucket: r9v_bucket(s),
        v: v_u32,
        q_bucket: r9v_bucket(k + 1),
        method: method_static,
        tree: fix.parents.is_some(),
        has_draft_probs: fix.draft_probs.is_some(),
    }));
    let abi = abi_for_op(OpId::Verify, &op_static).map_err(|e| kgen_error_to_t0("verify", e))?;
    let (method_id, eps_bits, delta_bits) = match method_static {
        VerifyMethodStatic::Rejection => (0u32, 0u32, 0u32),
        VerifyMethodStatic::Greedy => (1u32, 0u32, 0u32),
        VerifyMethodStatic::TypicalAcceptance {
            eps_bits,
            delta_bits,
        } => (2u32, eps_bits, delta_bits),
    };
    let defines = vec![
        ("R9V_V".to_string(), v.to_string()),
        ("R9V_VERIFY_METHOD".to_string(), method_id.to_string()),
        (
            "R9V_VERIFY_TREE".to_string(),
            if fix.parents.is_some() {
                "1".to_string()
            } else {
                "0".to_string()
            },
        ),
        (
            "R9V_HAS_DRAFT_PROBS".to_string(),
            if fix.draft_probs.is_some() {
                "1".to_string()
            } else {
                "0".to_string()
            },
        ),
        ("R9V_VERIFY_EPS_BITS".to_string(), eps_bits.to_string()),
        ("R9V_VERIFY_DELTA_BITS".to_string(), delta_bits.to_string()),
    ];
    let source = ctx.assemble(&abi, VERIFY_HIP, &defines);
    let key = format!(
        "verify_s{}_v{v}_k{k}_m{method_id}_t{}_d{}",
        r9v_bucket(s),
        fix.parents.is_some() as u8,
        fix.draft_probs.is_some() as u8
    );
    let module = ctx.module_for(&key, &source);

    let d_draft = ctx.alloc_filled(&u32_to_bytes(&fix.draft_tokens));
    let d_draft_probs = fix
        .draft_probs
        .as_ref()
        .map(|p| ctx.alloc_filled(&f32_to_bytes(p)));
    let d_target = ctx.alloc_filled(&f32_to_bytes(&fix.target_probs));
    let d_rng = ctx.alloc_filled(&rng_words_to_bytes(&fix.rng_words));
    let d_seq = ctx.alloc_filled(&u32_to_bytes(&fix.seq_ids));
    let parents_u32: Option<Vec<u8>> = fix.parents.as_ref().map(|p| {
        let mut b = Vec::with_capacity(p.len() * 4);
        for &v in p {
            b.extend_from_slice(&(v as u32).to_le_bytes());
        }
        b
    });
    let d_parents = parents_u32.as_ref().map(|b| ctx.alloc_filled(b));
    let d_ancestors = fix
        .parents
        .as_ref()
        .map(|_| ctx.alloc_filled(&vec![0u8; k * fix.t_max as usize]));
    let d_accepted = ctx.alloc_filled(&vec![0u8; s * (k + 1) * 4]);
    let d_accept_len = ctx.alloc_filled(&vec![0u8; s * 4]);

    let ptr = |o: &Option<DeviceBuffer>| o.as_ref().map_or(0u64, |b| b.as_ptr() as u64);
    let mut pointers: Vec<(&str, u64)> = vec![
        ("draft_tokens", d_draft.as_ptr() as u64),
        ("draft_probs", ptr(&d_draft_probs)),
        ("target_probs", d_target.as_ptr() as u64),
        ("rng_state", d_rng.as_ptr() as u64),
        ("seq_ids", d_seq.as_ptr() as u64),
        ("accepted", d_accepted.as_ptr() as u64),
        ("accept_len", d_accept_len.as_ptr() as u64),
    ];
    if fix.parents.is_some() {
        pointers.push(("tree_parents", ptr(&d_parents)));
        pointers.push(("tree_ancestors", ptr(&d_ancestors)));
    }
    let mut args = GpuCtx::pack_args(&abi, &pointers, &[("s", s as u32), ("k", k as u32)]);
    let grid = (s.min(1024)) as u32;
    ctx.launch(&module, "t1_verify", grid, 64, &mut args);

    let accepted = bytes_to_u32(&ctx.read_back(&d_accepted, s * (k + 1) * 4));
    let accept_len = bytes_to_u32(&ctx.read_back(&d_accept_len, s * 4));
    let rng_after = rng_words_from_bytes(&ctx.read_back(&d_rng, s * 24));
    assert_eq!(
        accepted, expected.accepted,
        "device accepted must equal T0 oracle"
    );
    assert_eq!(
        accept_len, expected.accept_len,
        "device accept_len must equal T0 oracle"
    );
    for (i, (w, &d)) in rng_after.iter().zip(expected_draws.iter()).enumerate() {
        assert_eq!(w.draw, d as u64, "seq {i} draw advance must match T0");
    }
    Ok(VerifyRunOut {
        accepted,
        accept_len,
        rng_after,
    })
}

// ---------------------------------------------------------------------------
// Pinned fixtures for batch invariance (Spec 1 §6.1)
// ---------------------------------------------------------------------------

/// Fixed seed for the pinned logical row/sequence (same bytes in every mode).
const PINNED_SEED: u64 = 0xA360_6E12_3456_7890;

/// Forces option presence on a PP fixture (all-valid defaults when added).
fn force_pp_presence(fix: &mut PpFixture, history: bool, mask: bool) {
    if history && fix.history.is_none() {
        fix.history = Some(vec![0u32; fix.s * fix.v]);
    }
    if !history {
        fix.history = None;
    }
    if mask && fix.mask_bool.is_none() {
        fix.mask_bool = Some(vec![true; fix.s * fix.q * fix.v]);
    }
    if !mask {
        fix.mask_bool = None;
    }
}

/// Overwrites sequence `r` of a PP fixture with the single-seq pinned fixture.
/// Both fixtures must carry the same option presence (enforced by
/// `derive_pp_pinned`).
fn pin_pp_row(fix: &mut PpFixture, pinned: &PpFixture, r: usize) {
    let (q, v) = (fix.q, fix.v);
    assert_eq!(fix.history.is_some(), pinned.history.is_some());
    assert_eq!(fix.mask_bool.is_some(), pinned.mask_bool.is_some());
    for j in 0..q {
        fix.logits[(r * q + j) * v..(r * q + j + 1) * v]
            .copy_from_slice(&pinned.logits[j * v..(j + 1) * v]);
        if let (Some(dst), Some(src)) = (fix.mask_bool.as_mut(), pinned.mask_bool.as_ref()) {
            dst[(r * q + j) * v..(r * q + j + 1) * v].copy_from_slice(&src[j * v..(j + 1) * v]);
        }
    }
    fix.params[r] = pinned.params[0].clone();
    if let (Some(dst), Some(src)) = (fix.history.as_mut(), pinned.history.as_ref()) {
        dst[r * v..(r + 1) * v].copy_from_slice(&src[..v]);
    }
    fix.seq_ids[r] = pinned.seq_ids[0];
}

/// Derives a PP fixture with sequence `row` pinned to the logical identity.
/// Option presence is decided once by the pinned seed so alone/padded/
/// embedded compile the same static.
fn derive_pp_pinned(shape: &[usize], row: usize) -> PpFixture {
    let mut presence = SeededRng::new(PINNED_SEED ^ 0xFFF0);
    let want_history = presence.next_u64().is_multiple_of(2);
    let want_mask = presence.next_u64().is_multiple_of(2);
    let mut fix = derive_pp(shape, 0xA360_6E12_0000_0001);
    force_pp_presence(&mut fix, want_history, want_mask);
    let mut pinned = derive_pp(&[1, shape[1], shape[2]], PINNED_SEED);
    force_pp_presence(&mut pinned, want_history, want_mask);
    pin_pp_row(&mut fix, &pinned, row);
    fix
}

/// Overwrites sequence `r` of a sample fixture with the pinned single-seq fixture.
fn pin_sample_row(fix: &mut SampleFixture, pinned: &SampleFixture, r: usize) {
    let v = fix.v;
    fix.probs[r * v..(r + 1) * v].copy_from_slice(&pinned.probs[..v]);
    fix.rng_words[r] = pinned.rng_words[0];
    fix.seq_ids[r] = pinned.seq_ids[0];
}

fn derive_sample_pinned(shape: &[usize], row: usize) -> SampleFixture {
    let mut fix = derive_sample(shape, 0xA360_6E12_0000_0002);
    let pinned = derive_sample(&[1, shape[1]], PINNED_SEED);
    pin_sample_row(&mut fix, &pinned, row);
    fix
}

/// Overwrites sequence `r` of a verify fixture with the pinned single-seq fixture.
fn pin_verify_row(fix: &mut VerifyFixture, pinned: &VerifyFixture, r: usize) {
    let (k, v) = (fix.k, fix.v);
    fix.draft_tokens[r * k..(r + 1) * k].copy_from_slice(&pinned.draft_tokens[..k]);
    if let (Some(dst), Some(src)) = (fix.draft_probs.as_mut(), pinned.draft_probs.as_ref()) {
        dst[r * k * v..(r + 1) * k * v].copy_from_slice(&src[..k * v]);
    }
    fix.target_probs[r * (k + 1) * v..(r + 1) * (k + 1) * v]
        .copy_from_slice(&pinned.target_probs[..(k + 1) * v]);
    fix.rng_words[r] = pinned.rng_words[0];
    fix.seq_ids[r] = pinned.seq_ids[0];
}

fn derive_verify_pinned(shape: &[usize], row: usize, tree: bool, has_draft: bool) -> VerifyFixture {
    let mut fix = derive_verify(shape, 0xA360_6E12_0000_0003, tree, has_draft);
    let pinned = derive_verify(&[1, shape[1], shape[2]], PINNED_SEED, tree, has_draft);
    if tree {
        fix.parents = pinned.parents.clone();
    }
    pin_verify_row(&mut fix, &pinned, row);
    fix
}

// ---------------------------------------------------------------------------
// GateCase: T1 logits_postprocess (Spec 4 §10 gate 1..4 vs T0/f64)
// ---------------------------------------------------------------------------

struct PpLast {
    device_probs: Vec<f32>,
}

/// Stashed run identity: (shape, seed, pinned row, illegal index).
type RunKey = (Vec<usize>, u64, Option<usize>, Option<usize>);

struct PpCase {
    ctx: std::rc::Rc<GpuCtx>,
    key: RefCell<Option<RunKey>>,
    last: RefCell<Option<PpLast>>,
}

impl PpCase {
    fn new(ctx: std::rc::Rc<GpuCtx>) -> Self {
        Self {
            ctx,
            key: RefCell::new(None),
            last: RefCell::new(None),
        }
    }

    fn fixture_for(&self) -> PpFixture {
        let (shape, seed, row, illegal) = self.key.borrow().clone().expect("build before execute");
        if let Some(idx) = illegal {
            return pp_illegal_fixture(idx);
        }
        match row {
            Some(r) => derive_pp_pinned(&shape, r),
            None => derive_pp(&shape, seed),
        }
    }
}

/// Explicit hostile PP fixtures, all of which T0 must refuse (Spec 1 §4.F).
fn pp_illegal_fixture(idx: usize) -> PpFixture {
    let mut fix = derive_pp(&[2, 2, 32], 0xA360_6E12_0BAD_0000 + idx as u64);
    match idx {
        0 => fix.logits[0] = f32::NAN,
        1 => fix.logits[5] = f32::INFINITY,
        2 => {
            let mut m = vec![true; 2 * 2 * 32];
            for b in m[..32].iter_mut() {
                *b = false;
            }
            fix.mask_bool = Some(m);
        }
        3 => fix.params[0].temperature = -1.0,
        4 => fix.params[0].logit_bias.push((32, 1.0)),
        5 => {
            fix.logits.pop();
        }
        _ => fix.logits.clear(),
    }
    fix
}

impl GateCase for PpCase {
    fn op_name(&self) -> &'static str {
        "logits_postprocess"
    }

    fn tolerance(&self) -> Tolerance {
        Tolerance::for_op("logits_postprocess").expect("tolerance row must exist")
    }

    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        vec![
            vec![1, 1, 8],
            vec![2, 3, 32],
            vec![5, 2, 257],
            vec![3, 1, 1024],
            vec![4, 4, 64],
            vec![33, 1, 32],
            vec![1, 5, 128],
            vec![8, 1, 4096],
        ]
    }

    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![
            vec![3, 2, 100],
            vec![2, 3, 1003],
            vec![7, 1, 3000],
            vec![64, 1, 16],
        ]
    }

    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        *self.key.borrow_mut() = Some((shape.to_vec(), seed, None, None));
        let fix = derive_pp(shape, seed);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[fix.s * fix.q * fix.v], &fix.logits)],
            vec![TypedBuffer::from_f32(
                &[fix.s * fix.q * fix.v],
                &vec![0.0f32; fix.s * fix.q * fix.v],
            )],
        ))
    }

    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        *self.key.borrow_mut() = Some((shape.to_vec(), PINNED_SEED, Some(row), None));
        let fix = derive_pp_pinned(shape, row);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[fix.s * fix.q * fix.v], &fix.logits)],
            vec![TypedBuffer::from_f32(
                &[fix.s * fix.q * fix.v],
                &vec![0.0f32; fix.s * fix.q * fix.v],
            )],
            row,
        ))
    }

    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), T0Error> {
        let fix = self.fixture_for();
        let device_probs = run_pp(&self.ctx, &fix)?;
        *self.last.borrow_mut() = Some(PpLast {
            device_probs: device_probs.clone(),
        });
        buffers.outputs = vec![TypedBuffer::from_f32(
            &[fix.s * fix.q * fix.v],
            &device_probs,
        )];
        Ok(())
    }

    fn verify(&self, buffers: &GateBuffers) -> Result<(), HarnessError> {
        let (shape, seed, row, _) = self.key.borrow().clone().expect("build before verify");
        let fix = match row {
            Some(r) => derive_pp_pinned(&shape, r),
            None => derive_pp(&shape, seed),
        };
        let last = self.last.borrow();
        let device = &last.as_ref().expect("execute before verify").device_probs;
        // Independent oracle: the f64 pipeline, never T0 f32 (Spec 4 §10).
        let logits_f64: Vec<f64> = fix.logits.iter().map(|&x| x as f64).collect();
        let expected = logits_postprocess_f64_reference(
            &logits_f64,
            fix.s,
            fix.q,
            fix.v,
            &fix.params,
            fix.history.as_deref(),
            fix.mask_bool.as_deref(),
        );
        check_f32_against_f64(self.tolerance(), device, &expected, "pp golden")?;
        let _ = buffers;
        Ok(())
    }

    fn output_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let last = self.last.borrow();
        let d = &last
            .as_ref()
            .expect("execute before output_bytes")
            .device_probs;
        Ok(f32_to_bytes(d))
    }

    fn logical_bytes(&self, buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let (shape, _, row_opt, _) = self
            .key
            .borrow()
            .clone()
            .expect("build before logical_bytes");
        let row = row_opt.expect("pinned run must carry a row");
        let last = self.last.borrow();
        let d = &last
            .as_ref()
            .expect("execute before logical_bytes")
            .device_probs;
        let (q, v) = (shape[1], shape[2]);
        let start = row * q * v;
        let _ = buffers;
        Ok(f32_to_bytes(&d[start..start + q * v]))
    }

    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1, 2, 64],
            padded: vec![4, 2, 64],
            embedded: vec![6, 2, 64],
            row_alone: 0,
            row: 2,
        }
    }

    fn illegal_count(&self) -> usize {
        6
    }

    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        if index >= 6 {
            return Err(HarnessError::FuzzVerdict {
                context: "logits_postprocess illegal".to_owned(),
                detail: format!("illegal index {index} out of range"),
            });
        }
        *self.key.borrow_mut() = Some((
            vec![2, 2, 32],
            0xA360_6E12_0BAD_0000 + index as u64,
            None,
            Some(index),
        ));
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[1], &[0.0])],
            vec![TypedBuffer::from_f32(&[1], &[0.0])],
        ))
    }
}

// ---------------------------------------------------------------------------
// GateCase: T1 sample (exact vs T0; Spec 4 §10)
// ---------------------------------------------------------------------------

struct SampleLast {
    tokens: Vec<u32>,
    rng_after: Vec<DeviceRngWords>,
}

struct SampleCase {
    ctx: std::rc::Rc<GpuCtx>,
    key: RefCell<Option<RunKey>>,
    last: RefCell<Option<SampleLast>>,
}

impl SampleCase {
    fn new(ctx: std::rc::Rc<GpuCtx>) -> Self {
        Self {
            ctx,
            key: RefCell::new(None),
            last: RefCell::new(None),
        }
    }

    fn fixture_for(&self) -> SampleFixture {
        let (shape, seed, row, illegal) = self.key.borrow().clone().expect("build before execute");
        if let Some(idx) = illegal {
            return sample_illegal_fixture(idx);
        }
        match row {
            Some(r) => derive_sample_pinned(&shape, r),
            None => derive_sample(&shape, seed),
        }
    }
}

/// Explicit hostile sample fixtures, all refused by T0 (Spec 1 §4.F).
fn sample_illegal_fixture(idx: usize) -> SampleFixture {
    let mut fix = derive_sample(&[3, 64], 0xA360_6E12_0BAD_1000 + idx as u64);
    match idx {
        0 => {
            for p in fix.probs[..64].iter_mut() {
                *p *= 2.0;
            }
        }
        1 => fix.probs[3] = -0.1,
        2 => fix.probs[7] = f32::NAN,
        3 => fix.rng_words[1].draw = u32::MAX as u64,
        _ => fix.probs.clear(),
    }
    fix
}

fn rng_bytes_of(words: &[DeviceRngWords]) -> Vec<u8> {
    rng_words_to_bytes(words)
}

/// RNG words as six u32 lanes per sequence for `TypedBuffer` outputs.
fn rng_words_to_u32(words: &[DeviceRngWords]) -> Vec<u32> {
    let mut out = Vec::with_capacity(words.len() * 6);
    for w in words {
        out.push(w.seed as u32);
        out.push((w.seed >> 32) as u32);
        out.push(w.step as u32);
        out.push((w.step >> 32) as u32);
        out.push(w.draw as u32);
        out.push((w.draw >> 32) as u32);
    }
    out
}

impl GateCase for SampleCase {
    fn op_name(&self) -> &'static str {
        "sample"
    }

    fn tolerance(&self) -> Tolerance {
        Tolerance::for_op("sample").expect("tolerance row must exist")
    }

    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        vec![
            vec![1, 8],
            vec![4, 257],
            vec![2, 1024],
            vec![33, 64],
            vec![8, 4096],
            vec![5, 32],
        ]
    }

    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        vec![vec![3, 100], vec![64, 16], vec![7, 3000]]
    }

    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        *self.key.borrow_mut() = Some((shape.to_vec(), seed, None, None));
        let fix = derive_sample(shape, seed);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[fix.s * fix.v], &fix.probs)],
            vec![TypedBuffer::from_u32(&[fix.s], &vec![0u32; fix.s])],
        ))
    }

    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        *self.key.borrow_mut() = Some((shape.to_vec(), PINNED_SEED, Some(row), None));
        let fix = derive_sample_pinned(shape, row);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_f32(&[fix.s * fix.v], &fix.probs)],
            vec![TypedBuffer::from_u32(&[fix.s], &vec![0u32; fix.s])],
            row,
        ))
    }

    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), T0Error> {
        let fix = self.fixture_for();
        let out = run_sample(&self.ctx, &fix)?;
        *self.last.borrow_mut() = Some(SampleLast {
            tokens: out.tokens.clone(),
            rng_after: out.rng_after.clone(),
        });
        buffers.outputs = vec![
            TypedBuffer::from_u32(&[fix.s], &out.tokens),
            TypedBuffer::from_u32(&[fix.s * 6], &rng_words_to_u32(&out.rng_after)),
        ];
        Ok(())
    }

    fn verify(&self, _buffers: &GateBuffers) -> Result<(), HarnessError> {
        // Oracle: T0 `sample` on the fixture inputs with pre-launch RNG state
        // (independent Rust path vs the HIP kernel; exact discrete contract).
        let (shape, seed, row, _) = self.key.borrow().clone().expect("build before verify");
        let fix = match row {
            Some(r) => derive_sample_pinned(&shape, r),
            None => derive_sample(&shape, seed),
        };
        let (expected, expected_draws) = t0_probe_sample(&fix).map_err(HarnessError::T0)?;
        let last = self.last.borrow();
        let got = last.as_ref().expect("execute before verify");
        if got.tokens != expected {
            return Err(HarnessError::FuzzVerdict {
                context: "sample golden".to_owned(),
                detail: "device tokens != T0".to_owned(),
            });
        }
        let expected_draws: Vec<u64> = expected_draws.iter().map(|&d| d as u64).collect();
        let got_draws: Vec<u64> = got.rng_after.iter().map(|w| w.draw).collect();
        if got_draws != expected_draws {
            return Err(HarnessError::FuzzVerdict {
                context: "sample draw accounting".to_owned(),
                detail: format!("device draws {got_draws:?} != T0 {expected_draws:?}"),
            });
        }
        Ok(())
    }

    fn output_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let last = self.last.borrow();
        let got = last.as_ref().expect("execute before output_bytes");
        let mut out = u32_to_bytes(&got.tokens);
        out.extend_from_slice(&rng_bytes_of(&got.rng_after));
        Ok(out)
    }

    fn logical_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let (_, _, row_opt, _) = self
            .key
            .borrow()
            .clone()
            .expect("build before logical_bytes");
        let row = row_opt.expect("pinned run must carry a row");
        let last = self.last.borrow();
        let got = last.as_ref().expect("execute before logical_bytes");
        let mut out = u32_to_bytes(&got.tokens[row..row + 1]);
        out.extend_from_slice(&rng_bytes_of(&got.rng_after[row..row + 1]));
        Ok(out)
    }

    fn batch_rows(&self) -> BatchRows {
        BatchRows {
            alone: vec![1, 64],
            padded: vec![4, 64],
            embedded: vec![6, 64],
            row_alone: 0,
            row: 2,
        }
    }

    fn illegal_count(&self) -> usize {
        4
    }

    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        if index >= 4 {
            return Err(HarnessError::FuzzVerdict {
                context: "sample illegal".to_owned(),
                detail: format!("illegal index {index} out of range"),
            });
        }
        *self.key.borrow_mut() = Some((
            vec![3, 64],
            0xA360_6E12_0BAD_1000 + index as u64,
            None,
            Some(index),
        ));
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_f32(&[1], &[0.0])],
            vec![TypedBuffer::from_u32(&[1], &[0])],
        ))
    }
}

// ---------------------------------------------------------------------------
// GateCase: T1 verify (exact vs T0, all methods × tree; Spec 7 §4, §5)
// ---------------------------------------------------------------------------

struct VerifyLast {
    accepted: Vec<u32>,
    accept_len: Vec<u32>,
    rng_after: Vec<DeviceRngWords>,
}

struct VerifyCase {
    ctx: std::rc::Rc<GpuCtx>,
    method: VerifyMethod,
    tree: bool,
    has_draft: bool,
    key: RefCell<Option<RunKey>>,
    last: RefCell<Option<VerifyLast>>,
}

impl VerifyCase {
    fn new(ctx: std::rc::Rc<GpuCtx>, method: VerifyMethod, tree: bool, has_draft: bool) -> Self {
        Self {
            ctx,
            method,
            tree,
            has_draft,
            key: RefCell::new(None),
            last: RefCell::new(None),
        }
    }

    fn fixture_for(&self) -> VerifyFixture {
        let (shape, seed, row, illegal) = self.key.borrow().clone().expect("build before execute");
        if let Some(idx) = illegal {
            return verify_illegal_fixture(idx, self.tree);
        }
        match row {
            Some(r) => derive_verify_pinned(&shape, r, self.tree, self.has_draft),
            None => derive_verify(&shape, seed, self.tree, self.has_draft),
        }
    }

    fn name(&self) -> String {
        let m = match &self.method {
            VerifyMethod::Rejection => "rejection",
            VerifyMethod::Greedy => "greedy",
            VerifyMethod::TypicalAcceptance { .. } => "typical",
        };
        format!(
            "verify_{m}_tree{}_draft{}",
            self.tree as u8, self.has_draft as u8
        )
    }
}

/// Explicit hostile verify fixtures, all refused by T0 or the device cap.
fn verify_illegal_fixture(idx: usize, tree: bool) -> VerifyFixture {
    let (s, k, v) = (2, 3, 32);
    let mut fix = derive_verify(&[s, k, v], 0xA360_6E12_0BAD_2000 + idx as u64, tree, true);
    match idx {
        0 => fix.draft_tokens[1] = v as u32,
        1 => {
            for p in fix.target_probs[..v].iter_mut() {
                *p *= 2.0;
            }
        }
        2 => fix.rng_words[0].draw = u32::MAX as u64 - k as u64,
        3 => {
            fix.k = DEVICE_MAX_K + 1;
            fix.draft_tokens = vec![0u32; s * (DEVICE_MAX_K + 1)];
            fix.draft_probs = Some(vec![1.0 / v as f32; s * (DEVICE_MAX_K + 1) * v]);
            fix.target_probs = vec![1.0 / v as f32; s * (DEVICE_MAX_K + 2) * v];
        }
        _ => {
            fix.target_probs.pop();
        }
    }
    fix
}

impl GateCase for VerifyCase {
    fn op_name(&self) -> &'static str {
        "verify"
    }

    fn tolerance(&self) -> Tolerance {
        Tolerance::for_op("verify").expect("tolerance row must exist")
    }

    fn legal_shapes(&self) -> Vec<Vec<usize>> {
        if self.tree {
            vec![vec![2, 3, 32], vec![1, 5, 64], vec![3, 7, 64]]
        } else {
            vec![
                vec![1, 2, 16],
                vec![2, 4, 64],
                vec![3, 1, 257],
                vec![2, 0, 16],
                vec![1, 8, 32],
            ]
        }
    }

    fn fuzz_legal(&self) -> Vec<Vec<usize>> {
        if self.tree {
            vec![vec![2, 4, 100]]
        } else {
            vec![vec![3, 2, 100], vec![5, 6, 48]]
        }
    }

    fn build(&self, shape: &[usize], seed: u64) -> Result<GateBuffers, HarnessError> {
        *self.key.borrow_mut() = Some((shape.to_vec(), seed, None, None));
        let fix = derive_verify(shape, seed, self.tree, self.has_draft);
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_u32(
                &[fix.s * fix.k.max(1)],
                &vec![0u32; fix.s * fix.k.max(1)],
            )],
            vec![
                TypedBuffer::from_u32(&[fix.s * (fix.k + 1)], &vec![0u32; fix.s * (fix.k + 1)]),
                TypedBuffer::from_u32(&[fix.s], &vec![0u32; fix.s]),
            ],
        ))
    }

    fn build_pinned(&self, shape: &[usize], row: usize) -> Result<GateBuffers, HarnessError> {
        *self.key.borrow_mut() = Some((shape.to_vec(), PINNED_SEED, Some(row), None));
        let fix = derive_verify_pinned(shape, row, self.tree, self.has_draft);
        Ok(GateBuffers::pinned(
            vec![TypedBuffer::from_u32(
                &[fix.s * fix.k.max(1)],
                &vec![0u32; fix.s * fix.k.max(1)],
            )],
            vec![
                TypedBuffer::from_u32(&[fix.s * (fix.k + 1)], &vec![0u32; fix.s * (fix.k + 1)]),
                TypedBuffer::from_u32(&[fix.s], &vec![0u32; fix.s]),
            ],
            row,
        ))
    }

    fn execute(&self, buffers: &mut GateBuffers) -> Result<(), T0Error> {
        let fix = self.fixture_for();
        let out = run_verify(&self.ctx, &fix, &self.method)?;
        *self.last.borrow_mut() = Some(VerifyLast {
            accepted: out.accepted.clone(),
            accept_len: out.accept_len.clone(),
            rng_after: out.rng_after.clone(),
        });
        buffers.outputs = vec![
            TypedBuffer::from_u32(&[fix.s * (fix.k + 1)], &out.accepted),
            TypedBuffer::from_u32(&[fix.s], &out.accept_len),
        ];
        Ok(())
    }

    fn verify(&self, _buffers: &GateBuffers) -> Result<(), HarnessError> {
        // Oracle: T0 `verify` on the fixture inputs with pre-launch RNG state
        // (independent Rust path vs the HIP kernel; exact discrete contract).
        let (shape, seed, row, _) = self.key.borrow().clone().expect("build before verify");
        let fix = match row {
            Some(r) => derive_verify_pinned(&shape, r, self.tree, self.has_draft),
            None => derive_verify(&shape, seed, self.tree, self.has_draft),
        };
        let (expected, expected_draws) =
            t0_probe_verify(&fix, &self.method).map_err(HarnessError::T0)?;
        let last = self.last.borrow();
        let got = last.as_ref().expect("execute before verify");
        if got.accepted != expected.accepted || got.accept_len != expected.accept_len {
            return Err(HarnessError::FuzzVerdict {
                context: format!("{} golden", self.name()),
                detail: "device accepted/accept_len != T0".to_owned(),
            });
        }
        let expected_draws: Vec<u64> = expected_draws.iter().map(|&d| d as u64).collect();
        let got_draws: Vec<u64> = got.rng_after.iter().map(|w| w.draw).collect();
        if got_draws != expected_draws {
            return Err(HarnessError::FuzzVerdict {
                context: format!("{} draw accounting", self.name()),
                detail: format!("device draws {got_draws:?} != T0 {expected_draws:?}"),
            });
        }
        Ok(())
    }

    fn output_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let last = self.last.borrow();
        let got = last.as_ref().expect("execute before output_bytes");
        let mut out = u32_to_bytes(&got.accepted);
        out.extend_from_slice(&u32_to_bytes(&got.accept_len));
        out.extend_from_slice(&rng_bytes_of(&got.rng_after));
        Ok(out)
    }

    fn logical_bytes(&self, _buffers: &GateBuffers) -> Result<Vec<u8>, HarnessError> {
        let (shape, _, row_opt, _) = self
            .key
            .borrow()
            .clone()
            .expect("build before logical_bytes");
        let row = row_opt.expect("pinned run must carry a row");
        let k = shape[1];
        let last = self.last.borrow();
        let got = last.as_ref().expect("execute before logical_bytes");
        let mut out = u32_to_bytes(&got.accepted[row * (k + 1)..(row + 1) * (k + 1)]);
        out.extend_from_slice(&u32_to_bytes(&got.accept_len[row..row + 1]));
        out.extend_from_slice(&rng_bytes_of(&got.rng_after[row..row + 1]));
        Ok(out)
    }

    fn batch_rows(&self) -> BatchRows {
        // Same k=3 in every mode so the pinned tree/chain is identical.
        BatchRows {
            alone: vec![1, 3, 64],
            padded: vec![4, 3, 64],
            embedded: vec![6, 3, 64],
            row_alone: 0,
            row: 2,
        }
    }

    fn illegal_count(&self) -> usize {
        5
    }

    fn build_illegal(&self, index: usize) -> Result<GateBuffers, HarnessError> {
        if index >= 5 {
            return Err(HarnessError::FuzzVerdict {
                context: "verify illegal".to_owned(),
                detail: format!("illegal index {index} out of range"),
            });
        }
        *self.key.borrow_mut() = Some((
            vec![2, 3, 32],
            0xA360_6E12_0BAD_2000 + index as u64,
            None,
            Some(index),
        ));
        Ok(GateBuffers::fresh(
            vec![TypedBuffer::from_u32(&[1], &[0])],
            vec![TypedBuffer::from_u32(&[1], &[0])],
        ))
    }
}

// ---------------------------------------------------------------------------
// Host tests: always run (no HIP needed)
// ---------------------------------------------------------------------------

/// Device record layouts are pinned on both sides of the boundary (Spec 4 §7).
#[test]
fn sampling_device_record_layouts_are_pinned() {
    assert_eq!(std::mem::size_of::<DeviceSamplingParams>(), 40);
    assert_eq!(offset_of!(DeviceSamplingParams, temperature), 0);
    assert_eq!(offset_of!(DeviceSamplingParams, top_p), 4);
    assert_eq!(offset_of!(DeviceSamplingParams, min_p), 8);
    assert_eq!(offset_of!(DeviceSamplingParams, repetition_penalty), 12);
    assert_eq!(offset_of!(DeviceSamplingParams, presence_penalty), 16);
    assert_eq!(offset_of!(DeviceSamplingParams, frequency_penalty), 20);
    assert_eq!(offset_of!(DeviceSamplingParams, top_k), 24);
    assert_eq!(offset_of!(DeviceSamplingParams, bias_start), 28);
    assert_eq!(offset_of!(DeviceSamplingParams, bias_count), 32);
    assert_eq!(std::mem::size_of::<DeviceBiasPair>(), 8);
    assert_eq!(offset_of!(DeviceBiasPair, token), 0);
    assert_eq!(offset_of!(DeviceBiasPair, bias), 4);
    assert_eq!(std::mem::size_of::<DeviceRngWords>(), 24);
    assert_eq!(offset_of!(DeviceRngWords, seed), 0);
    assert_eq!(offset_of!(DeviceRngWords, step), 8);
    assert_eq!(offset_of!(DeviceRngWords, draw), 16);
    // Packing round-trips through the exact byte layout the kernel reads.
    let words = vec![
        DeviceRngWords {
            seed: u64::MAX,
            step: 1u64 << 33,
            draw: 5,
        },
        DeviceRngWords {
            seed: 0,
            step: 0,
            draw: 0,
        },
    ];
    assert_eq!(rng_words_from_bytes(&rng_words_to_bytes(&words)), words);
    let params = vec![SamplingParams {
        temperature: 0.7,
        top_k: 3,
        top_p: 0.9,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: vec![(7, 1.5)],
    }];
    let packed = pack_device_params(&params);
    assert_eq!(packed.len(), 40 + 8);
    assert_eq!(u32::from_le_bytes(packed[24..28].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(packed[28..32].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(packed[32..36].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(packed[40..44].try_into().unwrap()), 7);
}

/// Reference kernels contain no arch-specific builtins or asm (Spec 4 §8).
/// Wave intrinsics are also absent: these kernels use plain portable HIP.
#[test]
fn reference_kernels_use_no_arch_specific_intrinsics() {
    let root = workspace_root().join("kernels/reference");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("kernels/reference must be readable")
        .map(|e| e.expect("dir entry must be readable").path())
        .filter(|p| p.extension().is_some_and(|e| e == "hip"))
        .collect();
    files.sort();
    assert!(
        files.len() >= 4,
        "expected common + 3 kernel sources, found {files:?}"
    );
    // Same list as `cmd_policy_t1_portable` in scripts/ci-gates.sh: only
    // arch/family-specific builtins are forbidden. Portable wave intrinsics
    // (__shfl, __ballot, WMMA where portable per Spec 4 §8) stay allowed so
    // future reference kernels can use them; these three sampling kernels
    // simply use none. Inline asm is covered repo-wide by `cmd_policy_asm`.
    let forbidden = [
        "__builtin_amdgcn",
        "__builtin_gfx",
        "__builtin_nv",
        "amdgcn_",
        "gfx1201",
        "gfx12_",
        "__AMDGCN__",
        "__CUDA_ARCH__",
        "__HIP_PLATFORM_NV__",
    ];
    for path in &files {
        let text = std::fs::read_to_string(path).expect("kernel source must be readable");
        for pat in &forbidden {
            assert!(
                !text.contains(pat),
                "{} contains forbidden intrinsic {pat}",
                path.display()
            );
        }
    }
}

/// Kernel bodies bind the generated ABI struct instead of declaring their own.
#[test]
fn kernel_bodies_bind_the_generated_abi_struct() {
    let root = workspace_root().join("kernels/reference");
    for (file, symbol) in [
        (PP_HIP, "t1_logits_postprocess"),
        (SAMPLE_HIP, "t1_sample"),
        (VERIFY_HIP, "t1_verify"),
    ] {
        let text = std::fs::read_to_string(root.join(file)).expect("body must be readable");
        assert!(
            text.contains(ARGS_TOKEN),
            "{file} must reference {ARGS_TOKEN}"
        );
        assert!(
            !text.contains("_args {") && !text.contains("_args{"),
            "{file} must not declare its own args struct"
        );
        assert!(
            text.contains(&format!("__global__ void {symbol}")),
            "{file} must define entry symbol {symbol}"
        );
    }
    let common = std::fs::read_to_string(root.join(COMMON_HIP)).expect("common must exist");
    assert!(
        common.contains("R9vSamplingParams"),
        "common header must define params record"
    );
    assert!(
        common.contains("R9vRngWords"),
        "common header must define RNG words"
    );
    assert!(
        common.contains("r9v_philox_word"),
        "common header must define Philox"
    );
}

/// Every sampling arg travels through the A3.2 generated ABI (Spec 4 §7).
#[test]
fn sampling_abi_structs_carry_every_arg() {
    let pp = OpStatic::Sampling(SamplingStatic::LogitsPostprocess(LogitsPostprocessStatic {
        s_bucket: 4,
        v: 64,
        q_bucket: 4,
        has_history_counts: true,
        has_grammar_mask: true,
    }));
    let abi = abi_for_op(OpId::LogitsPostprocess, &pp).expect("pp ABI must build");
    let names: Vec<&str> = abi.fields.iter().map(|f| f.name.as_str()).collect();
    for required in [
        "logits",
        "params",
        "history_counts",
        "grammar_mask",
        "probs",
        "workspace",
        "s",
        "q",
    ] {
        assert!(
            names.contains(&required),
            "pp ABI missing {required}: {names:?}"
        );
    }
    assert!(
        abi.name.starts_with("logits_postprocess_") && abi.name.ends_with("_args"),
        "unexpected canonical name {}",
        abi.name
    );
    assert_eq!(
        abi.name,
        canonical_struct_name(OpId::LogitsPostprocess, &pp)
    );
    let emitted = emit_hip_struct(&abi);
    assert!(emitted.contains(&format!("struct {} {{", abi.name)));

    let sm = OpStatic::Sampling(SamplingStatic::Sample(SampleStatic {
        s_bucket: 4,
        v: 64,
        rng: RngAlgorithm::Philox4x32,
    }));
    let abi = abi_for_op(OpId::Sample, &sm).expect("sample ABI must build");
    let names: Vec<&str> = abi.fields.iter().map(|f| f.name.as_str()).collect();
    for required in ["probs", "rng_state", "seq_ids", "tokens", "s"] {
        assert!(
            names.contains(&required),
            "sample ABI missing {required}: {names:?}"
        );
    }
    assert_eq!(abi.name, canonical_struct_name(OpId::Sample, &sm));

    for tree in [false, true] {
        let vf = OpStatic::Sampling(SamplingStatic::Verify(VerifyStatic {
            s_bucket: 4,
            v: 64,
            q_bucket: 4,
            method: VerifyMethodStatic::Rejection,
            tree,
            has_draft_probs: true,
        }));
        let abi = abi_for_op(OpId::Verify, &vf).expect("verify ABI must build");
        let names: Vec<&str> = abi.fields.iter().map(|f| f.name.as_str()).collect();
        for required in [
            "draft_tokens",
            "draft_probs",
            "target_probs",
            "rng_state",
            "seq_ids",
            "accepted",
            "accept_len",
            "s",
            "k",
        ] {
            assert!(
                names.contains(&required),
                "verify ABI missing {required}: {names:?}"
            );
        }
        assert_eq!(names.contains(&"tree_parents"), tree);
        assert_eq!(names.contains(&"tree_ancestors"), tree);
        assert_eq!(abi.name, canonical_struct_name(OpId::Verify, &vf));
    }
}

/// Assembled compilands carry the generated struct, the defines, and the
/// entry symbol with the canonical name bound (Spec 4 §7). Runs on every host.
#[test]
fn assembled_compilands_bind_the_canonical_struct() {
    let reference_dir = workspace_root().join("kernels/reference");
    let pp = OpStatic::Sampling(SamplingStatic::LogitsPostprocess(LogitsPostprocessStatic {
        s_bucket: 4,
        v: 64,
        q_bucket: 4,
        has_history_counts: true,
        has_grammar_mask: true,
    }));
    let abi = abi_for_op(OpId::LogitsPostprocess, &pp).expect("pp ABI must build");
    let src = assemble_compiland(
        &reference_dir,
        &abi,
        PP_HIP,
        &[
            ("R9V_V".to_string(), "64".to_string()),
            ("R9V_PP_HAS_HISTORY".to_string(), "1".to_string()),
            ("R9V_PP_HAS_MASK".to_string(), "1".to_string()),
        ],
    );
    assert!(src.contains(&format!("struct {} {{", abi.name)));
    assert!(
        src.contains(&format!(
            "__global__ void t1_logits_postprocess(const {} ",
            abi.name
        )) || src.contains(&format!("t1_logits_postprocess(const {}", abi.name))
    );

    let sm = OpStatic::Sampling(SamplingStatic::Sample(SampleStatic {
        s_bucket: 4,
        v: 64,
        rng: RngAlgorithm::Philox4x32,
    }));
    let abi = abi_for_op(OpId::Sample, &sm).expect("sample ABI must build");
    let src = assemble_compiland(
        &reference_dir,
        &abi,
        SAMPLE_HIP,
        &[("R9V_V".to_string(), "64".to_string())],
    );
    assert!(src.contains(&format!("struct {} {{", abi.name)));
    assert!(src.contains(&format!("t1_sample(const {}", abi.name)));

    let vf = OpStatic::Sampling(SamplingStatic::Verify(VerifyStatic {
        s_bucket: 4,
        v: 64,
        q_bucket: 4,
        method: VerifyMethodStatic::typical(0.05, 1.5),
        tree: true,
        has_draft_probs: true,
    }));
    let abi = abi_for_op(OpId::Verify, &vf).expect("verify ABI must build");
    let src = assemble_compiland(
        &reference_dir,
        &abi,
        VERIFY_HIP,
        &[
            ("R9V_V".to_string(), "64".to_string()),
            ("R9V_VERIFY_METHOD".to_string(), "2".to_string()),
            ("R9V_VERIFY_TREE".to_string(), "1".to_string()),
            ("R9V_HAS_DRAFT_PROBS".to_string(), "1".to_string()),
            (
                "R9V_VERIFY_EPS_BITS".to_string(),
                0.05f32.to_bits().to_string(),
            ),
            (
                "R9V_VERIFY_DELTA_BITS".to_string(),
                1.5f32.to_bits().to_string(),
            ),
        ],
    );
    assert!(src.contains(&format!("struct {} {{", abi.name)));
    assert!(src.contains(&format!("t1_verify(const {}", abi.name)));
}

/// Registry resolves every sampling variant to the T1 fallback (Spec 4 §9.2).
#[test]
fn registry_resolves_t1_fallback_for_sampling_ops() {
    fn add_t1(manifest: &mut BundleManifest, op: OpId, salt: u64) {
        let arch = manifest.archs.first().cloned().expect("arch must exist");
        manifest.insert_variant(
            VariantHash::new(0xA360_0001 ^ salt.wrapping_mul(0x9E3779B97F4A7C15)),
            ManifestVariantEntry {
                arch,
                file: format!("reference/{}.co", op.as_str()),
                tier: Tier::T1,
                entry_symbol: format!("t1_{}", op.as_str()),
                launch_geometry: LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
                workspace_bytes: 0,
                static_bytes: 0,
                static_flops: 0,
                op: Some(op),
                static_hash: None,
                validated: true,
                validated_on: Some("a3.6-test".to_string()),
            },
        );
    }
    let arch = ArchName::from("gfx942");
    let mut manifest = BundleManifest::new(1, vec![arch.clone()]);
    for (salt, op) in [OpId::LogitsPostprocess, OpId::Sample, OpId::Verify]
        .into_iter()
        .enumerate()
    {
        add_t1(&mut manifest, op, salt as u64 + 1);
    }
    let mut registry = Registry::new(RegistryConfig {
        allow_jit: false,
        ..Default::default()
    });
    registry
        .set_manifest(manifest, None)
        .expect("manifest must validate");

    let cases: Vec<(OpId, OpStatic)> = vec![
        (
            OpId::LogitsPostprocess,
            OpStatic::Sampling(SamplingStatic::LogitsPostprocess(LogitsPostprocessStatic {
                s_bucket: 4,
                v: 64,
                q_bucket: 4,
                has_history_counts: true,
                has_grammar_mask: false,
            })),
        ),
        (
            OpId::Sample,
            OpStatic::Sampling(SamplingStatic::Sample(SampleStatic {
                s_bucket: 4,
                v: 64,
                rng: RngAlgorithm::Philox4x32,
            })),
        ),
        (
            OpId::Verify,
            OpStatic::Sampling(SamplingStatic::Verify(VerifyStatic {
                s_bucket: 4,
                v: 64,
                q_bucket: 4,
                method: VerifyMethodStatic::Rejection,
                tree: true,
                has_draft_probs: true,
            })),
        ),
    ];
    for (op, static_) in &cases {
        let resolved = registry
            .resolve(*op, &arch, static_)
            .expect("T1 fallback must resolve on a listed arch");
        assert_eq!(resolved.tier, Tier::T1, "op {op:?} must resolve to T1");
        assert_eq!(resolved.entry_symbol, format!("t1_{}", op.as_str()));
    }
    // Unlisted arch still refuses (Spec 4 §9.2).
    let err = registry
        .resolve(OpId::Sample, &ArchName::from("gfx0000"), &cases[1].1)
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("gfx0000"),
        "unlisted arch refusal must name the arch: {err:?}"
    );
}

/// Hostile inputs are refused by the T0 probe with typed errors, before any
/// device touch (Spec 1 §4.F). Runs on every host.
#[test]
fn hostile_inputs_are_refused_before_device_touch() {
    for idx in 0..6 {
        let fix = pp_illegal_fixture(idx);
        assert!(
            t0_probe_pp(&fix).is_err(),
            "pp illegal #{idx} must be refused"
        );
    }
    for idx in 0..4 {
        let fix = sample_illegal_fixture(idx);
        assert!(
            t0_probe_sample(&fix).is_err(),
            "sample illegal #{idx} must be refused"
        );
    }
    for idx in 0..5 {
        for tree in [false, true] {
            let fix = verify_illegal_fixture(idx, tree);
            assert!(
                t0_probe_verify(&fix, &VerifyMethod::Rejection).is_err(),
                "verify illegal #{idx} tree={tree} must be refused"
            );
        }
    }
    // The device path cap refuses what T0 would accept (documented T1 bound).
    let mut wide = derive_verify(
        &[1, DEVICE_MAX_K + 1, 16],
        0xA360_6E12_0BAD_3000,
        false,
        true,
    );
    wide.k = DEVICE_MAX_K + 1;
    assert!(t0_probe_verify(&wide, &VerifyMethod::Rejection).is_err());
}

/// Pinned fixtures keep the logical row identical across batch modes (host
/// check of the invariance preconditions; the device comparison runs on GPU).
#[test]
fn pinned_fixtures_keep_the_logical_row_identical() {
    let alone = derive_pp_pinned(&[1, 2, 64], 0);
    let padded = derive_pp_pinned(&[4, 2, 64], 2);
    let embedded = derive_pp_pinned(&[6, 2, 64], 2);
    assert_eq!(alone.logits, padded.logits[2 * 128..3 * 128]);
    assert_eq!(alone.logits, embedded.logits[2 * 128..3 * 128]);
    assert_eq!(alone.params, vec![padded.params[2].clone()]);
    assert_eq!(alone.seq_ids, vec![padded.seq_ids[2]]);
    assert_eq!(alone.params, vec![embedded.params[2].clone()]);

    let alone = derive_sample_pinned(&[1, 64], 0);
    let padded = derive_sample_pinned(&[4, 64], 2);
    assert_eq!(alone.probs, padded.probs[128..192]);
    assert_eq!(alone.rng_words, vec![padded.rng_words[2]]);
    assert_eq!(alone.seq_ids, vec![padded.seq_ids[2]]);

    let alone = derive_verify_pinned(&[1, 3, 64], 0, true, true);
    let padded = derive_verify_pinned(&[4, 3, 64], 2, true, true);
    assert_eq!(alone.draft_tokens, padded.draft_tokens[6..9]);
    assert_eq!(alone.target_probs, padded.target_probs[2 * 256..3 * 256]);
    assert_eq!(alone.parents, padded.parents);
}

// ---------------------------------------------------------------------------
// GPU tests: run only with a compatible HIP runtime + device (Spec 4 §10)
// ---------------------------------------------------------------------------

fn gpu_ctx_or_skip() -> Option<std::rc::Rc<GpuCtx>> {
    probe_gpu().map(|lib| std::rc::Rc::new(GpuCtx::new(lib)))
}

#[test]
fn t1_logits_postprocess_passes_all_gates() {
    let Some(ctx) = gpu_ctx_or_skip() else { return };
    run_gates(&PpCase::new(ctx)).expect("pp must pass golden/invariance/determinism/fuzz");
}

#[test]
fn t1_sample_passes_all_gates() {
    let Some(ctx) = gpu_ctx_or_skip() else { return };
    run_gates(&SampleCase::new(ctx)).expect("sample must pass golden/invariance/determinism/fuzz");
}

#[test]
fn t1_verify_passes_all_gates() {
    let Some(ctx) = gpu_ctx_or_skip() else { return };
    let methods = [
        VerifyMethod::Rejection,
        VerifyMethod::Greedy,
        VerifyMethod::TypicalAcceptance {
            eps: 0.05,
            delta: 1.5,
        },
    ];
    for method in &methods {
        for (tree, has_draft) in [(false, true), (false, false), (true, true), (true, false)] {
            let case = VerifyCase::new(ctx.clone(), *method, tree, has_draft);
            run_gates(&case).expect("verify {} must pass all gates");
        }
    }
}

/// RNG raw-state boundaries agree bit-exactly with T0 on device (Spec 1 §4.F).
#[test]
fn t1_sampling_rng_raw_state_boundaries_agree_with_t0() {
    let Some(ctx) = gpu_ctx_or_skip() else { return };
    // Extreme seeds, full-64-bit steps (upper-bits must not collide), max
    // sequence word, and draws that advance exactly onto u32::MAX.
    let states = [
        (0u64, 0u64, 0u32, 0u64),
        (u64::MAX, u64::MAX, u32::MAX, u32::MAX as u64 - 1),
        (0x1234_5678_9ABC_DEF0, 1, 7, 1),
        (0x1234_5678_9ABC_DEF0, 1 | (1u64 << 32), 7, 1),
        (42, 1u64 << 32, 3, 99),
    ];
    for (i, &(seed, step, seq, draw)) in states.iter().enumerate() {
        let fix = SampleFixture {
            s: 1,
            v: 128,
            probs: derive_dist_rows(&mut SeededRng::new(0xA360_6E12_5000 + i as u64), 1, 128),
            rng_words: vec![DeviceRngWords { seed, step, draw }],
            seq_ids: vec![seq],
        };
        run_sample(&ctx, &fix).expect("raw-state sample must match T0");
    }
    // Draws landing exactly on u32::MAX are valid; one past refuses.
    let edge = SampleFixture {
        s: 1,
        v: 64,
        probs: derive_dist_rows(&mut SeededRng::new(0xA360_6E12_5001), 1, 64),
        rng_words: vec![DeviceRngWords {
            seed: 9,
            step: 9,
            draw: u32::MAX as u64 - 1,
        }],
        seq_ids: vec![5],
    };
    let out = run_sample(&ctx, &edge).expect("draw onto u32::MAX must succeed");
    assert_eq!(out.rng_after[0].draw, u32::MAX as u64);
    let over = SampleFixture {
        s: 1,
        v: 64,
        probs: edge.probs.clone(),
        rng_words: vec![DeviceRngWords {
            seed: 9,
            step: 9,
            draw: u32::MAX as u64,
        }],
        seq_ids: vec![5],
    };
    assert!(
        t0_probe_sample(&over).is_err(),
        "draw past u32::MAX must refuse"
    );

    // Verify advances k+1 onto u32::MAX exactly.
    let k = 3usize;
    let mut vfix = derive_verify(&[2, k, 64], 0xA360_6E12_5002, false, true);
    vfix.rng_words[0].draw = u32::MAX as u64 - k as u64 - 1;
    let out = run_verify(&ctx, &vfix, &VerifyMethod::Rejection)
        .expect("verify draw onto u32::MAX must succeed");
    assert_eq!(out.rng_after[0].draw, u32::MAX as u64);
}

/// Temperature zero is exact argmax with lowest-index ties (Spec 1 §4.F).
#[test]
fn t1_temperature_zero_is_exact_argmax_with_lowest_index_ties() {
    let Some(ctx) = gpu_ctx_or_skip() else { return };
    // All-equal logits: every tie breaks to index 0, even with penalties.
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.3,
        presence_penalty: 0.2,
        frequency_penalty: 0.1,
        logit_bias: vec![],
    };
    let fix = PpFixture {
        s: 1,
        q: 2,
        v: 256,
        logits: vec![1.0; 512],
        params: vec![params],
        history: Some(vec![2u32; 256]),
        mask_bool: None,
        seq_ids: vec![11],
    };
    let probs = run_pp(&ctx, &fix).expect("temp-zero must run");
    for row in 0..2 {
        assert_eq!(probs[row * 256], 1.0, "all-tie row {row} breaks to index 0");
        assert!(probs[row * 256 + 1..(row + 1) * 256]
            .iter()
            .all(|&p| p == 0.0));
    }
    // A two-way tie away from zero breaks to the lower index.
    let mut logits = vec![0.0f32; 128];
    logits[40] = 5.0;
    logits[90] = 5.0;
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        logit_bias: vec![],
    };
    let fix = PpFixture {
        s: 1,
        q: 1,
        v: 128,
        logits,
        params: vec![params],
        history: None,
        mask_bool: None,
        seq_ids: vec![12],
    };
    let probs = run_pp(&ctx, &fix).expect("temp-zero tie must run");
    assert_eq!(probs[40], 1.0, "two-way tie breaks to the lower index");
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "one-hot row sums to 1, got {sum}");
}

/// Tree verify path rules on hand-built shapes (Spec 7 §5).
#[test]
fn t1_verify_tree_path_rules_match_t0() {
    let Some(ctx) = gpu_ctx_or_skip() else { return };
    // Star tree: every draft node is a root; longest-accepted plus
    // lowest-first-token tie rules decide.
    let (s, k, v) = (1usize, 4usize, 8usize);
    let mut fix = derive_verify(&[s, k, v], 0xA360_6E12_6000, true, false);
    fix.parents = Some(vec![-1, -1, -1, -1]);
    run_verify(&ctx, &fix, &VerifyMethod::Greedy).expect("star tree must match T0");
    let typical = VerifyMethod::TypicalAcceptance {
        eps: 0.05,
        delta: 1.5,
    };
    run_verify(&ctx, &fix, &typical).expect("star tree typical must match T0");
    // Chain tree with a fork at the root.
    fix.parents = Some(vec![-1, 0, 0, 2]);
    run_verify(&ctx, &fix, &VerifyMethod::Greedy).expect("forked tree must match T0");
    run_verify(&ctx, &fix, &typical).expect("forked tree typical must match T0");
    // Degenerate k=0 still draws the bonus token from the root distribution.
    let fix0 = derive_verify(&[2, 0, 16], 0xA360_6E12_6001, false, false);
    run_verify(&ctx, &fix0, &VerifyMethod::Rejection).expect("k=0 must match T0");
    run_verify(&ctx, &fix0, &VerifyMethod::Greedy).expect("k=0 greedy must match T0");
    // One-hot draft path (deterministic proposer) under rejection.
    let fix1 = derive_verify(&[2, 3, 32], 0xA360_6E12_6002, false, false);
    run_verify(&ctx, &fix1, &VerifyMethod::Rejection).expect("one-hot path must match T0");
}

/// Chained launches consume consecutive draws exactly like T0 (Spec 1 §4.F).
#[test]
fn t1_chained_launches_consume_consecutive_draws() {
    let Some(ctx) = gpu_ctx_or_skip() else { return };
    let fix = derive_sample(&[3, 128], 0xA360_6E12_7000);
    let out1 = run_sample(&ctx, &fix).expect("first sample must match T0");
    let fix2 = SampleFixture {
        rng_words: out1.rng_after.clone(),
        ..fix_clone_without_rng(&fix)
    };
    let out2 = run_sample(&ctx, &fix2).expect("chained sample must match T0");
    for (w1, w2) in out1.rng_after.iter().zip(out2.rng_after.iter()) {
        assert_eq!(w2.draw, w1.draw + 1);
    }

    let method = VerifyMethod::Rejection;
    let vfix = derive_verify(&[2, 3, 64], 0xA360_6E12_7001, false, true);
    let vout1 = run_verify(&ctx, &vfix, &method).expect("first verify must match T0");
    let vfix2 = VerifyFixture {
        rng_words: vout1.rng_after.clone(),
        ..vfix_clone_without_rng(&vfix)
    };
    let vout2 = run_verify(&ctx, &vfix2, &method).expect("chained verify must match T0");
    for (w1, w2) in vout1.rng_after.iter().zip(vout2.rng_after.iter()) {
        assert_eq!(w2.draw, w1.draw + 4);
    }
}

fn fix_clone_without_rng(fix: &SampleFixture) -> SampleFixture {
    SampleFixture {
        s: fix.s,
        v: fix.v,
        probs: fix.probs.clone(),
        rng_words: Vec::new(),
        seq_ids: fix.seq_ids.clone(),
    }
}

fn vfix_clone_without_rng(fix: &VerifyFixture) -> VerifyFixture {
    VerifyFixture {
        s: fix.s,
        k: fix.k,
        v: fix.v,
        draft_tokens: fix.draft_tokens.clone(),
        draft_probs: fix.draft_probs.clone(),
        target_probs: fix.target_probs.clone(),
        rng_words: Vec::new(),
        seq_ids: fix.seq_ids.clone(),
        parents: fix.parents.clone(),
        t_max: fix.t_max,
    }
}
