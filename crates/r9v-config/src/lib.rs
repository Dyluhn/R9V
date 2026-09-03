//! R9V configuration schema, settings generator, and validation (Spec 12, Spec 14 §2).
//!
//! Spec 12 §2: the schema is code; docs, files, schema, and validation messages
//! are generated from it. Values retain their winning precedence source and
//! `auto` resolutions remain auditable.

extern crate self as r9v_config;

use std::path::PathBuf;

use r9v_common::ByteSize;

mod config;
mod generate;
mod index;

pub use config::{ConfigError, EffectiveConfig, Source, SourcedValue, CONFIG_VERSION};
pub use generate::{
    generate_artifacts, render_effective_toml, write_generated, GeneratedArtifacts,
};
pub use index::{check_settings_index, SettingsIndexError};
pub use r9v_config_macros::{section, setting};

/// `auto` marker: the effective value is resolved by the rule in the setting's
/// `default` text (Spec 12 §1, principle 2). `Auto::Auto` defers to that rule;
/// `Auto::Value(v)` pins an explicit value.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Auto<T> {
    /// Resolve via the setting's documented rule.
    #[default]
    Auto,
    /// Explicit value overriding the rule.
    Value(T),
}

impl<T> Auto<T> {
    /// True when the value defers to the documented resolution rule.
    pub fn is_auto(&self) -> bool {
        matches!(self, Auto::Auto)
    }

    /// Borrow the explicit value, if any.
    pub fn as_value(&self) -> Option<&T> {
        match self {
            Auto::Auto => None,
            Auto::Value(v) => Some(v),
        }
    }

    /// Resolve with a rule callback for the `Auto` case.
    pub fn resolve<U>(&self, rule: impl FnOnce() -> U, map: impl FnOnce(&T) -> U) -> U {
        match self {
            Auto::Auto => rule(),
            Auto::Value(v) => map(v),
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Auto<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Auto::Auto => write!(f, "auto"),
            Auto::Value(v) => write!(f, "{v}"),
        }
    }
}

impl<T: std::str::FromStr> std::str::FromStr for Auto<T> {
    type Err = T::Err;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            Ok(Self::Auto)
        } else {
            value.parse().map(Self::Value)
        }
    }
}

macro_rules! closed_setting {
    (
        $(#[$enum_meta:meta])*
        pub enum $name:ident {
            $($(#[$variant_meta:meta])* $variant:ident => $value:literal),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$variant_meta])* $variant),+
        }

        impl $name {
            /// Return the stable config-file spelling from Spec 12 §3.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $value),+
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

closed_setting! {
    /// Spec 12 §3. Concrete weight-read mode when `io.mode` is not `auto`.
    pub enum IoMode {
        /// Linux direct I/O.
        Direct => "direct",
        /// Memory-mapped reads.
        Mmap => "mmap",
    }
}

closed_setting! {
    /// Spec 12 §3. Stored KV-cache element representation.
    pub enum CacheDtype {
        /// E4M3 FP8 cache values.
        E4m3 => "e4m3",
        /// Signed 8-bit integer cache values.
        I8 => "i8",
        /// IEEE half-precision cache values.
        F16 => "f16",
    }
}

closed_setting! {
    /// Spec 12 §3. Concrete graph replay mode when `graph.mode` is not `auto`.
    pub enum GraphMode {
        /// Replay an explicit launch list.
        List => "list",
        /// Replay a captured HIP graph.
        HipGraph => "hipgraph",
    }
}

closed_setting! {
    /// Spec 12 §3. Concrete speculative proposer when `spec.proposer` is not `auto`.
    pub enum ProposerKind {
        /// Disable speculative proposals.
        None => "none",
        /// Host or device n-gram proposals.
        Ngram => "ngram",
        /// Multi-token-prediction head.
        Mtp => "mtp",
        /// Separate draft model.
        Draft => "draft",
        /// Eagle head.
        Eagle => "eagle",
    }
}

closed_setting! {
    /// Spec 12 §3 and Spec 11 §3. Profiling detail.
    pub enum ProfileMode {
        /// One device event per step graph.
        Step => "step",
        /// Per-kernel launch profiling.
        Kernel => "kernel",
        /// Disable timing instrumentation.
        Off => "off",
    }
}

closed_setting! {
    /// Spec 12 §3 and Spec 11 §11. Structured-log severity threshold.
    pub enum LogLevel {
        /// Most detailed diagnostic records.
        Trace => "trace",
        /// Debug records.
        Debug => "debug",
        /// Operational records.
        Info => "info",
        /// Warning records.
        Warn => "warn",
        /// Error records only.
        Error => "error",
    }
}

/// Declared mutability of a setting (Spec 12 §1, principle 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutability {
    /// Changeable at runtime (`POST /r9v/config`).
    Runtime,
    /// Changeable via reload (file/env/CLI, no fresh load).
    Reload,
    /// Fixed once a model is loaded.
    Load,
}

impl std::fmt::Display for Mutability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mutability::Runtime => write!(f, "Runtime"),
            Mutability::Reload => write!(f, "Reload"),
            Mutability::Load => write!(f, "Load"),
        }
    }
}

/// One setting's metadata: the single source of truth (Spec 12 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SettingSpec {
    /// Full `section.key` (e.g. `"scheduler.step_budget_ms"`).
    pub key: &'static str,
    /// Rust value type declared by the annotated field.
    pub type_name: &'static str,
    /// Doc string; also the generated file/docs/helper explanation.
    pub doc: &'static str,
    /// Default: a value or `auto` plus the resolving rule.
    pub default: &'static str,
    /// Range (`"1.0..=1000.0"`) or pipe-joined enum (`"direct|mmap|auto"`); `""` when neither applies.
    pub range_or_enum: &'static str,
    /// Unit (`"ms"`, `"bytes"`, ...); `""` when unitless.
    pub unit: &'static str,
    /// Declared mutability.
    pub mutability: Mutability,
    /// Keys whose meaning or resolution this setting affects.
    pub interacts: &'static [&'static str],
    /// Schema version introducing the setting.
    pub since: u32,
    /// Previous key name after a rename; `""` when never renamed.
    pub renamed_from: &'static str,
}

// --- Phase-A sections (Spec 12 §3) -----------------------------------------

/// Spec 12 §3. Model loading inputs.
#[section("load", doc = "Model and cache inputs for load.")]
pub struct LoadSection {
    /// Model file or directory.
    #[setting(kind = "path", doc = "Primary model artifact path.", default = "(none)", mutable = Load, since = 1)]
    pub model: PathBuf,
    /// Draft model for speculative decoding.
    #[setting(kind = "path", doc = "Draft model artifact path, if any.", default = "none", mutable = Load, since = 1)]
    pub draft_model: PathBuf,
    /// Eagle head weights, if any.
    #[setting(kind = "path", doc = "Eagle head artifact path, if any.", default = "none", mutable = Load, since = 1)]
    pub eagle_head: PathBuf,
    /// Repacked-cache directory.
    #[setting(kind = "Auto<path>", doc = "Cache directory. auto = beside the model.", default = "auto (beside model)", mutable = Load, since = 1)]
    pub cache_dir: Auto<PathBuf>,
    /// Refuse to load without the fast path.
    #[setting(doc = "Require the fast execution path at load.", default = "false", mutable = Load, since = 1)]
    pub require_fast_path: bool,
}

/// Spec 12 §3. IO behaviour for load/repack.
#[section("io", doc = "IO behaviour for load and repack.")]
pub struct IoSection {
    /// IO mode.
    #[setting(doc = "IO mode for weight reads. auto = direct I/O when supported, otherwise mmap.", default = "auto", values = ["direct", "mmap", "auto"], mutable = Load, since = 1)]
    pub mode: Auto<IoMode>,
    /// Read chunk size.
    #[setting(doc = "Read chunk size.", default = "16", range = "1..=1024", unit = "MB", mutable = Load, since = 1)]
    pub chunk_mb: u32,
    /// Queue depth.
    #[setting(doc = "IO queue depth.", default = "8", range = "1..=128", mutable = Load, since = 1)]
    pub queue_depth: u32,
    /// Repack threads.
    #[setting(doc = "Repack worker threads. auto = cores minus 2.", default = "auto (cores-2)", range = "1..=256", mutable = Load, since = 1)]
    pub repack_threads: Auto<u32>,
}

/// Spec 12 §3. Host memory budget.
#[section("host", doc = "Host memory budget.")]
pub struct HostSection {
    /// Pinned host-memory budget.
    #[setting(kind = "Auto<bytes>", doc = "Pinned host-memory budget. auto = min(free minus 4 GB, need).", default = "auto (min(free-4GB, need))", unit = "bytes", mutable = Load, since = 1)]
    pub pinned_budget: Auto<ByteSize>,
}

/// Spec 12 §3. Warmup buckets.
#[section("warmup", doc = "Warmup behaviour and buckets.")]
pub struct WarmupSection {
    /// Run warmup at load.
    #[setting(doc = "Run warmup over the bucket set at load.", default = "true", mutable = Load, since = 1)]
    pub enabled: bool,
    /// Warmup bucket set.
    #[setting(kind = "buckets", doc = "Warmup buckets over S, T_dec and T_pre.", default = "{S:[1,2,4], T_dec:[1,2,4,8,16,32], T_pre:[0,128,512,2048]}", mutable = Load, interacts = ["scheduler.prefill_max_chunk"], since = 1)]
    pub buckets: WarmupBuckets,
}

/// Strongly typed shape of `warmup.buckets`; serialized keys follow the spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmupBuckets {
    /// Concurrent-sequence buckets (`S`).
    pub sequences: Vec<u32>,
    /// Decode-token buckets (`T_dec`).
    pub decode_tokens: Vec<u32>,
    /// Prefill-token buckets (`T_pre`).
    pub prefill_tokens: Vec<u32>,
}

/// Spec 12 §3. Sequence-state limits.
#[section("state", doc = "Sequence-state limits and cache policy.")]
pub struct StateSection {
    /// Maximum context length.
    #[setting(doc = "Maximum context length; must be a multiple of 32.", default = "32768", range = "32..=1048576", mutable = Reload, since = 1)]
    pub max_ctx: u32,
    /// Maximum concurrent sequences.
    #[setting(doc = "Maximum concurrent sequences.", default = "8", range = "1..=1024", mutable = Reload, since = 1)]
    pub max_seqs: u32,
    /// KV cache dtype.
    #[setting(doc = "KV cache dtype.", default = "e4m3", values = ["e4m3", "i8", "f16"], mutable = Reload, since = 1)]
    pub cache_dtype: CacheDtype,
    /// Reserved bytes.
    #[setting(kind = "bytes", doc = "Bytes reserved outside the state pool.", default = "512 MB", unit = "bytes", mutable = Reload, since = 1)]
    pub reserve_bytes: ByteSize,
    /// Host block budget.
    #[setting(kind = "bytes", doc = "Host block spill budget; 0 disables spilling.", default = "0", unit = "bytes", mutable = Reload, since = 1)]
    pub host_block_budget: ByteSize,
    /// Session cache density.
    #[setting(doc = "Retained sessions per GB of cache.", default = "2", range = "0..=64", mutable = Runtime, since = 1)]
    pub session_cache: u32,
}

/// Spec 12 §3. Step scheduling.
#[section("scheduler", doc = "Step scheduling. Decode step time is the SLO.")]
pub struct SchedulerSection {
    /// Target wall time per step.
    #[setting(doc = "Target wall time per step in milliseconds. Prefill chunks and speculative depth are sized so the step fits. auto = 1.25 x measured single-sequence step time (latency) or 8 x (throughput), resolved at warmup on the loaded plan.", default = "auto", range = "1.0..=1000.0", unit = "ms", mutable = Runtime, interacts = ["scheduler.max_wait_ms", "spec.k_max", "parallel.profile"], since = 1)]
    pub step_budget_ms: Auto<f32>,
    /// Minimum prefill chunk.
    #[setting(doc = "Minimum prefill chunk.", default = "128", range = "1..=16384", unit = "tokens", mutable = Runtime, interacts = ["scheduler.prefill_max_chunk"], since = 1)]
    pub prefill_min_chunk: u32,
    /// Maximum prefill chunk.
    #[setting(doc = "Maximum prefill chunk.", default = "2048", range = "1..=16384", unit = "tokens", mutable = Runtime, interacts = ["scheduler.prefill_min_chunk"], since = 1)]
    pub prefill_max_chunk: u32,
    /// Maximum queue wait.
    #[setting(doc = "Maximum time a request waits before a step starts.", default = "500", range = "0..=60000", unit = "ms", mutable = Runtime, since = 1)]
    pub max_wait_ms: u32,
}

/// Spec 12 §3. Graph capture policy.
#[section("graph", doc = "Graph capture policy.")]
pub struct GraphSection {
    /// Capture mode.
    #[setting(doc = "Graph capture mode. auto = measured at warmup.", default = "auto (measured)", values = ["auto", "list", "hipgraph"], mutable = Reload, since = 1)]
    pub mode: Auto<GraphMode>,
}

/// Spec 12 §3. N-gram speculative policy (phase-A subset of `spec.*`).
#[section("spec", doc = "Speculative decoding policy.")]
pub struct SpecSection {
    /// Proposer selection.
    #[setting(doc = "Speculative proposer. auto = MTP, Eagle, draft, then n-gram according to loaded artifacts.", default = "auto", values = ["auto", "none", "ngram", "mtp", "draft", "eagle"], mutable = Reload, interacts = ["load.draft_model", "load.eagle_head", "spec.k_max"], since = 1)]
    pub proposer: Auto<ProposerKind>,
    /// Maximum linear draft depth.
    #[setting(doc = "Maximum speculative draft depth; k + 1 verified positions must fit the decode-class limit.", default = "8", range = "0..=15", mutable = Runtime, interacts = ["spec.tree_max", "scheduler.step_budget_ms"], since = 1)]
    pub k_max: u32,
    /// Maximum tree size.
    #[setting(doc = "Maximum speculative tree size.", default = "16", range = "1..=16", mutable = Runtime, interacts = ["spec.k_max"], since = 1)]
    pub tree_max: u32,
    /// Acceptance threshold.
    #[setting(doc = "Disable speculation temporarily when the recent acceptance EMA is below this value.", default = "0.3", range = "0.0..=1.0", mutable = Runtime, since = 1)]
    pub min_accept: f32,
    /// Permit lossy acceptance.
    #[setting(doc = "Permit opt-in lossy Typical acceptance.", default = "false", mutable = Runtime, since = 1)]
    pub lossy: bool,
}

/// Spec 12 §3. N-gram speculative policy.
#[section("spec.ngram", doc = "N-gram proposer policy.")]
pub struct SpecNgramSection {
    /// N-gram width.
    #[setting(doc = "N-gram width.", default = "3", range = "1..=16", mutable = Runtime, since = 1)]
    pub n: u32,
    /// Minimum match length.
    #[setting(doc = "Minimum match length to propose.", default = "2", range = "1..=16", mutable = Runtime, since = 1)]
    pub min_match: u32,
}

/// Spec 12 §3. Kernel policy.
#[section("kernels", doc = "Kernel build and determinism policy.")]
pub struct KernelsSection {
    /// Allow JIT-built kernels.
    #[setting(doc = "Allow just-in-time kernel builds.", default = "true", mutable = Load, since = 1)]
    pub allow_jit: bool,
    /// Allow nondeterministic kernels.
    #[setting(doc = "Allow kernels that may be nondeterministic.", default = "false", mutable = Load, since = 1)]
    pub allow_nondeterministic: bool,
    /// Autotune budget.
    #[setting(doc = "Autotune time budget.", default = "2000", range = "0..=600000", unit = "ms", mutable = Load, since = 1)]
    pub tune_budget_ms: u64,
}

/// Spec 12 §3. Profiling mode.
#[section("profile", doc = "Profiling mode.")]
pub struct ProfileSection {
    /// Profile mode.
    #[setting(doc = "Profiling mode.", default = "step", values = ["step", "kernel", "off"], mutable = Runtime, since = 1)]
    pub mode: ProfileMode,
}

/// Spec 12 §3. Logging.
#[section("log", doc = "Logging policy.")]
pub struct LogSection {
    /// Log level.
    #[setting(doc = "Log level.", default = "info", values = ["trace", "debug", "info", "warn", "error"], mutable = Runtime, since = 1)]
    pub level: LogLevel,
    /// Log file.
    #[setting(kind = "path", doc = "Log file path; none disables file logging.", default = "none", mutable = Runtime, since = 1)]
    pub file: PathBuf,
}

/// Spec 12 §3. Doctor bundle policy.
#[section("doctor", doc = "Doctor bundle policy.")]
pub struct DoctorSection {
    /// Include token text in bundles.
    #[setting(doc = "Include token text in the doctor bundle.", default = "false", mutable = Runtime, since = 1)]
    pub include_tokens: bool,
    /// Redact secrets in bundles.
    #[setting(doc = "Redact secrets in the doctor bundle.", default = "true", mutable = Runtime, since = 1)]
    pub redact: bool,
}

/// Spec 12 §3. Bench defaults.
#[section("bench", doc = "Benchmark defaults.")]
pub struct BenchSection {
    /// Measured repeats.
    #[setting(doc = "Measured repeats per benchmark.", default = "5", range = "1..=100", mutable = Runtime, since = 1)]
    pub repeats: u32,
    /// Warmup runs.
    #[setting(doc = "Warmup runs before measurement.", default = "2", range = "0..=100", mutable = Runtime, since = 1)]
    pub warmup: u32,
    /// Suites to run.
    #[setting(kind = "[str]", doc = "Benchmark suites to run.", default = "[decode, decode-spec, prefill, multi]", mutable = Runtime, since = 1)]
    pub suites: Vec<String>,
}

/// All phase-A settings in deterministic section order.
pub fn all_settings() -> Vec<&'static SettingSpec> {
    let mut out = Vec::new();
    out.extend(LoadSection::SETTINGS.iter());
    out.extend(IoSection::SETTINGS.iter());
    out.extend(HostSection::SETTINGS.iter());
    out.extend(WarmupSection::SETTINGS.iter());
    out.extend(StateSection::SETTINGS.iter());
    out.extend(SchedulerSection::SETTINGS.iter());
    out.extend(GraphSection::SETTINGS.iter());
    out.extend(SpecSection::SETTINGS.iter());
    out.extend(SpecNgramSection::SETTINGS.iter());
    out.extend(KernelsSection::SETTINGS.iter());
    out.extend(ProfileSection::SETTINGS.iter());
    out.extend(LogSection::SETTINGS.iter());
    out.extend(DoctorSection::SETTINGS.iter());
    out.extend(BenchSection::SETTINGS.iter());
    out
}

/// Look up one setting by full `section.key`.
pub fn find_setting(key: &str) -> Option<&'static SettingSpec> {
    all_settings().into_iter().find(|s| s.key == key)
}
