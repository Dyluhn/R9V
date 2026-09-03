// SPDX-License-Identifier: Apache-2.0
//! Arch descriptor (Spec 1 App. A).
//!
//! One instance per device, consumed by the kernel generator (Spec 4), the
//! partitioner (Spec 5), the loader (Spec 9) and the scheduler (Spec 6). It is
//! the only source of hardware facts in the engine: wave size, LDS size,
//! instruction availability and bandwidth all come from here, never from
//! literals in kernel-adjacent code (r9v-card-work §2).
//!
//! Fields are public: the spec states no cross-field invariants, and validity
//! is established by measurement (the doctor's pass, Spec 11 §7), not by
//! construction. The `measured` block starts empty and the doctor fills it.

use crate::{DType, IrError, LayoutId};

/// Device family (Spec 1 App. A).
///
/// `Cpu` is the T0/T0v device (Spec 4 §2): the scalar reference and SIMD
/// paths run there, and hosted CI claims only this tier (Spec 14 §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchFamily {
    /// RDNA4, e.g. gfx1201 (Spec 1 App. A).
    Rdna4,
    /// RDNA3 (Spec 1 App. A).
    Rdna3,
    /// CDNA3 (Spec 1 App. A).
    Cdna3,
    /// Portable reference device (Spec 1 App. A).
    Reference,
    /// The T0/T0v CPU device (Spec 1 App. A, Spec 4 §2).
    Cpu,
}

/// Relative matrix-throughput rate (Spec 1 App. A `RelRate`).
///
/// Multiple of the f16/bf16 baseline: gfx1201 lists f16/bf16 at 1×,
/// e4m3/e5m2, iu8 and iu4 at 2× nominal (Spec 1 App. A).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelRate(f32);

impl RelRate {
    /// Builds a rate; must be finite and positive (Spec 1 App. A).
    pub fn new(value: f32) -> Result<Self, IrError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(IrError::NonPositiveRate { value })
        }
    }

    /// Baseline multiple (1.0 = f16/bf16 rate on gfx1201).
    pub const fn as_f32(self) -> f32 {
        self.0
    }
}

/// One WMMA/MFMA form with its relative throughput
/// (Spec 1 App. A `matrix_ops`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixOp {
    /// Tile shape `(m, n, k)`, e.g. `(16, 16, 16)` (Spec 1 App. A).
    pub shape: [u32; 3],
    /// First operand dtype (Spec 1 App. A).
    pub a: DType,
    /// Second operand dtype; `e5m2` appears here only (Spec 1 §2.1).
    pub b: DType,
    /// Accumulator dtype: f32 or i32, never f16/bf16 (Spec 1 §6.1).
    pub acc: DType,
    /// Relative throughput (Spec 1 App. A).
    pub rate: RelRate,
}

impl MatrixOp {
    /// Builds a matrix-op entry; tile dims must be nonzero.
    pub fn new(
        shape: [u32; 3],
        a: DType,
        b: DType,
        acc: DType,
        rate: RelRate,
    ) -> Result<Self, IrError> {
        let mut problems = Vec::new();
        for (axis, dim) in shape.iter().enumerate() {
            if *dim == 0 {
                problems.push(IrError::ZeroExtent { axis });
            }
        }
        if !matches!(acc, DType::F32 | DType::I32) {
            problems.push(IrError::InvalidAccumulator { got: acc });
        }
        if problems.is_empty() {
            Ok(Self {
                shape,
                a,
                b,
                acc,
                rate,
            })
        } else if problems.len() == 1 {
            Err(problems
                // Internal invariant: this branch runs only when len == 1.
                .pop()
                .expect("problems holds exactly one entry"))
        } else {
            Err(IrError::Multiple {
                problems: problems.into_boxed_slice(),
            })
        }
    }
}

/// VALU dot-product forms (Spec 1 App. A `valu_dot`).
///
/// Closed enum for the named forms. A new form lands via RFC (Spec 1 §7);
// DECISION(A1.1): gfx1201() lists only dot4_i32_i8, the one form the spec
// names as present on gfx1201 ("dot4_i32_i8 present", App. A); rejected
// adding the dot2 forms without a spec statement for this arch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValuDot {
    /// `v_dot4`-style i8 dot product with i32 accumulation (Spec 1 App. A).
    Dot4I32I8,
    /// f16 dot product with f32 accumulation (Spec 1 App. A).
    Dot2F32F16,
    /// bf16 dot product with f32 accumulation (Spec 1 App. A).
    Dot2F32Bf16,
}

/// hipGraph capture support (Spec 1 App. A `graph_capture`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphCapture {
    /// Graphs capture and replay reliably (Spec 1 App. A; gfx1201).
    Supported,
    /// Captures but unreliably; the scheduler must not depend on it
    /// (Spec 1 App. A).
    Unstable,
    /// No capture support, e.g. the CPU reference device (Spec 1 App. A).
    None,
}

/// Peer transport for collectives (Spec 1 App. A `p2p`, Spec 1 §4.G).
///
/// The op is the same either way; transport only changes how the runtime
/// moves the bytes: P2P where the descriptor says the pair supports it,
/// host-staged otherwise (Spec 1 §4.G).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum P2pTransport {
    /// Direct peer access (Spec 1 App. A).
    Direct,
    /// Staged through host memory (Spec 1 App. A).
    HostStaged,
}

/// One peer link entry, mirrored in `Topology.links` (Spec 5 §2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct P2pLink {
    /// Peer rank.
    pub peer_rank: u32,
    /// How bytes move to that peer (Spec 1 §4.G).
    pub transport: P2pTransport,
    /// Measured throughput, once the doctor measures it; `None` until then.
    // DECISION(A1.1): Option, None until measured — mirrors the measured
    // block below. Rejected 0.0, which would read as "measured zero".
    pub measured_gbps: Option<f32>,
}

/// Doctor-measured values (Spec 1 App. A `measured`, Spec 11 §7).
///
/// Empty until the doctor's measurement pass fills it; spec-sheet values live
/// on [`ArchDescriptor`] itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    /// Measured memory bandwidth in GB/s (Spec 1 App. A).
    pub mem_bw_gbps: Option<f32>,
    /// Kernel dispatch overhead in µs (Spec 1 App. A).
    pub dispatch_overhead_us: Option<f32>,
    /// Measured matrix rates, parallel to `matrix_ops` (Spec 1 App. A).
    pub matrix_rates: Vec<RelRate>,
    /// Host-to-device bandwidth in GB/s (Spec 1 App. A).
    pub h2d_gbps: Option<f32>,
    /// Device-to-host bandwidth in GB/s (Spec 1 App. A).
    pub d2h_gbps: Option<f32>,
}

impl Measured {
    /// Empty measurement block: the pre-doctor state (Spec 1 App. A:
    /// "empty until then").
    pub fn empty() -> Self {
        Self {
            mem_bw_gbps: None,
            dispatch_overhead_us: None,
            matrix_rates: Vec::new(),
            h2d_gbps: None,
            d2h_gbps: None,
        }
    }

    /// True when no measurement has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.mem_bw_gbps.is_none()
            && self.dispatch_overhead_us.is_none()
            && self.matrix_rates.is_empty()
            && self.h2d_gbps.is_none()
            && self.d2h_gbps.is_none()
    }
}

/// Per-device hardware description (Spec 1 App. A).
///
/// The only source of hardware facts: the planner also gets a link matrix
/// (Spec 5 §2), and per-pair entries here mirror `Topology.links`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchDescriptor {
    /// Device name, e.g. `"gfx1201"` (Spec 1 App. A).
    pub name: String,
    /// Device family (Spec 1 App. A).
    pub family: ArchFamily,
    /// Wavefront size, 32 on gfx1201 (Spec 1 App. A).
    pub wave_size: u32,
    /// Compute-unit count, 64 on gfx1201 (Spec 1 App. A).
    pub cu_count: u32,
    /// LDS bytes per workgroup, 64 KiB on gfx1201 (Spec 1 App. A).
    pub lds_bytes_per_wg: u32,
    /// VGPRs available per lane (Spec 1 App. A).
    pub vgprs_per_lane: u32,
    /// L2 cache size in bytes (Spec 1 App. A).
    pub l2_bytes: u64,
    /// Infinity-cache size in bytes; 0 if none (Spec 1 App. A).
    pub l3_bytes: u64,
    /// VRAM in bytes; 32 GiB on gfx1201 (Spec 1 App. A).
    pub vram_bytes: u64,
    /// Spec-sheet bandwidth in GB/s; measured value lives in
    /// [`Measured::mem_bw_gbps`] (Spec 1 App. A).
    pub mem_bw_gbps: f32,
    /// Clock in MHz (Spec 1 App. A).
    pub clock_mhz: f32,
    /// Complete list of WMMA/MFMA forms with relative throughput
    /// (Spec 1 App. A).
    pub matrix_ops: Vec<MatrixOp>,
    /// VALU dot forms available (Spec 1 App. A).
    pub valu_dot: Vec<ValuDot>,
    /// Hardware conversion to/from e4m3/e5m2 (Spec 1 App. A; present on
    /// gfx1201).
    pub fp8_convert: bool,
    /// SWMMAC / 2:4 structured-sparse support (Spec 1 App. A; present on
    /// gfx1201).
    pub sparse_matrix: bool,
    /// Native B-fragment order the zero-copy loader checks against
    /// (Spec 1 App. A, Spec 2 §2.4).
    pub fragment_layout: LayoutId,
    /// Intra-block K/V element order the attention kernels use
    /// (Spec 1 App. A, Spec 3 §3.2, Spec 4 §5.3).
    pub attention_layout: LayoutId,
    /// Maximum workgroup size (Spec 1 App. A).
    pub max_wg_size: u32,
    /// hipGraph capture support (Spec 1 App. A).
    pub graph_capture: GraphCapture,
    /// Doctor-measured values; empty until measured (Spec 1 App. A,
    /// Spec 11 §7).
    pub measured: Measured,
    /// Peer links, mirrored in `Topology.links` (Spec 1 App. A, Spec 5 §2).
    pub p2p: Vec<P2pLink>,
}

impl ArchDescriptor {
    /// gfx1201 (R9700) initial values, to be overwritten by measurement
    /// (Spec 1 App. A).
    ///
    /// Spec-stated: wave 32; 64 CUs; 64 KiB LDS/WG; f16/bf16 at 1×,
    /// e4m3/e5m2, iu8 and iu4 (nominal, verify) at 2×; `dot4_i32_i8` present;
    /// fp8 convert present; SWMMAC present; 32 GiB VRAM; 640 GB/s spec
    /// bandwidth; graph capture `Supported`. `fragment_layout` is `L1`: the
    /// gfx12 native fragment order equals `L1` (Spec 2 §2.2).
    // DECISION(A1.1): fields omitted from App. A's initial-value sentence use
    // the reference gfx1201 capabilities recorded by the A0 spike hardware
    // fingerprints: 256 addressable VGPRs/lane, 8 MiB L2, 64 MiB L3, 2350 MHz,
    // and 1024 threads/WG. Rejected zero sentinels because Spec 4 §4.2 uses
    // these fields as hard search-space limits. `measured` remains empty and
    // p2p remains per-device/topology data; SI-2 owns the current rig's link.
    // DECISION(A1.1): the e5m2 matrix entry pairs an e4m3 first operand.
    // Spec 1 §2.1 constrains e5m2 to the second operand only but never names
    // the first; rejected omitting e5m2 (App. A lists it at 2×) and rejected
    // an (e5m2, e5m2) entry (contradicts §2.1). Spec 4 owns the correction.
    pub fn gfx1201() -> Self {
        let rate =
            |v: f32| RelRate::new(v).expect("A1.1 gfx1201 rates are positive finite literals");
        let mat = |shape: [u32; 3], a: DType, b: DType, acc: DType, r: f32| {
            MatrixOp::new(shape, a, b, acc, rate(r))
                .expect("A1.1 gfx1201 tile shapes are nonzero literals")
        };
        Self {
            name: "gfx1201".to_owned(),
            family: ArchFamily::Rdna4,
            wave_size: 32,
            cu_count: 64,
            lds_bytes_per_wg: 64 * 1024,
            vgprs_per_lane: 256,
            l2_bytes: 8 * 1024 * 1024,
            l3_bytes: 64 * 1024 * 1024,
            vram_bytes: 32 * 1024 * 1024 * 1024,
            mem_bw_gbps: 640.0,
            clock_mhz: 2350.0,
            matrix_ops: vec![
                mat([16, 16, 16], DType::F16, DType::F16, DType::F32, 1.0),
                mat([16, 16, 16], DType::Bf16, DType::Bf16, DType::F32, 1.0),
                mat([16, 16, 16], DType::E4m3, DType::E4m3, DType::F32, 2.0),
                mat([16, 16, 16], DType::E4m3, DType::E5m2, DType::F32, 2.0),
                mat([16, 16, 16], DType::I8, DType::I8, DType::I32, 2.0),
                mat([16, 16, 32], DType::I4, DType::I4, DType::I32, 2.0),
            ],
            valu_dot: vec![ValuDot::Dot4I32I8],
            fp8_convert: true,
            sparse_matrix: true,
            fragment_layout: LayoutId::L1,
            attention_layout: LayoutId::ATTENTION_GFX1201,
            max_wg_size: 1024,
            graph_capture: GraphCapture::Supported,
            measured: Measured::empty(),
            p2p: Vec::new(),
        }
    }

    /// The CPU reference device for T0/T0v (Spec 1 App. A family `CPU`,
    /// Spec 4 §2).
    ///
    /// Reports the scalar-reference identity: one lane, no matrix units, no
    /// capture. Numeric fields are unset (0/empty): the spec assigns no values
    /// here and the cost model must not mistake them for hardware facts.
    // DECISION(A1.1): wave 1 / single CU / Contiguous layouts / capture None
    // as the scalar-truth identity; rejected fabricating host core counts,
    // SIMD widths or DRAM bandwidth, which vary per machine and belong to
    // doctor measurement, not to a checked-in constructor.
    pub fn cpu() -> Self {
        Self {
            name: "cpu".to_owned(),
            family: ArchFamily::Cpu,
            wave_size: 1,
            cu_count: 1,
            lds_bytes_per_wg: 0,
            vgprs_per_lane: 0,
            l2_bytes: 0,
            l3_bytes: 0,
            vram_bytes: 0,
            mem_bw_gbps: 0.0,
            clock_mhz: 0.0,
            matrix_ops: Vec::new(),
            valu_dot: Vec::new(),
            fp8_convert: false,
            sparse_matrix: false,
            fragment_layout: LayoutId::CONTIGUOUS,
            attention_layout: LayoutId::CONTIGUOUS,
            max_wg_size: 1,
            graph_capture: GraphCapture::None,
            measured: Measured::empty(),
            p2p: Vec::new(),
        }
    }
}
