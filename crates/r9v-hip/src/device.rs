// SPDX-License-Identifier: Apache-2.0
//! Execution target devices, stable identities, and discovered device properties (Spec 5 §3.4, Spec 14 §2, §3).

use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use crate::error::{HipError, Result};
use crate::library::HipLibrary;
use crate::pcie::PciPathDiscovery;

/// Canonical PCI Bus/Device/Function address representing a physical PCI location (Spec 5 §3, Spec 14 §3).
///
/// Formats as `dddd:bb:dd.f` (e.g. `0000:03:00.0`), matching standard Linux sysfs naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PciBdf {
    /// PCI domain (segment) identifier.
    domain: u16,
    /// PCI bus identifier (0-255).
    bus: u8,
    /// PCI device (slot) identifier (0-31).
    device: u8,
    /// PCI function identifier (0-7).
    function: u8,
}

impl PciBdf {
    /// Creates a PCI BDF from trusted, already-decoded components.
    ///
    /// Panics if `device` or `function` exceeds the widths defined by PCI.
    /// Untrusted values must use [`Self::parse`] or [`Self::from_hip_fields`],
    /// which return typed errors.
    pub const fn new(domain: u16, bus: u8, device: u8, function: u8) -> Self {
        assert!(device <= 0x1f, "PCI device number must fit five bits");
        assert!(function <= 0x07, "PCI function number must fit three bits");
        Self {
            domain,
            bus,
            device,
            function,
        }
    }

    /// PCI domain (segment) identifier.
    pub const fn domain(self) -> u16 {
        self.domain
    }

    /// PCI bus identifier.
    pub const fn bus(self) -> u8 {
        self.bus
    }

    /// PCI device (slot) identifier.
    pub const fn device(self) -> u8 {
        self.device
    }

    /// PCI function identifier.
    pub const fn function(self) -> u8 {
        self.function
    }

    /// Validates the domain, bus and device fields exposed by
    /// `hipDeviceProp_t`. HIP does not expose the PCI function in that struct;
    /// The provisional value uses function zero and must be replaced by
    /// `hipDeviceGetPCIBusId` before it becomes a persistent identity.
    pub fn from_hip_fields(domain: i32, bus: i32, device: i32) -> Result<Self> {
        if !(0..=0xffff).contains(&domain) {
            return Err(HipError::InvalidPciBdf {
                input: format!("{domain}:{bus}:{device}"),
                details: "HIP PCI domain is outside 0x0000..0xffff",
            });
        }
        if !(0..=0xff).contains(&bus) {
            return Err(HipError::InvalidPciBdf {
                input: format!("{domain}:{bus}:{device}"),
                details: "HIP PCI bus is outside 0x00..0xff",
            });
        }
        if !(0..=0x1f).contains(&device) {
            return Err(HipError::InvalidPciBdf {
                input: format!("{domain}:{bus}:{device}"),
                details: "HIP PCI device is outside 0x00..0x1f",
            });
        }
        Ok(Self::new(domain as u16, bus as u8, device as u8, 0))
    }

    /// Parses a PCI BDF string in standard `[dddd:]bb:dd.f` format.
    pub fn parse(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        let dot_pos = trimmed.rfind('.').ok_or_else(|| HipError::InvalidPciBdf {
            input: s.to_string(),
            details: "missing '.' before function number",
        })?;
        let fn_str = &trimmed[dot_pos + 1..];
        if fn_str.is_empty() || fn_str.len() > 2 {
            return Err(HipError::InvalidPciBdf {
                input: s.to_string(),
                details: "function component must be 1-2 hex digits (0-7)",
            });
        }
        let function = u8::from_str_radix(fn_str, 16).map_err(|_| HipError::InvalidPciBdf {
            input: s.to_string(),
            details: "function component contains invalid hex digits",
        })?;
        if function > 7 {
            return Err(HipError::InvalidPciBdf {
                input: s.to_string(),
                details: "PCI function number must be 0-7",
            });
        }

        let prefix = &trimmed[..dot_pos];
        let colon_parts: Vec<&str> = prefix.split(':').collect();
        let (domain, bus, device) = match colon_parts.len() {
            2 => {
                let bus = u8::from_str_radix(colon_parts[0], 16).map_err(|_| {
                    HipError::InvalidPciBdf {
                        input: s.to_string(),
                        details: "bus component contains invalid hex digits",
                    }
                })?;
                let device = u8::from_str_radix(colon_parts[1], 16).map_err(|_| {
                    HipError::InvalidPciBdf {
                        input: s.to_string(),
                        details: "device component contains invalid hex digits",
                    }
                })?;
                (0u16, bus, device)
            }
            3 => {
                let domain = u16::from_str_radix(colon_parts[0], 16).map_err(|_| {
                    HipError::InvalidPciBdf {
                        input: s.to_string(),
                        details: "domain component contains invalid hex digits",
                    }
                })?;
                let bus = u8::from_str_radix(colon_parts[1], 16).map_err(|_| {
                    HipError::InvalidPciBdf {
                        input: s.to_string(),
                        details: "bus component contains invalid hex digits",
                    }
                })?;
                let device = u8::from_str_radix(colon_parts[2], 16).map_err(|_| {
                    HipError::InvalidPciBdf {
                        input: s.to_string(),
                        details: "device component contains invalid hex digits",
                    }
                })?;
                (domain, bus, device)
            }
            _ => {
                return Err(HipError::InvalidPciBdf {
                    input: s.to_string(),
                    details: "PCI BDF must format as [domain:]bus:device.function",
                });
            }
        };

        if device > 31 {
            return Err(HipError::InvalidPciBdf {
                input: s.to_string(),
                details: "PCI device number must be 0-31 (0x00-0x1f)",
            });
        }

        Ok(Self {
            domain,
            bus,
            device,
            function,
        })
    }
}

impl fmt::Display for PciBdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

impl FromStr for PciBdf {
    type Err = HipError;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// 16-byte AMD HIP device UUID (Spec 14 §3).
///
/// Formats as `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` per ROCm canonical convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HipUuid(pub [u8; 16]);

impl HipUuid {
    /// Creates a new [`HipUuid`] from 16 raw bytes.
    pub const fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Creates a new [`HipUuid`] from 16 raw bytes.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Returns a reference to the 16 raw bytes.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Returns the 16 raw bytes as an array.
    pub const fn bytes(&self) -> [u8; 16] {
        self.0
    }

    /// Returns `true` if all 16 bytes are zero (uninitialized or absent UUID).
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }

    /// Parses a UUID string, with or without `GPU-` prefix and with or without hyphens.
    pub fn parse(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        let hex_str = trimmed
            .strip_prefix("GPU-")
            .or_else(|| trimmed.strip_prefix("gpu-"))
            .unwrap_or(trimmed);
        let hex_clean: String = hex_str.chars().filter(|c| *c != '-').collect();
        if hex_clean.len() != 32 {
            return Err(HipError::InvalidHipUuid {
                input: s.to_owned(),
                details: format!("expected 32 hex digits, got {}", hex_clean.len()),
            });
        }
        let mut bytes = [0u8; 16];
        for i in 0..16 {
            bytes[i] = u8::from_str_radix(&hex_clean[i * 2..i * 2 + 2], 16).map_err(|_| {
                HipError::InvalidHipUuid {
                    input: s.to_owned(),
                    details: "contains a non-hex digit".to_owned(),
                }
            })?;
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for HipUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3],
            self.0[4], self.0[5],
            self.0[6], self.0[7],
            self.0[8], self.0[9],
            self.0[10], self.0[11], self.0[12], self.0[13], self.0[14], self.0[15],
        )
    }
}

impl From<[u8; 16]> for HipUuid {
    fn from(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
}

impl From<HipUuid> for [u8; 16] {
    fn from(uuid: HipUuid) -> Self {
        uuid.0
    }
}

/// Ephemeral process-local device ordinal index used for HIP runtime operations (Spec 14 §3).
///
/// HIP ordinals (0, 1, ... N-1) are explicitly ephemeral handles: they are assigned
/// dynamically by the HIP runtime and affected by environment variables such as
/// `HIP_VISIBLE_DEVICES`. They must NOT be used for persistent caching or cross-process
/// identity. For stable physical identity, use [`DeviceIdentity`] or [`PciBdf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HipOrdinal(u32);

impl HipOrdinal {
    /// Creates a new ephemeral HIP ordinal handle.
    pub const fn new(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Returns the ordinal index as a `u32`.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Returns the ordinal index as an `i32` for direct HIP C API calls.
    pub fn as_i32(self) -> Result<i32> {
        i32::try_from(self.0).map_err(|_| HipError::InvalidDeviceOrdinal {
            ordinal: i64::from(self.0),
        })
    }
}

impl fmt::Display for HipOrdinal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for HipOrdinal {
    fn from(val: u32) -> Self {
        Self(val)
    }
}

impl From<HipOrdinal> for u32 {
    fn from(ord: HipOrdinal) -> Self {
        ord.0
    }
}

/// Stable physical identity used by fingerprints, hardware descriptors, and cached plans (Spec 5 §2).
///
/// HIP ordinals are process-local ephemeral handles and are intentionally absent here.
/// A physical GPU is identified by its canonical PCI BDF address and optional hardware UUID.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceIdentity {
    /// The portable host CPU execution device.
    Cpu,
    /// A physical GPU identity.
    Gpu {
        /// Runtime-reported 16-byte UUID, when nonzero and available.
        uuid: Option<HipUuid>,
        /// Canonical PCI BDF address.
        pci_bdf: PciBdf,
    },
}

impl DeviceIdentity {
    /// Returns the CPU device identity.
    pub const fn cpu() -> Self {
        Self::Cpu
    }

    /// Returns a GPU device identity with given UUID and PCI BDF.
    pub const fn gpu(uuid: Option<HipUuid>, pci_bdf: PciBdf) -> Self {
        Self::Gpu { uuid, pci_bdf }
    }

    /// Returns `true` if this is the CPU identity.
    pub const fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns `true` if this is a GPU identity.
    pub const fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu { .. })
    }

    /// Returns the canonical PCI BDF address if this is a GPU identity.
    pub const fn pci_bdf(&self) -> Option<PciBdf> {
        match self {
            Self::Cpu => None,
            Self::Gpu { pci_bdf, .. } => Some(*pci_bdf),
        }
    }

    /// Returns the HIP UUID if this is a GPU identity with a valid UUID.
    pub const fn uuid(&self) -> Option<HipUuid> {
        match self {
            Self::Cpu => None,
            Self::Gpu { uuid, .. } => *uuid,
        }
    }
}

impl fmt::Display for DeviceIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu {
                uuid: Some(u),
                pci_bdf,
            } => write!(f, "{pci_bdf} ({u})"),
            Self::Gpu {
                uuid: None,
                pci_bdf,
            } => write!(f, "{pci_bdf}"),
        }
    }
}

/// Execution target device for buffers, operators, and kernel launches (Spec 5 §3.4, Spec 14 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Device {
    /// Host CPU execution tier (Spec 4 §1, Spec 14 §3).
    Cpu,
    /// AMD HIP GPU addressed by an ephemeral process-local ordinal.
    Hip(HipOrdinal),
}

impl Device {
    /// Returns the CPU device target (Spec 5 §3.4, Spec 14 §3).
    pub const fn cpu() -> Self {
        Self::Cpu
    }

    /// Returns a HIP GPU device target with the given rank (Spec 14 §3).
    pub const fn hip(rank: u32) -> Self {
        Self::Hip(HipOrdinal::new(rank))
    }

    /// Returns a HIP GPU device target with the given ephemeral ordinal (Spec 14 §3).
    pub const fn from_ordinal(ordinal: HipOrdinal) -> Self {
        Self::Hip(ordinal)
    }

    /// Returns `true` if this is the CPU target (Spec 5 §3.4, Spec 14 §3).
    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns `true` if this is a HIP GPU target (Spec 14 §3).
    pub const fn is_hip(self) -> bool {
        matches!(self, Self::Hip(_))
    }

    /// Returns the HIP device rank if this is a HIP device, or `None` if CPU (Spec 14 §3).
    pub const fn hip_rank(self) -> Option<u32> {
        match self {
            Self::Cpu => None,
            Self::Hip(ordinal) => Some(ordinal.as_u32()),
        }
    }

    /// Returns the ephemeral HIP ordinal handle if this is a HIP device, or `None` if CPU (Spec 14 §3).
    pub const fn hip_ordinal(self) -> Option<HipOrdinal> {
        match self {
            Self::Cpu => None,
            Self::Hip(ordinal) => Some(ordinal),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Hip(ordinal) => write!(f, "hip:{ordinal}"),
        }
    }
}

/// Safely extracts a bounded, trimmed UTF-8 string from a fixed-size driver char buffer (Spec 14 §3).
pub(crate) fn parse_fixed_c_string(bytes: &[std::ffi::c_char]) -> String {
    let u8_slice = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u8, bytes.len()) };
    let len = u8_slice
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(u8_slice.len());
    String::from_utf8_lossy(&u8_slice[..len]).trim().to_owned()
}

/// Discovered hardware capabilities and resource limits of a HIP GPU (Spec 14 §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProperties {
    /// Human-readable marketing and device name (Spec 14 §3).
    pub name: String,
    /// Total available global device memory in bytes (Spec 14 §3).
    pub total_global_mem: u64,
    /// Shared memory available per thread block in bytes (LDS) (Spec 14 §3).
    pub shared_mem_per_block: usize,
    /// 32-bit registers available per block (Spec 14 §3).
    pub regs_per_block: i32,
    /// SIMD execution width (warp/wavefront size; 32 or 64) (Spec 14 §3).
    pub warp_size: i32,
    /// Maximum work items per workgroup/block (Spec 14 §3).
    pub max_threads_per_block: i32,
    /// Maximum dimensions of a thread block [X, Y, Z] (Spec 14 §3).
    pub max_threads_dim: [i32; 3],
    /// Maximum dimensions of a grid [X, Y, Z] (Spec 14 §3).
    pub max_grid_size: [i32; 3],
    /// Core clock frequency in kilohertz (Spec 14 §3).
    pub clock_rate_khz: i32,
    /// Major compute capability architecture version (Spec 14 §3).
    pub major: i32,
    /// Minor compute capability architecture version (Spec 14 §3).
    pub minor: i32,
    /// Number of Compute Units (CUs) or Workgroup Processors (WGPs) (Spec 14 §3).
    pub multi_processor_count: i32,
    /// AMD GCN/RDNA ISA target identifier (e.g. "amdgcn-amd-amdhsa--gfx1201") (Spec 14 §3).
    pub gcn_arch_name: String,
    /// PCI bus identifier for device topology mapping (Spec 5 §3, Spec 14 §3).
    pub pci_bus_id: i32,
    /// PCI device identifier (Spec 5 §3, Spec 14 §3).
    pub pci_device_id: i32,
    /// PCI domain identifier (Spec 5 §3, Spec 14 §3).
    pub pci_domain_id: i32,
    /// Whether the device resides on a multi-GPU board (Spec 14 §3).
    pub is_multi_gpu_board: bool,
    /// Whether the device supports mapped host memory (Spec 14 §3).
    pub can_map_host_memory: bool,
    /// Whether concurrent kernel launches are supported (Spec 14 §3).
    pub concurrent_kernels: bool,
    /// Whether Error-Correcting Code memory is enabled (Spec 14 §3).
    pub ecc_enabled: bool,
    /// Whether cooperative kernel launch is supported (Spec 14 §3).
    pub cooperative_launch: bool,
    /// 16-byte HIP UUID reported by the driver, when non-zero.
    pub uuid: Option<HipUuid>,
    /// Typed PCI BDF address of the GPU device.
    pub pci_bdf: PciBdf,
}

impl DeviceProperties {
    /// Returns the stable physical identity corresponding to these device properties.
    pub fn identity(&self) -> DeviceIdentity {
        DeviceIdentity::Gpu {
            uuid: self.uuid,
            pci_bdf: self.pci_bdf,
        }
    }
}

/// Comprehensive record of an enumerated HIP GPU device (Spec 14 §2, §3).
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredDevice {
    /// Process-local ephemeral HIP ordinal handle used for runtime API calls.
    pub ordinal: HipOrdinal,
    /// Stable physical device identity (PCI BDF and optional UUID).
    pub identity: DeviceIdentity,
    /// Discovered hardware capabilities and resource limits.
    pub properties: DeviceProperties,
    /// Discovered Linux sysfs PCIe path topology and bottleneck analysis.
    /// A GPU inventory is not constructed if this path is unavailable.
    pub pcie_path: PciPathDiscovery,
}

impl DiscoveredDevice {
    /// Returns the process-local ephemeral HIP ordinal handle.
    pub fn ordinal(&self) -> HipOrdinal {
        self.ordinal
    }

    /// Returns the stable physical identity of the device.
    pub fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    /// Returns the typed PCI BDF address of the device.
    pub fn pci_bdf(&self) -> PciBdf {
        self.properties.pci_bdf
    }

    /// Returns the device UUID, if present and non-zero.
    pub fn uuid(&self) -> Option<HipUuid> {
        self.properties.uuid
    }

    /// Returns the device name.
    pub fn name(&self) -> &str {
        &self.properties.name
    }

    /// Returns the AMD GCN architecture target string.
    pub fn gcn_arch_name(&self) -> &str {
        &self.properties.gcn_arch_name
    }

    /// Returns the discovered PCIe bottleneck payload bandwidth in GB/s, if available.
    pub fn bottleneck_bandwidth_gbps(&self) -> Option<f64> {
        self.pcie_path.bottleneck_bandwidth_gbps()
    }
}

/// CPU-safe system device inventory (Spec 14 §2, §3).
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInventory {
    /// Discovered GPU devices. Empty if no HIP runtime is installed or 0 GPUs are present.
    pub gpus: Vec<DiscoveredDevice>,
}

impl DeviceInventory {
    /// Discovers system devices using default library discovery and `/sys`.
    ///
    /// If the HIP runtime is absent, returns an empty GPU list without error.
    /// If the HIP runtime is present but broken, returns a typed error.
    pub fn discover() -> Result<Self> {
        Self::discover_with_sys_root(Path::new("/sys"))
    }

    /// Discovers system devices using default library discovery and a specified sysfs root.
    pub fn discover_with_sys_root(sys_root: &Path) -> Result<Self> {
        Self::from_runtime_result(HipLibrary::load_system(), sys_root)
    }

    fn from_runtime_result(runtime: Result<Arc<HipLibrary>>, sys_root: &Path) -> Result<Self> {
        match runtime {
            Ok(lib) => {
                let gpus = lib.enumerate_devices_with_sys_root(sys_root)?;
                Ok(Self { gpus })
            }
            Err(HipError::LibraryNotFound { .. }) => {
                // Missing HIP library is normal empty-GPU state for CPU execution
                Ok(Self { gpus: Vec::new() })
            }
            Err(e) => Err(e),
        }
    }

    /// Creates an inventory from an explicitly loaded HIP library and sysfs root.
    pub fn from_library_with_sys_root(lib: &Arc<HipLibrary>, sys_root: &Path) -> Result<Self> {
        let gpus = lib.enumerate_devices_with_sys_root(sys_root)?;
        Ok(Self { gpus })
    }

    /// Creates an inventory from an explicitly loaded HIP library using default `/sys`.
    pub fn from_library(lib: &Arc<HipLibrary>) -> Result<Self> {
        Self::from_library_with_sys_root(lib, Path::new("/sys"))
    }

    /// Returns the number of discovered GPU devices.
    pub fn gpu_count(&self) -> usize {
        self.gpus.len()
    }

    /// The portable CPU execution device is unconditional.
    pub const fn has_cpu(&self) -> bool {
        true
    }

    /// Returns `true` if at least one GPU is available.
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    /// Finds a discovered GPU device by its ephemeral ordinal handle.
    pub fn gpu_by_ordinal(&self, ordinal: HipOrdinal) -> Option<&DiscoveredDevice> {
        self.gpus.iter().find(|d| d.ordinal == ordinal)
    }

    /// Finds a discovered GPU device by its stable PCI BDF address.
    pub fn gpu_by_bdf(&self, bdf: PciBdf) -> Option<&DiscoveredDevice> {
        self.gpus.iter().find(|d| d.properties.pci_bdf == bdf)
    }

    /// Finds a discovered GPU device by its HIP UUID.
    pub fn gpu_by_uuid(&self, uuid: &HipUuid) -> Option<&DiscoveredDevice> {
        self.gpus
            .iter()
            .find(|d| d.properties.uuid.as_ref() == Some(uuid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_char;

    #[test]
    fn test_fixed_c_string_parsing_bounded_without_nul() {
        // Array of non-NUL ASCII characters
        let filled_no_nul = ['A' as c_char; 256];
        let parsed = parse_fixed_c_string(&filled_no_nul);
        assert_eq!(parsed.len(), 256);
        assert_eq!(&parsed[..4], "AAAA");

        // Array with interior NUL
        let mut with_nul = ['B' as c_char; 256];
        with_nul[10] = 0;
        let parsed_nul = parse_fixed_c_string(&with_nul);
        assert_eq!(parsed_nul.len(), 10);
        assert_eq!(parsed_nul, "BBBBBBBBBB");
    }

    #[test]
    fn test_pci_bdf_parsing_and_display() {
        let bdf = PciBdf::parse("0000:03:00.0").unwrap();
        assert_eq!(bdf.domain(), 0);
        assert_eq!(bdf.bus(), 3);
        assert_eq!(bdf.device(), 0);
        assert_eq!(bdf.function(), 0);
        assert_eq!(bdf.to_string(), "0000:03:00.0");

        let bdf2 = PciBdf::parse("03:00.0").unwrap();
        assert_eq!(bdf2.domain(), 0);
        assert_eq!(bdf2.bus(), 3);
        assert_eq!(bdf2.device(), 0);
        assert_eq!(bdf2.function(), 0);
        assert_eq!(bdf2.to_string(), "0000:03:00.0");

        let bdf3 = PciBdf::parse("0001:1a:1f.7").unwrap();
        assert_eq!(bdf3.domain(), 1);
        assert_eq!(bdf3.bus(), 0x1a);
        assert_eq!(bdf3.device(), 31);
        assert_eq!(bdf3.function(), 7);
        assert_eq!(bdf3.to_string(), "0001:1a:1f.7");

        assert!(PciBdf::parse("invalid").is_err());
        assert!(PciBdf::parse("00:00").is_err());
        assert!(PciBdf::parse("0000:00:20.0").is_err()); // device > 31
        assert!(PciBdf::parse("0000:00:00.8").is_err()); // function > 7
        assert!(PciBdf::from_hip_fields(-1, 0, 0).is_err());
        assert!(PciBdf::from_hip_fields(0, 256, 0).is_err());
        assert!(PciBdf::from_hip_fields(0, 0, 32).is_err());
    }

    #[test]
    fn test_hip_uuid_parsing_and_display() {
        let bytes = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let uuid = HipUuid::new(bytes);
        assert_eq!(uuid.to_string(), "GPU-12345678-9abc-def0-1122-334455667788");
        assert!(!uuid.is_zero());

        let parsed = HipUuid::parse("GPU-12345678-9abc-def0-1122-334455667788").unwrap();
        assert_eq!(parsed, uuid);

        let parsed_no_prefix = HipUuid::parse("123456789abcdef01122334455667788").unwrap();
        assert_eq!(parsed_no_prefix, uuid);

        let zero_uuid = HipUuid::new([0u8; 16]);
        assert!(zero_uuid.is_zero());
    }

    #[test]
    fn test_hip_ordinal_semantics() {
        let ord0 = HipOrdinal::new(0);
        let ord1 = HipOrdinal::new(1);
        assert_eq!(ord0.as_u32(), 0);
        assert_eq!(ord0.as_i32().unwrap(), 0);
        assert_eq!(ord1.as_u32(), 1);
        assert_eq!(ord1.as_i32().unwrap(), 1);
        assert!(HipOrdinal::new(u32::MAX).as_i32().is_err());
        assert_eq!(ord0.to_string(), "0");
        assert!(ord0 < ord1);

        let dev = Device::from_ordinal(ord0);
        assert_eq!(dev, Device::Hip(HipOrdinal::new(0)));
        assert_eq!(dev.hip_ordinal(), Some(ord0));
        assert_eq!(Device::Cpu.hip_ordinal(), None);
    }

    #[test]
    fn test_device_identity_properties() {
        let cpu_id = DeviceIdentity::cpu();
        assert!(cpu_id.is_cpu());
        assert!(!cpu_id.is_gpu());
        assert_eq!(cpu_id.to_string(), "cpu");

        let bdf = PciBdf::new(0, 3, 0, 0);
        let uuid = HipUuid::new([1u8; 16]);
        let gpu_id = DeviceIdentity::gpu(Some(uuid), bdf);
        assert!(!gpu_id.is_cpu());
        assert!(gpu_id.is_gpu());
        assert_eq!(gpu_id.pci_bdf(), Some(bdf));
        assert_eq!(gpu_id.uuid(), Some(uuid));
        assert!(gpu_id.to_string().contains("0000:03:00.0"));
        assert!(gpu_id.to_string().contains("GPU-01010101"));
    }

    #[test]
    fn test_missing_runtime_is_a_deterministic_empty_gpu_inventory() {
        let inventory = DeviceInventory::from_runtime_result(
            Err(HipError::LibraryNotFound {
                searched: vec!["synthetic missing runtime".to_owned()],
            }),
            Path::new("/synthetic-sysfs-not-read"),
        )
        .expect("missing HIP is a normal CPU-only inventory");

        assert!(inventory.has_cpu());
        assert!(inventory.gpus.is_empty());
    }
}
