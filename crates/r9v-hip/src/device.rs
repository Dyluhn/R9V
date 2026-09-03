// SPDX-License-Identifier: Apache-2.0
//! Execution target devices and discovered device properties (Spec 5 §3.4, Spec 14 §2, §3).

use std::fmt;

/// Execution target device for buffers, operators, and kernel launches (Spec 5 §3.4, Spec 14 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Device {
    /// Host CPU execution tier (Spec 4 §1, Spec 14 §3).
    Cpu,
    /// AMD HIP GPU device identified by zero-based device rank (Spec 14 §3).
    Hip(u32),
}

impl Device {
    /// Returns the CPU device target (Spec 5 §3.4, Spec 14 §3).
    pub const fn cpu() -> Self {
        Self::Cpu
    }

    /// Returns a HIP GPU device target with the given rank (Spec 14 §3).
    pub const fn hip(rank: u32) -> Self {
        Self::Hip(rank)
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
            Self::Hip(rank) => Some(rank),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Hip(rank) => write!(f, "hip:{rank}"),
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
}
