// SPDX-License-Identifier: Apache-2.0
//! Raw C types, function signatures, and ABI definitions for AMD HIP (Spec 14 §2, §3).

use std::ffi::{c_char, c_int, c_uint, c_void};

pub(crate) type HipErrorT = i32;
pub(crate) const HIP_SUCCESS: HipErrorT = 0;

pub(crate) const HIP_ERROR_NO_DEVICE: HipErrorT = 100;
pub(crate) const HIP_ERROR_NOT_READY: HipErrorT = 600;
pub(crate) const HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED: HipErrorT = 704;
pub(crate) const HIP_ERROR_PEER_ACCESS_NOT_ENABLED: HipErrorT = 705;

pub(crate) type HipStreamT = *mut c_void;
pub(crate) type HipEventT = *mut c_void;
pub(crate) type HipModuleT = *mut c_void;
pub(crate) type HipFunctionT = *mut c_void;
pub(crate) type HipGraphT = *mut c_void;
pub(crate) type HipGraphExecT = *mut c_void;
pub(crate) type HipGraphNodeT = *mut c_void;

/// Raw C representation of `hipDeviceProp_t` (ROCm 6.0+ `hipDeviceProp_tR0600`, Spec 14 §3).
///
/// Pinned to exactly 1472 bytes matching ROCm 6.x and 7.x ABI.
#[repr(C)]
#[derive(Clone)]
pub(crate) struct RawDeviceProp {
    pub name: [c_char; 256],                // 0..256
    pub uuid: [u8; 16],                     // 256..272
    pub luid: [c_char; 8],                  // 272..280
    pub luid_device_node_mask: c_uint,      // 280..284
    pub _pad0: [u8; 4],                     // 284..288
    pub total_global_mem: usize,            // 288..296
    pub shared_mem_per_block: usize,        // 296..304
    pub regs_per_block: c_int,              // 304..308
    pub warp_size: c_int,                   // 308..312
    pub mem_pitch: usize,                   // 312..320
    pub max_threads_per_block: c_int,       // 320..324
    pub max_threads_dim: [c_int; 3],        // 324..336
    pub max_grid_size: [c_int; 3],          // 336..348
    pub clock_rate: c_int,                  // 348..352
    pub total_const_mem: usize,             // 352..360
    pub major: c_int,                       // 360..364
    pub minor: c_int,                       // 364..368
    pub texture_alignment: usize,           // 368..376
    pub texture_pitch_alignment: usize,     // 376..384
    pub device_overlap: c_int,              // 384..388
    pub multi_processor_count: c_int,       // 388..392
    pub kernel_exec_timeout_enabled: c_int, // 392..396
    pub integrated: c_int,                  // 396..400
    pub can_map_host_memory: c_int,         // 400..404
    pub compute_mode: c_int,                // 404..408
    pub _gap_to_concurrent: [u8; 168],      // 408..576
    pub concurrent_kernels: c_int,          // 576..580
    pub ecc_enabled: c_int,                 // 580..584
    pub pci_bus_id: c_int,                  // 584..588
    pub pci_device_id: c_int,               // 588..592
    pub pci_domain_id: c_int,               // 592..596
    pub _gap_to_multi_gpu: [u8; 60],        // 596..656
    pub is_multi_gpu_board: c_int,          // 656..660
    pub _gap_to_cooperative: [u8; 28],      // 660..688
    pub cooperative_launch: c_int,          // 688..692
    pub _gap_to_arch: [u8; 468],            // 692..1160
    pub gcn_arch_name: [c_char; 256],       // 1160..1416
    pub _gap_end: [u8; 56],                 // 1416..1472
}

const _: () = assert!(std::mem::size_of::<RawDeviceProp>() == 1472);

impl Default for RawDeviceProp {
    fn default() -> Self {
        // SAFETY: RawDeviceProp contains only primitive types (arrays of c_char/u8, usize, c_int, c_uint),
        // all of which are valid when bitwise zero-initialized.
        unsafe { std::mem::zeroed() }
    }
}
