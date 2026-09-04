// SPDX-License-Identifier: Apache-2.0
//! Integration tests for portability hardening: stable identity, ephemeral ordinals,
//! zero/one/three device enumeration, CPU-safe inventory, and PCIe path bottleneck discovery
//! (Spec 5 §3, Spec 14 §2, §3).

mod common;

use r9v_hip::{
    DeviceIdentity, DeviceInventory, HipError, HipLibrary, HipOrdinal, HipUuid, PciBdf,
    PciPathDiscovery,
};
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static TEST_TEMP_COUNTER: AtomicUsize = AtomicUsize::new(1);

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let count = TEST_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("{prefix}_{pid}_{count}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("failed to create temp test directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn make_flat_gpu_sysfs(sys_root: &Path, count: usize) {
    let by_bus = sys_root.join("bus/pci/devices");
    fs::create_dir_all(&by_bus).unwrap();
    for index in 0..count {
        let bus = 3 + index * 4;
        let bdf = format!("0000:{bus:02x}:00.0");
        let endpoint = sys_root.join("devices/pci0000:00").join(&bdf);
        fs::create_dir_all(&endpoint).unwrap();
        fs::write(endpoint.join("current_link_speed"), "32.0 GT/s\n").unwrap();
        fs::write(endpoint.join("current_link_width"), "16\n").unwrap();
        fs::write(endpoint.join("max_link_speed"), "32.0 GT/s\n").unwrap();
        fs::write(endpoint.join("max_link_width"), "16\n").unwrap();
        symlink(&endpoint, by_bus.join(bdf)).unwrap();
    }
}

#[test]
fn test_zero_device_enumeration_and_inventory() {
    let stub_so = common::get_or_compile_stub_with_count(0);
    let lib =
        Arc::new(HipLibrary::load_from_path(&stub_so).expect("failed to load stub with 0 devices"));

    // 1. Device count returns 0
    let count = lib.device_count().expect("device_count must succeed");
    assert_eq!(count, 0, "device count must be 0");

    // 2. Enumeration returns empty list
    let devices = lib
        .enumerate_devices()
        .expect("enumeration with 0 devices must succeed");
    assert!(
        devices.is_empty(),
        "devices list must be empty for 0 devices"
    );

    // 3. Inventory from library returns valid CPU-safe state with 0 GPUs
    let inv = DeviceInventory::from_library(&lib)
        .expect("inventory from library with 0 devices must succeed");
    assert!(inv.has_cpu(), "CPU execution tier must always be available");
    assert_eq!(inv.gpu_count(), 0);
    assert!(!inv.has_gpu());
    assert!(inv.gpus.is_empty());
}

#[test]
fn test_rocm_no_device_status_is_a_zero_gpu_inventory() {
    let stub_so = common::get_or_compile_no_device_stub();
    let lib = Arc::new(HipLibrary::load_from_path(&stub_so).expect("load no-device stub"));

    assert_eq!(lib.device_count().expect("no-device status is normal"), 0);
    let inventory = DeviceInventory::from_library(&lib).expect("construct CPU-only inventory");
    assert!(inventory.has_cpu());
    assert_eq!(inventory.gpu_count(), 0);
}

#[test]
fn test_one_device_enumeration_and_inventory() {
    let temp = TempDirGuard::new("r9v_one_gpu_sysfs");
    make_flat_gpu_sysfs(temp.path(), 1);
    let stub_so = common::get_or_compile_stub_with_count(1);
    let lib =
        Arc::new(HipLibrary::load_from_path(&stub_so).expect("failed to load stub with 1 device"));

    // 1. Device count returns 1
    let count = lib.device_count().expect("device_count must succeed");
    assert_eq!(count, 1, "device count must be 1");

    // 2. Enumeration returns exactly 1 device
    let devices = lib
        .enumerate_devices_with_sys_root(temp.path())
        .expect("enumeration with 1 device must succeed");
    assert_eq!(devices.len(), 1, "devices list must contain 1 device");

    let dev0 = &devices[0];
    assert_eq!(dev0.ordinal(), HipOrdinal::new(0));
    assert_eq!(dev0.pci_bdf(), PciBdf::new(0, 3, 0, 0));
    assert!(dev0.uuid().is_some());
    assert!(!dev0.uuid().unwrap().is_zero());
    assert_eq!(
        dev0.identity(),
        &DeviceIdentity::gpu(dev0.uuid(), dev0.pci_bdf())
    );
    assert_eq!(dev0.name(), "Stub AMD Radeon AI PRO R9700");
    assert_eq!(dev0.gcn_arch_name(), "amdgcn-amd-amdhsa--gfx1201");
    assert_eq!(dev0.properties.warp_size, 32);
    assert_eq!(dev0.properties.multi_processor_count, 64);

    // 3. Inventory records
    let inv = DeviceInventory::from_library_with_sys_root(&lib, temp.path())
        .expect("inventory from library with 1 device must succeed");
    assert!(inv.has_cpu());
    assert_eq!(inv.gpu_count(), 1);
    assert!(inv.has_gpu());
    assert_eq!(
        inv.gpu_by_ordinal(HipOrdinal::new(0)).unwrap().name(),
        "Stub AMD Radeon AI PRO R9700"
    );
    assert_eq!(
        inv.gpu_by_bdf(PciBdf::new(0, 3, 0, 0)).unwrap().ordinal(),
        HipOrdinal::new(0)
    );
}

#[test]
fn test_gpu_inventory_fails_typed_when_full_pcie_path_is_unavailable() {
    let temp = TempDirGuard::new("r9v_missing_gpu_sysfs");
    let stub_so = common::get_or_compile_stub_with_count(1);
    let lib = Arc::new(HipLibrary::load_from_path(&stub_so).expect("load one-device stub"));

    match DeviceInventory::from_library_with_sys_root(&lib, temp.path()) {
        Err(HipError::SysfsError { bdf, .. }) => assert_eq!(bdf, "0000:03:00.0"),
        Err(other) => panic!("expected SysfsError, got: {other}"),
        Ok(_) => panic!("GPU inventory silently accepted missing PCIe topology"),
    }
}

#[test]
fn test_two_device_enumeration_is_not_a_special_case() {
    let temp = TempDirGuard::new("r9v_two_gpu_sysfs");
    make_flat_gpu_sysfs(temp.path(), 2);
    let stub_so = common::get_or_compile_stub_with_count(2);
    let lib = Arc::new(HipLibrary::load_from_path(&stub_so).expect("load two-device stub"));

    let devices = lib
        .enumerate_devices_with_sys_root(temp.path())
        .expect("enumerate two devices");
    assert_eq!(
        devices
            .iter()
            .map(|device| device.ordinal())
            .collect::<Vec<_>>(),
        vec![HipOrdinal::new(0), HipOrdinal::new(1)]
    );
    assert_ne!(devices[0].identity(), devices[1].identity());
}

#[test]
fn test_three_devices_enumeration_ephemeral_ordinals_and_stable_identities() {
    let temp = TempDirGuard::new("r9v_three_gpu_sysfs");
    make_flat_gpu_sysfs(temp.path(), 3);
    let stub_so = common::get_or_compile_stub_with_count(3);
    let lib =
        Arc::new(HipLibrary::load_from_path(&stub_so).expect("failed to load stub with 3 devices"));

    // 1. Device count returns 3
    let count = lib.device_count().expect("device_count must succeed");
    assert_eq!(count, 3, "device count must be 3");

    // 2. Enumeration returns 3 devices
    let devices = lib
        .enumerate_devices_with_sys_root(temp.path())
        .expect("enumeration with 3 devices must succeed");
    assert_eq!(devices.len(), 3, "devices list must contain 3 devices");

    // 3. Verify ephemeral ordinals are sequential 0, 1, 2
    assert_eq!(devices[0].ordinal(), HipOrdinal::new(0));
    assert_eq!(devices[1].ordinal(), HipOrdinal::new(1));
    assert_eq!(devices[2].ordinal(), HipOrdinal::new(2));

    // 4. Verify stable physical identities are distinct and non-overlapping
    assert_eq!(devices[0].pci_bdf(), PciBdf::new(0, 3, 0, 0));
    assert_eq!(devices[1].pci_bdf(), PciBdf::new(0, 7, 0, 0));
    assert_eq!(devices[2].pci_bdf(), PciBdf::new(0, 11, 0, 0));

    let uuid0 = devices[0].uuid().expect("device 0 must have uuid");
    let uuid1 = devices[1].uuid().expect("device 1 must have uuid");
    let uuid2 = devices[2].uuid().expect("device 2 must have uuid");

    assert_ne!(uuid0, uuid1, "UUIDs must be distinct");
    assert_ne!(uuid1, uuid2, "UUIDs must be distinct");
    assert_ne!(uuid0, uuid2, "UUIDs must be distinct");

    assert_ne!(devices[0].identity(), devices[1].identity());
    assert_ne!(devices[1].identity(), devices[2].identity());

    // 5. Test inventory queries
    let inv = DeviceInventory::from_library_with_sys_root(&lib, temp.path())
        .expect("inventory from library with 3 devices must succeed");
    assert!(inv.has_cpu());
    assert_eq!(inv.gpu_count(), 3);
    assert!(inv.has_gpu());

    // Lookups by stable identity vs ephemeral ordinal
    assert_eq!(
        inv.gpu_by_bdf(PciBdf::new(0, 7, 0, 0)).unwrap().ordinal(),
        HipOrdinal::new(1)
    );
    assert_eq!(
        inv.gpu_by_uuid(&uuid2).unwrap().ordinal(),
        HipOrdinal::new(2)
    );
}

#[test]
fn test_missing_library_error_is_typed_without_host_assumptions() {
    let fake_path = Path::new("/nonexistent/test/path/libamdhip64.so.7");
    let result = HipLibrary::load_from_path(fake_path);

    // Loading nonexistent path produces LibraryNotFound
    match result {
        Err(HipError::LibraryNotFound { searched }) => {
            assert!(!searched.is_empty());
        }
        Err(other) => panic!("expected LibraryNotFound, got: {other}"),
        Ok(_) => panic!("expected LibraryNotFound, got Ok"),
    }
}

#[test]
fn test_present_unloadable_library_is_not_treated_as_absent() {
    let temp = TempDirGuard::new("r9v_invalid_hip_library");
    let invalid_so = temp.path().join("libamdhip64.so.7");
    fs::write(&invalid_so, b"not an ELF shared object").unwrap();

    match HipLibrary::load_from_path(&invalid_so) {
        Err(HipError::LibraryLoadFailed { attempts }) => {
            assert!(!attempts.is_empty());
        }
        Err(other) => panic!("expected LibraryLoadFailed, got: {other}"),
        Ok(_) => panic!("invalid shared object unexpectedly loaded"),
    }
}

#[test]
fn test_broken_present_hip_returns_typed_error() {
    let broken_so = common::get_or_compile_broken_stub();
    let lib = Arc::new(
        HipLibrary::load_from_path(&broken_so)
            .expect("loading stub library binary itself succeeds"),
    );

    // Calling inventory with a broken present HIP library must return a typed ApiError,
    // NEVER silently swallowing it as empty-GPU!
    let result = DeviceInventory::from_library(&lib);
    match result {
        Err(HipError::ApiError { op, code, message }) => {
            assert_eq!(op, "hipGetDeviceCount");
            assert_eq!(code, 101);
            assert!(message.contains("hipErrorInvalidDevice") || message.contains("101"));
        }
        other => {
            panic!("expected HipError::ApiError for broken present HIP, but got: {other:?}");
        }
    }
}

#[test]
fn test_pcie_path_discovery_endpoint_x16_upstream_x4_bottleneck() {
    let temp = TempDirGuard::new("r9v_sysfs_test");
    let sys_root = temp.path();

    // Set up synthetic sysfs tree:
    // Root port / bridge: 0000:00:01.0 (Gen4 x4 capacity, Gen1 idle)
    // GPU endpoint:       0000:03:00.0 (Gen5 x16)
    //
    // sys_root/
    //   devices/
    //     pci0000:00/
    //       0000:00:01.0/
    //         current_link_speed = "2.5 GT/s PCIe\n" (idle diagnostic)
    //         current_link_width = "4\n"
    //         max_link_speed = "16.0 GT/s PCIe\n"
    //         max_link_width = "4\n"
    //         0000:03:00.0/
    //           current_link_speed = "32.0 GT/s PCIe\n"
    //           current_link_width = "16\n"
    //           max_link_speed = "32.0 GT/s PCIe\n"
    //           max_link_width = "16\n"
    //   bus/
    //     pci/
    //       devices/
    //         0000:03:00.0 -> ../../../devices/pci0000:00/0000:00:01.0/0000:03:00.0

    let bridge_dir = sys_root.join("devices/pci0000:00/0000:00:01.0");
    let endpoint_dir = bridge_dir.join("0000:03:00.0");
    fs::create_dir_all(&endpoint_dir).expect("failed to create endpoint dir");

    fs::write(bridge_dir.join("current_link_speed"), "2.5 GT/s PCIe\n").unwrap();
    fs::write(bridge_dir.join("current_link_width"), "4\n").unwrap();
    fs::write(bridge_dir.join("max_link_speed"), "16.0 GT/s PCIe\n").unwrap();
    fs::write(bridge_dir.join("max_link_width"), "4\n").unwrap();

    fs::write(endpoint_dir.join("current_link_speed"), "32.0 GT/s PCIe\n").unwrap();
    fs::write(endpoint_dir.join("current_link_width"), "16\n").unwrap();
    fs::write(endpoint_dir.join("max_link_speed"), "32.0 GT/s PCIe\n").unwrap();
    fs::write(endpoint_dir.join("max_link_width"), "16\n").unwrap();

    let bus_devices_dir = sys_root.join("bus/pci/devices");
    fs::create_dir_all(&bus_devices_dir).expect("failed to create bus/pci/devices");
    symlink(&endpoint_dir, bus_devices_dir.join("0000:03:00.0")).expect("failed to create symlink");

    // Perform discovery for 0000:03:00.0
    let bdf = PciBdf::new(0, 3, 0, 0);
    let discovery = PciPathDiscovery::discover(sys_root, bdf)
        .expect("sysfs PCIe discovery must succeed on synthetic tree");

    // 1. Endpoint inspection: verified x16
    assert_eq!(discovery.endpoint.bdf, bdf);
    assert_eq!(discovery.endpoint.current_width, Some(16));
    assert_eq!(discovery.endpoint.current_speed_gts, Some(32.0));
    assert_eq!(discovery.endpoint.max_width, Some(16));
    assert_eq!(discovery.endpoint.max_speed_gts, Some(32.0));

    let ep_bw = discovery.endpoint_bandwidth_gbps().unwrap();
    // 32 GT/s * 16 lanes * (128/130) / 8 = 63.01538... GB/s
    assert!(
        (ep_bw - 63.01538).abs() < 1e-3,
        "endpoint payload bandwidth must be ~63.02 GB/s, got {ep_bw}"
    );

    // 2. Upstream ancestor inspection: verified x4
    assert_eq!(
        discovery.upstream_ancestors.len(),
        1,
        "must discover 1 upstream ancestor"
    );
    let ancestor = &discovery.upstream_ancestors[0];
    assert_eq!(ancestor.bdf, PciBdf::new(0, 0, 1, 0));
    assert_eq!(ancestor.current_width, Some(4));
    assert_eq!(ancestor.current_speed_gts, Some(2.5));
    assert_eq!(ancestor.max_speed_gts, Some(16.0));

    // 3. Bottleneck selection: MUST select the upstream x4 hop, NEVER endpoint alone!
    assert!(
        discovery.is_bottlenecked_upstream(),
        "is_bottlenecked_upstream must be true"
    );
    assert_eq!(
        discovery.capacity_bottleneck.bdf,
        PciBdf::new(0, 0, 1, 0),
        "bottleneck must be upstream bridge 0000:00:01.0, NEVER endpoint alone"
    );
    assert_eq!(
        discovery.capacity_bottleneck.capacity_width(),
        Some(4),
        "bottleneck width must be x4"
    );

    let bottleneck_bw = discovery.bottleneck_bandwidth_gbps().unwrap();
    // 16 GT/s * 4 lanes * (128/130) / 8 = 7.87692... GB/s
    assert!(
        (bottleneck_bw - 7.87692).abs() < 1e-3,
        "bottleneck payload bandwidth must be ~7.88 GB/s, got {bottleneck_bw}"
    );

    assert!(
        bottleneck_bw < ep_bw,
        "bottleneck payload capacity must be strictly less than endpoint"
    );
    assert_eq!(
        discovery.current_bottleneck.as_ref().map(|hop| (
            hop.bdf,
            hop.current_speed_gts,
            hop.current_width
        )),
        Some((PciBdf::new(0, 0, 1, 0), Some(2.5), Some(4)))
    );

    // 4. Test integrated enumeration with this synthetic sysfs
    let stub_so = common::get_or_compile_stub_with_count(1);
    let lib = Arc::new(HipLibrary::load_from_path(&stub_so).unwrap());
    let discovered = lib.enumerate_devices_with_sys_root(sys_root).unwrap();
    assert_eq!(discovered.len(), 1);
    let pcie_info = &discovered[0].pcie_path;
    assert_eq!(pcie_info.capacity_bottleneck.capacity_width(), Some(4));
    assert_eq!(
        discovered[0].bottleneck_bandwidth_gbps().unwrap(),
        bottleneck_bw
    );
}

#[test]
fn test_pcie_path_multi_hop_ancestors_picks_slowest_intermediate_hop() {
    let temp = TempDirGuard::new("r9v_multi_hop_sysfs");
    let sys_root = temp.path();

    // Topology:
    // Root Port (0000:00:01.0): Gen4 x16 (~31.51 GB/s)
    // Switch Upstream (0000:01:00.0): Gen4 x8 (~15.75 GB/s) <- BOTTLENECK
    // Switch Downstream (0000:02:01.0): Gen4 x16 (~31.51 GB/s)
    // GPU Endpoint (0000:03:00.0): Gen5 x16 (~63.02 GB/s)

    let root_dir = sys_root.join("devices/pci0000:00/0000:00:01.0");
    let sw_up_dir = root_dir.join("0000:01:00.0");
    let sw_down_dir = sw_up_dir.join("0000:02:01.0");
    let endpoint_dir = sw_down_dir.join("0000:03:00.0");
    fs::create_dir_all(&endpoint_dir).unwrap();

    fs::write(root_dir.join("current_link_speed"), "16.0 GT/s PCIe\n").unwrap();
    fs::write(root_dir.join("current_link_width"), "16\n").unwrap();

    // Intermediate bottleneck: x8
    fs::write(sw_up_dir.join("current_link_speed"), "16.0 GT/s PCIe\n").unwrap();
    fs::write(sw_up_dir.join("current_link_width"), "8\n").unwrap();

    fs::write(sw_down_dir.join("current_link_speed"), "16.0 GT/s PCIe\n").unwrap();
    fs::write(sw_down_dir.join("current_link_width"), "16\n").unwrap();

    fs::write(endpoint_dir.join("current_link_speed"), "32.0 GT/s PCIe\n").unwrap();
    fs::write(endpoint_dir.join("current_link_width"), "16\n").unwrap();

    let bus_dir = sys_root.join("bus/pci/devices");
    fs::create_dir_all(&bus_dir).unwrap();
    symlink(&endpoint_dir, bus_dir.join("0000:03:00.0")).unwrap();

    let discovery = PciPathDiscovery::discover(sys_root, PciBdf::new(0, 3, 0, 0)).unwrap();
    assert_eq!(discovery.upstream_ancestors.len(), 3);
    assert_eq!(discovery.endpoint.current_width, Some(16));
    assert_eq!(discovery.endpoint.current_speed_gts, Some(32.0));

    // Bottleneck must be the intermediate switch upstream port (0000:01:00.0, x8)
    assert_eq!(discovery.capacity_bottleneck.bdf, PciBdf::new(0, 1, 0, 0));
    assert_eq!(discovery.capacity_bottleneck.capacity_width(), Some(8));
    assert_eq!(
        discovery.capacity_bottleneck.capacity_speed_gts(),
        Some(16.0)
    );
    let bw = discovery.bottleneck_bandwidth_gbps().unwrap();
    assert!((bw - 15.75385).abs() < 1e-3);
}

#[test]
fn test_pcie_path_rejects_a_partial_hop_instead_of_guessing() {
    let temp = TempDirGuard::new("r9v_partial_pcie_sysfs");
    let sys_root = temp.path();
    let bridge = sys_root.join("devices/pci0000:00/0000:00:01.0");
    let endpoint = bridge.join("0000:03:00.0");
    fs::create_dir_all(&endpoint).unwrap();

    fs::write(endpoint.join("max_link_speed"), "32.0 GT/s\n").unwrap();
    fs::write(endpoint.join("current_link_width"), "16\n").unwrap();
    fs::write(bridge.join("max_link_speed"), "16.0 GT/s\n").unwrap();
    // Deliberately omit both width files for the bridge.

    let by_bus = sys_root.join("bus/pci/devices");
    fs::create_dir_all(&by_bus).unwrap();
    symlink(&endpoint, by_bus.join("0000:03:00.0")).unwrap();

    match PciPathDiscovery::discover(sys_root, PciBdf::new(0, 3, 0, 0)) {
        Err(HipError::SysfsError { path, details, .. }) => {
            assert!(path.ends_with("0000:00:01.0"));
            assert!(details.contains("complete positive speed/width"));
        }
        Err(other) => panic!("expected SysfsError, got: {other}"),
        Ok(_) => panic!("partial PCIe hop was silently omitted"),
    }
}

#[test]
fn test_pci_bdf_and_hip_uuid_types() {
    let bdf = PciBdf::new(0, 3, 0, 0);
    assert_eq!(bdf.to_string(), "0000:03:00.0");
    let parsed_bdf: PciBdf = "0000:03:00.0".parse().unwrap();
    assert_eq!(bdf, parsed_bdf);

    let bytes = [1u8; 16];
    let uuid = HipUuid::new(bytes);
    assert!(!uuid.is_zero());
    assert_eq!(uuid.to_string(), "GPU-01010101-0101-0101-0101-010101010101");
    let parsed_uuid = HipUuid::parse("GPU-01010101-0101-0101-0101-010101010101").unwrap();
    assert_eq!(uuid, parsed_uuid);
}
