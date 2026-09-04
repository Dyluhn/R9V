// SPDX-License-Identifier: Apache-2.0
//! API shape tests verifying public exports, trait implementations, and visibility (Spec 14 §2, §3, CONVENTIONS.md §4.1).

use r9v_hip::{
    default_library, device_count, enumerate_devices, inventory, is_available,
    pcie_payload_bandwidth_gbps, AllocationBudget, BudgetedDeviceBuffer, Device, DeviceBuffer,
    DeviceIdentity, DeviceInventory, DeviceProperties, DiscoveredDevice, Event, EventFlags,
    Function, Graph, GraphExec, HipError, HipLibrary, HipOrdinal, HipUuid, HostBuffer, MemcpyKind,
    Module, PciBdf, PciLinkHop, PciPathDiscovery, Result, Stream, StreamCaptureMode, StreamFlags,
};
use std::fmt::Display;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_display<T: Display>() {}
fn assert_clone<T: Clone>() {}
fn assert_copy<T: Copy>() {}

#[test]
fn test_api_shape_invariants() {
    // Top-level discovery helpers
    let _: fn() -> Result<std::sync::Arc<HipLibrary>> = default_library;
    let _: fn() -> bool = is_available;
    let _: fn() -> Result<u32> = device_count;
    let _: fn() -> Result<DeviceInventory> = inventory;
    let _: fn() -> Result<Vec<DiscoveredDevice>> = enumerate_devices;
    let _: fn(f64, u32) -> f64 = pcie_payload_bandwidth_gbps;
    let _: fn(&HipLibrary, HipOrdinal) -> Result<PciBdf> = HipLibrary::get_device_pci_bdf;

    // Device enum traits
    assert_copy::<Device>();
    assert_clone::<Device>();
    assert_send::<Device>();
    assert_sync::<Device>();
    assert_display::<Device>();

    // Ephemeral handle and stable identity types
    assert_copy::<HipOrdinal>();
    assert_clone::<HipOrdinal>();
    assert_send::<HipOrdinal>();
    assert_sync::<HipOrdinal>();
    assert_display::<HipOrdinal>();

    assert_copy::<PciBdf>();
    assert_clone::<PciBdf>();
    assert_send::<PciBdf>();
    assert_sync::<PciBdf>();
    assert_display::<PciBdf>();

    assert_copy::<HipUuid>();
    assert_clone::<HipUuid>();
    assert_send::<HipUuid>();
    assert_sync::<HipUuid>();
    assert_display::<HipUuid>();

    assert_clone::<DeviceIdentity>();
    assert_send::<DeviceIdentity>();
    assert_sync::<DeviceIdentity>();
    assert_display::<DeviceIdentity>();

    // Enumeration and inventory records
    assert_clone::<DiscoveredDevice>();
    assert_send::<DiscoveredDevice>();
    assert_sync::<DiscoveredDevice>();

    assert_clone::<DeviceInventory>();
    assert_send::<DeviceInventory>();
    assert_sync::<DeviceInventory>();

    // PCIe path discovery types
    assert_clone::<PciLinkHop>();
    assert_send::<PciLinkHop>();
    assert_sync::<PciLinkHop>();

    assert_clone::<PciPathDiscovery>();
    assert_send::<PciPathDiscovery>();
    assert_sync::<PciPathDiscovery>();

    // Typed parameter enums
    assert_copy::<MemcpyKind>();
    assert_clone::<MemcpyKind>();
    assert_send::<MemcpyKind>();
    assert_sync::<MemcpyKind>();

    assert_copy::<StreamCaptureMode>();
    assert_clone::<StreamCaptureMode>();
    assert_send::<StreamCaptureMode>();
    assert_sync::<StreamCaptureMode>();

    assert_copy::<StreamFlags>();
    assert_clone::<StreamFlags>();
    assert_send::<StreamFlags>();
    assert_sync::<StreamFlags>();

    assert_copy::<EventFlags>();
    assert_clone::<EventFlags>();
    assert_send::<EventFlags>();
    assert_sync::<EventFlags>();

    // DeviceProperties
    assert_clone::<DeviceProperties>();
    assert_send::<DeviceProperties>();
    assert_sync::<DeviceProperties>();

    // RAII handles
    assert_send::<Stream>();
    assert_sync::<Stream>();

    assert_send::<Event>();
    assert_sync::<Event>();

    assert_clone::<Module>();
    assert_send::<Module>();
    assert_sync::<Module>();

    assert_clone::<Function>();
    assert_send::<Function>();
    assert_sync::<Function>();

    assert_send::<Graph>();
    assert_sync::<Graph>();

    assert_send::<GraphExec>();
    assert_sync::<GraphExec>();

    assert_send::<DeviceBuffer>();
    assert_sync::<DeviceBuffer>();

    assert_clone::<AllocationBudget>();
    assert_send::<AllocationBudget>();
    assert_sync::<AllocationBudget>();

    assert_send::<BudgetedDeviceBuffer>();
    assert_sync::<BudgetedDeviceBuffer>();

    assert_send::<HostBuffer>();
    assert_sync::<HostBuffer>();

    assert_send::<HipLibrary>();
    assert_sync::<HipLibrary>();

    // HipError: Must implement Clone
    assert_clone::<HipError>();
    assert_send::<HipError>();
    assert_sync::<HipError>();
    assert_display::<HipError>();
}
