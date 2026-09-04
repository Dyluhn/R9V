// SPDX-License-Identifier: Apache-2.0
//! Kernel launch list record and deterministic replay, stub device execution, and profiling hooks (Spec 4 §7, §12).

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::error::{RegistryError, Result};
use crate::types::{LaunchGeometry, VariantHash};

/// A single recorded kernel launch operation (Spec 4 §7, §12).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchEntry {
    /// Identifier of the variant launched (Spec 4 §3).
    pub variant_hash: VariantHash,
    /// Exported entry point symbol (Spec 4 §9.1).
    pub entry_symbol: String,
    /// Grid and block launch geometry (Spec 4 §7).
    pub geometry: LaunchGeometry,
    /// Scratch workspace memory allocated for this launch in bytes (Spec 4 §7).
    pub workspace_bytes: u64,
    /// Static bytes transferred by this launch for memory bandwidth accounting (Spec 4 §12).
    pub static_bytes: u64,
    /// Static operations performed by this launch for rate accounting (Spec 4 §12).
    pub static_flops: u64,
    /// Serialized kernel ABI argument structure bytes (Spec 4 §7).
    pub args_blob: Vec<u8>,
}

impl LaunchEntry {
    /// Constructs a new launch entry.
    pub fn new(
        variant_hash: VariantHash,
        entry_symbol: impl Into<String>,
        geometry: LaunchGeometry,
        workspace_bytes: u64,
        static_bytes: u64,
        static_flops: u64,
        args_blob: Vec<u8>,
    ) -> Self {
        Self {
            variant_hash,
            entry_symbol: entry_symbol.into(),
            geometry,
            workspace_bytes,
            static_bytes,
            static_flops,
            args_blob,
        }
    }
}

/// Abstract device kernel launch interface for execution and replay (Spec 4 §7).
pub trait DeviceExecutor: Send + Sync {
    /// Executes or enqueues a single kernel launch (Spec 4 §7).
    fn launch_kernel(&self, entry: &LaunchEntry) -> Result<()>;
}

/// Telemetry record emitted during launch dispatch when profiling is enabled (Spec 4 §12).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRecord {
    /// Variant hash of the launched kernel (Spec 4 §12).
    pub variant_hash: VariantHash,
    /// Entry point symbol (Spec 4 §12).
    pub entry_symbol: String,
    /// Workspace bytes used (Spec 4 §12).
    pub workspace_used: u64,
    /// Static bytes transferred (Spec 4 §12).
    pub static_bytes: u64,
    /// Static floating-point or integer operations (Spec 4 §12).
    pub static_flops: u64,
    /// Launch geometry (Spec 4 §7).
    pub geometry: LaunchGeometry,
}

/// Sink trait receiving kernel launch profiling events (Spec 4 §12, Spec 11).
///
/// Supports recording start and end events around device kernel execution
/// so `r9v-obs` in card A4.1 can record `hipEvent` pairs (Spec 4 §12).
pub trait ProfileSink: Send + Sync {
    /// Records start of kernel launch before execution (e.g. `hipEventRecord` start) (Spec 4 §12).
    fn record_start(&self, entry: &LaunchEntry) -> Result<()>;

    /// Records completion of kernel launch after execution (e.g. `hipEventRecord` end) (Spec 4 §12).
    fn record_end(&self, entry: &LaunchEntry) -> Result<()>;
}

// DECISION(A3.1): profiling branch in dispatch_launch checks Option<&dyn ProfileSink>; when None, execution takes a single predictable forward branch directly to launch_kernel with zero timing or event allocation overhead. When Some, record_start is called before launch. If launch fails, record_end is still invoked to balance the stream event pair, but the original launch error takes precedence and is returned. Spec 4 §12.
/// Dispatches a kernel launch with an optional profiling sink hook (Spec 4 §12).
///
/// When `profiler` is `None`, the overhead is exactly one predictable conditional branch (Spec 4 §12).
/// When `profiler` is `Some`:
/// - `record_start` is invoked and its failure propagates immediately without calling `launch_kernel`.
/// - If `launch_kernel` fails, `record_end` is still invoked to keep event pairs balanced on stream, and the launch error is returned.
/// - If `launch_kernel` succeeds and `record_end` fails, the `record_end` error is returned.
#[inline]
pub fn dispatch_launch<E: DeviceExecutor + ?Sized>(
    executor: &E,
    entry: &LaunchEntry,
    profiler: Option<&dyn ProfileSink>,
) -> Result<()> {
    if let Some(sink) = profiler {
        sink.record_start(entry)?;
        let launch_res = executor.launch_kernel(entry);
        let end_res = sink.record_end(entry);
        match (launch_res, end_res) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), _) => Err(e),
            (Ok(()), Err(e)) => Err(e),
        }
    } else {
        executor.launch_kernel(entry)
    }
}

/// In-memory stub device for deterministic launch-list replay testing (Spec 4 §7).
#[derive(Debug, Clone, Default)]
pub struct StubDevice {
    records: Arc<Mutex<Vec<LaunchRecord>>>,
}

impl StubDevice {
    /// Constructs a new stub device with an empty launch record log.
    pub fn new() -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a copy of all launches recorded by this stub device.
    pub fn recorded_launches(&self) -> Result<Vec<LaunchRecord>> {
        let records = self
            .records
            .lock()
            .map_err(|_| RegistryError::LockPoisoned {
                resource: "stub_device_records".to_owned(),
            })?;
        Ok(records.clone())
    }

    /// Clears all recorded launches.
    pub fn clear(&self) -> Result<()> {
        let mut records = self
            .records
            .lock()
            .map_err(|_| RegistryError::LockPoisoned {
                resource: "stub_device_records".to_owned(),
            })?;
        records.clear();
        Ok(())
    }
}

impl DeviceExecutor for StubDevice {
    fn launch_kernel(&self, entry: &LaunchEntry) -> Result<()> {
        let mut list = self
            .records
            .lock()
            .map_err(|_| RegistryError::LockPoisoned {
                resource: "stub_device_records".to_owned(),
            })?;
        list.push(LaunchRecord {
            variant_hash: entry.variant_hash,
            entry_symbol: entry.entry_symbol.clone(),
            workspace_used: entry.workspace_bytes,
            static_bytes: entry.static_bytes,
            static_flops: entry.static_flops,
            geometry: entry.geometry,
        });
        Ok(())
    }
}

/// Ordered sequence of kernel launches captured during step execution or graph capture (Spec 4 §7).
///
/// Supports recording launches and replaying them deterministically against any [`DeviceExecutor`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchList {
    entries: Vec<LaunchEntry>,
}

impl LaunchList {
    /// Creates an empty launch list.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Appends a launch operation to the end of the launch list (Spec 4 §7).
    pub fn record(&mut self, entry: LaunchEntry) {
        self.entries.push(entry);
    }

    /// Returns the total number of recorded launches.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no launches have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a slice of the recorded launch entries in replay order.
    pub fn entries(&self) -> &[LaunchEntry] {
        &self.entries
    }

    /// Clears all recorded launches.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Replays all recorded launches in exact sequential order through [`dispatch_launch`] (Spec 4 §7, §12).
    ///
    /// Replay order is strictly ascending index order `0..len()`.
    pub fn replay<E: DeviceExecutor + ?Sized>(
        &self,
        executor: &E,
        profiler: Option<&dyn ProfileSink>,
    ) -> Result<()> {
        for entry in &self.entries {
            dispatch_launch(executor, entry, profiler)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_stub_device_lock_poisoning() {
        let stub = Arc::new(StubDevice::new());
        let stub_clone = Arc::clone(&stub);

        let _ = thread::spawn(move || {
            let _guard = stub_clone.records.lock().unwrap();
            panic!("deliberate panic to poison stub_device lock");
        })
        .join();

        // recorded_launches returns typed LockPoisoned error
        let err = stub.recorded_launches().unwrap_err();
        match err {
            RegistryError::LockPoisoned { resource } => {
                assert_eq!(resource, "stub_device_records");
            }
            other => panic!("expected LockPoisoned, got {other:?}"),
        }

        // clear returns typed LockPoisoned error
        let clear_err = stub.clear().unwrap_err();
        match clear_err {
            RegistryError::LockPoisoned { resource } => {
                assert_eq!(resource, "stub_device_records");
            }
            other => panic!("expected LockPoisoned, got {other:?}"),
        }

        // launch_kernel returns typed LockPoisoned error
        let entry = LaunchEntry::new(
            VariantHash::new(1),
            "k",
            LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
            0,
            0,
            0,
            vec![],
        );
        let launch_err = stub.launch_kernel(&entry).unwrap_err();
        match launch_err {
            RegistryError::LockPoisoned { resource } => {
                assert_eq!(resource, "stub_device_records");
            }
            other => panic!("expected LockPoisoned, got {other:?}"),
        }
    }
}
