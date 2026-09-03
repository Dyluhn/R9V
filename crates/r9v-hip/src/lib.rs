// SPDX-License-Identifier: Apache-2.0
//! Thin runtime-only AMD HIP binding for R9V (Spec 14 §2, §3).
//!
//! Provides dynamic lazy loading of `libamdhip64` with zero startup/build linkage.
//! Symbols are cached after initial resolution to eliminate `dlsym` overhead on hot paths.

mod raw;
mod symbol;

pub mod device;
pub mod error;
pub mod handles;
pub mod library;

use std::sync::Arc;

pub use device::{Device, DeviceProperties};
pub use error::{HipError, Result};
pub use handles::{DeviceBuffer, Event, Function, Graph, GraphExec, HostBuffer, Module, Stream};
pub use library::HipLibrary;

/// Memory copy direction for HIP transfer operations (Spec 14 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemcpyKind {
    /// Host to device transfer.
    HostToDevice,
    /// Device to host transfer.
    DeviceToHost,
    /// Device to device transfer on same device.
    DeviceToDevice,
    /// Direction inferred by driver from pointer attributes.
    Default,
}

impl MemcpyKind {
    /// Returns the raw integer value expected by `hipMemcpy` (Spec 14 §3).
    pub const fn as_raw(self) -> i32 {
        match self {
            Self::HostToDevice => 1,
            Self::DeviceToHost => 2,
            Self::DeviceToDevice => 3,
            Self::Default => 4,
        }
    }
}

/// Mode for stream graph capture (Spec 1 §3.1, Spec 6 §2, Spec 14 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamCaptureMode {
    /// Global mode: invalidates capturing on all non-participating streams.
    Global,
    /// Thread-local mode: capture only applies to calling thread.
    ThreadLocal,
    /// Relaxed mode: permits stream synchronization during capture if safe.
    Relaxed,
}

impl StreamCaptureMode {
    /// Returns the raw integer value expected by `hipStreamBeginCapture` (Spec 14 §3).
    pub const fn as_raw(self) -> i32 {
        match self {
            Self::Global => 0,
            Self::ThreadLocal => 1,
            Self::Relaxed => 2,
        }
    }
}

/// Flags controlling stream creation behavior (Spec 14 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamFlags {
    /// Default stream creation behavior.
    Default,
    /// Non-blocking stream: does not synchronize with the null stream.
    NonBlocking,
}

impl StreamFlags {
    /// Returns the raw integer value for stream creation (Spec 14 §3).
    pub const fn as_raw(self) -> u32 {
        match self {
            Self::Default => 0,
            Self::NonBlocking => 1,
        }
    }
}

/// Flags controlling event creation behavior (Spec 14 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventFlags {
    /// Default event creation behavior with timing enabled.
    Default,
    /// Timing disabled: reduces recording overhead when only used for synchronization.
    DisableTiming,
}

impl EventFlags {
    /// Returns the raw integer value for event creation (Spec 14 §3).
    pub const fn as_raw(self) -> u32 {
        match self {
            Self::Default => 0,
            Self::DisableTiming => 2,
        }
    }
}

/// Obtains a reference to the global default HIP runtime library instance (Spec 14 §3).
pub fn default_library() -> Result<Arc<HipLibrary>> {
    HipLibrary::default_or_load()
}

/// Returns `true` if the HIP runtime can be loaded AND at least one HIP device is available (Spec 14 §3).
pub fn is_available() -> bool {
    default_library()
        .and_then(|lib| lib.device_count())
        .map(|count| count > 0)
        .unwrap_or(false)
}

/// Queries the total number of available HIP GPU devices (Spec 14 §3).
pub fn device_count() -> Result<u32> {
    default_library()?.device_count()
}
