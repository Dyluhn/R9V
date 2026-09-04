// SPDX-License-Identifier: Apache-2.0
//! Typed opaque RAII handles for HIP runtime resources (Spec 14 §2, §3).

use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use crate::error::{HipError, Result};
use crate::library::HipLibrary;
use crate::raw::*;
use crate::{EventFlags, MemcpyKind, StreamCaptureMode, StreamFlags};

/// Safe RAII wrapper for a HIP execution stream (`hipStream_t`, Spec 14 §3).
pub struct Stream {
    raw: HipStreamT,
    lib: Arc<HipLibrary>,
}

// SAFETY: HIP streams are opaque thread-safe runtime references (hipStream_t) that can be sent
// across threads in multi-threaded asynchronous launch and synchronization architectures.
unsafe impl Send for Stream {}
// SAFETY: Multiple host threads may synchronize or enqueue operations concurrently to distinct or shared streams.
unsafe impl Sync for Stream {}

impl Stream {
    /// Creates a new execution stream with default flags (Spec 14 §3).
    pub fn new(lib: &Arc<HipLibrary>) -> Result<Self> {
        let raw = lib.stream_create()?;
        Ok(Self {
            raw,
            lib: Arc::clone(lib),
        })
    }

    /// Creates a new execution stream with custom flags (Spec 14 §3).
    pub fn with_flags(lib: &Arc<HipLibrary>, flags: StreamFlags) -> Result<Self> {
        let raw = lib.stream_create_with_flags(flags)?;
        Ok(Self {
            raw,
            lib: Arc::clone(lib),
        })
    }

    /// Returns the underlying raw C handle (Spec 14 §3).
    pub fn as_raw(&self) -> HipStreamT {
        self.raw
    }

    /// Synchronizes this stream, blocking the calling host thread until complete (Spec 14 §3).
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.lib.stream_synchronize(self.raw) }
    }

    /// Queries completion status without blocking (Spec 14 §3).
    pub fn is_complete(&self) -> Result<bool> {
        unsafe { self.lib.stream_query(self.raw) }
    }

    /// Instructs this stream to wait on an event before executing subsequent commands (Spec 14 §3).
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        unsafe { self.lib.stream_wait_event(self.raw, event.as_raw(), 0) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { self.lib.stream_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Safe RAII wrapper for a HIP timing/synchronization event (`hipEvent_t`, Spec 14 §3).
pub struct Event {
    raw: HipEventT,
    lib: Arc<HipLibrary>,
}

// SAFETY: HIP event handles (hipEvent_t) are thread-safe runtime references that can be transferred across threads.
unsafe impl Send for Event {}
// SAFETY: Concurrent threads may safely query or synchronize on recorded events.
unsafe impl Sync for Event {}

impl Event {
    /// Creates an event with default flags (Spec 14 §3).
    pub fn new(lib: &Arc<HipLibrary>) -> Result<Self> {
        let raw = lib.event_create()?;
        Ok(Self {
            raw,
            lib: Arc::clone(lib),
        })
    }

    /// Creates an event with custom flags (Spec 14 §3).
    pub fn with_flags(lib: &Arc<HipLibrary>, flags: EventFlags) -> Result<Self> {
        let raw = lib.event_create_with_flags(flags)?;
        Ok(Self {
            raw,
            lib: Arc::clone(lib),
        })
    }

    /// Returns the underlying raw C handle (Spec 14 §3).
    pub fn as_raw(&self) -> HipEventT {
        self.raw
    }

    /// Records the event in the specified stream (Spec 14 §3).
    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe { self.lib.event_record(self.raw, stream.as_raw()) }
    }

    /// Blocks host thread until this event has completed (Spec 14 §3).
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.lib.event_synchronize(self.raw) }
    }

    /// Queries event completion status without blocking (Spec 14 §3).
    pub fn is_complete(&self) -> Result<bool> {
        unsafe { self.lib.event_query(self.raw) }
    }

    /// Computes elapsed time in milliseconds between `start` and `self` (Spec 11 §7, Spec 14 §3).
    pub fn elapsed_since(&self, start: &Event) -> Result<f32> {
        unsafe { self.lib.event_elapsed_time(start.as_raw(), self.raw) }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { self.lib.event_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

pub(crate) struct ModuleInner {
    pub(crate) raw: HipModuleT,
    pub(crate) lib: Arc<HipLibrary>,
}

// SAFETY: HIP module handles (hipModule_t) are thread-safe references within the owning device context.
unsafe impl Send for ModuleInner {}
// SAFETY: Concurrent threads can safely look up functions or launch kernels from a loaded module.
unsafe impl Sync for ModuleInner {}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { self.lib.module_unload(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Safe RAII wrapper for a loaded compiled kernel module (`hipModule_t`, Spec 4 §10, Spec 14 §3).
#[derive(Clone)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

impl Module {
    /// Loads a compiled `.co` module from disk (Spec 4 §10, Spec 14 §3).
    pub fn load_file(lib: &Arc<HipLibrary>, path: &Path) -> Result<Self> {
        let raw = lib.module_load(path)?;
        Ok(Self {
            inner: Arc::new(ModuleInner {
                raw,
                lib: Arc::clone(lib),
            }),
        })
    }

    /// Loads a compiled `.co` module from memory bytes (Spec 4 §10, Spec 14 §3).
    pub fn load_data(lib: &Arc<HipLibrary>, image: &[u8]) -> Result<Self> {
        let raw = lib.module_load_data(image)?;
        Ok(Self {
            inner: Arc::new(ModuleInner {
                raw,
                lib: Arc::clone(lib),
            }),
        })
    }

    /// Resolves an exported kernel function by name (Spec 4 §10, Spec 14 §3).
    ///
    /// The returned [`Function`] keeps the parent [`Module`] alive even if the original
    /// [`Module`] handle is dropped.
    pub fn get_function(&self, name: &str) -> Result<Function> {
        let raw = unsafe { self.inner.lib.module_get_function(self.inner.raw, name)? };
        Ok(Function {
            raw,
            module: Arc::clone(&self.inner),
        })
    }

    /// Returns the underlying raw C handle (Spec 14 §3).
    pub fn as_raw(&self) -> HipModuleT {
        self.inner.raw
    }
}

/// Handle to an executable kernel function inside a loaded [`Module`] (Spec 4 §10, Spec 14 §3).
///
/// Holds an `Arc` reference to the module allocation, ensuring the module remains loaded
/// and valid for as long as the [`Function`] exists.
#[derive(Clone)]
pub struct Function {
    raw: HipFunctionT,
    module: Arc<ModuleInner>,
}

// SAFETY: HIP function handles (hipFunction_t) can be transferred across threads to enqueue kernel launches.
unsafe impl Send for Function {}
// SAFETY: Function handles are immutable and kernel launches take shared &Function references across threads.
unsafe impl Sync for Function {}

impl Function {
    /// Launches the kernel function on `stream` with given grid/block dimensions and parameters (Spec 4 §10, Spec 14 §3).
    ///
    /// # Safety
    /// The caller must ensure that kernel arguments match the declared parameter types
    /// and device pointer validity of the target kernel ABI.
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: &Stream,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        self.module.lib.module_launch_kernel(
            self.raw,
            grid,
            block,
            shared_mem,
            stream.as_raw(),
            args,
        )
    }

    /// Returns the underlying raw C handle (Spec 14 §3).
    pub fn as_raw(&self) -> HipFunctionT {
        self.raw
    }
}

/// Safe RAII wrapper for a captured graph definition (`hipGraph_t`, Spec 1 §3.1, Spec 6 §2, Spec 14 §3).
pub struct Graph {
    raw: HipGraphT,
    lib: Arc<HipLibrary>,
}

// SAFETY: Captured HIP graph handles (hipGraph_t) can be transferred across threads before instantiation.
unsafe impl Send for Graph {}
// SAFETY: HIP graph definitions are read-only once captured and safe to inspect concurrently.
unsafe impl Sync for Graph {}

impl Graph {
    /// Begins capturing execution operations enqueued to `stream` (Spec 1 §3.1, Spec 6 §2, Spec 14 §3).
    pub fn begin_capture(stream: &Stream, mode: StreamCaptureMode) -> Result<()> {
        unsafe { stream.lib.stream_begin_capture(stream.as_raw(), mode) }
    }

    /// Ends stream capture and returns the newly captured [`Graph`] (Spec 1 §3.1, Spec 6 §2, Spec 14 §3).
    pub fn end_capture(stream: &Stream) -> Result<Self> {
        let raw = unsafe { stream.lib.stream_end_capture(stream.as_raw())? };
        Ok(Self {
            raw,
            lib: Arc::clone(&stream.lib),
        })
    }

    /// Instantiates an executable graph from this definition (Spec 6 §2, Spec 14 §3).
    pub fn instantiate(&self) -> Result<GraphExec> {
        let raw = unsafe { self.lib.graph_instantiate(self.raw)? };
        Ok(GraphExec {
            raw,
            lib: Arc::clone(&self.lib),
        })
    }

    /// Returns the underlying raw C handle (Spec 14 §3).
    pub fn as_raw(&self) -> HipGraphT {
        self.raw
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { self.lib.graph_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Safe RAII wrapper for an instantiated executable graph (`hipGraphExec_t`, Spec 6 §2, Spec 14 §3).
pub struct GraphExec {
    raw: HipGraphExecT,
    lib: Arc<HipLibrary>,
}

// SAFETY: Instantiated graph execution handles (hipGraphExec_t) can be transferred across threads.
unsafe impl Send for GraphExec {}
// SAFETY: Graph launch operations dispatch to the stream and are safe to launch concurrently on distinct streams.
unsafe impl Sync for GraphExec {}

impl GraphExec {
    /// Launches the executable graph on `stream` (Spec 6 §2, Spec 14 §3).
    ///
    /// # Safety
    /// The caller must ensure that all memory allocations and resources captured within
    /// this graph definition remain allocated and valid throughout execution on `stream`.
    pub unsafe fn launch(&self, stream: &Stream) -> Result<()> {
        self.lib.graph_launch(self.raw, stream.as_raw())
    }

    /// Returns the underlying raw C handle (Spec 14 §3).
    pub fn as_raw(&self) -> HipGraphExecT {
        self.raw
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { self.lib.graph_exec_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Safe RAII wrapper for allocated linear device memory (Spec 14 §3).
pub struct DeviceBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
    lib: Arc<HipLibrary>,
}

// SAFETY: Linear device pointers (*mut c_void) represent distinct allocations in GPU VRAM
// and can be transferred safely between host threads.
unsafe impl Send for DeviceBuffer {}
// SAFETY: DeviceBuffer methods requiring mutable device writes take &mut self, while read copies take &self.
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocates `size_bytes` of linear device memory (Spec 14 §3).
    ///
    /// This path performs no budget accounting. Constrained execution must
    /// exclusively use the budgeted allocator path
    /// ([`BudgetedDeviceBuffer::allocate`](crate::budget::BudgetedDeviceBuffer::allocate));
    /// direct [`DeviceBuffer`] allocation is reserved for physical/unconstrained
    /// callers that intentionally bypass the budget.
    pub fn allocate(lib: &Arc<HipLibrary>, size_bytes: usize) -> Result<Self> {
        let ptr = lib.malloc(size_bytes)?;
        Ok(Self {
            ptr,
            size_bytes,
            lib: Arc::clone(lib),
        })
    }

    /// Returns the allocation size in bytes (Spec 14 §3).
    pub fn size(&self) -> usize {
        self.size_bytes
    }

    /// Returns the raw device memory pointer (Spec 14 §3).
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    /// Returns the mutable raw device memory pointer (Spec 14 §3).
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    /// Copies bytes synchronously from host slice to device buffer (Spec 14 §3).
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `src.len() > self.size()`.
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.size_bytes {
            return Err(HipError::BufferTooSmall {
                operation: "copy_from_host",
                required: src.len(),
                available: self.size_bytes,
            });
        }
        unsafe {
            self.lib.memcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                MemcpyKind::HostToDevice,
            )
        }
    }

    /// Copies bytes synchronously from device buffer to host slice (Spec 14 §3).
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `dst.len() < self.size()`.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() < self.size_bytes {
            return Err(HipError::BufferTooSmall {
                operation: "copy_to_host",
                required: self.size_bytes,
                available: dst.len(),
            });
        }
        unsafe {
            self.lib.memcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                self.size_bytes,
                MemcpyKind::DeviceToHost,
            )
        }
    }

    /// Asynchronously copies bytes from host slice to device buffer in `stream` (Spec 14 §3).
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `src.len() > self.size()`.
    ///
    /// # Safety
    /// The caller must ensure that `src` remains valid, allocated, and untouched
    /// until all operations queued in `stream` at the time of call have completed execution
    /// (e.g. by calling [`Stream::synchronize`]).
    pub unsafe fn copy_from_host_async(&mut self, src: &[u8], stream: &Stream) -> Result<()> {
        if src.len() > self.size_bytes {
            return Err(HipError::BufferTooSmall {
                operation: "copy_from_host_async",
                required: src.len(),
                available: self.size_bytes,
            });
        }
        self.lib.memcpy_async(
            self.ptr,
            src.as_ptr() as *const c_void,
            src.len(),
            MemcpyKind::HostToDevice,
            stream.as_raw(),
        )
    }

    /// Asynchronously copies bytes from device buffer to host slice in `stream` (Spec 14 §3).
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `dst.len() < self.size()`.
    ///
    /// # Safety
    /// The caller must ensure that `dst` remains valid, allocated, and untouched
    /// until all operations queued in `stream` at the time of call have completed execution
    /// (e.g. by calling [`Stream::synchronize`]).
    pub unsafe fn copy_to_host_async(&self, dst: &mut [u8], stream: &Stream) -> Result<()> {
        if dst.len() < self.size_bytes {
            return Err(HipError::BufferTooSmall {
                operation: "copy_to_host_async",
                required: self.size_bytes,
                available: dst.len(),
            });
        }
        self.lib.memcpy_async(
            dst.as_mut_ptr() as *mut c_void,
            self.ptr,
            self.size_bytes,
            MemcpyKind::DeviceToHost,
            stream.as_raw(),
        )
    }

    /// Copies bytes asynchronously between two device buffers on the same device in `stream` (Spec 14 §3).
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `self.size() > dst.size()`.
    ///
    /// # Safety
    /// The caller must ensure that `self`, `dst`, and `stream` remain valid until all operations
    /// queued in `stream` complete.
    pub unsafe fn copy_to_device_async(
        &self,
        dst: &mut DeviceBuffer,
        stream: &Stream,
    ) -> Result<()> {
        if self.size_bytes > dst.size_bytes {
            return Err(HipError::BufferTooSmall {
                operation: "copy_to_device_async",
                required: self.size_bytes,
                available: dst.size_bytes,
            });
        }
        self.lib.memcpy_async(
            dst.as_mut_ptr(),
            self.ptr,
            self.size_bytes,
            MemcpyKind::DeviceToDevice,
            stream.as_raw(),
        )
    }

    /// Asynchronously copies memory from this device buffer to `dst` on `dst_device` via peer transfer (Spec 5 §7, Spec 14 §3).
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `self.size() > dst.size()`.
    ///
    /// # Safety
    /// The caller must ensure that `self`, `dst`, and `stream` remain valid until all operations
    /// queued in `stream` complete, and that peer access is enabled between `src_device` and `dst_device`.
    pub unsafe fn copy_to_peer_async(
        &self,
        src_device: i32,
        dst: &mut DeviceBuffer,
        dst_device: i32,
        stream: &Stream,
    ) -> Result<()> {
        if self.size_bytes > dst.size_bytes {
            return Err(HipError::BufferTooSmall {
                operation: "copy_to_peer_async",
                required: self.size_bytes,
                available: dst.size_bytes,
            });
        }
        self.lib.memcpy_peer_async(
            dst.as_mut_ptr(),
            dst_device,
            self.ptr,
            src_device,
            self.size_bytes,
            stream.as_raw(),
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { self.lib.free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Safe RAII wrapper for page-locked (pinned) host memory (`hipHostMalloc`, Spec 14 §3).
pub struct HostBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
    lib: Arc<HipLibrary>,
}

// SAFETY: Pinned host memory buffers (*mut c_void) are unique system allocations safe to send across threads.
unsafe impl Send for HostBuffer {}
// SAFETY: Pinned host slices enforce Rust's borrow checker invariants (& vs &mut) over the mapped slice.
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    /// Allocates `size_bytes` of page-locked host memory (Spec 14 §3).
    pub fn allocate(lib: &Arc<HipLibrary>, size_bytes: usize, flags: u32) -> Result<Self> {
        let ptr = lib.host_malloc(size_bytes, flags)?;
        Ok(Self {
            ptr,
            size_bytes,
            lib: Arc::clone(lib),
        })
    }

    /// Returns allocation size in bytes (Spec 14 §3).
    pub fn size(&self) -> usize {
        self.size_bytes
    }

    /// Returns a slice over the pinned host memory (Spec 14 §3).
    pub fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() || self.size_bytes == 0 {
            &[]
        } else {
            // SAFETY: Allocation was verified non-null and size_bytes was returned from host_malloc.
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.size_bytes) }
        }
    }

    /// Returns a mutable slice over the pinned host memory (Spec 14 §3).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.ptr.is_null() || self.size_bytes == 0 {
            &mut []
        } else {
            // SAFETY: Allocation was verified non-null and mutable borrow ensures exclusivity.
            unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.size_bytes) }
        }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { self.lib.host_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
