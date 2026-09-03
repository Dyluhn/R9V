// SPDX-License-Identifier: Apache-2.0
//! Dynamic library loader and raw HIP runtime interface (Spec 14 §2, §3).

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::device::{parse_fixed_c_string, DeviceProperties};
use crate::error::{HipError, Result};
use crate::raw::*;
use crate::symbol::{DynamicLibrary, SymbolCache};
use crate::{EventFlags, MemcpyKind, StreamCaptureMode, StreamFlags};

static GLOBAL_LIBRARY: OnceLock<Result<Arc<HipLibrary>>> = OnceLock::new();

/// Thread-safe runtime binding to `libamdhip64` with cached lazy symbol lookup (Spec 14 §3).
pub struct HipLibrary {
    lib: DynamicLibrary,
    symbols: SymbolCache,
}

impl HipLibrary {
    /// Loads the default system HIP runtime library, or returns the cached instance (Spec 14 §3).
    ///
    /// Resolves search paths in order:
    /// 1. `R9V_HIP_PATH` environment variable
    /// 2. `ROCM_PATH/lib/libamdhip64.so.7` or `ROCM_PATH/lib/libamdhip64.so`
    /// 3. `/opt/rocm/lib/libamdhip64.so.7` or `/opt/rocm/lib/libamdhip64.so`
    /// 4. System library search paths (`libamdhip64.so.7`, `libamdhip64.so`)
    pub fn default_or_load() -> Result<Arc<Self>> {
        GLOBAL_LIBRARY
            .get_or_init(Self::load_system)
            .as_ref()
            .map(Arc::clone)
            .map_err(|e| e.clone())
    }

    /// Loads the system HIP library from standard candidate locations (Spec 14 §3).
    pub fn load_system() -> Result<Arc<Self>> {
        let mut searched = Vec::new();

        // 1. Explicit R9V_HIP_PATH environment variable
        if let Ok(path_str) = std::env::var("R9V_HIP_PATH") {
            let p = PathBuf::from(&path_str);
            if p.is_file() {
                match Self::load_from_path(&p) {
                    Ok(lib) => return Ok(Arc::new(lib)),
                    Err(e) => searched.push(format!("{}: {e}", p.display())),
                }
            } else if p.is_dir() {
                let cand1 = p.join("lib/libamdhip64.so.7");
                if cand1.is_file() {
                    match Self::load_from_path(&cand1) {
                        Ok(lib) => return Ok(Arc::new(lib)),
                        Err(e) => searched.push(format!("{}: {e}", cand1.display())),
                    }
                } else {
                    searched.push(format!("{}: file not found", cand1.display()));
                }

                let cand2 = p.join("libamdhip64.so.7");
                if cand2.is_file() {
                    match Self::load_from_path(&cand2) {
                        Ok(lib) => return Ok(Arc::new(lib)),
                        Err(e) => searched.push(format!("{}: {e}", cand2.display())),
                    }
                } else {
                    searched.push(format!("{}: file not found", cand2.display()));
                }
            } else {
                searched.push(format!("{path_str}: path does not exist"));
            }
        }

        // 2. Explicit ROCM_PATH environment variable
        if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
            let p = PathBuf::from(rocm_path);
            let candidates = [p.join("lib/libamdhip64.so.7"), p.join("lib/libamdhip64.so")];
            for cand in candidates {
                if cand.is_file() {
                    match Self::load_from_path(&cand) {
                        Ok(lib) => return Ok(Arc::new(lib)),
                        Err(e) => searched.push(format!("{}: {e}", cand.display())),
                    }
                } else {
                    searched.push(format!("{}: file not found", cand.display()));
                }
            }
        }

        // 3. Default /opt/rocm paths
        let opt_rocm_candidates = [
            "/opt/rocm/lib/libamdhip64.so.7",
            "/opt/rocm/lib/libamdhip64.so",
        ];
        for cand in opt_rocm_candidates {
            let p = Path::new(cand);
            if p.is_file() {
                match Self::load_from_path(p) {
                    Ok(lib) => return Ok(Arc::new(lib)),
                    Err(e) => searched.push(format!("{cand}: {e}")),
                }
            } else {
                searched.push(format!("{cand}: file not found"));
            }
        }

        // 4. System library sonames for pinned ROCm 7.x matrix
        let system_names = ["libamdhip64.so.7", "libamdhip64.so"];
        for name in system_names {
            match Self::load_from_path(Path::new(name)) {
                Ok(lib) => return Ok(Arc::new(lib)),
                Err(e) => searched.push(format!("{name}: {e}")),
            }
        }

        Err(HipError::LibraryNotFound { searched })
    }

    /// Loads a HIP shared library from an explicit path (Spec 14 §3).
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let lib = DynamicLibrary::open(path)?;
        Ok(Self {
            lib,
            symbols: SymbolCache::default(),
        })
    }

    /// Returns the file system path of the loaded dynamic library (Spec 14 §3).
    pub fn library_path(&self) -> &Path {
        self.lib.path()
    }

    fn check(&self, op: &'static str, code: HipErrorT) -> Result<()> {
        if code == HIP_SUCCESS {
            Ok(())
        } else {
            let desc = self.get_error_string(code);
            Err(HipError::api_error(op, code, desc.as_deref()))
        }
    }

    fn get_error_string(&self, code: HipErrorT) -> Option<String> {
        let sym_ptr = self
            .symbols
            .resolve(
                &self.symbols.hip_get_error_string,
                &self.lib,
                "hipGetErrorString",
            )
            .ok()?;
        // SAFETY: hipGetErrorString has signature `extern "C" fn(hipError_t) -> *const c_char`.
        let func: unsafe extern "C" fn(HipErrorT) -> *const c_char =
            unsafe { std::mem::transmute(sym_ptr) };
        // SAFETY: Invoking resolved hipGetErrorString with integer error code is safe.
        let ptr = unsafe { func(code) };
        if ptr.is_null() {
            None
        } else {
            // SAFETY: The driver returns a static NUL-terminated C string for standard error codes.
            Some(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    // --- Device Management ---

    /// Queries the number of available HIP GPU devices (Spec 14 §3).
    pub fn device_count(&self) -> Result<u32> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_get_device_count,
            &self.lib,
            "hipGetDeviceCount",
        )?;
        // SAFETY: hipGetDeviceCount signature is `extern "C" fn(*mut c_int) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut c_int) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut count: c_int = 0;
        // SAFETY: count is a valid local stack allocation.
        self.check("hipGetDeviceCount", unsafe { func(&mut count) })?;
        if count < 0 {
            return Err(HipError::InvalidDeviceCount { count });
        }
        Ok(count as u32)
    }

    /// Sets the active HIP device ordinal for the calling host thread (Spec 14 §3).
    pub fn set_device(&self, device_id: i32) -> Result<()> {
        let sym = self
            .symbols
            .resolve(&self.symbols.hip_set_device, &self.lib, "hipSetDevice")?;
        // SAFETY: hipSetDevice signature is `extern "C" fn(c_int) -> hipError_t`.
        let func: unsafe extern "C" fn(c_int) -> HipErrorT = unsafe { std::mem::transmute(sym) };
        // SAFETY: Passing integer device ID to hipSetDevice is safe.
        self.check("hipSetDevice", unsafe { func(device_id) })
    }

    /// Queries the currently active HIP device ordinal for the calling thread (Spec 14 §3).
    pub fn get_device(&self) -> Result<i32> {
        let sym = self
            .symbols
            .resolve(&self.symbols.hip_get_device, &self.lib, "hipGetDevice")?;
        // SAFETY: hipGetDevice signature is `extern "C" fn(*mut c_int) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut c_int) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut dev: c_int = 0;
        // SAFETY: dev is a valid local stack allocation.
        self.check("hipGetDevice", unsafe { func(&mut dev) })?;
        Ok(dev)
    }

    /// Queries hardware properties and capabilities of a HIP device (Spec 14 §3).
    pub fn get_device_properties(&self, device_id: i32) -> Result<DeviceProperties> {
        let sym = self.symbols.resolve_device_properties(&self.lib)?;
        // SAFETY: hipGetDevicePropertiesR0600 takes a pointer to RawDeviceProp (1472 bytes) and device ID.
        let func: unsafe extern "C" fn(*mut RawDeviceProp, c_int) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };

        let mut raw = RawDeviceProp::default();
        // SAFETY: raw is a valid stack-allocated 1472-byte RawDeviceProp struct.
        self.check("hipGetDevicePropertiesR0600", unsafe {
            func(&mut raw, device_id)
        })?;

        let name = parse_fixed_c_string(&raw.name);
        let gcn_arch_name = parse_fixed_c_string(&raw.gcn_arch_name);

        Ok(DeviceProperties {
            name,
            total_global_mem: raw.total_global_mem as u64,
            shared_mem_per_block: raw.shared_mem_per_block,
            regs_per_block: raw.regs_per_block,
            warp_size: raw.warp_size,
            max_threads_per_block: raw.max_threads_per_block,
            max_threads_dim: raw.max_threads_dim,
            max_grid_size: raw.max_grid_size,
            clock_rate_khz: raw.clock_rate,
            major: raw.major,
            minor: raw.minor,
            multi_processor_count: raw.multi_processor_count,
            gcn_arch_name,
            pci_bus_id: raw.pci_bus_id,
            pci_device_id: raw.pci_device_id,
            pci_domain_id: raw.pci_domain_id,
            is_multi_gpu_board: raw.is_multi_gpu_board != 0,
            can_map_host_memory: raw.can_map_host_memory != 0,
            concurrent_kernels: raw.concurrent_kernels != 0,
            ecc_enabled: raw.ecc_enabled != 0,
            cooperative_launch: raw.cooperative_launch != 0,
        })
    }

    // --- Memory Operations ---

    /// Allocates linear device memory on the current active device (Spec 14 §3).
    pub fn malloc(&self, size_bytes: usize) -> Result<*mut c_void> {
        let sym = self
            .symbols
            .resolve(&self.symbols.hip_malloc, &self.lib, "hipMalloc")?;
        // SAFETY: hipMalloc signature is `extern "C" fn(*mut *mut c_void, usize) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut *mut c_void, usize) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: ptr is a valid stack-allocated pointer.
        self.check("hipMalloc", unsafe { func(&mut ptr, size_bytes) })?;
        if ptr.is_null() && size_bytes > 0 {
            return Err(HipError::NullPointer("hipMalloc returned null pointer"));
        }
        Ok(ptr)
    }

    /// Frees linear device memory allocated by [`malloc`](Self::malloc) (Spec 14 §3).
    ///
    /// # Safety
    /// `ptr` must be a pointer returned from a HIP memory allocation function or null.
    pub unsafe fn free(&self, ptr: *mut c_void) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        let sym = self
            .symbols
            .resolve(&self.symbols.hip_free, &self.lib, "hipFree")?;
        // SAFETY: hipFree signature is `extern "C" fn(*mut c_void) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut c_void) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees ptr is a valid HIP allocation or null.
        self.check("hipFree", func(ptr))
    }

    /// Allocates page-locked (pinned) host memory accessible by device (Spec 14 §3).
    pub fn host_malloc(&self, size_bytes: usize, flags: u32) -> Result<*mut c_void> {
        let sym = self.symbols.resolve_host_malloc(&self.lib)?;
        // SAFETY: hipHostMalloc signature is `extern "C" fn(*mut *mut c_void, usize, c_uint) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: ptr is a valid stack-allocated pointer.
        self.check("hipHostMalloc", unsafe {
            func(&mut ptr, size_bytes, flags)
        })?;
        if ptr.is_null() && size_bytes > 0 {
            return Err(HipError::NullPointer("hipHostMalloc returned null pointer"));
        }
        Ok(ptr)
    }

    /// Frees page-locked host memory allocated by [`host_malloc`](Self::host_malloc) (Spec 14 §3).
    ///
    /// # Safety
    /// `ptr` must be a pointer returned from a HIP host memory allocation function or null.
    pub unsafe fn host_free(&self, ptr: *mut c_void) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        let sym = self.symbols.resolve_host_free(&self.lib)?;
        // SAFETY: hipHostFree signature is `extern "C" fn(*mut c_void) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut c_void) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees ptr is a valid HIP host allocation or null.
        self.check("hipHostFree", func(ptr))
    }

    /// Copies memory synchronously between host and device (Spec 14 §3).
    ///
    /// # Safety
    /// Caller must guarantee `dst` and `src` are valid for read/write of `count` bytes.
    pub unsafe fn memcpy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: MemcpyKind,
    ) -> Result<()> {
        let sym = self
            .symbols
            .resolve(&self.symbols.hip_memcpy, &self.lib, "hipMemcpy")?;
        // SAFETY: hipMemcpy signature is `extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> HipErrorT =
            std::mem::transmute(sym);
        // SAFETY: Caller guarantees valid pointers for `count` bytes.
        self.check("hipMemcpy", func(dst, src, count, kind.as_raw()))
    }

    /// Asynchronously copies memory between host, device, or device peers in a stream (Spec 14 §3).
    ///
    /// # Safety
    /// Caller must guarantee `dst` and `src` are valid for read/write of `count` bytes in `stream`.
    pub unsafe fn memcpy_async(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: MemcpyKind,
        stream: HipStreamT,
    ) -> Result<()> {
        let sym =
            self.symbols
                .resolve(&self.symbols.hip_memcpy_async, &self.lib, "hipMemcpyAsync")?;
        // SAFETY: hipMemcpyAsync signature is `extern "C" fn(*mut c_void, *const c_void, usize, c_int, HipStreamT) -> hipError_t`.
        let func: unsafe extern "C" fn(
            *mut c_void,
            *const c_void,
            usize,
            c_int,
            HipStreamT,
        ) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees valid pointers for `count` bytes in `stream`.
        self.check(
            "hipMemcpyAsync",
            func(dst, src, count, kind.as_raw(), stream),
        )
    }

    /// Copies memory asynchronously between two distinct device contexts (Spec 5 §7, Spec 14 §3).
    ///
    /// # Safety
    /// Caller must guarantee `dst` and `src` are valid device pointers on `dst_dev` and `src_dev`.
    pub unsafe fn memcpy_peer_async(
        &self,
        dst: *mut c_void,
        dst_dev: i32,
        src: *const c_void,
        src_dev: i32,
        count: usize,
        stream: HipStreamT,
    ) -> Result<()> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_memcpy_peer_async,
            &self.lib,
            "hipMemcpyPeerAsync",
        )?;
        // SAFETY: hipMemcpyPeerAsync signature is `extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize, HipStreamT) -> hipError_t`.
        let func: unsafe extern "C" fn(
            *mut c_void,
            c_int,
            *const c_void,
            c_int,
            usize,
            HipStreamT,
        ) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees valid pointers and enabled peer access between devices.
        self.check(
            "hipMemcpyPeerAsync",
            func(dst, dst_dev, src, src_dev, count, stream),
        )
    }

    // --- Stream Operations ---

    /// Creates a new execution stream with default flags (Spec 14 §3).
    pub fn stream_create(&self) -> Result<HipStreamT> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_stream_create,
            &self.lib,
            "hipStreamCreate",
        )?;
        // SAFETY: hipStreamCreate signature is `extern "C" fn(*mut HipStreamT) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut HipStreamT) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut stream: HipStreamT = std::ptr::null_mut();
        // SAFETY: stream is a valid local pointer.
        self.check("hipStreamCreate", unsafe { func(&mut stream) })?;
        Ok(stream)
    }

    /// Creates a new execution stream with custom flags (Spec 14 §3).
    pub fn stream_create_with_flags(&self, flags: StreamFlags) -> Result<HipStreamT> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_stream_create_with_flags,
            &self.lib,
            "hipStreamCreateWithFlags",
        )?;
        // SAFETY: hipStreamCreateWithFlags signature is `extern "C" fn(*mut HipStreamT, c_uint) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut HipStreamT, c_uint) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut stream: HipStreamT = std::ptr::null_mut();
        // SAFETY: stream is a valid local pointer.
        self.check("hipStreamCreateWithFlags", unsafe {
            func(&mut stream, flags.as_raw())
        })?;
        Ok(stream)
    }

    /// Destroys a stream handle (Spec 14 §3).
    ///
    /// # Safety
    /// `stream` must be a valid HIP stream handle or null.
    pub unsafe fn stream_destroy(&self, stream: HipStreamT) -> Result<()> {
        if stream.is_null() {
            return Ok(());
        }
        let sym = self.symbols.resolve(
            &self.symbols.hip_stream_destroy,
            &self.lib,
            "hipStreamDestroy",
        )?;
        // SAFETY: hipStreamDestroy signature is `extern "C" fn(HipStreamT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipStreamT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees stream is a valid handle or null.
        self.check("hipStreamDestroy", func(stream))
    }

    /// Blocks host thread until all queued operations in `stream` have completed (Spec 14 §3).
    ///
    /// # Safety
    /// `stream` must be a valid HIP stream handle or null.
    pub unsafe fn stream_synchronize(&self, stream: HipStreamT) -> Result<()> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_stream_synchronize,
            &self.lib,
            "hipStreamSynchronize",
        )?;
        // SAFETY: hipStreamSynchronize signature is `extern "C" fn(HipStreamT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipStreamT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees stream is a valid handle or null.
        self.check("hipStreamSynchronize", func(stream))
    }

    /// Returns `true` if all operations in `stream` have completed (Spec 14 §3).
    ///
    /// # Safety
    /// `stream` must be a valid HIP stream handle or null.
    pub unsafe fn stream_query(&self, stream: HipStreamT) -> Result<bool> {
        let sym =
            self.symbols
                .resolve(&self.symbols.hip_stream_query, &self.lib, "hipStreamQuery")?;
        // SAFETY: hipStreamQuery signature is `extern "C" fn(HipStreamT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipStreamT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees stream is a valid handle or null.
        let code = func(stream);
        if code == HIP_SUCCESS {
            Ok(true)
        } else if code == HIP_ERROR_NOT_READY {
            Ok(false)
        } else {
            self.check("hipStreamQuery", code)?;
            Ok(true)
        }
    }

    /// Enqueues a stream wait on an event (Spec 14 §3).
    ///
    /// # Safety
    /// `stream` and `event` must be valid HIP handles.
    pub unsafe fn stream_wait_event(
        &self,
        stream: HipStreamT,
        event: HipEventT,
        flags: u32,
    ) -> Result<()> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_stream_wait_event,
            &self.lib,
            "hipStreamWaitEvent",
        )?;
        // SAFETY: hipStreamWaitEvent signature is `extern "C" fn(HipStreamT, HipEventT, c_uint) -> hipError_t`.
        let func: unsafe extern "C" fn(HipStreamT, HipEventT, c_uint) -> HipErrorT =
            std::mem::transmute(sym);
        // SAFETY: Caller guarantees stream and event are valid handles.
        self.check("hipStreamWaitEvent", func(stream, event, flags))
    }

    // --- Event Operations ---

    /// Creates an event with default flags (Spec 14 §3).
    pub fn event_create(&self) -> Result<HipEventT> {
        let sym =
            self.symbols
                .resolve(&self.symbols.hip_event_create, &self.lib, "hipEventCreate")?;
        // SAFETY: hipEventCreate signature is `extern "C" fn(*mut HipEventT) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut HipEventT) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut event: HipEventT = std::ptr::null_mut();
        // SAFETY: event is a valid local pointer.
        self.check("hipEventCreate", unsafe { func(&mut event) })?;
        Ok(event)
    }

    /// Creates an event with custom flags (Spec 14 §3).
    pub fn event_create_with_flags(&self, flags: EventFlags) -> Result<HipEventT> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_event_create_with_flags,
            &self.lib,
            "hipEventCreateWithFlags",
        )?;
        // SAFETY: hipEventCreateWithFlags signature is `extern "C" fn(*mut HipEventT, c_uint) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut HipEventT, c_uint) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut event: HipEventT = std::ptr::null_mut();
        // SAFETY: event is a valid local pointer.
        self.check("hipEventCreateWithFlags", unsafe {
            func(&mut event, flags.as_raw())
        })?;
        Ok(event)
    }

    /// Destroys an event handle (Spec 14 §3).
    ///
    /// # Safety
    /// `event` must be a valid HIP event handle or null.
    pub unsafe fn event_destroy(&self, event: HipEventT) -> Result<()> {
        if event.is_null() {
            return Ok(());
        }
        let sym = self.symbols.resolve(
            &self.symbols.hip_event_destroy,
            &self.lib,
            "hipEventDestroy",
        )?;
        // SAFETY: hipEventDestroy signature is `extern "C" fn(HipEventT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipEventT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees event is a valid handle or null.
        self.check("hipEventDestroy", func(event))
    }

    /// Captures the execution state of `stream` into `event` (Spec 14 §3).
    ///
    /// # Safety
    /// `event` and `stream` must be valid handles.
    pub unsafe fn event_record(&self, event: HipEventT, stream: HipStreamT) -> Result<()> {
        let sym =
            self.symbols
                .resolve(&self.symbols.hip_event_record, &self.lib, "hipEventRecord")?;
        // SAFETY: hipEventRecord signature is `extern "C" fn(HipEventT, HipStreamT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipEventT, HipStreamT) -> HipErrorT =
            std::mem::transmute(sym);
        // SAFETY: Caller guarantees valid handles.
        self.check("hipEventRecord", func(event, stream))
    }

    /// Blocks host thread until `event` has completed (Spec 14 §3).
    ///
    /// # Safety
    /// `event` must be a valid event handle.
    pub unsafe fn event_synchronize(&self, event: HipEventT) -> Result<()> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_event_synchronize,
            &self.lib,
            "hipEventSynchronize",
        )?;
        // SAFETY: hipEventSynchronize signature is `extern "C" fn(HipEventT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipEventT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees event is a valid handle.
        self.check("hipEventSynchronize", func(event))
    }

    /// Computes elapsed time in milliseconds between two recorded events (Spec 11 §7, Spec 14 §3).
    ///
    /// # Safety
    /// `start` and `stop` must be valid recorded events.
    pub unsafe fn event_elapsed_time(&self, start: HipEventT, stop: HipEventT) -> Result<f32> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_event_elapsed_time,
            &self.lib,
            "hipEventElapsedTime",
        )?;
        // SAFETY: hipEventElapsedTime signature is `extern "C" fn(*mut f32, HipEventT, HipEventT) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut f32, HipEventT, HipEventT) -> HipErrorT =
            std::mem::transmute(sym);
        let mut ms: f32 = 0.0;
        // SAFETY: ms is a valid local stack float.
        self.check("hipEventElapsedTime", func(&mut ms, start, stop))?;
        Ok(ms)
    }

    /// Queries completion status of an event without blocking (Spec 14 §3).
    ///
    /// # Safety
    /// `event` must be a valid event handle.
    pub unsafe fn event_query(&self, event: HipEventT) -> Result<bool> {
        let sym =
            self.symbols
                .resolve(&self.symbols.hip_event_query, &self.lib, "hipEventQuery")?;
        // SAFETY: hipEventQuery signature is `extern "C" fn(HipEventT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipEventT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees event is a valid handle.
        let code = func(event);
        if code == HIP_SUCCESS {
            Ok(true)
        } else if code == HIP_ERROR_NOT_READY {
            Ok(false)
        } else {
            self.check("hipEventQuery", code)?;
            Ok(true)
        }
    }

    // --- Module & Kernel Launch Operations ---

    /// Loads a compiled code object file (`.co`) as a HIP module (Spec 4 §10, Spec 14 §3).
    pub fn module_load(&self, fname: &Path) -> Result<HipModuleT> {
        let sym =
            self.symbols
                .resolve(&self.symbols.hip_module_load, &self.lib, "hipModuleLoad")?;
        // SAFETY: hipModuleLoad signature is `extern "C" fn(*mut HipModuleT, *const c_char) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut HipModuleT, *const c_char) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let c_fname = CString::new(fname.as_os_str().as_encoded_bytes()).map_err(|e| {
            HipError::InvalidNulByte {
                context: "module file path",
                nul_position: e.nul_position(),
            }
        })?;
        let mut module: HipModuleT = std::ptr::null_mut();
        // SAFETY: module is a valid local pointer and c_fname is a valid NUL-terminated C string.
        self.check("hipModuleLoad", unsafe {
            func(&mut module, c_fname.as_ptr())
        })?;
        Ok(module)
    }

    /// Loads an in-memory compiled code object image as a HIP module (Spec 4 §10, Spec 14 §3).
    pub fn module_load_data(&self, image: &[u8]) -> Result<HipModuleT> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_module_load_data,
            &self.lib,
            "hipModuleLoadData",
        )?;
        // SAFETY: hipModuleLoadData signature is `extern "C" fn(*mut HipModuleT, *const c_void) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut HipModuleT, *const c_void) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut module: HipModuleT = std::ptr::null_mut();
        // SAFETY: image slice points to contiguous in-memory code object bytes.
        self.check("hipModuleLoadData", unsafe {
            func(&mut module, image.as_ptr() as *const c_void)
        })?;
        Ok(module)
    }

    /// Unloads a loaded HIP module (Spec 14 §3).
    ///
    /// # Safety
    /// `module` must be a valid HIP module handle or null.
    pub unsafe fn module_unload(&self, module: HipModuleT) -> Result<()> {
        if module.is_null() {
            return Ok(());
        }
        let sym = self.symbols.resolve(
            &self.symbols.hip_module_unload,
            &self.lib,
            "hipModuleUnload",
        )?;
        // SAFETY: hipModuleUnload signature is `extern "C" fn(HipModuleT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipModuleT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees module is a valid handle or null.
        self.check("hipModuleUnload", func(module))
    }

    /// Resolves an exported kernel function symbol from a module (Spec 4 §10, Spec 14 §3).
    ///
    /// # Safety
    /// `module` must be a valid loaded HIP module handle.
    pub unsafe fn module_get_function(
        &self,
        module: HipModuleT,
        name: &str,
    ) -> Result<HipFunctionT> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_module_get_function,
            &self.lib,
            "hipModuleGetFunction",
        )?;
        // SAFETY: hipModuleGetFunction signature is `extern "C" fn(*mut HipFunctionT, HipModuleT, *const c_char) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut HipFunctionT, HipModuleT, *const c_char) -> HipErrorT =
            std::mem::transmute(sym);
        let c_name = CString::new(name).map_err(|e| HipError::InvalidNulByte {
            context: "kernel function name",
            nul_position: e.nul_position(),
        })?;
        let mut function: HipFunctionT = std::ptr::null_mut();
        // SAFETY: Caller guarantees module is valid and c_name is NUL-terminated.
        self.check(
            "hipModuleGetFunction",
            func(&mut function, module, c_name.as_ptr()),
        )?;
        Ok(function)
    }

    /// Enqueues a device kernel launch on a stream with grid/block dimensions (Spec 4 §10, Spec 14 §3).
    ///
    /// # Safety
    /// `func` and `stream` must be valid handles, and `kernel_params` must match the kernel ABI.
    pub unsafe fn module_launch_kernel(
        &self,
        func: HipFunctionT,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: HipStreamT,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_module_launch_kernel,
            &self.lib,
            "hipModuleLaunchKernel",
        )?;
        // SAFETY: hipModuleLaunchKernel signature matches ROCm kernel launch ABI.
        let launch_fn: unsafe extern "C" fn(
            HipFunctionT,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            HipStreamT,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> HipErrorT = std::mem::transmute(sym);

        let params_ptr = if kernel_params.is_empty() {
            std::ptr::null_mut()
        } else {
            kernel_params.as_mut_ptr()
        };

        // SAFETY: Caller guarantees func, stream, and parameter types are valid.
        self.check(
            "hipModuleLaunchKernel",
            launch_fn(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem,
                stream,
                params_ptr,
                std::ptr::null_mut(),
            ),
        )
    }

    // --- Graph Operations ---

    /// Begins stream capture for graph replay (Spec 1 §3.1, Spec 6 §2, Spec 14 §3).
    ///
    /// # Safety
    /// `stream` must be a valid HIP stream handle.
    pub unsafe fn stream_begin_capture(
        &self,
        stream: HipStreamT,
        mode: StreamCaptureMode,
    ) -> Result<()> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_stream_begin_capture,
            &self.lib,
            "hipStreamBeginCapture",
        )?;
        // SAFETY: hipStreamBeginCapture signature is `extern "C" fn(HipStreamT, c_int) -> hipError_t`.
        let func: unsafe extern "C" fn(HipStreamT, c_int) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees stream is valid.
        self.check("hipStreamBeginCapture", func(stream, mode.as_raw()))
    }

    /// Ends stream capture and returns the constructed graph handle (Spec 1 §3.1, Spec 6 §2, Spec 14 §3).
    ///
    /// # Safety
    /// `stream` must be actively capturing.
    pub unsafe fn stream_end_capture(&self, stream: HipStreamT) -> Result<HipGraphT> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_stream_end_capture,
            &self.lib,
            "hipStreamEndCapture",
        )?;
        // SAFETY: hipStreamEndCapture signature is `extern "C" fn(HipStreamT, *mut HipGraphT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipStreamT, *mut HipGraphT) -> HipErrorT =
            std::mem::transmute(sym);
        let mut graph: HipGraphT = std::ptr::null_mut();
        // SAFETY: graph is a valid local pointer.
        self.check("hipStreamEndCapture", func(stream, &mut graph))?;
        Ok(graph)
    }

    /// Instantiates an executable graph from a captured graph topology (Spec 6 §2, Spec 14 §3).
    ///
    /// # Safety
    /// `graph` must be a valid captured graph handle.
    pub unsafe fn graph_instantiate(&self, graph: HipGraphT) -> Result<HipGraphExecT> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_graph_instantiate,
            &self.lib,
            "hipGraphInstantiate",
        )?;
        let mut exec: HipGraphExecT = std::ptr::null_mut();

        // SAFETY: hipGraphInstantiate 5-argument signature in ROCm 6.x/7.x ABI.
        let func: unsafe extern "C" fn(
            *mut HipGraphExecT,
            HipGraphT,
            *mut HipGraphNodeT,
            *mut c_char,
            usize,
        ) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: exec is a valid local pointer.
        self.check(
            "hipGraphInstantiate",
            func(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            ),
        )?;

        Ok(exec)
    }

    /// Launches an instantiated executable graph on a stream (Spec 6 §2, Spec 14 §3).
    ///
    /// # Safety
    /// `graph_exec` and `stream` must be valid handles.
    pub unsafe fn graph_launch(&self, graph_exec: HipGraphExecT, stream: HipStreamT) -> Result<()> {
        let sym =
            self.symbols
                .resolve(&self.symbols.hip_graph_launch, &self.lib, "hipGraphLaunch")?;
        // SAFETY: hipGraphLaunch signature is `extern "C" fn(HipGraphExecT, HipStreamT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipGraphExecT, HipStreamT) -> HipErrorT =
            std::mem::transmute(sym);
        // SAFETY: Caller guarantees graph_exec and stream are valid handles.
        self.check("hipGraphLaunch", func(graph_exec, stream))
    }

    /// Destroys a captured graph definition (Spec 14 §3).
    ///
    /// # Safety
    /// `graph` must be a valid graph handle or null.
    pub unsafe fn graph_destroy(&self, graph: HipGraphT) -> Result<()> {
        if graph.is_null() {
            return Ok(());
        }
        let sym = self.symbols.resolve(
            &self.symbols.hip_graph_destroy,
            &self.lib,
            "hipGraphDestroy",
        )?;
        // SAFETY: hipGraphDestroy signature is `extern "C" fn(HipGraphT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipGraphT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees graph is valid or null.
        self.check("hipGraphDestroy", func(graph))
    }

    /// Destroys an instantiated executable graph (Spec 14 §3).
    ///
    /// # Safety
    /// `graph_exec` must be a valid handle or null.
    pub unsafe fn graph_exec_destroy(&self, graph_exec: HipGraphExecT) -> Result<()> {
        if graph_exec.is_null() {
            return Ok(());
        }
        let sym = self.symbols.resolve(
            &self.symbols.hip_graph_exec_destroy,
            &self.lib,
            "hipGraphExecDestroy",
        )?;
        // SAFETY: hipGraphExecDestroy signature is `extern "C" fn(HipGraphExecT) -> hipError_t`.
        let func: unsafe extern "C" fn(HipGraphExecT) -> HipErrorT = std::mem::transmute(sym);
        // SAFETY: Caller guarantees graph_exec is valid or null.
        self.check("hipGraphExecDestroy", func(graph_exec))
    }

    // --- Peer Access Operations ---

    /// Queries whether peer memory access is supported from `device` to `peer_device` (Spec 5 §7, Spec 14 §3).
    pub fn device_can_access_peer(&self, device: i32, peer_device: i32) -> Result<bool> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_device_can_access_peer,
            &self.lib,
            "hipDeviceCanAccessPeer",
        )?;
        // SAFETY: hipDeviceCanAccessPeer signature is `extern "C" fn(*mut c_int, c_int, c_int) -> hipError_t`.
        let func: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let mut can_access: c_int = 0;
        // SAFETY: can_access is a valid local stack allocation.
        self.check("hipDeviceCanAccessPeer", unsafe {
            func(&mut can_access, device, peer_device)
        })?;
        Ok(can_access != 0)
    }

    /// Enables peer direct memory access from current device to `peer_device` (Spec 5 §7, Spec 14 §3).
    pub fn device_enable_peer_access(&self, peer_device: i32, flags: u32) -> Result<()> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_device_enable_peer_access,
            &self.lib,
            "hipDeviceEnablePeerAccess",
        )?;
        // SAFETY: hipDeviceEnablePeerAccess signature is `extern "C" fn(c_int, c_uint) -> hipError_t`.
        let func: unsafe extern "C" fn(c_int, c_uint) -> HipErrorT =
            unsafe { std::mem::transmute(sym) };
        let code = unsafe { func(peer_device, flags) };
        if code == HIP_SUCCESS || code == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            Ok(())
        } else {
            self.check("hipDeviceEnablePeerAccess", code)
        }
    }

    /// Disables peer direct memory access to `peer_device` (Spec 5 §7, Spec 14 §3).
    pub fn device_disable_peer_access(&self, peer_device: i32) -> Result<()> {
        let sym = self.symbols.resolve(
            &self.symbols.hip_device_disable_peer_access,
            &self.lib,
            "hipDeviceDisablePeerAccess",
        )?;
        // SAFETY: hipDeviceDisablePeerAccess signature is `extern "C" fn(c_int) -> hipError_t`.
        let func: unsafe extern "C" fn(c_int) -> HipErrorT = unsafe { std::mem::transmute(sym) };
        let code = unsafe { func(peer_device) };
        if code == HIP_SUCCESS || code == HIP_ERROR_PEER_ACCESS_NOT_ENABLED {
            Ok(())
        } else {
            self.check("hipDeviceDisablePeerAccess", code)
        }
    }
}
