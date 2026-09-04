// SPDX-License-Identifier: Apache-2.0
//! Dynamic symbol resolution and caching for libamdhip64 (Spec 14 §2, §3).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::error::{HipError, Result};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

/// Low-level dynamic library handle wrapping `dlopen` / `dlclose`.
pub(crate) struct DynamicLibrary {
    handle: *mut c_void,
    path: PathBuf,
}

// SAFETY: DynamicLibrary wraps a POSIX dlopen handle (*mut c_void). In Linux/glibc, dlopen handles
// are thread-safe and can be transferred safely across threads.
unsafe impl Send for DynamicLibrary {}
// SAFETY: DynamicLibrary operations via dlsym are thread-safe under POSIX and glibc dynamic linkers.
unsafe impl Sync for DynamicLibrary {}

impl DynamicLibrary {
    /// Attempts to open a shared library at `path`.
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| {
            HipError::InvalidNulByte {
                context: "library path",
                nul_position: e.nul_position(),
            }
        })?;

        // Clear existing dlerror
        unsafe { dlerror() };

        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            let err_msg = unsafe {
                let err_ptr = dlerror();
                if err_ptr.is_null() {
                    "unknown dlopen error".to_owned()
                } else {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                }
            };
            let requested = path.as_os_str().to_string_lossy();
            let missing_requested_library = err_msg.starts_with(requested.as_ref())
                && err_msg.contains("No such file or directory");
            if missing_requested_library {
                return Err(HipError::LibraryNotFound {
                    searched: vec![format!("{}: {err_msg}", path.display())],
                });
            }
            return Err(HipError::LibraryLoadFailed {
                attempts: vec![format!("{}: {err_msg}", path.display())],
            });
        }

        Ok(Self {
            handle,
            path: path.to_path_buf(),
        })
    }

    /// Resolves a raw symbol address using `dlsym`.
    pub(crate) fn dlsym(&self, name: &'static str) -> Result<*mut c_void> {
        let c_name = CString::new(name).map_err(|e| HipError::InvalidNulByte {
            context: "symbol name",
            nul_position: e.nul_position(),
        })?;

        unsafe { dlerror() };
        let sym = unsafe { dlsym(self.handle, c_name.as_ptr()) };
        if sym.is_null() {
            let err_msg = unsafe {
                let err_ptr = dlerror();
                if err_ptr.is_null() {
                    "symbol not found".to_owned()
                } else {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                }
            };
            return Err(HipError::SymbolNotFound {
                symbol: name,
                details: format!("{err_msg} in {}", self.path.display()),
            });
        }

        Ok(sym)
    }

    /// Returns the resolved path of the loaded shared library.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { dlclose(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Table of lazily resolved and cached function pointers for HIP runtime entry points.
#[derive(Default)]
pub(crate) struct SymbolCache {
    pub(crate) hip_get_device_count: AtomicPtr<c_void>,
    pub(crate) hip_set_device: AtomicPtr<c_void>,
    pub(crate) hip_get_device: AtomicPtr<c_void>,
    pub(crate) hip_get_device_properties_r0600: AtomicPtr<c_void>,
    pub(crate) hip_device_get_pci_bus_id: AtomicPtr<c_void>,
    pub(crate) hip_get_error_string: AtomicPtr<c_void>,

    pub(crate) hip_malloc: AtomicPtr<c_void>,
    pub(crate) hip_free: AtomicPtr<c_void>,
    pub(crate) hip_host_malloc: AtomicPtr<c_void>,
    pub(crate) hip_host_free: AtomicPtr<c_void>,
    pub(crate) hip_memcpy: AtomicPtr<c_void>,
    pub(crate) hip_memcpy_async: AtomicPtr<c_void>,
    pub(crate) hip_memcpy_peer_async: AtomicPtr<c_void>,

    pub(crate) hip_stream_create: AtomicPtr<c_void>,
    pub(crate) hip_stream_create_with_flags: AtomicPtr<c_void>,
    pub(crate) hip_stream_destroy: AtomicPtr<c_void>,
    pub(crate) hip_stream_synchronize: AtomicPtr<c_void>,
    pub(crate) hip_stream_query: AtomicPtr<c_void>,
    pub(crate) hip_stream_wait_event: AtomicPtr<c_void>,

    pub(crate) hip_event_create: AtomicPtr<c_void>,
    pub(crate) hip_event_create_with_flags: AtomicPtr<c_void>,
    pub(crate) hip_event_destroy: AtomicPtr<c_void>,
    pub(crate) hip_event_record: AtomicPtr<c_void>,
    pub(crate) hip_event_synchronize: AtomicPtr<c_void>,
    pub(crate) hip_event_elapsed_time: AtomicPtr<c_void>,
    pub(crate) hip_event_query: AtomicPtr<c_void>,

    pub(crate) hip_module_load: AtomicPtr<c_void>,
    pub(crate) hip_module_load_data: AtomicPtr<c_void>,
    pub(crate) hip_module_unload: AtomicPtr<c_void>,
    pub(crate) hip_module_get_function: AtomicPtr<c_void>,
    pub(crate) hip_module_launch_kernel: AtomicPtr<c_void>,

    pub(crate) hip_stream_begin_capture: AtomicPtr<c_void>,
    pub(crate) hip_stream_end_capture: AtomicPtr<c_void>,
    pub(crate) hip_graph_instantiate: AtomicPtr<c_void>,
    pub(crate) hip_graph_launch: AtomicPtr<c_void>,
    pub(crate) hip_graph_destroy: AtomicPtr<c_void>,
    pub(crate) hip_graph_exec_destroy: AtomicPtr<c_void>,

    pub(crate) hip_device_can_access_peer: AtomicPtr<c_void>,
    pub(crate) hip_device_enable_peer_access: AtomicPtr<c_void>,
    pub(crate) hip_device_disable_peer_access: AtomicPtr<c_void>,
}

impl SymbolCache {
    /// Loads a symbol pointer from `cell` or lazily resolves it from `lib` and caches it.
    pub(crate) fn resolve(
        &self,
        cell: &AtomicPtr<c_void>,
        lib: &DynamicLibrary,
        name: &'static str,
    ) -> Result<*mut c_void> {
        let ptr = cell.load(Ordering::Acquire);
        if !ptr.is_null() {
            return Ok(ptr);
        }

        let sym = lib.dlsym(name)?;
        cell.store(sym, Ordering::Release);
        Ok(sym)
    }

    /// Resolves `hipGetDevicePropertiesR0600` strictly without legacy fallbacks.
    pub(crate) fn resolve_device_properties(&self, lib: &DynamicLibrary) -> Result<*mut c_void> {
        self.resolve(
            &self.hip_get_device_properties_r0600,
            lib,
            "hipGetDevicePropertiesR0600",
        )
    }

    /// Resolves `hipHostMalloc` or falls back to `hipMallocHost`.
    pub(crate) fn resolve_host_malloc(&self, lib: &DynamicLibrary) -> Result<*mut c_void> {
        let ptr = self.hip_host_malloc.load(Ordering::Acquire);
        if !ptr.is_null() {
            return Ok(ptr);
        }

        let sym = lib
            .dlsym("hipHostMalloc")
            .or_else(|_| lib.dlsym("hipMallocHost"))?;

        self.hip_host_malloc.store(sym, Ordering::Release);
        Ok(sym)
    }

    /// Resolves `hipHostFree` or falls back to `hipFreeHost`.
    pub(crate) fn resolve_host_free(&self, lib: &DynamicLibrary) -> Result<*mut c_void> {
        let ptr = self.hip_host_free.load(Ordering::Acquire);
        if !ptr.is_null() {
            return Ok(ptr);
        }

        let sym = lib
            .dlsym("hipHostFree")
            .or_else(|_| lib.dlsym("hipFreeHost"))?;

        self.hip_host_free.store(sym, Ordering::Release);
        Ok(sym)
    }
}
