// SPDX-License-Identifier: Apache-2.0
//! Hard, thread-safe GPU allocation budget for constrained execution (Spec 14 §3).
//!
//! [`AllocationBudget`] enforces an immutable byte limit with shared atomic
//! accounting across clones, so a planning contract can feed its effective
//! VRAM bytes in once (as a plain `u64`/`usize` byte count) and every clone
//! draws from the same ledger. Reservation happens before `hipMalloc`; a
//! refusal returns [`HipError::BudgetExceeded`] and charges nothing, and a
//! reservation is rolled back if the underlying HIP allocation fails.
//!
//! Constrained execution must exclusively use the budgeted allocator path
//! ([`BudgetedDeviceBuffer::allocate`]). [`DeviceBuffer`] remains available
//! for physical/unconstrained callers that intentionally bypass the budget.
//!
//! This module has no dependency on `r9v-ir` and contains no workload-specific
//! branches: the budget only counts bytes.
//!
//! [`DeviceBuffer`]: crate::handles::DeviceBuffer
//! [`HipError::BudgetExceeded`]: crate::error::HipError::BudgetExceeded

use std::ffi::c_void;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crate::error::{HipError, Result};
use crate::handles::DeviceBuffer;
use crate::library::HipLibrary;
use crate::Stream;

/// Immutable byte limit with shared atomic accounting across clones.
///
/// Cloning shares the same ledger: all clones charge and release against one
/// atomic counter, so concurrent reservations from any clone never overcommit
/// the limit. [`AllocationBudget`] is `Send + Sync + Clone`.
#[derive(Debug, Clone)]
pub struct AllocationBudget {
    inner: Arc<BudgetInner>,
}

#[derive(Debug)]
struct BudgetInner {
    /// Immutable cap in bytes, fixed at construction.
    limit: u64,
    /// Currently charged bytes. Invariant: `used <= limit` at all times.
    used: AtomicU64,
}

impl AllocationBudget {
    /// Creates a budget capped at `limit_bytes` with zero bytes initially used.
    ///
    /// The `limit_bytes` value is intentionally a plain byte count so any
    /// planning contract can feed its effective VRAM bytes directly.
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            inner: Arc::new(BudgetInner {
                limit: limit_bytes,
                used: AtomicU64::new(0),
            }),
        }
    }

    /// Returns the immutable byte limit.
    pub fn limit(&self) -> u64 {
        self.inner.limit
    }

    /// Returns the bytes currently charged against the budget.
    pub fn used(&self) -> u64 {
        self.inner.used.load(Ordering::Acquire)
    }

    /// Returns the bytes remaining under the limit (`limit - used`, saturating).
    pub fn available(&self) -> u64 {
        self.inner.limit.saturating_sub(self.used())
    }

    /// Reserves `requested` bytes, or returns [`HipError::BudgetExceeded`].
    ///
    /// Uses a compare-and-swap loop so concurrent reservations never
    /// overcommit. `u64` arithmetic overflow is treated as a typed refusal,
    /// never a panic. A zero-byte request always succeeds without altering
    /// accounting.
    fn reserve(&self, requested: u64) -> Result<()> {
        let limit = self.inner.limit;
        let mut used = self.inner.used.load(Ordering::Acquire);
        loop {
            let Some(new_used) = used.checked_add(requested) else {
                return Err(Self::refusal(limit, used, requested));
            };
            if new_used > limit {
                return Err(Self::refusal(limit, used, requested));
            }
            match self.inner.used.compare_exchange_weak(
                used,
                new_used,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => used = actual,
            }
        }
    }

    /// Releases `charged` bytes back to the budget.
    ///
    /// Every successful [`reserve`](Self::reserve) is paired with exactly one
    /// release (on [`BudgetedDeviceBuffer`] drop or HIP-failure rollback), so
    /// the counter cannot underflow.
    fn release(&self, charged: u64) {
        if charged == 0 {
            return;
        }
        let previous = self.inner.used.fetch_sub(charged, Ordering::AcqRel);
        assert!(
            previous >= charged,
            "allocation budget released more bytes than charged"
        );
    }

    fn refusal(limit: u64, used: u64, requested: u64) -> HipError {
        HipError::BudgetExceeded {
            limit,
            used,
            requested,
            available: limit.saturating_sub(used),
        }
    }
}

/// RAII device buffer charged against an [`AllocationBudget`].
///
/// Created only via [`allocate`](Self::allocate), which reserves bytes before
/// calling `hipMalloc`, rolls the reservation back if HIP allocation fails,
/// and frees the HIP allocation before releasing exactly the charged bytes on
/// drop. The inner [`DeviceBuffer`]
/// is private with no accessor, `Deref`/`DerefMut` implementation, or
/// detach/extraction API, so accounting cannot be lost by moving the
/// underlying allocation out.
pub struct BudgetedDeviceBuffer {
    // `Option` lets `Drop` destroy the HIP allocation before returning its
    // bytes to the shared ledger. Releasing first would create a concurrent
    // window where the budget reports capacity while the old allocation is
    // still resident.
    buffer: Option<DeviceBuffer>,
    charged: u64,
    budget: AllocationBudget,
}

// SAFETY: BudgetedDeviceBuffer owns a DeviceBuffer (Send + Sync) plus an
// AllocationBudget (Send + Sync via Arc + atomics); ownership transfer across
// threads is safe and follows DeviceBuffer's borrowing rules (&mut for host
// writes, & for device reads).
unsafe impl Send for BudgetedDeviceBuffer {}
// SAFETY: Concurrent shared access is limited to read-style copies, matching
// DeviceBuffer's Sync contract.
unsafe impl Sync for BudgetedDeviceBuffer {}

impl BudgetedDeviceBuffer {
    /// Allocates `size_bytes` of device memory charged against `budget`.
    ///
    /// Reservation happens before `hipMalloc`: if the budget cannot cover the
    /// request, this returns [`HipError::BudgetExceeded`] without attempting
    /// any HIP call. If `hipMalloc` fails after a successful reservation, the
    /// reservation is rolled back before the error is returned.
    ///
    /// Zero-size allocations charge zero bytes and otherwise match
    /// [`DeviceBuffer::allocate`] behavior.
    pub fn allocate(
        budget: &AllocationBudget,
        lib: &Arc<HipLibrary>,
        size_bytes: usize,
    ) -> Result<Self> {
        // `usize` is at most 64 bits wide; saturate defensively without panicking.
        let requested = u64::try_from(size_bytes).unwrap_or(u64::MAX);
        budget.reserve(requested)?;
        match DeviceBuffer::allocate(lib, size_bytes) {
            Ok(buffer) => Ok(Self {
                buffer: Some(buffer),
                charged: requested,
                budget: budget.clone(),
            }),
            Err(e) => {
                budget.release(requested);
                Err(e)
            }
        }
    }

    /// Returns the allocation size in bytes.
    pub fn size(&self) -> usize {
        self.buffer().size()
    }

    /// Returns the bytes this buffer charged against its budget.
    pub fn charged_bytes(&self) -> u64 {
        self.charged
    }

    /// Returns the budget this buffer is charged against.
    pub fn budget(&self) -> &AllocationBudget {
        &self.budget
    }

    /// Returns the raw device memory pointer.
    pub fn as_ptr(&self) -> *const c_void {
        self.buffer().as_ptr()
    }

    /// Returns the mutable raw device memory pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.buffer_mut().as_mut_ptr()
    }

    /// Copies bytes synchronously from host slice to device buffer.
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `src.len() > self.size()`.
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        self.buffer_mut().copy_from_host(src)
    }

    /// Copies bytes synchronously from device buffer to host slice.
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `dst.len() < self.size()`.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        self.buffer().copy_to_host(dst)
    }

    /// Asynchronously copies bytes from host slice to device buffer in `stream`.
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `src.len() > self.size()`.
    ///
    /// # Safety
    /// The caller must ensure that `src` remains valid, allocated, and untouched
    /// until all operations queued in `stream` at the time of call have completed
    /// execution (e.g. by calling [`Stream::synchronize`]).
    pub unsafe fn copy_from_host_async(&mut self, src: &[u8], stream: &Stream) -> Result<()> {
        self.buffer_mut().copy_from_host_async(src, stream)
    }

    /// Asynchronously copies bytes from device buffer to host slice in `stream`.
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `dst.len() < self.size()`.
    ///
    /// # Safety
    /// The caller must ensure that `dst` remains valid, allocated, and untouched
    /// until all operations queued in `stream` at the time of call have completed
    /// execution (e.g. by calling [`Stream::synchronize`]).
    pub unsafe fn copy_to_host_async(&self, dst: &mut [u8], stream: &Stream) -> Result<()> {
        self.buffer().copy_to_host_async(dst, stream)
    }

    /// Copies bytes asynchronously into another budgeted buffer on the same device.
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `self.size() > dst.size()`.
    ///
    /// # Safety
    /// The caller must ensure that `self`, `dst`, and `stream` remain valid until
    /// all operations queued in `stream` complete.
    pub unsafe fn copy_to_device_async(
        &self,
        dst: &mut BudgetedDeviceBuffer,
        stream: &Stream,
    ) -> Result<()> {
        self.buffer().copy_to_device_async(dst.buffer_mut(), stream)
    }

    /// Asynchronously copies memory into `dst` on `dst_device` via peer transfer.
    ///
    /// # Errors
    /// Returns [`HipError::BufferTooSmall`] if `self.size() > dst.size()`.
    ///
    /// # Safety
    /// The caller must ensure that `self`, `dst`, and `stream` remain valid until
    /// all operations queued in `stream` complete, and that peer access is enabled
    /// between `src_device` and `dst_device`.
    pub unsafe fn copy_to_peer_async(
        &self,
        src_device: i32,
        dst: &mut BudgetedDeviceBuffer,
        dst_device: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.buffer()
            .copy_to_peer_async(src_device, dst.buffer_mut(), dst_device, stream)
    }

    fn buffer(&self) -> &DeviceBuffer {
        self.buffer
            .as_ref()
            .expect("budgeted buffer exists until Drop")
    }

    fn buffer_mut(&mut self) -> &mut DeviceBuffer {
        self.buffer
            .as_mut()
            .expect("budgeted buffer exists until Drop")
    }
}

impl Drop for BudgetedDeviceBuffer {
    fn drop(&mut self) {
        // Free device memory first. Only after hipFree returns may another
        // thread observe these bytes as available in the shared budget.
        drop(self.buffer.take());
        self.budget.release(self.charged);
    }
}
