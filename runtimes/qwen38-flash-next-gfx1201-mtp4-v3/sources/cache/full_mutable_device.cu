// SPDX-License-Identifier: Apache-2.0

#include "full_mutable_device.h"
#include "coop_choice.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>

namespace full_mutable_device {

namespace {

/**
 * Kernel: Initialize arena with warmstart configuration.
 * Validates hot IDs and uniqueness before mutating any state.
 */
__global__ void init_arena_kernel(
    uint8_t* __restrict__ arena_ptr,
    const int* __restrict__ hot_list,
    int num_hot,
    int rank,
    int capacity,
    int* __restrict__ hot_map,
    int* __restrict__ cold_map,
    int* __restrict__ cache_map) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  init_arena_core(arena_ptr, hot_list, num_hot, rank, capacity, hot_map, cold_map, cache_map);
}

/**
 * Kernel: Device Planner (single block, parallel over routes, race-free, deterministic).
 * Consumes expert_ids (diagnostic scope <= 64).
 * Enforces frozen V2 policy semantics matching MutablePoolState.
 */
__global__ void device_planner_kernel(
    uint8_t* __restrict__ arena_ptr,
    const int* __restrict__ expert_ids,
    int num_routes,
    int rank,
    int capacity,
    int max_inserts) {
  // One block cooperatively scans the existing slot set. Policy, tie order,
  // admission list, clocks and counters match device_planner_core exactly.
  constexpr int THREADS = 256;
  const int t = threadIdx.x;
  const auto p = unpack_arena(arena_ptr, rank, capacity);
  __shared__ int tags[kRank1TotalSlots], map[kNumExperts];
  __shared__ int protected_slot[kRank1TotalSlots];
  __shared__ int64_t clocks[kRank1TotalSlots], event;
  __shared__ int routes[kDiagnosticMaxRoutes], unique[kDiagnosticMaxRoutes];
  __shared__ int misses[kDiagnosticMaxRoutes];
  __shared__ int first_touch[kDiagnosticMaxRoutes], unique_is_miss[kDiagnosticMaxRoutes];
  __shared__ CoopChoice choices[THREADS];
  __shared__ int bad, unique_count, miss_count, admissions, evictions, skip, hits;

  if (t == 0) {
    p.counters[kScratchCurrentEventAdmissions] = 0;
    event = p.event_index[0];
    bad = ((rank != 0 && rank != 1) ||
           (rank == 0 && capacity != kRank0TotalSlots) ||
           (rank == 1 && capacity != kRank1TotalSlots) ||
           max_inserts < 0 || max_inserts > kMaxInsertsPerEvent ||
           (num_routes != 10 && num_routes != 50) || event < 0 || event == INT64_MAX);
    unique_count = miss_count = admissions = evictions = skip = hits = 0;
  }
  __syncthreads();
  if (bad) {
    if (t == 0) {
      p.counters[kCounterIdxInvalidRouteCount] += 1;
      p.counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
    }
    return;
  }
  if (t < num_routes) {
    const int e = expert_ids[t]; routes[t] = e;
    if (e < 0 || e >= kNumExperts) atomicExch(&bad, 1);
  }
  __syncthreads();
  if (bad) {
    if (t == 0) {
      p.counters[kCounterIdxInvalidRouteCount] += 1;
      p.counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
    }
    return; // uniform across the block, before any persistent state change
  }
  for (int s = t; s < capacity; s += THREADS) {
    tags[s] = p.pool_tags[s]; clocks[s] = p.pool_clock[s]; protected_slot[s] = 0;
  }
  for (int e = t; e < kNumExperts; e += THREADS) map[e] = p.pool_map[e];
  __syncthreads();
  // First-touch dedupe and miss list, one thread per route. A single thread
  // doing this serially cost ~40 us per layer; order matches the serial core.
  if (t < num_routes) {
    const int e = routes[t];
    int first = 1;
    for (int j = 0; j < t; ++j) if (routes[j] == e) { first = 0; break; }
    first_touch[t] = first;
  }
  __syncthreads();
  if (t < num_routes && first_touch[t]) {
    int pos = 0;
    for (int j = 0; j < t; ++j) pos += first_touch[j];
    const int e = routes[t], slot = map[e];
    const int miss = !(slot >= 0 && slot < capacity);
    unique[pos] = e;
    unique_is_miss[pos] = miss;
    if (!miss) protected_slot[slot] = 1;  // distinct experts own distinct slots
    atomicAdd(&unique_count, 1);
  }
  __syncthreads();
  if (t < unique_count && unique_is_miss[t]) {
    int pos = 0;
    for (int j = 0; j < t; ++j) pos += unique_is_miss[j];
    misses[pos] = unique[t];
    atomicAdd(&miss_count, 1);
  }
  if (t == 0) skip = unique_count > capacity / 2;
  __syncthreads();
  const int limit = skip ? 0 : (miss_count < max_inserts ? miss_count : max_inserts);
  for (int c = 0; c < limit; ++c) {
    CoopChoice best{INT64_MAX, INT_MAX, 2};
    for (int s = t; s < capacity; s += THREADS) {
      if (tags[s] < 0) best = better_choice(best, CoopChoice{0, s, 0});
      else if (!protected_slot[s]) best = better_choice(best, CoopChoice{clocks[s], s, 1});
    }
    choices[t] = best;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
      if (t < stride) choices[t] = better_choice(choices[t], choices[t + stride]);
      __syncthreads();
    }
    if (choices[0].kind == 2) break; // uniform, all slots protected
    if (t == 0) {
      const int target = choices[0].slot, cand = misses[c];
      const int old = tags[target];
      if (old >= 0 && old < kNumExperts) {
        map[old] = -1; p.pool_map[old] = -1; ++evictions;
        p.evicted_ids[admissions] = old;
      } else p.evicted_ids[admissions] = -1;
      tags[target] = cand; map[cand] = target; protected_slot[target] = 1;
      p.pool_tags[target] = cand; p.pool_map[cand] = target;
      p.miss_ids[admissions] = cand; p.miss_slots[admissions] = target;
      ++admissions;
    }
    __syncthreads();
  }
  // Each unique expert has a distinct slot, so these stores cannot race.
  if (t < unique_count) {
    const int slot = map[unique[t]];
    if (slot >= 0 && slot < capacity) p.pool_clock[slot] = event + 1;
  }
  __syncthreads();
  if (t < num_routes && map[routes[t]] >= 0) atomicAdd(&hits, 1);
  __syncthreads();
  if (t == 0) {
    p.event_index[0] = event + 1;
    p.counters[kCounterIdxTotalEvents] += 1;
    p.counters[kCounterIdxTokenHits] += hits;
    p.counters[kCounterIdxTokenMisses] += num_routes - hits;
    p.counters[kCounterIdxAdmittedCount] += admissions;
    p.counters[kCounterIdxEvictionsCount] += evictions;
    p.counters[kCounterIdxUncachedFallbackCount] += miss_count - admissions;
    if (skip) p.counters[kCounterIdxReadthroughEventsCount] += 1;
    p.counters[kCounterIdxPlanStatus] = skip ? kStatusReadthrough :
        (admissions < miss_count ? kStatusShortfall : kStatusOk);
    p.counters[kScratchCurrentEventAdmissions] = admissions;
  }
}

/**
 * Kernel: Parallel Gather for BOTH projections (w13 and w2).
 * Operates on device without host roundtrips.
 * Vectorized uint4 transfers (16 bytes per element) across threads and blocks
 * when 16-byte aligned; correct scalar fallback when unaligned.
 * Same-stream ordering with publish_map_kernel ensures completion before publication.
 */
__global__ void parallel_gather_kernel(
    const uint8_t* __restrict__ arena_ptr,
    const uint8_t* __restrict__ cold_w13,
    const uint8_t* __restrict__ cold_w2,
    uint8_t* __restrict__ hot_w13,
    uint8_t* __restrict__ hot_w2,
    uint8_t* __restrict__ cache_w13,
    uint8_t* __restrict__ cache_w2,
    int64_t w13_expert_bytes,
    int64_t w2_expert_bytes,
    int rank,
    int capacity) {
  const auto p = unpack_arena(const_cast<uint8_t*>(arena_ptr), rank, capacity);
  const int current_admissions = static_cast<int>(p.counters[kScratchCurrentEventAdmissions]);

  // blockIdx.y selects the admission index in [0, current_admissions - 1]
  const int adm_idx = blockIdx.y;
  if (adm_idx >= current_admissions || adm_idx >= kMaxInsertsPerEvent) {
    return;
  }

  const int cand = p.miss_ids[adm_idx];
  const int target = p.miss_slots[adm_idx];

  // Validate candidate and target bounds
  if (cand < 0 || cand >= kNumExperts) {
    return;
  }
  if (target < 0 || target >= capacity) {
    return;
  }

  // Resolve source pointers (from full 512 UVA host backing)
  const uint8_t* src_w13 = cold_w13 + static_cast<int64_t>(cand) * w13_expert_bytes;
  const uint8_t* src_w2 = cold_w2 + static_cast<int64_t>(cand) * w2_expert_bytes;

  // Resolve destination pointers (hot buffer or cache buffer)
  uint8_t* dst_w13 = nullptr;
  uint8_t* dst_w2 = nullptr;

  if (rank == 0) {
    if (target < kRank0HotSlots) {
      dst_w13 = hot_w13 + static_cast<int64_t>(target) * w13_expert_bytes;
      dst_w2 = hot_w2 + static_cast<int64_t>(target) * w2_expert_bytes;
    } else {
      const int cache_slot = target - kRank0HotSlots;
      if (cache_slot < 0 || cache_slot >= kRank0CacheSlots || cache_w13 == nullptr || cache_w2 == nullptr) {
        return;
      }
      dst_w13 = cache_w13 + static_cast<int64_t>(cache_slot) * w13_expert_bytes;
      dst_w2 = cache_w2 + static_cast<int64_t>(cache_slot) * w2_expert_bytes;
    }
  } else {
    if (target < 0 || target >= kRank1HotSlots || hot_w13 == nullptr || hot_w2 == nullptr) {
      return;
    }
    dst_w13 = hot_w13 + static_cast<int64_t>(target) * w13_expert_bytes;
    dst_w2 = hot_w2 + static_cast<int64_t>(target) * w2_expert_bytes;
  }

  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  // Transfer W13: vectorized uint4 (16-byte) if aligned, else scalar fallback
  const bool w13_aligned16 = (reinterpret_cast<uintptr_t>(src_w13) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(dst_w13) % 16 == 0);
  if (w13_aligned16) {
    const int64_t w13_vectors = w13_expert_bytes / sizeof(uint4);
    for (int64_t idx = tid; idx < w13_vectors; idx += stride) {
      reinterpret_cast<uint4*>(dst_w13)[idx] =
          reinterpret_cast<const uint4*>(src_w13)[idx];
    }
    const int64_t tail_start = w13_vectors * sizeof(uint4);
    for (int64_t b = tail_start + tid; b < w13_expert_bytes; b += stride) {
      dst_w13[b] = src_w13[b];
    }
  } else {
    for (int64_t b = tid; b < w13_expert_bytes; b += stride) {
      dst_w13[b] = src_w13[b];
    }
  }

  // Transfer W2: vectorized uint4 (16-byte) if aligned, else scalar fallback
  const bool w2_aligned16 = (reinterpret_cast<uintptr_t>(src_w2) % 16 == 0) &&
                            (reinterpret_cast<uintptr_t>(dst_w2) % 16 == 0);
  if (w2_aligned16) {
    const int64_t w2_vectors = w2_expert_bytes / sizeof(uint4);
    for (int64_t idx = tid; idx < w2_vectors; idx += stride) {
      reinterpret_cast<uint4*>(dst_w2)[idx] =
          reinterpret_cast<const uint4*>(src_w2)[idx];
    }
    const int64_t tail_start = w2_vectors * sizeof(uint4);
    for (int64_t b = tail_start + tid; b < w2_expert_bytes; b += stride) {
      dst_w2[b] = src_w2[b];
    }
  } else {
    for (int64_t b = tid; b < w2_expert_bytes; b += stride) {
      dst_w2[b] = src_w2[b];
    }
  }
}

/**
 * Kernel: Publish map updates to hot_map and cache_map.
 * Executed after gather completes on the same stream.
 * Uses __threadfence() for memory visibility across controllers.
 */
__global__ void publish_map_kernel(
    uint8_t* __restrict__ arena_ptr,
    int* __restrict__ hot_map,
    int* __restrict__ cache_map,
    int rank,
    int capacity) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  __threadfence();
  publish_map_core(arena_ptr, hot_map, cache_map, rank, capacity);
  __threadfence();
}

} // namespace

/**
 * PyTorch C++ Extension Function: Initialize arena.
 * Validates exact rank/capacity, contiguous int32 warmstart 62/427, metadata 512.
 */
void full_mutable_device_init(
    torch::Tensor arena,
    torch::Tensor hot_list,
    torch::Tensor hot_map,
    torch::Tensor cold_map,
    torch::optional<torch::Tensor> cache_map,
    int64_t rank,
    int64_t capacity) {
  TORCH_CHECK(arena.is_cuda() && hot_list.is_cuda() && hot_map.is_cuda() && cold_map.is_cuda(),
              "all tensors must be on GPU/UVA");
  const auto device = arena.device();
  TORCH_CHECK(hot_list.device() == device && hot_map.device() == device && cold_map.device() == device,
              "all tensors must share the same device");
  TORCH_CHECK(arena.scalar_type() == torch::kUInt8, "arena must be uint8");
  TORCH_CHECK(arena.is_contiguous(), "arena must be contiguous");
  TORCH_CHECK(arena.numel() == kArenaTotalBytes, "arena must be exactly 32768 bytes");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(arena.data_ptr<uint8_t>()) % alignof(int64_t) == 0,
              "arena pointer must be 8-byte aligned for int64");

  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK((rank == 0 && capacity == kRank0TotalSlots) ||
              (rank == 1 && capacity == kRank1TotalSlots),
              "capacity mismatch for specified rank: rank 0 expects 217, rank 1 expects 427");

  const int expected_hot = (rank == 0) ? kRank0HotSlots : kRank1HotSlots;
  TORCH_CHECK(hot_list.scalar_type() == torch::kInt32 && hot_list.is_contiguous(),
              "hot_list must be contiguous int32");
  TORCH_CHECK(hot_list.numel() == expected_hot,
              "hot_list size mismatch: rank 0 expects 62, rank 1 expects 427");

  TORCH_CHECK(hot_map.scalar_type() == torch::kInt32 && hot_map.is_contiguous() && hot_map.numel() == kNumExperts,
              "hot_map must be contiguous int32 of length 512");
  TORCH_CHECK(cold_map.scalar_type() == torch::kInt32 && cold_map.is_contiguous() && cold_map.numel() == kNumExperts,
              "cold_map must be contiguous int32 of length 512");

  int* cache_map_ptr = nullptr;
  if (rank == 0) {
    TORCH_CHECK(cache_map.has_value() && cache_map.value().is_cuda(),
                "rank 0 requires cache_map on GPU");
    TORCH_CHECK(cache_map.value().device() == device, "cache_map device mismatch");
    TORCH_CHECK(cache_map.value().scalar_type() == torch::kInt32 && cache_map.value().is_contiguous(),
                "cache_map must be contiguous int32");
    TORCH_CHECK(cache_map.value().numel() == kNumExperts,
                "cache_map must have length 512");
    cache_map_ptr = cache_map.value().data_ptr<int>();
  }

  c10::cuda::CUDAGuard guard(device);
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();

  init_arena_kernel<<<1, 1, 0, stream>>>(
      arena.data_ptr<uint8_t>(),
      hot_list.data_ptr<int>(),
      static_cast<int>(hot_list.numel()),
      static_cast<int>(rank),
      static_cast<int>(capacity),
      hot_map.data_ptr<int>(),
      cold_map.data_ptr<int>(),
      cache_map_ptr);
  AT_CUDA_CHECK(hipGetLastError());
}

/**
 * PyTorch C++ Extension Function: Step Planner + Parallel Gather + Map Publish.
 * Operates on the active stream without host roundtrips or device value inspection.
 *
 * NOTE ON EXECUTION PIPELINE AND COMPUTE:
 * full_mutable_device_step executes the device planner, parallel gather, and map publish.
 * It does NOT invoke the downstream compute kernel; that is queued by the inference pipeline caller.
 * In accordance with asynchronous execution, this extension does not inspect device counter values on host.
 * If invalid routes or event endpoint overflow occurs, the planner fails closed, leaving cache/tags/clocks
 * unmodified and zero admissions recorded for gather/publish. Callers executing negative tests must not
 * enqueue downstream compute on invalid inputs; no whole-graph suppression is invented.
 * Same-stream ordering ensures gather finishes before publication, and publication before compute.
 */
void full_mutable_device_step(
    torch::Tensor arena,
    torch::Tensor expert_ids,
    torch::Tensor cold_w13,
    torch::Tensor cold_w2,
    torch::Tensor hot_w13,
    torch::Tensor hot_w2,
    torch::Tensor hot_map,
    torch::Tensor cold_map,
    torch::optional<torch::Tensor> cache_w13,
    torch::optional<torch::Tensor> cache_w2,
    torch::optional<torch::Tensor> cache_map,
    int64_t rank,
    int64_t capacity,
    int64_t max_inserts) {
  // 1. Device validation
  TORCH_CHECK(arena.is_cuda() && expert_ids.is_cuda() &&
              cold_w13.is_cuda() && cold_w2.is_cuda() &&
              hot_w13.is_cuda() && hot_w2.is_cuda() &&
              hot_map.is_cuda() && cold_map.is_cuda(),
              "all core tensors must be GPU/UVA tensors");
  const auto device = arena.device();
  TORCH_CHECK(expert_ids.device() == device && cold_w13.device() == device &&
              cold_w2.device() == device && hot_w13.device() == device &&
              hot_w2.device() == device && hot_map.device() == device &&
              cold_map.device() == device,
              "all tensors must share the same accelerator device");

  // 2. Dtype validation
  TORCH_CHECK(arena.scalar_type() == torch::kUInt8 &&
              cold_w13.scalar_type() == torch::kUInt8 &&
              cold_w2.scalar_type() == torch::kUInt8 &&
              hot_w13.scalar_type() == torch::kUInt8 &&
              hot_w2.scalar_type() == torch::kUInt8,
              "arena and weight tensors must be uint8");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32 &&
              hot_map.scalar_type() == torch::kInt32 &&
              cold_map.scalar_type() == torch::kInt32,
              "metadata and expert_ids must be int32");

  // 3. Contiguity validation
  TORCH_CHECK(arena.is_contiguous() && expert_ids.is_contiguous() &&
              cold_w13.is_contiguous() && cold_w2.is_contiguous() &&
              hot_w13.is_contiguous() && hot_w2.is_contiguous() &&
              hot_map.is_contiguous() && cold_map.is_contiguous(),
              "all tensors must be contiguous");

  // 4. Alignment and storage offset checks
  TORCH_CHECK(arena.numel() == kArenaTotalBytes, "arena must be exactly 32768 bytes");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(arena.data_ptr<uint8_t>()) % alignof(int64_t) == 0,
              "arena pointer must be 8-byte aligned for int64");

  // 5. Exact rank and capacity validation
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK((rank == 0 && capacity == kRank0TotalSlots) ||
              (rank == 1 && capacity == kRank1TotalSlots),
              "capacity mismatch: rank 0 expects 217, rank 1 expects 427");

  // 6. max_inserts validation
  TORCH_CHECK(max_inserts >= 0 && max_inserts <= kMaxInsertsPerEvent,
              "max_inserts must be bounded in [0, 64]");

  // 7. Metadata shape validation
  TORCH_CHECK(hot_map.dim() == 1 && hot_map.numel() == kNumExperts,
              "hot_map must be 1D with 512 entries");
  TORCH_CHECK(cold_map.dim() == 1 && cold_map.numel() == kNumExperts,
              "cold_map must be 1D with 512 entries");

  // 8. Expert routes exact documented shapes (flat 10 or 50, or 2D 1x10 or 5x10)
  TORCH_CHECK(
      (expert_ids.dim() == 1 && (expert_ids.size(0) == 10 || expert_ids.size(0) == 50)) ||
      (expert_ids.dim() == 2 && ((expert_ids.size(0) == 1 && expert_ids.size(1) == 10) ||
                                  (expert_ids.size(0) == 5 && expert_ids.size(1) == 10))),
      "expert_ids must have exact documented shape: flat (10 or 50) or 2D (1x10 or 5x10)");
  const int num_routes = static_cast<int>(expert_ids.numel());

  // 9. Weight tensor shapes: 3D [slots, rows, bytes_per_row]
  TORCH_CHECK(cold_w13.dim() == 3 && cold_w2.dim() == 3 &&
              hot_w13.dim() == 3 && hot_w2.dim() == 3,
              "weight tensors must be 3D [slots, rows, bytes_per_row]");
  TORCH_CHECK(cold_w13.size(0) == kNumExperts && cold_w2.size(0) == kNumExperts,
              "cold weights must have full 512 expert backing");

  const int expected_hot = (rank == 0) ? kRank0HotSlots : kRank1HotSlots;
  TORCH_CHECK(hot_w13.size(0) == expected_hot && hot_w2.size(0) == expected_hot,
              "hot weight slots mismatch expected rank capacity: rank 0 expects 62, rank 1 expects 427");

  // Check that hot projection dimensions exactly match cold projection dimensions
  TORCH_CHECK(hot_w13.size(1) == cold_w13.size(1) && hot_w13.size(2) == cold_w13.size(2),
              "hot_w13 shape [rows, bytes_per_row] must match cold_w13");
  TORCH_CHECK(hot_w2.size(1) == cold_w2.size(1) && hot_w2.size(2) == cold_w2.size(2),
              "hot_w2 shape [rows, bytes_per_row] must match cold_w2");

  uint8_t* cache_w13_ptr = nullptr;
  uint8_t* cache_w2_ptr = nullptr;
  int* cache_map_ptr = nullptr;

  if (rank == 0) {
    TORCH_CHECK(cache_w13.has_value() && cache_w2.has_value() && cache_map.has_value(),
                "rank 0 requires cache_w13, cache_w2, and cache_map");
    auto& c13 = cache_w13.value();
    auto& c2 = cache_w2.value();
    auto& cm = cache_map.value();

    TORCH_CHECK(c13.is_cuda() && c2.is_cuda() && cm.is_cuda() &&
                c13.device() == device && c2.device() == device && cm.device() == device,
                "rank 0 cache tensors must be on the same device");
    TORCH_CHECK(c13.scalar_type() == torch::kUInt8 && c2.scalar_type() == torch::kUInt8 &&
                cm.scalar_type() == torch::kInt32,
                "rank 0 cache tensors dtype mismatch");
    TORCH_CHECK(c13.is_contiguous() && c2.is_contiguous() && cm.is_contiguous(),
                "rank 0 cache tensors must be contiguous");
    TORCH_CHECK(cm.dim() == 1 && cm.numel() == kNumExperts,
                "cache_map must be 1D with 512 entries");
    TORCH_CHECK(c13.dim() == 3 && c2.dim() == 3,
                "cache weight tensors must be 3D [slots, rows, bytes_per_row]");
    TORCH_CHECK(c13.size(0) == kRank0CacheSlots && c2.size(0) == kRank0CacheSlots,
                "rank 0 cache slots must be exactly 155");
    TORCH_CHECK(c13.size(1) == cold_w13.size(1) && c13.size(2) == cold_w13.size(2) &&
                c2.size(1) == cold_w2.size(1) && c2.size(2) == cold_w2.size(2),
                "rank 0 cache projection dimensions must match cold weights");

    cache_w13_ptr = c13.data_ptr<uint8_t>();
    cache_w2_ptr = c2.data_ptr<uint8_t>();
    cache_map_ptr = cm.data_ptr<int>();
  }

  c10::cuda::CUDAGuard guard(device);
  const auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();

  const int64_t w13_expert_bytes = cold_w13.size(1) * cold_w13.size(2);
  const int64_t w2_expert_bytes = cold_w2.size(1) * cold_w2.size(2);

  // Step 1: Device planner kernel
  device_planner_kernel<<<1, 256, 0, stream>>>(
      arena.data_ptr<uint8_t>(),
      expert_ids.data_ptr<int>(),
      num_routes,
      static_cast<int>(rank),
      static_cast<int>(capacity),
      static_cast<int>(max_inserts));

  // Step 2: Parallel gather kernel across admitted experts
  constexpr int kBlocksPerExpert = 32;
  const dim3 gather_grid(kBlocksPerExpert, kMaxInsertsPerEvent, 1);
  parallel_gather_kernel<<<gather_grid, 256, 0, stream>>>(
      arena.data_ptr<uint8_t>(),
      cold_w13.data_ptr<uint8_t>(),
      cold_w2.data_ptr<uint8_t>(),
      hot_w13.data_ptr<uint8_t>(),
      hot_w2.data_ptr<uint8_t>(),
      cache_w13_ptr,
      cache_w2_ptr,
      w13_expert_bytes,
      w2_expert_bytes,
      static_cast<int>(rank),
      static_cast<int>(capacity));

  // Step 3: Publish kernel
  publish_map_kernel<<<1, 1, 0, stream>>>(
      arena.data_ptr<uint8_t>(),
      hot_map.data_ptr<int>(),
      cache_map_ptr,
      static_cast<int>(rank),
      static_cast<int>(capacity));

  AT_CUDA_CHECK(hipGetLastError());
}

} // namespace full_mutable_device

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Full Mutable Device Planner and Gather Extension (Qwen3.8 GGUF)";
  m.def("full_mutable_device_init", &full_mutable_device::full_mutable_device_init,
        "Initialize full mutable device arena and warmstart maps");
  m.def("full_mutable_device_step", &full_mutable_device::full_mutable_device_step,
        "Device planner + parallel gather + map publish on current stream");
}
