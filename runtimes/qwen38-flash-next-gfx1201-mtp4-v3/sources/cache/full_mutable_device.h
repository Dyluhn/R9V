// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstddef>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define FMD_HOST_DEVICE __host__ __device__
#else
#define FMD_HOST_DEVICE
#endif

namespace full_mutable_device {

// Number of logical experts in the MoE layer
constexpr int kNumExperts = 512;

// Rank 0 capacities: 62 static hot + 155 cache slots = 217 resident slots
constexpr int kRank0HotSlots = 62;
constexpr int kRank0CacheSlots = 155;
constexpr int kRank0TotalSlots = 217;

// Rank 1 capacities: 427 static hot slots (0 cache) = 427 resident slots
constexpr int kRank1HotSlots = 427;
constexpr int kRank1CacheSlots = 0;
constexpr int kRank1TotalSlots = 427;

// Hard admission limit per event
constexpr int kMaxInsertsPerEvent = 64;

// Total preallocated arena bytes per rank per layer
constexpr size_t kArenaTotalBytes = 32768;

// Diagnostic scope limits
constexpr int kDiagnosticMaxRoutes = 64; // e.g. rows=1 (10 routes) or rows=5 (50 routes)
constexpr int kSupportedTopK = 10;

// Counter index constants (counters_int64[16])
constexpr int kCounterIdxTotalEvents = 0;
constexpr int kCounterIdxTokenHits = 1;
constexpr int kCounterIdxTokenMisses = 2;
constexpr int kCounterIdxAdmittedCount = 3;
constexpr int kCounterIdxEvictionsCount = 4;
constexpr int kCounterIdxUncachedFallbackCount = 5;
constexpr int kCounterIdxReadthroughEventsCount = 6;
constexpr int kCounterIdxInvalidRouteCount = 7;
constexpr int kCounterIdxPlanStatus = 8; // 0=OK, 1=READTHROUGH, 2=SHORTFALL, -1=INVALID_ROUTE
constexpr int kScratchCurrentEventAdmissions = 15;

// Status codes
constexpr int kStatusOk = 0;
constexpr int kStatusReadthrough = 1;
constexpr int kStatusShortfall = 2;
constexpr int kStatusInvalidRoute = -1;

/**
 * Typed layout view inside the single 32768-byte preallocated arena.
 * Exactly matches MEMORY_BUDGET.json planned_components.
 */
struct ArenaLayoutOffsets {
  size_t event_index_offset;   // int64_t[1] (8 bytes)
  size_t counters_offset;      // int64_t[16] (128 bytes)
  size_t pool_clock_offset;    // int64_t[capacity] (capacity * 8 bytes)
  size_t pool_map_offset;      // int32_t[512] (2048 bytes)
  size_t pool_tags_offset;     // int32_t[capacity] (capacity * 4 bytes)
  size_t miss_ids_offset;      // int32_t[64] (256 bytes)
  size_t miss_slots_offset;    // int32_t[64] (256 bytes)
  size_t evicted_ids_offset;   // int32_t[64] (256 bytes, within slack/scratch)
  size_t request_flags_offset; // int32_t[512] (2048 bytes)
  size_t total_used_bytes;
  size_t slack_bytes;
};

FMD_HOST_DEVICE inline ArenaLayoutOffsets get_arena_offsets(int rank, int capacity) {
  (void)rank;
  ArenaLayoutOffsets off{};
  off.event_index_offset = 0;  // 8B (aligned 8)
  off.counters_offset = 8;     // 128B (aligned 8)
  off.pool_clock_offset = 136; // capacity * 8B (aligned 8)
  const size_t after_clock = 136 + static_cast<size_t>(capacity) * 8;

  off.pool_map_offset = after_clock; // 512 * 4 = 2048B (aligned 4 & 8)
  const size_t after_map = off.pool_map_offset + kNumExperts * 4;

  off.pool_tags_offset = after_map; // capacity * 4B (aligned 4)
  const size_t after_tags = off.pool_tags_offset + static_cast<size_t>(capacity) * 4;

  off.miss_ids_offset = after_tags; // 64 * 4 = 256B (aligned 4)
  off.miss_slots_offset = off.miss_ids_offset + kMaxInsertsPerEvent * 4; // 256B (aligned 4)
  off.evicted_ids_offset = off.miss_slots_offset + kMaxInsertsPerEvent * 4; // 256B (aligned 4)
  off.request_flags_offset = off.evicted_ids_offset + kMaxInsertsPerEvent * 4; // 2048B (aligned 4)

  off.total_used_bytes = off.request_flags_offset + kNumExperts * 4;
  off.slack_bytes = kArenaTotalBytes - off.total_used_bytes;
  return off;
}

/**
 * Pointers unpacked from an arena buffer for a specific rank/capacity.
 */
struct ArenaPointers {
  int64_t* event_index;
  int64_t* counters;
  int64_t* pool_clock;
  int32_t* pool_map;
  int32_t* pool_tags;
  int32_t* miss_ids;
  int32_t* miss_slots;
  int32_t* evicted_ids;
  int32_t* request_flags;
  int capacity;
  int hot_capacity;
  int cache_capacity;
};

FMD_HOST_DEVICE inline ArenaPointers unpack_arena(uint8_t* arena_ptr, int rank, int capacity) {
  const auto off = get_arena_offsets(rank, capacity);
  ArenaPointers p{};
  p.event_index = reinterpret_cast<int64_t*>(arena_ptr + off.event_index_offset);
  p.counters = reinterpret_cast<int64_t*>(arena_ptr + off.counters_offset);
  p.pool_clock = reinterpret_cast<int64_t*>(arena_ptr + off.pool_clock_offset);
  p.pool_map = reinterpret_cast<int32_t*>(arena_ptr + off.pool_map_offset);
  p.pool_tags = reinterpret_cast<int32_t*>(arena_ptr + off.pool_tags_offset);
  p.miss_ids = reinterpret_cast<int32_t*>(arena_ptr + off.miss_ids_offset);
  p.miss_slots = reinterpret_cast<int32_t*>(arena_ptr + off.miss_slots_offset);
  p.evicted_ids = reinterpret_cast<int32_t*>(arena_ptr + off.evicted_ids_offset);
  p.request_flags = reinterpret_cast<int32_t*>(arena_ptr + off.request_flags_offset);
  p.capacity = capacity;
  p.hot_capacity = (rank == 0) ? kRank0HotSlots : kRank1HotSlots;
  p.cache_capacity = (rank == 0) ? kRank0CacheSlots : kRank1CacheSlots;
  return p;
}

/**
 * Core: Initialize arena with warmstart configuration.
 * Validates hot IDs and uniqueness before any cache/map state mutation.
 * Returns 0 on success, -1 on validation failure.
 */
FMD_HOST_DEVICE inline int init_arena_core(
    uint8_t* arena_ptr,
    const int* hot_list,
    int num_hot,
    int rank,
    int capacity,
    int* hot_map,
    int* cold_map,
    int* cache_map) {
  const auto off = get_arena_offsets(rank, capacity);
  auto* counters = reinterpret_cast<int64_t*>(arena_ptr + off.counters_offset);

  // Validate rank and capacity
  if ((rank == 0 && capacity != kRank0TotalSlots) ||
      (rank == 1 && capacity != kRank1TotalSlots)) {
    counters[kCounterIdxInvalidRouteCount] += 1;
    counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
    return -1;
  }

  // Validate num_hot
  const int expected_hot = (rank == 0) ? kRank0HotSlots : kRank1HotSlots;
  if (num_hot != expected_hot) {
    counters[kCounterIdxInvalidRouteCount] += 1;
    counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
    return -1;
  }

  // Validate hot IDs and uniqueness BEFORE modifying any state
  bool seen[kNumExperts] = {false};
  for (int i = 0; i < num_hot; ++i) {
    const int eid = hot_list[i];
    if (eid < 0 || eid >= kNumExperts || seen[eid]) {
      // Out of bounds or duplicate! Reject before any state mutation
      counters[kCounterIdxInvalidRouteCount] += 1;
      counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
      return -1;
    }
    seen[eid] = true;
  }

  // State mutation begins only after validation passes
  auto* event_index = reinterpret_cast<int64_t*>(arena_ptr + off.event_index_offset);
  auto* pool_clock = reinterpret_cast<int64_t*>(arena_ptr + off.pool_clock_offset);
  auto* pool_map = reinterpret_cast<int32_t*>(arena_ptr + off.pool_map_offset);
  auto* pool_tags = reinterpret_cast<int32_t*>(arena_ptr + off.pool_tags_offset);

  *event_index = 0;
  for (int i = 0; i < 16; ++i) {
    counters[i] = 0;
  }
  for (int s = 0; s < capacity; ++s) {
    pool_clock[s] = 0;
    pool_tags[s] = -1;
  }
  for (int e = 0; e < kNumExperts; ++e) {
    pool_map[e] = -1;
    if (cold_map != nullptr) cold_map[e] = e;
    if (hot_map != nullptr) hot_map[e] = -1;
    if (cache_map != nullptr) cache_map[e] = -1;
  }

  // Populate warmstart hot list
  for (int idx = 0; idx < num_hot; ++idx) {
    const int eid = hot_list[idx];
    pool_tags[idx] = eid;
    pool_map[eid] = idx;
    pool_clock[idx] = 0;

    if (rank == 0) {
      if (idx < kRank0HotSlots) {
        if (hot_map != nullptr) hot_map[eid] = idx;
      } else {
        if (cache_map != nullptr) cache_map[eid] = idx - kRank0HotSlots;
      }
    } else {
      if (hot_map != nullptr) hot_map[eid] = idx;
    }
  }

  return 0;
}

/**
 * Core: Device Planner (bounded serial prototype, race-free, deterministic).
 * Consumes expert_ids (diagnostic scope <= 64).
 * Enforces frozen V2 policy semantics matching MutablePoolState.
 * Returns admitted count on success, -1 on validation error.
 */
FMD_HOST_DEVICE inline int device_planner_core(
    uint8_t* arena_ptr,
    const int* expert_ids,
    int num_routes,
    int rank,
    int capacity,
    int max_inserts,
    bool strict_routes_10_50 = true) {
  const auto p = unpack_arena(arena_ptr, rank, capacity);

  // Scratch admission counter reset for this event
  p.counters[kScratchCurrentEventAdmissions] = 0;

  // Validate rank and capacity
  if ((rank == 0 && capacity != kRank0TotalSlots) ||
      (rank == 1 && capacity != kRank1TotalSlots)) {
    p.counters[kCounterIdxInvalidRouteCount] += 1;
    p.counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
    return -1;
  }

  // Validate max_inserts bounds [0, 64]
  if (max_inserts < 0 || max_inserts > kMaxInsertsPerEvent) {
    p.counters[kCounterIdxInvalidRouteCount] += 1;
    p.counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
    return -1;
  }

  // Route count validation
  if (strict_routes_10_50) {
    if (num_routes != 10 && num_routes != 50) {
      p.counters[kCounterIdxInvalidRouteCount] += 1;
      p.counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
      return -1;
    }
  } else {
    if (num_routes <= 0 || num_routes > kDiagnosticMaxRoutes) {
      p.counters[kCounterIdxInvalidRouteCount] += 1;
      p.counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
      return -1;
    }
  }

  // Validate expert ID bounds [0, kNumExperts - 1]
  for (int r = 0; r < num_routes; ++r) {
    const int eid = expert_ids[r];
    if (eid < 0 || eid >= kNumExperts) {
      p.counters[kCounterIdxInvalidRouteCount] += 1;
      p.counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
      return -1;
    }
  }

  // INT64_MAX event index endpoint refusal
  const int64_t cur_event = p.event_index[0];
  if (cur_event < 0 || cur_event >= INT64_MAX) {
    p.counters[kCounterIdxInvalidRouteCount] += 1;
    p.counters[kCounterIdxPlanStatus] = kStatusInvalidRoute;
    return -1;
  }

  // 1. Extract distinct requested experts preserving first-touch order
  int unique_requested[kDiagnosticMaxRoutes];
  bool seen[kNumExperts] = {false};
  int unique_cnt = 0;

  for (int r = 0; r < num_routes; ++r) {
    const int eid = expert_ids[r];
    if (!seen[eid]) {
      seen[eid] = true;
      unique_requested[unique_cnt++] = eid;
    }
  }

  // 2. Readthrough threshold check: distinct > floor(capacity * 0.5)
  const int threshold = capacity / 2;
  const bool skip_admission = (unique_cnt > threshold);

  // 3. Identify hits and misses; protect all current hits
  bool protected_slot[kRank1TotalSlots] = {false};
  int misses[kDiagnosticMaxRoutes];
  int miss_cnt = 0;
  int hit_cnt = 0;
  (void)hit_cnt;

  for (int i = 0; i < unique_cnt; ++i) {
    const int eid = unique_requested[i];
    const int slot = p.pool_map[eid];
    if (slot >= 0 && slot < capacity) {
      protected_slot[slot] = true;
      hit_cnt++;
    } else {
      misses[miss_cnt++] = eid;
    }
  }

  // 4. Bulk admission under frozen policy rules
  int admitted_cnt = 0;
  int evictions_cnt = 0;

  if (!skip_admission) {
    const int candidates_limit = (miss_cnt < max_inserts) ? miss_cnt : max_inserts;
    for (int c = 0; c < candidates_limit; ++c) {
      const int cand = misses[c];

      // a. Lowest empty slot
      int target = -1;
      for (int s = 0; s < capacity; ++s) {
        if (p.pool_tags[s] < 0) {
          target = s;
          break;
        }
      }

      // b. Victim selection among non-protected resident slots
      if (target < 0) {
        int64_t min_clock = -1;
        for (int s = 0; s < capacity; ++s) {
          if (p.pool_tags[s] >= 0 && !protected_slot[s]) {
            if (min_clock < 0 || p.pool_clock[s] < min_clock) {
              min_clock = p.pool_clock[s];
              target = s;
            }
          }
        }

        if (target < 0) {
          // Capacity shortfall: all resident slots protected in current event
          break;
        }

        // Evict old expert
        const int old_expert = p.pool_tags[target];
        if (old_expert >= 0 && old_expert < kNumExperts) {
          p.pool_map[old_expert] = -1;
          p.evicted_ids[admitted_cnt] = old_expert;
          evictions_cnt++;
        }
      } else {
        p.evicted_ids[admitted_cnt] = -1; // no eviction, filled empty slot
      }

      // Commit to planner arena state
      p.pool_tags[target] = cand;
      p.pool_map[cand] = target;
      protected_slot[target] = true; // protect newly admitted slot

      p.miss_ids[admitted_cnt] = cand;
      p.miss_slots[admitted_cnt] = target;
      admitted_cnt++;
    }
  }

  // 5. Recency update: all resident requested experts (hits + newly admitted)
  for (int i = 0; i < unique_cnt; ++i) {
    const int eid = unique_requested[i];
    const int slot = p.pool_map[eid];
    if (slot >= 0 && slot < capacity) {
      p.pool_clock[slot] = cur_event + 1;
    }
  }

  // 6. Increment event index
  p.event_index[0] = cur_event + 1;

  // 7. Update diagnostic counters
  int token_hits = 0;
  for (int r = 0; r < num_routes; ++r) {
    const int eid = expert_ids[r];
    if (p.pool_map[eid] >= 0) {
      token_hits++;
    }
  }
  const int token_misses = num_routes - token_hits;
  const int uncached_fallback = miss_cnt - admitted_cnt;

  p.counters[kCounterIdxTotalEvents] += 1;
  p.counters[kCounterIdxTokenHits] += token_hits;
  p.counters[kCounterIdxTokenMisses] += token_misses;
  p.counters[kCounterIdxAdmittedCount] += admitted_cnt;
  p.counters[kCounterIdxEvictionsCount] += evictions_cnt;
  p.counters[kCounterIdxUncachedFallbackCount] += uncached_fallback;

  if (skip_admission) {
    p.counters[kCounterIdxReadthroughEventsCount] += 1;
    p.counters[kCounterIdxPlanStatus] = kStatusReadthrough;
  } else if (admitted_cnt < miss_cnt) {
    p.counters[kCounterIdxPlanStatus] = kStatusShortfall;
  } else {
    p.counters[kCounterIdxPlanStatus] = kStatusOk;
  }

  // Device-side record of admissions for downstream gather/publish on same stream
  p.counters[kScratchCurrentEventAdmissions] = admitted_cnt;
  return admitted_cnt;
}

/**
 * Core: Publish map updates to hot_map and cache_map.
 * Executed after gather completes.
 * Returns number of published admissions on success, -1 on bounds error.
 */
FMD_HOST_DEVICE inline int publish_map_core(
    uint8_t* arena_ptr,
    int* hot_map,
    int* cache_map,
    int rank,
    int capacity) {
  const auto p = unpack_arena(arena_ptr, rank, capacity);
  const int current_admissions = static_cast<int>(p.counters[kScratchCurrentEventAdmissions]);
  if (current_admissions <= 0) return 0;
  if (current_admissions > kMaxInsertsPerEvent) return -1;

  for (int i = 0; i < current_admissions; ++i) {
    const int cand = p.miss_ids[i];
    const int target = p.miss_slots[i];
    const int old_expert = p.evicted_ids[i];

    if (target < 0 || target >= capacity) continue;

    // Clear old expert map entry
    if (old_expert >= 0 && old_expert < kNumExperts) {
      if (rank == 0) {
        if (target < kRank0HotSlots) {
          if (hot_map != nullptr) hot_map[old_expert] = -1;
        } else {
          if (cache_map != nullptr) cache_map[old_expert] = -1;
        }
      } else {
        if (hot_map != nullptr) hot_map[old_expert] = -1;
      }
    }

    // Publish new expert map entry
    if (cand >= 0 && cand < kNumExperts) {
      if (rank == 0) {
        if (target < kRank0HotSlots) {
          if (hot_map != nullptr) hot_map[cand] = target;
          if (cache_map != nullptr) cache_map[cand] = -1;
        } else {
          if (hot_map != nullptr) hot_map[cand] = -1;
          if (cache_map != nullptr) cache_map[cand] = target - kRank0HotSlots;
        }
      } else {
        if (hot_map != nullptr) hot_map[cand] = target;
      }
    }
  }
  return current_admissions;
}

/**
 * Projection copy helper: handles 16-byte aligned vectorization and scalar fallback.
 */
inline void copy_projection_bytes(
    const uint8_t* src,
    uint8_t* dst,
    int64_t nbytes) {
  const bool aligned16 = (reinterpret_cast<uintptr_t>(src) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(dst) % 16 == 0);
  if (aligned16) {
    const int64_t nvec = nbytes / 16;
    for (int64_t i = 0; i < nvec; ++i) {
      reinterpret_cast<uint64_t*>(dst)[i * 2] = reinterpret_cast<const uint64_t*>(src)[i * 2];
      reinterpret_cast<uint64_t*>(dst)[i * 2 + 1] = reinterpret_cast<const uint64_t*>(src)[i * 2 + 1];
    }
    for (int64_t b = nvec * 16; b < nbytes; ++b) {
      dst[b] = src[b];
    }
  } else {
    for (int64_t b = 0; b < nbytes; ++b) {
      dst[b] = src[b];
    }
  }
}

} // namespace full_mutable_device
