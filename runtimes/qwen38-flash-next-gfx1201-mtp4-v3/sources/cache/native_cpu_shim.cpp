// SPDX-License-Identifier: Apache-2.0
#include "full_mutable_device.h"

extern "C" {

void c_get_arena_offsets(int rank, int capacity, size_t* out_offsets) {
  const auto off = full_mutable_device::get_arena_offsets(rank, capacity);
  out_offsets[0] = off.event_index_offset;
  out_offsets[1] = off.counters_offset;
  out_offsets[2] = off.pool_clock_offset;
  out_offsets[3] = off.pool_map_offset;
  out_offsets[4] = off.pool_tags_offset;
  out_offsets[5] = off.miss_ids_offset;
  out_offsets[6] = off.miss_slots_offset;
  out_offsets[7] = off.evicted_ids_offset;
  out_offsets[8] = off.request_flags_offset;
  out_offsets[9] = off.total_used_bytes;
  out_offsets[10] = off.slack_bytes;
}

int c_init_arena(
    uint8_t* arena_ptr,
    const int* hot_list,
    int num_hot,
    int rank,
    int capacity,
    int* hot_map,
    int* cold_map,
    int* cache_map) {
  return full_mutable_device::init_arena_core(
      arena_ptr, hot_list, num_hot, rank, capacity, hot_map, cold_map, cache_map);
}

int c_device_planner(
    uint8_t* arena_ptr,
    const int* expert_ids,
    int num_routes,
    int rank,
    int capacity,
    int max_inserts,
    int strict_routes) {
  return full_mutable_device::device_planner_core(
      arena_ptr, expert_ids, num_routes, rank, capacity, max_inserts, strict_routes != 0);
}

int c_publish_map(
    uint8_t* arena_ptr,
    int* hot_map,
    int* cache_map,
    int rank,
    int capacity) {
  return full_mutable_device::publish_map_core(
      arena_ptr, hot_map, cache_map, rank, capacity);
}

void c_copy_projection_bytes(
    const uint8_t* src,
    uint8_t* dst,
    int64_t nbytes) {
  full_mutable_device::copy_projection_bytes(src, dst, nbytes);
}

}
