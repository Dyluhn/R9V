// Replays saved routing through the full mutable planner with the first `pinned` slots
// never evicted. At the compiled pin count it is checked against the real device_planner_core every event.
// stdin lines: rank layer pinned n_warm id... ; stdout: rank layer pinned split adm uncached token_miss
// Built and driven by run.py (its docstring has the g++ line); full_mutable_device.h is in the
// runtime's sources/cache.
#include "full_mutable_device.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace full_mutable_device;

static int pinned_planner(uint8_t* arena, const int* ids, int n, int rank, int cap, int pinned, int* uncached) {
  const auto p = unpack_arena(arena, rank, cap);
  int uniq[64], misses[64], nu = 0, nm = 0;
  bool seen[kNumExperts] = {false}, prot[kRank1TotalSlots] = {false};
  for (int r = 0; r < n; ++r) if (!seen[ids[r]]) { seen[ids[r]] = true; uniq[nu++] = ids[r]; }
  const bool skip = nu > cap / 2;
  for (int i = 0; i < nu; ++i) { int s = p.pool_map[uniq[i]]; if (s >= 0) prot[s] = true; else misses[nm++] = uniq[i]; }
  int adm = 0;
  const int64_t ev = p.event_index[0];
  for (int c = 0; !skip && c < nm && c < kMaxInsertsPerEvent; ++c) {
    int target = -1;
    for (int s = 0; s < cap; ++s) if (p.pool_tags[s] < 0) { target = s; break; }
    if (target < 0) {
      int64_t best = -1;
      for (int s = pinned; s < cap; ++s)
        if (p.pool_tags[s] >= 0 && !prot[s] && (best < 0 || p.pool_clock[s] < best)) { best = p.pool_clock[s]; target = s; }
      if (target < 0) break;
      p.pool_map[p.pool_tags[target]] = -1;
    }
    p.pool_tags[target] = misses[c]; p.pool_map[misses[c]] = target; prot[target] = true; ++adm;
  }
  for (int i = 0; i < nu; ++i) { int s = p.pool_map[uniq[i]]; if (s >= 0) p.pool_clock[s] = ev + 1; }
  p.event_index[0] = ev + 1;
  int hits = 0; for (int r = 0; r < n; ++r) hits += p.pool_map[ids[r]] >= 0;
  *uncached = nm - adm;
  return (adm << 8) | (n - hits);
}

int main(int argc, char** argv) {
  // routes.bin: int32 [events][48][50]; argv[2] = number of train events
  FILE* f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<int> routes(sz / 4); fread(routes.data(), 4, routes.size(), f); fclose(f);
  const int events = routes.size() / (48 * 50), train = atoi(argv[2]);
  int rank, layer, pinned, nwarm;
  while (scanf("%d %d %d %d", &rank, &layer, &pinned, &nwarm) == 4) {
    std::vector<int> warm(nwarm); for (auto& w : warm) scanf("%d", &w);
    const int cap = rank ? kRank1TotalSlots : kRank0TotalSlots, nhot = rank ? kRank1HotSlots : kRank0HotSlots;
    if (nwarm != nhot) { fprintf(stderr, "warm list must have %d ids\n", nhot); return 2; }
    std::vector<uint8_t> a(kArenaTotalBytes, 0), ref(kArenaTotalBytes, 0);
    init_arena_core(a.data(), warm.data(), nhot, rank, cap, nullptr, nullptr, nullptr);
    init_arena_core(ref.data(), warm.data(), nhot, rank, cap, nullptr, nullptr, nullptr);
    long long sum[2][3] = {{0}};
    for (int e = 0; e < events; ++e) {
      const int* ids = &routes[(e * 48 + layer) * 50];
      int unc, packed = pinned_planner(a.data(), ids, 50, rank, cap, pinned, &unc);
      int split = e >= train;
      sum[split][0] += packed >> 8; sum[split][1] += unc; sum[split][2] += packed & 255;
      if (pinned == pinned_slots(rank)) {  // the real planner pins exactly this many
        device_planner_core(ref.data(), ids, 50, rank, cap, 64);
        auto off = get_arena_offsets(rank, cap);  // compare persistent state: event, clocks, map, tags
        if (memcmp(a.data(), ref.data(), 8) || memcmp(a.data() + off.pool_clock_offset, ref.data() + off.pool_clock_offset, off.miss_ids_offset - off.pool_clock_offset)) {
          fprintf(stderr, "MISMATCH vs real planner rank %d layer %d event %d\n", rank, layer, e); return 3; }
      }
    }
    for (int s = 0; s < 2; ++s) printf("%d %d %d %s %lld %lld %lld\n", rank, layer, pinned, s ? "holdout" : "train", sum[s][0], sum[s][1], sum[s][2]);
  }
}
