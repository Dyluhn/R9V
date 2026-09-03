// Spike A0.S6: p2p (roadmap §A0, spec 5 §§1-2, 6; spec 11 measurement table).
// Self-verifying HIP diagnostic: peer-map both R9700s, measure 16 KiB
// direct (hipMemcpyPeerAsync, only when capability says yes) vs explicit
// pinned-host-staged latency in both directions, with content checks.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(expr)                                                        \
    do {                                                                   \
        hipError_t _e = (expr);                                            \
        if (_e != hipSuccess) {                                            \
            printf("FAIL-CLOSED: %s:%d: %s returned %d (%s)\n", __FILE__,   \
                   __LINE__, #expr, (int)_e, hipGetErrorString(_e));       \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

const size_t kBytes = 16 * 1024;  // 16 KiB payload (spec 11 p2p size class)
const int kWarmups = 10;
const int kTrials = 50;

void fill_pattern(uint8_t* p, size_t n, uint32_t seed) {
    // Deterministic content: byte i = (seed + 31*i + (i>>8)) & 0xFF.
    for (size_t i = 0; i < n; ++i)
        p[i] = (uint8_t)((seed + 31u * (uint32_t)i + (uint32_t)(i >> 8)) & 0xFFu);
}

bool verify_pattern(const uint8_t* p, size_t n, uint32_t seed, const char* what) {
    for (size_t i = 0; i < n; ++i) {
        uint8_t want = (uint8_t)((seed + 31u * (uint32_t)i + (uint32_t)(i >> 8)) & 0xFFu);
        if (p[i] != want) {
            printf("CONTENT-MISMATCH %s at byte %zu: got %u want %u (seed %u)\n", what, i,
                   (unsigned)p[i], (unsigned)want, seed);
            return false;
        }
    }
    return true;
}

double median_ms(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return -1.0;
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double gbps(double bytes, double ms) {
    if (ms <= 0) return -1.0;
    return (bytes / (ms / 1000.0)) / 1e9;  // decimal GB/s per task
}

struct DirResult {
    int src = -1, dst = -1;
    bool peer_cap = false;   // hipDeviceCanAccessPeer(src -> dst)
    bool peer_enabled = false;
    double direct_ms = -1.0;  // median, -1 = N/A
    double direct_gbps = -1.0;
    bool direct_ok = false;
    double staged_ms = -1.0;
    double staged_gbps = -1.0;
    bool staged_ok = false;
};

bool print_props(int dev) {
    hipDeviceProp_t p;
    memset(&p, 0, sizeof p);
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) {
        printf("device %d: <props query failed>\n", dev);
        return false;
    }
    char busid[64] = "?";
    if (hipDeviceGetPCIBusId(busid, sizeof busid, dev) != hipSuccess)
        snprintf(busid, sizeof busid, "?");
    printf("device %d: name=\"%s\" gcnArch=\"%s\" pciBusId=\"%s\" "
           "totalGlobalMem=%zuMB multiProcessorCount=%d maxThreadsPerBlock=%d\n",
           dev, p.name, p.gcnArchName, busid,
           (size_t)(p.totalGlobalMem >> 20), p.multiProcessorCount,
           p.maxThreadsPerBlock);
    return strstr(p.name, "R9700") != nullptr &&
           strcmp(p.gcnArchName, "gfx1201") == 0;
}

}  // namespace

int main() {
    printf("A0.S6 p2p diagnostic (roadmap A0; spec 5 ss1-2,6; spec 11 meas table)\n");

    int ndev = 0;
    CHECK(hipGetDeviceCount(&ndev));
    printf("device_count=%d\n", ndev);
    if (ndev < 2) {
        printf("FAIL-CLOSED: need >= 2 HIP devices, have %d\n", ndev);
        return 1;
    }
    bool reference_devices = true;
    for (int d = 0; d < ndev; ++d) {
        bool is_r9700 = print_props(d);
        if (d < 2) reference_devices &= is_r9700;
    }
    if (!reference_devices) {
        printf("FAIL-CLOSED: devices 0 and 1 must both be R9700/gfx1201\n");
        return 1;
    }
    // Hardware fingerprint: driver/runtime versions.
    int rt = 0, drv = 0;
    if (hipRuntimeGetVersion(&rt) == hipSuccess) printf("hip_runtime_version=%d\n", rt);
    if (hipDriverGetVersion(&drv) == hipSuccess) printf("hip_driver_version=%d\n", drv);

    // Scope: first two devices (the dual-R9700 reference rig).
    const int devs[2] = {0, 1};

    // Peer capability map, both directions. Never enable unless cap says yes.
    DirResult dirs[2];
    int pairs[2][2] = {{0, 1}, {1, 0}};
    for (int k = 0; k < 2; ++k) {
        dirs[k].src = pairs[k][0];
        dirs[k].dst = pairs[k][1];
        int can = 0;
        CHECK(hipDeviceCanAccessPeer(&can, dirs[k].src, dirs[k].dst));
        dirs[k].peer_cap = (can != 0);
        printf("peer_capability %d->%d: %s\n", dirs[k].src, dirs[k].dst,
               dirs[k].peer_cap ? "yes" : "no");
    }
    for (int k = 0; k < 2; ++k) {
        if (!dirs[k].peer_cap) continue;  // fail-open enable is forbidden
        CHECK(hipSetDevice(dirs[k].src));
        hipError_t e = hipDeviceEnablePeerAccess(dirs[k].dst, 0);
        if (e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) {
            dirs[k].peer_enabled = true;
        } else {
            printf("FAIL-CLOSED: peer_enable %d->%d failed: %d (%s)\n",
                   dirs[k].src, dirs[k].dst, (int)e, hipGetErrorString(e));
            return 1;
        }
        printf("peer_enabled %d->%d: %s\n", dirs[k].src, dirs[k].dst,
               dirs[k].peer_enabled ? "yes" : "no");
    }

    // Buffers: one device buffer and one device-owned stream per GPU, plus a
    // pinned host bounce buffer. A HIP stream is device-specific; using the
    // source stream for D2H and destination stream for H2D keeps the staged
    // path explicit and prevents the runtime from substituting a peer copy.
    void* dbuf[2] = {nullptr, nullptr};
    hipStream_t streams[2] = {nullptr, nullptr};
    for (int k = 0; k < 2; ++k) {
        CHECK(hipSetDevice(devs[k]));
        CHECK(hipMalloc(&dbuf[k], kBytes));
        CHECK(hipStreamCreate(&streams[k]));
    }
    void* stage = nullptr;
    CHECK(hipHostMalloc(&stage, kBytes, hipHostMallocDefault));

    bool all_content_ok = true;
    for (int k = 0; k < 2; ++k) {
        int src = dirs[k].src, dst = dirs[k].dst;
        uint32_t seed = 0x1000u + (uint32_t)k;
        std::vector<uint8_t> host_src(kBytes), host_check(kBytes);

        // Init src device buffer with deterministic pattern via staged H2D.
        fill_pattern(host_src.data(), kBytes, seed);
        CHECK(hipSetDevice(src));
        CHECK(hipMemcpy(dbuf[src], host_src.data(), kBytes, hipMemcpyHostToDevice));

        // --- Direct path: genuine hipMemcpyPeerAsync only. ---
        if (dirs[k].peer_cap && dirs[k].peer_enabled) {
            // Opposite pattern in dst first so a no-op copy cannot pass verify.
            std::vector<uint8_t> poison(kBytes, 0xA5);
            CHECK(hipSetDevice(dst));
            CHECK(hipMemcpy(dbuf[dst], poison.data(), kBytes, hipMemcpyHostToDevice));
            CHECK(hipSetDevice(src));
            for (int w = 0; w < kWarmups; ++w)
                CHECK(hipMemcpyPeerAsync(dbuf[dst], dst, dbuf[src], src, kBytes,
                                         streams[src]));
            CHECK(hipStreamSynchronize(streams[src]));
            std::vector<double> trials;
            for (int t = 0; t < kTrials; ++t) {
                auto t0 = std::chrono::steady_clock::now();
                CHECK(hipMemcpyPeerAsync(dbuf[dst], dst, dbuf[src], src, kBytes,
                                         streams[src]));
                CHECK(hipStreamSynchronize(streams[src]));
                auto t1 = std::chrono::steady_clock::now();
                trials.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            dirs[k].direct_ms = median_ms(trials);
            dirs[k].direct_gbps = gbps((double)kBytes, dirs[k].direct_ms);
            printf("direct_raw_ms %d->%d:", src, dst);
            for (double v : trials) printf(" %.4f", v);
            printf("\n");
            CHECK(hipSetDevice(dst));
            CHECK(hipMemcpy(host_check.data(), dbuf[dst], kBytes, hipMemcpyDeviceToHost));
            dirs[k].direct_ok = verify_pattern(host_check.data(), kBytes, seed, "direct");
            all_content_ok &= dirs[k].direct_ok;
            printf("direct %d->%d: median %.4f ms, %.4f GB/s(decimal), content %s\n", src, dst,
                   dirs[k].direct_ms, dirs[k].direct_gbps,
                   dirs[k].direct_ok ? "OK" : "MISMATCH");
            // Restore src content (direct overwrote nothing on src, but re-init
            // anyway so the staged leg starts from a known state).
            CHECK(hipSetDevice(src));
            CHECK(hipMemcpy(dbuf[src], host_src.data(), kBytes, hipMemcpyHostToDevice));
        } else {
            printf("direct %d->%d: N/A (peer capability %s)\n", src, dst,
                   dirs[k].peer_cap ? "yes-but-enable-failed" : "no");
        }

        // --- Host-staged path (always): D2H completion, then H2D. ---
        {
            std::vector<uint8_t> poison(kBytes, 0x5A);
            CHECK(hipSetDevice(dst));
            CHECK(hipMemcpy(dbuf[dst], poison.data(), kBytes, hipMemcpyHostToDevice));
            CHECK(hipSetDevice(src));
            for (int w = 0; w < kWarmups; ++w) {
                CHECK(hipMemcpyAsync(stage, dbuf[src], kBytes, hipMemcpyDeviceToHost,
                                     streams[src]));
                CHECK(hipStreamSynchronize(streams[src]));  // D2H completion first
                CHECK(hipSetDevice(dst));
                CHECK(hipMemcpyAsync(dbuf[dst], stage, kBytes, hipMemcpyHostToDevice,
                                     streams[dst]));
                CHECK(hipStreamSynchronize(streams[dst]));
                CHECK(hipSetDevice(src));
            }
            std::vector<double> trials;
            for (int t = 0; t < kTrials; ++t) {
                // Source is current before the timed interval. The interval
                // includes both completed DMA legs and the inter-device handoff.
                auto t0 = std::chrono::steady_clock::now();
                CHECK(hipMemcpyAsync(stage, dbuf[src], kBytes, hipMemcpyDeviceToHost,
                                     streams[src]));
                CHECK(hipStreamSynchronize(streams[src]));
                CHECK(hipSetDevice(dst));
                CHECK(hipMemcpyAsync(dbuf[dst], stage, kBytes, hipMemcpyHostToDevice,
                                     streams[dst]));
                CHECK(hipStreamSynchronize(streams[dst]));
                auto t1 = std::chrono::steady_clock::now();
                trials.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
                CHECK(hipSetDevice(src));
            }
            dirs[k].staged_ms = median_ms(trials);
            dirs[k].staged_gbps = gbps((double)kBytes, dirs[k].staged_ms);
            printf("staged_raw_ms %d->%d:", src, dst);
            for (double v : trials) printf(" %.4f", v);
            printf("\n");
            CHECK(hipSetDevice(dst));
            CHECK(hipMemcpy(host_check.data(), dbuf[dst], kBytes, hipMemcpyDeviceToHost));
            dirs[k].staged_ok = verify_pattern(host_check.data(), kBytes, seed, "staged");
            all_content_ok &= dirs[k].staged_ok;
            printf("staged %d->%d: median %.4f ms, %.4f GB/s(decimal), content %s\n", src,
                   dst, dirs[k].staged_ms, dirs[k].staged_gbps,
                   dirs[k].staged_ok ? "OK" : "MISMATCH");
        }
    }

    for (int k = 0; k < 2; ++k) {
        CHECK(hipSetDevice(devs[k]));
        CHECK(hipStreamDestroy(streams[k]));
        CHECK(hipFree(dbuf[k]));
    }
    CHECK(hipHostFree(stage));

    // Judgment: diagnostic PASS = peer map established in both directions AND
    // host-staged baseline measured with exact content in both directions.
    // Unavailable direct P2P is an expected valid result (roadmap risk text).
    printf("--- summary ---\n");
    bool staged_ok = true;
    for (int k = 0; k < 2; ++k) {
        printf("leg %d->%d peer_cap=%s direct=%s staged=%.4fms/%.4fGB/s content=%s/%s\n",
               dirs[k].src, dirs[k].dst, dirs[k].peer_cap ? "yes" : "no",
               (dirs[k].direct_ms >= 0 ? "measured" : "N/A"), dirs[k].staged_ms,
               dirs[k].staged_gbps, dirs[k].direct_ms >= 0
                                            ? (dirs[k].direct_ok ? "OK" : "MISMATCH")
                                            : "n/a",
               dirs[k].staged_ok ? "OK" : "MISMATCH");
        staged_ok &= (dirs[k].staged_ms >= 0 && dirs[k].staged_ok);
    }
    const char* transport = "HostStaged";  // direct only if measured available+correct
    bool direct_usable = true;
    for (int k = 0; k < 2; ++k)
        direct_usable &= (dirs[k].direct_ms >= 0 && dirs[k].direct_ok);
    if (direct_usable) transport = "Direct";
    printf("selected_transport=%s\n", transport);
    // Both peer-map queries are already established: CHECK returns before
    // this judgment on any query failure.
    bool pass = staged_ok && all_content_ok;
    printf("A0.S6 %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
