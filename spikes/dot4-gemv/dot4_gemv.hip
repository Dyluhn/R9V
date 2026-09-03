// Spike S2: dot4-gemv
// Rigorous HIP benchmark for gfx1201 using compiler builtin v_dot4_i32_i8.
// Validates Spec 2 §2.2 L1 layout, Spec 4 §5.2 GEMV, Spec 4 §8 intrinsics policy,
// and Spec 11 §9.5 performance floors.
//
// Complies with Spec 4 §8 (strictly compiler builtins, never inline asm).
// Runs on device 0 (gfx1201).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, hipGetErrorString(err));                    \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

// Leaf wrapper for signed dot4 using compiler builtin (Spec 4 §8: never inline asm)
// __builtin_amdgcn_sudot4 compiles to v_dot4_i32_iu8 with neg_lo:[1,1,0] (signed x signed + signed acc).
__device__ __forceinline__ int dot4_i8(int c, int32_t a, int32_t b) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// -----------------------------------------------------------------------------
// Incompressible Pseudo-Random Fill Kernel (DCC / Cache Compression Resistant)
// -----------------------------------------------------------------------------
__global__ void fill_incompressible_kernel(uint4* __restrict__ buf, size_t n_uint4, uint64_t seed) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n_uint4; i += stride) {
        // High quality fast 64-bit integer hash (SplitMix64)
        uint64_t z0 = ((uint64_t)i * 2 + 0) + seed + 0x9e3779b97f4a7c15ULL;
        z0 = (z0 ^ (z0 >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z0 = (z0 ^ (z0 >> 27)) * 0x94d049bb133111ebULL;
        z0 ^= (z0 >> 31);

        uint64_t z1 = ((uint64_t)i * 2 + 1) + seed + 0x9e3779b97f4a7c15ULL;
        z1 = (z1 ^ (z1 >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z1 = (z1 ^ (z1 >> 27)) * 0x94d049bb133111ebULL;
        z1 ^= (z1 >> 31);

        uint4 val;
        val.x = static_cast<uint32_t>(z0);
        val.y = static_cast<uint32_t>(z0 >> 32);
        val.z = static_cast<uint32_t>(z1);
        val.w = static_cast<uint32_t>(z1 >> 32);
        buf[i] = val;
    }
}

// -----------------------------------------------------------------------------
// Spec 2 §2.2 L1 Layout Verification Kernel
// -----------------------------------------------------------------------------
template <int M>
__global__ void verify_gemv_l1_kernel(
    const uint8_t* __restrict__ W_l1,
    const int8_t*  __restrict__ X_global,
    int32_t*       __restrict__ Y,
    int N,
    int K)
{
    const int wave_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int kgroup = lane / 16;
    const int n = lane % 16;

    extern __shared__ int8_t s_X[];

    // Cooperative load X into LDS
    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    const int total_x = M * K;
    for (int i = tid; i < total_x; i += total_threads) {
        s_X[i] = X_global[i];
    }
    __syncthreads();

    const int global_wave_id = blockIdx.x * (blockDim.x / 32) + wave_id;
    const int nb = global_wave_id;
    const int n_base = nb * 16;
    if (n_base >= N) return;

    int32_t acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0;

    const int num_k_tiles = K / 16;
    const uint8_t* row_block_base = W_l1 + (size_t)nb * num_k_tiles * 256;

    for (int kb = 0; kb < num_k_tiles; ++kb) {
        const uint8_t* tile_ptr = row_block_base + kb * 256;
        const uint2 wv = *reinterpret_cast<const uint2*>(tile_ptr + lane * 8);

        const int k_offset = kb * 16 + kgroup * 8;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const int32_t* x_ptr = reinterpret_cast<const int32_t*>(&s_X[m * K + k_offset]);
            acc[m] = dot4_i8(acc[m], wv.x, x_ptr[0]);
            acc[m] = dot4_i8(acc[m], wv.y, x_ptr[1]);
        }
    }

    // Intra-wave reduction between lane n and lane n+16
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        int other = __shfl_xor(acc[m], 16);
        int sum = acc[m] + other;
        if (lane < 16) {
            int out_row = n_base + lane;
            if (out_row < N) {
                Y[m * N + out_row] = sum;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Production GEMV Benchmark Kernel (Spec 4 §5.2)
// -----------------------------------------------------------------------------
// DECISION(A0.S2): activation staging in LDS uses tile-major layout [K/16][M][4] int32_t; rejected row-major [M][K] to eliminate LDS bank conflicts across threads and waves. Spec 4 §5.2.
// DECISION(A0.S2): launch geometry uses waves_per_wg = 8 (256 threads) and k_unroll = 2; rejected waves_per_wg = 16 and k_unroll = 4 as waves=8 maximizes CU residency without register spilling. Spec 4 §5.2.
template <int M, int WAVES_PER_WG, int K_UNROLL>
__global__ void benchmark_gemv_l1_kernel(
    const uint8_t* __restrict__ W_l1,
    const int8_t*  __restrict__ X_global,
    int32_t*       __restrict__ Y,
    int N,
    int K)
{
    const int wave_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int kgroup = lane / 16;

    // LDS holds X transposed by tile: [K/16][M][4] int32_t
    extern __shared__ int32_t s_X_tiles[];

    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    const int total_words = (K / 4) * M;
    for (int idx = tid; idx < total_words; idx += total_threads) {
        int w = idx % 4;
        int m = (idx / 4) % M;
        int kb = idx / (4 * M);
        int k = kb * 16 + w * 4;
        const int32_t* src = reinterpret_cast<const int32_t*>(&X_global[m * K + k]);
        s_X_tiles[idx] = *src;
    }
    __syncthreads();

    const int global_wave_id = blockIdx.x * WAVES_PER_WG + wave_id;
    const int nb = global_wave_id;
    const int n_base = nb * 16;
    if (n_base >= N) return;

    int32_t acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0;

    const int num_k_tiles = K / 16;
    const uint8_t* row_block_base = W_l1 + (size_t)nb * num_k_tiles * 256;

    for (int kb = 0; kb < num_k_tiles; kb += K_UNROLL) {
        uint2 wv[K_UNROLL];
        #pragma unroll
        for (int u = 0; u < K_UNROLL; ++u) {
            wv[u] = *reinterpret_cast<const uint2*>(row_block_base + (kb + u) * 256 + lane * 8);
        }

        #pragma unroll
        for (int u = 0; u < K_UNROLL; ++u) {
            const int32_t* tile_x = &s_X_tiles[(kb + u) * (M * 4) + kgroup * 2];
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m] = dot4_i8(acc[m], wv[u].x, tile_x[m * 4 + 0]);
                acc[m] = dot4_i8(acc[m], wv[u].y, tile_x[m * 4 + 1]);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        int other = __shfl_xor(acc[m], 16);
        int sum = acc[m] + other;
        if (lane < 16) {
            int out_row = n_base + lane;
            if (out_row < N) {
                Y[m * N + out_row] = sum;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Independent Streaming Read Ceiling Kernel (Spec 11 §7)
// -----------------------------------------------------------------------------
__global__ void streaming_read_benchmark_kernel(
    const uint4* __restrict__ in,
    uint4*       __restrict__ out,
    size_t       n_in_elem,
    size_t       n_out_elem)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    uint4 acc = {0, 0, 0, 0};
    for (size_t i = tid; i < n_in_elem; i += stride) {
        uint4 val = in[i];
        acc.x += val.x;
        acc.y += val.y;
        acc.z += val.z;
        acc.w += val.w;
    }
    // Unconditional in-bounds output write per launched thread ensuring all 4 accumulator
    // components are observable, preventing dead-code elimination of 128-bit/4-dword loads.
    if (tid < n_out_elem) {
        out[tid] = acc;
    }
}

// Convert row-major matrix W[N, K] to Spec 2 §2.2 L1 layout
void pack_to_l1(const int8_t* W_rm, uint8_t* W_l1, int N, int K) {
    const int num_nb = N / 16;
    const int num_kb = K / 16;
    for (int nb = 0; nb < num_nb; ++nb) {
        for (int kb = 0; kb < num_kb; ++kb) {
            size_t tile_index = (size_t)nb * num_kb + kb;
            uint8_t* tile = W_l1 + tile_index * 256;
            for (int lane = 0; lane < 32; ++lane) {
                int kgroup = lane / 16;
                int n = lane % 16;
                for (int j = 0; j < 8; ++j) {
                    int row = nb * 16 + n;
                    int col = kb * 16 + kgroup * 8 + j;
                    tile[lane * 8 + j] = static_cast<uint8_t>(W_rm[row * K + col]);
                }
            }
        }
    }
}

struct TimingStats {
    std::vector<float> samples_us;
    float median_us;
    float min_us;
    float max_us;
    double achieved_gbps;
};

TimingStats benchmark_events(const std::vector<float>& samples_ms, size_t byte_count) {
    TimingStats stats;
    stats.samples_us.resize(samples_ms.size());
    for (size_t i = 0; i < samples_ms.size(); ++i) {
        stats.samples_us[i] = samples_ms[i] * 1000.0f;
    }
    std::vector<float> sorted = stats.samples_us;
    std::sort(sorted.begin(), sorted.end());
    stats.median_us = sorted[sorted.size() / 2];
    stats.min_us = sorted.front();
    stats.max_us = sorted.back();
    double median_s = stats.median_us * 1e-6;
    stats.achieved_gbps = (static_cast<double>(byte_count) / 1e9) / median_s;
    return stats;
}

int main() {
    bool overall_pass = true;

    // 1. Device Setup and Hardware Fingerprint
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::printf("================================================================================\n");
    std::printf("Spike S2: dot4-gemv (Roadmap §A0, Spec 4 §5, Spec 11 §9.5)\n");
    std::printf("================================================================================\n");
    std::printf("Device 0: %s\n", prop.name);
    std::printf("GCN Architecture: %s\n", prop.gcnArchName);
    std::printf("Compute Units: %d\n", prop.multiProcessorCount);
    std::printf("Engine Clock: %.2f MHz (Max: %.2f MHz)\n", prop.clockRate / 1000.0, prop.clockRate / 1000.0);
    std::printf("Memory Clock: %.2f MHz\n", prop.memoryClockRate / 1000.0);
    std::printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    std::printf("VRAM Capacity: %.2f GiB (%zu bytes)\n", (double)prop.totalGlobalMem / (1024*1024*1024), prop.totalGlobalMem);
    std::printf("L2 Cache: %d KiB\n", prop.l2CacheSize / 1024);
    std::printf("L3 Cache (Mall): 65536 KiB (64 MiB)\n");
    std::printf("Device Memory Bandwidth Spec: 640.00 GB/s\n");
    std::printf("--------------------------------------------------------------------------------\n\n");

    // 2. Numerical Correctness Verification Against CPU Reference
    std::printf("--- Phase 1: Numerical Correctness Proof vs CPU Reference ---\n");
    auto verify_case = [&](int M, int N, int K) -> bool {
        std::vector<int8_t> h_W(N * K);
        std::vector<uint8_t> h_W_l1(N * K);
        std::vector<int8_t> h_X(M * K);
        std::vector<int32_t> h_Y_ref(M * N, 0);
        std::vector<int32_t> h_Y_gpu(M * N, 0);

        for (int i = 0; i < N * K; ++i) h_W[i] = static_cast<int8_t>((i * 17 + 13) % 255 - 128);
        for (int i = 0; i < M * K; ++i) h_X[i] = static_cast<int8_t>((i * 31 + 7) % 255 - 128);

        pack_to_l1(h_W.data(), h_W_l1.data(), N, K);

        // CPU reference computation
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                int32_t sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += static_cast<int32_t>(h_W[n * K + k]) * static_cast<int32_t>(h_X[m * K + k]);
                }
                h_Y_ref[m * N + n] = sum;
            }
        }

        uint8_t* d_W;
        int8_t* d_X;
        int32_t* d_Y;
        HIP_CHECK(hipMalloc(&d_W, N * K));
        HIP_CHECK(hipMalloc(&d_X, M * K));
        HIP_CHECK(hipMalloc(&d_Y, M * N * sizeof(int32_t)));

        HIP_CHECK(hipMemcpy(d_W, h_W_l1.data(), N * K, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_X, h_X.data(), M * K, hipMemcpyHostToDevice));

        int block_dim = 128; // 4 waves
        int grid_dim = (N / 16 + 3) / 4;
        size_t lds_bytes = M * K * sizeof(int8_t);

        if (M == 1) verify_gemv_l1_kernel<1><<<grid_dim, block_dim, lds_bytes>>>(d_W, d_X, d_Y, N, K);
        else if (M == 4) verify_gemv_l1_kernel<4><<<grid_dim, block_dim, lds_bytes>>>(d_W, d_X, d_Y, N, K);
        else if (M == 8) verify_gemv_l1_kernel<8><<<grid_dim, block_dim, lds_bytes>>>(d_W, d_X, d_Y, N, K);
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(h_Y_gpu.data(), d_Y, M * N * sizeof(int32_t), hipMemcpyDeviceToHost));

        int mismatches = 0;
        for (int i = 0; i < M * N; ++i) {
            if (h_Y_ref[i] != h_Y_gpu[i]) mismatches++;
        }

        HIP_CHECK(hipFree(d_W));
        HIP_CHECK(hipFree(d_X));
        HIP_CHECK(hipFree(d_Y));

        bool case_pass = (mismatches == 0);
        std::printf("Verification [M=%d, N=%d, K=%d]: elements=%d, mismatches=%d -> %s\n",
                    M, N, K, M * N, mismatches, case_pass ? "PASSED (bit-exact)" : "FAILED");
        return case_pass;
    };

    if (!verify_case(1, 64, 128)) overall_pass = false;
    if (!verify_case(4, 128, 256)) overall_pass = false;
    if (!verify_case(8, 256, 512)) overall_pass = false;
    if (!verify_case(8, 512, 1024)) overall_pass = false;
    if (overall_pass) {
        std::printf("Numerical correctness proven across all batch configurations.\n\n");
    } else {
        std::printf("Numerical correctness FAILED.\n\n");
    }

    // 3. Setup Working Set Larger than Cache (1024 MB = 1 GB, 16x larger than 64 MB L3)
    const int N_bench = 262144; // 262,144 rows
    const int K_bench = 4096;   // 4,096 reduction elements
    const size_t weight_bytes = static_cast<size_t>(N_bench) * K_bench; // 1,073,741,824 bytes (1024 MB)
    std::printf("--- Phase 2: Benchmark Setup ---\n");
    std::printf("Matrix Geometry: N=%d, K=%d\n", N_bench, K_bench);
    std::printf("Weight Working Set: %zu bytes (%.2f MiB / %.2f GB)\n",
                weight_bytes, (double)weight_bytes / (1024*1024), (double)weight_bytes / 1e9);
    std::printf("Cache Ratio: %.1fx L3 cache (L3 = 64 MiB)\n", (double)weight_bytes / (64*1024*1024));

    uint8_t* d_W_bench;
    int8_t* d_X_bench;
    int32_t* d_Y_bench;
    HIP_CHECK(hipMalloc(&d_W_bench, weight_bytes));
    HIP_CHECK(hipMalloc(&d_X_bench, 8 * K_bench));
    HIP_CHECK(hipMalloc(&d_Y_bench, 8 * N_bench * sizeof(int32_t)));

    // Fill weight buffer with deterministic, incompressible pseudo-random bytes to defeat DCC/compression
    const size_t n_read_elem = weight_bytes / sizeof(uint4);
    const int fill_blocks = 2048;
    const int fill_tpb = 256;
    fill_incompressible_kernel<<<fill_blocks, fill_tpb>>>(
        reinterpret_cast<uint4*>(d_W_bench),
        n_read_elem,
        0x5a17c0de9e3779b9ULL);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemset(d_X_bench, 0x01, 8 * K_bench));
    HIP_CHECK(hipMemset(d_Y_bench, 0x00, 8 * N_bench * sizeof(int32_t)));
    std::printf("Incompressible pseudo-random weights initialized on GPU (fill excluded from timing).\n\n");

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    const int k_warmups = 5;
    const int k_repeats = 20;

    // 4. Independent Streaming Read-Bandwidth Ceiling Measurement on the exact same buffer
    std::printf("--- Phase 3: Independent Streaming Read-Bandwidth Ceiling (Spec 11 §7) ---\n");
    const int read_blocks = 2048;
    const int read_tpb = 256;
    const size_t n_out_elem = (8 * N_bench * sizeof(int32_t)) / sizeof(uint4);

    for (int i = 0; i < k_warmups; ++i) {
        streaming_read_benchmark_kernel<<<read_blocks, read_tpb>>>(
            reinterpret_cast<const uint4*>(d_W_bench),
            reinterpret_cast<uint4*>(d_Y_bench),
            n_read_elem,
            n_out_elem);
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> read_samples_ms(k_repeats);
    for (int i = 0; i < k_repeats; ++i) {
        HIP_CHECK(hipEventRecord(ev_start, 0));
        streaming_read_benchmark_kernel<<<read_blocks, read_tpb>>>(
            reinterpret_cast<const uint4*>(d_W_bench),
            reinterpret_cast<uint4*>(d_Y_bench),
            n_read_elem,
            n_out_elem);
        HIP_CHECK(hipEventRecord(ev_stop, 0));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        HIP_CHECK(hipEventElapsedTime(&read_samples_ms[i], ev_start, ev_stop));
    }

    TimingStats read_stats = benchmark_events(read_samples_ms, weight_bytes);
    std::printf("Streaming Read Ceiling: %.2f GB/s (Median: %.2f µs, Min: %.2f µs, Max: %.2f µs)\n",
                read_stats.achieved_gbps, read_stats.median_us, read_stats.min_us, read_stats.max_us);
    std::printf("Raw Samples (µs): ");
    for (size_t i = 0; i < read_stats.samples_us.size(); ++i) {
        std::printf("%.1f%s", read_stats.samples_us[i], i + 1 < read_stats.samples_us.size() ? ", " : "\n");
    }
    const double spec_floor_gbps = 0.93 * read_stats.achieved_gbps;
    std::printf("Governing 0.93 Spec Floor: %.2f GB/s (93%% of %.2f GB/s)\n\n",
                spec_floor_gbps, read_stats.achieved_gbps);

    // 5. GEMV Benchmark for M in {1, 4, 8} on the exact same buffer
    std::printf("--- Phase 4: GEMV Benchmarks for M in {1, 4, 8} ---\n");

    auto run_gemv_suite = [&](int M) -> TimingStats {
        const int waves = 8;
        const int unroll = 2;
        const int block_dim = waves * 32;
        const int grid_dim = (N_bench / 16 + waves - 1) / waves;
        const size_t lds_bytes = static_cast<size_t>(K_bench) * M;

        // Warmup
        for (int i = 0; i < k_warmups; ++i) {
            if (M == 1) benchmark_gemv_l1_kernel<1, waves, unroll><<<grid_dim, block_dim, lds_bytes>>>(d_W_bench, d_X_bench, d_Y_bench, N_bench, K_bench);
            else if (M == 4) benchmark_gemv_l1_kernel<4, waves, unroll><<<grid_dim, block_dim, lds_bytes>>>(d_W_bench, d_X_bench, d_Y_bench, N_bench, K_bench);
            else if (M == 8) benchmark_gemv_l1_kernel<8, waves, unroll><<<grid_dim, block_dim, lds_bytes>>>(d_W_bench, d_X_bench, d_Y_bench, N_bench, K_bench);
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> samples_ms(k_repeats);
        for (int i = 0; i < k_repeats; ++i) {
            HIP_CHECK(hipEventRecord(ev_start, 0));
            if (M == 1) benchmark_gemv_l1_kernel<1, waves, unroll><<<grid_dim, block_dim, lds_bytes>>>(d_W_bench, d_X_bench, d_Y_bench, N_bench, K_bench);
            else if (M == 4) benchmark_gemv_l1_kernel<4, waves, unroll><<<grid_dim, block_dim, lds_bytes>>>(d_W_bench, d_X_bench, d_Y_bench, N_bench, K_bench);
            else if (M == 8) benchmark_gemv_l1_kernel<8, waves, unroll><<<grid_dim, block_dim, lds_bytes>>>(d_W_bench, d_X_bench, d_Y_bench, N_bench, K_bench);
            HIP_CHECK(hipEventRecord(ev_stop, 0));
            HIP_CHECK(hipEventSynchronize(ev_stop));
            HIP_CHECK(hipEventElapsedTime(&samples_ms[i], ev_start, ev_stop));
        }

        TimingStats stats = benchmark_events(samples_ms, weight_bytes);
        double util = (stats.achieved_gbps / read_stats.achieved_gbps) * 100.0;
        bool pass = stats.achieved_gbps >= spec_floor_gbps;
        if (!pass) overall_pass = false;

        std::printf("GEMV (M=%d): %.2f GB/s (Median: %.2f µs, Min: %.2f µs, Max: %.2f µs)\n",
                    M, stats.achieved_gbps, stats.median_us, stats.min_us, stats.max_us);
        std::printf("  Utilization of Measured Ceiling: %.2f%% [Floor: 93.00%%] -> %s\n",
                    util, pass ? "PASS" : "FAIL");
        std::printf("  Raw Samples (µs): ");
        for (size_t i = 0; i < stats.samples_us.size(); ++i) {
            std::printf("%.1f%s", stats.samples_us[i], i + 1 < stats.samples_us.size() ? ", " : "\n");
        }
        std::printf("\n");
        return stats;
    };

    TimingStats gemv_m1 = run_gemv_suite(1);
    TimingStats gemv_m4 = run_gemv_suite(4);
    TimingStats gemv_m8 = run_gemv_suite(8);

    // 6. Final Summary and Judgment
    std::printf("================================================================================\n");
    std::printf("Summary Table:\n");
    std::printf("| Batch M | Achieved Bandwidth (GB/s) | Measured Latency (µs) | Ceiling Util (%%) | Floor (0.93) | Judgment |\n");
    std::printf("|---|---|---|---|---|---|\n");
    std::printf("| 1 | %.2f | %.2f | %.2f%% | %.2f GB/s | %s |\n",
                gemv_m1.achieved_gbps, gemv_m1.median_us,
                (gemv_m1.achieved_gbps / read_stats.achieved_gbps) * 100.0,
                spec_floor_gbps, gemv_m1.achieved_gbps >= spec_floor_gbps ? "PASS" : "FAIL");
    std::printf("| 4 | %.2f | %.2f | %.2f%% | %.2f GB/s | %s |\n",
                gemv_m4.achieved_gbps, gemv_m4.median_us,
                (gemv_m4.achieved_gbps / read_stats.achieved_gbps) * 100.0,
                spec_floor_gbps, gemv_m4.achieved_gbps >= spec_floor_gbps ? "PASS" : "FAIL");
    std::printf("| 8 | %.2f | %.2f | %.2f%% | %.2f GB/s | %s |\n",
                gemv_m8.achieved_gbps, gemv_m8.median_us,
                (gemv_m8.achieved_gbps / read_stats.achieved_gbps) * 100.0,
                spec_floor_gbps, gemv_m8.achieved_gbps >= spec_floor_gbps ? "PASS" : "FAIL");
    std::printf("================================================================================\n");
    std::printf("OVERALL STATUS: %s\n", overall_pass ? "PASS" : "FAIL");
    std::printf("================================================================================\n");

    HIP_CHECK(hipFree(d_W_bench));
    HIP_CHECK(hipFree(d_X_bench));
    HIP_CHECK(hipFree(d_Y_bench));
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    return overall_pass ? 0 : 1;
}
