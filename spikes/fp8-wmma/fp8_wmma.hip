// Spike S3: fp8-wmma
// Tests the pinned-ROCm compiler FP8 WMMA builtin on gfx1201 (device 0) and
// decides whether the Spec 4 §8 leaf wrapper (`wmma_fp8`) needs inline asm.
//
// Claim (Roadmap §A0, Spec 4 §8): the fp8 builtins compile and run correctly
// on the pinned ROCm; the leaf wrapper uses builtins, no inline asm, unless a
// miscompile is demonstrated.
//
// Policy: this file uses exactly one arch instruction, reached ONLY through
// the pinned compiler builtin
//   __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
// There is no inline asm anywhere in this file (no `asm` keyword at all).
//
// What the program proves, and how:
//  - T1/T2: single 16x16x16 tile, lane-indexed exact patterns. Any
//    operand-lane or fragment-layout mistake changes outputs by O(1) against
//    a tight tolerance, so these catch layout errors.
//  - T3: E4M3 specials (max 448, min normal, subnormals incl. min 2^-9,
//    signed zero). Catches clipping/denormal mishandling in the unit.
//  - T4: 64x64x64 multi-tile GEMM (4x4xN tiles over a K loop) with
//    deterministic pseudo-random values. Catches tile-index/loop errors.
//  - T5: determinism (two runs bit-equal), the Spec 4 §10 gate-3 analogue.
//  - T6: register-bound peak microbench. Consumes the builtin in a
//    runtime-trip-count loop so the compiler cannot eliminate it, and records
//    the achieved rate.
// The CPU oracle decodes the same E4M3 bytes and accumulates in f64, so it is
// independent of the device path (different type, different order freedom).
// Any failed check makes the process exit nonzero.
// DECISION(A0.S3): Compile with Clang's supported -fno-fast-math spelling to
// preserve Spec 4 §4.3's no-fast-math requirement; the literal
// -ffast-math=off spelling is rejected by the pinned compiler (SI-1).
//
// FP8 encoding used here (OCP E4M3, documented so the oracle is auditable):
//   bit 7: sign; bits 6:3: exponent, bias 7; bits 2:0: mantissa.
//   e == 0:  subnormal (-1)^s * 2^-6 * (m/8); min |value| 2^-9.
//   1..14:   normal    (-1)^s * 2^(e-7) * (1+m/8); min normal 2^-6.
//   e == 15: m == 7 -> NaN; m < 7 -> normal with exponent 2^8 (max 448).
//   So max is 0x7E (448), min normal 0x08 (2^-6), min subnormal 0x01 (2^-9).
//   Encode: nearest-even over all non-NaN codes, saturate to +/-448.
//   (NaN codes 0x7F/0xFF are never generated as inputs.)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = (call);                                           \
        if (err != hipSuccess) {                                           \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,     \
                         __LINE__, hipGetErrorString(err));                \
            std::exit(2);                                                  \
        }                                                                  \
    } while (0)

typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------------------
// E4M3 codec (host oracle side; documented above)
// ---------------------------------------------------------------------------

static float fp8_e4m3_decode(uint8_t b) {
    unsigned s = b >> 7;
    unsigned e = (b >> 3) & 0xFu;
    unsigned m = b & 0x7u;
    float sign = s ? -1.0f : 1.0f;
    if (e == 0) {
        return sign * ldexpf((float)m / 8.0f, -6);  // subnormal
    }
    if (e == 15 && m == 7) {
        return sign * NAN;  // only NaN codes; never used as test input
    }
    return sign * ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
}

// Nearest-even encode over the 254 non-NaN codes; saturate beyond +/-448.
static uint8_t fp8_e4m3_encode(float v) {
    if (v != v) return 0x7Fu;  // NaN in -> NaN code (not used by tests)
    float av = std::fabs(v);
    if (av > 448.0f) return v < 0.0f ? 0xFEu : 0x7Eu;
    if (av == 0.0f) return std::signbit(v) ? 0x80u : 0x00u;
    uint8_t best = 0x00u;
    float best_d = 1e30f;
    for (int c = 0; c < 256; ++c) {
        if (c == 0x7F || c == 0xFF) continue;  // NaN codes excluded
        float d = std::fabs(fp8_e4m3_decode((uint8_t)c) - v);
        if (d < best_d || (d == best_d && (c & 1) == 0 && (best & 1) == 1)) {
            best_d = d;
            best = (uint8_t)c;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Deterministic host PRNG (LCG, fixed seed; no library RNG, fully
// reproducible). Documented seed: 0xC0FFEE.
// ---------------------------------------------------------------------------

static uint32_t lcg_state = 0xC0FFEEu;
static uint32_t lcg_next() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 8;
}

// ---------------------------------------------------------------------------
// Device kernels. One arch instruction, via builtin only.
// Lane mapping (same native WMMA 16x16x16 map the S1 spike verified for iu8;
// T1/T2 below re-verify it for fp8): lane l holds 8 K-consecutive elements of
// row (l % 16) starting at k-group (l / 16) * 8; accumulator lane l holds
// C[(l / 16) * 8 + v][l % 16], v = 0..7.
// ---------------------------------------------------------------------------

__global__ void fp8_single_tile_kernel(const i32x2* __restrict__ d_a,
                                       const i32x2* __restrict__ d_b,
                                       float* __restrict__ d_c) {
    const unsigned lane = threadIdx.x;  // one wave of 32
    const i32x2 a_frag = d_a[lane];
    const i32x2 b_frag = d_b[lane];
    f32x8 acc;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_frag, b_frag,
                                                              acc);
#pragma unroll
    for (int v = 0; v < 8; ++v) d_c[lane * 8 + v] = acc[v];
}

// 64x64x64 GEMM over a K-tile loop. B pre-packed tile-major in L1 lane order:
// tile_index = tile_n * k_tiles + kt, 32 lanes per tile.
__global__ void fp8_gemm_multitile_kernel(const unsigned char* __restrict__ A,
                                          const i32x2* __restrict__ B_l1,
                                          float* __restrict__ C, int M, int N,
                                          int K) {
    const int tile_n = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int lane = threadIdx.x;
    const int n_base = tile_n * 16;
    const int m_base = tile_m * 16;
    f32x8 acc;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    const int k_tiles = K / 16;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k_base = kt * 16;
        const i32x2 b_frag = B_l1[(tile_n * k_tiles + kt) * 32 + lane];
        const int a_row = m_base + (lane % 16);
        const int a_k = k_base + (lane / 16) * 8;
        i32x2 a_frag;
        unsigned char* a_bytes = reinterpret_cast<unsigned char*>(&a_frag);
#pragma unroll
        for (int j = 0; j < 8; ++j) a_bytes[j] = A[a_row * K + a_k + j];
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_frag,
                                                                  b_frag, acc);
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int c_row = m_base + 8 * (lane / 16) + v;
        const int c_col = n_base + (lane % 16);
        if (c_row < M && c_col < N) C[c_row * N + c_col] = acc[v];
    }
}

// Register-bound peak loop. `trips` is a runtime argument so the loop cannot
// be folded; every accumulator is stored to global and checksummed on host.
__global__ __launch_bounds__(256) void bench_fp8_peak(float* out, int seed,
                                                      unsigned trips) {
    i32x2 a, b;
    a[0] = 0x3C3C3C3C ^ seed;  // 0x3C = e4m3(+1.0): valid fp8 operand bytes
    a[1] = 0x3C3C3C3C ^ (seed * 31 + 7);
    b[0] = 0x3C3C3C3C ^ (seed * 17 + 3);
    b[1] = 0x38383838 ^ seed;  // 0x38 = e4m3(+0.5)
    constexpr unsigned kAcc = 8;
    f32x8 acc[kAcc];
#pragma unroll
    for (unsigned k = 0; k < kAcc; ++k)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[k][i] = 0.0f;
    for (unsigned t = 0; t < trips; ++t) {
#pragma unroll
        for (unsigned k = 0; k < kAcc; ++k)
            acc[k] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                a, b, acc[k]);
    }
    float s = 0.0f;
#pragma unroll
    for (unsigned k = 0; k < kAcc; ++k)
#pragma unroll
        for (int i = 0; i < 8; ++i) s += acc[k][i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------

// Pass criterion: |d - r| <= abs_tol + rel_tol * |r| (documented combined
// tolerance). T1/T2 use abs 1e-4 / rel 1e-4 (exact patterns: any lane or
// element permutation moves outputs by O(1)). T3/T4 use abs 1e-2 / rel 1e-3:
// the device accumulates in f32 (16-term dots, K-loop), the oracle in f64,
// so f32 rounding at large magnitude needs the looser bound while any tile
// or lane error still exceeds it by orders of magnitude.
struct Tol {
    double abs_tol;
    double rel_tol;
};

static int check_tile(const float* got, const double* ref, int n, Tol tol,
                      double* max_abs, double* max_rel, int* mism) {
    *max_abs = 0.0;
    *max_rel = 0.0;
    *mism = 0;
    for (int i = 0; i < n; ++i) {
        double d = (double)got[i] - ref[i];
        double ad = std::fabs(d);
        double ar = std::fabs(ref[i]);
        if (ad > *max_abs) *max_abs = ad;
        double rel = ad / (ar > 0.0 ? ar : 1.0);
        if (rel > *max_rel) *max_rel = rel;
        if (ad > tol.abs_tol + tol.rel_tol * ar) (*mism)++;
    }
    return *mism;
}

// Pack one 16x16 tile pair into lane fragments: lane l holds 8 K-consecutive
// bytes of row (l % 16) from k-group (l / 16) * 8.
static void pack_lane_order(const std::vector<uint8_t>& A,
                            const std::vector<uint8_t>& B, uint8_t* a_mem,
                            uint8_t* b_mem) {
    for (int lane = 0; lane < 32; ++lane) {
        int row = lane % 16;
        int k0 = (lane / 16) * 8;
        for (int j = 0; j < 8; ++j) {
            a_mem[lane * 8 + j] = A[row * 16 + k0 + j];
            b_mem[lane * 8 + j] = B[row * 16 + k0 + j];
        }
    }
}

// Reference: C[m][n] = sum_k decode(A[m][k]) * decode(B[n][k]) in f64.
static void ref_gemm(const std::vector<uint8_t>& A,
                     const std::vector<uint8_t>& B, std::vector<double>& C,
                     int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += (double)fp8_e4m3_decode(A[m * K + k]) *
                     (double)fp8_e4m3_decode(B[n * K + k]);
            C[m * N + n] = s;
        }
}

// Unpack accumulator lane order: lane l holds C[(l/16)*8+v][l%16].
static void unpack_acc_order(const std::vector<float>& raw,
                             std::vector<float>& C) {
    for (int lane = 0; lane < 32; ++lane)
        for (int v = 0; v < 8; ++v)
            C[((lane / 16) * 8 + v) * 16 + (lane % 16)] = raw[lane * 8 + v];
}

static uint64_t checksum_f32(const float* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &p[i], 4);
        h ^= u;
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char** argv) {
    unsigned bench_trips = argc > 1 ? (unsigned)std::strtoul(argv[1], 0, 10)
                                    : 20000u;
    if (bench_trips == 0) {
        std::fprintf(stderr, "trips must be > 0\n");
        return 2;
    }
    const int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    char bdf[64] = {0};
    HIP_CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), dev));

    std::printf("================================================================"
                "================\n");
    std::printf("R9V Spike S3: fp8-wmma (Roadmap §A0, Spec 4 §8)\n");
    std::printf("=================================================================="
                "==============\n");
    std::printf("Device %d: %s [%s] BDF %s, CUs %d, clock %.2f MHz, mem %.2f "
                "GiB\n",
                dev, prop.name, prop.gcnArchName, bdf, prop.multiProcessorCount,
                prop.clockRate / 1000.0,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("Builtin: __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 "
                "( Operands i32x2/i32x2, acc f32x8 )\n");
    std::printf("------------------------------------------------------------------"
                "--------------\n");

    int failures = 0;
    if (std::strstr(prop.gcnArchName, "gfx1201") == 0) {
        std::printf("[ARCH] FAILED: expected gfx1201, got %s\n",
                    prop.gcnArchName);
        failures++;
    } else {
        std::printf("[ARCH] gfx1201 confirmed\n");
    }

    uint64_t total_hash = 1469598103934665603ull;
    auto fold = [&](const float* p, size_t n) {
        uint64_t h = checksum_f32(p, n);
        total_hash ^= h + 0x9e3779b97f4a7c15ull;
    };

    // ---- T1: single tile, exact small integers (-7..7 / -6..6) ----
    {
        std::vector<uint8_t> A(256), B(256);
        for (int m = 0; m < 16; ++m)
            for (int k = 0; k < 16; ++k) {
                A[m * 16 + k] =
                    fp8_e4m3_encode((float)(((m * 5 + k * 3) % 15) - 7));
                B[m * 16 + k] =
                    fp8_e4m3_encode((float)(((m * 7 + k * 11) % 13) - 6));
            }
        std::vector<uint8_t> am(256), bm(256);
        pack_lane_order(A, B, am.data(), bm.data());
        i32x2 *d_a, *d_b;
        float* d_c;
        HIP_CHECK(hipMalloc(&d_a, 256));
        HIP_CHECK(hipMalloc(&d_b, 256));
        HIP_CHECK(hipMalloc(&d_c, 256 * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_a, am.data(), 256, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, bm.data(), 256, hipMemcpyHostToDevice));
        fp8_single_tile_kernel<<<1, 32>>>(d_a, d_b, d_c);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> raw(256);
        HIP_CHECK(hipMemcpy(raw.data(), d_c, 256 * sizeof(float),
                            hipMemcpyDeviceToHost));
        std::vector<float> C(256);
        unpack_acc_order(raw, C);
        fold(C.data(), 256);
        std::vector<double> R(256);
        // Reference over [M=16, N=16, K=16] with B stored [N,K].
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 16; ++n) {
                double s = 0.0;
                for (int k = 0; k < 16; ++k)
                    s += (double)fp8_e4m3_decode(A[m * 16 + k]) *
                         (double)fp8_e4m3_decode(B[n * 16 + k]);
                R[m * 16 + n] = s;
            }
        double ma, mr;
        int mm;
        check_tile(C.data(), R.data(), 256, {1e-4, 1e-4}, &ma, &mr, &mm);
        std::printf("[T1] exact-int tile: %s (mism %d/256, max_abs %.3g, "
                    "max_rel %.3g)\n",
                    mm == 0 ? "PASSED" : "FAILED", mm, ma, mr);
        if (mm != 0) failures++;
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
    }

    // ---- T2: single tile, exact fractions (halves x quarters) ----
    {
        std::vector<uint8_t> A(256), B(256);
        for (int m = 0; m < 16; ++m)
            for (int k = 0; k < 16; ++k) {
                A[m * 16 + k] = fp8_e4m3_encode(
                    ((float)(((m * 5 + k * 3) % 17) - 8)) * 0.5f);
                B[m * 16 + k] = fp8_e4m3_encode(
                    ((float)(((m * 7 + k * 11) % 15) - 7)) * 0.25f);
            }
        std::vector<uint8_t> am(256), bm(256);
        pack_lane_order(A, B, am.data(), bm.data());
        i32x2 *d_a, *d_b;
        float* d_c;
        HIP_CHECK(hipMalloc(&d_a, 256));
        HIP_CHECK(hipMalloc(&d_b, 256));
        HIP_CHECK(hipMalloc(&d_c, 256 * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_a, am.data(), 256, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, bm.data(), 256, hipMemcpyHostToDevice));
        fp8_single_tile_kernel<<<1, 32>>>(d_a, d_b, d_c);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> raw(256);
        HIP_CHECK(hipMemcpy(raw.data(), d_c, 256 * sizeof(float),
                            hipMemcpyDeviceToHost));
        std::vector<float> C(256);
        unpack_acc_order(raw, C);
        fold(C.data(), 256);
        std::vector<double> R(256);
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 16; ++n) {
                double s = 0.0;
                for (int k = 0; k < 16; ++k)
                    s += (double)fp8_e4m3_decode(A[m * 16 + k]) *
                         (double)fp8_e4m3_decode(B[n * 16 + k]);
                R[m * 16 + n] = s;
            }
        double ma, mr;
        int mm;
        check_tile(C.data(), R.data(), 256, {1e-4, 1e-4}, &ma, &mr, &mm);
        std::printf("[T2] fraction tile:  %s (mism %d/256, max_abs %.3g, "
                    "max_rel %.3g)\n",
                    mm == 0 ? "PASSED" : "FAILED", mm, ma, mr);
        if (mm != 0) failures++;
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
    }

    // ---- T3: extremes + subnormals + signed zero ----
    {
        const float vals[16] = {448.0f,   -448.0f, 240.0f,  -240.0f,
                                1.0f,     -1.0f,   0.5f,    -0.0f,
                                0.0f,     0.015625f,  // min normal 2^-6
                                0.0078125f,         // 2^-7 subnormal
                                0.001953125f,       // min subnormal 2^-9
                                -0.00390625f, 32.0f, -0.25f, 6.0f};
        std::vector<uint8_t> A(256), B(256);
        for (int m = 0; m < 16; ++m)
            for (int k = 0; k < 16; ++k) {
                A[m * 16 + k] = fp8_e4m3_encode(vals[(m + k) % 16]);
                B[m * 16 + k] = fp8_e4m3_encode(vals[(m * 3 + k * 5) % 16]);
            }
        std::vector<uint8_t> am(256), bm(256);
        pack_lane_order(A, B, am.data(), bm.data());
        i32x2 *d_a, *d_b;
        float* d_c;
        HIP_CHECK(hipMalloc(&d_a, 256));
        HIP_CHECK(hipMalloc(&d_b, 256));
        HIP_CHECK(hipMalloc(&d_c, 256 * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_a, am.data(), 256, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, bm.data(), 256, hipMemcpyHostToDevice));
        fp8_single_tile_kernel<<<1, 32>>>(d_a, d_b, d_c);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> raw(256);
        HIP_CHECK(hipMemcpy(raw.data(), d_c, 256 * sizeof(float),
                            hipMemcpyDeviceToHost));
        std::vector<float> C(256);
        unpack_acc_order(raw, C);
        fold(C.data(), 256);
        std::vector<double> R(256);
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 16; ++n) {
                double s = 0.0;
                for (int k = 0; k < 16; ++k)
                    s += (double)fp8_e4m3_decode(A[m * 16 + k]) *
                         (double)fp8_e4m3_decode(B[n * 16 + k]);
                R[m * 16 + n] = s;
            }
        double ma, mr;
        int mm;
        check_tile(C.data(), R.data(), 256, {1e-2, 1e-3}, &ma, &mr, &mm);
        std::printf("[T3] extreme tile:   %s (mism %d/256, max_abs %.3g, "
                    "max_rel %.3g)\n",
                    mm == 0 ? "PASSED" : "FAILED", mm, ma, mr);
        if (mm != 0) failures++;
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
    }

    // ---- T4: 64x64x64 multi-tile GEMM, deterministic LCG values ----
    // ---- T5: determinism (repeat T4 into a second buffer, bit-compare) ----
    {
        const int M = 64, N = 64, K = 64;
        lcg_state = 0xC0FFEEu;  // reset: T4 inputs reproducible every run
        std::vector<uint8_t> A(M * K), B(N * K);
        for (int i = 0; i < M * K; ++i)
            A[i] = fp8_e4m3_encode(((float)(lcg_next() % 1601) - 800.0f) *
                                   0.125f);
        for (int i = 0; i < N * K; ++i)
            B[i] = fp8_e4m3_encode(((float)(lcg_next() % 1601) - 800.0f) *
                                   0.125f);
        // B packed tile-major in L1 lane order.
        const int k_tiles = K / 16, n_tiles = N / 16;
        std::vector<uint8_t> Bl1(n_tiles * k_tiles * 256);
        for (int tn = 0; tn < n_tiles; ++tn)
            for (int kt = 0; kt < k_tiles; ++kt)
                for (int lane = 0; lane < 32; ++lane) {
                    int n = tn * 16 + (lane % 16);
                    int k0 = kt * 16 + (lane / 16) * 8;
                    for (int j = 0; j < 8; ++j)
                        Bl1[((tn * k_tiles + kt) * 32 + lane) * 8 + j] =
                            B[n * K + k0 + j];
                }
        unsigned char* d_a;
        i32x2* d_b;
        float *d_c, *d_c2;
        HIP_CHECK(hipMalloc(&d_a, (size_t)M * K));
        HIP_CHECK(hipMalloc(&d_b, Bl1.size()));
        HIP_CHECK(hipMalloc(&d_c, (size_t)M * N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_c2, (size_t)M * N * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_a, A.data(), (size_t)M * K,
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, Bl1.data(), Bl1.size(),
                            hipMemcpyHostToDevice));
        dim3 grid(n_tiles, M / 16);
        fp8_gemm_multitile_kernel<<<grid, 32>>>(d_a, d_b, d_c, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());
        fp8_gemm_multitile_kernel<<<grid, 32>>>(d_a, d_b, d_c2, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> C(M * N), C2(M * N);
        HIP_CHECK(hipMemcpy(C.data(), d_c, (size_t)M * N * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(C2.data(), d_c2, (size_t)M * N * sizeof(float),
                            hipMemcpyDeviceToHost));
        fold(C.data(), (size_t)M * N);
        std::vector<double> R(M * N);
        ref_gemm(A, B, R, M, N, K);
        double ma, mr;
        int mm;
        check_tile(C.data(), R.data(), M * N, {1e-2, 1e-3}, &ma, &mr, &mm);
        std::printf("[T4] 64x64x64 GEMM:  %s (mism %d/4096, max_abs %.3g, "
                    "max_rel %.3g)\n",
                    mm == 0 ? "PASSED" : "FAILED", mm, ma, mr);
        if (mm != 0) failures++;
        int det_mism = 0;
        for (int i = 0; i < M * N; ++i) {
            uint32_t u1, u2;
            std::memcpy(&u1, &C[i], 4);
            std::memcpy(&u2, &C2[i], 4);
            if (u1 != u2) det_mism++;
        }
        std::printf("[T5] determinism:    %s (bit-mism %d/4096)\n",
                    det_mism == 0 ? "PASSED" : "FAILED", det_mism);
        if (det_mism != 0) failures++;
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
        HIP_CHECK(hipFree(d_c2));
    }

    // ---- T6: peak microbench (anti-DCE + rate) ----
    {
        const int threads = 256, blocks = 64;  // 512 waves like S1
        const int reps = 7;
        float* d_out;
        HIP_CHECK(hipMalloc(&d_out, (size_t)threads * blocks * sizeof(float)));
        std::vector<float> h_out(threads * blocks);
        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0));
        HIP_CHECK(hipEventCreate(&e1));
        // Warmup.
        bench_fp8_peak<<<blocks, threads>>>(d_out, 1, bench_trips);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> times;
        for (int r = 0; r < reps; ++r) {
            HIP_CHECK(hipEventRecord(e0));
            bench_fp8_peak<<<blocks, threads>>>(d_out, 1234 + r, bench_trips);
            HIP_CHECK(hipEventRecord(e1));
            HIP_CHECK(hipEventSynchronize(e1));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
            times.push_back(ms);
        }
        HIP_CHECK(hipMemcpy(h_out.data(), d_out,
                            (size_t)threads * blocks * sizeof(float),
                            hipMemcpyDeviceToHost));
        fold(h_out.data(), h_out.size());
        std::sort(times.begin(), times.end());
        float med = times[reps / 2];
        // Per-wave accounting (same convention as the S1 spike): the 32
        // lanes of a wave cooperate on one matrix instruction, so
        // instructions = waves * trips * accs, not threads * trips * accs.
        double waves = (double)(threads * blocks) / 32.0;
        double ops = waves * (double)bench_trips * 8.0 * 2.0 * 16 * 16 * 16;
        double tflops = ops / (med / 1000.0) / 1e12;
        double cksum = 0.0;
        for (float v : h_out) cksum += (double)v;
        std::printf("[T6] peak: %.2f TFLOPS (median %.3f ms over %d reps, "
                    "range %.3f-%.3f ms, trips %u, checksum %.6g)\n",
                    tflops, med, reps, times.front(), times.back(),
                    bench_trips, cksum);
        if (!(cksum == cksum) || cksum == 0.0) {
            // NaN or all-zero output means the loop was eliminated or broken.
            std::printf("[T6] FAILED: checksum degenerate (DCE suspected)\n");
            failures++;
        }
        HIP_CHECK(hipEventDestroy(e0));
        HIP_CHECK(hipEventDestroy(e1));
        HIP_CHECK(hipFree(d_out));
    }

    std::printf("------------------------------------------------------------------"
                "--------------\n");
    std::printf("output checksum: 0x%016llx\n", (unsigned long long)total_hash);
    if (failures == 0) {
        std::printf("RESULT: PASS (no inline-asm fallback required)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (%d failed checks)\n", failures);
    return 1;
}
