// SPDX-License-Identifier: Apache-2.0
/**
 * native_cpu_shim.cpp
 * CPU-only verification and arithmetic simulation for gfx12 WMMA streamed dense matmul.
 * Tests exact fragment layout, chunk streaming, scale ordering, and zero-overread memory access.
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace prefill_q8_wmma_streamed_cpu {

constexpr int QK8_0 = 32;
constexpr int SIZEOF_BLOCK_Q8_0 = 34; // 2 byte half scale + 32 byte int8 quants
constexpr int SIZEOF_BLOCK_Q8_1 = 36; // 4 byte half2 ds + 32 byte int8 quants

constexpr int K_WAVE = 32;
constexpr int K_WAVES = 4;
constexpr int K_THREADS = K_WAVE * K_WAVES; // 128
constexpr int K_ROWS_PER_WAVE = 16;
constexpr int K_ROWS_PER_BLOCK = K_WAVES * K_ROWS_PER_WAVE; // 64
constexpr int K_TOKENS_PER_BLOCK = 32;

constexpr int K_CHUNK_BLOCKS = 8; // 8 blocks of 32 = 256 K elements per chunk
constexpr int K_CHUNK_ROW_BYTES = K_CHUNK_BLOCKS * SIZEOF_BLOCK_Q8_0; // 8 * 34 = 272 bytes
constexpr int K_QUADS_PER_CHUNK_ROW = K_CHUNK_ROW_BYTES / 16; // 17 quads
constexpr int K_TOTAL_QUADS_PER_WAVE = K_ROWS_PER_WAVE * K_QUADS_PER_CHUNK_ROW; // 272 quads
constexpr int K_LOADS_PER_LANE = (K_TOTAL_QUADS_PER_WAVE + K_WAVE - 1) / K_WAVE; // 9

// Contract validation
inline void validate_contract(int64_t M, int64_t N, int64_t K, int64_t packed_row_bytes) {
    if (M <= 0 || M > 65535) throw std::invalid_argument("M must be in 1..65535");
    if (N <= 0 || N % 128 != 0) throw std::invalid_argument("N must be a positive multiple of 128");
    if (K != 2560 && K != 3072) throw std::invalid_argument("K must be 2560 or 3072");
    int64_t expected_packed_bytes = (K / 32) * SIZEOF_BLOCK_Q8_0;
    if (packed_row_bytes != expected_packed_bytes) {
        throw std::invalid_argument("Packed row bytes mismatch: expected " +
                                    std::to_string(expected_packed_bytes) + " got " +
                                    std::to_string(packed_row_bytes));
    }
}

// Memory boundary verification for streamed dense W
// Returns byte offset read by load unit, or -1 if outside valid range
inline int64_t simulate_weight_load_offset(
    int64_t N, int64_t K, int block_idx_x, int wave, int lane, int chunk_idx, int load_j) {
    const int row0 = block_idx_x * K_ROWS_PER_BLOCK + wave * K_ROWS_PER_WAVE;
    if (row0 >= N) return -1;

    const int unit = lane + load_j * K_WAVE;
    if (unit >= K_TOTAL_QUADS_PER_WAVE) return -1;

    const int r = unit / K_QUADS_PER_CHUNK_ROW;
    const int q = unit - r * K_QUADS_PER_CHUNK_ROW;
    const int64_t global_r = row0 + r;
    if (global_r >= N) return -1;

    const int64_t row_stride = (K / 32) * SIZEOF_BLOCK_Q8_0;
    const int64_t chunk_start = static_cast<int64_t>(chunk_idx) * K_CHUNK_ROW_BYTES;
    const int64_t src_quad_start = global_r * row_stride + chunk_start + 16 * q;

    const int64_t total_bytes = N * row_stride;
    if (src_quad_start >= total_bytes) {
        return -1; // Guarded by weight_end boundary check
    }
    return src_quad_start;
}

// Exact WMMA fragment index mapping for a 32-element block
struct WmmaFragmentMap {
    static int get_output_row(int lane, int reg_i) {
        int half = lane >> 4;
        return 8 * half + reg_i;
    }

    static int get_output_col(int lane) {
        return lane & 15;
    }

    static void get_k_range_lo(int lane, int& start, int& count) {
        int half = lane >> 4;
        start = half * 8;
        count = 8;
    }

    static void get_k_range_hi(int lane, int& start, int& count) {
        int half = lane >> 4;
        start = 16 + half * 8;
        count = 8;
    }
};

// Arithmetic reference: exact 32-element int32 sum and scale ordering
inline float compute_block_dot_exact(
    const int8_t* w_quants,
    float w_scale,
    const int8_t* a_quants,
    float a_scale) {
    int32_t intsum = 0;
    for (int i = 0; i < 32; ++i) {
        intsum += static_cast<int32_t>(w_quants[i]) * static_cast<int32_t>(a_quants[i]);
    }
    // EXACT scale product order: (d0 * d1) * sumi
    return (w_scale * a_scale) * static_cast<float>(intsum);
}

} // namespace prefill_q8_wmma_streamed_cpu
