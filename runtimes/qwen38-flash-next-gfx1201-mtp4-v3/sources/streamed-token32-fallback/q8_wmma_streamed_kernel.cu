// SPDX-License-Identifier: Apache-2.0
#include "q8_wmma_streamed_kernel.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>
#include <climits>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "csrc/hip_compat.h"
#include "csrc/dispatch_utils.h"
#include "csrc/gguf/ggml-common_hip.h"
#include "csrc/gguf/vecdotq_hip.cuh"
#include "csrc/gguf/dequantize_hip.cuh"
#include "csrc/gguf/mmvq_hip.cuh"
#include "csrc/gguf/mmq_hip.cuh"

namespace prefill_q8_wmma_streamed {

using bf16 = __hip_bfloat16;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

constexpr int kWave = 32;
constexpr int kWaves = 4;
constexpr int kThreads = kWave * kWaves; // 128 threads per workgroup
constexpr int kRowsPerWave = 16;
constexpr int kRowsPerBlock = kWaves * kRowsPerWave; // 64 rows per block
constexpr int kTokensPerBlock = 32; // two 16-token tiles: n=0 and n=1

constexpr int kBlockBytes = 34; // sizeof(block_q8_0) = 2 byte fp16 scale + 32 byte int8 quants
constexpr int kChunkBlocks = 8; // 8 blocks of 32 = 256 K elements per chunk tile
constexpr int kChunkRowBytes = kChunkBlocks * kBlockBytes; // 8 * 34 = 272 bytes per row per chunk
constexpr int kQuadsPerChunkRow = kChunkRowBytes / 16; // 272 / 16 = 17 uint4 quads
constexpr int kTotalQuadsPerWave = kRowsPerWave * kQuadsPerChunkRow; // 16 * 17 = 272 quads
constexpr int kLoadsPerLane = (kTotalQuadsPerWave + kWave - 1) / kWave; // (272 + 31) / 32 = 9 uint4 loads per thread

// ============================================================================
// 1. Activation Quantization: quantize_row_q8_1_cuda
// Byte-identical to parent implementation in q8_small_k_wmma_kernel.cu
// ============================================================================

template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127.0f;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, hipStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = ::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL((quantize_q8_1), dim3(num_blocks), dim3(block_size), 0, stream, 
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

// ============================================================================
// 2. Working gfx12 WMMA Fragment Dequantizer for Q8_0
// Reused from research/tg-next/source_audit/image_extracted/retained-mtp4/kernels/r9v_moe_wmma.cu
// ============================================================================

__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t* p) {
  uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

struct DequantQ80 {
  static constexpr int kBlockBytes = 34; // sizeof(block_q8_0)
  static constexpr int kBlockCols = 32;
  __device__ __forceinline__ static void load(const uint8_t* row, int kstep,
                                              int h, v2i& lo, v2i& hi,
                                              float& scale) {
    const uint8_t* block = row + kstep * kBlockBytes;
    const uint8_t* q = block + 2 + 8 * h;
    lo[0] = static_cast<int>(load_u32_unaligned(q));
    lo[1] = static_cast<int>(load_u32_unaligned(q + 4));
    hi[0] = static_cast<int>(load_u32_unaligned(q + 16));
    hi[1] = static_cast<int>(load_u32_unaligned(q + 20));
    scale = __half2float(*reinterpret_cast<const half*>(block));
  }
};

// ============================================================================
// 3. Dedicated Streamed Dense Q8_0 x Q8_1 WMMA Kernel for Larger K
// Bounded LDS tile (8 blocks = 256 K elements per chunk).
// Streams successive K tiles in ascending order (K0..K/32-1).
// Accumulators persist across staged chunks.
// Workgroup producer/consumer barriers before reading and before overwriting.
// INT32 accumulator reset per 32-block; exact scale product (d0*d1)*sumi;
// sequential FP32 block accumulation and BF16 store.
// ============================================================================

__global__ __launch_bounds__(kThreads, 2)
void dense_q8_0_q8_1_wmma_streamed_kernel(
    const uint8_t* __restrict__ weight,
    const void* __restrict__ quant_x_ptr,
    bf16* __restrict__ output,
    int N,
    int K,
    int M,
    int token_stride_ints) {

  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int half = lane >> 4; // which 8-wide K slice of the 16-wide sub-step (0 or 1)
  const int idx = lane & 15;  // row within wave / token within tile (0..15)
  const int row0 = blockIdx.x * kRowsPerBlock + wave * kRowsPerWave;
  const bool wave_active = row0 < N;

  const int* quant_x = reinterpret_cast<const int*>(quant_x_ptr);

  // Setup activation pointers for this lane's two tokens (tiles n=0 and n=1)
  const int* act[2];
  int token_ids[2];
  const int col0 = blockIdx.y * kTokensPerBlock;
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int t = col0 + n * 16 + idx;
    token_ids[n] = t;
    act[n] = (t < M) ? (quant_x + static_cast<int64_t>(t) * token_stride_ints) : nullptr;
  }

  __shared__ __align__(16) uint8_t staged[kWaves][kRowsPerWave * kChunkRowBytes];
  uint8_t* wave_stage = staged[wave];
  const int64_t row_stride = static_cast<int64_t>(K / 32) * kBlockBytes;
  const uint8_t* weight_end = weight + static_cast<int64_t>(N) * row_stride;

  // Persistent accumulator registers for 2 token tiles x 8 rows across all K chunks
  float acc[2][8];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[n][i] = 0.0f;
    }
  }

  const int num_chunks = (K / 32) / kChunkBlocks;

  // Stream successive K tiles in strictly ascending order: K0..K/32-1
  // Keep outer loop rolled; actual compiler resources require offline review
#pragma unroll 1
  for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    // 1. Stage bounded chunk of 8 blocks (272 bytes per row) into LDS
    uint4 values[kLoadsPerLane];
#pragma unroll
    for (int j = 0; j < kLoadsPerLane; ++j) {
      const int unit = lane + j * kWave;
      if (wave_active && unit < kTotalQuadsPerWave) {
        const int r = unit / kQuadsPerChunkRow;
        const int q = unit - r * kQuadsPerChunkRow;
        const int64_t global_r = row0 + r;
        const int64_t chunk_start = static_cast<int64_t>(chunk_idx) * kChunkRowBytes;
        const uint8_t* src = weight + global_r * row_stride + chunk_start + 16 * q;
        if (global_r < N && src + 16 <= weight_end) {
          values[j] = *reinterpret_cast<const uint4*>(src);
        } else {
          uint32_t words[4] = {0, 0, 0, 0};
          for (int b = 0; b < 16 && src + b < weight_end; ++b) {
            words[b / 4] |= static_cast<uint32_t>(src[b]) << (8 * (b % 4));
          }
          values[j] = make_uint4(words[0], words[1], words[2], words[3]);
        }
      } else {
        values[j] = make_uint4(0, 0, 0, 0);
      }
    }

#pragma unroll
    for (int j = 0; j < kLoadsPerLane; ++j) {
      const int unit = lane + j * kWave;
      if (wave_active && unit < kTotalQuadsPerWave) {
        const int r = unit / kQuadsPerChunkRow;
        const int q = unit - r * kQuadsPerChunkRow;
        *reinterpret_cast<uint4*>(wave_stage + r * kChunkRowBytes + 16 * q) = values[j];
      }
    }

    // Workgroup producer barrier: ensure all staged chunks are visible before reading
    __syncthreads();

    // 2. Process 8 blocks (256 K elements) in this staged chunk
    const uint8_t* staged_row = wave_stage + idx * kChunkRowBytes;

#pragma unroll 1
    for (int b = 0; b < kChunkBlocks; ++b) {
      const int kstep = chunk_idx * kChunkBlocks + b;

      v2i a_lo, a_hi;
      float row_scale;
      DequantQ80::load(staged_row, b, half, a_lo, a_hi, row_scale);

      // Scales of the eight rows this lane accumulates: rows 8*half + i
      float scales[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        scales[i] = __shfl(row_scale, 8 * half + i, kWave);
      }

#pragma unroll
      for (int n = 0; n < 2; ++n) {
        v2i b_lo = {0, 0}, b_hi = {0, 0};
        float d8 = 0.0f;
        if (act[n] != nullptr) {
          // block_q8_1 = { half2 ds; int8_t qs[32]; }: scale first, then 8 ints of quants
          const int* block = act[n] + kstep * 9;
          d8 = __low2float(*reinterpret_cast<const half2*>(block));
          b_lo[0] = block[1 + 2 * half];
          b_lo[1] = block[2 + 2 * half];
          b_hi[0] = block[5 + 2 * half];
          b_hi[1] = block[6 + 2 * half];
        }

        v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_lo, true, b_lo, c, false);
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_hi, true, b_hi, c, false);

        // EXACT per-block scale product order: (d_weight * d_activation) * float(intsum)
        // Preserves source d0*d1 then *sumi; compiled numerical parity remains to be tested.
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float d0 = scales[i];
          const float d1 = d8;
          acc[n][i] += (d0 * d1) * static_cast<float>(c[i]);
        }
      }
    }

    // Workgroup consumer barrier: ensure all threads finish reading before next chunk overwrites LDS
    __syncthreads();
  }

  // Store: lane owns column (token_ids[n]) and rows row0 + 8*half .. row0 + 8*half + 7 of tile
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int t = token_ids[n];
    if (!wave_active || t >= M) continue;

    bf16* out = output + static_cast<int64_t>(t) * N + row0 + 8 * half;
    uint32_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bf16 lo_v = __float2bfloat16(acc[n][2 * i]);
      const bf16 hi_v = __float2bfloat16(acc[n][2 * i + 1]);
      packed[i] = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo_v)) |
                  (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi_v)) << 16);
    }
    *reinterpret_cast<uint4*>(out) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

// ============================================================================
// 4. Entrypoint: q8_matmul(W, X, variant)
// ============================================================================

torch::Tensor q8_matmul(
    torch::Tensor W,
    torch::Tensor X,
    py::object variant_obj) {

    // 1. Dtype validation
    TORCH_CHECK(W.scalar_type() == torch::kUInt8,
        "W must be uint8 packed Q8_0 weights, got ", W.scalar_type());
    TORCH_CHECK(X.scalar_type() == torch::kBFloat16,
        "X must be bfloat16 input activations, got ", X.scalar_type());

    // 2. Device validation
    TORCH_CHECK(W.is_cuda(), "W must be on CUDA/ROCm device");
    TORCH_CHECK(X.is_cuda(), "X must be on CUDA/ROCm device");
    TORCH_CHECK(W.device() == X.device(),
        "W and X must be on same device, got W=", W.device(), " vs X=", X.device());

    // 3. Layout validation
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    // 4. Shape validation
    TORCH_CHECK(W.dim() == 2, "W must be 2D [N, packed_row_bytes], got dim=", W.dim());
    TORCH_CHECK(X.dim() == 2, "X must be 2D [M, K], got dim=", X.dim());

    const int64_t N = W.size(0);
    const int64_t packed_row_bytes = W.size(1);
    const int64_t M = X.size(0);
    const int64_t K = X.size(1);

    // 5. Dimension contract validation
    TORCH_CHECK(K == 2560 || K == 3072,
        "K must be 2560 or 3072 for streamed WMMA specialization, got ", K);
    const int64_t expected_packed_bytes = (K / 32) * kBlockBytes;
    TORCH_CHECK(packed_row_bytes == expected_packed_bytes,
        "Packed row bytes mismatch: expected ", expected_packed_bytes, " got ", packed_row_bytes);

    int variant = 0;
    if (py::isinstance<py::int_>(variant_obj)) {
        variant = variant_obj.cast<int>();
    } else if (py::isinstance<py::str>(variant_obj)) {
        std::string s = variant_obj.cast<std::string>();
        if (s == "candidate" || s == "wmma") variant = 1;
        else if (s == "control" || s == "mmq") variant = 0;
        else TORCH_CHECK(false, "Unknown variant string: ", s);
    } else {
        TORCH_CHECK(false, "variant must be int (0/1) or string ('control'/'candidate')");
    }

    TORCH_CHECK(variant == 0 || variant == 1, "Invalid variant");
    TORCH_CHECK(M > 0 && M <= 65535 && N > 0 && N % 128 == 0,
                "Diagnostic requires positive M<=65535, N multiple of 128");
    TORCH_CHECK(N <= INT_MAX - 127 && M * N <= INT_MAX && N * (K / 32) <= INT_MAX,
                "Diagnostic integer limits");
    const c10::cuda::CUDAGuard device_guard(X.device());

    // 6. Allocate output Y: [M, N] bfloat16
    auto Y = torch::zeros({M, N}, X.options());

    // 7. Activation Quantization Buffer: padded to multiple of 512
    const int64_t padded = (K + 512 - 1) / 512 * 512;
    const int token_stride_ints = static_cast<int>((padded / 32) * 9);
    auto quant_X = torch::empty(
        {M, token_stride_ints * 4},
        X.options().dtype(torch::kUInt8));

    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    // 8. Launch Activation Quantizer
    quantize_row_q8_1_cuda<c10::BFloat16>(
        reinterpret_cast<const c10::BFloat16*>(X.data_ptr()),
        quant_X.data_ptr(),
        static_cast<int>(K),
        static_cast<int>(M),
        stream);

    // 9. Dispatch Matmul based on variant
    if (variant == 0) {
        // Control: Reference source MMQ path with guarded allocation
        ggml_mul_mat_q8_0_q8_1_cuda<c10::BFloat16>(
            W.data_ptr(),
            quant_X.data_ptr(),
            reinterpret_cast<c10::BFloat16*>(Y.data_ptr()),
            static_cast<int>(K),
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(padded),
            static_cast<int>(N),
            stream);
    } else {
        // Candidate: Dedicated Streamed gfx12 WMMA Kernel
        const int block_num_x = static_cast<int>((N + kRowsPerBlock - 1) / kRowsPerBlock);
        const int block_num_y = static_cast<int>((M + kTokensPerBlock - 1) / kTokensPerBlock);
        const dim3 block_nums(block_num_x, block_num_y, 1);
        const dim3 block_dims(kThreads, 1, 1);

        hipLaunchKernelGGL(
            (dense_q8_0_q8_1_wmma_streamed_kernel),
            block_nums,
            block_dims,
            0,
            stream,
            reinterpret_cast<const uint8_t*>(W.data_ptr()),
            quant_X.data_ptr(),
            reinterpret_cast<bf16*>(Y.data_ptr()),
            static_cast<int>(N),
            static_cast<int>(K),
            static_cast<int>(M),
            token_stride_ints);
    }

    AT_CUDA_CHECK(hipGetLastError());
    return Y;
}

} // namespace prefill_q8_wmma_streamed

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Prefill Q8_0 Streamed WMMA Extension (Control MMQ vs Candidate Streamed WMMA K=2560/3072)";
    m.def("q8_matmul", &prefill_q8_wmma_streamed::q8_matmul,
          "Execute Q8_0 matmul: Y = X @ W^T with selectable variant (0/control MMQ, 1/candidate Streamed WMMA)",
          py::arg("W"), py::arg("X"), py::arg("variant") = py::int_(0));
    m.def("q8_matmul_control", [](torch::Tensor W, torch::Tensor X) {
        return prefill_q8_wmma_streamed::q8_matmul(W, X, py::int_(0));
    }, "Execute Q8_0 matmul with Control MMQ", py::arg("W"), py::arg("X"));
    m.def("q8_matmul_candidate", [](torch::Tensor W, torch::Tensor X) {
        return prefill_q8_wmma_streamed::q8_matmul(W, X, py::int_(1));
    }, "Execute Q8_0 matmul with Candidate Streamed WMMA", py::arg("W"), py::arg("X"));
}
