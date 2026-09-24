// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

namespace prefill_q8_wmma_streamed {

/**
 * Standard PyTorch HIP entrypoint for Q8_0 matmul on actual prefill shapes.
 * Variant 0 (Control): Reference source MMQ path with guarded allocation.
 * Variant 1 (Candidate): Dedicated gfx12 int8 WMMA streamed kernel with bounded
 *                        LDS tiles (8 Q8_0 blocks = 256 K elements per wave),
 *                        exact scale product ordering ((d0 * d1) * sumi),
 *                        INT32 accumulator reset per 32-element block,
 *                        persistent FP32 accumulators across streamed K-tiles,
 *                        and zero overread on tightly-packed [N, (K/32)*34] weights.
 *
 * Parameters:
 *   W: [N, (K/32)*34] uint8 (packed Q8_0 weights)
 *   X: [M, K] bfloat16 input activations
 *   variant: 0 (Control MMQ) or 1 (Candidate Streamed WMMA)
 *
 * Supported Dimensions:
 *   K in (2560, 3072) (80 or 96 blocks, divisible by 8)
 *   N multiple of 128
 *   M in 1..65535
 *
 * Returns:
 *   Y: [M, N] bfloat16 output activations
 */
torch::Tensor q8_matmul(
    torch::Tensor W,
    torch::Tensor X,
    py::object variant = py::int_(0)
);

} // namespace prefill_q8_wmma_streamed
