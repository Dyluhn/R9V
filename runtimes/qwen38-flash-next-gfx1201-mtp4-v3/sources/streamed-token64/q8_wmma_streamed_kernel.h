// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

namespace prefill_q8_wmma_streamed {

/**
 * Standard PyTorch HIP entrypoint for Q8_0 matmul on actual prefill shapes.
 * Variant 0 (MMQ Oracle): Reference source MMQ path with guarded allocation.
 * Variant 1 (Control Token32): Dedicated gfx12 int8 WMMA streamed kernel with
 *                              32 tokens/workgroup (2 WMMA fragments per wave),
 *                              bounded LDS tile (8 Q8_0 blocks = 256 K elements per wave),
 *                              exact scale product ordering ((d0 * d1) * sumi),
 *                              INT32 accumulator reset per 32-element block,
 *                              persistent FP32 accumulators across streamed K-tiles.
 * Variant 2 (Candidate Token64): Dedicated gfx12 int8 WMMA streamed kernel with
 *                              64 tokens/workgroup (4 WMMA fragments per wave),
 *                              reusing each staged 8-block weight tile across 4 fragments,
 *                              halving weight-stage workgroups, with identical 17,408B LDS.
 *
 * Parameters:
 *   W: [N, (K/32)*34] uint8 (packed Q8_0 weights)
 *   X: [M, K] bfloat16 input activations
 *   variant: 0 (MMQ Oracle), 1 (Token32 Control), 2 (Token64 Candidate)
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
    py::object variant = py::int_(2)
);

} // namespace prefill_q8_wmma_streamed
