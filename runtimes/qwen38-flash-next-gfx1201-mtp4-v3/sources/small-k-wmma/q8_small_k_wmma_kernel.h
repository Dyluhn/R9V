// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

namespace prefill_q8_small_k_wmma {

/**
 * Standard PyTorch HIP entrypoint for Q8_0 small-K matmul.
 * Variant 0 (Control): Reference source MMQ path with explicit 68-byte guarded allocation.
 * Variant 1 (Candidate): Dedicated gfx12 int8 WMMA dense K=320 kernel with exact
 *                        scale product ordering ((d0 * d1) * sumi), int32 accumulation,
 *                        and zero overread on tightly-packed [N, 340] weight tensor.
 *
 * Parameters:
 *   W: [N, 340] uint8 (packed Q8_0 weights: 10 blocks of 34 bytes = 340 bytes/row)
 *   X: [M, 320] bfloat16 input activations
 *   variant: 0 (Control MMQ) or 1 (Candidate Dense WMMA)
 *
 * Returns:
 *   Y: [M, N] bfloat16 output activations
 */
torch::Tensor q8_matmul(
    torch::Tensor W,
    torch::Tensor X,
    py::object variant = py::int_(0)
);

} // namespace prefill_q8_small_k_wmma
