// SPDX-License-Identifier: Apache-2.0
// Empty kernel fixture for real GPU smoke verification (Spec 4 §10, Spec 14 §3).

#include <hip/hip_runtime.h>

extern "C" __global__ void empty_kernel() {
    // Intentionally empty kernel for runtime launch verification
}
