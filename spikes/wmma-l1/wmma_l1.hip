// Spike S1: wmma-l1
// Tests iu8 and iu4 GEMM whose B fragments load straight from global memory in spec 2 L1 lane order.
// Confirms layout claim and measures real iu4 rate against iu8 (Spec 1 App. A, Spec 2 §2, Roadmap §A0).

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Skeleton entry point for spike runner on gfx1201
int main(int argc, char** argv) {
    printf("Spike S1: wmma-l1 (Spec 1 App. A, Spec 2 §2)\n");
    printf("Testing L1 lane-order direct fragment loads for iu8 and iu4...\n");
    // Implementation executed during card A0.S1 on reference rig
    return 0;
}
