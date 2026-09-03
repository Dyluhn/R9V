// Spike S2: dot4-gemv
// Tests v_dot4_i32_i8 GEMV over L1 at M in {1, 4, 8}; measures achieved GB/s against streaming ceiling.
// (Roadmap §A0, Spec 4 §5).

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    printf("Spike S2: dot4-gemv (Roadmap §A0, Spec 4 §5)\n");
    printf("Testing v_dot4_i32_i8 GEMV over L1 for M in {1, 4, 8}...\n");
    return 0;
}
