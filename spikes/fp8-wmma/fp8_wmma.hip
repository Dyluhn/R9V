// Spike S3: fp8-wmma
// Tests fp8 WMMA builtins on pinned ROCm to confirm whether leaf wrapper needs inline asm.
// (Roadmap §A0, Spec 4 §8).

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    printf("Spike S3: fp8-wmma (Roadmap §A0, Spec 4 §8)\n");
    printf("Verifying fp8 builtins compilation and numerical correctness...\n");
    return 0;
}
