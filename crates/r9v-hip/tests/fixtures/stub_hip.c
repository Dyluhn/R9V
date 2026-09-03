// SPDX-License-Identifier: Apache-2.0
// Stub libamdhip64 for deterministic hosted testing of r9v-hip (Spec 14 §2, §3).

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int hipError_t;
#define HIP_SUCCESS 0
#define HIP_ERROR_INVALID_DEVICE 101
#define HIP_ERROR_OUT_OF_MEMORY 2
#define HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED 704
#define HIP_ERROR_PEER_ACCESS_NOT_ENABLED 705

// Mock device properties struct (1472 bytes)
typedef struct {
    char name[256];
    uint8_t uuid[16];
    char luid[8];
    uint32_t luid_device_node_mask;
    uint8_t _pad0[4];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    uint8_t _gap_to_concurrent[168];
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    uint8_t _gap_to_multi_gpu[60];
    int isMultiGpuBoard;
    uint8_t _gap_to_cooperative[28];
    int cooperativeLaunch;
    uint8_t _gap_to_arch[468];
    char gcnArchName[256];
    uint8_t _gap_end[56];
} MockDeviceProp;

hipError_t hipGetDeviceCount(int *count) {
    if (!count) return 1;
    *count = 2;
    return HIP_SUCCESS;
}

hipError_t hipSetDevice(int device) {
    if (device < 0 || device >= 2) {
        return HIP_ERROR_INVALID_DEVICE;
    }
    return HIP_SUCCESS;
}

hipError_t hipGetDevice(int *device) {
    if (!device) return 1;
    *device = 0;
    return HIP_SUCCESS;
}

hipError_t hipGetDevicePropertiesR0600(void *prop_ptr, int device) {
    if (!prop_ptr) return 1;
    if (device < 0 || device >= 2) return HIP_ERROR_INVALID_DEVICE;

    MockDeviceProp *prop = (MockDeviceProp *)prop_ptr;
    memset(prop, 0, sizeof(MockDeviceProp));
    strncpy(prop->name, "Stub AMD Radeon AI PRO R9700", sizeof(prop->name) - 1);
    strncpy(prop->gcnArchName, "amdgcn-amd-amdhsa--gfx1201", sizeof(prop->gcnArchName) - 1);
    prop->totalGlobalMem = 34359738368ULL; // 32 GiB
    prop->sharedMemPerBlock = 65536;
    prop->regsPerBlock = 512;
    prop->warpSize = 32;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 1024;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRate = 2400000;
    prop->major = 12;
    prop->minor = 0;
    prop->multiProcessorCount = 64;
    prop->pciBusID = 3;
    prop->pciDeviceID = 0;
    prop->pciDomainID = 0;
    prop->isMultiGpuBoard = 0;
    prop->canMapHostMemory = 1;
    prop->concurrentKernels = 1;
    prop->ECCEnabled = 1;
    prop->cooperativeLaunch = 1;

    return HIP_SUCCESS;
}

const char *hipGetErrorString(hipError_t error) {
    if (error == HIP_SUCCESS) return "hipSuccess";
    if (error == HIP_ERROR_INVALID_DEVICE) return "hipErrorInvalidDevice";
    if (error == HIP_ERROR_OUT_OF_MEMORY) return "hipErrorOutOfMemory";
    if (error == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED) return "hipErrorPeerAccessAlreadyEnabled";
    if (error == HIP_ERROR_PEER_ACCESS_NOT_ENABLED) return "hipErrorPeerAccessNotEnabled";
    return "hipErrorUnknown";
}

hipError_t hipMalloc(void **ptr, size_t size) {
    if (!ptr) return 1;
    if (size == 0xDEADBEEF) {
        return HIP_ERROR_OUT_OF_MEMORY;
    }
    *ptr = calloc(1, size ? size : 1);
    return HIP_SUCCESS;
}

hipError_t hipFree(void *ptr) {
    if (ptr) free(ptr);
    return HIP_SUCCESS;
}

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
    (void)flags;
    if (!ptr) return 1;
    *ptr = calloc(1, size ? size : 1);
    return HIP_SUCCESS;
}

hipError_t hipHostFree(void *ptr) {
    if (ptr) free(ptr);
    return HIP_SUCCESS;
}

hipError_t hipMemcpy(void *dst, const void *src, size_t count, int kind) {
    (void)kind;
    if (!dst || !src) return 1;
    memmove(dst, src, count);
    return HIP_SUCCESS;
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t count, int kind, void *stream) {
    (void)stream;
    return hipMemcpy(dst, src, count, kind);
}

hipError_t hipMemcpyPeerAsync(void *dst, int dst_dev, const void *src, int src_dev, size_t count, void *stream) {
    (void)dst_dev;
    (void)src_dev;
    (void)stream;
    return hipMemcpy(dst, src, count, 3);
}

hipError_t hipStreamCreate(void **stream) {
    if (!stream) return 1;
    *stream = malloc(sizeof(int));
    return HIP_SUCCESS;
}

hipError_t hipStreamCreateWithFlags(void **stream, unsigned int flags) {
    (void)flags;
    return hipStreamCreate(stream);
}

hipError_t hipStreamDestroy(void *stream) {
    if (stream) free(stream);
    return HIP_SUCCESS;
}

hipError_t hipStreamSynchronize(void *stream) {
    (void)stream;
    return HIP_SUCCESS;
}

hipError_t hipStreamQuery(void *stream) {
    (void)stream;
    return HIP_SUCCESS;
}

hipError_t hipStreamWaitEvent(void *stream, void *event, unsigned int flags) {
    (void)stream;
    (void)event;
    (void)flags;
    return HIP_SUCCESS;
}

hipError_t hipEventCreate(void **event) {
    if (!event) return 1;
    *event = malloc(sizeof(int));
    return HIP_SUCCESS;
}

hipError_t hipEventCreateWithFlags(void **event, unsigned int flags) {
    (void)flags;
    return hipEventCreate(event);
}

hipError_t hipEventDestroy(void *event) {
    if (event) free(event);
    return HIP_SUCCESS;
}

hipError_t hipEventRecord(void *event, void *stream) {
    (void)event;
    (void)stream;
    return HIP_SUCCESS;
}

hipError_t hipEventSynchronize(void *event) {
    (void)event;
    return HIP_SUCCESS;
}

hipError_t hipEventElapsedTime(float *ms, void *start, void *stop) {
    (void)start;
    (void)stop;
    if (!ms) return 1;
    *ms = 1.25f;
    return HIP_SUCCESS;
}

hipError_t hipEventQuery(void *event) {
    (void)event;
    return HIP_SUCCESS;
}

hipError_t hipModuleLoad(void **module, const char *fname) {
    (void)fname;
    if (!module) return 1;
    *module = malloc(sizeof(int));
    return HIP_SUCCESS;
}

hipError_t hipModuleLoadData(void **module, const void *image) {
    (void)image;
    if (!module) return 1;
    *module = malloc(sizeof(int));
    return HIP_SUCCESS;
}

hipError_t hipModuleUnload(void *module) {
    if (module) free(module);
    return HIP_SUCCESS;
}

hipError_t hipModuleGetFunction(void **func, void *module, const char *name) {
    (void)module;
    (void)name;
    if (!func) return 1;
    *func = malloc(sizeof(int));
    return HIP_SUCCESS;
}

hipError_t hipModuleLaunchKernel(void *func, unsigned int gx, unsigned int gy, unsigned int gz,
                                unsigned int bx, unsigned int by, unsigned int bz,
                                unsigned int sm, void *stream, void **params, void **extra) {
    (void)func; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)sm; (void)stream; (void)params; (void)extra;
    return HIP_SUCCESS;
}

hipError_t hipStreamBeginCapture(void *stream, int mode) {
    (void)stream; (void)mode;
    return HIP_SUCCESS;
}

hipError_t hipStreamEndCapture(void *stream, void **graph) {
    (void)stream;
    if (!graph) return 1;
    *graph = malloc(sizeof(int));
    return HIP_SUCCESS;
}

hipError_t hipGraphInstantiate(void **exec, void *graph, void **err_node, char *log_buf, size_t buf_size) {
    (void)graph; (void)err_node; (void)log_buf; (void)buf_size;
    if (!exec) return 1;
    *exec = malloc(sizeof(int));
    return HIP_SUCCESS;
}

#ifndef OMIT_HIP_GRAPH_LAUNCH
hipError_t hipGraphLaunch(void *exec, void *stream) {
    (void)exec; (void)stream;
    return HIP_SUCCESS;
}
#endif

hipError_t hipGraphDestroy(void *graph) {
    if (graph) free(graph);
    return HIP_SUCCESS;
}

hipError_t hipGraphExecDestroy(void *exec) {
    if (exec) free(exec);
    return HIP_SUCCESS;
}

hipError_t hipDeviceCanAccessPeer(int *can_access, int device, int peer_device) {
    (void)device; (void)peer_device;
    if (!can_access) return 1;
    *can_access = 1;
    return HIP_SUCCESS;
}

static int g_peer_access_enabled[2] = {0, 0};

hipError_t hipDeviceEnablePeerAccess(int peer_device, unsigned int flags) {
    (void)flags;
    if (peer_device < 0 || peer_device >= 2) return HIP_ERROR_INVALID_DEVICE;
    if (g_peer_access_enabled[peer_device]) {
        return HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED;
    }
    g_peer_access_enabled[peer_device] = 1;
    return HIP_SUCCESS;
}

hipError_t hipDeviceDisablePeerAccess(int peer_device) {
    if (peer_device < 0 || peer_device >= 2) return HIP_ERROR_INVALID_DEVICE;
    if (!g_peer_access_enabled[peer_device]) {
        return HIP_ERROR_PEER_ACCESS_NOT_ENABLED;
    }
    g_peer_access_enabled[peer_device] = 0;
    return HIP_SUCCESS;
}
