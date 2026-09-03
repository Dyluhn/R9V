// Spike S4: hipgraph (Card A0.S4; Roadmap §A0, Spec 6 §5.2, Spec 14).
// Captures an exact 400-kernel launch list into a hipGraph on gfx1201,
// repeatedly replays it, and compares against the same 400 sequential
// launches for correctness, dispatch overhead, and replay stability.
//
// Pass criteria (strict, nonzero exit on any failure):
//  1. All HIP calls succeed (device 0).
//  2. Captured graph contains exactly 400 nodes.
//  3. Every sequential and every graph replay reproduces the host-computed
//     reference buffer bit-exactly (memcmp), proving launches are real,
//     observable, and semantically equivalent across paths.
//  4. Graph replay is stable: max sample <= 2x median and CV <= 10%.
//  5. Graph replay median total time is strictly less than the sequential
//     launch-list median (the Spec 6 §5.2 / Roadmap §A0 claim). Otherwise
//     FAIL is recorded for SPEC-ISSUES; the claim is not weakened.
//
// Capture + instantiate happen once, outside all replay timing. Both timed
// paths reset the output buffer first, are bracketed by hipEvents on the
// same stream, and are synchronized before the stop event is read.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,          \
                         __LINE__, hipGetErrorString(err));                    \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

#define FAIL(...)                                                              \
    do {                                                                       \
        std::fprintf(stderr, "FAIL: " __VA_ARGS__);                            \
        std::fprintf(stderr, "\n");                                            \
        g_fail = true;                                                         \
    } while (0)

static bool g_fail = false;

static const int kLaunches = 400;      // exact launch-list length (card A0.S4)
static const int kChunkElem = 2048;    // uint32 words per launch (8 KiB)
static const int kBlock = 256;         // threads per block
static const int kWarmups = 10;
static const int kSamples = 50;        // timed replays per path
static const uint32_t kMul = 1664525u;
static const uint32_t kAdd = 1013904223u;

// Deterministic SplitMix64 fill in [-1, 1).
static uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Nontrivial per-launch kernel: 8-iteration integer multiply-add chain with
// a per-launch term, writing a full 8 KiB chunk. Integer arithmetic is
// bit-exact on host and device, so the host reference is a strict oracle.
// Every output word is read back and memcmp'd, so no launch can be eliminated.
__global__ void step_kernel(const uint32_t* __restrict__ in,
                            uint32_t* __restrict__ out, int chunk) {
    int base = chunk * kChunkElem;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < kChunkElem) {
        uint32_t c = kAdd + (uint32_t)chunk * 0x9E3779B9u;
        uint32_t x = in[base + i];
#pragma unroll
        for (int r = 0; r < 8; ++r) x = x * kMul + c + (uint32_t)r;
        out[base + i] = x;
    }
}

static void launch_list(hipStream_t s, const uint32_t* d_in, uint32_t* d_out) {
    for (int c = 0; c < kLaunches; ++c) {
        hipLaunchKernelGGL(step_kernel, dim3((kChunkElem + kBlock - 1) / kBlock),
                           dim3(kBlock), 0, s, d_in, d_out, c);
    }
}

struct Stats {
    double median, min, max, mean, stddev;
};

static Stats summarize(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    Stats st;
    st.min = v.front();
    st.max = v.back();
    st.median = (v[v.size() / 2 - (v.size() % 2 == 0 ? 1 : 0)] + v[v.size() / 2]) / 2.0;
    st.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double acc = 0.0;
    for (float x : v) acc += (x - st.mean) * (x - st.mean);
    st.stddev = std::sqrt(acc / v.size());
    return st;
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0;
    HIP_CHECK(hipDriverGetVersion(&driver));
    HIP_CHECK(hipRuntimeGetVersion(&runtime));
    std::printf("device: %s arch=%s\n", prop.name, prop.gcnArchName);
    std::printf("driver=%d runtime=%d\n", driver, runtime);

    const size_t n = (size_t)kLaunches * kChunkElem;
    std::vector<uint32_t> h_in(n), h_ref(n);
    uint64_t seed = 0x243F6A8885A308D3ULL;
    for (size_t i = 0; i < n; ++i) h_in[i] = (uint32_t)splitmix64(seed);
    // Host reference: exact same integer arithmetic as the kernel.
    for (int c = 0; c < kLaunches; ++c) {
        uint32_t cc = kAdd + (uint32_t)c * 0x9E3779B9u;
        for (int i = 0; i < kChunkElem; ++i) {
            uint32_t x = h_in[(size_t)c * kChunkElem + i];
            for (int r = 0; r < 8; ++r) x = x * kMul + cc + (uint32_t)r;
            h_ref[(size_t)c * kChunkElem + i] = x;
        }
    }

    uint32_t *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, n * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_out, n * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), n * sizeof(uint32_t), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    std::vector<uint32_t> h_out(n);
    auto reset_and_sync = [&]() {
        HIP_CHECK(hipMemsetAsync(d_out, 0, n * sizeof(uint32_t), stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    };
    auto verify = [&](const char* tag, int iter) {
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, n * sizeof(uint32_t),
                            hipMemcpyDeviceToHost));
        if (std::memcmp(h_out.data(), h_ref.data(), n * sizeof(uint32_t)) != 0) {
            size_t bad = 0;
            for (size_t i = 0; i < n; ++i)
                if (h_out[i] != h_ref[i]) { bad = i; break; }
            FAIL("%s iter %d: output mismatch at word %zu (got 0x%08x expected 0x%08x)",
                 tag, iter, bad, h_out[bad], h_ref[bad]);
        }
    };

    // Sequential warmups (also proves the launch list is correct pre-capture).
    for (int w = 0; w < kWarmups; ++w) {
        reset_and_sync();
        launch_list(stream, d_in, d_out);
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    reset_and_sync();
    launch_list(stream, d_in, d_out);
    HIP_CHECK(hipStreamSynchronize(stream));
    verify("seq-warmup", 0);

    // Capture the exact same 400-launch list. Timing excluded.
    hipGraph_t graph;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    launch_list(stream, d_in, d_out);
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    size_t num_nodes = 0;
    HIP_CHECK(hipGraphGetNodes(graph, nullptr, &num_nodes));
    std::printf("captured nodes: %zu (expected %d)\n", num_nodes, kLaunches);
    if (num_nodes != (size_t)kLaunches)
        FAIL("graph has %zu nodes, expected exactly %d", num_nodes, kLaunches);
    hipGraphExec_t exec;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // Graph warmups.
    for (int w = 0; w < kWarmups; ++w) {
        reset_and_sync();
        HIP_CHECK(hipGraphLaunch(exec, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    reset_and_sync();
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    verify("graph-warmup", 0);

    // Timed runs, alternating paths to share thermal/drift conditions.
    // Reset + sync happen before the start event, so only replay time is measured.
    std::vector<float> t_seq, t_graph;
    t_seq.reserve(kSamples);
    t_graph.reserve(kSamples);
    for (int s = 0; s < kSamples; ++s) {
        reset_and_sync();
        HIP_CHECK(hipEventRecord(ev_start, stream));
        launch_list(stream, d_in, d_out);
        HIP_CHECK(hipEventRecord(ev_stop, stream));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        t_seq.push_back(ms * 1000.0f);
        verify("seq", s);

        reset_and_sync();
        HIP_CHECK(hipEventRecord(ev_start, stream));
        HIP_CHECK(hipGraphLaunch(exec, stream));
        HIP_CHECK(hipEventRecord(ev_stop, stream));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        t_graph.push_back(ms * 1000.0f);
        verify("graph", s);
    }

    Stats ss = summarize(t_seq), sg = summarize(t_graph);
    std::printf("seq_us median=%.2f min=%.2f max=%.2f mean=%.2f std=%.2f\n", ss.median,
                ss.min, ss.max, ss.mean, ss.stddev);
    std::printf("graph_us median=%.2f min=%.2f max=%.2f mean=%.2f std=%.2f\n", sg.median,
                sg.min, sg.max, sg.mean, sg.stddev);
    std::printf("seq raw:");
    for (float x : t_seq) std::printf(" %.1f", x);
    std::printf("\ngraph raw:");
    for (float x : t_graph) std::printf(" %.1f", x);
    std::printf("\n");
    double per_seq = ss.median / kLaunches, per_graph = sg.median / kLaunches;
    double speedup = ss.median / sg.median;
    double cv = sg.stddev / sg.median;
    std::printf("per-launch seq=%.3f us graph=%.3f us dispatch-saved=%.3f us speedup=%.3f\n",
                per_seq, per_graph, per_seq - per_graph, speedup);
    std::printf("graph stability: max/median=%.3f cv=%.4f\n", sg.max / sg.median, cv);

    // Stability gates (generous: only gross instability fails).
    if (sg.max > 2.0 * sg.median)
        FAIL("graph replay unstable: max %.1f us > 2x median %.1f us", sg.max, sg.median);
    if (cv > 0.10)
        FAIL("graph replay unstable: CV %.4f > 0.10", cv);
    // The spec claim, unweakened: replay must reduce dispatch overhead.
    if (!(sg.median < ss.median))
        FAIL("graph replay median %.1f us is not below sequential %.1f us", sg.median,
             ss.median);

    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    if (g_fail) {
        std::printf("RESULT: FAIL\n");
        return 1;
    }
    std::printf("RESULT: PASS\n");
    return 0;
}
