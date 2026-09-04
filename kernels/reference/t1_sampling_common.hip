// SPDX-License-Identifier: Apache-2.0
// T1 portable sampling shared device code (Spec 1 §4.F, §6.5; Spec 4 §2, §5.8).
//
// Included textually into each T1 sampling compiland by the runner test after
// the A3.2 generated ABI struct. Portable HIP only: plain C++ control flow,
// block-level sync, fixed-order arithmetic. No wave intrinsics, no builtins,
// no inline asm, no WMMA (Spec 4 §8; the CI portability grep enforces this).
#pragma once

// DECISION(A3.6): `rng_state` words are 3xu64 `{seed, step, draw}` per sequence,
// matching T0 `RngState { seed: u64, step: u64, draw_index: u32 }` with the
// sequence id carried by the `seq_ids` BatchMeta buffer (the canonical u32
// Philox counter word, same constraint as T0). The A3.2 ABI types the field as
// a mutable u64 pointer with an `[S]` doc note; one word cannot hold the full
// (seed, step, draw) state, so T1 defines the record width here. Rejected
// packing seed/step/draw into one u64 because any bit split truncates valid
// T0 states (u64 seeds, steps above 2^32). Spec 1 §4.F, Spec 4 §7.
struct R9vRngWords {
    unsigned long long seed;
    unsigned long long step;
    unsigned long long draw;
};

// DECISION(A3.6): the A3.2 ABI types `params` as `const void*` (heterogeneous
// per-sequence records), so T1 defines the device record once here: scalars
// plus an offset/count pair into a trailing `R9vBiasPair` run in the same
// allocation (S records, then all sparse pairs). Rejected fixed inline bias
// slots because the op contract admits arbitrary sparse bias counts.
// Spec 1 §4.F, Spec 4 §7.
struct R9vSamplingParams {
    float temperature;
    float top_p;
    float min_p;
    float repetition_penalty;
    float presence_penalty;
    float frequency_penalty;
    unsigned int top_k;
    unsigned int bias_start;
    unsigned int bias_count;
    unsigned int _pad;
};

struct R9vBiasPair {
    unsigned int token;
    float bias;
};

// Philox4x32-10 block function (Spec 1 §4.F, Salmon et al. 2011). Exact port of
// the T0 scalar reference: unsigned wrap is defined in C++, mulhilo via u64.
__device__ __forceinline__ void r9v_philox_round(unsigned int ctr[4],
                                                unsigned int key[2]) {
    const unsigned int M0 = 0xD2511F53u;
    const unsigned int M1 = 0xCD9E8D57u;
    unsigned long long p0 = (unsigned long long)M0 * (unsigned long long)ctr[0];
    unsigned long long p1 = (unsigned long long)M1 * (unsigned long long)ctr[2];
    unsigned int hi0 = (unsigned int)(p0 >> 32);
    unsigned int lo0 = (unsigned int)(p0 & 0xFFFFFFFFu);
    unsigned int hi1 = (unsigned int)(p1 >> 32);
    unsigned int lo1 = (unsigned int)(p1 & 0xFFFFFFFFu);
    unsigned int c0 = hi1 ^ ctr[1] ^ key[0];
    unsigned int c1 = lo1;
    unsigned int c2 = hi0 ^ ctr[3] ^ key[1];
    unsigned int c3 = lo0;
    ctr[0] = c0;
    ctr[1] = c1;
    ctr[2] = c2;
    ctr[3] = c3;
}

__device__ __forceinline__ unsigned int r9v_philox_word(unsigned int draw,
                                                       unsigned int step_lo,
                                                       unsigned int step_hi,
                                                       unsigned int seq_word,
                                                       unsigned int seed_lo,
                                                       unsigned int seed_hi) {
    // Round 0 uses the initial key; rounds 1..9 bump by the Weyl constants
    // before the round (matches T0 `philox4x32_10` exactly).
    const unsigned int W0 = 0x9E3779B9u;
    const unsigned int W1 = 0xBB67AE85u;
    unsigned int ctr[4];
    ctr[0] = draw;
    ctr[1] = step_lo;
    ctr[2] = step_hi;
    ctr[3] = seq_word;
    unsigned int key[2];
    key[0] = seed_lo;
    key[1] = seed_hi;
    r9v_philox_round(ctr, key);
    for (int r = 1; r < 10; ++r) {
        key[0] += W0;
        key[1] += W1;
        r9v_philox_round(ctr, key);
    }
    return ctr[0];
}

// Maps a 32-bit word to f32 uniform on the open interval (0, 1) (Spec 1 §4.F).
// Upper 23 bits centered by +0.5 scaled by 2^-23; never exactly 0 or 1, so
// inverse-CDF and rejection thresholds never saturate at the boundary.
__device__ __forceinline__ float r9v_u32_to_unit_f32(unsigned int word) {
    return ((float)(word >> 9) + 0.5f) * (1.0f / 8388608.0f);
}

// One Philox uniform draw for (seed, seq, step, draw_index) (Spec 1 §4.F).
// Counter words are [draw, step_lo, step_hi, seq] and the key is the seed;
// the 64-bit step is split losslessly so steps above 2^32 never collide.
__device__ __forceinline__ float r9v_draw_at(unsigned long long seed,
                                            unsigned int seq_word,
                                            unsigned long long step,
                                            unsigned int draw) {
    unsigned int seed_lo = (unsigned int)(seed & 0xFFFFFFFFu);
    unsigned int seed_hi = (unsigned int)(seed >> 32);
    unsigned int step_lo = (unsigned int)(step & 0xFFFFFFFFu);
    unsigned int step_hi = (unsigned int)(step >> 32);
    unsigned int word =
        r9v_philox_word(draw, step_lo, step_hi, seq_word, seed_lo, seed_hi);
    return r9v_u32_to_unit_f32(word);
}

// Exact ascending-order comparator for (-logit, index) candidate pairs
// (Spec 1 §4.F, §6.5). Float equality (never `<`) decides ties so -0.0 and
// +0.0 compare equal exactly like T0's `partial_cmp`, with the lower index
// first. NaN cannot occur: the host wrapper rejects NaN/+Inf logits before
// launch, mirroring T0.
__device__ __forceinline__ bool r9v_pair_less(float ka, unsigned int ia,
                                             float kb, unsigned int ib) {
    if (ka < kb) {
        return true;
    }
    if (ka == kb && ia < ib) {
        return true;
    }
    return false;
}
