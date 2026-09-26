#pragma once
#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"

namespace ninfer::ops::detail {

__device__ __forceinline__ int4 int8_kv_dequant_f16x8(const std::int8_t* codes8, __half scale) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    const __half2 s2     = __halves2half2(scale, scale);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 code2 =
            __floats2half2_rn(static_cast<float>(c[2 * i]), static_cast<float>(c[2 * i + 1]));
        const __half2 value2 = __hmul2(code2, s2);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

__device__ __forceinline__ void int8_kv_fast_mma_f16_acc(unsigned& c0, unsigned& c1, unsigned a0,
                                                          unsigned a1, unsigned a2, unsigned a3,
                                                          unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One ldmatrix.trans b16 lane of an INT8 [key][d] tile holds the codes
// {V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]}. Returns the FP16 B-fragment halves
// {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code widened exactly and multiplied
// once by its key's represented group scale.
__device__ __forceinline__ void int8_kv_fast_decode_v_pair(unsigned codes, unsigned scales,
                                                           unsigned& even, unsigned& odd) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte.
    const unsigned biased = codes ^ 0x80808080u;
    unsigned e            = __byte_perm(biased, 0x64646464u, 0x4240);
    unsigned o            = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 s2      = load_vec<__half2>(&scales);
    const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), s2);
    const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), s2);
    even                  = load_vec<unsigned>(&ve);
    odd                   = load_vec<unsigned>(&vo);
}

} // namespace ninfer::ops::detail
