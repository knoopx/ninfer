#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The single-parent ternary attention projection parent stores rows in physical order
// query/key/output gate/value ([6144,1024,6144,1024]) while the public output argument order
// is q, gate, k, v; the kernel consumes the stored row order as-is and publishes each segment
// to its own contiguous BF16 output, mirroring Nvfp4GdnInputOutput.
struct TernaryAttnInputOutput {
    static constexpr std::int32_t kQRows    = 6144;
    static constexpr std::int32_t kKvRows   = 1024;
    static constexpr std::int32_t kGateRows = 6144;

    __nv_bfloat16* q;
    __nv_bfloat16* gate;
    __nv_bfloat16* k;
    __nv_bfloat16* v;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        constexpr std::int32_t kKeyBegin   = kQRows;
        constexpr std::int32_t kGateBegin  = kQRows + kKvRows;
        constexpr std::int32_t kValueBegin = kQRows + kKvRows + kGateRows;
        if (parent_row < kQRows) {
            return q + static_cast<std::int64_t>(token) * kQRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return k + static_cast<std::int64_t>(token) * kKvRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return v + static_cast<std::int64_t>(token) * kKvRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert((TernaryAttnInputOutput::kQRows % 128) == 0);
static_assert((TernaryAttnInputOutput::kKvRows % 128) == 0);
static_assert(TernaryAttnInputOutput::kQRows + TernaryAttnInputOutput::kKvRows +
                   TernaryAttnInputOutput::kGateRows + TernaryAttnInputOutput::kKvRows ==
               14336);

} // namespace ninfer::ops::detail
