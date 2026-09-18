#include "ops/linear_pair/ternary/ternary_pair_plan.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/linear/ternary/ternary_gemv.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_pair/ternary/ternary_pair_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Contiguous [N, K/128] 34-byte block plane: one 1024 K block is 8 groups (8*34 bytes in-row).
template <class Geometry>
constexpr std::uint64_t kKBlockBytes =
    Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;

// T=1 paired decode route: one 256-thread CTA owns Schedule::kRowsPerCta rows of both weights
// (the same row index in first and second) for the single input token. Each 1024 K block is
// staged as `x * sign * (1/32)` into the shared transform row with the weight's own sign
// vector, run through ternary_fwt_butterfly(), and multiplied against the exact FP32 trit
// decode (code - 1) * binary16 scale. Because A(x) = (1/32) * FWT(x * s) is per-tensor, the
// row is transformed once per weight (the two transforms coincide when the sign vectors are
// shared, the [1024,2048] row-view case). Packed weights are never materialized.
template <class Geometry, class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void ternary_pair_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                const std::uint8_t* __restrict__ first_blocks,
                                const std::uint8_t* __restrict__ second_blocks,
                                const float* __restrict__ first_signs,
                                const float* __restrict__ second_signs,
                                __nv_bfloat16* __restrict__ first_out,
                                __nv_bfloat16* __restrict__ second_out) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Schedule::kRowsPerCta % 4) == 0);

    __shared__ alignas(16) float fwt_row[kTernaryFwtBlock];
    const int cta_row0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;

    float accum_first[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    float accum_second[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};

#pragma unroll 1
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        stage_ternary_fwt_row<Geometry, Schedule>(x, first_signs, k_block, fwt_row);
        __syncthreads();
        ternary_fwt_butterfly(fwt_row);
        __syncthreads();
        float* transformed = fwt_row;
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const std::int64_t row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
            const std::uint8_t* row_blocks =
                first_blocks + row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                k_block * kKBlockBytes<Geometry>;
            accumulate_ternary_row<Geometry, Schedule>(row_blocks, transformed, lane,
                                                       accum_first[local_row]);
        }
        __syncthreads();

        stage_ternary_fwt_row<Geometry, Schedule>(x, second_signs, k_block, fwt_row);
        __syncthreads();
        ternary_fwt_butterfly(fwt_row);
        __syncthreads();
        transformed = fwt_row;
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const std::int64_t row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
            const std::uint8_t* row_blocks =
                second_blocks + row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                k_block * kKBlockBytes<Geometry>;
            accumulate_ternary_row<Geometry, Schedule>(row_blocks, transformed, lane,
                                                       accum_second[local_row]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float first_total  = 0.0F;
        float second_total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            first_total += accum_first[local_row][chain];
            second_total += accum_second[local_row][chain];
        }
        first_total  = warp_reduce_sum(first_total);
        second_total = warp_reduce_sum(second_total);
        if (lane == 0) {
            const int parent_row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
            first_out[parent_row]  = __float2bfloat16_rn(first_total);
            second_out[parent_row] = __float2bfloat16_rn(second_total);
        }
    }
}

template <class Geometry>
void launch_decode(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                   Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    constexpr int kBlocks = Geometry::kOutputRows / TernaryPairGemv::kRowsPerCta;
    ternary_pair_decode_kernel<Geometry, TernaryPairGemv>
        <<<kBlocks, TernaryPairGemv::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(first_weight.qdata),
            static_cast<const std::uint8_t*>(second_weight.qdata),
            ternary_signs(first_weight), ternary_signs(second_weight),
            static_cast<__nv_bfloat16*>(first_out.data),
            static_cast<__nv_bfloat16*>(second_out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void ternary_pair_gemv_launch(const Tensor& x, const Weight& first_weight,
                              const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                              cudaStream_t stream) {
    if (first_weight.k == 5120) {
        launch_decode<TernaryPairN1024K5120>(x, first_weight, second_weight, first_out, second_out,
                                             stream);
        return;
    }
    if (first_weight.k == 2048) {
        launch_decode<TernaryPairN1024K2048>(x, first_weight, second_weight, first_out, second_out,
                                             stream);
        return;
    }
    throw std::invalid_argument("ternary linear_pair: unsupported decode shape");
}

} // namespace ninfer::ops::detail
