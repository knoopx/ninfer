#include "ops/linear_add/ternary/ternary_linear_add_plan.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/linear/ternary/ternary_gemv.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_add/ternary/ternary_linear_add_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Contiguous [N, K/128] 34-byte block plane: one 1024 K block is 8 groups (8*34 bytes in-row).
template <class Geometry>
constexpr std::uint64_t kKBlockBytes =
    Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;

// T=1 residual-add decode route: one 256-thread CTA owns Schedule::kRowsPerCta weight rows for
// the single input token. Each 1024 K block is staged as `x * sign * (1/32)` into the shared
// transform row, run through ternary_fwt_butterfly(), and multiplied against the exact FP32 trit
// decode (code - 1) * binary16 scale. The fused epilogue adds the running residual in place.
template <class Geometry, class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void ternary_linear_add_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                    const std::uint8_t* __restrict__ blocks,
                                    const float* __restrict__ signs,
                                    const __nv_bfloat16* __restrict__ residual,
                                    __nv_bfloat16* __restrict__ out) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Schedule::kRowsPerCta % 4) == 0);

    __shared__ alignas(16) float fwt_row[kTernaryFwtBlock];
    const int cta_row0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;

    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};

#pragma unroll 1
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        stage_ternary_fwt_row<Geometry, Schedule>(x, signs, k_block, fwt_row);
        __syncthreads();
        ternary_fwt_butterfly(fwt_row);
        __syncthreads();
        float* transformed = fwt_row;
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const std::int64_t row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
            const std::uint8_t* row_blocks =
                blocks + row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                k_block * kKBlockBytes<Geometry>;
            accumulate_ternary_row<Geometry, Schedule>(row_blocks, transformed, lane,
                                                       accumulators[local_row]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        total = warp_reduce_sum(total);
        if (lane == 0) {
            const int parent_row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
            const float value = total + __bfloat162float(
                                           residual[static_cast<std::int64_t>(parent_row)]);
            out[static_cast<std::int64_t>(parent_row)] = __float2bfloat16_rn(value);
        }
    }
}

} // namespace

void ternary_linear_add_gemv_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    const auto* residual_bf16 = static_cast<const __nv_bfloat16*>(residual.data);
    auto* out_bf16            = static_cast<__nv_bfloat16*>(residual.data);
    switch (weight.n) {
    case 5120:
        if (weight.k == 6144) {
            constexpr int kBlocks = TernaryAddN5120K6144::kOutputRows / TernaryAddGemv::kRowsPerCta;
            ternary_linear_add_gemv_kernel<TernaryAddN5120K6144, TernaryAddGemv>
                <<<kBlocks, TernaryAddGemv::kThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const std::uint8_t*>(weight.qdata), ternary_signs(weight),
                    residual_bf16, out_bf16);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (weight.k == 17408) {
            constexpr int kBlocks = TernaryAddN5120K17408::kOutputRows / TernaryAddGemv::kRowsPerCta;
            ternary_linear_add_gemv_kernel<TernaryAddN5120K17408, TernaryAddGemv>
                <<<kBlocks, TernaryAddGemv::kThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const std::uint8_t*>(weight.qdata), ternary_signs(weight),
                    residual_bf16, out_bf16);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case 2048:
        if (weight.k == 4096) {
            constexpr int kBlocks = TernaryAddN2048K4096::kOutputRows / TernaryAddGemv::kRowsPerCta;
            ternary_linear_add_gemv_kernel<TernaryAddN2048K4096, TernaryAddGemv>
                <<<kBlocks, TernaryAddGemv::kThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const std::uint8_t*>(weight.qdata), ternary_signs(weight),
                    residual_bf16, out_bf16);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (weight.k == 6144) {
            constexpr int kBlocks = TernaryAddN2048K6144::kOutputRows / TernaryAddGemv::kRowsPerCta;
            ternary_linear_add_gemv_kernel<TernaryAddN2048K6144, TernaryAddGemv>
                <<<kBlocks, TernaryAddGemv::kThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const std::uint8_t*>(weight.qdata), ternary_signs(weight),
                    residual_bf16, out_bf16);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    }
    throw std::invalid_argument("ternary linear_add: unsupported gemv shape");
}

} // namespace ninfer::ops::detail
