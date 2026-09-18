#include "ops/linear_pair/ternary/ternary_pair_plan.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/linear/ternary/ternary_simt.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_pair/ternary/ternary_pair_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Contiguous [N, K/128] 34-byte block plane: one 1024 K block is 8 groups (8*34 bytes in-row).
template <class Geometry>
constexpr std::uint64_t kKBlockBytes = Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;

// Small-T (2..32) paired SIMT route: one 256-thread CTA owns Schedule::kRowsPerCta rows of both
// weights (the same row index in first and second) and up to 16 tokens (the P3a 16-token chunk
// shape). The 1024-wide transform row is shared: each token is staged with the weight's own
// sign vector, run through ternary_fwt_butterfly(), and multiplied against the exact FP32 trit
// decode (code - 1) * binary16 scale for both weights before the next token reuses the row
// (the two transforms coincide when the sign vectors are shared, the [1024,2048] row-view case).
// Same numerics as the decode route; the packed weights stay packed.
template <class Geometry, class Schedule, bool RuntimeColumns>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void ternary_pair_simt_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ first_blocks,
                              const std::uint8_t* __restrict__ second_blocks,
                              const float* __restrict__ first_signs,
                              const float* __restrict__ second_signs,
                              __nv_bfloat16* __restrict__ first_out,
                              __nv_bfloat16* __restrict__ second_out,
                              int columns = Schedule::kTokenTile) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    const int live_tokens = RuntimeColumns ? columns : Schedule::kTokenTile;

    __shared__ alignas(16) float fwt_row[kTernaryFwtBlock];
    const int cta_row0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int token0   = static_cast<int>(blockIdx.y) * Schedule::kTokenTile;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t rows = Geometry::kOutputRows;

    float accum_first[Schedule::kRowsPerWarp][Schedule::kTokenTile] = {};
    float accum_second[Schedule::kRowsPerWarp][Schedule::kTokenTile] = {};

#pragma unroll 1
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        // The weights are token independent: hoist the per-group byte and scale decode of both
        // weights out of the token loop.
        std::uint8_t code_bytes[2][Schedule::kRowsPerWarp][Geometry::kGroupsPerKBlock];
        float group_scale[2][Schedule::kRowsPerWarp][Geometry::kGroupsPerKBlock];
#pragma unroll
        for (int which = 0; which < 2; ++which) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::int64_t row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
                const std::uint8_t* plane = which == 0 ? first_blocks : second_blocks;
                const std::uint8_t* row_blocks =
                    plane + row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                    k_block * kKBlockBytes<Geometry>;
#pragma unroll
                for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
                    const std::uint8_t* block = row_blocks + group * kTernaryPq2BlockBytes;
                    code_bytes[which][local_row][group] = block[2 + lane];
                    const std::uint16_t scale_bits = load_vec<std::uint16_t>(block);
                    group_scale[which][local_row][group] =
                        __half2float(__half(__half_raw{scale_bits}));
                }
            }
        }
#pragma unroll
        for (int token = 0; token < Schedule::kTokenTile; ++token) {
            const int live = token0 + token;
            if (live >= live_tokens) { break; }
            stage_ternary_simt_token<Geometry, Schedule>(x, first_signs, k_block, live, fwt_row);
            __syncthreads();
            ternary_fwt_butterfly(fwt_row);
            __syncthreads();
            float* transformed = fwt_row;
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
                for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
                    const int feature0 = group * kTernaryPq2GroupSize + lane * 4;
                    float partial      = 0.0F;
#pragma unroll
                    for (int slot = 0; slot < 4; ++slot) {
                        const int code =
                            (code_bytes[0][local_row][group] >> (2 * slot)) & 3;
                        partial = fmaf(static_cast<float>(code - 1) *
                                           group_scale[0][local_row][group],
                                       transformed[feature0 + slot], partial);
                    }
                    accum_first[local_row][token] += partial;
                }
            }
            __syncthreads();

            stage_ternary_simt_token<Geometry, Schedule>(x, second_signs, k_block, live, fwt_row);
            __syncthreads();
            ternary_fwt_butterfly(fwt_row);
            __syncthreads();
            transformed = fwt_row;
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
                for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
                    const int feature0 = group * kTernaryPq2GroupSize + lane * 4;
                    float partial      = 0.0F;
#pragma unroll
                    for (int slot = 0; slot < 4; ++slot) {
                        const int code =
                            (code_bytes[1][local_row][group] >> (2 * slot)) & 3;
                        partial = fmaf(static_cast<float>(code - 1) *
                                           group_scale[1][local_row][group],
                                       transformed[feature0 + slot], partial);
                    }
                    accum_second[local_row][token] += partial;
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
        for (int token = 0; token < Schedule::kTokenTile; ++token) {
            const int live = token0 + token;
            if (live < live_tokens) {
                float first_total  = warp_reduce_sum(accum_first[local_row][token]);
                float second_total = warp_reduce_sum(accum_second[local_row][token]);
                if (lane == 0) {
                    const int parent_row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
                    const std::int64_t index =
                        static_cast<std::int64_t>(live) * rows + parent_row;
                    first_out[index]  = __float2bfloat16_rn(first_total);
                    second_out[index] = __float2bfloat16_rn(second_total);
                }
            }
        }
    }
}

template <class Geometry>
void launch_simt(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    const dim3 grid(Geometry::kOutputRows / TernaryPairSimt::kRowsPerCta,
                    (tokens + TernaryPairSimt::kTokenTile - 1) / TernaryPairSimt::kTokenTile);
    ternary_pair_simt_kernel<Geometry, TernaryPairSimt, true>
        <<<grid, TernaryPairSimt::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(first_weight.qdata),
            static_cast<const std::uint8_t*>(second_weight.qdata),
            ternary_signs(first_weight), ternary_signs(second_weight),
            static_cast<__nv_bfloat16*>(first_out.data),
            static_cast<__nv_bfloat16*>(second_out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void ternary_pair_simt_launch(const Tensor& x, const Weight& first_weight,
                              const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                              cudaStream_t stream) {
    if (first_weight.k == 5120) {
        launch_simt<TernaryPairN1024K5120>(x, first_weight, second_weight, first_out, second_out,
                                           stream);
        return;
    }
    if (first_weight.k == 2048) {
        launch_simt<TernaryPairN1024K2048>(x, first_weight, second_weight, first_out, second_out,
                                           stream);
        return;
    }
    throw std::invalid_argument("ternary linear_pair: unsupported simt shape");
}

} // namespace ninfer::ops::detail
