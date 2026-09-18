#include "ops/linear_add/ternary/ternary_linear_add_plan.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/linear/ternary/ternary_simt.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_add/ternary/ternary_linear_add_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
constexpr std::uint64_t kKBlockBytes = Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;

// Small-T (2..32) residual-add SIMT route: one 256-thread CTA owns Schedule::kRowsPerCta weight
// rows and up to 16 tokens (the P3a 16-token chunk shape). The 1024-wide transform row is shared
// across tokens; each token is staged, transformed, and multiplied against the exact FP32 trit
// decode before the next token reuses the row. The fused epilogue adds the running residual in
// place per (row, token). Packed weights stay packed.
template <class Geometry, class Schedule, bool RuntimeColumns>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void ternary_linear_add_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                    const std::uint8_t* __restrict__ blocks,
                                    const float* __restrict__ signs,
                                    const __nv_bfloat16* __restrict__ residual,
                                    __nv_bfloat16* __restrict__ out, int columns = Schedule::kTokenTile) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    const int live_tokens = RuntimeColumns ? columns : Schedule::kTokenTile;

    __shared__ alignas(16) float fwt_row[kTernaryFwtBlock];
    const int cta_row0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int token0   = static_cast<int>(blockIdx.y) * Schedule::kTokenTile;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t rows = Geometry::kOutputRows;

    float accumulators[Schedule::kRowsPerWarp][Schedule::kTokenTile] = {};

#pragma unroll 1
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        std::uint8_t code_bytes[Schedule::kRowsPerWarp][Geometry::kGroupsPerKBlock];
        float group_scale[Schedule::kRowsPerWarp][Geometry::kGroupsPerKBlock];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const std::int64_t row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
            const std::uint8_t* row_blocks =
                blocks + row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                k_block * kKBlockBytes<Geometry>;
#pragma unroll
            for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
                const std::uint8_t* block = row_blocks + group * kTernaryPq2BlockBytes;
                code_bytes[local_row][group] = block[2 + lane];
                const std::uint16_t scale_bits = load_vec<std::uint16_t>(block);
                group_scale[local_row][group] = __half2float(__half(__half_raw{scale_bits}));
            }
        }
#pragma unroll
        for (int token = 0; token < Schedule::kTokenTile; ++token) {
            const int live = token0 + token;
            if (live >= live_tokens) { break; }
            stage_ternary_simt_token<Geometry, Schedule>(x, signs, k_block, live, fwt_row);
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
                            (code_bytes[local_row][group] >> (2 * slot)) & 3;
                        partial = fmaf(static_cast<float>(code - 1) *
                                           group_scale[local_row][group],
                                       transformed[feature0 + slot], partial);
                    }
                    accumulators[local_row][token] += partial;
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
                float total = warp_reduce_sum(accumulators[local_row][token]);
                if (lane == 0) {
                    const int parent_row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
                    const std::int64_t index =
                        static_cast<std::int64_t>(live) * rows + parent_row;
                    out[index] = __float2bfloat16_rn(
                        total + __bfloat162float(residual[index]));
                }
            }
        }
    }
}

template <class Geometry>
void launch_simt(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    const int tokens = x.ne[1];
    const dim3 grid(Geometry::kOutputRows / TernaryAddSimt::kRowsPerCta,
                    (tokens + TernaryAddSimt::kTokenTile - 1) / TernaryAddSimt::kTokenTile);
    ternary_linear_add_simt_kernel<Geometry, TernaryAddSimt, true>
        <<<grid, TernaryAddSimt::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata), ternary_signs(weight),
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<__nv_bfloat16*>(residual.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void ternary_linear_add_simt_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    if (weight.n == 5120 && weight.k == 6144) {
        launch_simt<TernaryAddN5120K6144>(x, weight, residual, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 17408) {
        launch_simt<TernaryAddN5120K17408>(x, weight, residual, stream);
        return;
    }
    if (weight.n == 2048 && weight.k == 4096) {
        launch_simt<TernaryAddN2048K4096>(x, weight, residual, stream);
        return;
    }
    if (weight.n == 2048 && weight.k == 6144) {
        launch_simt<TernaryAddN2048K6144>(x, weight, residual, stream);
        return;
    }
    throw std::invalid_argument("ternary linear_add: unsupported simt shape");
}

} // namespace ninfer::ops::detail
