#include "ops/linear_swiglu/ternary/ternary_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/ternary/ternary_simt.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_swiglu/ternary/ternary_linear_swiglu_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = TernarySwiGluGeometry;
using Schedule = TernarySwiGluSimt;
constexpr std::int32_t kIntermediate = kTernarySwiGluIntermediate;

// Contiguous [N, K/128] 34-byte block plane: one 1024 K block is 8 groups (8*34 bytes in-row).
constexpr std::uint64_t kKBlockBytes =
    Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;

struct SwiGluSimtSharedStorage {
    alignas(16) float fwt_row[kTernaryFwtBlock];
};

// Small-T (2..32) gate/up SIMT route: one 256-thread CTA owns Schedule::kRowsPerCta gate rows
// (and their up counterparts) and up to 16 tokens (the P3a 16-token chunk shape). The 1024-wide
// transform row is shared across tokens; each token is staged, transformed, and multiplied
// against the exact FP32 trit decode for both the gate and up rows before the next token reuses
// the row. Same numerics as the decode route; packed weights stay packed. The fused epilogue is
// silu(gate) * up per (row, token).
template <int Capacity, bool RuntimeColumns>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void ternary_linear_swiglu_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::uint8_t* __restrict__ blocks,
                                       const float* __restrict__ signs,
                                       __nv_bfloat16* __restrict__ out, int columns = Capacity) {
    static_assert(Capacity == Schedule::kTokenTile);
    static_assert((kIntermediate % Schedule::kRowsPerCta) == 0);
    const int live_tokens = RuntimeColumns ? columns : Capacity;

    __shared__ SwiGluSimtSharedStorage shared;
    const int cta_row0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int token0   = static_cast<int>(blockIdx.y) * Capacity;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;

    float accum_gate[Schedule::kRowsPerWarp][Schedule::kTokenTile] = {};
    float accum_up[Schedule::kRowsPerWarp][Schedule::kTokenTile]   = {};

#pragma unroll 1
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        // Weights are token independent: hoist the per-group byte and scale decode (gate and up)
        // out of the token loop.
        std::uint8_t code_bytes[2][Schedule::kRowsPerWarp][Geometry::kGroupsPerKBlock];
        float group_scale[2][Schedule::kRowsPerWarp][Geometry::kGroupsPerKBlock];
#pragma unroll
        for (int which = 0; which < 2; ++which) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::int64_t row =
                    cta_row0 + warp * Schedule::kRowsPerWarp + local_row +
                    (which == 1 ? kIntermediate : 0);
                const std::uint8_t* row_blocks =
                    blocks + row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                    k_block * kKBlockBytes;
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
            stage_ternary_simt_token<Geometry, Schedule>(x, signs, k_block, live,
                                                         shared.fwt_row);
            __syncthreads();
            ternary_fwt_butterfly(shared.fwt_row);
            __syncthreads();
            float* fwt_row = shared.fwt_row;
#pragma unroll
            for (int which = 0; which < 2; ++which) {
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
                    for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
                        const int feature0 = group * kTernaryPq2GroupSize + lane * 4;
                        float partial      = 0.0F;
#pragma unroll
                        for (int slot = 0; slot < 4; ++slot) {
                            const int code =
                                (code_bytes[which][local_row][group] >> (2 * slot)) & 3;
                            partial = fmaf(static_cast<float>(code - 1) *
                                               group_scale[which][local_row][group],
                                           fwt_row[feature0 + slot], partial);
                        }
                        if (which == 0) {
                            accum_gate[local_row][token] += partial;
                        } else {
                            accum_up[local_row][token] += partial;
                        }
                    }
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
                float gate = warp_reduce_sum(accum_gate[local_row][token]);
                float up   = warp_reduce_sum(accum_up[local_row][token]);
                if (lane == 0) {
                    const int parent_row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
                    out[static_cast<std::int64_t>(live) * kIntermediate + parent_row] =
                        __float2bfloat16_rn(silu(gate) * up);
                }
            }
        }
    }
}

} // namespace

void ternary_linear_swiglu_simt_launch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                      cudaStream_t stream) {
    const int tokens = x.ne[1];
    const dim3 grid(kTernarySwiGluIntermediate / Schedule::kRowsPerCta,
                    (tokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile);
    ternary_linear_swiglu_simt_kernel<Schedule::kTokenTile, true>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(gate_up_weight.qdata),
            ternary_signs(gate_up_weight), static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
