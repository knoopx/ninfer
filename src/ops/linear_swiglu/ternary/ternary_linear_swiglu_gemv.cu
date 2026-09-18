#include "ops/linear_swiglu/ternary/ternary_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/ternary/ternary_gemv.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_swiglu/ternary/ternary_linear_swiglu_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = TernarySwiGluGeometry;
using Schedule = TernarySwiGluGemv;
constexpr std::int32_t kIntermediate = kTernarySwiGluIntermediate;

// The block plane is the contiguous interleaved [N, K/128] 34-byte block array (spec 2,
// tools/artifact/codecs/ternary_pq2.py): group g of row n sits at (n*groups+g)*34. One
// 1024-wide K block is 8 groups, so the in-row K-block stride is 8*34 bytes.
constexpr std::uint64_t kKBlockBytes =
    Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;

struct SwiGluSharedStorage {
    alignas(16) float fwt_row[kTernaryFwtBlock];
};

// T=1 gate/up decode route: one 256-thread CTA owns Schedule::kRowsPerCta gate rows (and their
// up counterparts) for the single input token. Each 1024 K block is staged as
// `x * sign * (1/32)` into the shared transform row, run through ternary_fwt_butterfly(), and
// multiplied against the exact FP32 trit decode for both the gate and up rows. The fused
// epilogue is silu(gate) * up. Packed weights are never materialized.
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void ternary_linear_swiglu_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::uint8_t* __restrict__ blocks,
                                       const float* __restrict__ signs,
                                       __nv_bfloat16* __restrict__ out) {
    static_assert((kIntermediate % Schedule::kRowsPerCta) == 0);
    static_assert((Schedule::kRowsPerCta % 4) == 0);

    __shared__ SwiGluSharedStorage shared;
    const int cta_row0 = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;

    float accum_gate[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    float accum_up[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains]   = {};

#pragma unroll 1
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        stage_ternary_fwt_row<Geometry, Schedule>(x, signs, k_block, shared.fwt_row);
        __syncthreads();
        ternary_fwt_butterfly(shared.fwt_row);
        __syncthreads();
        float* fwt_row = shared.fwt_row;
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const std::int64_t row   = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
            const std::uint8_t* gate =
                blocks + row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes +
                k_block * kKBlockBytes;
            const std::uint8_t* up =
                blocks + (row + kIntermediate) * Geometry::kGroupsPerRow *
                             kTernaryPq2BlockBytes +
                k_block * kKBlockBytes;
            accumulate_ternary_row<Geometry, Schedule>(gate, fwt_row, lane,
                                                       accum_gate[local_row]);
            accumulate_ternary_row<Geometry, Schedule>(up, fwt_row, lane,
                                                       accum_up[local_row]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float gate = 0.0F;
        float up   = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            gate += accum_gate[local_row][chain];
            up   += accum_up[local_row][chain];
        }
        gate = warp_reduce_sum(gate);
        up   = warp_reduce_sum(up);
        if (lane == 0) {
            const int parent_row = cta_row0 + warp * Schedule::kRowsPerWarp + local_row;
            out[static_cast<std::int64_t>(parent_row)] =
                __float2bfloat16_rn(silu(gate) * up);
        }
    }
}

} // namespace

void ternary_linear_swiglu_gemv_launch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                      cudaStream_t stream) {
    constexpr std::int32_t kBlocks = kTernarySwiGluIntermediate / Schedule::kRowsPerCta;
    ternary_linear_swiglu_gemv_kernel<<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(gate_up_weight.qdata), ternary_signs(gate_up_weight),
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
