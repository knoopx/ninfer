#include "ops/linear_swiglu/ternary/ternary_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/ternary/ternary_gemm_mma.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_swiglu/ternary/ternary_linear_swiglu_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = TernarySwiGluGeometry;
using Schedule = TernarySwiGluMma;
constexpr std::int32_t kIntermediate = kTernarySwiGluIntermediate;

// Large-T gate/up mma route: one 256-thread CTA owns a 16-token FWT slab x Schedule::kBlockN
// gate rows (and their up counterparts). Every 1024 K block is cp.async staged as BF16, run
// through ternary_fwt_mma_slab (fwt.cuh), and multiplied against the packed ternary gate and up
// weights with mma.sync.m16n8k16 (BF16 operands, FP32 accumulation). The fused epilogue is
// silu(gate) * up per (row, token). Pinned numerics match the P3a mma route (pinned FWT chunk
// rounding, in-register B unpack, FP32 accumulation, RN-BF16 output). Packed weights are never
// materialized.
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void ternary_linear_swiglu_mma_kernel(const __nv_bfloat16* __restrict__ x,
                                      const std::uint8_t* __restrict__ blocks,
                                      const float* __restrict__ signs, std::int32_t tokens,
                                      __nv_bfloat16* __restrict__ out) {
    static_assert((kIntermediate % Schedule::kBlockN) == 0);
    static_assert((Geometry::kInputRows % kTernaryPq2RotationBlockSize) == 0);
    static_assert((Geometry::kInputRows % 16) == 0);

    alignas(16) __nv_bfloat16 slab[Schedule::kBlockM * kTernaryPq2RotationBlockSize];
    auto* slab_raw = reinterpret_cast<std::uint8_t*>(slab);

    const int token_begin = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const int row_begin   = static_cast<int>(blockIdx.x) * Schedule::kBlockN;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;

    // ldmatrix A fragment (m16 tokens x 16 features, row-major slab).
    const int a_row_off = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int a_col_off = ((lane >> 4) & 1) * 8;

    // B unpack geometry (shared by gate and up).
    const int b_byte = (lane & 3) >> 1;
    const int b_bit  = 4 * ((lane & 3) & 1);

    // Per-lane B weight-row block-plane bases for the gate and up rows (constant across the GEMM).
    const std::uint8_t* b_base_gate[Schedule::kMmaN];
    const std::uint8_t* b_base_up[Schedule::kMmaN];
#pragma unroll
    for (int nt = 0; nt < Schedule::kMmaN; ++nt) {
        const std::int64_t weight_row =
            row_begin + warp * Schedule::kWarpN + nt * 8 + (lane >> 2);
        b_base_gate[nt] = blocks + weight_row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes;
        b_base_up[nt]   = blocks + (weight_row + kIntermediate) * Geometry::kGroupsPerRow *
                                  kTernaryPq2BlockBytes;
    }

    float accum_gate[Schedule::kMmaN][4] = {};
    float accum_up[Schedule::kMmaN][4]   = {};

#pragma unroll 1
    for (int b = 0; b < Geometry::kKBlocks; ++b) {
        stage_ternary_mma_slab<Geometry, Schedule>(x, slab_raw, b, token_begin, tokens);
        __syncthreads();

        ternary_fwt_mma::ternary_fwt_mma_slab<Geometry::kInputRows>(slab, signs, b);
        __syncthreads();

#pragma unroll
        for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
            const int global_group = b * Geometry::kGroupsPerKBlock + group;
#pragma unroll
            for (int s = 0; s < Schedule::kK16PerBlock / Geometry::kGroupsPerKBlock; ++s) {
                const int k16 = group * (Schedule::kK16PerBlock / Geometry::kGroupsPerKBlock) + s;
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3,
                             smem_addr(slab + a_row_off * kTernaryPq2RotationBlockSize +
                                       k16 * 16 + a_col_off));
#pragma unroll
                for (int nt = 0; nt < Schedule::kMmaN; ++nt) {
                    const int code_offset = 2 + 4 * s + b_byte;
                    const std::uint8_t* gb = b_base_gate[nt] + global_group * kTernaryPq2BlockBytes;
                    const std::uint8_t* ub  = b_base_up[nt] + global_group * kTernaryPq2BlockBytes;
                    const std::uint8_t g_lo = gb[code_offset];
                    const std::uint8_t g_hi = gb[code_offset + 2];
                    const std::uint8_t u_lo = ub[code_offset];
                    const std::uint8_t u_hi = ub[code_offset + 2];
                    const float g_scale =
                        __half2float(__half(__half_raw{load_vec<std::uint16_t>(gb)}));
                    const float u_scale =
                        __half2float(__half(__half_raw{load_vec<std::uint16_t>(ub)}));
                    const int g00 = (g_lo >> b_bit) & 3;
                    const int g01 = (g_lo >> (b_bit + 2)) & 3;
                    const int g10 = (g_hi >> b_bit) & 3;
                    const int g11 = (g_hi >> (b_bit + 2)) & 3;
                    const int u00 = (u_lo >> b_bit) & 3;
                    const int u01 = (u_lo >> (b_bit + 2)) & 3;
                    const int u10 = (u_hi >> b_bit) & 3;
                    const int u11 = (u_hi >> (b_bit + 2)) & 3;
                    const __nv_bfloat162 b_g = __floats2bfloat162_rn(
                        static_cast<float>(g00 - 1) * g_scale,
                        static_cast<float>(g01 - 1) * g_scale);
                    const __nv_bfloat162 b_u = __floats2bfloat162_rn(
                        static_cast<float>(u00 - 1) * u_scale,
                        static_cast<float>(u01 - 1) * u_scale);
                    const __nv_bfloat162 b_g_hi = __floats2bfloat162_rn(
                        static_cast<float>(g10 - 1) * g_scale,
                        static_cast<float>(g11 - 1) * g_scale);
                    const __nv_bfloat162 b_u_hi = __floats2bfloat162_rn(
                        static_cast<float>(u10 - 1) * u_scale,
                        static_cast<float>(u11 - 1) * u_scale);
                    mma_bf16(accum_gate[nt][0], accum_gate[nt][1], accum_gate[nt][2],
                             accum_gate[nt][3], a0, a1, a2, a3, load_vec<unsigned>(&b_g),
                             load_vec<unsigned>(&b_g_hi));
                    mma_bf16(accum_up[nt][0], accum_up[nt][1], accum_up[nt][2], accum_up[nt][3],
                             a0, a1, a2, a3, load_vec<unsigned>(&b_u),
                             load_vec<unsigned>(&b_u_hi));
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: reuse the slab as the 16 x (kBlockN + 8) BF16 output plane (the +8 padding avoids
    // bank conflicts; only the first kBlockN columns are live). The fused swiglu value is
    // silu(gate) * up.
    constexpr int kOutputStride = Schedule::kBlockN + 8;
    static_assert(sizeof(slab) >= Schedule::kBlockM * kOutputStride * sizeof(__nv_bfloat16));
    auto* shared_output = reinterpret_cast<__nv_bfloat16*>(slab);
    const int token0 = token_begin + (lane >> 2);
    const int token1 = token0 + 8;
    const int col_base = warp * Schedule::kWarpN + 2 * (lane & 3);
#pragma unroll
    for (int nt = 0; nt < Schedule::kMmaN; ++nt) {
        const int col = col_base + nt * 8;
        float g0 = accum_gate[nt][0];
        float g1 = accum_gate[nt][1];
        float u0 = accum_up[nt][0];
        float u1 = accum_up[nt][1];
        float g2 = accum_gate[nt][2];
        float g3 = accum_gate[nt][3];
        float u2 = accum_up[nt][2];
        float u3 = accum_up[nt][3];
        if (token0 < tokens) {
            g0 = silu(g0) * u0;
            g1 = silu(g1) * u1;
        }
        if (token1 < tokens) {
            g2 = silu(g2) * u2;
            g3 = silu(g3) * u3;
        }
        if (token0 < tokens) {
            *reinterpret_cast<__nv_bfloat162*>(
                shared_output + (lane >> 2) * kOutputStride + col) =
                __floats2bfloat162_rn(g0, g1);
        }
        if (token1 < tokens) {
            *reinterpret_cast<__nv_bfloat162*>(
                shared_output + ((lane >> 2) + 8) * kOutputStride + col) =
                __floats2bfloat162_rn(g2, g3);
        }
    }
    __syncthreads();

    // uint4 store pass over the live kBlockN columns (the output plane is kIntermediate rows).
    constexpr int kOutputVectorsPerRow = Schedule::kBlockN / 8;
    constexpr int kOutputVectors       = Schedule::kBlockM * kOutputVectorsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kOutputVectors;
         task += Schedule::kThreads) {
        const int token_local = task / kOutputVectorsPerRow;
        const int vec         = task - token_local * kOutputVectorsPerRow;
        const int token       = token_begin + token_local;
        if (token < tokens) {
            const uint4 values =
                load_vec<uint4>(shared_output + token_local * kOutputStride + vec * 8);
            const int parent_row = row_begin + vec * 8;
            store_vec(static_cast<__nv_bfloat16*>(out) +
                          static_cast<std::int64_t>(token) * kIntermediate + parent_row,
                      values);
        }
    }
}

} // namespace

void ternary_linear_swiglu_mma_launch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                     cudaStream_t stream) {
    const int tokens = x.ne[1];
    const dim3 grid(kTernarySwiGluIntermediate / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    ternary_linear_swiglu_mma_kernel<<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(gate_up_weight.qdata), ternary_signs(gate_up_weight),
        tokens, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
