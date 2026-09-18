#include "ops/linear_pair/ternary/ternary_pair_plan.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/ternary/ternary_gemm_mma.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_pair/ternary/ternary_pair_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Large-T (T>=33) paired mma route: one 256-thread CTA owns a 16-token FWT slab x
// Schedule::kBlockN weight rows of both weights. Every 1024 K block is cp.async staged as BF16
// into the single 32KB slab, run through ternary_fwt_mma_slab (fwt.cuh) with the weight's own
// sign vector, and multiplied against the packed ternary weights with mma.sync.m16n8k16 (BF16
// operands, FP32 accumulation). Because A(x) = (1/32) * FWT(x * s) is per-tensor, the slab is
// transformed separately for each weight's sign vector: with shared signs (the [1024,2048]
// row-view case) the one transform serves both GEMMs; with independent signs (the paired
// [1024,5120] case) the raw slab is re-staged and re-transformed for the second weight. The
// sequential single-slab design stays under the 48KB static shared limit. Pinned numerics
// match the P3a mma route (pinned FWT chunk rounding, in-register B unpack, FP32 accumulation,
// RN-BF16 output). Packed weights are never materialized.
template <class Geometry, class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void ternary_pair_mma_kernel(const __nv_bfloat16* __restrict__ x,
                             const std::uint8_t* __restrict__ first_blocks,
                             const std::uint8_t* __restrict__ second_blocks,
                             const float* __restrict__ first_signs,
                             const float* __restrict__ second_signs, std::int32_t tokens,
                             __nv_bfloat16* __restrict__ first_out,
                             __nv_bfloat16* __restrict__ second_out) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockN) == 0);
    static_assert((Geometry::kInputRows % kTernaryPq2RotationBlockSize) == 0);
    static_assert((Geometry::kInputRows % 16) == 0);

    alignas(16) __nv_bfloat16 slab[Schedule::kBlockM * kTernaryPq2RotationBlockSize];
    auto* slab_raw = reinterpret_cast<std::uint8_t*>(slab);

    const int token_begin  = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const int row_begin    = static_cast<int>(blockIdx.x) * Schedule::kBlockN;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const bool shared_signs = first_signs == second_signs;

    // ldmatrix A fragment (m16 tokens x 16 features, row-major slab): lanes 0-7 / 8-15 / 16-23
    // / 24-31 address the four 8x8 matrices; the 8 row addresses of a matrix all start at the
    // same column.
    const int a_row_off = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int a_col_off = ((lane >> 4) & 1) * 8;

    // B unpack: the lane's k-pairs of each 16-wide k-step sit in code bytes b_byte and
    // b_byte + 2 of the 128-group, at bit offset b_bit (slots 0-1 for even lanes, 2-3 for odd).
    const int b_byte = (lane & 3) >> 1;
    const int b_bit  = 4 * ((lane & 3) & 1);

    // The per-lane B weight-row block-plane bases are constant across the whole GEMM.
    const std::uint8_t* b_base_first[Schedule::kMmaN];
    const std::uint8_t* b_base_second[Schedule::kMmaN];
#pragma unroll
    for (int nt = 0; nt < Schedule::kMmaN; ++nt) {
        const std::int64_t weight_row =
            row_begin + warp * Schedule::kWarpN + nt * 8 + (lane >> 2);
        b_base_first[nt]  = first_blocks + weight_row * Geometry::kGroupsPerRow *
                                      kTernaryPq2BlockBytes;
        b_base_second[nt] = second_blocks + weight_row * Geometry::kGroupsPerRow *
                                       kTernaryPq2BlockBytes;
    }

    float accum_first[Schedule::kMmaN][4] = {};
    float accum_second[Schedule::kMmaN][4] = {};

    // One GEMM pass against a weight's B plane: 64 m16n8k16 k-steps (16 features each);
    // k-step s of group g decodes group g's 32 code bytes + binary16 scale and builds the B
    // fragment in registers. The A fragment comes from the (already transformed) slab.
    auto gemm_weight = [&](const std::uint8_t* const* b_base, float (*acc)[4], int b) {
#pragma unroll
        for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
            const int global_group = b * Geometry::kGroupsPerKBlock + group;
#pragma unroll
            for (int s = 0; s < Schedule::kK16PerBlock / Geometry::kGroupsPerKBlock; ++s) {
                const int k16 = group * (Schedule::kK16PerBlock / Geometry::kGroupsPerKBlock) + s;
                unsigned a0, a1, a2, a3;
                // The slab holds this k block's 1024 local features (row-major, token-major);
                // the feature offset is k16 * 16 + a_col_off (the b * 1024 offset applies to
                // x, not the local slab).
                ldmatrix_x4(a0, a1, a2, a3,
                             smem_addr(slab + a_row_off * kTernaryPq2RotationBlockSize +
                                       k16 * 16 + a_col_off));
#pragma unroll
                for (int nt = 0; nt < Schedule::kMmaN; ++nt) {
                    const std::uint8_t* block_bytes =
                        b_base[nt] + global_group * kTernaryPq2BlockBytes;
                    const int code_offset = 2 + 4 * s + b_byte;
                    const std::uint8_t byte_lo = block_bytes[code_offset];
                    const std::uint8_t byte_hi = block_bytes[code_offset + 2];
                    const float scale =
                        __half2float(__half(__half_raw{load_vec<std::uint16_t>(block_bytes)}));
                    const int code00 = (byte_lo >> b_bit) & 3;
                    const int code01 = (byte_lo >> (b_bit + 2)) & 3;
                    const int code10 = (byte_hi >> b_bit) & 3;
                    const int code11 = (byte_hi >> (b_bit + 2)) & 3;
                    const __nv_bfloat162 b_lo = __floats2bfloat162_rn(
                        static_cast<float>(code00 - 1) * scale,
                        static_cast<float>(code01 - 1) * scale);
                    const __nv_bfloat162 b_hi = __floats2bfloat162_rn(
                        static_cast<float>(code10 - 1) * scale,
                        static_cast<float>(code11 - 1) * scale);
                    mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3,
                             load_vec<unsigned>(&b_lo), load_vec<unsigned>(&b_hi));
                }
            }
        }
    };

#pragma unroll 1
    for (int b = 0; b < Geometry::kKBlocks; ++b) {
        // (1) cp.async stage the 16 x 1024 BF16 slab.
        stage_ternary_mma_slab<Geometry, Schedule>(x, slab_raw, b, token_begin, tokens);
        __syncthreads();

        // (2) FWT slab transform with the first weight's signs (sign flip + 1/32 fused, pinned
        // chunk rounding).
        ternary_fwt_mma::ternary_fwt_mma_slab<Geometry::kInputRows>(slab, first_signs, b);
        __syncthreads();

        // (3) GEMM against the first weight's B plane.
        gemm_weight(b_base_first, accum_first, b);
        __syncthreads();

        if (!shared_signs) {
            // (4) Independent sign vectors: re-stage the raw slab and re-transform it with the
            // second weight's signs (the one slab is reused sequentially).
            stage_ternary_mma_slab<Geometry, Schedule>(x, slab_raw, b, token_begin, tokens);
            __syncthreads();
            ternary_fwt_mma::ternary_fwt_mma_slab<Geometry::kInputRows>(slab, second_signs, b);
            __syncthreads();
        }

        // (5) GEMM against the second weight's B plane (the shared-signs transform serves both).
        gemm_weight(b_base_second, accum_second, b);
        // The next k block's cp.async reuses the slab; all warps must be done reading it.
        __syncthreads();
    }

    // Epilogue: reuse the slab as the 16 x (kBlockN + 8) BF16 output plane (the +8 padding
    // avoids bank conflicts; only the first kBlockN columns are live), one weight at a time.
    constexpr int kOutputStride = Schedule::kBlockN + 8;
    static_assert(sizeof(slab) >= Schedule::kBlockM * kOutputStride * sizeof(__nv_bfloat16));
    auto* shared_output = reinterpret_cast<__nv_bfloat16*>(slab);
    constexpr int kRows = Geometry::kOutputRows;

    auto store_plane = [&](float (*acc)[4], __nv_bfloat16* out) {
        const int token0 = token_begin + (lane >> 2);
        const int token1 = token0 + 8;
#pragma unroll
        for (int nt = 0; nt < Schedule::kMmaN; ++nt) {
            const int col = warp * Schedule::kWarpN + nt * 8 + 2 * (lane & 3);
            float value00 = acc[nt][0];
            float value01 = acc[nt][1];
            float value10 = acc[nt][2];
            float value11 = acc[nt][3];
            if (token0 < tokens) {
                *reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (lane >> 2) * kOutputStride + col) =
                    __floats2bfloat162_rn(value00, value01);
            }
            if (token1 < tokens) {
                *reinterpret_cast<__nv_bfloat162*>(
                    shared_output + ((lane >> 2) + 8) * kOutputStride + col) =
                    __floats2bfloat162_rn(value10, value11);
            }
        }
        __syncthreads();

        // uint4 store pass over the live kBlockN columns (32 sixteen-byte vectors per row).
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
                store_vec(out + static_cast<std::int64_t>(token) * kRows +
                              (row_begin + vec * 8),
                          values);
            }
        }
        __syncthreads();
    };
    store_plane(accum_first, first_out);
    store_plane(accum_second, second_out);
}

} // namespace

void ternary_pair_mma_launch(const Tensor& x, const Weight& first_weight,
                             const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                             cudaStream_t stream) {
    using Schedule = TernaryPairMma;
    if (first_weight.k == 5120) {
        const dim3 grid(TernaryPairN1024K5120::kOutputRows / Schedule::kBlockN,
                        (x.ne[1] + Schedule::kBlockM - 1) / Schedule::kBlockM);
        ternary_pair_mma_kernel<TernaryPairN1024K5120, Schedule>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(first_weight.qdata),
                static_cast<const std::uint8_t*>(second_weight.qdata),
                ternary_signs(first_weight), ternary_signs(second_weight), x.ne[1],
                static_cast<__nv_bfloat16*>(first_out.data),
                static_cast<__nv_bfloat16*>(second_out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (first_weight.k == 2048) {
        const dim3 grid(TernaryPairN1024K2048::kOutputRows / Schedule::kBlockN,
                        (x.ne[1] + Schedule::kBlockM - 1) / Schedule::kBlockM);
        ternary_pair_mma_kernel<TernaryPairN1024K2048, Schedule>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(first_weight.qdata),
                static_cast<const std::uint8_t*>(second_weight.qdata),
                ternary_signs(first_weight), ternary_signs(second_weight), x.ne[1],
                static_cast<__nv_bfloat16*>(first_out.data),
                static_cast<__nv_bfloat16*>(second_out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    throw std::invalid_argument("ternary linear_pair: unsupported mma shape");
}

} // namespace ninfer::ops::detail
