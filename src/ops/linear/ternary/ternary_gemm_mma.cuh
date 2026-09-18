#pragma once

#include "ops/linear/ternary/ternary_config.h"
#include "ops/linear/ternary/ternary_output.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/fwt.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Large-T mma route: one 256-thread CTA owns a 16-token FWT slab x Schedule::kBlockN weight
// rows. Every 1024-wide K block is cp.async staged as BF16, run through
// ternary_fwt_mma::ternary_fwt_mma_slab (fwt.cuh), and multiplied against the packed ternary
// weights with mma.sync.m16n8k16 (BF16 operands, FP32 accumulation).
//
// Pinned numerics:
//   - A operand: the BF16 FWT slab. Its rounding is the pinned chunk rule of fwt.cuh (the sign
//     flip and the 1/32 scale fused into the operand load with RN, FP32 accumulation within a
//     16x16 chunk, one RN-BF16 conversion at the chunk boundary).
//   - B operand: no BF16 weight materialization. Each lane builds its mma B fragment in
//     registers from the 2-bit code slots and the binary16 group scale of the 34-byte block:
//     v = (code - 1) * scale computed in exact FP32, converted to BF16 with RN
//     (__floats2bfloat162_rn).
//   - the mma accumulation is FP32; the epilogue converts the final accumulators to BF16 with
//     RN.
//
// The 32KB static slab is reused as the 16 x (kBlockN + 8) BF16 output plane after the GEMM
// (16 x 264 x 2 = 8.4KB <= 32KB, mirroring nvfp4_w4a4_mma). Only the first kBlockN columns of
// the plane are live; the +8 padding avoids shared-memory bank conflicts.

// Stages one 16 x 1024 BF16 slab from `x` (2048 sixteen-byte vectors, 8 per thread). Padded
// token rows copy zero bytes from a clamped valid source row.
template <class Geometry, class Schedule>
__device__ __forceinline__ void stage_ternary_mma_slab(const __nv_bfloat16* __restrict__ x,
                                                        std::uint8_t* __restrict__ slab_raw,
                                                        int k_block, int token_begin,
                                                        int tokens) {
    constexpr int kVectorsPerSlabRow = kTernaryPq2RotationBlockSize * 2 / 16;  // 128
    constexpr int kSlabVectors       = Schedule::kBlockM * kVectorsPerSlabRow;
    constexpr int kLoadsPerThread    = kSlabVectors / Schedule::kThreads;
    const int tid                    = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < kLoadsPerThread; ++i) {
        const int task  = i * Schedule::kThreads + tid;
        const int row   = task / kVectorsPerSlabRow;
        const int vec   = task - row * kVectorsPerSlabRow;
        const int token = token_begin + row;
        const auto* source =
            x + static_cast<std::int64_t>(token < tokens ? token : 0) * Geometry::kInputRows +
            k_block * kTernaryPq2RotationBlockSize + vec * 8;
        cp_async_zfill<16, Cache::cg>(slab_raw + task * 16, source, token < tokens ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
}

template <class Geometry, class Schedule, class Epilogue, class Output>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
ternary_mma_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ blocks,
                   const float* __restrict__ signs, std::int32_t tokens, Epilogue epilogue,
                   Output output) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockN) == 0);
    static_assert((Geometry::kInputRows % kTernaryPq2RotationBlockSize) == 0);
    static_assert((Geometry::kInputRows % 16) == 0);

    alignas(16) __nv_bfloat16 slab[Schedule::kBlockM * kTernaryPq2RotationBlockSize];
    auto* slab_raw = reinterpret_cast<std::uint8_t*>(slab);

    const int token_begin = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const int row_begin   = static_cast<int>(blockIdx.x) * Schedule::kBlockN;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;

    // ldmatrix A fragment (m16 tokens x 16 features, row-major slab): lanes 0-7 / 8-15 / 16-23
    // / 24-31 address the four 8x8 matrices (m 0-7 x k 0-7, m 8-15 x k 0-7, m 0-7 x k 8-15,
    // m 8-15 x k 8-15); the 8 row addresses of a matrix all start at the same column.
    const int a_row_off = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int a_col_off = ((lane >> 4) & 1) * 8;

    // B unpack: the lane's k-pairs are {2*(lane&3), 2*(lane&3)+1} (b0) and {8 + 2*(lane&3),
    // 8 + 2*(lane&3)+1} (b1) of each 16-wide k-step; the pair sits in code bytes (lane&3)/2
    // and 2 + (lane&3)/2 of the 128-group, at bit offset 4 * ((lane&3) & 1) (slots 0-1 for
    // even lanes, slots 2-3 for odd lanes).
    const int b_byte = (lane & 3) >> 1;
    const int b_bit  = 4 * ((lane & 3) & 1);

    // The per-lane B weight-row block-plane bases are constant across the whole GEMM.
    const std::uint8_t* b_base[Schedule::kMmaN];
#pragma unroll
    for (int nt = 0; nt < Schedule::kMmaN; ++nt) {
        const std::int64_t weight_row =
            row_begin + warp * Schedule::kWarpN + nt * 8 + (lane >> 2);
        b_base[nt] = blocks + weight_row * Geometry::kGroupsPerRow * kTernaryPq2BlockBytes;
    }

    float accumulators[Schedule::kMmaN][4] = {};

#pragma unroll 1
    for (int b = 0; b < Geometry::kKBlocks; ++b) {
        // (1) cp.async stage the 16 x 1024 BF16 slab.
        stage_ternary_mma_slab<Geometry, Schedule>(x, slab_raw, b, token_begin, tokens);
        __syncthreads();

        // (2) FWT slab transform (sign flip + 1/32 fused, pinned chunk rounding).
        ternary_fwt_mma::ternary_fwt_mma_slab<Geometry::kInputRows>(slab, signs, b);
        __syncthreads();

        // (3) GEMM: 64 m16n8k16 k-steps (16 features each); k-step s of group g decodes
        // group g's 32 code bytes + binary16 scale and builds the B fragment in registers.
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
                    mma_bf16(accumulators[nt][0], accumulators[nt][1], accumulators[nt][2],
                             accumulators[nt][3], a0, a1, a2, a3,
                             load_vec<unsigned>(&b_lo), load_vec<unsigned>(&b_hi));
                }
            }
        }
        // The next k block's cp.async reuses the slab; all warps must be done reading it.
        __syncthreads();
    }

    // Epilogue: reuse the slab region as the 16 x (kBlockN + 8) BF16 output plane (the +8
    // padding avoids bank conflicts; only the first kBlockN columns are live).
    constexpr int kOutputStride = Schedule::kBlockN + 8;
    static_assert(sizeof(slab) >= Schedule::kBlockM * kOutputStride * sizeof(__nv_bfloat16));
    auto* shared_output = reinterpret_cast<__nv_bfloat16*>(slab);
    const int token0 = token_begin + (lane >> 2);
    const int token1 = token0 + 8;
#pragma unroll
    for (int nt = 0; nt < Schedule::kMmaN; ++nt) {
        const int col = warp * Schedule::kWarpN + nt * 8 + 2 * (lane & 3);
        float value00 = accumulators[nt][0];
        float value01 = accumulators[nt][1];
        float value10 = accumulators[nt][2];
        float value11 = accumulators[nt][3];
        if (token0 < tokens) {
            value00 = epilogue.apply(row_begin + col, token0, value00);
            value01 = epilogue.apply(row_begin + col + 1, token0, value01);
        }
        if (token1 < tokens) {
            value10 = epilogue.apply(row_begin + col, token1, value10);
            value11 = epilogue.apply(row_begin + col + 1, token1, value11);
        }
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
            output.store_vector(row_begin + vec * 8, token, values);
        }
    }
}

} // namespace ninfer::ops::detail
