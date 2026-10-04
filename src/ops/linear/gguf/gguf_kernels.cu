#include "ops/linear/gguf/gguf_dispatch.h"

#include "core/device.h"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
// NInfer-local pass (not a Strata kernel): bulk BF16 -> FP32 in 8-element vector
// loads. The per-element conversion is strata::kernels::f32_from_bf16 (exact: bf16
// is the top half of f32); only the vectorized pass shape is local.
#if defined(__CUDACC__)
__global__ void gguf_bf16_to_f32_kernel(const uint4* __restrict__ src, float* __restrict__ dst,
                                        std::int64_t vectors) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vectors) {
        return;
    }
    const uint4 raw = src[i];
    const auto* halves = reinterpret_cast<const std::uint16_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        dst[i * 8 + j] = strata::kernels::f32_from_bf16(halves[j]);
    }
}
#endif  // defined(__CUDACC__)

// ---------------------------------------------------------------------------
// Q2_K vector product, ported from llama.cpp's ggml-cuda (the vec_dot_q2_K_q8_1 reduction and the
// mul_mat_vec_q schedule, as carried by the gsq-rco bridge). Q2_K is the one GGUF type neither
// Strata route covers, so the engine carries this kernel locally.
//
// A warp owns two rows; lane l takes the 32-value slices l, l + 32, ... of each. Every slice of a
// row is decoded once into eight int8x4 words in value order (word i holds values 4i..4i+3) plus
// the slice's (scale, min) factors, then dotted with that slice of every activation column through
// __dp4a. The activation is llama.cpp's block_q8_1 staging the native route already produces:
// 36 bytes per 32 values, an fp16 d and the fp16 sum of the 32 values followed by 32 int8.
//
// The 84-byte weight supers-block covers 256 values: 16 packed 4-bit (scale, min) pairs, 64 bytes
// of 2-bit quants, then fp16 d and dmin. One slice contributes
// d_act * (d*sc*(w.a) - dmin*m*sum(a)), with the min term using the exact integer sum of the
// activation's quantized values.
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)
constexpr int kQ2KBlockElems  = 256;
constexpr int kQ2KBlockBytes  = 84;
constexpr int kQ2KBlockSlices = kQ2KBlockElems / 32;  // eight lanes' sub-blocks per block
constexpr int kQ2KRowsPerWarp = 2;
constexpr int kQ2KWarps       = 4;
constexpr int kQ2KColumns     = 8;

struct Q2KSlice {
    int w[8];  // int8x4 words: word i holds values 4i..4i+3
    float d_sc_lo, d_sc_hi, dmin_m_lo, dmin_m_hi;
};

__device__ __forceinline__ void decode_q2_k(const std::uint8_t* b, int sub, Q2KSlice& s) {
    const int half_index = sub / 4, pair = sub % 4;
    const float2 dm      = __half22float2(*reinterpret_cast<const half2*>(b + 80));
    const std::uint32_t lo = b[8 * half_index + 2 * pair];
    const std::uint32_t hi = b[8 * half_index + 2 * pair + 1];
    s.d_sc_lo              = dm.x * static_cast<float>(lo & 0xF);
    s.d_sc_hi              = dm.x * static_cast<float>(hi & 0xF);
    s.dmin_m_lo            = dm.y * static_cast<float>(lo >> 4);
    s.dmin_m_hi            = dm.y * static_cast<float>(hi >> 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s.w[i] = int((*reinterpret_cast<const std::uint32_t*>(b + 16 + 32 * half_index + 4 * i) >>
                      (2 * pair)) &
                     0x03030303u);
    }
}

__device__ __forceinline__ float dot_q2_k(const Q2KSlice& s, const int* a, float d) {
    int lo = 0, hi = 0, sum_lo = 0, sum_hi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        lo     = __dp4a(s.w[i], a[i], lo);
        hi     = __dp4a(s.w[i + 4], a[i + 4], hi);
        sum_lo = __dp4a(a[i], 0x01010101, sum_lo);
        sum_hi = __dp4a(a[i + 4], 0x01010101, sum_hi);
    }
    return d * (s.d_sc_lo * static_cast<float>(lo) + s.d_sc_hi * static_cast<float>(hi) -
                s.dmin_m_lo * static_cast<float>(sum_lo) -
                s.dmin_m_hi * static_cast<float>(sum_hi));
}

__global__ void gguf_q2_k_mmvq_kernel(const std::uint8_t* __restrict__ weight,
                                      std::int64_t row_bytes,
                                      const std::uint8_t* __restrict__ activation, int k, int rows,
                                      int columns, float* __restrict__ out) {
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int slices = k / 32;
    const int sub    = lane % kQ2KBlockSlices;
    const std::int64_t lane_offset = (lane / kQ2KBlockSlices) * kQ2KBlockBytes;

    for (int row0 = (blockIdx.x * kQ2KWarps + warp) * kQ2KRowsPerWarp; row0 < rows;
         row0 += gridDim.x * kQ2KWarps * kQ2KRowsPerWarp) {
        const std::uint8_t* rowp[kQ2KRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kQ2KRowsPerWarp; ++r) {
            rowp[r] = weight + std::int64_t(min(row0 + r, rows - 1)) * row_bytes + lane_offset;
        }
        float acc[kQ2KRowsPerWarp][kQ2KColumns];
#pragma unroll
        for (int r = 0; r < kQ2KRowsPerWarp; ++r) {
#pragma unroll
            for (int j = 0; j < kQ2KColumns; ++j) { acc[r][j] = 0.0f; }
        }
        for (int s = lane; s < slices; s += 32) {
            int a[kQ2KColumns][8];
            float d[kQ2KColumns];
#pragma unroll
            for (int j = 0; j < kQ2KColumns; ++j) {
                if (j >= columns) { continue; }
                const auto* blk = activation + (std::int64_t(j) * slices + s) * 36;
                d[j]            = __half2float(*reinterpret_cast<const half*>(blk));
                const auto* q   = reinterpret_cast<const int*>(blk + 4);
#pragma unroll
                for (int w = 0; w < 8; ++w) { a[j][w] = q[w]; }
            }
#pragma unroll
            for (int r = 0; r < kQ2KRowsPerWarp; ++r) {
                Q2KSlice sl;
                decode_q2_k(rowp[r] + std::int64_t(s / kQ2KBlockSlices) * kQ2KBlockBytes, sub,
                            sl);
#pragma unroll
                for (int j = 0; j < kQ2KColumns; ++j) {
                    if (j >= columns) { continue; }
                    acc[r][j] += dot_q2_k(sl, a[j], d[j]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kQ2KRowsPerWarp; ++r) {
#pragma unroll
            for (int j = 0; j < kQ2KColumns; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < kQ2KRowsPerWarp; ++r) {
                if (row0 + r >= rows) { continue; }
#pragma unroll
                for (int j = 0; j < kQ2KColumns; ++j) {
                    if (j >= columns) { continue; }
                    out[std::int64_t(j) * rows + (row0 + r)] = acc[r][j];
                }
            }
        }
    }
}
#endif  // defined(__CUDACC__)

} // namespace

void gguf_q2_k_mmvq_launch(const void* weight, std::int64_t row_bytes, const void* activation,
                           int k, int rows, int columns, float* out, cudaStream_t stream) {
    if (columns < 1 || columns > kQ2KColumns) {
        throw std::invalid_argument("gguf q2_k: columns must be 1..8");
    }
    if (k <= 0 || k % kQ2KBlockElems != 0) {
        throw std::invalid_argument("gguf q2_k: K must be a positive multiple of 256");
    }
    if (rows <= 0 || row_bytes < static_cast<std::int64_t>(k / kQ2KBlockElems) * kQ2KBlockBytes) {
        throw std::invalid_argument("gguf q2_k: row is shorter than its blocks");
    }
    constexpr int kThreads = kQ2KWarps * 32;
    const int groups = (rows + kQ2KRowsPerWarp * kQ2KWarps - 1) / (kQ2KRowsPerWarp * kQ2KWarps);
#if defined(__CUDACC__)
    gguf_q2_k_mmvq_kernel<<<groups, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight), row_bytes,
        static_cast<const std::uint8_t*>(activation), k, rows, columns, out);
#else
    (void)weight;
    (void)row_bytes;
    (void)activation;
    (void)k;
    (void)rows;
    (void)columns;
    (void)out;
    (void)stream;
    (void)groups;
#endif
    CUDA_CHECK(cudaGetLastError());
}

void gguf_bf16_to_f32_launch(const void* x_bf16, float* x_f32, std::int64_t elements,
                             cudaStream_t stream) {
    if (elements <= 0) {
        throw std::invalid_argument("gguf linear: K*T must be positive");
    }
    if (elements % 8 != 0) {
        throw std::invalid_argument("gguf linear: K*T must be a multiple of 8");
    }
    constexpr int kThreads = 256;
    const std::int64_t vectors = elements / 8;
#if defined(__CUDACC__)
    gguf_bf16_to_f32_kernel
        <<<static_cast<unsigned>((vectors + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
            static_cast<const uint4*>(x_bf16), x_f32, vectors);
#else
    // Host-side syntax check only: the device launch (<<<>>> and blockIdx/
    // threadIdx) is nvcc's job in the build; g++ -fsyntax-only cannot parse it.
    (void)x_bf16;
    (void)x_f32;
    (void)vectors;
    (void)stream;
#endif
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
