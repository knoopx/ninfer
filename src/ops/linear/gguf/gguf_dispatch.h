#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// ---------------------------------------------------------------------------
// GGUF weight dispatch, sm_120. Strata-only: every weight kernel is a Strata
// kernel; NInfer carries no third-party kernel tree.
//
// Scope
//   Weight payloads are raw GGUF block bytes (QuantLayout::GgufNative); the n axis is
//   contiguous. One fixed pipeline for every LinearPolicy, with the weight
//   kernel chosen by type from two Strata families:
//
//     x BF16 [K,T] --(NInfer-local bf16->f32 pass)--> x_f32 [K,T]
//     then, by weight type:
//       iq route (strata::kernels::iq_supported: ggml 6, 7, 8, 11, 12, 13, 16, 17, 18,
//       20, 21, 22, 23, 29, 42):
//          --strata::kernels::quantize_q8_1_rows--> q8_1 staging [T, K/32] (34 B/block)
//          --strata::kernels::iq_mmvq(weight.ggml_type, ...)--> y_f32 [N,T]
//       native route (strata::kernels::native_mmvq_supported: ggml 2, 6, 7, 8, 11, 12,
//       13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42):
//          --strata::kernels::native_quantize_q8_1--> q8_1 staging [T, K/32] (36 B/block)
//          --strata::kernels::native_mmvq(weight.ggml_type, ...)--> y_f32 [N,T]
//       Q2_K route (gguf_q2_k_mmvq_launch, ported from llama.cpp's ggml-cuda; no Strata kernel):
//          --strata::kernels::native_quantize_q8_1--> the same llama.cpp block_q8_1 staging
//          --gguf_q2_k_mmvq_launch(...)--> y_f32 [N,T], 1..8 activation columns per launch
//     --strata::kernels::f32_to_bf16_bulk--> out BF16 [N,T]
//
//   K must be a multiple of 32 on the iq route and a multiple of the native route's
//   block granularity (256 for Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS, 64 for Q2_0, 32 for
//   Q4_0/Q5_0/Q8_0/IQ4_NL) or the Q2_K route's 256. The native and Q2_K kernels take 1..8
//   activation columns per launch, so a wider product chunks T (the iq route runs up to the
//   caller workspace cap).
//
// Numerical contract
//   Both routes quantize activations to q8_1 blocks (32 values, fp16 scale) and
//   dequantize weights with the unmodified llama.cpp GGUF block layout, so a weight
//   means exactly what it means in llama.cpp. The FP32 staging passes are exact
//   (bf16 is the top half of f32); the only rounding is the activation
//   quantization and the final f32->bf16 output store (round-to-nearest-even,
//   strata::kernels::bf16_bits).
//
// Refuse list
//   GGUF types with no kernel of any route throw std::invalid_argument naming the type:
//   Q4_1, Q8_1, Q8_K, IQ1_S, I8, Q1_0. GGUF F16, F32 and BF16 tensors never
//   reach this dispatch: the artifact reader registers them as QType::GGUF_F16 /
//   QType::FP32 / QType::BF16, which own their own routes.
//
// Workspace
//   The call-scoped planes for one T slice: x_f32 = K*t*4, q8_1 = t*(K/32)*34 (iq route),
//   q8_1_native = t*(K/32)*36 (native route), y_f32 = N*t*4, where t is the slice the dispatch
//   chunks T into (bounded, so a long prefill does not size the planes for every column). The
//   capacity is the union of the planes so it is safe for either route.
// ---------------------------------------------------------------------------

struct GgufWorkspace {
    float*        x_f32         = nullptr;  // K x T FP32 activations
    std::uint8_t* q8_1          = nullptr;  // T x (K/32) q8_1 blocks (34 bytes each), iq route
    std::uint8_t* q8_1_native   = nullptr;  // T x (K/32) q8_1 blocks (36 bytes each), native route
    float*        y_f32         = nullptr;  // N x T FP32 result
};

[[nodiscard]] std::size_t gguf_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                                std::int32_t input_rows,
                                                                LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

void gguf_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream);

// NInfer-local bulk BF16 -> FP32 pass (device kernels in gguf_kernels.cu; the
// per-element conversion is strata::kernels::f32_from_bf16). Not a Strata kernel.
void gguf_bf16_to_f32_launch(const void* x_bf16, float* x_f32, std::int64_t elements,
                             cudaStream_t stream);

// Q2_K vector product for 1..8 activation columns: the one GGUF type neither Strata route covers.
// Ported from llama.cpp's ggml-cuda (vec_dot_q2_K_q8_1 and the mul_mat_vec_q schedule);
// `activation` is the native route's block_q8_1 staging (36 bytes per 32 values) and `weight`
// points at row 0 with rows `row_bytes` apart.
void gguf_q2_k_mmvq_launch(const void* weight, std::int64_t row_bytes, const void* activation,
                           int k, int rows, int columns, float* out, cudaStream_t stream);

} // namespace ninfer::ops::detail
