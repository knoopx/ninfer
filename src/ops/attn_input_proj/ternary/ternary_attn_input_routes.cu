#include "core/weight.h"
#include "ops/attn_input_proj/ternary/ternary_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/ternary/ternary_attn_input_output.cuh"
#include "ops/linear/ternary/ternary_config.h"
#include "ops/linear/ternary/ternary_gemm_mma.cuh"
#include "ops/linear/ternary/ternary_gemv.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear/ternary/ternary_output.cuh"
#include "ops/linear/ternary/ternary_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {
// The attn_input_proj parent is the [14336,5120] ternary problem; the P3a schedules every
// registered shape shares. Parent rows [0,6144) publish to q, [6144,7168) to k, [7168,13312)
// to the output gate, and [13312,14336) to v (the kernel consumes the stored row order as-is;
// no permutation).
using Geometry     = TernaryN14336K5120;
using GemvSchedule = TernaryGemvSchedule<2, 4, 2>;
using SimtSchedule = TernarySimtSchedule<16, 2, 1>;
using MmaSchedule  = TernaryMmaSchedule<256, 8, 1>;
using Epilogue     = TernaryIdentityEpilogue;
using Output       = TernaryAttnInputOutput;
constexpr int kSimtCapacity = 16;

Output make_output(Tensor& q, Tensor& gate, Tensor& k, Tensor& v) {
    return Output{static_cast<__nv_bfloat16*>(q.data),
                  static_cast<__nv_bfloat16*>(gate.data),
                  static_cast<__nv_bfloat16*>(k.data),
                  static_cast<__nv_bfloat16*>(v.data)};
}

// T=1 decode route: one CTA per GemvSchedule::kRowsPerCta weight rows, one input token.
void launch_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                 Tensor& v, cudaStream_t stream) {
    const Output output = make_output(q, gate, k, v);
    ternary_gemv_kernel<Geometry, GemvSchedule, Epilogue, Output>
        <<<Geometry::kOutputRows / GemvSchedule::kRowsPerCta, GemvSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata), ternary_signs(weight),
            Epilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

// 16-token SIMT route over one chunk: FullColumns pins the live column count to the chunk
// capacity (no runtime compare); the ragged final chunk passes its live count through the
// RuntimeColumns path (the chunk's x.ne[1]).
template <bool FullColumns>
void launch_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                 Tensor& v, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / SimtSchedule::kRowsPerCta,
                    (x.ne[1] + kSimtCapacity - 1) / kSimtCapacity);
    const Output output = make_output(q, gate, k, v);
    ternary_simt_kernel<Geometry, kSimtCapacity, SimtSchedule, Epilogue, Output, !FullColumns>
        <<<grid, SimtSchedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                       static_cast<const std::uint8_t*>(weight.qdata),
                                                       ternary_signs(weight), Epilogue{}, output,
                                                       x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// Large-T route: one CTA per 16-token FWT slab x MmaSchedule::kBlockN rows, full column range.
void launch_mma(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / MmaSchedule::kBlockN,
                    (x.ne[1] + MmaSchedule::kBlockM - 1) / MmaSchedule::kBlockM);
    const Output output = make_output(q, gate, k, v);
    ternary_mma_kernel<Geometry, MmaSchedule, Epilogue, Output>
        <<<grid, MmaSchedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                      static_cast<const std::uint8_t*>(weight.qdata),
                                                      ternary_signs(weight), x.ne[1], Epilogue{},
                                                      output);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void ternary_attn_input_gemv_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_gemv(x, weight, q, gate, k, v, stream);
}

void ternary_attn_input_simt_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, bool full_columns, cudaStream_t stream) {
    if (full_columns) {
        launch_simt<true>(x, weight, q, gate, k, v, stream);
    } else {
        launch_simt<false>(x, weight, q, gate, k, v, stream);
    }
}

void ternary_attn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mma(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
