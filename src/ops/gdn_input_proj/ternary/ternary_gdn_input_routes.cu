#include "core/weight.h"
#include "ops/gdn_input_proj/ternary/ternary_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/ternary/ternary_gdn_input_output.cuh"
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
// The gdn_input_proj parent is the [16384,5120] ternary problem; the P3a schedules every
// registered shape shares. Parent rows 0..10239 publish to qkv, 10240..16383 to z (the kernel
// consumes the stored row order as-is; no permutation).
using Geometry     = TernaryN16384K5120;
using GemvSchedule = TernaryGemvSchedule<2, 4, 2>;
using SimtSchedule = TernarySimtSchedule<16, 2, 1>;
using MmaSchedule  = TernaryMmaSchedule<256, 8, 1>;
using Epilogue     = TernaryIdentityEpilogue;
using Output       = TernaryGdnInputOutput;
constexpr int kSimtCapacity = 16;

Output make_output(Tensor& qkv, Tensor& z) {
    return Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
}

// T=1 decode route: one CTA per GemvSchedule::kRowsPerCta weight rows, one input token.
void launch_gemv(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                 cudaStream_t stream) {
    const Output output = make_output(qkv, z);
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
void launch_simt(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                 cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / SimtSchedule::kRowsPerCta,
                    (x.ne[1] + kSimtCapacity - 1) / kSimtCapacity);
    const Output output = make_output(qkv, z);
    ternary_simt_kernel<Geometry, kSimtCapacity, SimtSchedule, Epilogue, Output, !FullColumns>
        <<<grid, SimtSchedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                       static_cast<const std::uint8_t*>(weight.qdata),
                                                       ternary_signs(weight), Epilogue{}, output,
                                                       x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// Large-T route: one CTA per 16-token FWT slab x MmaSchedule::kBlockN rows, full column range.
void launch_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / MmaSchedule::kBlockN,
                    (x.ne[1] + MmaSchedule::kBlockM - 1) / MmaSchedule::kBlockM);
    const Output output = make_output(qkv, z);
    ternary_mma_kernel<Geometry, MmaSchedule, Epilogue, Output>
        <<<grid, MmaSchedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                      static_cast<const std::uint8_t*>(weight.qdata),
                                                      ternary_signs(weight), x.ne[1], Epilogue{},
                                                      output);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void ternary_gdn_input_gemv_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    launch_gemv(x, weight, qkv, z, stream);
}

void ternary_gdn_input_simt_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   bool full_columns, cudaStream_t stream) {
    if (full_columns) {
        launch_simt<true>(x, weight, qkv, z, stream);
    } else {
        launch_simt<false>(x, weight, qkv, z, stream);
    }
}

void ternary_gdn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  cudaStream_t stream) {
    launch_mma(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
