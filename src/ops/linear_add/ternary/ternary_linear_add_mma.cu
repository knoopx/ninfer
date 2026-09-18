#include "ops/linear_add/ternary/ternary_linear_add_plan.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/ternary/ternary_gemm_mma.cuh"
#include "ops/linear/ternary/ternary_launch.cuh"
#include "ops/linear_add/ternary/ternary_linear_add_config.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Large-T (T>=33) residual-add mma route: one 256-thread CTA owns a 16-token FWT slab x
// Schedule::kBlockN weight rows. Every 1024 K block is cp.async staged as BF16, run through
// ternary_fwt_mma_slab (fwt.cuh), and multiplied against the packed ternary weights with
// mma.sync.m16n8k16 (BF16 operands, FP32 accumulation). The fused epilogue adds the running
// residual in place per (row, token). Pinned numerics match the P3a mma route (pinned FWT
// chunk rounding, in-register B unpack, FP32 accumulation, RN-BF16 output). The P3a kernel
// already stages the in-row K-block offset with the contiguous [N, K/128] block-plane stride
// (8 groups x 34 bytes), so it is instantiated directly with the residual epilogue.
template <class Geometry>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = TernaryAddMma;
    static_assert((Geometry::kOutputRows % Schedule::kBlockN) == 0);

    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (x.ne[1] + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const TernaryAddResidualEpilogue epilogue{
        static_cast<const __nv_bfloat16*>(residual.data), Geometry::kOutputRows};
    const TernaryContiguousOutput output{static_cast<__nv_bfloat16*>(residual.data),
                                        Geometry::kOutputRows};
    ternary_mma_kernel<Geometry, Schedule, TernaryAddResidualEpilogue, TernaryContiguousOutput>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata), ternary_signs(weight), x.ne[1],
            epilogue, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void ternary_linear_add_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    if (weight.n == 5120 && weight.k == 6144) {
        launch_mma<TernaryAddN5120K6144>(x, weight, residual, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 17408) {
        launch_mma<TernaryAddN5120K17408>(x, weight, residual, stream);
        return;
    }
    if (weight.n == 2048 && weight.k == 4096) {
        launch_mma<TernaryAddN2048K4096>(x, weight, residual, stream);
        return;
    }
    if (weight.n == 2048 && weight.k == 6144) {
        launch_mma<TernaryAddN2048K6144>(x, weight, residual, stream);
        return;
    }
    throw std::invalid_argument("ternary linear_add: unsupported mma shape");
}

} // namespace ninfer::ops::detail
