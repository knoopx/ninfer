#pragma once
#include "core/device.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_gemv.cuh"
#include "ops/linear/ternary/ternary_simt.cuh"
#include "ops/linear/ternary/ternary_gemm_mma.cuh"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Rotation auxiliary layout: a 16-byte header followed by K float sign words
// (validate_ternary_weight pins the header size and the +/−1.0 words).
inline const float* ternary_signs(const Weight& weight) {
    return static_cast<const float*>(weight.rotation) +
           kTernaryPq2RotationHeaderBytes / sizeof(float);
}

template <class Geometry, class Schedule>
void launch_ternary_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const TernaryContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};
    ternary_gemv_kernel<Geometry, Schedule>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata), ternary_signs(weight),
            TernaryIdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int Capacity, class Schedule, bool FullColumns = false>
void launch_ternary_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kRowsPerCta,
                    (x.ne[1] + Capacity - 1) / Capacity);
    const TernaryContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};
    ternary_simt_kernel<Geometry, Capacity, Schedule, TernaryIdentityEpilogue,
                        TernaryContiguousOutput, !FullColumns>
        <<<grid, Schedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                   static_cast<const std::uint8_t*>(weight.qdata),
                                                   ternary_signs(weight), TernaryIdentityEpilogue{},
                                                   output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, template <int> class Schedule, std::size_t... Indices>
constexpr auto ternary_exact_launchers(std::index_sequence<Indices...>) {
    return std::array<TernaryLaunch, sizeof...(Indices)>{
        &launch_ternary_simt<Geometry, First + static_cast<int>(Indices),
                             Schedule<First + static_cast<int>(Indices)>, true>...};
}

// Each shape supplies its measured exact interval and schedule.
template <class Geometry, int First, int Last, template <int> class Schedule>
TernaryLaunch select_ternary_exact(std::int32_t tokens) {
    static constexpr auto launchers =
        ternary_exact_launchers<Geometry, First, Schedule>(
            std::make_index_sequence<Last - First + 1>{});
    return launchers.at(static_cast<std::size_t>(tokens - First));
}

template <int Chunk, TernaryLaunch (*Select)(std::int32_t)>
void launch_ternary_a16_chunks(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += Chunk) {
        const int count = std::min(Chunk, x.ne[1] - offset);
        auto input      = x.slice(1, offset, count);
        auto output     = out.slice(1, offset, count);
        Select(count)(input, weight, output, stream);
    }
}

template <class Geometry, class Schedule>
void launch_ternary_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (x.ne[1] + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const TernaryContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};
    ternary_mma_kernel<Geometry, Schedule, TernaryIdentityEpilogue, TernaryContiguousOutput>
        <<<grid, Schedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                   static_cast<const std::uint8_t*>(weight.qdata),
                                                   ternary_signs(weight), x.ne[1],
                                                   TernaryIdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
