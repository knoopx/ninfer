// Implements: include/ninfer/ops/candidate_slice_softmax.h
// Match: wrapper-validated contiguous FP32 [N,C] logits, I32 [N,C] ids/groups, and FP32 [N,C] out.
// Algorithm assumptions: one 256-thread CTA per row; the 256-slot shared pool covers C<=256.
#include "ops/launcher/candidate_slice_softmax.h"

#include "core/device.h"
#include "ops/kernel/candidate_slice_softmax.cuh"

namespace ninfer::ops::detail {

void candidate_slice_softmax_launch(const Tensor& logits, const Tensor& ids, const Tensor& groups,
                                    float temperature, Tensor& out, cudaStream_t stream) {
    (void)ids; // accepted for API symmetry; the kernel only consumes logits, groups, and out.
    const std::int32_t rows    = logits.ne[0];
    const std::int32_t columns = logits.ne[1];
    candidate_slice_softmax_kernel<<<rows, kCandidateSliceBlock, 0, stream>>>(
        static_cast<const float*>(logits.data), static_cast<const std::int32_t*>(groups.data),
        static_cast<float*>(out.data), columns, temperature);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
