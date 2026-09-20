#pragma once

// ninfer::ops::detail - private launch prototype for candidate_slice_softmax.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void candidate_slice_softmax_launch(const Tensor& logits, const Tensor& ids, const Tensor& groups,
                                    float temperature, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
