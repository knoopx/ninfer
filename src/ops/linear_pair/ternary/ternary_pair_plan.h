#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Ternary PQ2_0 paired route (A16-only, P3b). Registered problems: the paired [1024,5120]
// projection (two independent ternary weights, each with its own block plane and per-tensor
// rotation auxiliary) and the exact adjacent [1024,2048] K/V row views of one [6144,2048]
// parent (first = parent rows [4096,5120), second = parent rows [5120,6144), sharing the
// parent's rotation auxiliary). x [K,T]; first_out/second_out each [1024,T]. Every route is
// A16; there is no transient workspace.
[[nodiscard]] std::size_t ternary_pair_workspace_capacity_bytes(std::int32_t input_rows,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

void ternary_pair_gemv_launch(const Tensor& x, const Weight& first_weight,
                              const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                              cudaStream_t stream);
void ternary_pair_simt_launch(const Tensor& x, const Weight& first_weight,
                              const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                              cudaStream_t stream);
void ternary_pair_mma_launch(const Tensor& x, const Weight& first_weight,
                             const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                             cudaStream_t stream);

void ternary_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                           Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
