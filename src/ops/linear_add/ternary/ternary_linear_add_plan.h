#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Ternary PQ2_0 residual-add route (A16-only, P3b). Registered problems: [5120,6144],
// [5120,17408], [2048,4096], [2048,6144] (weight [N,K], x [K,T], residual [N,T]). The weight is
// the packed ternary block plane plus its per-tensor rotation auxiliary; the fused epilogue is
// residual += projection (in place on the residual tensor). Every policy resolves to the A16
// routes; there is no A8/A4 route and no transient workspace.
[[nodiscard]] std::size_t ternary_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                      std::int32_t input_rows,
                                                                      LinearPolicy policy,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);

void ternary_linear_add_gemv_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);
void ternary_linear_add_simt_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);
void ternary_linear_add_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);

void ternary_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail
