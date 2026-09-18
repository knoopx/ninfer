#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The [16384,5120] single-parent ternary PQ2_0 GDN input projection. Every policy resolves to
// the A16 routes (no A8/A4 route); the routes need no transient workspace.

[[nodiscard]] std::size_t ternary_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens);

void ternary_gdn_input_gemv_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);

void ternary_gdn_input_simt_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   bool full_columns, cudaStream_t stream);

void ternary_gdn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  cudaStream_t stream);

void ternary_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
