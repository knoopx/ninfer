#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Ternary PQ2_0 gate/up route (A16-only, P3b). Registered problem: gate_up [34816,5120],
// x [5120,T], out [17408,T]. The weight is the packed ternary block plane plus its per-tensor
// rotation auxiliary; the gate rows [0,17408) precede their up rows [17408,34816) and the
// fused epilogue is silu(gate) * up (the existing op contract). Every policy resolves to the
// A16 routes; there is no A8/A4 route and no transient workspace.
[[nodiscard]] std::size_t ternary_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                         std::int32_t min_tokens,
                                                                         std::int32_t max_tokens);

void ternary_linear_swiglu_gemv_launch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                       cudaStream_t stream);
void ternary_linear_swiglu_simt_launch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                       cudaStream_t stream);
void ternary_linear_swiglu_mma_launch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                      cudaStream_t stream);

void ternary_linear_swiglu_dispatch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                    LinearPolicy policy, WorkspaceArena& workspace,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
