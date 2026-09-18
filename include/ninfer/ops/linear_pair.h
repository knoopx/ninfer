#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Validates the native weight pair and inclusive T interval. Existing routes use no scratch.
[[nodiscard]] std::size_t linear_pair_workspace_capacity_bytes(const Weight& first_weight,
                                                               const Weight& second_weight,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

/**
 * Computes two same-shaped projections of one input:
 *
 *   first_out[:,t]  = linear(x[:,t], first_weight)
 *   second_out[:,t] = linear(x[:,t], second_weight).
 *
 * `x` is contiguous BF16 [K,T] and both outputs are distinct contiguous BF16 [1024,T]. Both
 * weights have the same logical [1024,K] shape and the same encoding: RowSplit Q8_G32_FP16
 * with FP16 scales, or TERNARY_PQ2_0 TernaryPq2Block with the per-tensor rotation auxiliary.
 * The registry admits the paired [1024,5120] physical projection (two independent ternary
 * weights, each with its own block plane and rotation auxiliary) and the exact adjacent
 * [1024,2048] K/V row views (parent rows [4096,5120) and [5120,6144) of one [6144,2048]
 * parent, sharing the parent's rotation auxiliary) for every positive T. The ternary routes
 * are A16-only and use zero transient storage (plain writing). Numeric semantics are those of
 * linear(). Inputs, outputs, and weight planes must be mutually non-overlapping. The Op
 * requires no caller-owned transient storage and has no persistent state side effect.
 */
void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops
