#pragma once

// Integer-activation routes for PQ2_0 row-split weights (the codes are exact int8 {-1, 0, +1};
// the activations are quantised to int8 with one scale per token and 64- or 128-wide group): the
// small-T kernel of ternary_small_t_i8.cuh for T in [kTernaryI8SmallMinTokens, kTernaryI8SmallMaxTokens], and
// above it, from ternary_a8_min_tokens() up, the tile GEMM of ternary_prefill_i8.cuh, which pads T to a
// multiple of 64 internally (NINFER_PQ2_A8_TILE=off: the shared int8 GEMM of
// ops/common/rowsplit_a8_mma.cuh with the ternary codec).

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kTernaryA8MinTokens = 64;
// The widths the small-T kernel takes by default; NINFER_PQ2_I8_SMALL=off|lo,hi overrides them.
// Against the 128x64 prefill tile the small-T launches stop paying from 96 columns on the RTX 3090
// (a GDN layer's four projections: 790 vs 918 us at 96, 1091 vs 1840 us at 192; even at 64).
inline constexpr std::int32_t kTernaryI8SmallMinTokens = 1;
inline constexpr std::int32_t kTernaryI8SmallMaxTokens = 64;
// kTernaryA8MinTokens, or NINFER_PQ2_A8_MIN when it is set (benchmark A/B of the crossover).
[[nodiscard]] std::int32_t ternary_a8_min_tokens();

[[nodiscard]] bool ternary_a8_admits(LinearPolicy policy);
[[nodiscard]] bool ternary_a8_shape_supported(std::int32_t output_rows, std::int32_t input_rows);
[[nodiscard]] bool ternary_a8_supported(const Weight& w, std::int32_t tokens);
// Activation planes for T in [1, max_tokens]; zero when no admitted call can take the route.
[[nodiscard]] std::size_t ternary_a8_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                LinearPolicy policy, std::int32_t max_tokens);

// Bytes of one quantised activation plane pair for T up to max_tokens.
[[nodiscard]] std::size_t ternary_a8_activation_bytes(std::int32_t input_rows, std::int32_t max_tokens);

// Records into `layout` exactly the planes ternary_a8_quantize allocates for `tokens`; false when no
// integer route takes that width.
bool ternary_a8_layout_activations(WorkspaceLayoutBuilder& layout, std::int32_t input_rows,
                              std::int32_t tokens);

// x quantised once for several PQ2 GEMMs over the same activations; the planes live in the caller's
// workspace scope.
struct TernaryA8Activations {
    const std::int8_t* codes;
    const __half* scales;
    std::int32_t tokens;
    std::int32_t input_rows;
};

[[nodiscard]] TernaryA8Activations ternary_a8_quantize(const Tensor& x, WorkspaceArena& workspace,
                                             cudaStream_t stream);
// One pass over a parent weight: rows [0, split) go to first[first_offset + row, t], the rest to
// second[second_offset + row - split, t].
void ternary_a8_project_split(const TernaryA8Activations& x, const Weight& w, Tensor& first,
                         std::int32_t first_offset, std::int32_t split, Tensor& second,
                         std::int32_t second_offset, cudaStream_t stream);

struct TernaryA8Split {
    const Weight& weight;
    Tensor& first;
    std::int32_t first_offset;
    std::int32_t split;
    Tensor& second;
    std::int32_t second_offset;
};

// Both parents of a split projection over the same activations: one launch on the small-T route,
// one per parent on the prefill route.
void ternary_a8_project_split_pair(const TernaryA8Activations& x, const TernaryA8Split& a, const TernaryA8Split& b,
                              cudaStream_t stream);

void ternary_a8_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                  cudaStream_t stream);
void ternary_a8_linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& workspace,
                      cudaStream_t stream);

} // namespace ninfer::ops::detail
