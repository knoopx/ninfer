#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/sparse_moe.h"

#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace ninfer::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
};

struct PairedProjectionWeights {
    Weight first, second;
};

// One part of a fused input projection whose parent is a native GGUF weight. `output` selects the
// fused output slot (q/gate/k/v for attention, qkv/z for GDN) and `row` the first row that part
// fills within it. Consecutive parts of one parent merge, so a combined q/k/v parent contributes
// one part per output while separate q/k/v parents contribute one part each.
struct GgufProjectionPart {
    Weight weight;
    std::int32_t output = 0;
    std::int32_t row    = 0;
};

struct GgufProjectionWeights {
    std::vector<GgufProjectionPart> parts;
};

using ProjectionWeights =
    std::variant<SingleProjectionWeight, PairedProjectionWeights, GgufProjectionWeights>;

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                                     const WeightInput& key,
                                                                     const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query,
                                                                const WeightInput& key,
                                                                const WeightInput& gate,
                                                                const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query,
                                                               const WeightInput& key,
                                                               const WeightInput& value,
                                                               const WeightInput& z);
[[nodiscard]] ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a,
                                                                const WeightInput& b);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);
[[nodiscard]] SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down);

} // namespace ninfer::ops
