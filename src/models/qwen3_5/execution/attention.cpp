#include "models/qwen3_5/execution/attention.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rope.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        // Native GGUF parents are separate Linear operands, so the fused Op's workspace does not
        // apply; each part needs its own Linear capacity.
        std::size_t bytes = 0;
        for (const auto& part : gguf->parts) {
            bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                        part.weight.qtype, part.weight.n, part.weight.k,
                                        ops::LinearPolicy::A16Only, first, last));
        }
        return bytes;
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                             single->policy, first, last);
    }
    return 0;
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        // Native GGUF q/k/v/gate are separate parents (or row ranges of one combined parent), so
        // each part is projected on its own into the public output slot it fills: 0=q, 1=gate,
        // 2=k, 3=v.
        for (const auto& part : gguf->parts) {
            if (part.row != 0) {
                throw std::invalid_argument(
                    "attention projection: a GGUF part must fill its output from row zero");
            }
            Tensor* out = part.output == 0   ? &query
                          : part.output == 1 ? &gate
                          : part.output == 2 ? &key
                                             : &value;
            ops::linear(hidden, part.weight, *out, ops::LinearPolicy::A16Only, workspace, stream);
        }
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(hidden, single.weight, query, gate, key, value, single.policy,
                             workspace, stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               DeviceExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, execution);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               DeviceExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, execution);
}

} // namespace ninfer::models::qwen3_5::execution
