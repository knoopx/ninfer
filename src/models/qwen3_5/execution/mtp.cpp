#include "models/qwen3_5/execution/mtp.h"

#include "core/layout.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/mtp_pack.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                           std::int32_t first, std::int32_t last) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.packed)) {
        // Native GGUF parents are separate Linear operands.
        std::size_t bytes = 0;
        for (const auto& part : gguf->parts) {
            bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                        part.weight.qtype, part.weight.n, part.weight.k,
                                        ops::LinearPolicy::A16Only, first, last));
        }
        return bytes;
    }
    const auto& p = std::get<LinearParameters>(parameters.packed);
    const auto& w = p.weight;
    if (!parameters.rows) {
        return ops::attn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first,
                                                             last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                   const AttentionConfig& config, std::int32_t first,
                                   std::int32_t last) {
    if (parameters.rows) {
        const auto& k = (*parameters.rows)[1];
        const auto& v = (*parameters.rows)[3];
        if (k.weight.layout == QuantLayout::GgufNative ||
            v.weight.layout == QuantLayout::GgufNative) {
            // Native GGUF K/V are separate parents, so the Q8 pair form does not apply.
            return std::max(
                ops::linear_workspace_capacity_bytes(k.weight.qtype, k.weight.n, k.weight.k,
                                                     ops::LinearPolicy::A16Only, first, last),
                ops::linear_workspace_capacity_bytes(v.weight.qtype, v.weight.n, v.weight.k,
                                                     ops::LinearPolicy::A16Only, first, last));
        }
        return ops::linear_pair_workspace_capacity_bytes(k.weight, v.weight, first, last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                           const AttentionConfig& config, std::int32_t first,
                                           std::int32_t last) {
    if (parameters.rows) {
        const auto bytes = [&](std::size_t index) {
            const auto& p = (*parameters.rows)[index];
            const auto& w = p.weight;
            return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
        };
        return std::max(bytes(0), bytes(2));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.packed)) {
        // Public output order is q, gate, k, v.
        for (const auto& part : gguf->parts) {
            if (part.row != 0) {
                throw std::invalid_argument(
                    "MTP projection: a GGUF part must fill its output from row zero");
            }
            Tensor* out = part.output == 0   ? &query
                          : part.output == 1 ? &gate
                          : part.output == 2 ? &key
                                             : &value;
            ops::linear(hidden, part.weight, *out, ops::LinearPolicy::A16Only, workspace, stream);
        }
        return;
    }
    const auto& p = std::get<LinearParameters>(parameters.packed);
    if (!parameters.rows) {
        ops::attn_input_proj(hidden, p.weight, query, gate, key, value, p.policy, workspace,
                             stream);
        return;
    }
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    Tensor packed      = workspace.alloc(DType::BF16, {p.weight.n, columns});
    ops::linear(hidden, p.weight, packed, p.policy, workspace, stream);
    Tensor q =
        query.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor k =
        key.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    Tensor g =
        gate.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor v =
        value.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    ops::mtp_split_attn_in(packed, q, k, g, v, stream);
}

void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        const auto& k = (*parameters.rows)[1];
        const auto& v = (*parameters.rows)[3];
        if (k.weight.layout == QuantLayout::GgufNative ||
            v.weight.layout == QuantLayout::GgufNative) {
            // Native GGUF K/V are separate parents: project each on its own.
            ops::linear(hidden, k.weight, key, ops::LinearPolicy::A16Only, workspace, stream);
            ops::linear(hidden, v.weight, value, ops::LinearPolicy::A16Only, workspace, stream);
            return;
        }
        ops::linear_pair(hidden, k.weight, v.weight, key, value, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    Tensor gate  = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        const auto& q = (*parameters.rows)[0];
        const auto& g = (*parameters.rows)[2];
        {
            auto scope = workspace.scope();
            ops::linear(hidden, q.weight, query, q.policy, workspace, stream);
        }
        ops::linear(hidden, g.weight, gate, g.policy, workspace, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor key   = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    Tensor value = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
