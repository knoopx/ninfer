#include "models/qwen3_5/execution/gdn.h"

#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

// A native GGUF GDN projection has separate parents (a combined qkv parent and a z parent), so the
// fused snapshot/record forms cannot apply. The projection is applied part by part into a combined
// BF16 plane, which the convolution-only Op then consumes.
std::size_t gguf_plane_capacity_bytes(const ops::GgufProjectionWeights& projection,
                                      const GdnConfig& config, std::int32_t batch,
                                      std::int32_t first_width, std::int32_t last_width) {
    const auto channels = 2 * static_cast<std::int32_t>(config.key_width()) +
                          static_cast<std::int32_t>(config.value_width());
    std::size_t bytes =
        static_cast<std::size_t>(channels) * last_width * batch * sizeof(std::uint16_t);
    for (const auto& part : projection.parts) {
        bytes += ops::linear_workspace_capacity_bytes(
            part.weight.qtype, part.weight.n, part.weight.k, ops::LinearPolicy::A16Only,
            first_width * batch, last_width * batch);
    }
    return bytes;
}

Tensor gguf_projected_plane(const Tensor& hidden, const Tensor& z,
                            const ops::GgufProjectionWeights& projection, const GdnConfig& config,
                            WorkspaceArena& workspace, cudaStream_t stream) {
    const auto channels = 2 * static_cast<std::int32_t>(config.key_width()) +
                          static_cast<std::int32_t>(config.value_width());
    const auto columns = hidden.ne[1] * hidden.ne[2];
    Tensor projected   = workspace.alloc(DType::BF16, {channels, columns});
    Tensor hidden_flat = hidden.view({hidden.ne[0], columns});
    Tensor z_flat      = z.view({z.ne[0], columns});
    for (const auto& part : projection.parts) {
        if (part.row != 0) {
            throw std::invalid_argument(
                "GDN projection: a GGUF part must fill its output from row zero");
        }
        Tensor* out = part.output == 0 ? &projected : &z_flat;
        ops::linear(hidden_flat, part.weight, *out, ops::LinearPolicy::A16Only, workspace, stream);
    }
    return Tensor(projected.data, DType::BF16, {channels, hidden.ne[1], hidden.ne[2]});
}

} // namespace

std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters, std::int32_t first,
                                           std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("GDN projection: invalid column interval");
    }
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        // Native GGUF parents are separate Linear operands, so each part needs its own Linear
        // capacity rather than the fused Op's.
        std::size_t bytes = 0;
        for (const auto& part : gguf->parts) {
            bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                        part.weight.qtype, part.weight.n, part.weight.k,
                                        ops::LinearPolicy::A16Only, first, last));
        }
        return bytes;
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& w = single->weight;
        return ops::gdn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, single->policy,
                                                            first, last);
    }
    return 0;
}

std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                         std::int32_t batch, std::int32_t first_width,
                                         std::int32_t last_width) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        return std::max(std::size_t{1},
                        gguf_plane_capacity_bytes(*gguf, config, batch, first_width, last_width));
    }
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    // The native overlap contract takes a disjoint span even for a zero-scratch fused route.
    return std::max(std::size_t{1}, bytes);
}

std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                       std::int32_t batch, std::int32_t first_width,
                                       std::int32_t last_width) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        return std::max(std::size_t{1},
                        gguf_plane_capacity_bytes(*gguf, config, batch, first_width, last_width));
    }
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    return std::max(std::size_t{1}, bytes);
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        // Native GGUF: output 0 is the combined qkv parent, output 1 the z parent; both are
        // projected on their own into the full output tensor.
        for (const auto& part : gguf->parts) {
            if (part.row != 0) {
                throw std::invalid_argument(
                    "GDN projection: a GGUF part must fill its output from row zero");
            }
            Tensor* out = part.output == 0 ? &qkv : &z;
            ops::linear(hidden, part.weight, *out, ops::LinearPolicy::A16Only, workspace, stream);
        }
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj(hidden, pair->first, pair->second, qkv, z, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj(hidden, single.weight, qkv, z, single.policy, workspace, stream);
    }
}

void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      WorkspaceArena& workspace, DeviceExecutionView execution) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.control)) {
        ops::gdn_norm_gating_proj(residual, norm, epsilon, pair->first, pair->second,
                                  parameters.a_log, parameters.dt_bias, workspace, hidden, g, beta,
                                  execution);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.control);
        ops::gdn_norm_gating_proj(residual, norm, epsilon, single.weight, parameters.a_log,
                                  parameters.dt_bias, workspace, hidden, g, beta, execution);
    }
}

void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(gdn_snapshot_workspace_bytes(
        parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        const Tensor projected =
            gguf_projected_plane(hidden, z, *gguf, config, scratch, stream);
        ops::gdn_projected_conv_snapshot(projected, parameters.convolution, conv_states,
                                         valid_columns, initial_slots, destination_slots, query,
                                         key, value, stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, pair->first, pair->second, parameters.convolution,
                                          conv_states, valid_columns, initial_slots,
                                          destination_slots, query, key, value, z, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_snapshot(
            hidden, single.weight, parameters.convolution, conv_states, valid_columns,
            initial_slots, destination_slots, query, key, value, z, single.policy, scratch, stream);
    }
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(
        gdn_record_workspace_bytes(parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        // The record plane is the caller's conv_record: the projection fills it and the
        // convolution records into it.
        const auto columns = hidden.ne[1] * hidden.ne[2];
        Tensor hidden_flat = hidden.view({hidden.ne[0], columns});
        Tensor record_flat = conv_record.view({conv_record.ne[0], columns});
        Tensor z_flat      = z.view({z.ne[0], columns});
        for (const auto& part : gguf->parts) {
            if (part.row != 0) {
                throw std::invalid_argument(
                    "GDN projection: a GGUF part must fill its output from row zero");
            }
            Tensor* out = part.output == 0 ? &record_flat : &z_flat;
            ops::linear(hidden_flat, part.weight, *out, ops::LinearPolicy::A16Only, scratch, stream);
        }
        ops::gdn_projected_conv_record(conv_record, parameters.convolution, conv_states,
                                       valid_columns, initial_slots, query, key, value, stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_record(hidden, pair->first, pair->second, parameters.convolution,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, z, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_record(hidden, single.weight, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, conv_record, query, key,
                                        value, z, single.policy, scratch, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
