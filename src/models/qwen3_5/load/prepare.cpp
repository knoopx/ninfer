#include "models/qwen3_5/load/bindings.h"

#include "artifact/views.h"

#include <cmath>

namespace ninfer::models::qwen3_5::loading {

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference =
        binder.parameter(name, std::move(shape), artifact::Residency::Device, exact_format);
    for (const auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
        }
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role != "activation_input_divisor") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            const auto value = binder.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation divisor must be positive finite FP32");
            }
            result.activation_input_divisor = value;
        }
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format) {
    return parameter(std::move(name), std::move(shape), {}, format);
}

// --- folded (rotated-basis) sign table ---------------------------------------
//
// These constants describe the artifact contract the ternary port writes: 1024-wide normalized
// Sylvester-Hadamard blocks, 28672 explicit signs, and widths that partition the signs exactly.
inline constexpr std::size_t kHadamardSignValues = 28672;
inline constexpr std::size_t kHadamardWidthCount = 3;

std::optional<HadamardSigns> bind_hadamard_signs(Bindings& bindings) {
    auto& binder = bindings.binder;
    if (!binder.contains("text/hadamard_signs")) { return std::nullopt; }

    HadamardSigns out;
    out.values = binder.parameter("text/hadamard_signs", {kHadamardSignValues},
                                  artifact::Residency::Device, QType::FP32);
    // The widths only need to reach the host: they are read here to derive each width's element
    // offset, which is what lets a weight find its block from its input dimension alone.
    const auto widths_ref = binder.parameter("text/hadamard_widths", {kHadamardWidthCount},
                                             artifact::Residency::Values, QType::INT32);
    const auto widths = binder.values(widths_ref.binding, QType::INT32).integers();
    if (widths.size() != kHadamardWidthCount) {
        throw artifact::ArtifactError("text/hadamard_widths is shorter than its declared count");
    }
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < widths.size(); ++i) {
        if (widths[i] <= 0) {
            throw artifact::ArtifactError("text/hadamard_widths entries must be positive");
        }
        out.width_offsets.emplace_back(widths[i], offset);
        offset += static_cast<std::uint64_t>(widths[i]);
    }
    if (offset != kHadamardSignValues) {
        throw artifact::ArtifactError("text/hadamard_widths must sum to " +
                                      std::to_string(kHadamardSignValues) + ", got " +
                                      std::to_string(offset));
    }
    return out;
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
