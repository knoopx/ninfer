#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

    // Sign block for a folded ternary weight's input width, or nullptr when the artifact is not
    // folded (no sign table) or the width has no block. The block owns width/1024 rows of 1024
    // F32 +-1 values.
    [[nodiscard]] const float* hadamard_signs(std::int32_t width) const noexcept {
        if (hadamard_signs_base_ == nullptr) { return nullptr; }
        for (const auto& [w, offset] : hadamard_width_offsets_) {
            if (w == width) { return hadamard_signs_base_ + offset; }
        }
        return nullptr;
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing,
          const float* hadamard_signs,
          std::vector<std::pair<std::int32_t, std::uint64_t>> hadamard_width_offsets);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    const float* hadamard_signs_base_ = nullptr;
    std::vector<std::pair<std::int32_t, std::uint64_t>> hadamard_width_offsets_;
};

} // namespace ninfer::models::qwen3_5
