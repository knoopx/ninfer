#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "models/load_options.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/weights.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;
using artifact::ArtifactError;
using artifact::Residency;
using ninfer::test::artifact_fixture::require;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
}

// Selected DFlash2 device objects are the device placements whose object id names the dflash2/
// domain. A legacy artifact without the bundle carries no such object in any selected plan.
std::size_t dflash2_device_objects(const artifact::Reader& reader,
                                   const artifact::MaterializationPlan& plan) {
    std::size_t count = 0;
    for (const auto& item : plan.device_objects) {
        if (artifact::object_id(reader.directory().object(item.object)).starts_with("dflash2/")) {
            ++count;
        }
    }
    return count;
}

std::set<std::size_t> device_object_set(const artifact::MaterializationPlan& plan) {
    std::set<std::size_t> set;
    for (const auto& item : plan.device_objects) { (void)set.insert(item.object.index); }
    return set;
}

bool parameter_on_device(const qwen::LoadPlan& plan, qwen::WeightId id,
                         const std::set<std::size_t>& device_objects) {
    for (const auto& part : plan.parameter(id).binding.parts) {
        if (device_objects.contains(part.object.index)) { return true; }
    }
    return false;
}

QType tensor_format(const artifact::Reader& reader, const qwen::LoadPlan& plan, qwen::WeightId id) {
    const auto& reference = plan.parameter(id);
    require(!reference.binding.parts.empty(), "parameter has no bound object parts");
    const auto& object = reader.directory().object(reference.binding.parts.front().object);
    const auto* tensor = std::get_if<artifact::TensorObject>(&object);
    require(tensor != nullptr, "parameter part is not a tensor object");
    return artifact::parse_format(tensor->format);
}

// Placement invariants, re-derived from the plan: unique, aligned, non-overlapping device
// placements whose tail is the declared capacity.
void check_placements(const qwen::LoadPlan& plan) {
    const auto& materialization = plan.materialization();
    std::set<std::size_t> parents;
    std::uint64_t end = 0;
    for (const auto& item : materialization.device_objects) {
        require(parents.insert(item.object.index).second,
                "device parent was planned more than once");
        require(item.offset >= end && item.offset % item.alignment == 0,
                "device placement overlaps or is unaligned");
        end = item.offset + item.bytes;
    }
    require(end == materialization.device_capacity_bytes,
            "weight capacity differs from actual placements");
}

void check_selection(const qwen::LoadPlan& plan, const LoadOptions& options) {
    const auto& weights = plan.weights();
    require(weights.vision.has_value() == options.vision,
            "Vision demand differs from startup selection");
    require(weights.mtp.has_value() == (options.speculative == SpeculativeBackend::Mtp),
            "MTP selection differs");
    require(weights.draft.has_value() ==
                (options.speculative == SpeculativeBackend::DFlash ||
                 options.speculative == SpeculativeBackend::DFlash2),
            "draft selection differs");
    require(weights.proposal.has_value() == options.proposal_enabled(),
            "proposal selection differs");
    require(weights.text.layers.size() == plan.config().text.num_hidden_layers,
            "layer data differs from instance config");
}

// Groupwise-int 27B: per-layer storage-format inventory. The exact per-layer format mix
// (which Q4/Q5/Q6) is artifact-specific and not re-derivable from the plan; assert the
// groupwise-int class instead.
int verify_groupwise(const std::filesystem::path& path) {
    const LoadOptions options{.vision    = true,
                              .speculative = SpeculativeBackend::Mtp,
                              .proposal_head = ProposalHead::Optimized};
    artifact::Reader reader(path);
    const auto plan = qwen::plan_load(reader, options);
    check_selection(plan, options);
    const auto& weights = plan.weights();
    const auto& text    = plan.config().text;
    require(text.num_hidden_layers > 0, "text config has no layers");
    require(text.full_attention_layers + text.linear_attention_layers == text.num_hidden_layers,
            "attention kind counts differ from layer types");
    const auto& materialization = plan.materialization();
    require(materialization.object_count > materialization.device_objects.size(),
            "materialization plan lost objects");
    require(materialization.device_capacity_bytes > 0, "device capacity is empty");
    require(materialization.host_objects.size() > 0, "no host resource placements");
    check_placements(plan);
    const auto g64 = [](QType format) {
        return format == QType::Q4_G64_FP16 || format == QType::Q5_G64_FP16 ||
               format == QType::Q6_G64_FP16;
    };
    require(g64(tensor_format(reader, plan, weights.text.token_embedding)),
            "groupwise token embedding left the groupwise-int profile");
    require(g64(tensor_format(reader, plan, weights.text.output_head)),
            "groupwise output head left the groupwise-int profile");
    for (const auto& layer : weights.text.layers) {
        if (const auto* mlp = std::get_if<qwen::DenseWeights>(&layer.ffn)) {
            require(g64(tensor_format(reader, plan, mlp->gate)), "groupwise MLP gate format");
            require(g64(tensor_format(reader, plan, mlp->up)), "groupwise MLP up format");
            require(g64(tensor_format(reader, plan, mlp->down)), "groupwise MLP down format");
        } else {
            const auto& moe = std::get<qwen::MoeWeights>(layer.ffn);
            for (const auto& expert : moe.experts) {
                require(g64(tensor_format(reader, plan, expert.gate)), "groupwise expert gate");
                require(g64(tensor_format(reader, plan, expert.up)), "groupwise expert up");
                require(g64(tensor_format(reader, plan, expert.down)), "groupwise expert down");
            }
            require(g64(tensor_format(reader, plan, moe.shared.gate)), "groupwise shared gate");
            require(g64(tensor_format(reader, plan, moe.shared.up)), "groupwise shared up");
            require(g64(tensor_format(reader, plan, moe.shared.down)), "groupwise shared down");
        }
    }
    std::cout << "groupwise: objects=" << materialization.object_count
              << " device=" << materialization.device_objects.size()
              << " host=" << materialization.host_objects.size()
              << " capacity_bytes=" << materialization.device_capacity_bytes << '\n';
    return 0;
}

// NVFP4 27B: per-layer storage-format inventory. NVFP4 MLP weights are re-derived exactly; the
// attention bf16/nvfp4 split is reported, not pinned (the old API's layer counts were
// artifact-specific).
int verify_nvfp4(const std::filesystem::path& path) {
    const LoadOptions options{.vision    = true,
                              .speculative = SpeculativeBackend::Mtp,
                              .proposal_head = ProposalHead::Optimized};
    artifact::Reader reader(path);
    const auto plan = qwen::plan_load(reader, options);
    check_selection(plan, options);
    const auto& weights = plan.weights();
    const auto& text    = plan.config().text;
    require(text.num_hidden_layers > 0, "text config has no layers");
    const auto& materialization = plan.materialization();
    require(materialization.object_count > materialization.device_objects.size(),
            "materialization plan lost objects");
    require(materialization.device_capacity_bytes > 0, "device capacity is empty");
    require(materialization.host_objects.size() > 0, "no host resource placements");
    check_placements(plan);
    const auto mixed = [](QType format) {
        return format == QType::NVFP4 || format == QType::BF16;
    };
    require(mixed(tensor_format(reader, plan, weights.text.token_embedding)),
            "NVFP4 token embedding left the NVFP4 profile");
    require(mixed(tensor_format(reader, plan, weights.text.output_head)),
            "NVFP4 output head left the NVFP4 profile");
    std::size_t nvfp4_mlp      = 0;
    std::size_t bf16_attention_inputs = 0;
    std::size_t bf16_attention_outputs = 0;
    for (const auto& layer : weights.text.layers) {
        const auto count_mlp = [&](const qwen::DenseWeights& mlp) {
            for (const qwen::WeightId id : {mlp.gate, mlp.up, mlp.down}) {
                require(tensor_format(reader, plan, id) == QType::NVFP4,
                        "NVFP4 MLP weight left NVFP4");
                ++nvfp4_mlp;
            }
        };
        if (const auto* mlp = std::get_if<qwen::DenseWeights>(&layer.ffn)) {
            count_mlp(*mlp);
        } else {
            const auto& moe = std::get<qwen::MoeWeights>(layer.ffn);
            for (const auto& expert : moe.experts) { count_mlp(expert); }
            count_mlp(moe.shared);
        }
        if (const auto* attention = std::get_if<qwen::AttentionWeights>(&layer.mixer)) {
            for (const qwen::WeightId id : {attention->query, attention->key, attention->gate,
                                            attention->value}) {
                const auto format = tensor_format(reader, plan, id);
                require(mixed(format), "NVFP4 attention input left the NVFP4/BF16 profile");
                if (format == QType::BF16) { ++bf16_attention_inputs; }
            }
            if (tensor_format(reader, plan, attention->output) == QType::BF16) {
                ++bf16_attention_outputs;
            }
        } else {
            const auto& gdn = std::get<qwen::GdnWeights>(layer.mixer);
            for (const qwen::WeightId id : {gdn.query, gdn.key, gdn.value, gdn.z}) {
                const auto format = tensor_format(reader, plan, id);
                require(mixed(format), "NVFP4 GDN input left the NVFP4/BF16 profile");
                if (format == QType::BF16) { ++bf16_attention_inputs; }
            }
            if (tensor_format(reader, plan, gdn.output) == QType::BF16) {
                ++bf16_attention_outputs;
            }
        }
    }
    require(nvfp4_mlp > 0, "NVFP4 artifact lost its NVFP4 MLP weights");
    std::cout << "nvfp4: objects=" << materialization.object_count
              << " device=" << materialization.device_objects.size()
              << " host=" << materialization.host_objects.size()
              << " capacity_bytes=" << materialization.device_capacity_bytes
              << " nvfp4_mlp=" << nvfp4_mlp << " bf16_attention_input=" << bf16_attention_inputs
              << " bf16_attention_output=" << bf16_attention_outputs << '\n';
    return 0;
}

// Legacy artifact without a dflash2 component: None and Mtp selections bind no draft, and a
// selected DFlash2 backend is rejected naming the missing component.
int verify_legacy_dflash2_compatibility(const std::filesystem::path& path) {
    {
        artifact::Reader reader(path);
        const auto plan = qwen::plan_load(reader);
        require(!plan.weights().draft && dflash2_device_objects(reader, plan.materialization()) == 0,
                "legacy artifact unexpectedly bound DFlash2");
    }
    {
        artifact::Reader reader(path);
        const auto plan =
            qwen::plan_load(reader, LoadOptions{.speculative = SpeculativeBackend::Mtp});
        const auto& weights = plan.weights();
        require(weights.mtp.has_value() && !weights.draft,
                "legacy artifact did not preserve MTP-only binding");
        require(dflash2_device_objects(reader, plan.materialization()) == 0,
                "legacy artifact bound DFlash2 device objects");
        const auto device = device_object_set(plan.materialization());
        require(plan.parameter(weights.mtp->input_projection).residency == Residency::Device &&
                    parameter_on_device(plan, weights.mtp->input_projection, device),
                "MTP input projection left device residency");
    }
    {
        artifact::Reader reader(path);
        try {
            (void)qwen::plan_load(reader, LoadOptions{.speculative = SpeculativeBackend::DFlash2});
        } catch (const ArtifactError& error) {
            if (std::string(error.what()).find("dflash2") != std::string::npos) { return 0; }
            std::cerr << "legacy DFlash2 rejection lost the component name: " << error.what()
                      << '\n';
            return 1;
        }
        std::cerr << "legacy artifact did not reject selected DFlash2: " << path << '\n';
        return 1;
    }
}

// DFlash2 bundle artifact: inactive backends are validate-only (bundle present in file, not
// bound), and the selected DFlash2 backend places the bundle, the draft head, and the proposal
// head on device.
int verify_dflash2_bundle(const std::filesystem::path& path) {
    for (const SpeculativeBackend backend :
         {SpeculativeBackend::None, SpeculativeBackend::Mtp}) {
        artifact::Reader reader(path);
        LoadOptions options;
        options.speculative = backend;
        const auto plan = qwen::plan_load(reader, options);
        check_selection(plan, options);
        const auto& weights = plan.weights();
        require(!weights.draft && dflash2_device_objects(reader, plan.materialization()) == 0,
                "inactive DFlash2 bundle was not validate-only");
        if (weights.mtp) {
            const auto device = device_object_set(plan.materialization());
            require(parameter_on_device(plan, weights.mtp->input_projection, device),
                    "MTP placement does not match backend selection");
        }
    }
    artifact::Reader reader(path);
    const auto plan = qwen::plan_load(
        reader, LoadOptions{.speculative = SpeculativeBackend::DFlash2,
                            .proposal_head = ProposalHead::Optimized});
    check_selection(plan, LoadOptions{.speculative   = SpeculativeBackend::DFlash2,
                                      .proposal_head = ProposalHead::Optimized});
    const auto& weights = plan.weights();
    require(weights.draft.has_value(), "DFlash2 draft bundle was not bound");
    require(dflash2_device_objects(reader, plan.materialization()) > 0,
            "DFlash2 device objects are missing");
    require(!weights.mtp, "MTP was bound alongside DFlash2");
    const auto device = device_object_set(plan.materialization());
    require(plan.parameter(weights.draft->output_head).residency == Residency::Device &&
                parameter_on_device(plan, weights.draft->output_head, device),
            "draft output head left device residency");
    if (weights.proposal) {
        require(plan.parameter(weights.proposal->head).residency == Residency::Device &&
                    parameter_on_device(plan, weights.proposal->head, device),
                "proposal head left device residency");
        if (weights.proposal->token_ids) {
            require(plan.parameter(weights.proposal->token_ids.value()).residency ==
                        Residency::Device,
                    "proposal token ids left device residency");
        }
    }
    std::cout << "dflash2 bundle: dflash2_device_objects="
              << dflash2_device_objects(reader, plan.materialization()) << '\n';
    return 0;
}

} // namespace

int main() {
    try {
        const std::filesystem::path groupwise =
            artifact_path("NINFER_QWEN3_5_WEIGHTS", "qwen3_5_27b.ninfer");
        const std::filesystem::path nvfp4 =
            artifact_path("NINFER_QWEN3_5_NVFP4_WEIGHTS", "qwen3_5_27b_nvfp4.ninfer");
        const std::filesystem::path dflash2_groupwise =
            artifact_path("NINFER_QWEN3_5_DFLASH2_WEIGHTS", "qwen3_5_27b_dflash2.ninfer");
        const std::filesystem::path dflash2_nvfp4 = artifact_path(
            "NINFER_QWEN3_5_NVFP4_DFLASH2_WEIGHTS", "qwen3_5_27b_nvfp4_dflash2.ninfer");
        if (!std::filesystem::is_regular_file(groupwise) ||
            !std::filesystem::is_regular_file(nvfp4)) {
            std::cerr << "skip: both real 27B artifacts are required: groupwise=" << groupwise
                      << " nvfp4=" << nvfp4 << '\n';
            return 77;
        }
        if (const int result = verify_groupwise(groupwise); result != 0) { return result; }
        if (const int result = verify_nvfp4(nvfp4); result != 0) { return result; }
        if (const int result = verify_legacy_dflash2_compatibility(groupwise); result != 0) {
            return result;
        }
        if (const int result = verify_legacy_dflash2_compatibility(nvfp4); result != 0) {
            return result;
        }
        if (!std::filesystem::is_regular_file(dflash2_groupwise) ||
            !std::filesystem::is_regular_file(dflash2_nvfp4)) {
            std::cerr << "skip DFlash2 binding matrix: both DFlash2 27B artifacts are required: "
                      << dflash2_groupwise << ' ' << dflash2_nvfp4 << '\n';
            return 77;
        }
        if (const int result = verify_dflash2_bundle(dflash2_groupwise); result != 0) {
            return result;
        }
        if (const int result = verify_dflash2_bundle(dflash2_nvfp4); result != 0) {
            return result;
        }
        std::cout << "qwen3_5 27B load plan: placement, inventory, and DFlash2 binding passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
