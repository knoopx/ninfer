#include "serve/model_config.h"

#include "product/speculative_options.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json;

// String → KvCacheStorage (mirrors serve_options.cpp parse_kv_dtype; duplicated here because
// model_config.cpp is a standalone parsing unit that does not depend on serve_options).
KvCacheStorage parse_kv_cache_storage(std::string_view text) {
    if (text == "bf16")  { return KvCacheStorage::BFloat16; }
    if (text == "int8")  { return KvCacheStorage::Int8Group64; }
    if (text == "fp8")   { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4")  { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument(std::string("config: invalid kvDtype: ") + std::string(text));
}

// Reads the top-level JSON document into a Config. Malformed documents throw a nlohmann parse
// error, which parse_config converts to std::invalid_argument. Config-level validation errors
// (non-object top level, a model missing its required `artifact`) are thrown here as
// std::invalid_argument and propagate through unchanged.
Config parse_impl(const std::string& text) {
    Json root = Json::parse(text);
    if (!root.is_object()) {
        throw std::invalid_argument("config: top-level value must be a JSON object");
    }

    Config cfg;

    // `models` (ordered): iterate in key order; the first entry is the resident model.
    if (root.contains("models")) {
        const Json& models = root.at("models");
        if (!models.is_object()) {
            throw std::invalid_argument("config: `models` must be a JSON object");
        }
        for (const auto& [id, value] : models.items()) {
            const std::string model_id = id;
            if (!value.is_object()) {
                throw std::invalid_argument("config: model '" + model_id + "' must be a JSON object");
            }
            ModelConfig model;
            model.id = model_id;
            // `artifact` is the ninfer-native translation of llama-swap's per-model `cmd`; it is
            // the `.ninfer` artifact path and is required.
            model.artifact = value.value<std::string>("artifact", "");
            if (model.artifact.empty()) {
                throw std::invalid_argument("config: model '" + model_id + "' requires an `artifact`");
            }
            // `identity` is the registered model identity; default it to the public model id.
            model.identity = value.value<std::string>("identity", "");
            if (model.identity.empty()) {
                model.identity = model.id;
            }
            model.aliases = value.value<std::vector<std::string>>("aliases", {});
            model.ttl = value.value<int>("ttl", 0);
            model.name = value.value<std::string>("name", "");
            model.description = value.value<std::string>("description", "");
            model.check_endpoint = value.value<std::string>("checkEndpoint", "/health");
            if (value.contains("capabilities") && value.at("capabilities").is_object()) {
                const Json& caps = value.at("capabilities");
                model.capabilities.input = caps.value<std::vector<std::string>>("in", {});
                model.capabilities.output = caps.value<std::vector<std::string>>("out", {});
                model.capabilities.tools = caps.value<bool>("tools", false);
                model.capabilities.reranker = caps.value<bool>("reranker", false);
                model.capabilities.context = caps.value<int>("context", 0);
            }
            // Per-model engine overrides (all optional; nullopt = use global ServeOptions value).
            if (value.contains("maxContext")) {
                model.overrides.max_context = value.at("maxContext").get<std::uint32_t>();
            }
            if (value.contains("defaultMaxTokens")) {
                model.overrides.default_max_tokens = value.at("defaultMaxTokens").get<int>();
            }
            // kvCapacity: a JSON number, the string "auto", or a numeric string (the Nix config
            // emits it as a string). "auto" -> automatic sizing; otherwise an explicit token count.
            if (value.contains("kvCapacity")) {
                const Json& kv_capacity = value.at("kvCapacity");
                if (kv_capacity.is_string()) {
                    const std::string kv_text = kv_capacity.get<std::string>();
                    model.overrides.kv_capacity =
                        kv_text == "auto"
                            ? KvCapacityPolicy::automatic()
                            : KvCapacityPolicy::explicit_capacity(
                                  static_cast<std::uint32_t>(std::stoul(kv_text)));
                } else {
                    model.overrides.kv_capacity =
                        KvCapacityPolicy::explicit_capacity(kv_capacity.get<std::uint32_t>());
                }
            }
            if (value.contains("kvDtype")) {
                model.overrides.kv_cache = parse_kv_cache_storage(value.at("kvDtype").get<std::string>());
            }
            if (value.contains("spec") || value.contains("draftTokens") ||
                value.contains("lmHeadDraft")) {
                SpeculativeOptions spec;
                if (value.contains("spec")) {
                    spec.backend =
                        product::parse_speculative_backend(value.at("spec").get<std::string>());
                }
                if (value.contains("draftTokens")) {
                    spec.draft_tokens = value.at("draftTokens").get<std::uint32_t>();
                }
                if (value.contains("lmHeadDraft")) {
                    spec.proposal_head = value.at("lmHeadDraft").get<bool>()
                                             ? ProposalHead::Optimized
                                             : ProposalHead::Full;
                }
                model.overrides.speculative = spec;
            }
            if (value.contains("prefillChunk")) {
                model.overrides.prefill_chunk = value.at("prefillChunk").get<std::uint32_t>();
            }
            if (value.contains("vision")) {
                model.overrides.enable_vision = value.at("vision").get<bool>();
            }
            // --- per-model validation (grounded in the engine's validate_target_options) ---
            // kvCapacity (explicit, not "auto") must be at least maxContext. Ground: qwen3_6
            // layouts_impl.h validate_target_options ("kv_capacity must be at least max_context");
            // the Automatic policy is exempt (sized from the context at load time).
            if (model.overrides.kv_capacity.has_value() && model.overrides.max_context.has_value() &&
                model.overrides.kv_capacity->mode == ninfer::KvCapacityMode::Explicit &&
                model.overrides.kv_capacity->explicit_tokens < *model.overrides.max_context) {
                throw std::invalid_argument(
                    "config: model '" + model_id + "' kvCapacity (" +
                    std::to_string(model.overrides.kv_capacity->explicit_tokens) +
                    ") must be at least maxContext (" +
                    std::to_string(*model.overrides.max_context) + ")");
            }
            // prefillChunk must be a positive multiple of 128. Ground: qwen3_6 layouts_impl.h
            // validate_target_options ("prefill_chunk must be a nonzero multiple of 128").
            if (model.overrides.prefill_chunk.has_value() &&
                (*model.overrides.prefill_chunk == 0 || *model.overrides.prefill_chunk % 128u != 0)) {
                throw std::invalid_argument(
                    "config: model '" + model_id +
                    "' prefillChunk must be a nonzero multiple of 128");
            }
            // speculative draftTokens range is backend-specific; the config carries the backend, so
            // the exact per-backend range is checkable model-agnostically. Ground: product/
            // speculative_options.h validate_speculative_cli_options (mtp [1,5]; dflash/dflash2
            // [1,15]; none requires 0). Reuse the engine's own validation so the ranges never drift
            // from the target code (a spec-less draftTokens hits the None case and is rejected).
            if (model.overrides.speculative.has_value()) {
                try {
                    product::validate_speculative_cli_options(*model.overrides.speculative);
                } catch (const std::invalid_argument& e) {
                    throw std::invalid_argument(
                        "config: model '" + model_id + "' speculative options: " + e.what());
                }
            }
            cfg.models.push_back(std::move(model));
        }
    }

    // Top-level scalar fields (with their defaults).
    cfg.global_ttl = root.value<int>("globalTTL", 0);
    cfg.unload_timeout = root.value<int>("unloadTimeout", 10);
    cfg.health_check_timeout = root.value<int>("healthCheckTimeout", 120);
    cfg.start_port = root.value<int>("startPort", 5800);

    // `routing` (nested, optional). A missing `routing`/`scheduler`/`router` object keeps the
    // header defaults (scheduler "fifo", router "group").
    if (root.contains("routing") && root.at("routing").is_object()) {
        const Json& routing = root.at("routing");
        if (routing.contains("scheduler") && routing.at("scheduler").is_object()) {
            cfg.routing.scheduler.use = routing.at("scheduler").value<std::string>("use", "fifo");
        }
        if (routing.contains("router") && routing.at("router").is_object()) {
            cfg.routing.router.use = routing.at("router").value<std::string>("use", "group");
        }
    }

    // `groups` (optional), stored as a map keyed by group id.
    if (root.contains("groups") && root.at("groups").is_object()) {
        const Json& groups = root.at("groups");
        for (const auto& [id, value] : groups.items()) {
            const std::string group_id = id;
            GroupConfig group;
            if (value.is_object()) {
                group.swap = value.value<bool>("swap", true);
                group.exclusive = value.value<bool>("exclusive", true);
                group.persistent = value.value<bool>("persistent", false);
                group.members = value.value<std::vector<std::string>>("members", {});
            }
            cfg.groups[group_id] = std::move(group);
        }
    }

    return cfg;
}

} // namespace

Config parse_config(std::string_view json_text) {
    try {
        return parse_impl(std::string(json_text));
    } catch (const std::invalid_argument&) {
        throw; // Config validation errors are already std::invalid_argument; propagate unchanged.
    } catch (const nlohmann::json::exception& e) {
        // Malformed JSON (parse error) or a field with an unexpected JSON type.
        throw std::invalid_argument(std::string("config parse error: ") + e.what());
    }
}

Config load_config(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::invalid_argument("config: cannot open " + path.string());
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return parse_config(buffer.str());
}

const ModelConfig* Config::model(std::string_view id) const {
    for (const auto& model : models) {
        if (model.id == id) { return &model; }
    }
    return nullptr;
}

const ModelConfig& Config::resident() const {
    if (models.empty()) {
        throw std::logic_error("config: no models are defined; cannot select a resident model");
    }
    return models.front();
}

} // namespace ninfer::serve
