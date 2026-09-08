#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <string_view>
#include <utility>

namespace ninfer::serve {

namespace {

std::string chat_identifier(std::string_view prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx",
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + buffer.data();
}

std::string responses_identifier(std::string_view prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 48> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx%016llx",
                  static_cast<unsigned long long>(distribution(random)),
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + "_" + buffer.data();
}

} // namespace

using Json = nlohmann::json;

bool parse_openai_prompt_cache_breakpoint(const RequestJson& value, std::string_view param) {
    if (!value.contains("prompt_cache_breakpoint") ||
        value.at("prompt_cache_breakpoint").is_null()) {
        return false;
    }
    const RequestJson& breakpoint = value.at("prompt_cache_breakpoint");
    if (!breakpoint.is_object() || !breakpoint.contains("mode") ||
        !breakpoint.at("mode").is_string() ||
        breakpoint.at("mode").get<std::string>() != "explicit") {
        bad_request("prompt_cache_breakpoint must be {mode:'explicit'}", std::string(param),
                    "invalid_cache_breakpoint");
    }
    return true;
}

OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body) {
    auto require_string_hint = [&](const char* field, std::optional<std::size_t> maximum = {}) {
        if (!body.contains(field) || body.at(field).is_null()) { return; }
        if (!body.at(field).is_string()) {
            bad_request(std::string(field) + " must be a string", field);
        }
        if (maximum && body.at(field).get_ref<const std::string&>().size() > *maximum) {
            bad_request(std::string(field) + " must be at most " + std::to_string(*maximum) +
                            " characters",
                        field);
        }
    };
    require_string_hint("prompt_cache_key", 64U);
    require_string_hint("safety_identifier", 64U);
    require_string_hint("user");

    if (body.contains("prompt_cache_retention") && !body.at("prompt_cache_retention").is_null()) {
        if (!body.at("prompt_cache_retention").is_string()) {
            bad_request("prompt_cache_retention must be a string", "prompt_cache_retention");
        }
        const std::string value = body.at("prompt_cache_retention").get<std::string>();
        if (value != "in_memory" && value != "24h") {
            bad_request("prompt_cache_retention must be 'in_memory' or '24h'",
                        "prompt_cache_retention");
        }
    }

    OpenAIPromptCachePolicy policy;
    if (!body.contains("prompt_cache_options") || body.at("prompt_cache_options").is_null()) {
        return policy;
    }
    const RequestJson& options = body.at("prompt_cache_options");
    if (!options.is_object()) {
        bad_request("prompt_cache_options must be an object", "prompt_cache_options");
    }
    policy.automatic = OpenAIPromptCacheAutomatic::Requested;
    if (options.contains("mode") && !options.at("mode").is_null()) {
        if (!options.at("mode").is_string()) {
            bad_request("prompt_cache_options.mode must be a string", "prompt_cache_options");
        }
        const std::string mode = options.at("mode").get<std::string>();
        if (mode == "explicit") {
            policy.automatic = OpenAIPromptCacheAutomatic::Disabled;
        } else if (mode != "implicit") {
            bad_request("prompt_cache_options.mode must be 'implicit' or 'explicit'",
                        "prompt_cache_options");
        }
    }
    if (options.contains("ttl") && !options.at("ttl").is_null() &&
        (!options.at("ttl").is_string() || options.at("ttl").get<std::string>() != "30m")) {
        bad_request("prompt_cache_options.ttl must be '30m'", "prompt_cache_options");
    }
    return policy;
}

void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy) {
    std::vector<std::optional<CacheBoundary>*> explicit_boundaries;
    for (ToolDefinition& tool : request.tools) {
        if (tool.cache_boundary_after) {
            explicit_boundaries.push_back(&tool.cache_boundary_after);
        }
    }
    for (ChatTurn& turn : request.messages) {
        for (ContentPart& part : turn.content) {
            if (part.cache_boundary_after) {
                explicit_boundaries.push_back(&part.cache_boundary_after);
            }
        }
        if (turn.cache_boundary_after) {
            explicit_boundaries.push_back(&turn.cache_boundary_after);
        }
    }

    std::optional<CacheBoundary>* automatic_target = nullptr;
    for (auto turn = request.messages.rbegin(); turn != request.messages.rend(); ++turn) {
        if (!turn->tool_calls.empty()) {
            automatic_target = &turn->cache_boundary_after;
            break;
        }
        if (!turn->content.empty()) {
            automatic_target = &turn->content.back().cache_boundary_after;
            break;
        }
    }
    if (automatic_target == nullptr && !request.tools.empty()) {
        automatic_target = &request.tools.back().cache_boundary_after;
    }

    const bool automatic_enabled =
        policy.automatic != OpenAIPromptCacheAutomatic::Disabled && automatic_target != nullptr;
    const bool automatic_merges_explicit   = automatic_enabled && automatic_target->has_value();
    const std::size_t explicit_write_slots = !automatic_enabled || automatic_merges_explicit
                                                 ? kMaximumExplicitPromptCacheMarkers
                                                 : kMaximumExplicitPromptCacheMarkers - 1U;
    const std::size_t first_selected       = explicit_boundaries.size() > explicit_write_slots
                                                 ? explicit_boundaries.size() - explicit_write_slots
                                                 : 0U;
    for (std::size_t index = 0; index < explicit_boundaries.size(); ++index) {
        if (index < first_selected) {
            explicit_boundaries[index]->reset();
        } else {
            explicit_boundaries[index]->value().kind =
                ninfer::PromptCacheMarkerKind::SharedStablePrefix;
            explicit_boundaries[index]->value().evidence =
                ninfer::SharedCandidateEvidence::ExplicitBoundary;
        }
    }

    if (automatic_enabled) {
        const ninfer::SharedCandidateEvidence evidence =
            policy.automatic == OpenAIPromptCacheAutomatic::Default
                ? ninfer::SharedCandidateEvidence::DefaultAutomatic
                : ninfer::SharedCandidateEvidence::RequestedAutomatic;
        if (*automatic_target) {
            automatic_target->value().evidence |= evidence;
        } else {
            *automatic_target = CacheBoundary{.evidence = evidence};
        }
    }
    // OpenAI already defines the automatic/explicit write policy for every request. Existing
    // exact shared residents are still considered by the Engine independently of this switch.
    request.allow_engine_automatic_shared_prefixes = false;
}

std::string make_models_list(const std::vector<ModelConfig>& models, std::int64_t created,
                             std::uint32_t default_max_context, std::string_view loaded_id) {
    // vLLM/llama.cpp-compatible discovery metadata. Each model reports its OWN effective
    // per-request context limit: the per-model `maxContext` override when set, else the server
    // default (the single value passed in). `status` carries the webui "loaded" indicator the
    // model browser expects; it is set dynamically from the router's live loaded model (an empty
    // loaded_id means no model is currently loaded).
    Json data = Json::array();
    for (const auto& model : models) {
        const bool loaded = !loaded_id.empty() && model.id == loaded_id;
        const std::uint32_t max_model_len =
            model.overrides.max_context.value_or(default_max_context);
        data.push_back(Json{{"id", model.id},
                            {"object", "model"},
                            {"created", created},
                            {"owned_by", "ninfer"},
                            {"max_model_len", max_model_len},
                            {"status", Json{{"value", loaded ? "loaded" : "unloaded"}}}});
    }
    const Json payload = {{"object", "list"}, {"data", std::move(data)}};
    return payload.dump();
}

std::string make_model_object(const ModelConfig& model, std::int64_t created,
                              std::uint32_t default_max_context, bool loaded) {
    // vLLM/llama.cpp-compatible discovery metadata; `status` carries the webui "loaded"
    // indicator. The reported limit is this model's effective per-request context (its
    // per-model `maxContext` override when set, else the server default passed in).
    const std::uint32_t max_model_len = model.overrides.max_context.value_or(default_max_context);
    const Json payload = {{"id", model.id},
                          {"object", "model"},
                          {"created", created},
                          {"owned_by", "ninfer"},
                          {"max_model_len", max_model_len},
                          {"status", Json{{"value", loaded ? "loaded" : "unloaded"}}}};
    return payload.dump();
}

std::string make_slots_list(std::uint32_t max_concurrency, std::string_view loaded_id,
                           std::string_view model_filter, int in_flight_total, bool is_swapping,
                           int prefilling_requests) {
    // The target model: an explicit ?model= filter, else the resident model. Only the resident
    // model can run generations (one GPU, one resident); a non-resident (or absent) target has no
    // in-flight generations, so its slots are all idle (a load/swap is required to use them).
    // This keeps the webui's "is the server free?" probe (areAllSlotsIdle) honest: it is true
    // exactly when no generation for that model is running.
    const std::string target =
        !model_filter.empty() ? std::string(model_filter) : std::string(loaded_id);
    const bool target_is_resident = !target.empty() && target == loaded_id;
    // Slots actually in use: the resident's in-flight generations, bounded by the concurrency.
    const int busy =
        target_is_resident ? std::min(in_flight_total, static_cast<int>(max_concurrency)) : 0;
    const auto slot_state = [&](int index) {
        if (!target_is_resident) { return std::string("idle"); } // a load/swap is required
        if (is_swapping) { return std::string("loading"); } // the model set is in transition
        if (index < busy) { return prefilling_requests > 0 ? std::string("prefill")
                                                            : std::string("decode"); }
        return std::string("idle");
    };
    const Json slots = [&] {
        Json array = Json::array();
        for (std::uint32_t i = 0; i < max_concurrency; ++i) {
            const bool processing = static_cast<int>(i) < busy;
            array.push_back(Json{{"id", i},
                                 {"model", target},
                                 {"is_processing", processing},
                                 {"state", slot_state(static_cast<int>(i))}});
        }
        return array;
    }();
    return slots.dump();
}

std::string make_props_stub(const ServeOptions& options, const std::string& model_id) {
    // The webui expects the llama.cpp server-properties shape; NInfer fills the fields it knows
    // from the serve options and leaves the rest at neutral defaults so the panel renders.
    const auto optional_float = [](std::optional<float> value) {
        return value.has_value() ? Json(static_cast<double>(*value)) : Json();
    };
    const auto optional_int = [](std::optional<std::int32_t> value) {
        return value.has_value() ? Json(*value) : Json();
    };

    const auto& ov = options.sampling_overrides;
    Json params = Json::object();
    params["n_predict"]         = options.default_max_tokens;
    params["seed"]              = ov.seed ? static_cast<double>(*ov.seed) : 0;
    params["temperature"]       = ov.temperature ? static_cast<double>(*ov.temperature) : 0;
    params["top_p"]             = optional_float(ov.top_p);
    params["top_k"]             = optional_int(ov.top_k);
    params["min_p"]             = optional_float(ov.min_p);
    params["presence_penalty"]  = optional_float(ov.presence_penalty);
    params["frequency_penalty"] = optional_float(ov.frequency_penalty);
    // Neutral stub values for fields NInfer does not expose through its sampling overrides.
    params["repeat_penalty"]     = Json();
    params["repeat_last_n"]      = Json();
    params["dynatemp_exponent"]  = Json();
    params["typical_p"]          = Json();
    params["mlock"]              = false;
    params["mmap"]               = false;
    params["n_gpu_layers"]       = -1;
    params["n_batch"]            = 2048;
    params["n_ubatch"]           = 512;
    params["n_threads"]          = 8;
    params["n_threads_batch"]    = 8;
    params["n_cpu_mempolicy"]    = 0;
    params["n_probs"]            = 0;
    params["logdir"]             = "";
    params["main_gpu"]           = 0;
    params["no_kv_transfer"]     = false;
    params["flash_attn"]         = false;
    params["rag_ch"]             = 0;
    params["cache_type_k"]       = "f16";
    params["cache_type_v"]       = "f16";
    params["cache_state_tiling"] = false;
    params["cache_reuse"]        = 0;
    params["numa"]               = "none";
    params["n_par"]              = 1;
    params["n_ctx"]              = options.max_context;
    params["n_ctx_overwrite"]    = 0;
    params["n_batch_overwrite"]  = 0;
    params["min_keep"]           = 0;
    params["stop"]               = Json::array();
    params["jinja"]              = false;
    params["reasoning_format"]   = "deepseek";
    params["mcp_server"]         = "";
    params["image"]              = Json::array();
    params["audio"]              = Json::array();
    params["max_tokens"]         = options.default_max_tokens;
    params["grammar"]            = "";
    params["penalty_prompt"]     = Json::array();
    params["mirostat"]           = 0;
    params["mirostat_mu"]        = 2.0;
    params["mirostat_tau"]       = 0.5;
    params["mirostat_skill"]     = 1.0;
    params["n_save"]             = 0;
    params["n_keep"]             = 0;
    params["lcm"]                = false;
    params["lcm_s"]              = 1.0;
    params["smin"]               = 0.0;
    params["smoe"]               = 0.0;
    params["pmin"]               = 0.0;

    Json next_token = Json{{"id", 0},
                           {"token", "<|endoftext|>"},
                           {"text", ""},
                           {"prob", 0.0},
                           {"utf8", ""},
                           {"lprob", 0.0},
                           {"tprob", 0.0},
                           {"timestart", 0.0},
                           {"timeend", 0.0},
                           {"is_unknown", false},
                           {"content", ""},
                           {"type", 0},
                           {"is_special", false}};
    Json models = Json::array();
    models.push_back(Json{{"name", model_id}, {"id", model_id}, {"meta_path", ""}});

    Json props = Json::object();
    props["default_generation_settings"] = {
        {"id", 0},
        {"id_task", 0},
        {"n_ctx", static_cast<int>(options.max_context)},
        {"speculative", options.speculative.backend != SpeculativeBackend::None},
        {"is_processing", false},
        {"params", params},
        {"prompt", ""},
        {"next_token", Json::array({std::move(next_token)})},
    };
    props["slots"]        = Json::array();
    props["server"]       = Json{{"models", std::move(models)}};
    props["status"]       = Json{{"msg", "ok"},
                                  {"slots_total", 0},
                                  {"slots_used", 0},
                                  {"n_ctx_max", options.max_context},
                                  {"n_ctx_now", options.max_context},
                                  {"n_cpu_mempolicy", 0},
                                  {"ngl", -1},
                                  {"use_mmap", false},
                                  {"use_mlock", false}};
    // The llama-ui webui enters multi-model (router) mode only when the /props role is "router";
    // otherwise it falls back to single-model mode and lists only one model. Report "router" when
    // the config declares more than one model so the webui's model browser shows every configured
    // model (the list is served by /v1/models); a single-model config keeps the single-model "model" role.
    props["role"] = options.model_config.models.size() > 1 ? "router" : "model";
    props["public_key"]   = "unknown";
    props["capabilities"] = Json::array();
    Json modalities = Json::object();
    modalities["vision"] = options.enable_vision;
    modalities["audio"]  = false;
    props["modalities"] = modalities;
    return props.dump();
}

std::string make_error_body(const ApiError& error) {
    Json rendered     = {{"message", error.message}, {"type", error.type}};
    rendered["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    rendered["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    return Json{{"error", std::move(rendered)}}.dump();
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string new_openai_chat_completion_id() { return chat_identifier("chatcmpl-"); }

std::string new_openai_chat_tool_call_id() { return chat_identifier("call_"); }

std::string new_openai_request_id() { return responses_identifier("req"); }

std::string new_openai_response_id() { return responses_identifier("resp"); }

std::string new_openai_response_item_id(std::string_view prefix) {
    return responses_identifier(prefix);
}

} // namespace ninfer::serve
