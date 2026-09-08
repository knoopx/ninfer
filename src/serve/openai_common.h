#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/model_config.h"
#include "serve/request.h"
#include "serve/request_json.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
};

[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const RequestJson& value,
                                                        std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len, const ModelMetadata& metadata);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len, const ModelMetadata& metadata);
// Multi-model (no-resident) discovery: list every configured model, marking the loaded one. Each
// entry carries the webui status marker (status.value = loaded|unloaded); no engine metadata
// (nothing is loaded at discovery time).
std::string make_models_list(const std::vector<ModelConfig>& models, std::int64_t created,
                             std::uint32_t max_model_len, std::string_view loaded_id);
std::string make_model_object(const ModelConfig& model, std::int64_t created,
                              std::uint32_t max_model_len, bool loaded);
// The webui's live /slots probe: the resident model's slot occupancy (pure, host-testable).
std::string make_slots_list(std::uint32_t max_concurrency, std::string_view loaded_id,
                            std::string_view model_filter, int in_flight_total, bool is_swapping,
                            int prefilling_requests);
// The webui's /props panel: the llama.cpp server-properties shape filled from the serve options
// (neutral defaults for fields NInfer does not track) so the panel renders.
std::string make_props_stub(const ServeOptions& options, const std::string& model_id);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace ninfer::serve
