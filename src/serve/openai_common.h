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

std::string make_models_list(const std::vector<ModelConfig>& models, std::int64_t created,
                             std::uint32_t default_max_context, std::string_view loaded_id);
std::string make_model_object(const ModelConfig& model, std::int64_t created,
                              std::uint32_t default_max_context, bool loaded);
std::string make_props_stub(const ServeOptions& options, const std::string& model_id);
// llama.cpp(-swap) webui parity (design §3): the GET /slots live slot state. Returns a JSON array
// of one slot object per configured concurrency slot (max_concurrency). Pure (no router access):
// the caller gathers the live inputs and passes them in, so the slot-shaping is host-testable.
//   - model_filter  : the ?model= query value ("" = no filter -> the resident model).
//   - loaded_id     : the resident model id ("" = no-resident / mid-swap).
//   - in_flight_total : the resident's granted-but-not-completed generations (ModelRouter::in_flight).
//   - is_swapping   : a model load/swap is in flight (ModelRouter::is_swapping).
//   - prefilling_requests : the resident backend's live prefill-lane occupancy (RuntimeStats).
// Each slot: {"id", "model", "is_processing", "state"}. is_processing is the only field the webui
// reads (areAllSlotsIdle does .every(s=>!s.is_processing)); the rest are informational.
std::string make_slots_list(std::uint32_t max_concurrency, std::string_view loaded_id,
                            std::string_view model_filter, int in_flight_total, bool is_swapping,
                            int prefilling_requests);

std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace ninfer::serve
