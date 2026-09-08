#pragma once

// Multi-model serve configuration. The schema mirrors llama-swap's `internal/config` (Config,
// ModelConfig, GroupConfig, RoutingConfig, ModelCapConfig) with the per-model `cmd` translated to
// ninfer-native `artifact` + `identity` (the `.ninfer` artifact path and its registered identity).
// STEP 1 (this step) consumes `models` (the ordered model list) and the resident model's artifact;
// `routing`/`groups`/`globalTTL`/`unloadTimeout`/`healthCheckTimeout`/`startPort` are parsed and
// stored for the deferred router step.

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// llama-swap ModelCapConfig. JSON keys: `in`, `out`, `tools`, `reranker`, `context`.
struct ModelCapConfig {
    std::vector<std::string> input;
    std::vector<std::string> output;
    bool tools    = false;
    bool reranker = false;
    int context   = 0;
};

// llama-swap ModelConfig (the value of the `models` map). JSON keys: `artifact` (was `cmd`),
// `identity`, `aliases`, `ttl`, `name`, `description`, `checkEndpoint`,
// `capabilities`. `id` is the `models` map key (the public model id).
// Per-model engine parameter overrides. Each field is std::optional: nullopt means "use the
// global ServeOptions value"; a set value overrides the global for this model only. The router
// applies these in EngineModelBackend's constructor by copying the base ServeOptions and then
// overwriting each set field before constructing the GenerationService.
struct EngineOverrides {
    std::optional<std::uint32_t> max_context;
    std::optional<int> default_max_tokens;
    std::optional<KvCapacityPolicy> kv_capacity;
    std::optional<KvCacheStorage> kv_cache;
    std::optional<SpeculativeOptions> speculative;
    std::optional<std::uint32_t> prefill_chunk;
    std::optional<bool> enable_vision;
};

struct ModelConfig {
    std::string id;
    std::string artifact;
    std::string identity;
    std::vector<std::string> aliases;
    int ttl               = 0;
    std::string name;
    std::string description;
    std::string check_endpoint = "/health";
    ModelCapConfig capabilities;
    EngineOverrides overrides;
};

// llama-swap GroupConfig. JSON keys: `swap`, `exclusive`, `persistent`, `members`.
struct GroupConfig {
    bool swap       = true;
    bool exclusive  = true;
    bool persistent = false;
    std::vector<std::string> members;
};

// llama-swap SchedulerConfig / RouterConfig (the `routing.scheduler` / `routing.router` objects).
struct SchedulerConfig {
    std::string use = "fifo";
};
struct RouterConfig {
    std::string use = "group";
};
// llama-swap RoutingConfig. JSON keys: `scheduler` {`use`}, `router` {`use`}.
struct RoutingConfig {
    SchedulerConfig scheduler;
    RouterConfig router;
};

// llama-swap Config (the ninfer-relevant subset). JSON keys: `models` (object, key = public model
// id, value = ModelConfig), `groups`, `routing`, `globalTTL`, `unloadTimeout`,
// `healthCheckTimeout`, `startPort`.
struct Config {
    // File order is preserved (parsed from an ordered JSON object); the first model is the
    // resident (STEP 1 placeholder until the router step selects it dynamically).
    std::vector<ModelConfig> models;
    std::map<std::string, GroupConfig> groups;
    RoutingConfig routing;
    int global_ttl          = 0;
    int unload_timeout      = 10;
    int health_check_timeout = 120;
    int start_port          = 5800;

    [[nodiscard]] const ModelConfig* model(std::string_view id) const;
    // The resident model (the first in file order). Throws if the config has no models.
    [[nodiscard]] const ModelConfig& resident() const;
};

// Parses a JSON config document into a Config. Throws std::invalid_argument on malformed JSON,
// a non-object top level, or a model missing its required `artifact`.
Config parse_config(std::string_view json_text);

// Reads the JSON file at `path` and parses it. Throws std::invalid_argument if the file cannot be
// opened or does not parse.
Config load_config(const std::filesystem::path& path);

} // namespace ninfer::serve
