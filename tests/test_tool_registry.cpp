// Host-side (no GPU) test for the serving-owned ToolRegistry (llama-parity real-design §4, step
// 7). Verifies the registry is a REAL mutable structure -- NOT a hardcoded response: it starts
// empty (GET /tools -> []), upsert makes GET reflect a registered tool, and the execution
// dispatch is real (look-up -> handler -> result): an unhandled/unknown tool yields a structured
// {error}, while a tool with a registered handler yields {plain_text_response}. The response
// shapes mirror exactly what the HTTP handlers emit (the webui's FI.PLAIN_TEXT / FI.ERROR
// contract, design §0.6). No Engine, no model, no network -- a pure serving-state object. Bare
// main() + a small CHECK helper, matching test_stream_registry.cpp.

#include "serve/tool_registry.h"

#include <cstdio>
#include <string>
#include <vector>

namespace ns = ninfer::serve;
using ns::Tool;
using ns::ToolHandler;
using ns::ToolRegistry;
using ns::ToolResult;

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

// A trivial in-test handler: it proves the dispatch is real by (a) receiving the tool name and
// (b) reading params through, returning a deterministic plain_text_response.
ToolResult echo_handler(std::string_view name, const nlohmann::ordered_json& params) {
    if (!params.is_object()) { return ToolResult{.ok = false, .error = "params must be an object"}; }
    std::string greeting =
        (params.contains("greeting") && params["greeting"].is_string())
            ? params["greeting"].get<std::string>()
            : std::string("(none)");
    return ToolResult{.ok = true, .plain_text_response = "greeting:" + std::string(name) + ":" + greeting};
}

// NInfer ships no built-in server tools: a fresh registry lists nothing, so GET /tools -> [].
int test_starts_empty() {
    int failures = 0;
    ToolRegistry registry;
    const auto tools = registry.list();
    failures += check(tools.empty(), "registry must start empty (no built-in server tools)");
    failures += check(registry.find("anything") == std::nullopt,
                      "find on an empty registry must miss");
    // GET /tools handler shape for the empty baseline: a JSON array of {name, definition, enabled}
    // -- empty here. (Mirrors handle_tools.)
    nlohmann::json get_response = nlohmann::json::array();
    for (const auto& tool : tools) {
        get_response.push_back(nlohmann::json{{"name", tool.name},
                                              {"definition", tool.definition.is_null()
                                                               ? nlohmann::ordered_json::object()
                                                               : tool.definition},
                                              {"enabled", tool.enabled}});
    }
    failures += check(get_response.is_array() && get_response.empty(),
                      "GET /tools must be [] on an empty registry");
    return failures;
}

// upsert a schema-only tool -> GET /tools reflects it (name / definition / enabled round-trip).
int test_upsert_reflected_in_list() {
    int failures = 0;
    ToolRegistry registry;
    const nlohmann::ordered_json definition = nlohmann::ordered_json::parse(
        R"({"description":"search the web","parameters":{"type":"object"}})");
    registry.upsert(Tool{.name = "search", .definition = definition, .enabled = true});

    const auto tools = registry.list();
    failures += check(tools.size() == 1, "upsert did not register exactly one tool");
    if (!tools.empty()) {
        failures += check(tools[0].name == "search", "upserted tool name mismatch");
        failures += check(tools[0].enabled, "upserted tool enabled flag lost");
        failures += check(tools[0].definition == definition, "upserted tool definition lost");
    }
    const auto found = registry.find("search");
    failures += check(found.has_value() && found->name == "search", "find did not locate the tool");

    // GET /tools handler shape for a registered tool: an array with the tool's fields present.
    nlohmann::json get_response = nlohmann::json::array();
    for (const auto& tool : tools) {
        get_response.push_back(nlohmann::json{{"name", tool.name},
                                              {"definition", tool.definition.is_null()
                                                               ? nlohmann::ordered_json::object()
                                                               : tool.definition},
                                              {"enabled", tool.enabled}});
    }
    failures += check(get_response.size() == 1 && get_response[0].is_object() &&
                          get_response[0].contains("name") && get_response[0].contains("definition") &&
                          get_response[0].contains("enabled"),
                      "GET /tools must expose {name, definition, enabled} per registered tool");
    return failures;
}

// upsert is idempotent-by-name (an update, not a duplicate): re-upserting the same name keeps one
// entry and overwrites its definition/enabled.
int test_upsert_updates_in_place() {
    int failures = 0;
    ToolRegistry registry;
    registry.upsert(Tool{.name = "search", .definition = nlohmann::ordered_json::object(),
                         .enabled = true});
    registry.upsert(Tool{.name = "search",
                         .definition = nlohmann::ordered_json{{"v", 2}},
                         .enabled = false});
    const auto tools = registry.list();
    failures += check(tools.size() == 1, "re-upsert of the same name must not duplicate");
    if (!tools.empty()) {
        failures += check(!tools[0].enabled, "re-upsert did not overwrite enabled");
        failures += check(tools[0].definition.contains("v"), "re-upsert did not overwrite definition");
    }
    return failures;
}

// A registered (stub) tool WITH a handler: execute() is a real dispatch (look-up -> handler ->
// result) that receives params and returns {plain_text_response}.
int test_execute_registered_handler() {
    int failures = 0;
    ToolRegistry registry;
    registry.upsert(Tool{.name = "weather", .definition = nlohmann::ordered_json::object(),
                         .enabled = true},
                    echo_handler);

    const ToolResult result = registry.execute("weather", nlohmann::ordered_json{{"greeting", "hi"}});
    failures += check(result.ok, "registered handler must report success");
    failures += check(result.plain_text_response == "greeting:weather:hi",
                      "handler did not receive name + params and produce a plain_text_response");

    // POST /tools handler shape for success: a single {plain_text_response} object.
    nlohmann::json response;
    if (result.ok) {
        response["plain_text_response"] = result.plain_text_response;
    } else {
        response["error"] = result.error;
    }
    failures += check(response.is_object() && response.contains("plain_text_response") &&
                          !response.contains("error") &&
                          response["plain_text_response"].get<std::string>() == "greeting:weather:hi",
                      "POST /tools success shape must be {plain_text_response}");
    return failures;
}

// A registered tool with NO handler (schema-only) and an unknown tool both yield the honest
// structured {error} (NInfer ships no built-in executable tools).
int test_execute_unhandled_and_unknown() {
    int failures = 0;
    ToolRegistry registry;
    registry.upsert(Tool{.name = "search", .definition = nlohmann::ordered_json::object(),
                         .enabled = true}); // schema-only, no handler

    const ToolResult unhandled = registry.execute("search", nlohmann::ordered_json::object());
    failures += check(!unhandled.ok && unhandled.error.find("search") != std::string::npos,
                      "an unhandled (schema-only) tool must yield a structured {error}");

    const ToolResult unknown = registry.execute("nope", nlohmann::ordered_json::object());
    failures += check(!unknown.ok && unknown.error.find("nope") != std::string::npos,
                      "an unknown tool must yield a structured {error}");

    // POST /tools handler shape for the error case: a single {error} object.
    nlohmann::json response;
    if (unknown.ok) {
        response["plain_text_response"] = unknown.plain_text_response;
    } else {
        response["error"] = unknown.error;
    }
    failures += check(response.is_object() && response.contains("error") &&
                          !response.contains("plain_text_response"),
                      "POST /tools error shape must be {error}");
    return failures;
}

} // namespace

int main() {
    int failures = test_starts_empty() + test_upsert_reflected_in_list() +
                   test_upsert_updates_in_place() + test_execute_registered_handler() +
                   test_execute_unhandled_and_unknown();
    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
