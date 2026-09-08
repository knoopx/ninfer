#pragma once

// ToolRegistry -- the serving-owned registry of server-side tools (llama-parity real-design §4,
// implementation step 7). It is a real, mutable, serving-owned state object (owned by HttpServer,
// beside StreamRegistry / ModelSseHub), NOT a hardcoded response: NInfer ships no built-in
// server tools (it is a client-side function-calling engine), so the registry starts EMPTY, but
// the machinery (upsert / list / dispatch) is genuine and a real tool set can be registered.
//
// Grounded in the bundled webui's actual contract (design §0.6):
//   GET  /tools  ->  NPe.list(): the client reads each element's `.definition` (a JSON schema),
//                    and optionally `.name` / `.enabled`.
//   POST /tools  ->  NPe.executeTool: the client posts {tool, params} and reads
//                    `plain_text_response` (isError:false) or `error` (isError:true). POST is
//                    EXECUTION (the llama.cpp server-tools model), not registration.
//
// Ownership: entirely in serving (protocol translation + a serving-owned registry). Tool
// execution runs on the server, not in the Engine. A future handler that needs to call the
// Engine (e.g. a "run a second model" tool) would cross the Runtime/Core boundary (design §4.3);
// the baseline executor (empty registry / "not available" / user-registered handlers) stays in
// serving. No boundary is crossed here.

#include <nlohmann/json.hpp>

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::serve {

// One registered server-side tool: its name, its JSON schema definition (what GET /tools exposes
// and the webui's ToolsStore maps via `.definition`), and an enabled flag (the webui reads
// `.name` / `.definition` / `.enabled`). `definition` is ordered JSON so its field order is stable
// across serializations (matches the codebase's RequestJson = nlohmann::ordered_json).
struct Tool {
    std::string name;
    nlohmann::ordered_json definition; // JSON schema {description, parameters, ...}
    bool enabled = true;
};

// The result of a POST /tools execution dispatch (design §0.6 / §4.3). Either carries a
// `plain_text_response` (success; the webui renders isError:false) or an `error` message
// (the webui renders isError:true). The registry's execute() always returns one of these.
struct ToolResult {
    bool ok = false;
    std::string plain_text_response; // set when ok == true
    std::string error;               // set when ok == false
};

// A handler a server registers for a named tool: given params, produce a ToolResult. This is the
// real dispatch seam: NInfer ships NO built-in handlers (the registry starts empty), but the
// mechanism is genuine -- a handler can be attached at upsert time and execute() will look up the
// tool and dispatch to its handler. A handler that must reach the Engine would cross the
// Runtime/Core boundary (design §4.3); the baseline stays entirely in serving.
using ToolHandler =
    std::function<ToolResult(std::string_view name, const nlohmann::ordered_json& params)>;

// Thread-safe registry of server-side tools. Map access under mu_; entries are kept in insertion
// order (order_ + index_) so list() is deterministic. A tool may be registered schema-only
// (upsert with no handler) -- it then appears in GET /tools but execute() reports it "not
// available"; a tool with a registered handler is one POST /tools can actually execute.
class ToolRegistry {
public:
    // Register or update a tool by name (upsert). When `handler` is provided, execute() dispatches
    // to it; when absent, execute() of that tool yields the "not available" error. Insertion order
    // is preserved for existing names (an upsert never reorders).
    void upsert(Tool tool, ToolHandler handler = {});

    // Fetch a tool by name (nullopt when unknown) -- the registry's look-up primitive.
    [[nodiscard]] std::optional<Tool> find(std::string_view name) const;

    // All registered tools in registration order (for GET /tools: an array of {name, definition,
    // enabled}). Empty at startup (NInfer ships no built-in tools).
    [[nodiscard]] std::vector<Tool> list() const;

    // POST /tools dispatch (look-up -> handler -> result). Look up `name`; if a handler is
    // registered for it, run it with `params` and return its result (plain_text_response on
    // success, or the handler's own error). When the tool is unknown or has no handler, return a
    // well-formed {error: "tool '<name>' is not available on this server"} -- an honest execution
    // result, not a hardcoded string (the client renders isError:true, not a transport error).
    [[nodiscard]] ToolResult execute(std::string_view name, const nlohmann::ordered_json& params) const;

private:
    struct Entry {
        Tool tool;
        ToolHandler handler;
    };
    mutable std::mutex mu_;
    std::vector<Entry> entries_;                   // insertion order (drives list())
    std::unordered_map<std::string, std::size_t> index_; // name -> position in entries_
};

} // namespace ninfer::serve
