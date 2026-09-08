#include "serve/tool_registry.h"

namespace ninfer::serve {

void ToolRegistry::upsert(Tool tool, ToolHandler handler) {
    std::lock_guard lock(mu_);
    const auto key = tool.name;
    const auto existing = index_.find(key);
    if (existing != index_.end()) {
        // Update in place (preserve insertion order): overwrite the tool + handler.
        Entry& entry = entries_[existing->second];
        entry.tool   = std::move(tool);
        entry.handler = std::move(handler);
        return;
    }
    const std::size_t position = entries_.size();
    entries_.push_back(Entry{.tool = std::move(tool), .handler = std::move(handler)});
    index_.emplace(key, position);
}

std::optional<Tool> ToolRegistry::find(std::string_view name) const {
    std::lock_guard lock(mu_);
    const auto it = index_.find(std::string(name));
    if (it == index_.end()) { return std::nullopt; }
    return entries_[it->second].tool;
}

std::vector<Tool> ToolRegistry::list() const {
    std::lock_guard lock(mu_);
    std::vector<Tool> tools;
    tools.reserve(entries_.size());
    for (const auto& entry : entries_) { tools.push_back(entry.tool); }
    return tools;
}

ToolResult ToolRegistry::execute(std::string_view name, const nlohmann::ordered_json& params) const {
    ToolHandler handler;
    {
        std::lock_guard lock(mu_);
        const auto key = std::string(name);
        const auto it = index_.find(key);
        // Unknown tool, or a schema-only tool (registered with no handler): an honest execution
        // result ("not available"), NOT a transport error and NOT a hardcoded no-op.
        if (it == index_.end()) {
            return ToolResult{.ok = false,
                              .plain_text_response = {},
                              .error = "tool '" + key + "' is not available on this server"};
        }
        handler = entries_[it->second].handler; // copy out under the lock; may be empty
    }
    if (!handler) {
        // Registered but no handler attached (NInfer ships no built-in executable tools): the
        // dispatch mechanism is real, but this specific tool has nothing to run.
        return ToolResult{.ok = false,
                          .plain_text_response = {},
                          .error = "tool '" + std::string(name) + "' is not available on this server"};
    }
    return handler(name, params);
}

} // namespace ninfer::serve
