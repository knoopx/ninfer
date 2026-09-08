#pragma once

#include "serve/generation_service.h"
#include "serve/model_router.h"
#include "serve/model_sse_hub.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"
#include "serve/stream_registry.h"
#include "serve/tool_registry.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::serve {

void write_openai_error(httplib::Response& response, const ApiError& error);
void write_anthropic_error(httplib::Response& response, const ApiError& error,
                           const std::string& request_id);

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;

class HttpServer {
public:
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger);
    // Clears the /models/sse status hook on the attached router (the router outlives this server,
    // so its final unload must not fire a hook into a destroyed server). The stats reporter thread
    // is already stopped by listen() before the server is torn down.
    ~HttpServer();

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(ModelRouter& router);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, RequestLogContext context);

        void done(const GenerationOutcome& outcome);
        void failure(const RequestFailure& failure);
        void response_failure(const RequestFailure& failure);

        [[nodiscard]] std::uint64_t request_id() const noexcept { return context_.id; }

    private:
        enum class State : std::uint8_t {
            Pending,
            Done,
            Error,
        };

        [[nodiscard]] bool claim(State terminal) noexcept;

        HttpServer* owner_ = nullptr;
        RequestLogContext context_;
        std::atomic<State> state_{State::Pending};
    };

    [[nodiscard]] std::shared_ptr<RequestLifecycle> begin_request(RequestLogContext context);

    void register_routes();
    void mount_webui(const std::string& webui_dir);
    void register_webui_mime();
    [[nodiscard]] bool webui_spa_path(const std::string& path) const;
    [[nodiscard]] bool is_api_path(const std::string& path) const;
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;
    void handle_model_load(const httplib::Request& req, httplib::Response& res) const;
    void handle_model_unload(const httplib::Request& req, httplib::Response& res) const;
    void handle_model_sse(const httplib::Request& req, httplib::Response& res);
    // llama-parity live /slots (design §3, step 8): read-only slot-state introspection of the
    // ModelRouter + the resident backend (in_flight/loaded_id/is_swapping + runtime_stats). No new
    // state, no scheduler mutation, no preemption; the slot-shaping is a pure function
    // (make_slots_list) that is host-testable without a GPU.
    void handle_slots(const httplib::Request& req, httplib::Response& res) const;
    // llama-parity stream-registry routes (design §1.3, steps 3-6): the webui's lookup / resume /
    // cancel routes consume StreamRegistry (the serving-owned registry of in-flight streaming
    // completions). All four are serving-only protocol translation over the registry + the existing
    // CancellationView (no Engine/Core change, no preemption).
    void handle_stream_lookup(const httplib::Request& req, httplib::Response& res);
    void handle_stream_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_completions_control(const httplib::Request& req, httplib::Response& res);
    void handle_stream(const httplib::Request& req, httplib::Response& res);
    // llama-parity tool-registry routes (design §4.3, step 7): GET /tools lists the serving-owned
    // ToolRegistry (an array of {name, definition, enabled}); POST /tools EXECUTES a registered
    // tool ({tool, params} -> {plain_text_response} on success, {error} otherwise). Both are
    // serving-only protocol translation over the registry (no Engine/Core change; a future
    // Engine-backed tool handler would cross the boundary -- flagged, not built).
    void handle_tools(const httplib::Request& req, httplib::Response& res) const;
    void handle_tools_execute(const httplib::Request& req, httplib::Response& res);
    void handle_props(const httplib::Request& req, httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();

    ModelRouter* router_ = nullptr;
    ModelSseHub model_sse_hub_; // serving-owned emitter for GET /models/sse (design §3.3)
    // Serving-owned registry of in-flight streaming chat completions (llama-parity design §1,
    // step 1): tees the SSE bytes the handler renders + carries the owner cancel token. It does
    // not schedule or run any generation (no Runtime/Core boundary crossing). The 4 stream/cancel
    // endpoint handlers (lookup / GET / DELETE / control) consume it in the next step; this step
    // builds the registry + wires the producer only.
    StreamRegistry stream_registry_;
    // Serving-owned registry of server-side tools (llama-parity design §4, step 7): a real,
    // mutable, in-memory structure (empty at startup -- NInfer ships no built-in server tools).
    // GET /tools lists it; POST /tools dispatches execution through it. Serving-owned; does not
    // reach into Core/Runtime (a future Engine-backed tool handler would cross the boundary).
    ToolRegistry tool_registry_;
    ServeOptions options_;
    std::string public_model_id_;
    bool webui_serving_ = false; // true once a static webui dir is mounted
    std::string webui_index_html_; // cached index.html for the SPA fallback
    OpenAIResponsesStore openai_responses_store_;
    OperationalLog operational_log_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
};

} // namespace ninfer::serve
