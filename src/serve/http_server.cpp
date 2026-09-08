#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) { return path.starts_with("/v1/messages"); }

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

// Parse the ?from=N byte offset for GET /v1/stream/{id} (design §0.3, §1.3): N is a non-negative
// byte offset into the buffered SSE stream (0 = fresh attach; N = bytes already received). An
// absent, non-numeric, or out-of-range value resumes from the start.
std::size_t parse_stream_offset(const std::string& value) {
    if (value.empty()) { return 0; }
    for (const char c : value) {
        if (c < '0' || c > '9') { return 0; }
    }
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (const std::exception&) {
        return 0; // offset overflows size_t; resume from the start
    }
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view path) noexcept {
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && request.path.rfind("/v1/messages", 0) == 0) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger)) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    if (!options_.webui_dir.empty()) {
        mount_webui(options_.webui_dir);
        register_webui_mime();
    }
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock = std::chrono::steady_clock;
    // Read the resident backend's runtime stats. The router can be mid-swap (the resident is
    // destroyed before the target is installed) or have TTL-unloaded an idle resident; in either
    // window router_->resident() is null. A null read skips the sample and carries the previous
    // snapshot forward so a resumed interval reports no spurious delta. A swap that completes
    // WITHIN one interval produces two non-null samples of DIFFERENT models; the sampled model id
    // is tracked alongside the snapshot and a model change resets the baseline the same way.
    struct ResidentSample {
        bool present = false;
        std::string model_id; // the resident backend's id (the identity of the sampled model)
        ninfer::RuntimeStats stats;
    };
    auto read_resident_stats = [this] {
        const auto resident = router_->resident();
        if (resident == nullptr) {
            return ResidentSample{};
        }
        // Take the model id from the SAME resident copy whose stats are sampled, so a swap that
        // completes mid-interval is detected (a different id resets the baseline below) and a
        // model's stats are never paired with another model's baseline.
        return ResidentSample{true, resident->id(), resident->runtime_stats()};
    };
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;
    ResidentSample previous = read_resident_stats();

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        // Reap finished streaming-chat streams whose retention window has elapsed (design §1.2):
        // rides this reporter's cadence so a done entry's buffered SSE bytes are dropped after the
        // window instead of accumulating. Runs every tick (even a mid-swap skip), so a finished
        // stream is reaped on schedule regardless of resident state. The per-tick count is not a
        // metric here, so it is intentionally discarded.
        (void)stream_registry_.reap(Clock::now());

        const auto current = read_resident_stats();
        if (!current.present) {
            // Mid-swap / idle-unloaded: no resident to sample. Drop the stale snapshot (it may be
            // a different model's stats after a swap, which would make a spurious cross-model
            // delta) and skip the interval; the next valid sample starts fresh.
            previous.present = false;
            previous_time    = Clock::now();
            next_deadline   += interval;
            continue;
        }
        if (!previous.present || previous.model_id != current.model_id) {
            // First valid sample (resuming from a null window) or a swap completed within this
            // interval (two non-null samples of DIFFERENT models): reset the baseline so no
            // cross-model throughput delta is reported, exactly like the null case.
            previous      = current;
            previous_time = Clock::now();
        }
        const Clock::time_point now = Clock::now();
        const ThroughputReport report =
            make_throughput_report(previous.stats, current.stats,
                                   std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const auto current = read_resident_stats();
    if (previous.present && current.present && previous.model_id == current.model_id) {
        const Clock::time_point now = Clock::now();
        const ThroughputReport tail = make_throughput_report(
            previous.stats, current.stats,
            std::chrono::duration<double>(now - previous_time).count());
        // The exact partial interval remains useful to measurement consumers. Pretty throughput is
        // a fixed-cadence operational record and deliberately has no irregular shutdown tail.
        if (report_has_activity(tail)) { request_jsonl_.write_throughput(tail); }
    }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::mount_webui(const std::string& webui_dir) {
    std::ifstream index(webui_dir + "/index.html", std::ios::binary);
    if (!index) {
        throw std::runtime_error("webui dir has no index.html: " + webui_dir);
    }
    webui_index_html_ =
        std::string((std::istreambuf_iterator<char>(index)), std::istreambuf_iterator<char>());
    webui_serving_ = true;
    if (!server_.set_mount_point("/", webui_dir)) {
        throw std::runtime_error("cannot mount webui directory: " + webui_dir);
    }
}

void HttpServer::register_webui_mime() {
    // The vendored httplib's built-in map covers the stock webui's asset types; pin the ones the
    // UI's runtime loading depends on.
    server_.set_file_extension_and_mimetype_mapping("js", "text/javascript");
    server_.set_file_extension_and_mimetype_mapping("css", "text/css");
    server_.set_file_extension_and_mimetype_mapping("html", "text/html");
    server_.set_file_extension_and_mimetype_mapping("json", "application/json");
    server_.set_file_extension_and_mimetype_mapping("svg", "image/svg+xml");
    server_.set_file_extension_and_mimetype_mapping("ico", "image/x-icon");
}

bool HttpServer::webui_spa_path(const std::string& path) const {
    // The SPA fallback must only catch client-side routes: the static file handler already
    // served every real asset, and API paths (/v1/...), /props, /health, and hashed bundle
    // paths (_app/...) are never SPA routes. A missing _app file or an unknown /v1 path must
    // keep falling through to its natural 404.
    if (path.size() < 2 || path[0] != '/') { return false; }
    if (path == "/") { return false; }
    if (path.rfind("/v1", 0) == 0 && (path.size() == 3 || path[3] == '/')) { return false; }
    if (path == "/props" || path == "/health" || path == "/slots" || path == "/tools" ||
        path == "/v1/streams/lookup") {
        return false;
    }
    if (path.rfind("/models/", 0) == 0) { return false; } // ModelRouter load/unload wrappers
    if (path.rfind("/_app/", 0) == 0) { return false; }
    if (path.find_first_of('.') != std::string::npos) { return false; }
    return true;
}

bool HttpServer::is_api_path(const std::string& path) const {
    // The endpoints that require an API key. Deliberately dot-free: static assets
    // (favicon.ico, app.js) and the UI shell (/, index.html, SPA routes) are never
    // API paths, so the UI loads freely even when --api-key is set (llama-server parity).
    if (path.rfind("/v1", 0) == 0 && (path.size() == 3 || path[3] == '/')) { return true; }
    if (path == "/props" || path == "/slots" || path == "/tools") { return true; }
    if (path.rfind("/v1/streams/lookup", 0) == 0) { return true; }
    if (path.rfind("/models/", 0) == 0) { return true; } // ModelRouter load/unload wrappers
    if (path.rfind("/_app/", 0) == 0) { return false; } // served by the static mount
    return false;
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        // SPA fallback: the static file handler already served every real asset and every
        // registered API route has already been tried, so an unmatched 404 on a GET/HEAD is a
        // client-side route (e.g. /chat/123). Hand it the SPA shell; leave every other error
        // (413, 405 on a real API path, 404 on a missing asset or API path) to the JSON handler.
        if (webui_serving_ && response.status == 404 &&
            (request.method == "GET" || request.method == "HEAD") &&
            webui_spa_path(request.path)) {
            response.set_content(webui_index_html_, "text/html");
            return httplib::Server::HandlerResponse::Handled;
        }
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key. Echo the
        // requested preflight headers back so browser clients can preflight custom headers
        // (e.g. the webui's Authorization / x-llama-api-key) beyond the fixed defaults.
        server_.Options(R"(.*)", [](const httplib::Request& req, httplib::Response& res) {
            res.status = 204;
            const std::string requested = req.get_header_value("Access-Control-Request-Headers");
            if (!requested.empty()) { res.set_header("Access-Control-Allow-Headers", requested); }
        });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
        // Only API endpoints require a key. The UI shell and every static asset load freely so
        // the webui can prompt for and send the key on API calls (llama-server parity). /health
        // stays open and OPTIONS is a CORS preflight.
        if (options_.api_key.empty() || req.method == "OPTIONS" || req.path == "/health" ||
            !is_api_path(req.path)) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_anthropic_error(res, error, new_anthropic_request_id());
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        // /health is a pure liveness probe (grounded in llama-swap's handleHealth, api.go:434:
        // always 200 "OK"), answering whether the SERVER is up, independent of whether any model
        // is currently loaded. With the no-resident, on-demand-load router a startup /health
        // (nothing loaded yet) is 200, NOT 503. Model readiness is observable via /v1/models
        // `status` (loaded/unloaded), not via /health -- so this deliberately ignores the router.
        res.status = 200;
        res.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    // The llama.cpp webui queries /props for its server-properties stub (loaded model, modalities,
    // default generation settings); it is an API path so it requires a key like the other /v1 routes.
    server_.Get("/props", [this](const httplib::Request& req, httplib::Response& res) {
        handle_props(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });

    // ---- llama.cpp(-swap) webui parity (Group 1: live /slots; real tool registry; real stream registry) ----
    // The bundled webui (ggml-org/llama-ui) probes server-management endpoints. /slots is REAL
    // (design §3, step 8): it returns live slot state -- one object per configured concurrency
    // slot (max_concurrency 1..8) on the single resident model -- via read-only introspection of
    // ModelRouter (in_flight/loaded_id/is_swapping) + the resident backend's runtime_stats (no new
    // state, no scheduler mutation, no preemption). /tools is REAL (design §4, step 7): it
    // consumes the serving-owned ToolRegistry -- GET lists registered tools, POST executes one
    // (llama.cpp server-tools model). The two stream routes below (/v1/streams/lookup, DELETE
    // /v1/stream/{id}) are REAL: they consume the serving-owned StreamRegistry of in-flight
    // streaming completions (design §1, steps 3-6). Every route here is already key-gated by
    // is_api_path (the /v1 prefix rule covers /v1/streams/lookup and /v1/stream/{id}; the explicit
    // /slots and /tools entries cover the rest) and SPA-excluded by webui_spa_path. Each calls a
    // named handler (the slots handler reads the router + resident backend; the tool + stream
    // handlers read their registries).
    server_.Get("/slots", [this](const httplib::Request& req, httplib::Response& res) {
        handle_slots(req, res);
    });
    server_.Get("/tools", [this](const httplib::Request& req, httplib::Response& res) {
        handle_tools(req, res);
    });
    server_.Post("/tools", [this](const httplib::Request& req, httplib::Response& res) {
        handle_tools_execute(req, res);
    });
    server_.Post("/v1/streams/lookup",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_stream_lookup(req, res);
                 });
    // The {id} is the registry key (X-Conversation-Id = conversationId::model) URL-encoded by the
    // webui; the model part can contain '/', so the route captures the whole decoded remainder. The
    // handler raises the owner cancel token (the existing CancellationView channel) and removes the
    // entry -- a real owner-initiated stop (design §1.3, step 5).
    server_.Delete(R"(/v1/stream/(.*))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_stream_cancel(req, res);
                   });

    // ---- llama.cpp(-swap) webui parity: ModelRouter load/unload wrappers (Group 2a) ----
    // Thin protocol-translation wrappers over the existing ModelRouter route()/unload() primitives
    // (serving owns protocol translation; the router owns load/eviction -- no new state, no
    // preemption, no device allocation). POST /models/load performs the on-demand load/swap
    // (evicts the current resident; one-GPU, one-resident, no-preemption) and POST /models/unload
    // forces the no-resident state. The webui reads 2xx + valid JSON; it observes model state via
    // GET /v1/models and the SSE feed.
    server_.Post("/models/load", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model_load(req, res);
    });
    server_.Post("/models/unload", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model_unload(req, res);
    });
    // GET /models/sse (Group 2b): the minimal keepalive SSE stream (design §3.3). Key-gated by the
    // /models/ prefix rule (is_api_path) and SPA-excluded (webui_spa_path), like load/unload.
    server_.Get("/models/sse", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model_sse(req, res);
    });

    // ---- llama.cpp(-swap) webui parity (Group 3: owner cancel + stream replay/tail) ----
    // Both are key-gated by the /v1 prefix rule in is_api_path and SPA-excluded by webui_spa_path,
    // so only the handlers are added here (no new gate entries, like the Group 1 routes above).
    // POST /v1/chat/completions/control (design §2, step 6): the webui's stopReasoning posts the
    // active completion id and requires `ok && success===true`. The handler resolves the id through
    // the stream registry's completion index and raises that stream's owner cancel token (the
    // existing CancellationView channel) -- a real owner-initiated stop, not a no-op.
    server_.Post("/v1/chat/completions/control",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_completions_control(req, res);
                 });
    // GET /v1/stream/{id}?from=N (design §1.3, step 4): resume a registered stream from a byte
    // offset and tail it live (replay the buffered SSE bytes, then follow new frames until done).
    // Same URL-encoded registry key as DELETE (the model part can contain '/', so the route
    // captures the whole decoded remainder).
    server_.Get(R"(/v1/stream/(.*))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_stream(req, res);
                });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(options_.model_config.models, unix_time_now(),
                                     options_.max_context, router_->loaded_id()),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const ModelConfig* model = options_.model_config.model(id);
    if (model == nullptr) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    const bool loaded = (model->id == router_->loaded_id());
    res.set_content(make_model_object(*model, unix_time_now(), options_.max_context, loaded),
                    "application/json");
}

void HttpServer::handle_slots(const httplib::Request& req, httplib::Response& res) const {
    // Real live slot state (llama-parity design §3, step 8). The webui asks "is the server free?"
    // via GET /slots (and GET /slots?model=<name>); the busy slots are the resident model's
    // in-flight generations. This is read-only introspection of ModelRouter + the resident
    // backend's runtime_stats (no new state, no scheduler mutation, no preemption). The slot
    // shaping is a pure function (make_slots_list) so it is host-testable without a GPU.
    const std::string model_filter = req.get_param_value("model");
    // The resident backend (null when no-resident / mid-swap) supplies the live prefill-lane
    // occupancy; a null read keeps the shape honest (all slots idle when there is no resident).
    const auto resident = router_->resident();
    const int prefilling =
        resident ? static_cast<int>(resident->runtime_stats().prefilling_requests) : 0;
    const std::string payload =
        make_slots_list(options_.max_concurrency, router_->loaded_id(), model_filter,
                        router_->in_flight(), router_->is_swapping(), prefilling);
    res.status = 200;
    res.set_content(payload, "application/json");
}

void HttpServer::handle_model_load(const httplib::Request& req, httplib::Response& res) const {
    RequestJson body;
    try {
        body = parse_json_body(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    std::string model;
    if (body.is_object() && body.contains("model") && body["model"].is_string()) {
        model = body["model"].get<std::string>();
    }
    if (model.empty()) {
        ApiError error;
        error.status  = 400;
        error.type    = "invalid_request_error";
        error.param   = "model";
        error.code    = "model_required";
        error.message = "the 'model' field is required";
        write_openai_error(res, error);
        return;
    }
    // Route through the in-process router: the on-demand load/swap (destroy the current resident,
    // construct the target, gate on readiness). The exception mapping mirrors handle_chat_completions
    // (openai_chat_http.cpp): an unknown id/alias -> 404, a concurrency-limit / queue-full breach ->
    // 429. In the single-GPU, one-resident model this load evicts the current resident (a sequential
    // swap, no preemption); extra_args are ignored (NInfer model config is fixed at registration).
    ModelRouter::Grant grant;
    try {
        grant = router_->route(model);
    } catch (const std::out_of_range& exception) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.param   = "model";
        error.code    = "model_not_found";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    } catch (const std::overflow_error& exception) {
        ApiError error;
        error.status  = 429;
        error.type    = "invalid_request_error";
        error.param   = "model";
        error.code    = "rate_limit_exceeded";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    } catch (const std::exception& exception) {
        // A rethrown factory/Engine exception from route() (a failed load, or a readiness-timeout /
        // shutdown-during-install "model is not ready").
        ApiError error;
        error.status  = 503;
        error.type    = "server_error";
        error.param   = "model";
        error.code    = "model_not_ready";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }
    // Thin wrapper: release the in-flight reservation now (the model persists as the resident via
    // the router's TTL; this endpoint only triggers the load/swap and does not run generation).
    grant.reset();
    res.status = 200;
    res.set_content(nlohmann::json{{"model", model}, {"status", "loaded"}}.dump(),
                    "application/json");
}

void HttpServer::handle_model_unload(const httplib::Request& req, httplib::Response& res) const {
    RequestJson body;
    try {
        body = parse_json_body(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    std::string model;
    if (body.is_object() && body.contains("model") && body["model"].is_string()) {
        model = body["model"].get<std::string>();
    }
    // Validate the id is in the config (mirrors handle_model); a model that is not the loaded one
    // still unloads the single resident (NInfer keeps exactly one resident) and returns 200.
    if (model.empty() || options_.model_config.model(model) == nullptr) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.param   = "model";
        error.code    = "model_not_found";
        error.message = "model '" + model + "' not found";
        write_openai_error(res, error);
        return;
    }
    // Capture the actually-unloaded model id before the unload so the response is honest about
    // what was evicted (the single-resident system may have a different model loaded than the
    // one requested; e.g. the webui requests "other" but "res" is the loaded model).
    const std::string unloaded_id = router_->loaded_id();
    router_->unload();
    res.status = 200;
    res.set_content(nlohmann::json{{"model", unloaded_id.empty() ? model : unloaded_id},
                                   {"status", "unloaded"}}.dump(),
                    "application/json");
}

void HttpServer::handle_model_sse(const httplib::Request&, httplib::Response& res) {
    // GET /models/sse (llama.cpp(-swap) webui parity, design §5, router-hooked pub/sub tier). The
    // webui opens this in a reconnect loop (runStatusReader) as a live model-status feed and
    // treats a closed stream as non-fatal. On connect: register this connection as a ModelSseHub
    // subscriber and emit the CURRENT model status immediately. Thereafter the router's status
    // hook (wired in attach) fans every status transition (load/swap/unload/TTL-evict) out to all
    // connected subscribers; this connection's writer thread (this lambda, on the httplib worker)
    // drains its own bounded queue, writes each record, and holds the socket open with periodic
    // keepalive comments between events. A slow client whose queue overflows is DROPPED (it never
    // stalls the router's event path, design §5.1), and on disconnect the subscriber is
    // unsubscribed (clean teardown, no leak). Copies the chat-completions SSE pattern exactly
    // (prepare_sse_response + set_chunked_content_provider + SseTransport + sink.done()).
    prepare_sse_response(res);
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, cancelled](std::size_t, httplib::DataSink& sink) -> bool {
            SseTransport transport(sink, *cancelled, model_sse_hub_.keepalive_interval());
            // Register this connection as a subscriber BEFORE emitting the connect record, so a
            // status change published in the meantime is enqueued (not lost) and delivered next.
            const auto sub = model_sse_hub_.subscribe();
            try {
                // current-status-on-connect: emit the current model status immediately.
                render_and_write(transport, [&] { return model_sse_hub_.current_status_record(); });
            } catch (const ClientDisconnected&) {
                model_sse_hub_.unsubscribe(sub);
                sink.done();
                return false;
            } catch (const std::exception&) {
                // The record build only reads the router and cannot realistically fail; treat a
                // render failure as a clean end of the stream.
                model_sse_hub_.unsubscribe(sub);
                sink.done();
                return false;
            }
            // Drain the subscriber's bounded queue: each published status record is written; on a
            // quiet interval (a pop timeout on the keepalive cadence) a no-op keepalive comment
            // holds the socket live. A dropped subscriber (its queue overflowed -> slow client)
            // or a closed/failed socket (client disconnect) ends the loop. Only this per-connection
            // writer blocks on a slow socket; the router-hook publish() path never does (design
            // §5.1 "drop-on-slow-client").
            for (;;) {
                std::string record;
                const auto result =
                    model_sse_hub_.pop_blocking(sub, &record, model_sse_hub_.keepalive_interval());
                if (result == ModelSseHub::PopResult::Inactive) {
                    break; // dropped (slow client) or unsubscribed: stop
                }
                if (result == ModelSseHub::PopResult::TimedOut) {
                    // Idle within the keepalive window: hold the socket open (the SseTransport
                    // heartbeat writes the no-op comment when due and reports a closed/failed
                    // socket so the loop terminates cleanly on a disconnect).
                    if (transport.poll()) { break; }
                    continue;
                }
                // A status record was delivered: write it. A client disconnect surfaces as a
                // ClientDisconnected (the socket write fails); stop cleanly.
                try {
                    transport.write(record);
                } catch (const ClientDisconnected&) {
                    break;
                }
                if (transport.poll()) { break; } // closed/failed socket after the write
            }
            model_sse_hub_.unsubscribe(sub); // clean teardown (idempotent; a dropped sub too)
            sink.done();
            return true;
        },
        [cancelled](bool successful) {
            // Mark the stream cancelled if it did not complete cleanly (mirrors the chat-stream
            // resource releaser; the client-disconnect path is already handled in the loop above).
            if (!successful) { cancelled->store(true, std::memory_order_release); }
        });
}

void HttpServer::handle_stream_lookup(const httplib::Request& req, httplib::Response& res) {
    // POST /v1/streams/lookup (design §1.3, step 3): the webui's probeServerStream /
    // syncRemoteRunningStreams post {conversation_ids:[...]} (each a registry key =
    // conversationId::model) and read the active server streams back (selectActiveStream filters
    // !is_done and picks the newest started_at). Filter the registry's active entries to the
    // requested set; when the field is absent or not a string array, return all active. Only the
    // fields the webui reads are emitted (conversation_id, is_done, started_at).
    RequestJson body;
    try {
        body = parse_json_body(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    std::vector<std::string> requested;
    if (body.is_object() && body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
        for (const auto& item : body["conversation_ids"]) {
            if (item.is_string()) { requested.push_back(item.get<std::string>()); }
        }
    }
    const auto active = stream_registry_.lookup_active();
    nlohmann::json records = nlohmann::json::array();
    for (const auto& record : active) {
        if (!requested.empty()) {
            const bool wanted =
                std::any_of(requested.begin(), requested.end(), [&record](const std::string& id) {
                    return record.conversation_id == id;
                });
            if (!wanted) { continue; }
        }
        records.push_back(nlohmann::json{{"conversation_id", record.conversation_id},
                                         {"is_done", record.is_done},
                                         {"started_at", record.started_at}});
    }
    res.status = 200;
    res.set_content(records.dump(), "application/json");
}

void HttpServer::handle_stream_cancel(const httplib::Request& req, httplib::Response& res) {
    // DELETE /v1/stream/{id} (design §1.3, step 5): the webui's cancelServerStream is
    // fire-and-forget and reads no response field. {id} is the registry key (matches[1] is the
    // decoded key; the route /v1/stream/(.*) captured the whole remainder). Raise the owner cancel
    // token (observed by the in-flight generation's existing CancellationView) and remove the entry.
    // cancel() reports whether the stream was found: a done-but-not-yet-reaped stream is still
    // found+removed -> cancelled:true; only an already-reaped (no longer registered) or a
    // never-registered id -> false. Always 200 (the client must not see a 404). This is an
    // owner-initiated stop, not scheduler preemption.
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    // cancel() reports whether the stream was found (it raises the owner token); remove() erases
    // it. Both values are consumed: the response reports whether an entry was actually found+removed.
    const bool cancelled =
        !id.empty() && stream_registry_.cancel(id) && stream_registry_.remove(id);
    res.status = 200;
    res.set_content(nlohmann::json{{"cancelled", cancelled}}.dump(), "application/json");
}

void HttpServer::handle_completions_control(const httplib::Request& req, httplib::Response& res) {
    // POST /v1/chat/completions/control (design §2, step 6): the webui's stopReasoning posts
    // {id:<completion id>, action:"reasoning_end"} and requires `ok && success===true`. Resolve the
    // id through the registry's completion index; if that misses, treat the id as a stream id
    // (design §2.2 secondary path) and cancel + remove. `action` is accepted but the effect is
    // always an owner-initiated cancel of the in-flight generation (the existing CancellationView
    // channel) -- a real stop, not a no-op. Unknown id -> success:false (an honest no-op, not a
    // fake success).
    RequestJson body;
    try {
        body = parse_json_body(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    std::string id;
    if (body.is_object() && body.contains("id") && body["id"].is_string()) {
        id = body["id"].get<std::string>();
    }
    bool cancelled = !id.empty() && stream_registry_.cancel_by_completion(id);
    if (!cancelled && !id.empty() && stream_registry_.cancel(id)) {
        // Secondary path hit: the id is a stream id. Remove the entry (the cancel token is already
        // raised above); the response reflects whether the entry was actually removed.
        cancelled = stream_registry_.remove(id);
    }
    res.status = 200;
    res.set_content(nlohmann::json{{"success", cancelled}}.dump(), "application/json");
}

void HttpServer::handle_stream(const httplib::Request& req, httplib::Response& res) {
    // GET /v1/stream/{id}?from=N (design §1.3, step 4): resume a registered stream from a byte
    // offset and tail it live. {id} is the registry key (X-Conversation-Id = conversationId::model);
    // the route /v1/stream/(.*) captures the whole decoded remainder (httplib decodes req.path via
    // decode_path_component before routing, so matches[1] is already the decoded key; the model part
    // may contain '/', hence (.*) rather than [^/]+). ?from=N is the byte offset to resume from
    // (0 = fresh attach; N = bytes the client already received).
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto entry = stream_registry_.get(id);
    if (entry == nullptr) {
        // Unknown (never registered, or already reaped). A well-formed 404 tells the webui's
        // attachServerStream "no such stream -- start fresh locally" (task contract; design §1.3
        // originally proposed emitting [DONE] and closing -- 404 is the honest not-found answer).
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "stream_not_found";
        error.message = "stream '" + id + "' is not registered (or was reaped)";
        write_openai_error(res, error);
        return;
    }
    const std::size_t from = parse_stream_offset(req.get_param_value("from"));

    prepare_sse_response(res);
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    res.set_chunked_content_provider(
        "text/event-stream",
        [entry, from, cancelled](std::size_t, httplib::DataSink& sink) -> bool {
            // Replay the buffered SSE bytes from `from`, then live-tail new frames until the stream
            // is done. The buffered bytes are EXACTLY what the producer teed to the live client
            // (including the terminal "data: [DONE]" frame, which the producer appends via
            // StreamDoneGuard + the terminal-frame loop), so they are replayed verbatim and no
            // second [DONE] is emitted here (a double-emit would be wrong; the terminator is
            // already in the buffer). SseTransport::write writes the bytes verbatim (no keepalive
            // injection -- keepalives only fire via poll(), which is never called here, so the
            // byte-offset contract the webui relies on is preserved) and throws ClientDisconnected
            // on a closed socket (a mid-tail disconnect is detected on the next frame; the tail
            // always terminates when the generation completes or is cancelled).
            SseTransport transport(sink, *cancelled);
            std::size_t offset = from;
            try {
                for (;;) {
                    const std::string have = entry->replay_from(offset);
                    if (!have.empty()) {
                        transport.write(have);
                        offset += have.size();
                    }
                    const bool done = entry->wait_for_bytes(offset);
                    if (done) {
                        // Final drain: frames appended just before is_done is observed.
                        const std::string tail = entry->replay_from(offset);
                        if (!tail.empty()) { transport.write(tail); }
                        break;
                    }
                }
            } catch (const ClientDisconnected&) {
                sink.done();
                return false;
            }
            sink.done();
            return true;
        },
        [cancelled](bool successful) {
            // Mark the stream cancelled if it did not complete cleanly (mirrors handle_model_sse).
            if (!successful) { cancelled->store(true, std::memory_order_release); }
        });
}

void HttpServer::handle_tools(const httplib::Request&, httplib::Response& res) const {
    // GET /tools (llama-parity design §4.3, step 7): the webui's ToolsStore maps each element's
    // `.definition` (and reads `.name` / `.enabled`). Return the serving-owned registry's registered
    // tools as a JSON array of {name, definition, enabled}. NInfer ships no built-in server tools,
    // so this is [] at startup (the honest "no tools" state the client expects -- it only disables
    // the feature on a transport error, not on an empty list, design §0.6). A registered tool set
    // (upserted at serve startup) is reflected here faithfully.
    nlohmann::json tools = nlohmann::json::array();
    for (const auto& tool : tool_registry_.list()) {
        tools.push_back(nlohmann::json{{"name", tool.name},
                                       {"definition", tool.definition.is_null()
                                            ? nlohmann::ordered_json::object()
                                            : tool.definition},
                                       {"enabled", tool.enabled}});
    }
    res.status = 200;
    res.set_content(tools.dump(), "application/json");
}

void HttpServer::handle_tools_execute(const httplib::Request& req, httplib::Response& res) {
    // POST /tools (llama-parity design §4.3, step 7): EXECUTION, not registration. The webui's
    // NPe.executeTool posts {tool, params} and reads `error` (isError:true) or
    // `plain_text_response` (isError:false). Dispatch through the serving-owned registry:
    //   - a tool with a registered handler -> run it, return {plain_text_response} (or the handler's error);
    //   - an unknown / unhandled tool -> a well-formed {error: "tool '<name>' is not available on this server"}
    // (200; the client renders isError:true, NOT a transport error). This is the llama.cpp
    // server-tools execution model -- no Core/Runtime reach (a future Engine-backed tool handler
    // would cross the boundary, design §4.3).
    RequestJson body;
    try {
        body = parse_json_body(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    std::string tool;
    if (body.is_object() && body.contains("tool") && body["tool"].is_string()) {
        tool = body["tool"].get<std::string>();
    }
    const nlohmann::ordered_json params =
        (body.is_object() && body.contains("params") && body["params"].is_object())
            ? body["params"]
            : nlohmann::ordered_json::object();
    const ToolResult result = tool_registry_.execute(tool, params);
    nlohmann::json response;
    if (result.ok) {
        response["plain_text_response"] = result.plain_text_response;
    } else {
        response["error"] = result.error;
    }
    res.status = 200;
    res.set_content(response.dump(), "application/json");
}

void HttpServer::handle_props(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_props_stub(options_, public_model_id_), "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(ModelRouter& router) {
    if (router_ != nullptr) {
        throw std::logic_error("HTTP model router is already attached");
    }
    router_ = &router;
    model_sse_hub_.bind(router);
    // Router-hooked /models/sse pub/sub (design §5.2): the router fires a ModelStatusEvent at each
    // status transition (load/swap-start/swap-complete/unload/TTL-evict), OUTSIDE its lock (so a
    // blocking subscriber cannot stall route()); the hub fans it out to every connected
    // /models/sse subscriber (a bounded, non-blocking enqueue; slow clients are dropped). Serving-
    // owned: the hook is a thin callback held by the router; the fan-out state (the hub) is owned
    // by this HttpServer. Cleared again in ~HttpServer (the router outlives the server) so the
    // router's final unload cannot fire a hook into a destroyed server.
    router.set_status_hook([this](const ModelStatusEvent& ev) { model_sse_hub_.publish(ev); });
    // No-resident, on-demand-load: the router starts UNLOADED, so there is no resident backend
    // whose diagnostics to record at attach time. public_model_id_ is the default model (the first
    // in config order, or an explicit override) used as the default for model-less requests; it is
    // NOT a loaded model. The server-start record therefore carries neutral (empty) engine
    // diagnostics; model diagnostics are captured when the first model is loaded on demand.
    public_model_id_ = resolve_public_model_id(options_, options_.model_config.resident().id);
    request_jsonl_.write_server_start(options_, /*engine_options=*/ ninfer::EngineOptions{},
                                      /*sampling_defaults=*/ ninfer::ModelSamplingDefaults{},
                                      public_model_id_, /*load=*/ ninfer::LoadSummary{},
                                      /*memory=*/ ninfer::MemorySummary{});
}

bool HttpServer::listen() {
    if (router_ == nullptr) { throw std::logic_error("HTTP model router is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

HttpServer::~HttpServer() {
    // The router outlives this server (the serving main destroys the server before the router).
    // Clear the /models/sse status hook so the router's final unload (in its own shutdown, run
    // when the router is destroyed) cannot fire a hook into this destroyed server. In-service
    // transitions (route/unload/TTL) fire the hook only while this server is alive; once it is
    // torn down the hook is cleared and those transitions become no-ops. (The stats reporter
    // thread is already stopped by listen() before the server is torn down.)
    if (router_ != nullptr) {
        router_->set_status_hook({});
    }
}

} // namespace ninfer::serve
