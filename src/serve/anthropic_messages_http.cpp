#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);
    AnthropicCountTokensRequest request;
    try {
        request = parse_anthropic_count_tokens_request(parse_json_body(req));
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_count_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    }
    // Route (supersedes model validation): unknown model -> 404 model-not-found; a concurrency /
    // queue breach -> 429. The Grant keeps the backend alive for the (short) count call.
    ModelRouter::Grant grant;
    try {
        grant = router_->route(request.model);
    } catch (const std::out_of_range& exception) {
        ApiError error;
        error.status  = 404;
        error.code    = "model_not_found";
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::overflow_error& exception) {
        ApiError error;
        error.status  = 429;
        error.code    = "rate_limit_exceeded";
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        // A rethrown factory/Engine exception from route() (a failed load, or a readiness-timeout /
        // shutdown-during-install "model is not ready").
        ApiError error;
        error.status  = 503;
        error.code    = "model_not_ready";
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    }
    auto* backend = grant.backend.get();
    try {
        const int input_tokens =
            backend->count_prompt_tokens(request.generation,
                                         [&req] { return client_disconnected(req); });
        res.set_content(make_anthropic_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_count_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);

    AnthropicMessagesRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_anthropic_messages_request(parse_json_body(req), limits);
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_messages",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};

    // Route through the in-process router: unknown model -> 404 model-not-found; a concurrency /
    // queue breach -> 429. The Grant keeps the granted backend alive for the whole request; the
    // streaming path moves it into the stream.
    ModelRouter::Grant grant;
    try {
        grant = router_->route(request.model);
    } catch (const std::out_of_range& exception) {
        ApiError error;
        error.status  = 404;
        error.code    = "model_not_found";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata,
            normalize_anthropic_error(error)));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::overflow_error& exception) {
        ApiError error;
        error.status  = 429;
        error.code    = "rate_limit_exceeded";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata,
            normalize_anthropic_error(error)));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        // A rethrown factory/Engine exception from route() (a failed load, or a readiness-timeout /
        // shutdown-during-install "model is not ready").
        ApiError error;
        error.status  = 503;
        error.code    = "model_not_ready";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata,
            normalize_anthropic_error(error)));
        write_anthropic_error(res, error, request_id);
        return;
    }
    auto* backend = grant.backend.get();

    PreparedRequest prepared;
    try {
        prepared = backend->prepare(request.generation,
                                    request.stream ? GenerationConsumerMode::Streaming
                                                   : GenerationConsumerMode::Aggregate,
                                    {}, [&req] { return client_disconnected(req); },
                                    ninfer::ContextCacheHints{});
    } catch (const ApiException& exception) {
        const ApiError error = normalize_anthropic_error(exception.error());
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    }

    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity(request_id, request.model);
    const int input_tokens = prepared.prompt_tokens;

    auto lifecycle = begin_request(make_request_log_context(
        req_id, "anthropic_messages", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = backend->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->failure(make_generation_request_failure(error));
            write_anthropic_error(res, error, request_id);
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_owned_json_content(res, make_anthropic_messages_response(identity, outcome),
                                   prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::ResponseRender, error));
            write_anthropic_error(res, error, request_id);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
        }
        return;
    }

    try {
        auto stream  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<AnthropicMessagesStream>(identity, input_tokens);

        prepare_sse_response(res);
        // Move the Grant into the stream so the granted backend (Engine) outlives the streaming
        // generation (see the OpenAI handlers for the copyability-wrapper rationale).
        auto grant_keep = std::make_shared<ModelRouter::Grant>(std::move(grant));
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, grant_keep](
                std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        if (!encoder->started()) {
                            render_and_write(transport, [&] { return encoder->start(); });
                        }
                        render_and_write(transport, [&] { return encoder->error(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    }
                };

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const ninfer::GenerationStart& start) {
                        render_and_write(transport, [&] { return encoder->start(start); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = grant_keep->backend->run(stream->prepared, &output, {});
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    const ApiError error = normalize_anthropic_error(exception.error());
                    lifecycle->failure(make_generation_request_failure(error));
                    return send_error(error);
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                }

                lifecycle->done(outcome);
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

} // namespace ninfer::serve
