#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

} // namespace

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(parse_json_body(req), limits,
                                                                   public_model_id_);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};

    // Route through the in-process router (supersedes validate_openai_model): an unknown model or
    // alias -> 404; a concurrency-limit / queue-full breach -> 429. The Grant keeps the granted
    // backend (Engine) alive for the whole request; the streaming path moves it into the stream.
    ModelRouter::Grant grant;
    try {
        grant = router_->route(request.model);
    } catch (const std::out_of_range& exception) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.param   = "model";
        error.code    = "model_not_found";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    } catch (const std::overflow_error& exception) {
        ApiError error;
        error.status  = 429;
        error.type    = "invalid_request_error";
        error.param   = "model";
        error.code    = "rate_limit_exceeded";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    } catch (const std::exception& exception) {
        // A rethrown factory/Engine exception from route() (a failed load, or a readiness-timeout /
        // shutdown-during-install "model is not ready"): the model cannot serve this request.
        ApiError error;
        error.status  = 503;
        error.type    = "server_error";
        error.param   = "model";
        error.code    = "model_not_ready";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }
    auto* backend = grant.backend.get();

    PreparedRequest prepared;
    try {
        const ninfer::GenerationObservationOptions observation{
            .phase_timings   = true,
            .live_timings    = request.stream && request.timings_per_token,
            .prompt_progress = request.stream && request.return_progress,
        };
        prepared = backend->prepare(request.generation,
                                    request.stream ? GenerationConsumerMode::Streaming
                                                   : GenerationConsumerMode::Aggregate,
                                    observation, [&req] { return client_disconnected(req); },
                                    ninfer::ContextCacheHints{});
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "openai_chat_completions", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    const OpenAIChatResponseIdentity identity = make_openai_chat_response_identity(request.model);
    auto lifecycle                            = begin_request(make_request_log_context(
        req_id, "openai_chat_completions", request.generation, metadata, prepared));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = backend->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            const RequestFailure failure =
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what());
            lifecycle->failure(failure);
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_owned_json_content(res, make_chat_completion_response(identity, outcome),
                                   prepared.lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
        }
        return;
    }

    // llama-parity stream registry (design §1, step 2): when the request carries the webui's
    // X-Conversation-Id header, register this streaming completion so its rendered SSE bytes are
    // buffered (resume-by-byte-offset + live tail) and its owner cancel token is addressable. A
    // request WITHOUT the header (a non-llama-ui client) gets no entry -> identical behavior to
    // before (the regression guard).
    std::shared_ptr<StreamEntry> stream_entry;
    {
        const std::string conversation_id = req.get_header_value("X-Conversation-Id");
        if (!conversation_id.empty()) {
            stream_entry = stream_registry_.create(conversation_id, identity.id, unix_time_now());
        }
    }

    try {
        const bool return_progress   = request.return_progress;
        const bool timings_per_token = request.timings_per_token;
        auto stream                  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<OpenAIChatStream>(identity, request.include_usage,
                                                          timings_per_token, return_progress);

        prepare_sse_response(res);
        // Move the Grant into the stream so the granted backend (Engine) outlives the streaming
        // generation: if the router swaps the resident mid-stream, the in-flight Grant's
        // shared_ptr keeps this backend alive. The wrapper keeps the chunked-content lambda
        // copy-constructible (required by httplib's DataSink::ContentProviderWithoutLength).
        auto grant_keep = std::make_shared<ModelRouter::Grant>(std::move(grant));
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle, return_progress, timings_per_token, grant_keep,
             stream_entry](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                // Mark the registry entry done when this stream ends (any exit path), starting its
                // retention window (design §1.2). The guard holds the entry for the generation's
                // lifetime; with no X-Conversation-Id, stream_entry is null and the guard is a no-op.
                struct StreamDoneGuard {
                    std::shared_ptr<StreamEntry> entry;
                    ~StreamDoneGuard() {
                        if (entry != nullptr) { entry->mark_done(); }
                    }
                } stream_done_guard{stream_entry};
                // Tee every rendered SSE frame into the registry buffer (the exact bytes written to
                // the live client) in addition to the transport write, so a client can resume from a
                // byte offset and tail live (design §1.1). With no entry the tee is a no-op and the
                // write path is identical to before (regression guard). Error semantics mirror the
                // render_and_write free function: render() exceptions become ResponseRenderFailure.
                auto tee_write = [&transport, &stream_entry]<class Render>(Render&& render) {
                    std::string payload;
                    try {
                        payload = std::forward<Render>(render)();
                    } catch (const ClientDisconnected&) { throw; } catch (const ResponseRenderFailure&) {
                        throw;
                    } catch (const std::exception& exception) {
                        throw ResponseRenderFailure(exception.what());
                    }
                    if (stream_entry != nullptr) { stream_entry->append(payload); }
                    transport.write(payload);
                };
                // OR-combine the owner cancel token into the EXISTING cancellation channel
                // (design §2.1): transport.poll() is the client-disconnect path (unchanged); the
                // cancel flag lets DELETE/control stop this generation via the existing
                // CancellationView (no new Engine mechanism, no preemption). Null token (no header)
                // -> behavior identical to before.
                const auto cancel_token = stream_entry != nullptr ? stream_entry->cancel : nullptr;
                const auto send_error = [&](const ApiError& error) {
                    try {
                        tee_write([&] { return sse_error_event(error); });
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
                try {
                    tee_write([&] { return encoder->start(); });
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const ninfer::GenerationStart& start) {
                        encoder->note_start(start);
                        if (return_progress) {
                            tee_write([&] { return encoder->initial_prompt_progress(); });
                        }
                    };
                    if (return_progress) {
                        output.on_progress = [&](const ninfer::PromptProgress& progress) {
                            tee_write([&] { return encoder->prompt_progress(progress); });
                        };
                    }
                    if (timings_per_token) {
                        output.on_timing = [&](const ninfer::GenerationTimingObservation& timing) {
                            encoder->note_timing(timing);
                        };
                    }
                    output.on_content = [&](const std::string& text) {
                        tee_write([&] { return encoder->content_delta(text); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        tee_write([&] { return encoder->reasoning_delta(text); });
                    };
                    output.is_cancelled = [transport_ptr = &transport, cancel_token] {
                        // Existing client-disconnect observation OR the owner cancel token (design
                        // §2.1). A null cancel_token (no X-Conversation-Id) reduces to poll() only.
                        return transport_ptr->poll() ||
                               (cancel_token != nullptr &&
                                cancel_token->load(std::memory_order_acquire));
                    };

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
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
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
                    error.type    = "internal_error";
                    error.message = exception.what();
                    return send_error(error);
                }
                try {
                    for (const std::string& item : terminal) {
                        if (stream_entry != nullptr) { stream_entry->append(item); }
                        transport.write(item);
                    }
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
        // The streaming setup (or content-provider install) failed after the registry entry was
        // created, so the content-provider done-guard never ran; mark the entry done here so it is
        // reaped on schedule instead of leaking. No-op when there was no X-Conversation-Id.
        if (stream_entry != nullptr) { stream_entry->mark_done(); }
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace ninfer::serve
