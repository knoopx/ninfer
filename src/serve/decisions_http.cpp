#include "serve/http_server.h"

#include "serve/decisions.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"

#include <chrono>
#include <exception>
#include <string>

namespace ninfer::serve {

void HttpServer::handle_decisions(const httplib::Request& req, httplib::Response& res) {
    const auto received = std::chrono::steady_clock::now();
    DecisionsRequest request;
    try {
        request = parse_decisions_request(parse_json_body(req));
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    // The request parsed, so its shape is known: every later failure emits one decision_error
    // terminal, and a request that reaches the decision service is bracketed by decision_start.
    const std::uint64_t request_id = ++request_seq_;
    DecisionLogContext log_context;
    log_context.id             = request_id;
    log_context.question_count = request.questions.size();
    log_context.raw_logits     = request.raw_logits;
    for (const ninfer::DecisionQuestion& question : request.questions) {
        log_context.candidate_count += question.criteria.size();
    }
    for (const ChatTurn& turn : request.messages) {
        for (const ContentPart& part : turn.content) {
            if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) {
                ++log_context.media_item_count;
            }
        }
    }

    // One decision_error terminal for a failure that never opened the request lifecycle.
    const auto fail = [&](const ApiError& error, RequestFailurePhase phase) {
        record_decision_failure(log_context, make_request_failure(phase, error));
        write_openai_error(res, error);
    };

    // The route runs on the model the router has loaded (the resident); a client-sent
    // top-level "model" field is accepted and ignored by the request parser. Nothing loaded ->
    // 503 model_not_ready; a loaded model routes without a swap (pinned for the request
    // duration).
    const std::string model_id = router_->loaded_id();
    if (model_id.empty()) {
        ApiError error;
        error.status  = 503;
        error.type    = "server_error";
        error.param   = "model";
        error.code    = "model_not_ready";
        error.message = "no model is loaded";
        fail(error, RequestFailurePhase::Prepare);
        return;
    }
    ModelRouter::Grant grant;
    try {
        grant = router_->route(model_id);
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 503;
        error.type    = "server_error";
        error.param   = "model";
        error.code    = "model_not_ready";
        error.message = exception.what();
        fail(error, RequestFailurePhase::Prepare);
        return;
    }

    // The decision seam is off the ModelBackend/SSE interface: resolve the backend's
    // GenerationService (which runs decisions on the model's loaded engine) and call decide on
    // it. A backend with no decision service (test fakes) answers 503 decision route
    // unavailable; production backends always expose the service.
    const GenerationService* decision_service = grant.backend->decision_service();
    if (decision_service == nullptr) {
        ApiError error;
        error.status  = 503;
        error.type    = "server_error";
        error.param   = "model";
        error.code    = "decision_route_unavailable";
        error.message = "decision route unavailable";
        fail(error, RequestFailurePhase::Prepare);
        return;
    }

    log_context.model = model_id;
    log_context.prepare_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - received).count();
    record_decision_start(log_context);

    // decide() runs on the model's loaded engine and maps its own failures to the shared error
    // contract (422 invalid questions, 400 context length, 429/503 admission); a render
    // failure (std::logic_error) escapes to the server's exception handler, which renders a 500.
    const auto started = std::chrono::steady_clock::now();
    ninfer::DecisionResult result;
    try {
        result = decision_service->decide(request, 1.0f);
    } catch (const ApiException& exception) {
        record_decision_failure(
            log_context, make_request_failure(RequestFailurePhase::Generation, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_decision_failure(log_context,
                                make_request_failure(RequestFailurePhase::Generation, error));
        write_openai_error(res, error);
        return;
    }
    DecisionLogOutcome outcome;
    outcome.answer_count  = result.answers.size();
    outcome.input_tokens  = result.input_tokens;
    outcome.total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    record_decision_done(log_context, outcome);
    res.status = 200;
    res.set_content(render_decisions_response(request, result, model_id), "application/json");
}

} // namespace ninfer::serve
