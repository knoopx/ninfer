#include "serve/http_server.h"

#include "serve/decisions.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"

#include <exception>
#include <string>

namespace ninfer::serve {

void HttpServer::handle_decisions(const httplib::Request& req, httplib::Response& res) {
    DecisionsRequest request;
    try {
        request = parse_decisions_request(parse_json_body(req));
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    // The request carries no model: the route runs on the model the router has loaded (the
    // resident); a client-sent top-level "model" field is a 422 unknown field (rejected by the
    // request parser). Nothing loaded -> 503 model_not_ready; a loaded model routes without a
    // swap (pinned for the request duration).
    const std::string model_id = router_->loaded_id();
    if (model_id.empty()) {
        ApiError error;
        error.status  = 503;
        error.type    = "server_error";
        error.param   = "model";
        error.code    = "model_not_ready";
        error.message = "no model is loaded";
        write_openai_error(res, error);
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
        write_openai_error(res, error);
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
        write_openai_error(res, error);
        return;
    }

    // decide() runs on the model's loaded engine and maps its own failures to the shared error
    // contract (422 invalid questions, 400 context length, 429/503 admission); a render
    // failure (std::logic_error) escapes to the server's exception handler, which renders a 500.
    ninfer::DecisionResult result;
    try {
        result = decision_service->decide(request, 1.0f);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }
    res.status = 200;
    res.set_content(render_decisions_response(request, result, model_id), "application/json");
}

} // namespace ninfer::serve
