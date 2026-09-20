#pragma once
#include "ninfer/decision.h"
#include "serve/request.h"
#include "serve/request_json.h"
#include <string>
#include <vector>
namespace ninfer::serve {
// Parsed POST /v1/decisions request (the decision-scoring wire contract). questions and
// question_ids are parallel, in JSON object order. Exactly one of the two context fields is
// populated: state_text is the rendered state input (string state verbatim; structured state
// JSON-dumped UTF-8; JSON null state renders as the text "null") for the `state` path, and
// messages is the chat context (text + image parts) for the `messages` path; image/video media
// sources in messages remain unresolved until the product service acquires owning bytes. The
// request carries no model (a top-level "model" field is a 422 unknown field): the route runs
// on the model the router has loaded (the resident), and the response's model field echoes the
// loaded model id.
struct DecisionsRequest {
    std::string state_text;
    std::vector<ChatTurn> messages;
    std::vector<ninfer::DecisionQuestion> questions;
    std::vector<std::string> question_ids;

    // True when the request's optional top-level `options` object set raw_logits: each answer
    // then carries the per-candidate pre-softmax readout logits as a diagnostic field.
    bool raw_logits = false;

    // True when any message part is an image or video source (the vision gating check).
    [[nodiscard]] bool has_media() const noexcept {
        for (const ChatTurn& turn : messages) {
            for (const ContentPart& part : turn.content) {
                if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) {
                    return true;
                }
            }
        }
        return false;
    }
};
// ApiException{422, <field>} on: a top-level "model" field (unknown field); an `options`
// object that is not an object, carries a non-boolean raw_logits, or names an unknown option
// field; missing context (exactly one non-null of state/messages; state may be JSON null); a
// messages entry with an unsupported role or content part; missing/empty questions; unknown
// question type; extra fields in a question object; choice criteria <2 or >255; score criteria
// <2 or >50; noul criteria keys other than "true"/"false". The route runs on the router's
// loaded (resident) model.
[[nodiscard]] DecisionsRequest parse_decisions_request(const RequestJson& body);
// The response shape: {model, answers{<id>: Answer}, usage{input_tokens, output_tokens: 0}}.
// When the request set options.raw_logits, each answer carries a raw_logits map (option to
// pre-softmax logit) parallel to probabilities.
[[nodiscard]] std::string render_decisions_response(const DecisionsRequest& request,
                                                     const ninfer::DecisionResult& result,
                                                     const std::string& model_id);
} // namespace ninfer::serve
