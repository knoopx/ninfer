#include "serve/decisions.h"

#include "serve/request.h"

#include <cstddef>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

// The 422 variant of the request_validation bad_request pattern (which is a 400): every
// decisions-contract violation is an unprocessable 422, not a 400.
[[noreturn]] void unprocessable(std::string message, std::string param) {
    ApiError error;
    error.status  = 422;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    throw ApiException(std::move(error));
}

// Content value: str | dict | list | null. Any JSON value is accepted; the raw text is the
// canonical compact dump (null renders as "null"), preserving the value verbatim in the
// compiled prompt.
std::string content_json(const RequestJson& value) { return value.dump(); }

// An image_url content part: the `image_url` value is a string URL or an object {url, detail?}.
// The URL must be an HTTP(S) URL or a base64 data URI. Detail is omitted or "auto" (an explicit
// profile is rejected, matching the chat route). Returns the media source for the product layer
// to acquire.
ninfer::product::media_acquire::Source
parse_decision_image_source(const RequestJson& part, const std::string& param_prefix) {
    const std::string field = "image_url";
    if (!part.contains(field)) {
        unprocessable(field + " content part must contain " + field, param_prefix);
    }
    const RequestJson& value = part.at(field);
    std::string url;
    if (value.is_string()) {
        url = value.get<std::string>();
    } else if (value.is_object()) {
        if (!value.contains("url") || !value.at("url").is_string()) {
            unprocessable(field + " must contain a string url", param_prefix);
        }
        url = value.at("url").get<std::string>();
        if (value.contains("detail") && !value.at("detail").is_null()) {
            if (!value.at("detail").is_string()) {
                unprocessable("image_url.detail must be a string", param_prefix);
            }
            const std::string detail = value.at("detail").get<std::string>();
            if (detail != "auto") {
                unprocessable("image_url.detail='" + detail +
                                 "' requests an explicit preprocessing profile that NInfer's "
                                 "fixed Vision frontend cannot apply; use 'auto'",
                              param_prefix);
            }
        }
    } else {
        unprocessable(field + " must be a URL string or object", param_prefix);
    }
    if (url.empty()) { unprocessable(field + " URL must not be empty", param_prefix); }

    ninfer::product::media_acquire::Source source;
    source.value = std::move(url);
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        unprocessable(field + " must use HTTP(S) or a data URI", param_prefix);
    }
    return source;
}

// One decision message: {role, content}. role is a supported chat role; content is a string
// (one text part) or an array of text/image_url parts. Unknown fields are rejected.
void parse_decision_message(const RequestJson& entry, std::size_t index, ChatTurn& turn) {
    const std::string param_prefix = "messages[" + std::to_string(index) + "]";
    if (!entry.is_object()) {
        unprocessable("message must be an object", param_prefix);
    }
    for (const auto& [key, value] : entry.items()) {
        if (key != "role" && key != "content") {
            unprocessable("unknown field \"" + key + "\"", param_prefix + "." + key);
        }
    }
    if (!entry.contains("role") || !entry.at("role").is_string()) {
        unprocessable("message role is required and must be a string", param_prefix + ".role");
    }
    const std::string role = entry.at("role").get<std::string>();
    if (role == "system") {
        turn.role = ChatRole::System;
    } else if (role == "developer") {
        turn.role = ChatRole::Developer;
    } else if (role == "user") {
        turn.role = ChatRole::User;
    } else if (role == "assistant") {
        turn.role = ChatRole::Assistant;
    } else {
        unprocessable("message role must be one of \"system\", \"developer\", \"user\", "
                      "\"assistant\"; got \"" +
                          role + "\"",
                      param_prefix + ".role");
    }
    if (!entry.contains("content") || entry.at("content").is_null()) {
        unprocessable("message content is required", param_prefix + ".content");
    }
    const RequestJson& content = entry.at("content");
    if (content.is_string()) {
        turn.content.push_back(
            ContentPart{.kind = ContentKind::Text, .text = content.get<std::string>(),
                        .type_raw = "text"});
        return;
    }
    if (!content.is_array()) {
        unprocessable("message content must be a string or an array of parts",
                      param_prefix + ".content");
    }
    for (std::size_t part_index = 0; part_index < content.size(); ++part_index) {
        const RequestJson& part = content[part_index];
        const std::string part_param = param_prefix + ".content[" + std::to_string(part_index) + "]";
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string()) {
            unprocessable("content part must be an object with a string type", part_param);
        }
        const std::string type = part.at("type").get<std::string>();
        ContentPart parsed;
        parsed.type_raw = type;
        if (type == "text") {
            if (!part.contains("text") || !part.at("text").is_string()) {
                unprocessable("text content part must contain a string text", part_param);
            }
            parsed.kind = ContentKind::Text;
            parsed.text = part.at("text").get<std::string>();
        } else if (type == "image_url") {
            parsed.kind   = ContentKind::Image;
            parsed.source = parse_decision_image_source(part, part_param);
        } else {
            unprocessable("content type \"" + type +
                             "\" is not supported; decisions messages accept \"text\" and "
                             "\"image_url\" parts",
                          part_param);
        }
        turn.content.push_back(std::move(parsed));
    }
}

} // namespace

DecisionsRequest parse_decisions_request(const RequestJson& body) {
    if (!body.is_object()) {
        unprocessable("request body must be a JSON object", "body");
    }
    DecisionsRequest request;

    // The request carries no model: the route runs on the model the router has loaded (the
    // resident). A top-level "model" field is a 422 (unknown field). Extra top-level fields
    // are forbidden (the allowed set is {state, messages, questions, options}).
    for (const auto& [key, value] : body.items()) {
        if (key != "state" && key != "messages" && key != "questions" && key != "options") {
            unprocessable("unknown field \"" + key + "\"", "body." + key);
        }
    }

    // options: the optional diagnostics object. The only supported key is raw_logits (a
    // boolean); an unknown option field is a 422 (the offending field is named).
    if (body.contains("options")) {
        const RequestJson& options = body["options"];
        if (!options.is_object()) {
            unprocessable("options must be an object", "options");
        }
        for (const auto& [key, value] : options.items()) {
            if (key != "raw_logits") {
                unprocessable("unknown option field \"" + key + "\"", "options." + key);
            }
        }
        if (options.contains("raw_logits")) {
            const RequestJson& raw_logits = options["raw_logits"];
            if (!raw_logits.is_boolean()) {
                unprocessable("options.raw_logits must be a boolean", "options.raw_logits");
            }
            request.raw_logits = raw_logits.get<bool>();
        }
    }

    // Context: exactly one non-null of state/messages. A JSON null state renders as the text
    // "null" (json.dumps(None)); a string state is verbatim; structured state is JSON-dumped
    // UTF-8. An explicit null messages is not a supplied context.
    const bool has_state    = body.contains("state") && !body.at("state").is_null();
    const bool has_messages = body.contains("messages") && !body.at("messages").is_null();
    if (has_state && has_messages) {
        unprocessable("supply exactly one of state or messages", "body");
    }
    if (has_state) {
        const RequestJson& state = body.at("state");
        if (state.is_string()) {
            request.state_text = state.get<std::string>();
        } else if (state.is_object() || state.is_array()) {
            request.state_text = state.dump();
        } else {
            unprocessable("state must be a string, object, array, or null", "state");
        }
    } else if (has_messages) {
        const RequestJson& messages = body.at("messages");
        if (!messages.is_array() || messages.empty()) {
            unprocessable("messages must be a non-empty array", "messages");
        }
        request.messages.reserve(messages.size());
        for (std::size_t index = 0; index < messages.size(); ++index) {
            ChatTurn turn;
            parse_decision_message(messages[index], index, turn);
            request.messages.push_back(std::move(turn));
        }
    } else {
        unprocessable("state or messages is required", "body");
    }

    // questions: required non-empty object, iterated in JSON (insertion) order.
    if (!body.contains("questions") || !body["questions"].is_object()) {
        unprocessable("questions is required and must be a non-empty object", "questions");
    }
    const RequestJson& questions = body["questions"];
    if (questions.empty()) {
        unprocessable("questions must be a non-empty object", "questions");
    }

    for (const auto& [id, entry] : questions.items()) {
        const std::string param_prefix = "questions." + id;
        if (!entry.is_object()) {
            unprocessable("question must be an object", param_prefix);
        }

        // Extra fields are forbidden in a question object (the allowed set is
        // {type, instructions, criteria}).
        for (const auto& [key, value] : entry.items()) {
            if (key != "type" && key != "instructions" && key != "criteria") {
                unprocessable("unknown field \"" + key + "\"", param_prefix + "." + key);
            }
        }

        // type: required string in {noul, choice, score}; the message names the offending value.
        if (!entry.contains("type") || !entry["type"].is_string()) {
            const std::string offending =
                entry.contains("type") && !entry["type"].is_null() ? entry["type"].dump() : "null";
            unprocessable("question type must be one of \"noul\", \"choice\", \"score\"; got " +
                              offending,
                          param_prefix + ".type");
        }
        const std::string type = entry["type"].get<std::string>();
        ninfer::DecisionQuestionType question_type;
        if (type == "noul") {
            question_type = ninfer::DecisionQuestionType::Noul;
        } else if (type == "choice") {
            question_type = ninfer::DecisionQuestionType::Choice;
        } else if (type == "score") {
            question_type = ninfer::DecisionQuestionType::Score;
        } else {
            unprocessable("question type must be one of \"noul\", \"choice\", \"score\"; got \"" +
                              type + "\"",
                          param_prefix + ".type");
        }

        // instructions: Content, optional, default null (any JSON value).
        ninfer::DecisionQuestion question;
        question.type = question_type;
        if (entry.contains("instructions")) {
            question.instructions_json = content_json(entry["instructions"]);
        } else {
            question.instructions_json = "null";
        }

        const std::string criteria_param = param_prefix + ".criteria";
        switch (question_type) {
        case ninfer::DecisionQuestionType::Noul:
            // noul: the fixed ["false","true"] option pair; an optional criteria map
            // with keys restricted to "true"/"false" supplies the (any-JSON) descriptions.
            question.criteria             = {"false", "true"};
            question.criteria_values_json = {"null", "null"};
            if (entry.contains("criteria") && !entry["criteria"].is_null()) {
                const RequestJson& criteria = entry["criteria"];
                if (!criteria.is_object()) {
                    unprocessable("noul criteria must be an object with keys \"true\" and \"false\"",
                                  criteria_param);
                }
                for (const auto& [key, value] : criteria.items()) {
                    if (key != "true" && key != "false") {
                        unprocessable("noul criteria has an unsupported key \"" + key +
                                          "\"; only \"true\" and \"false\" are allowed",
                                      criteria_param);
                    }
                }
                question.criteria_values_json[0] = content_json(criteria.value("false", RequestJson()));
                question.criteria_values_json[1] = content_json(criteria.value("true", RequestJson()));
            }
            break;
        case ninfer::DecisionQuestionType::Choice: {
            if (!entry.contains("criteria") || !entry["criteria"].is_object() ||
                entry["criteria"].size() < 2) {
                unprocessable("choice criteria must be a map with at least 2 options", criteria_param);
            }
            const RequestJson& criteria = entry["criteria"];
            if (criteria.size() > 255) {
                unprocessable("choice criteria must have at most 255 options", criteria_param);
            }
            for (const auto& [key, value] : criteria.items()) {
                question.criteria.push_back(key);
                question.criteria_values_json.push_back(content_json(value));
            }
            break;
        }
        case ninfer::DecisionQuestionType::Score: {
            if (!entry.contains("criteria") || !entry["criteria"].is_array()) {
                unprocessable("score criteria must be an array of level descriptions",
                              criteria_param);
            }
            const RequestJson& criteria = entry["criteria"];
            if (criteria.size() < 2) {
                unprocessable("score criteria must have at least 2 levels", criteria_param);
            }
            if (criteria.size() > 50) {
                unprocessable("score criteria must have at most 50 levels", criteria_param);
            }
            for (std::size_t level = 0; level < criteria.size(); ++level) {
                question.criteria.push_back(std::to_string(level));
                question.criteria_values_json.push_back(content_json(criteria[level]));
            }
            break;
        }
        }

        request.question_ids.push_back(id);
        request.questions.push_back(std::move(question));
    }
    return request;
}

} // namespace ninfer::serve
