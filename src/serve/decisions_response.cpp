#include "serve/decisions.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

using OrderedJson = nlohmann::ordered_json;

// Numbers are emitted as doubles (cast from float) so the wire values are exact JSON numbers.
OrderedJson make_float_option_map(const std::vector<std::string>& keys,
                                  const std::vector<float>& values) {
    OrderedJson map = OrderedJson::object();
    for (std::size_t index = 0; index < keys.size(); ++index) {
        map[keys[index]] = static_cast<double>(values[index]);
    }
    return map;
}

OrderedJson make_json_option_map(const std::vector<std::string>& keys,
                                 const std::vector<std::string>& json_values) {
    OrderedJson map = OrderedJson::object();
    for (std::size_t index = 0; index < keys.size(); ++index) {
        map[keys[index]] = OrderedJson::parse(json_values[index]);
    }
    return map;
}

} // namespace

std::string render_decisions_response(const DecisionsRequest& request,
                                      const ninfer::DecisionResult& result,
                                      const std::string& model_id) {
    if (result.answers.size() != request.questions.size()) {
        throw std::logic_error("decision answer count does not match the question count");
    }
    OrderedJson answers = OrderedJson::object();
    for (std::size_t index = 0; index < request.questions.size(); ++index) {
        const auto& question = request.questions[index];
        const auto& answer   = result.answers[index];
        if (answer.options.size() != answer.probabilities.size() ||
            answer.options.size() != answer.raw_logits.size() ||
            answer.options.size() != question.criteria.size()) {
            throw std::logic_error("decision answer options/probabilities/raw_logits are not "
                                   "parallel to the question criteria");
        }
        OrderedJson item;
        if (request.raw_logits) {
            item["raw_logits"] = make_float_option_map(answer.options, answer.raw_logits);
        }
        switch (question.type) {
        case ninfer::DecisionQuestionType::Noul:
            // The noul answer reports P(true); no confidence field.
            item["type"] = "noul";
            item["noul"] = static_cast<double>(answer.noul);
            break;
        case ninfer::DecisionQuestionType::Choice:
            item["type"]               = "choice";
            item["choice"]             = answer.winning_option;
            item["probabilities"]      = make_float_option_map(answer.options, answer.probabilities);
            item["confidence"]         = static_cast<double>(answer.confidence);
            break;
        case ninfer::DecisionQuestionType::Score:
            // legend maps each option (the 0-based level index) to its level description, a
            // raw JSON value (string, object, array, or null) exactly as the request sent it.
            item["type"]          = "score";
            item["score"]         = static_cast<double>(answer.score);
            item["legend"]        = make_json_option_map(answer.options, question.criteria_values_json);
            item["probabilities"] = make_float_option_map(answer.options, answer.probabilities);
            item["confidence"]    = static_cast<double>(answer.confidence);
            break;
        }
        answers[request.question_ids[index]] = std::move(item);
    }
    OrderedJson body;
    body["model"] = model_id;
    body["answers"] = std::move(answers);
    body["usage"]   = OrderedJson{{"input_tokens", result.input_tokens},
                                 {"output_tokens", 0}};
    return body.dump();
}

} // namespace ninfer::serve
