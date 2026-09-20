#include "serve/decisions.h"

#include <iostream>
#include <string>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

DecisionsRequest parse(Json body) { return parse_decisions_request(body); }

// A valid single-question object (noul needs no criteria) used to isolate the context and
// role/content rules under test.
Json valid_questions() { return Json{{"q1", Json{{"type", "noul"}}}}; }

// (a) The messages path with text parts.
int test_text_messages() {
    int failures = 0;
    const Json body =
        Json::parse(R"({"messages":[{"role":"user","content":"hello"}],"questions":{"q1":{"type":"noul"}}})");
    const DecisionsRequest request = parse(body);
    failures += check(request.state_text.empty(), "state_text is empty on the messages path");
    failures += check(request.messages.size() == 1, "one message parsed");
    failures += check(request.messages[0].role == ninfer::ChatRole::User &&
                          request.messages[0].content.size() == 1 &&
                          request.messages[0].content[0].kind == ContentKind::Text &&
                          request.messages[0].content[0].text == "hello",
                      "a string message content becomes one text part");
    failures += check(!request.has_media(), "text-only messages report no media");
    failures += check(request.questions.size() == 1 && request.question_ids.size() == 1 &&
                          request.question_ids[0] == "q1" &&
                          request.questions[0].type == ninfer::DecisionQuestionType::Noul,
                      "questions and ids parse in JSON order");
    return failures;
}

// (b) An image_url data-URI part: the CPU-testable precondition of the vision gate
// (the 400 vision_disabled rejection itself is thrown by GenerationService::decide).
int test_image_media() {
    int failures = 0;
    const Json body = Json::parse(
        R"({"messages":[{"role":"user","content":[{"type":"text","text":"look"},)"
        R"({"type":"image_url","image_url":"data:image/png;base64,AAA="}]}],)"
        R"("questions":{"q1":{"type":"noul"}}})");
    const DecisionsRequest request = parse(body);
    failures += check(request.messages.size() == 1 && request.messages[0].content.size() == 2,
                      "text and image parts both parse");
    failures += check(request.messages[0].content[0].kind == ContentKind::Text &&
                          request.messages[0].content[1].kind == ContentKind::Image &&
                          request.messages[0].content[1].source.value ==
                              "data:image/png;base64,AAA=",
                      "an image_url data URI normalizes to an Image source");
    failures += check(request.has_media(),
                      "an image_url data URI reports media (vision gate precondition)");
    return failures;
}

// (c) Context exclusivity: exactly one non-null of state/messages.
int test_context_exclusivity() {
    int failures                          = 0;
    const ApiError both                   = api_error(
        [&] { (void)parse(Json::parse(R"({"state":"s","messages":[{"role":"user","content":"x"}],"questions":{"q1":{"type":"noul"}}})")); });
    failures += check(both.status == 422 && both.param == "body",
                      "state and messages together are a 422");
    const ApiError neither                =
        api_error([&] { (void)parse(Json::parse(R"({"questions":{"q1":{"type":"noul"}}})")); });
    failures += check(neither.status == 422 && neither.param == "body" &&
                          neither.message == "state or messages is required",
                      "neither state nor messages is a 422");
    const DecisionsRequest null_state     = parse(Json::parse(
        R"({"state":null,"messages":[{"role":"user","content":"x"}],"questions":{"q1":{"type":"noul"}}})"));
    failures += check(null_state.messages.size() == 1 && null_state.state_text.empty(),
                      "a JSON-null state is not a supplied context; the messages path parses");
    const ApiError null_state_only        =
        api_error([&] { (void)parse(Json::parse(R"({"state":null,"questions":{"q1":{"type":"noul"}}})")); });
    failures += check(null_state_only.status == 422 &&
                          null_state_only.message == "state or messages is required",
                      "a null state alone is a 422 (state or messages is required)");
    return failures;
}

// (d) Unsupported roles and content types.
int test_unsupported_roles_and_content() {
    int failures      = 0;
    const ApiError role = api_error(
        [&] { (void)parse(Json::parse(R"({"messages":[{"role":"tool","content":"x"}],"questions":{"q1":{"type":"noul"}}})")); });
    failures += check(role.status == 422 && role.param == "messages[0].role" &&
                          role.message.find("tool") != std::string::npos,
                      "an unsupported tool role is a 422 naming the role");
    const ApiError audio     = api_error([&] {
        (void)parse(Json::parse(
            R"({"messages":[{"role":"user","content":[{"type":"audio"}]}],"questions":{"q1":{"type":"noul"}}})"));
    });
    failures += check(audio.status == 422 && audio.param == "messages[0].content[0]" &&
                          audio.message.find("audio") != std::string::npos,
                      "an unsupported audio content type is a 422 naming the part");
    return failures;
}

// (e) options: the optional diagnostics object. raw_logits defaults to false, parses when
// true/false, and a non-object options, an unknown option field, or a non-boolean raw_logits
// is a 422 naming the field.
int test_options() {
    int failures      = 0;
    const DecisionsRequest absent = parse(Json::parse(
        R"({"state":"s","questions":{"q1":{"type":"noul"}}})"));
    failures += check(!absent.raw_logits, "options omitted leaves raw_logits false");
    const DecisionsRequest on = parse(Json::parse(
        R"({"state":"s","options":{"raw_logits":true},"questions":{"q1":{"type":"noul"}}})"));
    failures += check(on.raw_logits, "options.raw_logits=true sets the flag");
    const DecisionsRequest off = parse(Json::parse(
        R"({"state":"s","options":{"raw_logits":false},"questions":{"q1":{"type":"noul"}}})"));
    failures += check(!off.raw_logits, "options.raw_logits=false leaves the flag unset");
    const ApiError not_object = api_error([&] {
        (void)parse(Json::parse(
            R"({"state":"s","options":"raw","questions":{"q1":{"type":"noul"}}})"));
    });
    failures += check(not_object.status == 422 && not_object.param == "options",
                      "a non-object options is a 422");
    const ApiError unknown_key = api_error([&] {
        (void)parse(Json::parse(
            R"({"state":"s","options":{"verbose":true},"questions":{"q1":{"type":"noul"}}})"));
    });
    failures += check(unknown_key.status == 422 && unknown_key.param == "options.verbose" &&
                          unknown_key.message.find("verbose") != std::string::npos,
                      "an unknown option field is a 422 naming the field");
    const ApiError non_boolean = api_error([&] {
        (void)parse(Json::parse(
            R"({"state":"s","options":{"raw_logits":"yes"},"questions":{"q1":{"type":"noul"}}})"));
    });
    failures += check(non_boolean.status == 422 && non_boolean.param == "options.raw_logits",
                      "a non-boolean raw_logits is a 422");
    return failures;
}

// (f) Criteria bounds: choice takes 2-255 options and score 2-50 levels.
int test_criteria_bounds() {
    int failures      = 0;
    const ApiError one_option = api_error([&] {
        (void)parse(Json::parse(
            R"({"state":"s","questions":{"q1":{"type":"choice","criteria":{"a":null}}}})"));
    });
    failures += check(one_option.status == 422 && one_option.param == "questions.q1.criteria",
                      "a single-option choice is a 422");
    const DecisionsRequest two_options = parse(Json::parse(
        R"({"state":"s","questions":{"q1":{"type":"choice","criteria":{"a":null,"b":null}}}})"));
    failures += check(two_options.questions.size() == 1 &&
                          two_options.questions[0].criteria.size() == 2,
                      "a two-option choice parses");
    Json crit = Json::object();
    for (int i = 0; i < 255; ++i) { crit["o" + std::to_string(i)] = nullptr; }
    const DecisionsRequest max_choice = parse(
        Json::parse(R"({"state":"s","questions":{"q1":{"type":"choice","criteria":)" +
                    crit.dump() + "}}}")
    );
    failures += check(max_choice.questions[0].criteria.size() == 255,
                      "a 255-option choice parses");
    crit["o255"] = nullptr;
    const ApiError choice_256 = api_error([&] {
        (void)parse(Json::parse(R"({"state":"s","questions":{"q1":{"type":"choice","criteria":)" +
                                crit.dump() + "}}}")
        );
    });
    failures += check(choice_256.status == 422 && choice_256.param == "questions.q1.criteria",
                      "a 256-option choice is a 422");
    const ApiError one_level = api_error([&] {
        (void)parse(Json::parse(
            R"({"state":"s","questions":{"q1":{"type":"score","criteria":["only"]}}})"));
    });
    failures += check(one_level.status == 422, "a single-level score is a 422");
    Json levels = Json::array();
    for (int i = 0; i < 50; ++i) { levels.push_back("l" + std::to_string(i)); }
    const DecisionsRequest max_score = parse(
        Json::parse(R"({"state":"s","questions":{"q1":{"type":"score","criteria":)" +
                    levels.dump() + "}}}")
    );
    failures += check(max_score.questions[0].criteria.size() == 50,
                      "a 50-level score parses");
    levels.push_back("l50");
    const ApiError score_51 = api_error([&] {
        (void)parse(Json::parse(R"({"state":"s","questions":{"q1":{"type":"score","criteria":)" +
                                levels.dump() + "}}}")
        );
    });
    failures += check(score_51.status == 422, "a 51-level score is a 422");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_text_messages();
    failures += test_image_media();
    failures += test_context_exclusivity();
    failures += test_unsupported_roles_and_content();
    failures += test_options();
    failures += test_criteria_bounds();
    if (failures == 0) { std::cout << "decisions request tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
