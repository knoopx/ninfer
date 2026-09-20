#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ninfer {

// The media state prefix is the model's media-expanded prepared prompt (the concrete type is
// model-internal; only the qwen3_5 implementation is carried). Forward-declared here so the
// public carrier stays a pointer; the destructor is defined where the type is complete.
namespace models {
namespace qwen3_5 {
struct PreparedPromptData;
}
}

// Decision-scoring question type. The criteria semantics follow the type: noul is the fixed
// ["false","true"] pair, choice carries the option keys, and score carries the 0-based level
// indices "0".."K-1".
enum class DecisionQuestionType : std::uint8_t { Noul, Choice, Score };

// One decision question. The criteria meaning follows the type: noul is the fixed
// ["false","true"] pair, choice carries the option keys, and score the 0-based level
// indices "0".."K-1". instructions_json and criteria_values_json are raw JSON values (the
// text "null" when the request omitted them) so the compiled prompt keeps the values
// verbatim (values may be strings, objects, arrays, or null).
struct DecisionQuestion {
    DecisionQuestionType type;
    // Raw JSON value of the question instructions; "null" when the request omitted the field.
    std::string instructions_json;
    // Option keys: choice carries the criteria keys in question order, score the 0-based level
    // indices "0".."K-1", noul the fixed ["false","true"] pair.
    std::vector<std::string> criteria;
    // Raw JSON values parallel to criteria: choice criteria values, score level descriptions,
    // noul criteria values in fixed order ("null" when the request omitted them).
    std::vector<std::string> criteria_values_json;
};

// One question branch: the rendered suffix following the shared state prefix plus the
// candidate token set. candidate_groups is the per-candidate choice-group index for
// max-pooling; -1 disables pooling (the ops::candidate_slice_softmax group contract).
// options carries the option label per candidate, parallel to candidate_ids: noul is the
// fixed ["false","true"] pair, choice the criteria keys in question order, and score the
// level indices "0".."K-1".
struct DecisionBranch {
    DecisionQuestionType type;
    std::vector<TokenId> suffix_tokens;
    std::vector<TokenId> candidate_ids;
    std::vector<std::int32_t> candidate_groups;
    std::vector<std::string> options;
};

// Prepared decision input: the shared state prefix and one branch per question.
//
// The state prefix is carried in exactly one of two forms. For a text-only state, state_tokens
// holds the rendered prefix and state_media is null. For a media state (image/video parts),
// state_media holds the media-expanded prefix (its token_ids include the Vision tokens, plus the
// token types, positions, media payloads, and Vision items the execution core needs); in that
// case state_tokens is empty. state_media is the model's PreparedPromptData (prepared_prompt.h)
// held by pointer because the concrete type is model-internal; the destructor is defined where
// it is complete (engine.cpp).
struct DecisionPrepared {
    std::vector<TokenId> state_tokens;
    std::vector<DecisionBranch> branches;
    std::unique_ptr<models::qwen3_5::PreparedPromptData> state_media;

    [[nodiscard]] bool has_media() const noexcept { return state_media != nullptr; }

    DecisionPrepared();
    DecisionPrepared(DecisionPrepared&&);
    DecisionPrepared& operator=(DecisionPrepared&&);
    DecisionPrepared(const DecisionPrepared&)            = delete;
    DecisionPrepared& operator=(const DecisionPrepared&) = delete;
    ~DecisionPrepared();
};

// One question answer. options is parallel to probabilities and raw_logits. Per type: noul
// reports noul as P(true) with no confidence; choice reports the winning key plus a normalized
// Gini confidence; score reports the expected 0-based level index plus a normalized Gini
// confidence. raw_logits holds the per-candidate pre-softmax readout logits (the diagnostic
// `options.raw_logits` response field carries them when the request asked for them).
struct DecisionAnswer {
    DecisionQuestionType type;
    std::string winning_option;
    std::vector<std::string> options;
    std::vector<float> probabilities;
    std::vector<float> raw_logits;
    float score      = 0.0f;
    float noul       = 0.0f;
    float confidence = 0.0f;
};

// One decision result: answers parallel to the prepared branches. output_tokens is always 0
// because decision scoring generates no tokens.
struct DecisionResult {
    std::vector<DecisionAnswer> answers;
    std::uint32_t input_tokens  = 0;
    std::uint32_t output_tokens = 0;
};

} // namespace ninfer
