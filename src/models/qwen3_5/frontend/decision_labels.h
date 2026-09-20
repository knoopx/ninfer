#pragma once

#include "models/qwen3_5/frontend/tokenizer.h"
#include <ninfer/types.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5 {

// Per-artifact table of single-token decision answer codes, compiled once at Frontend startup.
// codes and token_ids are parallel: codes[i] is the i-th letter code and token_ids[i] is its
// single-token id. The table holds what the tokenizer supports: a variable-size set of 0..255
// letter codes. max_options() is the per-artifact option cap: a question needing more options
// than the cap is rejected upstream (422).
struct DecisionLabelTable {
    std::vector<std::string> codes;  // 0..255 letter codes (what the tokenizer supports)
    std::vector<TokenId> token_ids;

    [[nodiscard]] std::size_t max_options() const noexcept { return codes.size(); }
};

// Compiles the decision label table with the "Answer:"-boundary scan: for widths 1..3 over
// A..Z letter products, a code is accepted only when
// encode("Answer: " + code) is the tokenization of "Answer:" extended by exactly one new,
// unseen token (the code token, usually the space-merged form such as " A"). No decode
// round-trip is part of the acceptance rule. The result is a variable-size table
// (0..255 codes): a tokenizer that cannot supply 255 distinct codes yields an empty or
// smaller table, a valid outcome; the compile does not throw.
[[nodiscard]] DecisionLabelTable
compile_decision_label_table(const frontend::Tokenizer& tokenizer);

} // namespace ninfer::models::qwen3_5
