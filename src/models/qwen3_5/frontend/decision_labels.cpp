#include "models/qwen3_5/frontend/decision_labels.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

namespace fi = frontend;

// Choice-option cap of the decision wire contract; the label table is compiled to this cap
// (the per-artifact cap).
constexpr std::size_t kDecisionMaxOptions = 255;

// All A..Z products of the given width, in lexicographic (itertools.product) order: the last
// position varies fastest.
std::vector<std::string> letter_products(std::size_t width) {
    static constexpr char kLetters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::size_t total = 1;
    for (std::size_t i = 0; i < width; ++i) { total *= 26; }
    std::vector<std::string> result;
    result.reserve(total);
    std::string buffer(width, 'A');
    for (std::size_t index = 0; index < total; ++index) {
        std::size_t remainder = index;
        for (std::size_t pos = width; pos-- > 0;) {
            buffer[pos] = kLetters[remainder % 26];
            remainder /= 26;
        }
        result.push_back(buffer);
    }
    return result;
}

} // namespace

DecisionLabelTable compile_decision_label_table(const fi::Tokenizer& tokenizer) {
    // Answer-boundary scan: accept a code when encode("Answer: " + code) is the tokenization
    // of "Answer:" extended by exactly one new, unseen token. The code token is normally the
    // space-merged form (e.g. " A"), so no decode round-trip is required.
    const std::string boundary = "Answer:";
    const std::vector<int> prefix = tokenizer.encode(boundary);
    DecisionLabelTable table;
    table.codes.reserve(kDecisionMaxOptions);
    table.token_ids.reserve(kDecisionMaxOptions);
    std::unordered_set<int> seen;
    for (std::size_t width = 1; width <= 3 && table.codes.size() < kDecisionMaxOptions; ++width) {
        for (const std::string& code : letter_products(width)) {
            if (table.codes.size() == kDecisionMaxOptions) break;
            const std::vector<int> ids = tokenizer.encode(boundary + " " + code);
            // Accept only when the boundary tokenization is extended by exactly one token...
            if (ids.size() != prefix.size() + 1) continue;
            if (!std::equal(prefix.begin(), prefix.end(), ids.begin())) continue;
            const int token = ids.back();
            // ...that has not been claimed by an earlier code.
            if (!seen.insert(token).second) continue;
            table.codes.push_back(code);
            table.token_ids.push_back(static_cast<TokenId>(token));
        }
    }
    return table;
}

} // namespace ninfer::models::qwen3_5
