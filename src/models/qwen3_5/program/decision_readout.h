#pragma once

// Decision-path readout helpers shared with the tests. The decision path scores each branch's
// candidate tokens by gathering their next-token logits out of the [vocab_size, rows] BF16
// readout written by the final projection, then runs candidate_slice_softmax over the slice.
// These helpers isolate the gather (the index math that historically produced NaNs when the
// candidate index was mistaken for a vocab position) so it can be qualified independently.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Exact BF16 -> FP32: a BF16 value is the high 16 bits of its FP32 encoding.
inline float decision_bf16_to_f32(std::uint16_t bits) {
    const std::uint32_t words = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &words, sizeof(value));
    return value;
}

// Gather one branch's candidate logits from the [vocab_size, rows] BF16 decision readout.
// The projection writes readout[vocab][row] at element offset row * vocab_size + vocab (vocab,
// ne[0], is the fastest dimension -- the same layout target_logprobs gathers from), so a
// candidate's logit sits at row * vocab_size + its vocab id. The candidate index is NOT a vocab
// position; indexing by the candidate index reads unrelated logits (the historical NaN bug).
inline std::vector<float> gather_decision_candidate_logits(const std::uint16_t* readout,
                                                           std::size_t vocab_size, std::size_t row,
                                                           const std::vector<std::int32_t>& candidate_ids) {
    std::vector<float> slice(candidate_ids.size());
    for (std::size_t column = 0; column < candidate_ids.size(); ++column) {
        slice[column] = decision_bf16_to_f32(
            readout[row * vocab_size + static_cast<std::size_t>(candidate_ids[column])]);
    }
    return slice;
}

} // namespace ninfer::models::qwen3_5::detail
