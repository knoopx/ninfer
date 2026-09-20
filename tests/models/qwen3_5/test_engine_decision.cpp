#include "ninfer/decision.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Pipeline oracle for the decision-scoring readout: run a real artifact through
// prepare_decision + decision_score and check the per-candidate probabilities are finite,
// in [0,1], and sum to one per question. A NaN/inf in any candidate probability means the
// decision readout (state prefill -> suffix -> output_head projection -> gather -> softmax)
// produced a non-finite value, which this test pinpoints at the pipeline level.
int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::Generation;
    options.max_context   = 32768;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(32768);
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    // The decision fork needs a branch KV tail per candidate; the default context-cache
    // capacities (P=2C, S=max(C,4)) are too small for the branch fork at C=1, so raise them.
    options.context_cache.max_private_continuations = 64;
    options.context_cache.max_shared_prefixes       = 64;
    ninfer::Engine engine(options);

    // A short, self-contained state plus one 2-option choice and one 3-level score.
    const std::string state =
        "A customer reports their account was locked after a password reset. "
        "Support policy: locked accounts after a reset go to the access queue; "
        "billing disputes go to billing; everything else goes to general.";
    std::vector<ninfer::DecisionQuestion> questions;

    // instructions_json and criteria_values_json are raw JSON values (the engine parses them
    // verbatim), so the text is JSON-encoded as a JSON string.
    ninfer::DecisionQuestion choice;
    choice.type                = ninfer::DecisionQuestionType::Choice;
    choice.instructions_json   = "\"Which queue should handle this request?\"";
    choice.criteria            = {"access", "billing"};
    choice.criteria_values_json = {"\"Account access support.\"", "\"Billing support.\""};
    questions.push_back(choice);

    ninfer::DecisionQuestion score;
    score.type                = ninfer::DecisionQuestionType::Score;
    score.instructions_json   = "\"How urgent is this request on a 0-2 scale?\"";
    score.criteria            = {"0", "1", "2"};
    score.criteria_values_json = {"\"Not urgent.\"", "\"Moderate.\"", "\"Critical.\""};
    questions.push_back(score);

    ninfer::DecisionPrepared prepared = engine.prepare_decision(state, questions);
    // decision_score consumes its prepared input by move, so capture the per-branch candidate
    // counts before the move to validate the readout answers against them.
    std::vector<std::size_t> candidate_counts;
    candidate_counts.reserve(prepared.branches.size());
    for (const auto& branch : prepared.branches) {
        candidate_counts.push_back(branch.candidate_ids.size());
    }
    const ninfer::DecisionResult result = engine.decision_score(std::move(prepared), 1.0F);
    if (result.answers.size() != questions.size()) {
        std::cerr << "decision returned " << result.answers.size() << " answers for "
                  << questions.size() << " questions\n";
        return 1;
    }
    if (result.output_tokens != 0) {
        std::cerr << "decision scoring must generate zero tokens\n";
        return 1;
    }
    for (std::size_t q = 0; q < result.answers.size(); ++q) {
        const ninfer::DecisionAnswer& answer = result.answers[q];
        if (answer.probabilities.size() != candidate_counts[q]) {
            std::cerr << "question " << q << " probability count " << answer.probabilities.size()
                      << " != candidate count " << candidate_counts[q] << "\n";
            return 1;
        }
        double sum = 0.0;
        for (std::size_t c = 0; c < answer.probabilities.size(); ++c) {
            const float p = answer.probabilities[c];
            sum += p;
            if (!std::isfinite(p) || p < 0.0F || p > 1.0F) {
                std::cerr << "question " << q << " candidate " << c << " ("
                          << answer.options[c] << ") has non-finite/invalid probability " << p
                          << "\n";
                return 1;
            }
            std::cout << "q" << q << " " << answer.options[c] << " = " << p << "\n";
        }
        if (std::fabs(sum - 1.0) > 1e-3) {
            std::cerr << "question " << q << " probabilities sum to " << sum << ", expected 1\n";
            return 1;
        }
    }
    std::cout << "OK decision\n";
    return 0;
}
