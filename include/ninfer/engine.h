#pragma once

#include "ninfer/decision.h"
#include "ninfer/types.h"

#include <chrono>
#include <memory>
#include <string_view>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    // Pimpl. The nested class NAME is public (its body stays opaque in this header) so a
    // GenerationHandle's Impl can hold a typed std::shared_ptr<Engine::Impl> keep-alive
    // back-reference: each live handle pins the Engine's VRAM and its destructor releases
    // that pin (active_handle_pins / wait_handle_pins_zero).
    class Impl;

    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()).
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    // Renders the shared state prefix + one branch per question via the model Frontend (the
    // input of Engine::decision_score); throws std::invalid_argument on framing/label-table
    // contract violations.
    [[nodiscard]] DecisionPrepared prepare_decision(std::string state_text,
                                                    std::vector<DecisionQuestion> questions) const;

    // Media-capable decision preparation: the state context is a PromptInput (chat messages that
    // may carry image/video parts). The Frontend prepends the decision system prompt, expands any
    // media to Vision tokens, and stores the media-expanded prefix in DecisionPrepared.state_media
    // (a text-only context leaves state_media null and fills state_tokens). Throws
    // std::invalid_argument on framing/label-table contract violations.
    [[nodiscard]] DecisionPrepared prepare_decision(PromptInput input,
                                                    std::vector<DecisionQuestion> questions) const;

    // Returns one DecisionAnswer per prepared branch for a DecisionScoring Engine; zero tokens
    // are generated.
    [[nodiscard]] DecisionResult decision_score(DecisionPrepared prepared,
                                                float temperature = 1.0f);

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink. Observation options request protocol-neutral publication facts without changing the
    // execution request.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] ModelMetadata model_metadata() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] bool is_available() const;

    // Live VRAM pin count: the number of GenerationHandles currently pinning this Engine's
    // VRAM (weights, KV cache, CUDA graphs). Each handle returned by submit() pins the Engine
    // until it is destroyed; ~Engine() alone does not free VRAM (it drops one ref, and every
    // live handle holds another). The serve router's swap drain gates on this count before
    // destroying an evicted resident: the router's Grant in-flight counter can reach 0 while a
    // retained/streaming GenerationHandle still pins the VRAM (concurrent residency -> OOM).
    [[nodiscard]] std::size_t active_handle_pins() const;
    // Bounded wake on the live-handle pin count reaching zero: blocks until no GenerationHandle
    // pins the Engine's VRAM (active_handle_pins() == 0) or `deadline` passes; returns true when
    // zero live pins are observed (a no-engine Engine has no pins, so it returns true immediately).
    // Lets the serve router's swap drain gate on the real pin without busy-polling.
    [[nodiscard]] bool wait_handle_pins_zero(std::chrono::steady_clock::time_point deadline) const;

    void reset_memory_peaks() noexcept;

private:
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
