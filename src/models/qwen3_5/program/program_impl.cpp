#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/transactions/decision_branches.h"
#include "models/qwen3_5/program/decision_readout.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/startup.h"
#include "core/device.h"
#include "ninfer/decision.h"
#include "ninfer/ops/candidate_slice_softmax.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

namespace {

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options);

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options) {
    if (!options.max_private_continuations || *options.max_private_continuations == 0) {
        throw std::logic_error("Qwen3.5 context cache private capacity is not normalized");
    }
    return *options.max_private_continuations;
}

} // namespace

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache),
      continuation_capacity(normalized_private_capacity(plan.context_cache)),
      shared_prefix_capacity(plan.context_cache.max_shared_prefixes.value_or(0)),
      prefill_chunk(plan.prefill_chunk), draft_window(plan.draft_window),
      speculative_backend(plan.speculative_backend), kv_storage(plan.kv_storage),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), causal_scoring(plan.causal_scoring),
      decision_scoring(plan.decision_scoring),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      continuation_states(continuation_capacity), continuation_slots(continuation_capacity),
      shared_prefix_states(shared_prefix_capacity), shared_prefix_slots(shared_prefix_capacity),
      round_host(plan.causal_scoring ? std::nullopt
                                     : std::make_optional<PinnedHostBuffer>(sizeof(TokenId))),
      score_logprobs_host(plan.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                    kCausalScoreTile * sizeof(float))
                                              : std::nullopt),
      decision_readout_logits_host(
          plan.decision_scoring
              ? std::make_optional<PinnedHostBuffer>(
                    workspace_plan.decision_score.readout_logits * 2)
              : std::nullopt),
      decision_probabilities_host(
          plan.decision_scoring
              ? std::make_optional<PinnedHostBuffer>(
                    workspace_plan.decision_score.host_probabilities * sizeof(float))
              : std::nullopt),
      ordinary_host(
          !plan.causal_scoring && plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_5::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::MtpDecodeIngress) +
                                                          sizeof(qwen3_5::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(is_masked_draft_backend(plan.speculative_backend)
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)} {
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        plan.decision_scoring != (workspace_plan.decision_score.bytes != 0) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    if (!plan.context_cache.max_private_continuations || !plan.context_cache.max_shared_prefixes) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::uint64_t address_capacity64 =
        static_cast<std::uint64_t>(*plan.context_cache.max_private_continuations) +
        *plan.context_cache.max_shared_prefixes;
    if (address_capacity64 == 0 || address_capacity64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV address-space capacity exceeds uint32");
    }
    // One unpublished descriptor is reserved for the single in-flight active-capture snapshot.
    // Published private/shared address spaces remain bounded by P + S; the transaction slot lets a
    // full shared catalog replace one entry without releasing the old checkpoint before the new
    // snapshot has been prepared.
    if (address_capacity64 == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV transaction address capacity exceeds uint32");
    }
    const auto address_capacity      = static_cast<std::uint32_t>(address_capacity64 + 1U);
    const auto logical_page_capacity = [&](const DeviceKVPagePool& pool) {
        const HostKVPageLayout host_layout = plan_host_kv_page_layout(pool.geometry());
        const std::uint64_t host_pages =
            plan.context_cache.host_kv_capacity_bytes / host_layout.page_stride;
        const std::uint64_t total = static_cast<std::uint64_t>(pool.capacity_pages()) + host_pages;
        if (total > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 logical KV page capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(total);
    };

    decoder = std::make_unique<qwen3_5::DecoderState>(backing, plan.persistent.decoder);
    text_host_kv_page_stride =
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()).page_stride;
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_page_capacity(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    state_images =
        std::make_unique<qwen3_5::StateImageDevicePool>(backing, plan.persistent.state_images);
    if (plan.context_cache.host_state_slots != 0) {
        const std::uint64_t host_state_bytes =
            static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
            plan.context_cache.host_state_slots;
        StartupPhaseScope host_state_phase(startup_observer, StartupPhase::HostStatePin,
                                           StartupProgressUnit::Bytes, host_state_bytes);
        host_state_images = std::make_unique<qwen3_5::HostStatePool>(
            state_images->host_layout(), plan.context_cache.host_state_slots);
        host_state_phase.complete(host_state_bytes, host_state_bytes);
    }
    const std::uint64_t logical_state_capacity =
        static_cast<std::uint64_t>(state_images->slot_count()) +
        plan.context_cache.host_state_slots;
    if (logical_state_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 logical StateImage capacity exceeds uint32");
    }
    state_store = std::make_unique<StateImageStore>(
        *state_images, host_state_images.get(), static_cast<std::uint32_t>(logical_state_capacity));
    pressure_private_owner_scratch_.resize(continuation_capacity);
    pressure_shared_owner_scratch_.resize(shared_prefix_capacity);
    pressure_private_drop_scratch_.resize(continuation_capacity);
    const std::size_t pressure_checkpoint_capacity =
        2U + context_cache.max_long_anchors_per_continuation.value_or(0U);
    for (auto& dropped : pressure_private_drop_scratch_) {
        dropped.reserve(pressure_checkpoint_capacity);
    }
    pressure_state_scratch_.reserve(static_cast<std::size_t>(logical_state_capacity));
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None) ||
        replay_fold.has_value() != replay_records.has_value()) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) {
        CyclicKVCache* local = state_images->dflash_local();
        if (local == nullptr) {
            throw std::logic_error("DFlash StateImage has no local fixed state");
        }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    if (dflash.has_value() != plan.features.masked_draft()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }
    if (qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
        backend_host_kv_page_stride =
            plan_host_kv_page_layout(backend->page_pool().geometry()).page_stride;
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_page_capacity(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }
    pressure_text_page_scratch_.resize(text_kv_pages->capacity());
    pressure_text_selected_pages_.reserve(text_kv_pages->capacity());
    if (backend_kv_pages) {
        pressure_backend_page_scratch_.resize(backend_kv_pages->capacity());
        pressure_backend_selected_pages_.reserve(backend_kv_pages->capacity());
    }
    if (plan.context_cache.host_kv_capacity_bytes != 0) {
        std::vector<HostKVPageLayout> layouts;
        layouts.push_back(plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()));
        if (const qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
            HostKVPageLayout backend_layout =
                plan_host_kv_page_layout(backend->page_pool().geometry());
            if (backend_layout != layouts.front()) { layouts.push_back(std::move(backend_layout)); }
        }
        StartupPhaseScope host_kv_phase(
            startup_observer, StartupPhase::HostKvPin, StartupProgressUnit::Bytes,
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        host_kv_arena = std::make_unique<HostKVArena>(
            plan.context_cache.host_kv_capacity_bytes,
            std::span<const HostKVPageLayout>(layouts.data(), layouts.size()));
        host_kv_phase.complete(
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes),
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }
        const std::size_t extent_capacity =
            plan.context_cache.host_kv_capacity_bytes / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 Host KV extent capacity exceeds uint32");
        }
        if (extent_capacity != 0) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(
                *host_kv_arena, static_cast<std::uint32_t>(extent_capacity));
        }
    }

    io = qwen3_5::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() !=
        (!causal_scoring && speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    if (plan.persistent.decision_readout_hidden) {
        decision_readout_hidden = plan.persistent.decision_readout_hidden->bind(backing);
    }
    if (plan.persistent.decision_readout_logits) {
        decision_readout_logits = plan.persistent.decision_readout_logits->bind(backing);
    }
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    active_continuations.fill(continuation_capacity);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) { lane_epochs[lane] = 1; }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        SequenceState& sequence = continuation_states[index];
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.long_anchors.reserve(context_cache.max_long_anchors_per_continuation.value_or(0));
        // One retained shared resume source can coexist with every fixed per-request candidate.
        sequence.shared_prefix_references.reserve(8U);
    }
    materialization_ledger_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_identity_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_prefix_digests_.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    set_device_i32(io.text_kv_table_row, 0);
    if (!causal_scoring) { set_device_i32(io.backend_kv_table_row, 0); }

    host_tokens = round_host ? static_cast<TokenId*>(round_host->data()) : nullptr;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_5::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_5::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_5::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_5::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_5::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_5::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_5::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_5::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    }
    device.synchronize();
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        const std::size_t free_before = device.free_bytes();
        prepare_graphs();
        device.synchronize();
        const std::size_t free_after = device.free_bytes();
        graph_measured_bytes         = free_before > free_after ? free_before - free_after : 0;
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

ProgramImpl::~ProgramImpl() noexcept {
    if (device.transfer_stream != nullptr) { (void)cudaStreamSynchronize(device.transfer_stream); }
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

std::vector<float> ProgramImpl::causal_score(PreparedPromptData&& prompt,
                                             std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t entitlement            = kv_pages_for_frontier(predictor_count);
    if (entitlement == 0) { throw std::logic_error("causal score has no KV entitlement"); }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    std::vector<float> output;
    output.reserve(token_count_size - first_target);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;

    try {
        state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(entitlement, 0);
        if (!address) { throw std::bad_alloc(); }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, device.stream);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(
                DType::BF16, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids = work.alloc(DType::I32, {columns});
            Tensor logprobs   = work.alloc(DType::FP32, {columns});
            Tensor hidden     = score_hidden->slice(1, 0, columns);
            execution::project(hidden, parameters.text.output_head, logits, work, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                                    cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids,
                                              dimension(parameters.model.resources().public_token_count),
                                              logprobs, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                                    cudaMemcpyDeviceToHost, device.stream));
            device.synchronize();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);

            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = kCausalScoreTile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == kCausalScoreTile) { flush(); }
            }
        }
        flush();
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

std::unique_ptr<DecisionAdmissionCandidateImpl>
ProgramImpl::inspect_decision_admission(DecisionPrepared prepared) {
    if (!decision_readout_logits_host || !decision_probabilities_host ||
        !decision_readout_hidden || !decision_readout_logits ||
        workspace_plan.decision_score.bytes == 0) {
        return nullptr;
    }
    // The decision path is self-contained (temp state + Main KV row 0 + branch fork + batched
    // suffix prefill + readout) and is serialized by the Generation core against in-flight
    // generation rounds, so it runs on the model's loaded (Generation-purpose) Program.
    if (prepared.branches.empty()) {
        throw std::invalid_argument("decision scoring requires at least one branch");
    }
    const std::uint32_t branch_count = static_cast<std::uint32_t>(prepared.branches.size());
    if (branch_count > workspace_plan.decision_score.branch_kv_rows) {
        throw std::invalid_argument("decision branches exceed the planned branch KV rows");
    }
    const std::uint32_t state_length = prepared.has_media()
                                           ? static_cast<std::uint32_t>(
                                                 prepared.state_media->token_ids.size())
                                           : static_cast<std::uint32_t>(
                                                 prepared.state_tokens.size());
    if (state_length == 0 || state_length > capacity) {
        throw std::invalid_argument("decision state token count must be in [1,capacity]");
    }
    const std::uint32_t vocabulary = dimension(parameters.model.resources().public_token_count);
    std::uint32_t input_tokens = state_length;
    std::uint32_t max_suffix_length = 0;
    for (const DecisionBranch& branch : prepared.branches) {
        if (branch.suffix_tokens.empty()) {
            throw std::invalid_argument("decision branch suffix must not be empty");
        }
        max_suffix_length =
            std::max(max_suffix_length, static_cast<std::uint32_t>(branch.suffix_tokens.size()));
        if (static_cast<std::uint64_t>(state_length) + branch.suffix_tokens.size() > capacity) {
            throw std::invalid_argument("decision branch exceeds the sequence capacity");
        }
        input_tokens += static_cast<std::uint32_t>(branch.suffix_tokens.size());
        if (branch.candidate_ids.empty() || branch.candidate_ids.size() > kDecisionMaxCandidates) {
            throw std::invalid_argument("decision branch candidate count must be in [1,256]");
        }
        if (branch.candidate_groups.size() != branch.candidate_ids.size() ||
            branch.options.size() != branch.candidate_ids.size()) {
            throw std::invalid_argument("decision branch fields must be parallel");
        }
        for (const TokenId candidate : branch.candidate_ids) {
            if (candidate < 0 || static_cast<std::uint32_t>(candidate) >= vocabulary) {
                throw std::invalid_argument("decision candidate token id is out of range");
            }
        }
        for (const std::int32_t group : branch.candidate_groups) {
            if (group != -1 &&
                (group < 0 || group >= static_cast<std::int32_t>(branch.candidate_ids.size()))) {
                throw std::invalid_argument("decision candidate group is out of range");
            }
        }
        if (branch.type == DecisionQuestionType::Noul && branch.candidate_ids.size() != 2) {
            throw std::invalid_argument("noul branch requires the fixed candidate pair");
        }
    }

    const std::uint32_t state_pages = kv_pages_for_frontier(state_length);
    // Size the per-branch KV page entitlement to the actual state + suffix pages, not the planned
    // cap: a short-suffix decision must not reserve a full tail page group per branch, which
    // exhausts the device page pool when several branches fork at once.
    const std::uint32_t branch_entitlement =
        std::min(workspace_plan.decision_score.branch_tail_pages,
                 state_pages + kv_pages_for_frontier(max_suffix_length));
    const std::uint32_t entitlement = branch_entitlement;
    if (state_pages == 0 || state_pages > entitlement) {
        throw std::invalid_argument("decision state pages exceed the branch KV tail entitlement");
    }

    // Soft feasibility check: a decision reserves branch_count state rows and branch_count KV
    // addresses (one continuation per branch, plus the caller-owned prefilled row 0). The KV page
    // demand (branch_count * entitlement) is enforced per-address by create_active, not by the
    // address count, so infeasibility is a normal admission result, reported without reserving.
    if (state_store->capacity() - state_store->occupied() < branch_count ||
        text_kv_addresses->capacity() - text_kv_addresses->occupied() < branch_count) {
        return nullptr;
    }

    auto candidate = std::make_unique<DecisionAdmissionCandidateImpl>();
    candidate->prepared     = std::move(prepared);
    candidate->branch_count = branch_count;
    candidate->state_length = state_length;
    candidate->entitlement  = entitlement;
    candidate->input_tokens = input_tokens;
    candidate->revision     = resource_revision();
    return candidate;
}

runtime::ContextTransactionReserveStatus
ProgramImpl::start_decision_transaction(DecisionAdmissionCandidateImpl&& candidate,
                                        runtime::CancellationFlagView cancellation) {
    if (decision_candidate_.has_value()) {
        throw std::logic_error("decision transaction already open");
    }
    if (cancellation.requested()) { return runtime::ContextTransactionReserveStatus::Aborted; }
    try {
        candidate.state = state_store->reserve_reset(device.stream);
        if (!candidate.state) {
            return runtime::ContextTransactionReserveStatus::Aborted;
        }
        candidate.address = text_kv_addresses->create_active(candidate.entitlement, 0);
        if (!candidate.address) {
            (void)state_store->release(*candidate.state);
            candidate.state.reset();
            return runtime::ContextTransactionReserveStatus::Aborted;
        }
        if (text_kv_addresses->bound_row(*candidate.address) != 0) {
            throw std::logic_error("decision transaction did not bind the unique Main KV row");
        }
        candidate.transaction_open = true;
        decision_candidate_ = std::move(candidate);
        return runtime::ContextTransactionReserveStatus::Reserved;
    } catch (...) {
        if (candidate.state) { (void)state_store->release(*candidate.state); }
        if (candidate.address) {
            if (text_kv_addresses->active(*candidate.address)) {
                text_kv_addresses->deactivate(*candidate.address);
            }
            (void)text_kv_addresses->release(*candidate.address);
        }
        throw;
    }
}

DecisionResult ProgramImpl::progress_decision_transaction(float temperature,
                                                          runtime::CancellationFlagView cancellation) {
    (void)cancellation;
    DecisionAdmissionCandidateImpl& candidate = *decision_candidate_;
    const DecisionPrepared& prepared          = candidate.prepared;
    const std::uint32_t state_length          = candidate.state_length;
    const std::uint32_t branch_count          = candidate.branch_count;
    const std::uint32_t entitlement           = candidate.entitlement;
    const std::uint32_t branch_entitlement    = candidate.entitlement;
    // Release forked rows/states first, then the caller-owned prefilled row 0 / state 0
    // (deactivated by open_branches), mirroring the causal_score cleanup.
    const auto cleanup = [&] {
        if (candidate.reservation_open) {
            release_branches(*this, std::move(candidate.reservation));
            candidate.reservation_open = false;
        }
        if (candidate.address) {
            if (text_kv_addresses->active(*candidate.address)) {
                text_kv_addresses->deactivate(*candidate.address);
            }
            (void)text_kv_addresses->release(*candidate.address);
            candidate.address.reset();
        }
        if (candidate.state) {
            (void)state_store->release(*candidate.state);
            candidate.state.reset();
        }
    };

    DecisionResult result;
    result.input_tokens = candidate.input_tokens;
    try {
        text_kv_addresses->ensure_mapped_to_tokens(*candidate.address, state_length,
                                                   device.stream);

        // Shared state prefill on row 0 (the causal_score chunk loop without score staging).
        const std::int32_t state_slot = state_store->physical_slot(*candidate.state);
        std::uint32_t cursor = 0;
        qwen3_5::PreparedPromptData* media = prepared.state_media.get();
        if (media != nullptr) {
            if (!workspace_plan.vision) {
                throw std::logic_error("decision media prefill has no vision workspace plan");
            }
            VisionPrefillPlan vision_plan;
            vision_plan.control_plan = std::make_shared<const qwen3_5::VisionControlPlan>(
                qwen3_5::plan_vision_control(*media, *parameters.model.config().vision));
            vision_plan.uses.reserve(vision_plan.control_plan->items.size());
            for (std::size_t index = 0; index < vision_plan.control_plan->items.size(); ++index) {
                const qwen3_5::VisionItemControlPlan& item =
                    vision_plan.control_plan->items[index];
                vision_plan.uses.push_back(VisionUseSpan{
                    .begin               = item.token_begin,
                    .end                 = item.token_end,
                    .prepared_item_index = static_cast<std::uint32_t>(index),
                });
                vision_plan.max_merged_count =
                    std::max(vision_plan.max_merged_count, item.merged_count);
            }
            vision_plan.control = std::make_shared<const qwen3_5::VisionControl>(
                qwen3_5::build_vision_control(*media, *vision_plan.control_plan, 0));
            auto vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, *media, vision_plan, vision_handoff_peak_bytes);
            while (cursor < state_length) {
                const std::uint32_t nominal = std::min(prefill_chunk, state_length - cursor);
                execution::PrefillContext schedule_state{
                    {device, parameters, work, state_images->linear(), nullptr, io,
                     prefill_hidden, prefill_chunk, proposal_head},
                    decoder->text_kv.execution_view(
                        text_kv_addresses->execution_row(*candidate.address)),
                    {},
                    decoder->text_kv,
                    nullptr,
                    nullptr,
                    cursor,
                    nullptr,
                    nullptr,
                    state_slot,
                    state_slot,
                    0,
                    nullptr};
                mark_workspace_usage(workspace_plan.vision->capacity_bytes);
                const auto chunk = execution::prefill_multimodal_chunk(
                    schedule_state, *media, *vision, nominal, std::nullopt,
                    cursor + nominal == state_length);
                vision->release_encoded_media_payloads();
                if (chunk.processed_tokens == 0 || chunk.processed_tokens > nominal) {
                    throw std::logic_error("decision score state Prefill made invalid progress");
                }
                cursor += chunk.processed_tokens;
                text_kv_addresses->commit_frontier(*candidate.address, cursor);
            }
        } else {
            while (cursor < state_length) {
                const std::uint32_t nominal = std::min(prefill_chunk, state_length - cursor);
                execution::PrefillContext schedule_state{
                    {device, parameters, work, state_images->linear(), nullptr, io,
                     prefill_hidden, prefill_chunk, proposal_head},
                    decoder->text_kv.execution_view(
                        text_kv_addresses->execution_row(*candidate.address)),
                    {},
                    decoder->text_kv,
                    nullptr,
                    nullptr,
                    cursor,
                    nullptr,
                    nullptr,
                    state_slot,
                    state_slot,
                    0,
                    nullptr};
                mark_workspace_usage(workspace_plan.text_prefill);
                const execution::PrefillChunkResult chunk =
                    execution::prefill_text_chunk(schedule_state,
                                                  std::span<const TokenId>(prepared.state_tokens),
                                                  nominal, std::nullopt, false);
                if (chunk.finalized || chunk.processed_tokens == 0 ||
                    chunk.processed_tokens > nominal) {
                    throw std::logic_error("decision score state Prefill made invalid progress");
                }
                cursor += chunk.processed_tokens;
                text_kv_addresses->commit_frontier(*candidate.address, cursor);
            }
        }

        // Fork the prefilled row/state into the branch rows; row 0 stays with this route.
        candidate.reservation =
            open_branches(*this, *candidate.address, *candidate.state, branch_count, state_length,
                          branch_entitlement, device.stream);
        candidate.reservation_open = true;

        // open_branches deactivated row 0 so the fork can pin and copy the source pages; re-acquire
        // it (with its tail entitlement) so the branch-0 suffix appends on the shared row.
        if (branch_count > 1) {
            text_kv_addresses->activate(*candidate.address, entitlement, 0);
        }

        // Right-padded suffix matrix plus per-row execution views and GDN state slots.
        const std::size_t rows = prepared.branches.size();
        std::uint32_t width = 0;
        for (const DecisionBranch& branch : prepared.branches) {
            width = std::max(width, static_cast<std::uint32_t>(branch.suffix_tokens.size()));
        }
        std::vector<TokenId> suffix_tokens(rows * width);
        std::vector<std::uint32_t> valid_lengths(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            const auto& suffix = prepared.branches[row].suffix_tokens;
            std::copy(suffix.begin(), suffix.end(), suffix_tokens.begin() + row * width);
            valid_lengths[row] = static_cast<std::uint32_t>(suffix.size());
        }
        std::vector<qwen3_5::PagedKVCacheView> kv_rows(rows);
        std::vector<std::int32_t> state_source_slots(rows);
        std::vector<std::int32_t> state_destination_slots(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            kv_rows[row] = decoder->text_kv.execution_view(
                text_kv_addresses->execution_row(candidate.reservation.kv_rows[row]));
            state_source_slots[row] =
                state_store->physical_slot(candidate.reservation.states[row]);
            state_destination_slots[row] = state_source_slots[row];
        }

        work.reset();
        mark_workspace_usage(workspace_plan.decision_score.bytes);
        // The readout buffers live in the persistent layout, not the scratch arena: the branch
        // suffix prefill resets the arena to offset 0 for its chunk intermediates, which would
        // clobber readout tensors allocated here (the last-position gather runs after each
        // branch's prefill, so a clobbered readout_hidden feeds the projection). The persistent
        // readout buffers are planned for the maximum branch capacity; slice to the active rows.
        const Tensor readout_hidden_slice =
            decision_readout_hidden->slice(1, 0, static_cast<std::int32_t>(rows));
        const Tensor readout_logits_slice =
            decision_readout_logits->slice(1, 0, static_cast<std::int32_t>(rows));
        Tensor readout_hidden = readout_hidden_slice;
        Tensor readout_logits = readout_logits_slice;
        execution::PrefillContext batch_state{
            {device, parameters, work, state_images->linear(), nullptr, io, prefill_hidden,
             prefill_chunk, proposal_head},
            kv_rows.front(),
            {},
            decoder->text_kv,
            nullptr,
            nullptr,
            0,
            nullptr,
            nullptr,
            0,
            0,
            0,
            nullptr};
        const std::span<const TokenId> state_span =
            media != nullptr ? std::span<const TokenId>(media->token_ids)
                             : std::span<const TokenId>(prepared.state_tokens);
        (void)execution::prefill_decision_batch(
            batch_state, state_span,
            std::span<const TokenId>(suffix_tokens), width,
            std::span<const std::uint32_t>(valid_lengths),
            std::span<const qwen3_5::PagedKVCacheView>(kv_rows),
            std::span<const std::int32_t>(state_source_slots),
            std::span<const std::int32_t>(state_destination_slots), state_length,
            &readout_hidden);
        execution::project(readout_hidden, parameters.text.output_head, readout_logits, work,
                           device.stream);
        // The projection is BF16 and no gather-cast op exists, so pull the readout rows and
        // build each row's FP32 candidate slice on the host before the op call.
        CUDA_CHECK(cudaMemcpyAsync(decision_readout_logits_host->data(), readout_logits.data,
                                   readout_logits.bytes(), cudaMemcpyDeviceToHost, device.stream));
        device.synchronize();
        const auto* readout =
            static_cast<const std::uint16_t*>(decision_readout_logits_host->data());
        const auto normalized_gini = [](const float* probs, std::size_t count) {
            if (count == 1) { return 1.0f; }
            double squares = 0.0;
            for (std::size_t column = 0; column < count; ++column) {
                squares += static_cast<double>(probs[column]) * probs[column];
            }
            const double gini =
                (static_cast<double>(count) * squares - 1.0) / static_cast<double>(count - 1);
            return static_cast<float>(std::clamp(gini, 0.0, 1.0));
        };

        const std::size_t vocab_size =
            static_cast<std::size_t>(dimension(parameters.model.config().text.vocab_size));
        result.answers.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            const DecisionBranch& branch = prepared.branches[row];
            const std::size_t candidates = branch.candidate_ids.size();
            // readout is the [vocab_size, rows] BF16 projection with vocab (ne[0]) fastest, so a
            // candidate's logit sits at row * vocab_size + its vocab id — the same gather
            // target_logprobs performs; the candidate index is NOT a vocab position (see
            // decision_readout.h).
            const std::vector<float> slice = gather_decision_candidate_logits(readout, vocab_size,
                                                                              row, branch.candidate_ids);
            // One op call per row (N=1, C = that row's candidate count): candidate counts differ
            // per question type, so no padding may enter any softmax row.
            work.reset();
            Tensor slice_logits =
                work.alloc(DType::FP32, {1, static_cast<std::int32_t>(candidates)});
            Tensor candidate_ids =
                work.alloc(DType::I32, {1, static_cast<std::int32_t>(candidates)});
            Tensor candidate_groups =
                work.alloc(DType::I32, {1, static_cast<std::int32_t>(candidates)});
            Tensor probabilities =
                work.alloc(DType::FP32, {1, static_cast<std::int32_t>(candidates)});
            CUDA_CHECK(cudaMemcpyAsync(candidate_ids.data, branch.candidate_ids.data(),
                                       candidates * sizeof(TokenId), cudaMemcpyHostToDevice,
                                       device.stream));
            CUDA_CHECK(cudaMemcpyAsync(candidate_groups.data, branch.candidate_groups.data(),
                                       candidates * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            CUDA_CHECK(cudaMemcpyAsync(slice_logits.data, slice.data(),
                                       candidates * sizeof(float), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::candidate_slice_softmax(slice_logits, candidate_ids, candidate_groups, temperature,
                                         probabilities, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(decision_probabilities_host->data(), probabilities.data,
                                       probabilities.bytes(), cudaMemcpyDeviceToHost,
                                       device.stream));
            device.synchronize();
            const float* probs =
                static_cast<const float*>(decision_probabilities_host->data());

            DecisionAnswer answer;
            answer.type      = branch.type;
            answer.options   = branch.options;
            answer.probabilities.assign(probs, probs + candidates);
            answer.raw_logits = slice; // pre-softmax readout logits (the options.raw_logits diagnostic)
            if (branch.type == DecisionQuestionType::Noul) {
                answer.noul           = probs[1];
                answer.winning_option = (probs[1] >= probs[0]) ? "true" : "false";
            } else if (branch.type == DecisionQuestionType::Choice) {
                std::size_t winner = 0;
                for (std::size_t column = 1; column < candidates; ++column) {
                    if (probs[column] > probs[winner]) { winner = column; }
                }
                answer.winning_option = branch.options[winner];
                answer.confidence     = normalized_gini(probs, candidates);
            } else {
                float expected = 0.0f;
                for (std::size_t column = 0; column < candidates; ++column) {
                    expected += static_cast<float>(column) * probs[column];
                }
                answer.score      = expected;
                answer.confidence = normalized_gini(probs, candidates);
            }
            result.answers.push_back(std::move(answer));
        }
        cleanup();
        decision_candidate_.reset();
        return result;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

void ProgramImpl::finalize_decision_transaction() noexcept {
    if (!decision_candidate_) {
        return;
    }
    DecisionAdmissionCandidateImpl& candidate = *decision_candidate_;
    if (candidate.reservation_open) {
        try {
            release_branches(*this, std::move(candidate.reservation));
            candidate.reservation_open = false;
        } catch (...) {}
    }
    if (candidate.address) {
        try {
            if (text_kv_addresses->active(*candidate.address)) {
                text_kv_addresses->deactivate(*candidate.address);
            }
            (void)text_kv_addresses->release(*candidate.address);
        } catch (...) {}
        candidate.address.reset();
    }
    if (candidate.state) {
        try {
            (void)state_store->release(*candidate.state);
        } catch (...) {}
        candidate.state.reset();
    }
    decision_candidate_.reset();
}

bool ProgramImpl::has_decision_transaction() const noexcept {
    return decision_candidate_ && decision_candidate_->transaction_open;
}

DecisionResult ProgramImpl::decision_score(DecisionPrepared prepared, float temperature) {
    auto candidate = inspect_decision_admission(std::move(prepared));
    if (!candidate) {
        throw std::runtime_error("decision admission infeasible (state/KV capacity)");
    }
    if (start_decision_transaction(std::move(*candidate), runtime::CancellationFlagView{}) !=
        runtime::ContextTransactionReserveStatus::Reserved) {
        throw std::runtime_error("decision reservation aborted");
    }
    return progress_decision_transaction(temperature, runtime::CancellationFlagView{});
}

void ProgramImpl::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImpl::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

runtime::ContextTransferObservation ProgramImpl::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    out.device          = device.device;
    out.max_context     = capacity;
    out.kv_capacity     = kv_capacity;
    out.kv_cache        = kv_storage;
    const auto& weights = parameters.model.storage_stats();
    out.weights = ArenaMemorySummary{weights.device_capacity_bytes, weights.device_capacity_bytes,
                                     weights.device_capacity_bytes};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    std::size_t active_workspace_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        active_workspace_bytes =
            std::max(active_workspace_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (workspace_plan.vision) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = workspace_plan.vision->max_merged_tokens,
            .general_capacity_bytes = workspace_plan.vision->general_capacity_bytes,
            .encode_peak_bytes      = workspace_plan.vision->encode_peak_bytes,
            .handoff_offset_bytes   = workspace_plan.vision->handoff_offset_bytes,
            .handoff_capacity_bytes = workspace_plan.vision->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
        };
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_measured_bytes    = graph_measured_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    if (host_state_images) {
        out.host_state_capacity_slots = host_state_images->capacity();
        out.host_state_occupied_slots = host_state_images->occupied();
    }
    if (host_kv_arena) {
        out.host_kv_capacity_bytes = host_kv_arena->capacity_bytes();
        out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes();
    }
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes    = active_handoff_bytes;
    workspace_logical_peak_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes =
            std::max(workspace_logical_peak_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
}


} // namespace ninfer::models::qwen3_5::detail
