#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
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

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      kv_storage(plan.kv_storage), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_cuda_graph(plan.use_cuda_graph),
      causal_scoring(plan.causal_scoring), decision_scoring(plan.decision_scoring),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      round_host(plan.causal_scoring
                     ? std::nullopt
                     : std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::PrefillRoundHost))),
      score_logprobs_host(plan.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                    kCausalScoreTile * sizeof(float))
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
                                                             sizeof(qwen3_5::DFlashDecodeEgress) +
                                                             sizeof(qwen3_5::DFlashPrefillIngress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)},
      prefill_gpu_timer_(device_in) {
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder      = std::make_unique<qwen3_5::DecoderState>(backing, plan.persistent.decoder);
    state_images = std::make_unique<StateImageDevicePool>(backing, plan.persistent.state_images);
    // The decision path binds execution rows above the Generation lanes' rows, so a decision job
    // that runs between generation rounds never collides with an in-flight request's row (the
    // lanes bind themselves to rows 0..max_concurrency-1 for a request's whole lifetime). The plan
    // sizes the table as lanes + D + 1, so the decision's rows are its trailing block.
    if (plan.decision_scoring) {
        const std::uint32_t table_rows =
            static_cast<std::uint32_t>(decoder->text_kv.execution_tables().row_count());
        const std::uint32_t decision_rows = plan.persistent.decision_rows + 1U;
        if (table_rows < decision_rows) {
            throw std::logic_error("Qwen3.5 decision execution rows exceed the planned KV table");
        }
        decision_row_base_ = table_rows - decision_rows;
    }
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    if (plan.persistent.dflash) {
        auto* local = state_images->dflash_local();
        if (!local) { throw std::logic_error("DFlash StateImage has no local state"); }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    const auto host_bytes = plan.context_cache.host_capacity_bytes.value();
    std::vector<HostKVPageLayout> layouts{
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry())};
    text_host_kv_page_stride = layouts.front().page_stride;
    if (const auto* backend = backend_kv_cache()) {
        auto layout                 = plan_host_kv_page_layout(backend->page_pool().geometry());
        backend_host_kv_page_stride = layout.page_stride;
        if (layout != layouts.front()) { layouts.push_back(std::move(layout)); }
    }
    std::size_t minimum_stride    = state_images->host_layout().image_bytes;
    std::size_t minimum_kv_stride = layouts.front().page_stride;
    for (const auto& layout : layouts) {
        minimum_stride    = std::min(minimum_stride, layout.page_stride);
        minimum_kv_stride = std::min(minimum_kv_stride, layout.page_stride);
    }
    const auto checked_count = [](std::uint64_t value) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("context resource descriptor capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(value);
    };
    const auto logical_states = checked_count(state_images->slot_count() +
                                              host_bytes / state_images->host_layout().image_bytes);
    // One immutable image can serve private replay, an explicit anchor, a public view and an
    // exact-hit endpoint. Alias descriptors are bounded separately from physical state slots.
    checkpoints.resize(checked_count(4ULL * logical_states + 2ULL * max_concurrency));
    // The decision path's reserved row 0 plus D branch rows are counted on top, so a decision
    // job always has room for them and Generation traffic cannot wedge the decision region.
    const auto address_capacity = checked_count(
        checkpoints.size() + max_concurrency + 2U +
        (plan.decision_scoring ? static_cast<std::uint64_t>(plan.persistent.decision_rows) + 1U
                               : 0U));
    if (host_bytes) {
        StartupPhaseScope phase(startup_observer, StartupPhase::HostContextPin,
                                StartupProgressUnit::Bytes, host_bytes);
        host_context_arena = std::make_unique<HostContextArena>(host_bytes, minimum_stride);
        host_state_images =
            std::make_unique<HostStatePool>(*host_context_arena, state_images->host_layout());
        host_kv_arena      = std::make_unique<HostKVArena>(*host_context_arena, layouts);
        const auto extents = checked_count(host_bytes / minimum_kv_stride);
        if (extents) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(*host_kv_arena, extents);
        }
        phase.complete(host_bytes, host_bytes);
    }
    state_store =
        std::make_unique<StateImageStore>(*state_images, host_state_images.get(), logical_states);
    const auto logical_pages = [&](const DeviceKVPagePool& pool) {
        const auto stride = plan_host_kv_page_layout(pool.geometry()).page_stride;
        return checked_count(pool.capacity_pages() + host_bytes / stride);
    };
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_pages(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    if (auto* backend = backend_kv_cache()) {
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_pages(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
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
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    if (plan.persistent.grammar_masks) {
        grammar_masks_device = plan.persistent.grammar_masks->bind(backing);
        grammar_masks_host.emplace(grammar_masks_device.bytes());
    }
    if (is_masked_draft_backend(speculative_backend)) {
        dflash_draft_handoff.emplace(device, draft_window * max_concurrency);
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        lane_epochs[lane]    = 1;
        sequences[lane].lane = lane;
    }
    host_tokens = round_host
                      ? &static_cast<qwen3_5::PrefillRoundHost*>(round_host->data())->sampled_token
                      : nullptr;
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
        *dflash_host_ingress        = {};
        *dflash_host_egress         = {};
        dflash_prefill_host_ingress = reinterpret_cast<qwen3_5::DFlashPrefillIngress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress) + sizeof(qwen3_5::DFlashDecodeEgress));
        *dflash_prefill_host_ingress = {};
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
        prepare_graphs();
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
    const std::uint32_t required_pages         = kv_pages_for_frontier(predictor_count);
    if (required_pages == 0) { throw std::logic_error("causal score requires KV pages"); }

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
        address = text_kv_addresses->create_active(required_pages, 0, device.stream);
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
    if (!decision_scoring) {
        return nullptr;
    }
    // The decision path is self-contained (temp state + Main KV row 0 + branch fork + batched
    // suffix prefill + readout) and is serialized by the Generation core against in-flight
    // generation rounds, so it runs on the model's loaded (Generation-purpose) Program.
    if (prepared.branches.empty()) {
        throw std::invalid_argument("decision scoring requires at least one branch");
    }
    const std::uint32_t branch_count = static_cast<std::uint32_t>(prepared.branches.size());
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
    // exhausts the device page pool when several branches fork at once. A decision whose state +
    // suffix exceeds the planned tail cap is rejected (nullptr), never capped below the need:
    // capping makes the branch suffix prefill write past the row's page capacity (an OOB KV write
    // that corrupts the heap). The state alone always fits (state_pages <= the cap is implied by
    // the rejection below, since suffix pages >= 1).
    const std::uint32_t branch_entitlement =
        state_pages + kv_pages_for_frontier(max_suffix_length);
    if (branch_entitlement > workspace_plan.decision_score.branch_tail_pages) {
        return nullptr; // the branch tail exceeds the planned decision KV capacity
    }
    const std::uint32_t entitlement = branch_entitlement;

    // Feasibility check against the reserved decision region: the plan holds row 0 plus one
    // branch row in the Device StateImage pool, in the KV address space and in the decision's own
    // execution rows, so a shortfall means Generation traffic overran its configured axes. The
    // runtime still forks in rounds sized from the live free counts, so a larger free region only
    // widens a round. Physical KV page demand is checked against the shared pool's live free
    // pages (row 0 plus one branch here, per round at sizing); create_active enforces it per
    // address at reservation.
    const std::uint32_t free_device_slots =
        state_store->device_capacity() - state_store->device_occupied();
    const std::uint32_t free_addresses =
        text_kv_addresses->capacity() - text_kv_addresses->occupied();
    const std::uint32_t free_decision_rows =
        static_cast<std::uint32_t>(decoder->text_kv.execution_tables().row_count()) -
        decision_row_base_;
    // Page axis: row 0 takes `entitlement` pages at start; each forked branch reserves
    // `entitlement - full_pages + (partial tail ? 1 : 0)` new physical pages (the exact
    // prepare_prefix_fork demand open_branches makes). The reserved decision region guarantees
    // the state/address/row axes, not the page axis, so Generation traffic can leave the shared
    // pool short: without this check the job passes admission, opens row 0, and the first fork
    // throws a bare std::bad_alloc from the page pool instead of a capacity rejection.
    const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t pages_per_branch =
        entitlement - state_length / page_size + (state_length % page_size != 0 ? 1U : 0U);
    const std::uint32_t free_pages = text_kv_pages->physical_pool().available_pages();
    if (free_device_slots < 2 || free_addresses < 2 || free_decision_rows < 2 ||
        free_pages < entitlement + pages_per_branch) {
        // The plan reserves a decision region, so a shortfall means Generation traffic overran its
        // own capacity axes (or the plan is inconsistent). Report the exact axes instead of a
        // bare infeasible verdict: this block used to wedge every later decision request with no
        // way to tell which resource had run out.
        throw std::logic_error(
            "decision scoring has no capacity for a branch fork (free device state slots=" +
            std::to_string(free_device_slots) + " free KV addresses=" +
            std::to_string(free_addresses) + " free decision rows=" +
            std::to_string(free_decision_rows) + " free KV pages=" + std::to_string(free_pages) +
            " required pages for row 0 plus one branch=" +
            std::to_string(entitlement + pages_per_branch) + ")");
    }

    auto candidate = std::make_unique<DecisionAdmissionCandidateImpl>();
    candidate->prepared     = std::move(prepared);
    candidate->branch_count = branch_count;
    candidate->state_length = state_length;
    candidate->entitlement  = entitlement;
    candidate->input_tokens = input_tokens;
    return candidate;
}

[[nodiscard]] bool ProgramImpl::start_decision_transaction(DecisionAdmissionCandidateImpl&& candidate,
                                                           runtime::CancellationFlagView cancellation) {
    if (decision_candidate_.has_value()) {
        throw std::logic_error("decision transaction already open");
    }
    if (cancellation.requested()) { return false; }
    try {
        candidate.state = state_store->reserve_reset(device.stream);
        // The dedicated decision store guarantees D state slots and at most one decision job
        // is in flight, so exhaustion here is an invariant failure, not admission.
        if (!candidate.state) {
            throw std::logic_error("decision state reservation failed (dedicated capacity)");
        }
        candidate.address = text_kv_addresses->create_active(
            candidate.entitlement, static_cast<std::int32_t>(decision_row_base_), device.stream);
        if (!candidate.address) {
            (void)state_store->release(*candidate.state);
            candidate.state.reset();
            // create_active fails only when the per-branch entitlement exceeds the dedicated
            // page capacity: a genuinely impossible input, not Generation occupancy.
            throw std::logic_error(
                "decision branch entitlement exceeds the dedicated KV page capacity");
        }
        if (text_kv_addresses->bound_row(*candidate.address) !=
            static_cast<std::int32_t>(decision_row_base_)) {
            throw std::logic_error("decision transaction did not bind the decision KV row");
        }
        candidate.transaction_open = true;
        decision_candidate_ = std::move(candidate);
        return true;
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
    const std::uint32_t branch_entitlement    = candidate.entitlement;
    // Release forked rows/states first, then the caller-owned prefilled row 0 / state 0
    // (deactivated by open_branches), mirroring the causal_score cleanup.
    const auto cleanup = [&] {
        if (candidate.reservation_open) {
            release_branches(*state_store, *text_kv_addresses,
                             std::move(candidate.reservation));
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
                    .control_index       = static_cast<std::uint32_t>(index),
                });
                vision_plan.max_merged_count =
                    std::max(vision_plan.max_merged_count, item.merged_count);
            }
            vision_plan.control = std::make_shared<const qwen3_5::VisionControl>(
                qwen3_5::build_vision_control(*media, *vision_plan.control_plan, 0));
            auto vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, *media, vision_plan, vision_handoff,
                vision_handoff_peak_bytes);
            while (cursor < state_length) {
                const std::uint32_t nominal = std::min(prefill_chunk, state_length - cursor);
                execution::PrefillContext schedule_state{
                    {device, parameters, work, state_images->linear(), nullptr, io,
                     prefill_hidden, prefill_chunk, proposal_head},
                    decoder->text_kv.execution_view(
                        decoder->text_kv.execution_tables(), text_kv_addresses->execution_row(*candidate.address)),
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
                    0,
                    nullptr};
                mark_workspace_usage(workspace_plan.vision->capacity_bytes);
                const auto chunk = execution::prefill_multimodal_chunk(
                    schedule_state, *media, *vision, nominal, std::nullopt,
                    cursor + nominal == state_length);
                vision->retire_handoff();
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
                        decoder->text_kv.execution_tables(), text_kv_addresses->execution_row(*candidate.address)),
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

        // The decision path re-uses Generation's working set, so one fork round can only hold as
        // many branch rows as the state-image slots, the KV addresses and the execution-table
        // rows allow next to the caller-owned row 0. A job that names more branches runs as
        // successive fork/prefill/readout/release rounds: peak residency stays at one round, so
        // the branch count is bounded by the job, not by the plan.
        // A fork copies the GDN state into a Device slot (reserve_reset + copy_slot), so the
        // round is bounded by the free Device slots, not by the logical capacity (which adds the
        // host replicas a fork never claims).
        const std::uint32_t free_state =
            state_store->device_capacity() - state_store->device_occupied();
        const std::uint32_t free_kv =
            text_kv_addresses->capacity() - text_kv_addresses->occupied();
        // The fork binds destination i to decision_row_base_ + i + 1, so the rows above the
        // decision's own row 0 are the ceiling; count them from the table so a misplanned table
        // fails here instead of inside the fork.
        const std::uint32_t table_rows =
            static_cast<std::uint32_t>(decoder->text_kv.execution_tables().row_count());
        const std::uint32_t free_decision_rows =
            table_rows > decision_row_base_ ? table_rows - decision_row_base_ - 1U : 0U;
        // Page axis: each forked branch reserves `pages_per_branch` new physical pages (the
        // open_branches prepare_prefix_fork demand: growth pages plus the partial tail copy);
        // row 0's `branch_entitlement` pages are already reserved. The shared pool's live free
        // pages bound how many branches fit in a round, so size the round from them or the fork's
        // resize_reservation throws a bare std::bad_alloc under Generation page pressure.
        // pages_per_branch is always >= 1 (branch suffixes are non-empty, so the entitlement
        // exceeds the full prefix pages), making the division well-defined.
        const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t pages_per_branch =
            branch_entitlement - state_length / page_size +
            (state_length % page_size != 0 ? 1U : 0U);
        const std::uint32_t free_pages = text_kv_pages->physical_pool().available_pages();
        const std::uint32_t free_by_pages = free_pages / pages_per_branch;
        const std::uint32_t round_capacity =
            std::min({free_state, free_kv, free_decision_rows, free_by_pages});
        if (round_capacity == 0) {
            throw std::logic_error(
                "decision branch fork has no capacity (free device state slots=" +
                std::to_string(free_state) + " free KV addresses=" + std::to_string(free_kv) +
                " free decision rows=" + std::to_string(free_decision_rows) +
                " free KV pages=" + std::to_string(free_pages) + " pages per branch=" +
                std::to_string(pages_per_branch) + ")");
        }
        const std::uint32_t round_rows = std::min(round_capacity, branch_count);

        // The readout buffers are allocated per job (sized to one round, the widest batch any
        // single prefill or projection sees) outside the scratch arena: the branch suffix prefill
        // resets the arena to offset 0 for its chunk intermediates, which would clobber readout
        // tensors allocated in the arena (the last-position gather runs after each row's prefill,
        // so a clobbered readout_hidden feeds the projection). The per-job allocation sits outside
        // the arena, so the clobber problem is gone. DeviceBuffer/PinnedHostBuffer are RAII and
        // freed on scope exit.
        const std::size_t hidden = dimension(parameters.model.config().text.hidden_size);
        const std::size_t vocab  = dimension(parameters.model.config().text.vocab_size);
        const std::size_t vocab_size = static_cast<std::size_t>(vocab);
        DeviceBuffer readout_hidden_storage(hidden * round_rows * 2);
        DeviceBuffer readout_logits_storage(vocab * round_rows * 2);
        const std::size_t project_scratch_bytes = ops::linear_workspace_capacity_bytes(
            parameters.text.output_head.weight.qtype, parameters.text.output_head.weight.n,
            parameters.text.output_head.weight.k, parameters.text.output_head.policy, 1,
            static_cast<std::int32_t>(round_rows));
        // Zero-capacity qtypes (BF16, Q4..Q8, FP32, INT32) need no linear scratch, but the
        // borrowed DeviceArena asserts non-empty storage; the 1-byte floor is never touched.
        DeviceBuffer project_scratch_storage(std::max<std::size_t>(project_scratch_bytes, 1));
        DeviceArena project_workspace{DeviceSpan{project_scratch_storage.p,
                                                  project_scratch_storage.bytes}};
        PinnedHostBuffer readout_logits_host(vocab * round_rows * 2);
        PinnedHostBuffer probabilities_host(kDecisionMaxCandidates * sizeof(float));
        const std::span<const TokenId> state_span =
            media != nullptr ? std::span<const TokenId>(media->token_ids)
                             : std::span<const TokenId>(prepared.state_tokens);
        result.answers.reserve(branch_count);
        for (std::uint32_t begin = 0; begin < branch_count; begin += round_rows) {
            const std::uint32_t count = std::min(round_rows, branch_count - begin);
            // Forking count + 1 rows keeps the prefilled row 0 as the source instead of a branch
            // row: open_branches binds destination i above decision_row_base_ and leaves the
            // source inactive with its page membership and GDN state intact, so every round forks
            // the same prefix. The outer catch's cleanup releases the round if the body throws.
            candidate.reservation = open_branches(
                *state_store, *state_images, *text_kv_addresses, *candidate.address,
                *candidate.state, count + 1U, state_length, branch_entitlement,
                decision_row_base_, device.stream);
            candidate.reservation_open = true;
            // The fork publishes only the shared prefix pages (state_length tokens): each branch
            // row's growth pages are reserved but unmapped, so extend the row's block table
            // through its suffix before the suffix prefill writes KV there, or the writes land
            // in unpublished block-table slots (an illegal device write).
            for (std::uint32_t index = 0; index < count; ++index) {
                text_kv_addresses->ensure_mapped_to_tokens(
                    candidate.reservation.kv_rows[index + 1U],
                    state_length + prepared.branches[begin + index].suffix_tokens.size(),
                    device.stream);
            }

            // Right-padded suffix matrix plus per-row execution views and GDN state slots.
            // reservation.kv_rows[0] / states[0] are the source row; rows 1..count are branches.
            std::uint32_t width = 0;
            for (std::uint32_t index = 0; index < count; ++index) {
                width = std::max(
                    width, static_cast<std::uint32_t>(
                               prepared.branches[begin + index].suffix_tokens.size()));
            }
            std::vector<TokenId> suffix_tokens(static_cast<std::size_t>(count) * width);
            std::vector<std::uint32_t> valid_lengths(count);
            for (std::uint32_t index = 0; index < count; ++index) {
                const auto& suffix = prepared.branches[begin + index].suffix_tokens;
                std::copy(suffix.begin(), suffix.end(),
                          suffix_tokens.begin() + static_cast<std::size_t>(index) * width);
                valid_lengths[index] = static_cast<std::uint32_t>(suffix.size());
            }
            std::vector<qwen3_5::PagedKVCacheView> kv_rows(count);
            std::vector<std::int32_t> state_source_slots(count);
            std::vector<std::int32_t> state_destination_slots(count);
            for (std::uint32_t index = 0; index < count; ++index) {
                kv_rows[index] = decoder->text_kv.execution_view(
                    decoder->text_kv.execution_tables(),
                    text_kv_addresses->execution_row(candidate.reservation.kv_rows[index + 1U]));
                state_source_slots[index] =
                    state_store->physical_slot(candidate.reservation.states[index + 1U]);
                state_destination_slots[index] = state_source_slots[index];
            }

            work.reset();
            // The round re-slices the per-job storage to its live row count: prefill_decision_batch
            // and project size their work from the tensor shape, not from the allocation.
            Tensor readout_hidden =
                Tensor(readout_hidden_storage.p, DType::BF16,
                       {static_cast<std::int32_t>(hidden), static_cast<std::int32_t>(count)});
            Tensor readout_logits =
                Tensor(readout_logits_storage.p, DType::BF16,
                       {static_cast<std::int32_t>(vocab), static_cast<std::int32_t>(count)});
            execution::PrefillContext batch_state{
                {device, parameters, work, state_images->linear(), nullptr, io,
                 prefill_hidden, prefill_chunk, proposal_head},
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
                0,
                nullptr};
            (void)execution::prefill_decision_batch(
                batch_state, state_span,
                std::span<const TokenId>(suffix_tokens), width,
                std::span<const std::uint32_t>(valid_lengths),
                std::span<const qwen3_5::PagedKVCacheView>(kv_rows),
                std::span<const std::int32_t>(state_source_slots),
                std::span<const std::int32_t>(state_destination_slots), state_length,
                &readout_hidden);
            execution::project(readout_hidden, parameters.text.output_head, readout_logits,
                               project_workspace, device.stream);
            // The projection is BF16 and no gather-cast op exists, so pull the readout rows and
            // build each row's FP32 candidate slice on the host before the op call.
            CUDA_CHECK(cudaMemcpyAsync(readout_logits_host.data(), readout_logits.data,
                                       readout_logits.bytes(), cudaMemcpyDeviceToHost,
                                       device.stream));
            device.synchronize();
            const auto* readout =
                static_cast<const std::uint16_t*>(readout_logits_host.data());
            for (std::uint32_t index = 0; index < count; ++index) {
                const DecisionBranch& branch = prepared.branches[begin + index];
                const std::size_t candidates = branch.candidate_ids.size();
                // readout is the [vocab_size, count] BF16 projection with vocab (ne[0]) fastest,
                // so a candidate's logit sits at row * vocab_size + its vocab id — the same
                // gather target_logprobs performs; the candidate index is NOT a vocab position
                // (see decision_readout.h).
                const std::vector<float> slice = gather_decision_candidate_logits(
                    readout, vocab_size, index, branch.candidate_ids);
                // One op call per row (N=1, C = that row's candidate count): candidate counts
                // differ per question type, so no padding may enter any softmax row.
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
                                           candidates * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, device.stream));
                CUDA_CHECK(cudaMemcpyAsync(slice_logits.data, slice.data(),
                                           candidates * sizeof(float), cudaMemcpyHostToDevice,
                                           device.stream));
                ops::candidate_slice_softmax(slice_logits, candidate_ids, candidate_groups,
                                             temperature, probabilities, device.stream);
                CUDA_CHECK(cudaMemcpyAsync(probabilities_host.data(), probabilities.data,
                                           probabilities.bytes(), cudaMemcpyDeviceToHost,
                                           device.stream));
                device.synchronize();
                const float* probs =
                    static_cast<const float*>(probabilities_host.data());

                DecisionAnswer answer;
                answer.type      = branch.type;
                answer.options   = branch.options;
                answer.probabilities.assign(probs, probs + candidates);
                answer.raw_logits = slice; // pre-softmax readout logits (raw_logits diagnostic)

                if (branch.type == DecisionQuestionType::Noul) {
                    answer.noul           = probs[1];
                    answer.winning_option = (probs[1] >= probs[0]) ? "true" : "false";
                } else if (branch.type == DecisionQuestionType::Choice) {
                    std::size_t winner = 0;
                    for (std::size_t column = 1; column < candidates; ++column) {
                        if (probs[column] > probs[winner]) { winner = column; }
                    }
                    answer.winning_option = branch.options[winner];
                    // Top probability: the abstention signal the reference measures.
                    answer.confidence = probs[winner];
                } else {
                    double expected = 0.0;
                    for (std::size_t column = 0; column < candidates; ++column) {
                        expected +=
                            static_cast<double>(column) * static_cast<double>(probs[column]);
                    }
                    answer.score = static_cast<float>(expected);
                    // Spread, not height: an even split between adjacent levels knows where the
                    // answer is. variance is over the 0-based level indices; half_range is (K-1)/2.
                    double variance = 0.0;
                    for (std::size_t column = 0; column < candidates; ++column) {
                        const double offset = static_cast<double>(column) - expected;
                        variance += static_cast<double>(probs[column]) * offset * offset;
                    }
                    const double half_range = static_cast<double>(candidates - 1) / 2.0;
                    answer.confidence = static_cast<float>(
                        std::clamp(1.0 - std::sqrt(variance) / half_range, 0.0, 1.0));
                }
                result.answers.push_back(std::move(answer));
            }
            release_branches(*state_store, *text_kv_addresses,
                             std::move(candidate.reservation));
            candidate.reservation_open = false;
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
            release_branches(*state_store, *text_kv_addresses,
                             std::move(candidate.reservation));
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

std::uint32_t ProgramImpl::decision_max_branches() const {
    if (!decision_scoring || !state_store || !text_kv_addresses || !decoder) { return 0; }
    // Largest branch count one fork round can hold when the shared working set is otherwise
    // idle (the live counts the runtime actually rounds with): row 0 plus as many state rows,
    // KV addresses and execution-table rows as the plan provides.
    const std::uint32_t execution_rows =
        static_cast<std::uint32_t>(decoder->text_kv.execution_tables().row_count());
    if (execution_rows <= decision_row_base_ + 1U) { return 0; }
    return std::min({state_store->device_capacity(), text_kv_addresses->capacity(),
                     execution_rows - decision_row_base_ - 1U});
}

DecisionResult ProgramImpl::decision_score(DecisionPrepared prepared, float temperature) {
    if (!decision_scoring) {
        throw std::logic_error("decision scoring is disabled for this Program");
    }
    auto candidate = inspect_decision_admission(std::move(prepared));
    if (!candidate) {
        // A null candidate means the shared working set cannot hold the caller-owned row 0 plus
        // one branch row, so no fork round can start.
        throw std::logic_error("decision scoring has no capacity for a branch fork");
    }
    // The fresh empty cancellation flag never requests cancellation, and the dedicated stores
    // make the capacity aborts unreachable, so the reservation is Reserved by construction.
    (void)start_decision_transaction(std::move(*candidate), runtime::CancellationFlagView{});
    // A failed job must release its reservation (mirroring the Generation core's decide): a
    // stuck open transaction pins the decision state/KV rows and wedges every later decision
    // job ("decision transaction already open") until a restart.
    try {
        return progress_decision_transaction(temperature, runtime::CancellationFlagView{});
    } catch (...) {
        finalize_decision_transaction();
        throw;
    }
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
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
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
    out.kv_payload_bytes             = kv_payload_bytes;
    if (host_state_images) { out.host_state_occupied_slots = host_state_images->occupied(); }
    if (host_kv_arena) { out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes(); }
    if (host_context_arena) {
        out.host_context_capacity_bytes = host_context_arena->capacity_bytes();
        out.host_context_occupied_bytes = host_context_arena->occupied_bytes();
        out.host_context_reserved_bytes = host_context_arena->reserved_bytes();
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
        if (request.replay && request.replay->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.replay->vision->active_handoff_bytes());
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
