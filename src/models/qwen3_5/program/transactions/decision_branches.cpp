#include "models/qwen3_5/program/transactions/decision_branches.h"

#include "models/qwen3_5/program/program_impl.h"

#include <cstddef>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

DecisionBranchReservation
open_branches(ProgramImpl& program, KVAddressSpaceHandle source_row, StateImageHandle source_state,
              std::uint32_t branch_count, std::uint32_t frontier, std::uint32_t entitlement,
              cudaStream_t stream) {
    const auto& decision = program.workspace_plan.decision_score;
    if (branch_count < 1) {
        throw std::invalid_argument("decision branches require at least one row");
    }
    if (branch_count > decision.branch_kv_rows) {
        throw std::invalid_argument("decision branches exceed the planned branch KV rows");
    }
    if (entitlement == 0) {
        throw std::invalid_argument("decision branch entitlement must be nonzero");
    }

    DecisionBranchReservation reservation;
    reservation.frontier = frontier;
    reservation.kv_rows.resize(branch_count);
    reservation.states.resize(branch_count);
    reservation.kv_rows[0] = source_row;
    reservation.states[0]  = source_state;

    const auto rollback = [&]() {
        for (std::uint32_t index = 1; index < branch_count; ++index) {
            if (reservation.kv_rows[index].valid()) {
                (void)program.text_kv_addresses->release(reservation.kv_rows[index]);
            }
            if (reservation.states[index].valid()) {
                (void)program.state_store->release(reservation.states[index]);
            }
        }
    };

    const std::int32_t source_slot = program.state_store->physical_slot(source_state);
    try {
        for (std::uint32_t index = 1; index < branch_count; ++index) {
            auto forked_state = program.state_store->reserve_reset(stream);
            if (!forked_state) {
                throw std::runtime_error(
                    "decision: fork state reserve_reset exhausted (branch=" +
                    std::to_string(index) + " state capacity=" +
                    std::to_string(program.state_store->capacity()) + " occupied=" +
                    std::to_string(program.state_store->occupied()) + " device_capacity=" +
                    std::to_string(program.state_store->device_capacity()) +
                    " device_occupied=" +
                    std::to_string(program.state_store->device_occupied()) + ")");
            }
            const std::int32_t forked_slot = program.state_store->physical_slot(*forked_state);
            // Full-slot copy across every layer: the hybrid GDN recurrent state must be
            // replicated per branch, not masked.
            program.state_images->linear().copy_slot(source_slot, forked_slot, stream);
            program.state_images->copy_slot(source_slot, forked_slot, stream);
            reservation.states[index] = std::move(*forked_state);
        }
    } catch (...) {
        rollback();
        throw;
    }

    // A single-branch decision has no fork: the source row stays active with its execution row.
    if (branch_count == 1) { return reservation; }

    std::vector<KVAddressSpaceHandle> forks;
    forks.reserve(branch_count - 1U);
    try {
        for (std::uint32_t index = 1; index < branch_count; ++index) {
            auto forked_row = program.text_kv_addresses->create_inactive();
            if (!forked_row) {
                throw std::runtime_error(
                    "decision: fork KV create_inactive exhausted (branch=" +
                    std::to_string(index) + " kv capacity=" +
                    std::to_string(program.text_kv_addresses->capacity()) + " occupied=" +
                    std::to_string(program.text_kv_addresses->occupied()) + ")");
            }
            reservation.kv_rows[index] = *forked_row;
            forks.push_back(std::move(*forked_row));
        }
        if (program.text_kv_addresses->active(source_row)) {
            program.text_kv_addresses->deactivate(source_row);
        }
        program.text_kv_addresses->prepare_branch_fork_batch(source_row, forks, frontier, entitlement,
                                                             stream);
    } catch (...) {
        rollback();
        throw;
    }
    return reservation;
}

void release_branches(ProgramImpl& program, DecisionBranchReservation&& reservation) {
    bool released = true;
    for (std::size_t index = 1; index < reservation.kv_rows.size(); ++index) {
        const KVAddressSpaceHandle row = reservation.kv_rows[index];
        if (row.valid()) {
            if (program.text_kv_addresses->active(row)) {
                program.text_kv_addresses->deactivate(row);
            }
            released = program.text_kv_addresses->release(row) && released;
        }
    }
    for (std::size_t index = 1; index < reservation.states.size(); ++index) {
        const StateImageHandle state = reservation.states[index];
        if (state.valid()) {
            released = program.state_store->release(state) && released;
        }
    }
    if (!released) {
        throw std::logic_error("decision branch resources could not be released");
    }
}

} // namespace ninfer::models::qwen3_5::detail
