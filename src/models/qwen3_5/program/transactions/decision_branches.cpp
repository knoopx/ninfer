#include "models/qwen3_5/program/transactions/decision_branches.h"

#include <cstddef>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

DecisionBranchReservation
open_branches(StateImageStore& state_store, qwen3_5::StateImageDevicePool& state_images,
              KVAddressSpaceStore& kv_addresses, KVAddressSpaceHandle source_row,
              StateImageHandle source_state, std::uint32_t branch_count, std::uint32_t frontier,
              std::uint32_t entitlement, std::uint32_t row_base, cudaStream_t stream) {
    if (branch_count < 1) {
        throw std::invalid_argument("decision branches require at least one row");
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
                (void)kv_addresses.release(reservation.kv_rows[index]);
            }
            if (reservation.states[index].valid()) {
                (void)state_store.release(reservation.states[index]);
            }
        }
    };

    const std::int32_t source_slot = state_store.physical_slot(source_state);
    try {
        for (std::uint32_t index = 1; index < branch_count; ++index) {
            auto forked_state = state_store.reserve_reset(stream);
            if (!forked_state) {
                throw std::runtime_error(
                    "decision: fork state reserve_reset exhausted (branch=" +
                    std::to_string(index) + " state capacity=" +
                    std::to_string(state_store.capacity()) + " occupied=" +
                    std::to_string(state_store.occupied()) + " device_capacity=" +
                    std::to_string(state_store.device_capacity()) +
                    " device_occupied=" + std::to_string(state_store.device_occupied()) + ")");
            }
            const std::int32_t forked_slot = state_store.physical_slot(*forked_state);
            // Full-slot copy across every layer: the hybrid GDN recurrent state must be
            // replicated per branch, not masked.
            state_images.linear().copy_slot(source_slot, forked_slot, stream);
            state_images.copy_slot(source_slot, forked_slot, stream);
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
            auto forked_row = kv_addresses.create_inactive();
            if (!forked_row) {
                throw std::runtime_error(
                    "decision: fork KV create_inactive exhausted (branch=" +
                    std::to_string(index) + " kv capacity=" +
                    std::to_string(kv_addresses.capacity()) + " occupied=" +
                    std::to_string(kv_addresses.occupied()) + ")");
            }
            reservation.kv_rows[index] = *forked_row;
            forks.push_back(std::move(*forked_row));
        }
        if (kv_addresses.active(source_row)) {
            kv_addresses.deactivate(source_row);
        }
        kv_addresses.prepare_branch_fork_batch(source_row, forks, frontier, entitlement, row_base,
                                               stream);
    } catch (...) {
        rollback();
        throw;
    }
    return reservation;
}

void release_branches(StateImageStore& state_store, KVAddressSpaceStore& kv_addresses,
                      DecisionBranchReservation&& reservation) {
    bool released = true;
    for (std::size_t index = 1; index < reservation.kv_rows.size(); ++index) {
        const KVAddressSpaceHandle row = reservation.kv_rows[index];
        if (row.valid()) {
            if (kv_addresses.active(row)) {
                kv_addresses.deactivate(row);
            }
            released = kv_addresses.release(row) && released;
        }
    }
    for (std::size_t index = 1; index < reservation.states.size(); ++index) {
        const StateImageHandle state = reservation.states[index];
        if (state.valid()) {
            released = state_store.release(state) && released;
        }
    }
    if (!released) {
        throw std::logic_error("decision branch resources could not be released");
    }
}

} // namespace ninfer::models::qwen3_5::detail
