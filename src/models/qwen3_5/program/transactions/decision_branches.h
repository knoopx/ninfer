#pragma once

#include "models/qwen3_5/program/storage/kv_store.h"
#include "models/qwen3_5/program/storage/state_store.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class ProgramImpl;

// One prefilled decision state/KV row forked into N branch rows. Index 0 is the caller's
// prefilled row and state; 1..N-1 are the forked branches.
struct DecisionBranchReservation {
    std::vector<KVAddressSpaceHandle> kv_rows;
    std::vector<StateImageHandle> states;
    std::uint32_t frontier = 0;
};

// Forks the prefilled row/state into branch_count - 1 branch rows: each forked state is a
// full-slot copy of the source state slot, and each forked KV row aliases the prefix pages
// [0, frontier) as reader references while owning its own tail pages. The source row is
// deactivated by the fork; the forked rows come back active. Row 0 / state 0 stay caller-owned.
[[nodiscard]] DecisionBranchReservation
open_branches(ProgramImpl& program, KVAddressSpaceHandle source_row, StateImageHandle source_state,
              std::uint32_t branch_count, std::uint32_t frontier, std::uint32_t entitlement,
              cudaStream_t stream);

// Releases the forked rows (1..N-1, deactivate + release) and forked states, mirroring the
// causal_score cleanup; throws on a partial release. Row 0 / state 0 belong to the caller.
void release_branches(ProgramImpl& program, DecisionBranchReservation&& reservation);

} // namespace ninfer::models::qwen3_5::detail
