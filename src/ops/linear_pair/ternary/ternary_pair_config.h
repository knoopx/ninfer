#pragma once

// Ternary PQ2_0 paired K/V geometry and schedules for LinearPair (P3b). Reuses the P3a shared
// geometry/schedule templates of the linear ternary route; the op-local pieces are the N1024
// geometry aliases for the paired projection and the adjacent K/V row views.

#include "ops/linear/ternary/ternary_config.h"

namespace ninfer::ops::detail {

// Registered LinearPair ternary problems (mirror the op's registered parent set): the paired
// [1024,5120] projection (two independent ternary weights, each with its own block plane and
// rotation auxiliary) and the exact adjacent [1024,2048] K/V row views of one [6144,2048]
// parent (first = parent rows [4096,5120), second = parent rows [5120,6144), sharing the
// parent's rotation auxiliary). N=1024 divides the CTA row tiles of every schedule.
using TernaryPairN1024K5120 = TernaryGeometry<1024, 5120>;
using TernaryPairN1024K2048 = TernaryGeometry<1024, 2048>;

// All three T ranges share these schedules (until P4 tuning).
using TernaryPairGemv = TernaryGemvSchedule<2, 4, 2>;
using TernaryPairSimt = TernarySimtSchedule<16, 2, 1>;
using TernaryPairMma  = TernaryMmaSchedule<256, 8, 1>;

} // namespace ninfer::ops::detail
