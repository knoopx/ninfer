#pragma once

// Ternary PQ2_0 gate/up geometry and schedules for LinearSwiGLU (P3b). Reuses the P3a
// shared geometry/schedule templates of the linear ternary route; the op-local pieces are
// the gate/up pair schedule aliases and the intermediate (output) row count.

#include "ops/linear/ternary/ternary_config.h"

namespace ninfer::ops::detail {

// Gate/up parent geometry: weight [34816,5120], gate rows [0,17408), up rows [17408,34816),
// output [17408,T]. Reuses the P3a N34816K5120 geometry (kOutputRows = 34816 = gate_up rows).
using TernarySwiGluGeometry = TernaryN34816K5120;
constexpr std::int32_t kTernarySwiGluIntermediate = TernarySwiGluGeometry::kOutputRows / 2;

// All three T ranges share the P3a N34816K5120 schedules (until P4 tuning).
using TernarySwiGluGemv = TernaryGemvSchedule<2, 4, 2>;
using TernarySwiGluSimt = TernarySimtSchedule<16, 2, 1>;
using TernarySwiGluMma  = TernaryMmaSchedule<256, 8, 1>;

} // namespace ninfer::ops::detail
