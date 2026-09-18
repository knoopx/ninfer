#pragma once

// Ternary PQ2_0 residual-add geometry and schedules for LinearAdd (P3b). Reuses the P3a shared
// geometry/schedule templates of the linear ternary route; the op-local pieces are the N2048
// geometry aliases and the residual-add epilogue.

#include "ops/linear/ternary/ternary_config.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

// Registered LinearAdd ternary problems (mirror the op's registered parent set):
// [5120,6144], [5120,17408] (P3a geometries) and [2048,4096], [2048,6144] (N2048 aliases).
using TernaryAddN5120K6144  = TernaryN5120K6144;
using TernaryAddN5120K17408 = TernaryN5120K17408;
using TernaryAddN2048K4096  = TernaryGeometry<2048, 4096>;
using TernaryAddN2048K6144  = TernaryGeometry<2048, 6144>;

// All four shapes share these schedules (until P4 tuning).
using TernaryAddGemv = TernaryGemvSchedule<2, 4, 2>;
using TernaryAddSimt = TernarySimtSchedule<16, 2, 1>;
using TernaryAddMma  = TernaryMmaSchedule<256, 8, 1>;

// Fused residual-add epilogue: out = projection + residual (in place on the residual tensor).
struct TernaryAddResidualEpilogue {
    const __nv_bfloat16* residual;
    std::int32_t rows;

    __device__ __forceinline__ float apply(std::int32_t row, std::int32_t token,
                                           float value) const {
        return value + __bfloat162float(
                          residual[static_cast<std::int64_t>(token) * rows + row]);
    }
};

} // namespace ninfer::ops::detail
