#pragma once
#include "ops/linear/ternary/ternary_launch.h"

namespace ninfer::ops::detail {
struct TernaryLinearShape {
    std::int32_t n, k;
    // T-dependent route selection: the shape's A16 entry resolves the token count to a launcher
    // (GEMV / chunked SIMT / MMA), so it is a select function, not a single launcher.
    TernaryLaunch (*a16)(std::int32_t tokens);
};

extern const TernaryLinearShape kTernaryN14336K5120;
extern const TernaryLinearShape kTernaryN16384K5120;
extern const TernaryLinearShape kTernaryN34816K5120;
extern const TernaryLinearShape kTernaryN248320K5120;
extern const TernaryLinearShape kTernaryN5120K6144;
extern const TernaryLinearShape kTernaryN5120K17408;
} // namespace ninfer::ops::detail
