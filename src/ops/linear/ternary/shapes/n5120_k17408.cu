#include "ops/linear/ternary/ternary_shapes.h"
#include "ops/linear/ternary/ternary_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = TernaryN5120K17408;
using Gemv = TernaryGemvSchedule<2, 4, 2>;
using Simt = TernarySimtSchedule<16, 2, 1>;
using Mma  = TernaryMmaSchedule<256, 8, 1>;

// P3a T routing (all six shapes share these schedules until P4 tuning): T=1 uses the GEMV
// route; 2..32 use the 16-token SIMT route in 16-token chunks; T>=33 uses the MMA route.
// select_simt_chunk re-selects per chunk: a full 16-token chunk takes the FullColumns path
// (live == Capacity, no runtime compare); the ragged last chunk of 17..32 passes its live
// count through the RuntimeColumns path (x.ne[1] after slicing).
TernaryLaunch select_simt_chunk(std::int32_t tokens) {
    if (tokens >= 16) return launch_ternary_simt<Geometry, 16, Simt, true>;
    return launch_ternary_simt<Geometry, 16, Simt, false>;
}

TernaryLaunch select_ternary_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_ternary_gemv<Geometry, Gemv>;
    if (tokens <= 32) return launch_ternary_a16_chunks<16, select_simt_chunk>;
    return launch_ternary_mma<Geometry, Mma>;
}
} // namespace

const TernaryLinearShape kTernaryN5120K17408{5120, 17408, select_ternary_a16};
} // namespace ninfer::ops::detail
