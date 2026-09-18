#include "ops/linear_swiglu/ternary/ternary_linear_swiglu_plan.h"

#include "ops/linear/ternary/ternary_format.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The registered gate/up problem (mirrors the NVFP4 parent of the op contract).
constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kInputRows  = 5120;

// P3b T routing (until P4 tuning): T=1 uses the GEMV route; 2..32 use the 16-token SIMT route;
// T>=33 uses the MMA route. All resolve to A16 compute for every policy.
enum class TernarySwiGluRoute {
    Gemv,
    Simt,
    Mma,
};

TernarySwiGluRoute resolve_route(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("ternary linear_swiglu: T must be positive"); }
    if (tokens == 1) { return TernarySwiGluRoute::Gemv; }
    if (tokens <= 32) { return TernarySwiGluRoute::Simt; }
    return TernarySwiGluRoute::Mma;
}

} // namespace

std::size_t ternary_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                           std::int32_t min_tokens,
                                                           std::int32_t max_tokens) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("ternary linear_swiglu: unsupported policy");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("ternary linear_swiglu: invalid token interval");
    }
    (void)resolve_route(min_tokens);
    (void)resolve_route(max_tokens);
    // The A16-only ternary route needs no transient storage.
    return 0;
}

void ternary_linear_swiglu_dispatch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                    LinearPolicy policy, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    (void)policy;
    (void)workspace;
    if (x.ne[1] <= 0) { throw std::invalid_argument("ternary linear_swiglu: T must be positive"); }
    if (gate_up_weight.n != kGateUpRows || gate_up_weight.k != kInputRows) {
        throw std::invalid_argument("ternary linear_swiglu: unsupported shape");
    }
    (void)validate_ternary_weight_metadata(gate_up_weight, "ternary linear_swiglu");

    const TernarySwiGluRoute route = resolve_route(x.ne[1]);
    switch (route) {
    case TernarySwiGluRoute::Gemv:
        ternary_linear_swiglu_gemv_launch(x, gate_up_weight, out, stream);
        return;
    case TernarySwiGluRoute::Simt:
        ternary_linear_swiglu_simt_launch(x, gate_up_weight, out, stream);
        return;
    case TernarySwiGluRoute::Mma:
        ternary_linear_swiglu_mma_launch(x, gate_up_weight, out, stream);
        return;
    }
    throw std::logic_error("ternary linear_swiglu: unrouted token count");
}

} // namespace ninfer::ops::detail
