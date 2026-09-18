#include "ops/linear_add/ternary/ternary_linear_add_plan.h"

#include "ops/linear/ternary/ternary_format.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The registered residual-add problems (mirror the op's registered parent set).
constexpr bool admits_shape(std::int32_t n, std::int32_t k) {
    return (n == 5120 && k == 6144) || (n == 5120 && k == 17408) || (n == 2048 && k == 4096) ||
           (n == 2048 && k == 6144);
}

// P3b T routing (until P4 tuning): T=1 uses the GEMV route; 2..32 use the 16-token SIMT route;
// T>=33 uses the MMA route. All resolve to A16 compute for every policy.
enum class TernaryAddRoute {
    Gemv,
    Simt,
    Mma,
};

TernaryAddRoute resolve_route(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("ternary linear_add: T must be positive"); }
    if (tokens == 1) { return TernaryAddRoute::Gemv; }
    if (tokens <= 32) { return TernaryAddRoute::Simt; }
    return TernaryAddRoute::Mma;
}

} // namespace

std::size_t ternary_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                        std::int32_t input_rows,
                                                        LinearPolicy policy, std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("ternary linear_add: unsupported policy");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("ternary linear_add: invalid token interval");
    }
    if (!admits_shape(output_rows, input_rows)) {
        throw std::invalid_argument("ternary linear_add: unsupported shape");
    }
    (void)resolve_route(min_tokens);
    (void)resolve_route(max_tokens);
    // The A16-only ternary route needs no transient storage (plain writing).
    return 0;
}

void ternary_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream) {
    (void)policy;
    (void)workspace;
    if (x.ne[1] <= 0) { throw std::invalid_argument("ternary linear_add: T must be positive"); }
    if (!admits_shape(weight.n, weight.k)) {
        throw std::invalid_argument("ternary linear_add: unsupported shape");
    }
    (void)validate_ternary_weight_metadata(weight, "ternary linear_add");

    switch (resolve_route(x.ne[1])) {
    case TernaryAddRoute::Gemv:
        ternary_linear_add_gemv_launch(x, weight, residual, stream);
        return;
    case TernaryAddRoute::Simt:
        ternary_linear_add_simt_launch(x, weight, residual, stream);
        return;
    case TernaryAddRoute::Mma:
        ternary_linear_add_mma_launch(x, weight, residual, stream);
        return;
    }
    throw std::logic_error("ternary linear_add: unrouted token count");
}

} // namespace ninfer::ops::detail
