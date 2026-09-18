#include "ops/linear/ternary/ternary_dispatch.h"
#include "ops/linear/ternary/ternary_shapes.h"
#include "ops/linear/ternary/ternary_format.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kTernaryN14336K5120, &kTernaryN16384K5120, &kTernaryN34816K5120,
                         &kTernaryN248320K5120, &kTernaryN5120K6144, &kTernaryN5120K17408};

const TernaryLinearShape& resolve_shape(std::int32_t n, std::int32_t k) {
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("ternary linear: unsupported shape");
}
} // namespace

std::size_t ternary_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                    LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("ternary linear: unsupported policy");
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("ternary linear: invalid token interval");
    resolve_shape(n, k);
    return 0;
}

void ternary_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    (void)validate_ternary_weight_metadata(weight, "ternary linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("ternary linear: T must be positive");
    (void)policy;
    (void)workspace;
    const auto& shape = resolve_shape(weight.n, weight.k);
    shape.a16(x.ne[1])(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
