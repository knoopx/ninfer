#include "core/weight.h"
#include "ops/attn_input_proj/ternary/ternary_attn_input_plan.h"

#include "ops/linear/ternary/ternary_format.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

std::size_t ternary_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("ternary attn_input_proj workspace: unsupported policy");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("ternary attn_input_proj workspace: invalid token interval");
    }
    // Every policy resolves to the A16 routes, which need no transient storage.
    return 0;
}

void ternary_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                                 cudaStream_t stream) {
    if (x.ne[1] <= 0) { throw std::invalid_argument("ternary attn_input_proj: T must be positive"); }
    (void)validate_ternary_weight_metadata(weight, "ternary attn_input_proj");
    if (weight.n != 14336 || weight.k != 5120) {
        throw std::invalid_argument("ternary attn_input_proj: unsupported weight shape");
    }
    (void)policy;
    (void)workspace;

    const std::int32_t tokens = x.ne[1];
    if (tokens == 1) {
        ternary_attn_input_gemv_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (tokens <= 32) {
        // 16-token SIMT chunks; the ragged final chunk (1..15 live columns) passes its live
        // count through the RuntimeColumns path.
        for (std::int32_t offset = 0; offset < tokens; offset += 16) {
            const std::int32_t count = std::min(16, tokens - offset);
            auto input               = x.slice(1, offset, count);
            auto query               = q.slice(1, offset, count);
            auto output_gate         = gate.slice(1, offset, count);
            auto key                 = k.slice(1, offset, count);
            auto value               = v.slice(1, offset, count);
            ternary_attn_input_simt_launch(input, weight, query, output_gate, key, value,
                                           count == 16, stream);
        }
        return;
    }
    ternary_attn_input_mma_launch(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
