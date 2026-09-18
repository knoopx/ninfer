#include "core/weight.h"
#include "ops/gdn_input_proj/ternary/ternary_gdn_input_plan.h"

#include "ops/linear/ternary/ternary_format.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

std::size_t ternary_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("ternary gdn_input_proj workspace: unsupported policy");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("ternary gdn_input_proj workspace: invalid token interval");
    }
    // Every policy resolves to the A16 routes, which need no transient storage.
    return 0;
}

void ternary_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                LinearPolicy policy, WorkspaceArena* workspace,
                                cudaStream_t stream) {
    if (x.ne[1] <= 0) { throw std::invalid_argument("ternary gdn_input_proj: T must be positive"); }
    (void)validate_ternary_weight_metadata(weight, "ternary gdn_input_proj");
    if (weight.n != 16384 || weight.k != 5120) {
        throw std::invalid_argument("ternary gdn_input_proj: unsupported weight shape");
    }
    (void)policy;
    (void)workspace;

    const std::int32_t tokens = x.ne[1];
    if (tokens == 1) {
        ternary_gdn_input_gemv_launch(x, weight, qkv, z, stream);
        return;
    }
    if (tokens <= 32) {
        // 16-token SIMT chunks; the ragged final chunk (1..15 live columns) passes its live
        // count through the RuntimeColumns path.
        for (std::int32_t offset = 0; offset < tokens; offset += 16) {
            const std::int32_t count = std::min(16, tokens - offset);
            auto input               = x.slice(1, offset, count);
            auto qkv_chunk           = qkv.slice(1, offset, count);
            auto z_chunk             = z.slice(1, offset, count);
            ternary_gdn_input_simt_launch(input, weight, qkv_chunk, z_chunk, count == 16, stream);
        }
        return;
    }
    ternary_gdn_input_mma_launch(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
