#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

namespace ninfer::models::qwen3_5::execution {

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear(input, p.weight, output, p.policy, workspace, stream);
}

inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    if (p.weight.layout == QuantLayout::GgufNative) {
        // The fused linear_add route has no native GGUF form, so the projection is applied on its
        // own and added to the residual.
        const auto columns = residual.ne[1];
        Tensor delta       = workspace.alloc(DType::BF16, {p.weight.n, columns});
        {
            auto scope = workspace.scope();
            ops::linear(input, p.weight, delta, p.policy, workspace, stream);
        }
        ops::residual_add(delta, residual, stream);
        return;
    }
    auto scope = workspace.scope();
    ops::linear_add(input, p.weight, residual, p.policy, workspace, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_swiglu(input, p.weight, output, p.policy, workspace, stream);
}

// The native GGUF form: the gate and up parents are separate tensors, so the two projections are
// applied on their own and combined by silu_mul.
inline void project_swiglu(const Tensor& input, const LinearParameters& gate,
                           const LinearParameters& up, Tensor& output, WorkspaceArena& workspace,
                           cudaStream_t stream) {
    const auto columns = output.ne[1];
    Tensor g = workspace.alloc(DType::BF16, {gate.weight.n, columns});
    Tensor u = workspace.alloc(DType::BF16, {up.weight.n, columns});
    {
        auto scope = workspace.scope();
        ops::linear(input, gate.weight, g, gate.policy, workspace, stream);
    }
    {
        auto scope = workspace.scope();
        ops::linear(input, up.weight, u, up.policy, workspace, stream);
    }
    ops::silu_mul(g, u, output, stream);
}

} // namespace ninfer::models::qwen3_5::execution
