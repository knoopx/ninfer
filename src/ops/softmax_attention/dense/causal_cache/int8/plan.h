#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"

namespace ninfer::ops::detail {

enum class Int8KvFamily { Grouped, ParallelGrouped, Tiled };

// Widest prompt-route row block (the eight-warp fast INT8 kernel's Br); every prompt
// kernel's row block divides it. Consumed by causal_softmax_attention_prompt_wave_tokens.
inline constexpr std::int32_t kInt8PromptWaveRows = 128;

static_assert(kInt8PromptWaveRows == Int8KvFastMmaSchedule<8>::Br);
static_assert(kInt8PromptWaveRows % Int8KvFastMmaSchedule<4>::Br == 0);

struct Int8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Int8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope);
std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
