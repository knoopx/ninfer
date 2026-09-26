#pragma once

#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/int8/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/tiled_mma.cuh"
#include "ops/softmax_attention/common/causal_merge.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          bool ParallelQueries = false>
void launch_int8_kv_grouped_mma(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                                Input input, CausalKvPartition partition, CausalPartialView partial,
                                cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_quantized_causal_operands<G>(p, cache);
    if ((!ParallelQueries && p.width != S::kTokenTile) || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > CausalKvPartition::kMaxSplits || partition.target < 1 ||
        partition.key_shift < 6 || partition.key_shift > 12 ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("INT8 grouped attention: invalid schedule/partials");
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("INT8 append requires K/V");
    constexpr auto kernel =
        int8_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, ParallelQueries>;
    constexpr int bytes = S::kDynamicArena ? S::kArenaBytes : 0;
    if constexpr (S::kDynamicArena) {
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
    }
    const dim3 grid(G::KVHeads * (ParallelQueries ? div_up(p.width, S::kTokenTile) : 1),
                    partition.capacity, p.batch);
    kernel<<<grid, S::kThreads, bytes, stream>>>(
        p.q, input, p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales,
        cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride, p.width,
        p.visible_capacity, partition, p.scale, partial.acc, partial.maximum, partial.sum);
    CUDA_CHECK(cudaGetLastError());
}

// Narrow (4-warp) / wide (8-warp) selection. Both fast CTA shapes run one CTA per SM and every
// CTA of a launch sweeps a similar key range, so a launch costs about (waves) x (one CTA's
// sweep). A four-warp CTA sweeps in about 0.72 of an eight-warp CTA's time (measured on RTX 5090
// at 64K context) but covers half the rows.
bool int8_kv_tiled_mma_fast_prefers_narrow(std::int32_t tokens, std::int32_t q_heads) {
    static const int multiprocessors = [] {
        int device = 0;
        int count  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    const auto waves = [&](int rows) {
        return div_up(div_up(tokens, rows) * q_heads, multiprocessors);
    };
    constexpr int NarrowCostPercent = 72;
    return waves(Int8KvFastMmaSchedule<4>::Br) * NarrowCostPercent <
           waves(Int8KvFastMmaSchedule<8>::Br) * 100;
}

template <class G>
void launch_int8_kv_tiled_mma_fast(const CausalAttentionOperands& p, Int8KvReadView cache,
                                   cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("INT8 tiled attention requires a complete single query row");
    const auto tokens = p.width;
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        // Both fast CTA shapes exceed the default 48 KiB dynamic-smem ceiling.
        static const cudaError_t attr_wide = cudaFuncSetAttribute(
            int8_kv_tiled_mma_fast_kernel<G, Metadata, 8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Int8KvFastMmaSchedule<8>::SmemBytes);
        CUDA_CHECK(attr_wide);
        static const cudaError_t attr_narrow = cudaFuncSetAttribute(
            int8_kv_tiled_mma_fast_kernel<G, Metadata, 4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Int8KvFastMmaSchedule<4>::SmemBytes);
        CUDA_CHECK(attr_narrow);
        const auto launch = [&]<int Warps>() {
            using Shape = Int8KvFastMmaSchedule<Warps>;
            const dim3 grid(div_up(tokens, Shape::Br), G::QHeads);
            int8_kv_tiled_mma_fast_kernel<G, Metadata, Warps>
                <<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                    p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
                    p.positions, p.scale, p.out, p.width);
            CUDA_CHECK(cudaGetLastError());
        };
        if (int8_kv_tiled_mma_fast_prefers_narrow(tokens, G::QHeads))
            launch.template operator()<4>();
        else
            launch.template operator()<8>();
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
}

} // namespace ninfer::ops::detail
