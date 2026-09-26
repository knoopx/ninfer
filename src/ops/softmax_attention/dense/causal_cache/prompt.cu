// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
static_assert(kPromptWaveRows == CausalPromptI8FastShape<8>::Br);
static_assert(kPromptWaveRows % CausalPromptI8FastShape<4>::Br == 0);
static_assert(kPromptWaveRows % kCausalPromptBr == 0);

namespace {

// Both fast INT8 CTA shapes run one CTA per SM and every CTA of a launch sweeps a similar key
// range, so a launch costs about (waves) x (one CTA's sweep). A four-warp CTA sweeps in about
// 0.72 of an eight-warp CTA's time (measured on RTX 5090 at 64K context) but covers half the
// rows.
bool causal_attention_prompt_i8_fast_prefers_narrow(std::int32_t tokens, std::int32_t q_heads) {
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
    return waves(CausalPromptI8FastShape<4>::Br) * NarrowCostPercent <
           waves(CausalPromptI8FastShape<8>::Br) * 100;
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_i8_fast_launch_for(const Tensor& q, const Tensor& positions,
                                                float scale, const CacheView& cache,
                                                Metadata metadata, Tensor& out,
                                                cudaStream_t stream) {
    static const cudaError_t attr_wide = cudaFuncSetAttribute(
        causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, 8>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, CausalPromptI8FastShape<8>::SmemBytes);
    CUDA_CHECK(attr_wide);
    static const cudaError_t attr_narrow = cudaFuncSetAttribute(
        causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, CausalPromptI8FastShape<4>::SmemBytes);
    CUDA_CHECK(attr_narrow);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const auto launch = [&]<int Warps>() {
        using Shape = CausalPromptI8FastShape<Warps>;
        const dim3 grid(static_cast<unsigned>(div_up(tokens, Shape::Br)),
                        static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, Warps>
            <<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache.k_pages.data),
                static_cast<const std::int8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    };
    if (causal_attention_prompt_i8_fast_prefers_narrow(tokens, Geometry::QHeads)) {
        launch.template operator()<4>();
    } else {
        launch.template operator()<8>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Int8Group64) {
        causal_attention_prompt_i8_fast_launch_for<Geometry>(q, positions, scale, cache, metadata,
                                                             out, stream);
        return;
    }
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptSmemBytes);
    CUDA_CHECK(attr_bf16);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_bf16_kernel<Geometry, Metadata>
        <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(cache_k.data),
            static_cast<const __half*>(cache_v.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
