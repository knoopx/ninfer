#include "ops/linear/gguf/gguf_dispatch.h"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

std::size_t gguf_checked_bytes(std::int64_t count, std::size_t bytes_each) {
    if (count <= 0) {
        throw std::invalid_argument("gguf linear workspace: token count must be positive");
    }
    if (bytes_each != 0 && count > std::numeric_limits<std::size_t>::max() / bytes_each) {
        throw std::overflow_error("gguf linear workspace size overflow");
    }
    return static_cast<std::size_t>(count) * bytes_each;
}

// The Q2_K block type id in a GGUF tensor directory (ggml_type) and its super-block geometry:
// 256 values in 84 bytes (16 packed 4-bit scale/min pairs, 64 bytes of 2-bit quants, fp16 d/dmin).
constexpr std::uint32_t kGgmlQ2K      = 10;
constexpr std::size_t kQ2KBlockElems  = 256;
constexpr std::size_t kQ2KBlockBytes  = 84;

// The staging planes are [K,T] and [N,T], so one pass over a long prefill would reserve gigabytes
// for a single projection (a 248320-row head at the 4096-token prefill chunk is 4 GiB). Every
// kernel in the GGUF routes is per-column, so the projection is chunked over T and one projection
// is bounded to this many bytes per slice.
constexpr std::uint64_t kGgufSliceBudgetBytes = 64ULL * 1024 * 1024;

std::int32_t gguf_token_chunk(std::int32_t input_rows, std::int32_t output_rows) {
    const auto k = static_cast<std::uint64_t>(input_rows);
    const auto n = static_cast<std::uint64_t>(output_rows);
    // x_f32 + y_f32 + both q8_1 stagings, per column.
    const std::uint64_t per_token = (k + n) * 4 + (k / 32) * (34 + 36);
    const std::uint64_t chunk =
        per_token == 0 ? 1 : std::max<std::uint64_t>(1, kGgufSliceBudgetBytes / per_token);
    return static_cast<std::int32_t>(std::min<std::uint64_t>(chunk, 4096));
}

// K granularity the native route requires (its q8_1 block / super-block width):
// 256 for the K-quants and IQ4_XS, 64 for Q2_0, 32 for Q4_0/Q5_0/Q8_0/IQ4_NL.
std::uint32_t native_k_granularity(std::uint32_t ggml_type) {
    switch (ggml_type) {
        case 11:  // Q3_K
        case 12:  // Q4_K
        case 13:  // Q5_K
        case 14:  // Q6_K
        case 23:  // IQ4_XS
            return 256;
        case 42:  // Q2_0
            return 64;
        default:
            return 32;
    }
}

template <class Arena>
GgufWorkspace allocate_gguf_workspace(Arena& arena, std::int32_t tokens, std::int32_t input_rows,
                                      std::int32_t output_rows) {
    if (tokens <= 0) {
        throw std::invalid_argument("gguf linear workspace: T must be positive");
    }
    if (input_rows <= 0 || (input_rows % 32) != 0) {
        throw std::invalid_argument("gguf linear workspace: K must be a positive multiple of 32");
    }
    if (output_rows <= 0) {
        throw std::invalid_argument("gguf linear workspace: N must be positive");
    }
    const std::int64_t k = input_rows;
    const std::int64_t t = tokens;
    const std::int64_t n = output_rows;
    GgufWorkspace workspace;
    workspace.x_f32 = static_cast<float*>(arena.alloc_bytes(gguf_checked_bytes(k * t, 4), 256).data);
    // iq route: Strata's q8_1 staging (34 bytes per 32 values).
    workspace.q8_1 = static_cast<std::uint8_t*>(
        arena.alloc_bytes(gguf_checked_bytes(t * (k / 32) * 34, 256), 256).data);
    // native route: llama.cpp block_q8_1 staging (36 bytes per 32 values).
    workspace.q8_1_native = static_cast<std::uint8_t*>(
        arena.alloc_bytes(gguf_checked_bytes(t * (k / 32) * 36, 256), 256).data);
    workspace.y_f32 = static_cast<float*>(arena.alloc_bytes(gguf_checked_bytes(n * t, 4), 256).data);
    return workspace;
}

void validate_gguf_weight(const Weight& weight, const char* operation) {
    if (weight.layout != QuantLayout::GgufNative) {
        throw std::invalid_argument(std::string(operation) + ": weight layout must be GgufNative");
    }
    if (weight.payload == nullptr) {
        throw std::invalid_argument(std::string(operation) + ": weight payload is null");
    }
    if (weight.ggml_type == 0) {
        throw std::invalid_argument(std::string(operation) + ": weight ggml type is unset");
    }
    const bool iq_route     = strata::kernels::iq_supported(weight.ggml_type);
    const bool native_route = strata::kernels::native_mmvq_supported(weight.ggml_type);
    // Q2_K is served by the ported vector kernel (see gguf_kernels.cu), so it passes this gate.
    const bool q2_k_route = weight.ggml_type == kGgmlQ2K;
    if (!iq_route && !native_route && !q2_k_route) {
        throw std::invalid_argument(std::string(operation) + ": unsupported ggml type " +
                                    std::to_string(weight.ggml_type) + " (" +
                                    strata::ggml_type_name(weight.ggml_type) + ")");
    }
    const std::size_t row_bytes =
        iq_route ? strata::kernels::iq_row_bytes(weight.ggml_type, weight.k)
                 : (native_route
                        ? strata::kernels::native_mmvq_weight_bytes(weight.ggml_type, weight.k, 1)
                        : static_cast<std::size_t>(weight.k / kQ2KBlockElems) * kQ2KBlockBytes);
    if (weight.payload_bytes != static_cast<std::uint64_t>(weight.n) * row_bytes) {
        throw std::invalid_argument(std::string(operation) + ": weight payload size mismatch");
    }
}

} // namespace

std::size_t gguf_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                 LinearPolicy policy, std::int32_t min_tokens,
                                                 std::int32_t max_tokens) {
    (void)policy;
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gguf linear workspace: invalid token interval");
    }
    // The dispatch chunks T, so the capacity is one slice's footprint, not the whole interval.
    const std::int32_t tokens = std::min(max_tokens, gguf_token_chunk(input_rows, output_rows));
    WorkspaceLayoutBuilder layout;
    (void)allocate_gguf_workspace(layout, std::max(tokens, 1), input_rows, output_rows);
    return layout.peak_bytes(1);
}

void gguf_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream) {
    // The policy selects no alternative for GGUF weights (see the header contract).
    (void)policy;
    validate_gguf_weight(weight, "gguf linear");
    const std::int32_t k = weight.k;
    const std::int32_t n = weight.n;
    const std::int32_t t = x.ne[1];
    if (t <= 0) {
        throw std::invalid_argument("gguf linear: T must be positive");
    }
    const bool iq_route = strata::kernels::iq_supported(weight.ggml_type);
    const bool native_route = !iq_route && strata::kernels::native_mmvq_supported(weight.ggml_type);
    // Q2_K: the one type neither Strata route covers, served by the ported vector kernel below.
    const bool q2_k_route = !iq_route && !native_route && weight.ggml_type == kGgmlQ2K;
    if (!iq_route && !native_route && !q2_k_route) {
        throw std::invalid_argument("gguf linear: unsupported ggml type " +
                                    std::to_string(weight.ggml_type) + " (" +
                                    strata::ggml_type_name(weight.ggml_type) + ")");
    }
    const std::uint32_t k_gran =
        iq_route ? 32u : (native_route ? native_k_granularity(weight.ggml_type) : 256u);
    if (k % k_gran != 0) {
        throw std::invalid_argument("gguf linear: K must be a multiple of " + std::to_string(k_gran) +
                                    ", got " + std::to_string(k));
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("gguf linear requires caller workspace");
    }
    auto scope = workspace->scope();
    // Chunk over T: the FP32 staging planes are [K,T] and [N,T], and every kernel below is
    // per-column, so projecting a long prefill in slices keeps one projection's workspace bounded
    // instead of reserving a plane for every column at once.
    // The iq route runs up to the caller's workspace cap; the native and Q2_K vector kernels take
    // 1..8 activation columns per launch, so a wider product chunks T (every kernel is per-column).
    const std::int32_t t_chunk =
        iq_route ? gguf_token_chunk(k, n) : std::min<std::int32_t>(t, 8);
    for (std::int32_t t0 = 0; t0 < t; t0 += t_chunk) {
        const std::int32_t slice = std::min(t_chunk, t - t0);
        auto slice_scope         = workspace->scope();
        const GgufWorkspace ws   = allocate_gguf_workspace(*workspace, slice, k, n);
        const auto* x_slice =
            static_cast<const std::uint16_t*>(x.data) + static_cast<std::int64_t>(t0) * k;
        auto* out_slice =
            static_cast<std::uint16_t*>(out.data) + static_cast<std::int64_t>(t0) * n;
        gguf_bf16_to_f32_launch(x_slice, ws.x_f32, static_cast<std::int64_t>(k) * slice, stream);
        if (iq_route) {
            strata::kernels::quantize_q8_1_rows(ws.x_f32, slice, k, ws.q8_1, stream);
            strata::kernels::iq_mmvq(weight.ggml_type, weight.payload, ws.q8_1, ws.y_f32, k, n,
                                     slice, stream);
        } else if (native_route) {
            strata::kernels::native_quantize_q8_1(ws.x_f32, ws.q8_1_native, k, slice, stream);
            strata::kernels::native_mmvq(weight.ggml_type, weight.payload, ws.q8_1_native,
                                         ws.y_f32, k, n, slice, stream);
        } else {
            // Q2_K: Strata stages the same llama.cpp block_q8_1 the ported kernel reads.
            strata::kernels::native_quantize_q8_1(ws.x_f32, ws.q8_1_native, k, slice, stream);
            const auto row_bytes =
                static_cast<std::int64_t>(weight.payload_bytes / static_cast<std::uint64_t>(n));
            gguf_q2_k_mmvq_launch(weight.payload, row_bytes, ws.q8_1_native, k, n, slice,
                                  ws.y_f32, stream);
        }
        strata::kernels::f32_to_bf16_bulk(ws.y_f32, out_slice,
                                          static_cast<std::int64_t>(n) * slice, stream);
    }
}

} // namespace ninfer::ops::detail
