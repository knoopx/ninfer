#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    TERNARY_PQ2_0       = 9,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    TernaryPq2Block     = 4,
};

// Ternary PQ2_0 block constants: a 34-byte block holds one binary16 scale and 32 code
// bytes (2 bits per weight) for 128 weights, followed by a per-tensor rotation auxiliary.
inline constexpr std::uint32_t kTernaryPq2GroupSize       = 128;
inline constexpr std::uint32_t kTernaryPq2BlockBytes      = 34;
inline constexpr std::uint32_t kTernaryPq2RotationBlockSize = 1024;
inline constexpr std::uint32_t kTernaryPq2RotationHeaderBytes = 16;
inline constexpr std::uint32_t kTernaryPq2SignWordBytes   = 4;

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
    // Ternary PQ2_0 per-tensor rotation auxiliary (Hadamard header + sign vector).
    const void* rotation         = nullptr;
    std::uint64_t rotation_bytes = 0;
};

} // namespace ninfer
