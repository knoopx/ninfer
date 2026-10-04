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
    // GGUF (ggml) formats, one value per supported ggml type. The GGUF float types reuse
    // the existing .ninfer values: GGUF F32 is FP32 and GGUF BF16 is BF16, so no duplicate
    // format spellings are registered for them.
    GGUF_F16     = 9,
    GGUF_Q4_0    = 10,
    GGUF_Q4_1    = 11,
    GGUF_Q5_0    = 12,
    GGUF_Q5_1    = 13,
    GGUF_Q8_0    = 14,
    GGUF_Q8_1    = 15,
    GGUF_Q2_K    = 16,
    GGUF_Q3_K    = 17,
    GGUF_Q4_K    = 18,
    GGUF_Q5_K    = 19,
    GGUF_Q6_K    = 20,
    GGUF_Q8_K    = 21,
    GGUF_IQ2_XXS = 22,
    GGUF_IQ2_XS  = 23,
    GGUF_IQ2_S   = 24,
    GGUF_IQ3_XXS = 25,
    GGUF_IQ1_S   = 26,
    GGUF_IQ1_M   = 27,
    GGUF_IQ3_S   = 28,
    GGUF_IQ4_NL  = 29,
    GGUF_IQ4_XS  = 30,
    GGUF_I8      = 31,
    GGUF_Q1_0    = 32,
    GGUF_Q2_0    = 33,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // Raw GGUF payload (strata ggml block encodings). GgufReader builds the geometry
    // directly; weight_geometry() is never used for this layout, so GGUF stays out of
    // the repacking paths.
    GgufNative          = 4,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    // ggml type id (strata::ggml_type_name); meaningful only for GGUF qtypes
    // (GgufNative layout), where it carries the ggml block-encoding id. 0 otherwise.
    std::uint16_t ggml_type        = 0;
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
};

} // namespace ninfer
