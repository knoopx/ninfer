#include "artifact/formats.h"

#include "artifact/schema.h"

#include <array>

namespace ninfer::artifact {
namespace {

constexpr std::array kFormats = {
    std::pair{QType::BF16, std::string_view{"bf16"}},
    std::pair{QType::FP32, std::string_view{"fp32"}},
    std::pair{QType::INT32, std::string_view{"int32"}},
    std::pair{QType::Q4_G64_FP16, std::string_view{"q4_g64_fp16"}},
    std::pair{QType::Q5_G64_FP16, std::string_view{"q5_g64_fp16"}},
    std::pair{QType::Q6_G64_FP16, std::string_view{"q6_g64_fp16"}},
    std::pair{QType::Q8_G32_FP16, std::string_view{"q8_g32_fp16"}},
    std::pair{QType::NVFP4, std::string_view{"nvfp4"}},
    std::pair{QType::FP8_E4M3FN_ROW_BF16, std::string_view{"fp8_e4m3fn_row_bf16"}},
    // GGUF (ggml) quantized formats. GGUF F32/BF16 tensors reuse the existing "fp32"/
    // "bf16" spellings (QType::FP32/QType::BF16) above, so only the quantized ggml types
    // register new spellings here.
    std::pair{QType::GGUF_F16, std::string_view{"f16"}},
    std::pair{QType::GGUF_Q4_0, std::string_view{"q4_0"}},
    std::pair{QType::GGUF_Q4_1, std::string_view{"q4_1"}},
    std::pair{QType::GGUF_Q5_0, std::string_view{"q5_0"}},
    std::pair{QType::GGUF_Q5_1, std::string_view{"q5_1"}},
    std::pair{QType::GGUF_Q8_0, std::string_view{"q8_0"}},
    std::pair{QType::GGUF_Q8_1, std::string_view{"q8_1"}},
    std::pair{QType::GGUF_Q2_K, std::string_view{"q2_k"}},
    std::pair{QType::GGUF_Q3_K, std::string_view{"q3_k"}},
    std::pair{QType::GGUF_Q4_K, std::string_view{"q4_k"}},
    std::pair{QType::GGUF_Q5_K, std::string_view{"q5_k"}},
    std::pair{QType::GGUF_Q6_K, std::string_view{"q6_k"}},
    std::pair{QType::GGUF_Q8_K, std::string_view{"q8_k"}},
    std::pair{QType::GGUF_IQ2_XXS, std::string_view{"iq2_xxs"}},
    std::pair{QType::GGUF_IQ2_XS, std::string_view{"iq2_xs"}},
    std::pair{QType::GGUF_IQ2_S, std::string_view{"iq2_s"}},
    std::pair{QType::GGUF_IQ3_XXS, std::string_view{"iq3_xxs"}},
    std::pair{QType::GGUF_IQ1_S, std::string_view{"iq1_s"}},
    std::pair{QType::GGUF_IQ1_M, std::string_view{"iq1_m"}},
    std::pair{QType::GGUF_IQ3_S, std::string_view{"iq3_s"}},
    std::pair{QType::GGUF_IQ4_NL, std::string_view{"iq4_nl"}},
    std::pair{QType::GGUF_IQ4_XS, std::string_view{"iq4_xs"}},
    std::pair{QType::GGUF_I8, std::string_view{"i8"}},
    std::pair{QType::GGUF_Q1_0, std::string_view{"q1_0"}},
    std::pair{QType::GGUF_Q2_0, std::string_view{"q2_0"}},
};
constexpr std::array kLayouts = {
    std::pair{QuantLayout::Contiguous, std::string_view{"contiguous_le_v1"}},
    std::pair{QuantLayout::RowSplit, std::string_view{"row_split_k128_v1"}},
    std::pair{QuantLayout::RowScale, std::string_view{"row_scale_v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4, std::string_view{"block_scale_k16_m128x4_v1"}},
    std::pair{QuantLayout::GgufNative, std::string_view{"gguf_native"}},
};

} // namespace

QType parse_format(std::string_view name) {
    for (const auto& [format, spelling] : kFormats) {
        if (spelling == name) { return format; }
    }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

QuantLayout parse_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

std::string_view format_name(QType format) noexcept {
    for (const auto& [value, spelling] : kFormats) {
        if (value == format) { return spelling; }
    }
    return {};
}

std::string_view layout_name(QuantLayout layout) noexcept {
    for (const auto& [value, spelling] : kLayouts) {
        if (value == layout) { return spelling; }
    }
    return {};
}

} // namespace ninfer::artifact
