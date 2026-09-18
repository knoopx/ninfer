#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstring>

namespace ninfer::ops::detail {

// Ternary PQ2_0 decode contract (spec: docs/maintainer/ternary-pq2-0.md, sections 2-3).
// A 34-byte block (src/core/weight.h) holds 128 weights: bytes 0..1 are a little-endian
// binary16 scale d, bytes 2..33 are 32 code bytes. Weight j of the block sits in code
// byte j/4, bits 2*(j%4) through 2*(j%4)+1, low bits first. The fixed codebook is
// 00 -> -1, 01 -> 0, 10 -> +1, 11 -> +2, and the represented weight is (code - 1) * d.
//
// The per-tensor rotation auxiliary follows the block plane: a 16-byte header
// (u32 block_size = 1024, u8 transform = 0, u8 gdn_v_grouped, u8 inverse, u8 reserved,
// u32, u32) followed by K little-endian binary32 signs, every sign exactly +1.0
// (0x3F800000) or -1.0 (0xBF800000). The runtime applies the rotation to activations:
// A(x) = (1/sqrt(1024)) * FWT(x * s), fused into the GEMM A-tile load.

// Fixed 2-bit codebook, indexed by the 2-bit code word (host decode path; the device GEMM
// routes compute (code - 1) inline, matching the codebook exactly).
inline constexpr std::int32_t kTernaryPq2Codebook[4] = {-1, 0, 1, 2};

// Slot [0,4) within one code byte, low bits first.
[[nodiscard]] inline constexpr int ternary_pq2_code(std::uint8_t byte, int slot) noexcept {
    return (byte >> (2 * slot)) & 3;
}

// The 4 slot codes of one code byte, low bits first.
[[nodiscard]] inline constexpr int ternary_pq2_code_at(const std::uint8_t* code_bytes,
                                                        int weight) noexcept {
    return ternary_pq2_code(code_bytes[weight / 4], weight % 4);
}

// A stored binary16 scale word is finite when its exponent field (bits 14:10) is not all
// ones. Mirrors tools/artifact/formats.py:valid_ternary_pq2_scale_word.
[[nodiscard]] inline constexpr bool ternary_pq2_scale_word_finite(std::uint16_t word) noexcept {
    return (word & 0x7C00u) != 0x7C00u;
}

// A binary16 word expanded exactly to binary32. The caller passes a finite word. The
// subnormal path uses the double 0.5 + m*2^-24 - 0.5 normalization trick, which is exact in
// binary64 and exact in binary32 (every binary16 subnormal is representable in binary32).
[[nodiscard]] inline float ternary_pq2_scale_word_to_float(std::uint16_t word) noexcept {
    const std::uint32_t sign     = static_cast<std::uint32_t>(word & 0x8000u) << 16;
    const std::uint32_t exponent = (word >> 10) & 0x1Fu;
    const std::uint32_t mantissa = word & 0x3FFu;
    if (exponent == 0) {
        // m * 2^-24: the binary16 subnormal weight. 0.5f in binary64 with the mantissa in its
        // top 10 fraction bits gives 0.5 + m * 2^-24 exactly; subtracting 0.5 leaves the value.
        // (A binary64 fraction bit k has weight 2^(k-52) on the 1.0 ulp; the value is
        // 2^-1 * (1 + m * 2^(29-52)), so the mantissa sits at bits 29..38 and
        // 2^-1 * m * 2^-23 = m * 2^-24.)
        const std::uint64_t bits = 0x3FE0000000000000ull | (static_cast<std::uint64_t>(mantissa) << 29) |
                                  (sign ? 0x8000000000000000ull : 0ull);
        double result;
        std::memcpy(&result, &bits, sizeof(double));
        return static_cast<float>(result - 0.5);
    }
    const std::uint32_t bits =
        sign | ((exponent + 112u) << 23) | (mantissa << 13); // 127 - 15 exponent rebias.
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// Decode one code byte against a scale word: 4 represented weights (code - 1) * d, exact
// in binary32 (the binary16 scale is exact in binary32 and |code - 1| <= 2).
inline void ternary_pq2_decode_byte(std::uint8_t code_byte, std::uint16_t scale_word,
                                                  float (&values)[4]) noexcept {
    const float scale = ternary_pq2_scale_word_to_float(scale_word);
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        values[slot] = static_cast<float>(kTernaryPq2Codebook[ternary_pq2_code(code_byte, slot)]) * scale;
    }
}

struct TernaryWeightView {
    const std::uint8_t* blocks; ///< The 34-byte block plane (weight.qdata), N * K/128 blocks.
    const float* signs;         ///< The K sign words of the rotation auxiliary.
    std::uint32_t groups_per_row;
    std::uint64_t block_plane_bytes;
};

// Validates the TERNARY_PQ2_0 weight metadata: block geometry, the qdata/rotation pointer
// wiring (non-null, 16-byte aligned, payload bounds), and the payload byte accounting.
// Never dereferences qdata/rotation bytes, so it is safe with device-resident payloads (the
// CUDA dispatch/plan path); the returned view's .blocks/.signs are computed, not read. The
// content scan lives in the full host-side validators below and in the artifact materializer
// (validate_ternary_payload). Throws std::invalid_argument on the first violation.
[[nodiscard]] TernaryWeightView validate_ternary_weight_metadata(const Weight& weight,
                                                                 const char* operation);

// Validates a TERNARY_PQ2_0 row view: the same metadata checks as
// validate_ternary_weight_metadata, but the block plane head (qdata) is a row-aligned offset
// into the parent payload rather than the payload head itself. Used by the [1024,2048]
// adjacent K/V row-view registration of linear_pair. Never dereferences qdata/rotation bytes.
[[nodiscard]] TernaryWeightView validate_ternary_row_view_metadata(const Weight& weight,
                                                                   const char* operation);

// Validates the TERNARY_PQ2_0 weight: the metadata checks plus the content scan (every block
// scale word finite, the rotation header, and every sign word +/-1.0). Host-side only: the
// content scan dereferences qdata/rotation, so it must not run on device-resident payloads.
// Throws std::invalid_argument on the first violation.
[[nodiscard]] TernaryWeightView validate_ternary_weight(const Weight& weight,
                                                        const char* operation);

// Host-side full validation of a TERNARY_PQ2_0 row view: the row-view metadata checks plus
// the content scan (see validate_ternary_weight).
[[nodiscard]] TernaryWeightView validate_ternary_row_view(const Weight& weight,
                                                          const char* operation);

} // namespace ninfer::ops::detail
