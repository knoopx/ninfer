#include "ops/linear/ternary/ternary_format.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(std::string(operation) + ": ternary PQ2_0 geometry overflows");
    }
    return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(operation) + ": ternary PQ2_0 geometry overflows");
    }
    return left + right;
}

std::uint32_t read_u32_le(const std::byte* pointer) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(pointer);
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint16_t read_u16_le(const std::byte* pointer) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(pointer);
    return static_cast<std::uint16_t>(bytes[0] | (static_cast<std::uint16_t>(bytes[1]) << 8));
}

std::uint32_t read_u32_bits(const float* pointer) {
    std::uint32_t bits;
    std::memcpy(&bits, pointer, sizeof(bits));
    return bits;
}

} // namespace

// Metadata-only validation: geometry, pointer wiring, and payload accounting. Never
// dereferences qdata/rotation bytes (the returned view's .blocks/.signs are computed, not
// read), so it is safe with device-resident payloads. Throws std::invalid_argument on the
// first violation.
TernaryWeightView validate_ternary_weight_metadata_impl(const Weight& weight, const char* operation,
                                                       bool row_view) {
    if (weight.n <= 0 || weight.k <= 0 || (weight.k % kTernaryPq2GroupSize) != 0) {
        throw std::invalid_argument(std::string(operation) +
                                    ": ternary PQ2_0 requires positive N and K%128=0");
    }

    const std::uint64_t groups = static_cast<std::uint64_t>(weight.k) / kTernaryPq2GroupSize;
    const std::uint64_t block_plane_bytes =
        checked_mul(checked_mul(static_cast<std::uint64_t>(weight.n), groups, operation),
                    kTernaryPq2BlockBytes, operation);
    const std::uint64_t required_rotation_bytes =
        checked_add(kTernaryPq2RotationHeaderBytes,
                    checked_mul(static_cast<std::uint64_t>(weight.k),
                                kTernaryPq2SignWordBytes, operation),
                    operation);

    if (weight.qtype != QType::TERNARY_PQ2_0 || weight.layout != QuantLayout::TernaryPq2Block ||
        weight.group_size != kTernaryPq2GroupSize || weight.group != kTernaryPq2GroupSize ||
        weight.ndim != 2 || weight.shape[0] != weight.n || weight.shape[1] != weight.k ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k ||
        weight.payload == nullptr || weight.qdata == nullptr || weight.qhigh != nullptr ||
        weight.scales != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < checked_add(block_plane_bytes, required_rotation_bytes, operation) ||
        !aligned_to(weight.qdata, 16) || weight.rotation == nullptr ||
        weight.rotation_bytes != required_rotation_bytes || !aligned_to(weight.rotation, 16)) {
        throw std::invalid_argument(std::string(operation) + ": invalid ternary PQ2_0 weight");
    }

    const auto* payload = static_cast<const std::byte*>(weight.payload);
    if (row_view) {
        // A row view addresses a row-aligned slice of the parent block plane; its qdata is an
        // offset into payload (not the plane head) and must stay inside the parent payload.
        const std::uintptr_t q  = reinterpret_cast<std::uintptr_t>(weight.qdata);
        const std::uintptr_t p  = reinterpret_cast<std::uintptr_t>(payload);
        if (q < p || q - p >= weight.payload_bytes || q % 16 != 0) {
            throw std::invalid_argument(std::string(operation) +
                                        ": invalid ternary PQ2_0 row-view plane geometry");
        }
    } else if (weight.qdata != payload) {
        throw std::invalid_argument(std::string(operation) + ": invalid ternary PQ2_0 plane geometry");
    }

    // Pointer arithmetic only: the view's .blocks/.signs are computed, never read.
    const auto* blocks = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* signs  = reinterpret_cast<const float*>(static_cast<const std::byte*>(
                                                                     weight.rotation) +
                                                         kTernaryPq2RotationHeaderBytes);
    return {blocks, signs, static_cast<std::uint32_t>(groups), block_plane_bytes};
}

// Content scan: dereferences the qdata/rotation bytes. Host-side only (the bytes are
// device-resident in the CUDA build, where this scan SIGSEGVs); the full validators run it
// after the metadata checks, and the artifact materializer runs the same scan on the host
// payload at load time (validate_ternary_payload).
void validate_ternary_weight_content(const Weight& weight, const char* operation) {
    const std::uint64_t groups = static_cast<std::uint64_t>(weight.k) / kTernaryPq2GroupSize;

    // Every block starts with its binary16 scale word; a stored scale must be finite.
    const auto* blocks = static_cast<const std::uint8_t*>(weight.qdata);
    for (std::uint64_t block = 0; block < weight.n * groups; ++block) {
        const auto* start = blocks + block * kTernaryPq2BlockBytes;
        if (!ternary_pq2_scale_word_finite(read_u16_le(reinterpret_cast<const std::byte*>(start)))) {
            throw std::invalid_argument(
                std::string(operation) + ": ternary PQ2_0 binary16 scale is nonfinite");
        }
    }

    // Rotation auxiliary: the pinned 16-byte header plus K sign words of exactly +/-1.0.
    const auto* rotation = static_cast<const std::byte*>(weight.rotation);
    if (read_u32_le(rotation) != kTernaryPq2RotationBlockSize) {
        throw std::invalid_argument(std::string(operation) +
                                    ": ternary PQ2_0 rotation block size differs from 1024");
    }
    const auto transform = static_cast<unsigned>(rotation[4]);
    if (transform != 0) {
        throw std::invalid_argument(std::string(operation) +
                                    ": unsupported ternary PQ2_0 rotation transform");
    }
    if (static_cast<unsigned>(rotation[7]) != 0) {
        throw std::invalid_argument(std::string(operation) +
                                    ": ternary PQ2_0 rotation reserved byte is not zero");
    }
    const auto* signs = reinterpret_cast<const float*>(rotation + kTernaryPq2RotationHeaderBytes);
    for (std::int32_t lane = 0; lane < weight.k; ++lane) {
        const std::uint32_t word = read_u32_bits(signs + lane);
        if (word != 0x3F800000u && word != 0xBF800000u) {
            throw std::invalid_argument(std::string(operation) +
                                        ": ternary PQ2_0 rotation sign is not +/-1");
        }
    }
}

TernaryWeightView validate_ternary_weight_metadata(const Weight& weight, const char* operation) {
    return validate_ternary_weight_metadata_impl(weight, operation, /*row_view=*/false);
}

TernaryWeightView validate_ternary_row_view_metadata(const Weight& weight, const char* operation) {
    return validate_ternary_weight_metadata_impl(weight, operation, /*row_view=*/true);
}

TernaryWeightView validate_ternary_weight(const Weight& weight, const char* operation) {
    const TernaryWeightView view =
        validate_ternary_weight_metadata_impl(weight, operation, /*row_view=*/false);
    validate_ternary_weight_content(weight, operation);
    return view;
}

TernaryWeightView validate_ternary_row_view(const Weight& weight, const char* operation) {
    const TernaryWeightView view =
        validate_ternary_weight_metadata_impl(weight, operation, /*row_view=*/true);
    validate_ternary_weight_content(weight, operation);
    return view;
}

} // namespace ninfer::ops::detail
