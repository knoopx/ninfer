#include "artifact/layouts.h"

#include "artifact/formats.h"
#include "artifact/framing.h"
#include "artifact/schema.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::artifact {
namespace {

std::uint16_t read_u16_le(const std::byte* bytes) noexcept {
    return static_cast<std::uint16_t>(std::to_integer<unsigned>(bytes[0]) |
                                       std::to_integer<unsigned>(bytes[1]) << 8);
}

// A binary16 word is non-finite when its exponent field (bits 14:10) is all ones.
bool fp16_nonfinite(std::uint16_t word) noexcept { return (word & 0x7C00) == 0x7C00; }

} // namespace

WeightGeometry describe_tensor(const TensorObject& object) {
    try {
        auto geometry =
            weight_geometry(parse_format(object.format), parse_layout(object.layout), object.shape);
        if (geometry.bytes != object.bytes || object.offset % geometry.alignment) {
            throw ArtifactError("encoded size or object alignment differs from layout");
        }
        return geometry;
    } catch (const std::exception& error) { throw ArtifactError(object.id + ": " + error.what()); }
}

void validate_ternary_payload(const std::byte* payload, const WeightGeometry& geometry,
                              std::string_view id) {
    if (geometry.format != QType::TERNARY_PQ2_0) { return; }
    const auto n = geometry.shape[0];
    const auto k = geometry.shape[1];
    const auto groups = k / kTernaryPq2GroupSize;
    const auto blocks = n * groups;
    for (std::uint64_t block = 0; block < blocks; ++block) {
        const auto* start = payload + block * kTernaryPq2BlockBytes;
        if (fp16_nonfinite(read_u16_le(start))) {
            throw ArtifactError(std::string(id) + ": ternary PQ2_0 binary16 scale is nonfinite");
        }
    }
    const auto* rotation = payload + geometry.rotation_offset;
    if (read_u32_le(rotation) != kTernaryPq2RotationBlockSize) {
        throw ArtifactError(std::string(id) +
                            ": ternary PQ2_0 rotation block size differs from " +
                            std::to_string(kTernaryPq2RotationBlockSize));
    }
    const auto transform = std::to_integer<unsigned>(rotation[4]);
    if (transform != 0) {
        throw ArtifactError(std::string(id) + ": unsupported ternary PQ2_0 rotation transform");
    }
    const auto* signs = rotation + kTernaryPq2RotationHeaderBytes;
    for (std::uint64_t lane = 0; lane < k; ++lane) {
        const auto word = read_u32_le(signs + lane * kTernaryPq2SignWordBytes);
        if (word != 0x3F800000u && word != 0xBF800000u) {
            throw ArtifactError(std::string(id) + ": ternary PQ2_0 rotation sign is not +/-1");
        }
    }
}

} // namespace ninfer::artifact
