#include "artifact/fixture.h"
#include "artifact/layouts.h"
#include "core/weight_view.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

// Shape (2, 256): two rows of two 128-weight groups. Block array = 4 * 34 = 136 bytes,
// 256-aligned rotation auxiliary at offset 256 (16-byte header + 256 binary32 signs).
std::vector<std::byte> ternary_payload() {
    const std::uint64_t n = 2, k = 256;
    const auto geometry =
        weight_geometry(QType::TERNARY_PQ2_0, QuantLayout::TernaryPq2Block,
                        std::array<std::uint64_t, 2>{n, k});
    std::vector<std::byte> buffer(geometry.bytes);
    for (std::size_t block = 0; block < n * (k / kTernaryPq2GroupSize); ++block) {
        auto* start = buffer.data() + block * kTernaryPq2BlockBytes;
        start[0] = std::byte{0x00};
        start[1] = std::byte{0x38}; // binary16 0x3800 = +0.5, a finite scale.
        for (std::size_t i = 2; i < kTernaryPq2BlockBytes; ++i) {
            start[i] = std::byte{0x55}; // 01 codes: zero values.
        }
    }
    auto* rotation = buffer.data() + geometry.rotation_offset;
    for (unsigned i = 0; i < 4; ++i) { // u32 block size 1024, little-endian.
        rotation[i] = std::byte((kTernaryPq2RotationBlockSize >> (8 * i)) & 255);
    }
    auto* signs = rotation + kTernaryPq2RotationHeaderBytes;
    for (std::uint64_t lane = 0; lane < k; ++lane) {
        const auto word = lane < k / 2 ? 0x3F800000u : 0xBF800000u; // +/-1.0.
        for (unsigned i = 0; i < 4; ++i) {
            signs[lane * 4 + i] = std::byte((word >> (8 * i)) & 255);
        }
    }
    return buffer;
}

void ternary_geometry_and_native_weight() {
    const auto geometry =
        weight_geometry(QType::TERNARY_PQ2_0, QuantLayout::TernaryPq2Block,
                        std::array<std::uint64_t, 2>{2, 256});
    require(geometry.bytes == 1296 && geometry.rotation_offset == 256 &&
                geometry.rotation_bytes == 16 + 256 * 4 && geometry.code_bytes == 136 &&
                geometry.code_bytes_per_row == 2 * kTernaryPq2BlockBytes &&
                geometry.group_size == kTernaryPq2GroupSize && geometry.scale_bytes == 0,
            "ternary geometry did not keep the block array and rotation auxiliary");
    rejects<std::invalid_argument>(
        [&] {
            (void)weight_geometry(QType::TERNARY_PQ2_0, QuantLayout::TernaryPq2Block,
                                  std::array<std::uint64_t, 2>{2, 257});
        },
        "K not divisible by the ternary group size was accepted");
    rejects<std::invalid_argument>(
        [&] {
            (void)weight_geometry(QType::BF16, QuantLayout::TernaryPq2Block,
                                  std::array<std::uint64_t, 2>{2, 256});
        },
        "a non-ternary format was accepted in the ternary layout");

    auto buffer = ternary_payload();
    WeightParent parent{geometry, buffer.data()};
    const WeightView view{{2, 256}, {{&parent, 0, 2 * 256}}};
    const auto weight = native_weight(view);
    require(weight.qtype == QType::TERNARY_PQ2_0 && weight.layout == QuantLayout::TernaryPq2Block &&
                weight.n == 2 && weight.k == 256 && weight.group_size == 128 &&
                weight.qdata == buffer.data() &&
                weight.rotation == buffer.data() + 256 && weight.rotation_bytes == 1040 &&
                weight.payload == buffer.data() && weight.payload_bytes == 1296,
            "native ternary parent lost its planes or rotation auxiliary");
}

void ternary_payload_validation() {
    const auto geometry =
        weight_geometry(QType::TERNARY_PQ2_0, QuantLayout::TernaryPq2Block,
                        std::array<std::uint64_t, 2>{2, 256});
    auto buffer = ternary_payload();
    validate_ternary_payload(buffer.data(), geometry, "ternary"); // Accepts the well-formed parent.

    auto reject_bytes = [&](auto mutate, const char* message) {
        auto broken = ternary_payload();
        mutate(broken);
        rejects([&] { validate_ternary_payload(broken.data(), geometry, "ternary"); }, message);
    };
    reject_bytes(
        [](std::vector<std::byte>& bytes) {
            bytes[0] = std::byte{0x00};
            bytes[1] = std::byte{0x7C}; // +infinity binary16 scale.
        },
        "infinite binary16 scale accepted");
    reject_bytes(
        [](std::vector<std::byte>& bytes) {
            bytes[0] = std::byte{0x00};
            bytes[1] = std::byte{0x7E}; // NaN binary16 scale.
        },
        "NaN binary16 scale accepted");
    reject_bytes(
        [&](std::vector<std::byte>& bytes) {
            bytes[geometry.rotation_offset] = std::byte{0x00};
            bytes[geometry.rotation_offset + 1] = std::byte{0x02}; // u32 512.
        },
        "rotation block size other than 1024 accepted");
    reject_bytes(
        [&](std::vector<std::byte>& bytes) {
            auto* sign = bytes.data() + geometry.rotation_offset + kTernaryPq2RotationHeaderBytes;
            sign[0] = std::byte{0x00};
            sign[1] = std::byte{0x00};
            sign[2] = std::byte{0x00};
            sign[3] = std::byte{0x40}; // binary32 2.0, not exactly +/-1.0.
        },
        "rotation sign outside +/-1 accepted");
}

} // namespace

int main() {
    try {
        ternary_geometry_and_native_weight();
        ternary_payload_validation();
        std::cout << "ternary PQ2_0 geometry, native weight and payload checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
