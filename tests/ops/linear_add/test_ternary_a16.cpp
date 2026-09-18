// Host-only TERNARY_PQ2_0 checks for LinearAdd (P3b): registered capacity, weight
// validation, and the trit codebook / binary16 scale / FWT decode contract. Every section
// runs on the CPU; no CUDA kernel is launched.

#include "ninfer/ops/linear_add.h"
#include "ops/linear/ternary/ternary_format.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops::detail;

// Registered problems (mirror src/ops/linear_add/ternary/ternary_linear_add_config.h):
// weight [N,K] in {(5120,6144), (5120,17408), (2048,4096), (2048,6144)}.
const std::pair<std::int32_t, std::int32_t> kRegisteredShapes[] = {
    {5120, 6144}, {5120, 17408}, {2048, 4096}, {2048, 6144}};

// ---------------------------------------------------------------------------
// Shared FP64 FWT / Hadamard helpers (host-side mirrors of the pinned device
// butterfly convention in src/ops/common/fwt.cuh: low = x + y, high = x - y).
// ---------------------------------------------------------------------------

// Sylvester Hadamard matrix H_N by iterative doubling H[2n] = [[H, H], [H, -H]].
template <std::size_t N>
std::vector<std::vector<double>> sylvester() {
    std::vector<std::vector<double>> H(1, {1.0});
    std::size_t m = 1;
    while (m < N) {
        m *= 2;
        std::vector<std::vector<double>> next(2 * m, std::vector<double>(2 * m));
        for (std::size_t i = 0; i < m / 2; ++i) {
            for (std::size_t j = 0; j < m / 2; ++j) {
                next[i][j] = H[i][j];
                next[i][j + m / 2] = H[i][j];
                next[i + m / 2][j] = H[i][j];
                next[i + m / 2][j + m / 2] = -H[i][j];
            }
        }
        H = std::move(next);
    }
    return H;
}

// In-place Sylvester Walsh-Hadamard butterfly (low = x + y, high = x - y), full
// per-block form: at stage h the pairs (k, k+h) are coupled for every block.
template <std::size_t N>
void fwt_block_form(std::array<double, N>& y) {
    std::size_t h = 1;
    while (h < N) {
        for (std::size_t j = 0; j < N; j += 2 * h) {
            for (std::size_t k = j; k < j + h; ++k) {
                const double lo = y[k] + y[k + h];
                const double hi = y[k] - y[k + h];
                y[k] = lo;
                y[k + h] = hi;
            }
        }
        h *= 2;
    }
}

// Run `action` and count 0 failures when it throws std::invalid_argument, 1 otherwise.
int expect_invalid_argument(const char* label, const std::function<void()>& action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return 0;
    } catch (const std::exception& error) {
        std::cerr << label << ": threw unexpected exception: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected std::invalid_argument, no throw\n";
    return 1;
}

// ---------------------------------------------------------------------------
// Host ternary weight + buffer fixture (mirrors src/core/weight_view.cpp: the
// rotation auxiliary is 256-byte aligned after the block plane).
// ---------------------------------------------------------------------------

struct TernaryFixture {
    std::vector<std::byte> buf;
    Weight weight;
    std::uint64_t rot_offset;
    std::uint64_t rot_bytes;
};

TernaryFixture make_ternary_fixture(std::int32_t n, std::int32_t k) {
    const std::uint64_t groups   = static_cast<std::uint64_t>(k) / kTernaryPq2GroupSize;
    const std::uint64_t code_bytes =
        static_cast<std::uint64_t>(n) * groups * kTernaryPq2BlockBytes;
    const std::uint64_t rot_offset = (code_bytes + 255u) / 256u * 256u;  // aligned(code_bytes, 256)
    const std::uint64_t rot_bytes =
        kTernaryPq2RotationHeaderBytes + static_cast<std::uint64_t>(k) * kTernaryPq2SignWordBytes;

    // Zero-initialized: zero blocks (binary16 scale word 0x0000 is a finite +0) and the
    // padding between the block plane and the 256-byte-aligned rotation auxiliary.
    std::vector<std::byte> buf(rot_offset + rot_bytes);

    // 16-byte rotation header at rot_offset: [u32 1024][u8 0 transform][u8 1 gdn_v_grouped]
    // [u8 0 inverse][u8 0 reserved][u32 0][u32 0].
    auto* rot = buf.data() + rot_offset;
    const std::uint8_t header[16] = {
        0x00, 0x04, 0x00, 0x00,  // u32 block_size = 1024 (little-endian)
        0x00, 0x01, 0x00, 0x00,  // transform, gdn_v_grouped, inverse, reserved
        0x00, 0x00, 0x00, 0x00,  // u32 0
        0x00, 0x00, 0x00, 0x00,  // u32 0
    };
    std::memcpy(rot, header, sizeof(header));

    // K binary32 signs, alternating +1.0 (0x3F800000) / -1.0 (0xBF800000).
    auto* signs = reinterpret_cast<std::uint32_t*>(rot + kTernaryPq2RotationHeaderBytes);
    for (std::int32_t lane = 0; lane < k; ++lane) {
        signs[lane] = (lane % 2 == 0) ? 0x3F800000u : 0xBF800000u;
    }

    Weight w;
    w.qtype          = QType::TERNARY_PQ2_0;
    w.layout         = QuantLayout::TernaryPq2Block;
    w.group_size     = kTernaryPq2GroupSize;
    w.group          = kTernaryPq2GroupSize;
    w.ndim           = 2;
    w.n              = n;
    w.k              = k;
    w.shape[0]       = n;
    w.shape[1]       = k;
    w.shape[2]       = 1;
    w.shape[3]       = 1;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    w.padded_shape[2] = 1;
    w.padded_shape[3] = 1;
    w.payload        = buf.data();
    w.payload_bytes  = buf.size();
    w.qdata          = buf.data();
    w.rotation       = buf.data() + rot_offset;
    w.rotation_bytes = rot_bytes;
    return {std::move(buf), w, rot_offset, rot_bytes};
}

// ---------------------------------------------------------------------------
// 1. capacity_admits_registered_shapes
// ---------------------------------------------------------------------------

int capacity_admits_registered_shapes() {
    int failures = 0;
    for (auto [n, k] : kRegisteredShapes) {
        for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}) {
            const std::size_t capacity =
                ops::linear_add_workspace_capacity_bytes(QType::TERNARY_PQ2_0, n, k, policy, 1, 8);
            if (capacity != 0) {
                std::cerr << "capacity: [" << n << ',' << k << "] expected 0 transient bytes, got "
                          << capacity << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------
// 2. capacity_rejects_unregistered_shapes
// ---------------------------------------------------------------------------

int capacity_rejects_unregistered_shapes() {
    int failures = 0;
    const std::pair<std::int32_t, std::int32_t> shapes[] = {
        {5120, 4096}, {2048, 5120}, {100, 5120}, {1, 1}};
    for (auto [n, k] : shapes) {
        failures += expect_invalid_argument(
            ("capacity: unregistered [" + std::to_string(n) + ',' + std::to_string(k) +
             "] must throw").c_str(),
            [n, k]() {
                (void)ops::linear_add_workspace_capacity_bytes(QType::TERNARY_PQ2_0, n, k,
                                                               ops::LinearPolicy::A16Only, 1, 8);
            });
    }
    return failures;
}

// ---------------------------------------------------------------------------
// 3. capacity_rejects_bad_intervals
// ---------------------------------------------------------------------------

int capacity_rejects_bad_intervals() {
    int failures = 0;
    for (auto [min_tokens, max_tokens] : {std::pair{1, 0}, std::pair{5, 3}}) {
        failures += expect_invalid_argument(
            ("capacity: bad interval [" + std::to_string(min_tokens) + ',' +
             std::to_string(max_tokens) + "] must throw").c_str(),
            [min_tokens, max_tokens]() {
                (void)ops::linear_add_workspace_capacity_bytes(QType::TERNARY_PQ2_0, 5120, 6144,
                                                               ops::LinearPolicy::A16Only,
                                                               min_tokens, max_tokens);
            });
    }
    return failures;
}

// ---------------------------------------------------------------------------
// 4. weight_validation
// ---------------------------------------------------------------------------

int weight_validation() {
    int failures = 0;

    // The valid (5120, 6144) fixture must validate, with view.signs pointing just past the
    // rotation header.
    {
        auto f = make_ternary_fixture(5120, 6144);
        bool ok = true;
        try {
            const TernaryWeightView view = validate_ternary_weight(f.weight, "test");
            if (reinterpret_cast<const std::byte*>(view.signs) !=
                f.buf.data() + f.rot_offset + kTernaryPq2RotationHeaderBytes) {
                std::cerr << "weight: view.signs does not point past the rotation header\n";
                ok = false;
            }
            if (view.block_plane_bytes !=
                static_cast<std::uint64_t>(5120) *
                    (static_cast<std::uint64_t>(6144) / kTernaryPq2GroupSize) *
                    kTernaryPq2BlockBytes) {
                std::cerr << "weight: view.block_plane_bytes is wrong\n";
                ok = false;
            }
        } catch (const std::exception& error) {
            std::cerr << "weight: valid (5120,6144) fixture rejected: " << error.what() << '\n';
        }
        if (!ok) ++failures;
    }

    // Variations that must throw std::invalid_argument.
    {
        auto f = make_ternary_fixture(5120, 6144);
        f.weight.rotation = nullptr;
        failures += expect_invalid_argument("weight: rotation=nullptr must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(5120, 6144);
        f.weight.rotation_bytes = f.rot_bytes + 4;  // mismatch with the required byte count.
        failures += expect_invalid_argument("weight: rotation_bytes mismatch must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(5120, 6144);
        // Overwrite the header block_size u32 with 512 (little-endian).
        auto* rot = f.buf.data() + f.rot_offset;
        rot[0] = std::byte{0x00};
        rot[1] = std::byte{0x02};
        rot[2] = std::byte{0x00};
        rot[3] = std::byte{0x00};
        failures += expect_invalid_argument("weight: header block_size=512 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(5120, 6144);
        (f.buf.data() + f.rot_offset)[4] = std::byte{0x01};  // transform byte
        failures += expect_invalid_argument("weight: transform byte=1 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(5120, 6144);
        (f.buf.data() + f.rot_offset)[7] = std::byte{0x01};  // reserved byte
        failures += expect_invalid_argument("weight: reserved byte=1 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(5120, 6144);
        auto* signs =
            reinterpret_cast<std::uint32_t*>(f.buf.data() + f.rot_offset +
                                              kTernaryPq2RotationHeaderBytes);
        signs[0] = 0x40000000u;  // not +1.0 / -1.0
        failures += expect_invalid_argument("weight: sign word 0x40000000 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(5120, 6144);
        f.buf[0] = std::byte{0x00};
        f.buf[1] = std::byte{0x7C};  // first block scale word 0x7C00 (nonfinite)
        failures += expect_invalid_argument("weight: scale word 0x7C00 (nonfinite) must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(5120, 129);  // K must be a multiple of the 128 group size.
        failures += expect_invalid_argument("weight: k=129 (K%128!=0) must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(0, 6144);  // N must be positive.
        failures += expect_invalid_argument("weight: n=0 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    return failures;
}

// ---------------------------------------------------------------------------
// 5. decode_unit (trit codebook + binary16 scale decode, FWT convention in FP64)
// ---------------------------------------------------------------------------

int decode_unit() {
    int failures = 0;

    // scale word 0x3800 -> 0.5f exactly.
    {
        const float scale = ternary_pq2_scale_word_to_float(0x3800);
        if (scale != 0.5f) {
            std::cerr << "decode: 0x3800 -> " << scale << " (expected 0.5)\n";
            ++failures;
        }
    }

    // byte 0x64 with scale 0x3800 -> {-0.5, 0.0, +0.5, 0.0} (spec: source w3=+0.25, code 01 -> 0).
    {
        float values[4];
        ternary_pq2_decode_byte(0x64, 0x3800, values);
        const float expected[4] = {-0.5f, 0.0f, 0.5f, 0.0f};
        for (int slot = 0; slot < 4; ++slot) {
            if (values[slot] != expected[slot]) {
                std::cerr << "decode: 0x64/0x3800 slot " << slot << " = " << values[slot] << " (expected "
                          << expected[slot] << ")\n";
                ++failures;
            }
        }
    }

    // scale word 0x3C00 (1.0f): codes 0,1,2,3 one per slot -> {-1, 0, +1, +2}.
    // byte = 0b11_10_01_00 = 0xE4 (slot0=00, slot1=01, slot2=10, slot3=11, low bits first).
    {
        float values[4];
        ternary_pq2_decode_byte(0xE4, 0x3C00, values);
        const float expected[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
        for (int slot = 0; slot < 4; ++slot) {
            if (values[slot] != expected[slot]) {
                std::cerr << "decode: 0xE4/0x3C00 slot " << slot << " = " << values[slot] << " (expected "
                          << expected[slot] << ")\n";
                ++failures;
            }
        }
    }

    // scale word 0x0000 (zero) -> every slot 0.0f.
    {
        float values[4];
        ternary_pq2_decode_byte(0xE4, 0x0000, values);
        for (int slot = 0; slot < 4; ++slot) {
            if (values[slot] != 0.0f) {
                std::cerr << "decode: 0x0000 slot " << slot << " = " << values[slot]
                          << " (expected 0.0)\n";
                ++failures;
            }
        }
    }

    // scale word 0x0001 (binary16 subnormal 2^-24) with code 3 (0b11) -> +2 * 2^-24 exactly.
    {
        float values[4];
        ternary_pq2_decode_byte(0x03, 0x0001, values);  // slot 0 = code 3
        const float expected = std::ldexp(2.0f, -24);   // 2 * 2^-24 = 2^-23
        if (values[0] != expected) {
            std::cerr << "decode: 0x0001 code3 = " << values[0] << " (expected " << expected
                      << " = 2*2^-24)\n";
            ++failures;
        }
    }

    // ternary_pq2_scale_word_finite.
    {
        const std::uint16_t nonfinite[] = {0x7C00, 0x7F80, 0xFFC0};
        const std::uint16_t finite[]    = {0x0000, 0x3C00, 0x0001};
        for (const std::uint16_t word : nonfinite) {
            if (ternary_pq2_scale_word_finite(word)) {
                std::cerr << "decode: 0x" << std::hex << std::uppercase
                          << static_cast<unsigned>(word) << std::dec
                          << " must be nonfinite\n";
                ++failures;
            }
        }
        for (const std::uint16_t word : finite) {
            if (!ternary_pq2_scale_word_finite(word)) {
                std::cerr << "decode: 0x" << std::hex << std::uppercase
                          << static_cast<unsigned>(word) << std::dec
                          << " must be finite\n";
                ++failures;
            }
        }
    }

    // 1024-wide FWT: the pinned device butterfly (low = x + y, high = x - y) with the sign
    // flip and the 1/32 (1/sqrt(1024)) scale folded in must match the naive FP64 Sylvester
    // H_1024 product. Pure host FP64 math.
    {
        constexpr int N     = 1024;
        constexpr double kScale = 0.03125;  // 1/32, exact in binary.

        std::vector<double> x(N), s(N), y_in(N);
        for (int i = 0; i < N; ++i) {
            x[i] = (i * 0.125) - 63.5 + std::sin(0.37 * i);
            s[i] = (i % 3 == 0) ? -1.0 : 1.0;
            y_in[i] = s[i] * x[i];
        }

        // Reference A: naive Sylvester H_1024, ref[j] = 0.03125 * sum_k H[j][k] * (s[k]*x[k]).
        const std::vector<std::vector<double>> H = sylvester<N>();
        std::vector<double> A(N);
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k)
                acc += H[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] * y_in[k];
            A[j] = kScale * acc;
        }

        // Mirror B: in-place FP64 pinned butterfly over y = s[i]*x[i]*0.03125.
        std::array<double, N> B{};
        for (int i = 0; i < N; ++i)
            B[static_cast<std::size_t>(i)] = y_in[static_cast<std::size_t>(i)] * kScale;
        fwt_block_form(B);

        // max |A[j] - B[j]| <= 1e-9 (independent FP64 accumulation orders).
        double max_diff = 0.0;
        for (int j = 0; j < N; ++j) {
            max_diff = std::max(max_diff, std::fabs(A[static_cast<std::size_t>(j)] -
                                                     B[static_cast<std::size_t>(j)]));
        }
        if (max_diff > 1e-9) {
            std::cerr << "fwt: max |A-B| = " << max_diff << " > 1e-9\n";
            ++failures;
        }

        // 1/32 normalization: the orthogonal transform preserves energy, so
        // sum_j A[j]^2 == sum_i (s[i]*x[i])^2.
        double sum_a2 = 0.0, sum_in2 = 0.0;
        for (int j = 0; j < N; ++j)
            sum_a2 += A[static_cast<std::size_t>(j)] * A[static_cast<std::size_t>(j)];
        for (int i = 0; i < N; ++i)
            sum_in2 += y_in[static_cast<std::size_t>(i)] * y_in[static_cast<std::size_t>(i)];
        if (std::fabs(sum_a2 - sum_in2) > 1e-6 * sum_in2) {
            std::cerr << "fwt: 1/32 normalization not preserved (|sumA^2 - sumin^2| = "
                      << std::fabs(sum_a2 - sum_in2) << ")\n";
            ++failures;
        }

        // Small-case convention pin: x4 = {0.5, -1.25, 3.0, 0.125}, all signs +1, no scaling.
        // After the raw butterfly (h = 1, 2) the output is dyadic-exact.
        const double a = 0.5, b = -1.25, c = 3.0, d = 0.125;
        std::array<double, 4> y{{a, b, c, d}};
        fwt_block_form(y);
        const double expected[4] = {a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d};
        for (int i = 0; i < 4; ++i) {
            if (y[static_cast<std::size_t>(i)] != expected[i]) {
                std::cerr << "fwt: small-case slot " << i << " = " << y[static_cast<std::size_t>(i)]
                          << " (expected " << expected[i] << ")\n";
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        int failures = 0;
        failures += capacity_admits_registered_shapes();
        failures += capacity_rejects_unregistered_shapes();
        failures += capacity_rejects_bad_intervals();
        failures += weight_validation();
        failures += decode_unit();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TERNARY_PQ2_0 LinearAdd (host)\n";
        return failures;
    } catch (const std::exception& error) {
        std::cerr << "TERNARY_PQ2_0 LinearAdd (host): " << error.what() << '\n';
        return 1;
    }
}
