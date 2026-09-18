#include "ninfer/ops/linear.h"
#include "ops/linear/ternary/ternary_config.h"
#include "ops/linear/ternary/ternary_format.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops::detail;

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

// A Weight with correct ternary metadata and an all-zero dummy payload (not a valid ternary
// payload: the rotation header block size reads 0, not 1024). Mirrors make_ternary_fixture's
// geometry so every metadata field is correct and the pointers are 16-byte aligned.
struct DummyTernary {
    std::vector<std::byte> buf;
    Weight weight;
    std::uint64_t rot_offset;
};

DummyTernary make_dummy_ternary_weight(std::int32_t n, std::int32_t k) {
    const std::uint64_t groups   = static_cast<std::uint64_t>(k) / kTernaryPq2GroupSize;
    const std::uint64_t code_bytes =
        static_cast<std::uint64_t>(n) * groups * kTernaryPq2BlockBytes;
    const std::uint64_t rot_offset = (code_bytes + 255u) / 256u * 256u;  // aligned(code_bytes, 256)
    const std::uint64_t rot_bytes =
        kTernaryPq2RotationHeaderBytes + static_cast<std::uint64_t>(k) * kTernaryPq2SignWordBytes;
    // All-zero: every block scale word is 0x0000 (finite +0) and the rotation header block
    // size is 0 (invalid: must be 1024).
    std::vector<std::byte> buf(rot_offset + rot_bytes, std::byte{0});

    Weight w;
    w.qtype           = QType::TERNARY_PQ2_0;
    w.layout          = QuantLayout::TernaryPq2Block;
    w.group_size      = kTernaryPq2GroupSize;
    w.group           = kTernaryPq2GroupSize;
    w.ndim            = 2;
    w.n               = n;
    w.k               = k;
    w.shape[0]        = n;
    w.shape[1]        = k;
    w.shape[2]        = 1;
    w.shape[3]        = 1;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    w.padded_shape[2] = 1;
    w.padded_shape[3] = 1;
    w.payload         = buf.data();
    w.payload_bytes   = buf.size();
    w.qdata           = buf.data();
    w.rotation        = buf.data() + rot_offset;
    w.rotation_bytes  = rot_bytes;
    return {std::move(buf), w, rot_offset};
}

// ---------------------------------------------------------------------------
// a. capacity_admits_registered_shapes
// ---------------------------------------------------------------------------

int capacity_admits_registered_shapes() {
    int failures = 0;
    const std::pair<std::int32_t, std::int32_t> shapes[] = {
        {14336, 5120}, {16384, 5120}, {34816, 5120}, {248320, 5120}, {5120, 6144}, {5120, 17408}};
    for (auto [n, k] : shapes) {
        for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}) {
            const std::size_t capacity =
                ops::linear_workspace_capacity_bytes(QType::TERNARY_PQ2_0, n, k, policy, 1, 8);
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
// b. capacity_rejects_unregistered_shapes
// ---------------------------------------------------------------------------

int capacity_rejects_unregistered_shapes() {
    int failures = 0;
    const std::pair<std::int32_t, std::int32_t> shapes[] = {
        {14336, 5121}, {100, 5120}, {248320, 512}, {1, 1}};
    for (auto [n, k] : shapes) {
        failures += expect_invalid_argument(
            ("capacity: unregistered [" + std::to_string(n) + ',' + std::to_string(k) +
             "] must throw").c_str(),
            [n, k]() {
                (void)ops::linear_workspace_capacity_bytes(QType::TERNARY_PQ2_0, n, k,
                                                           ops::LinearPolicy::A16Only, 1, 8);
            });
    }
    return failures;
}

// ---------------------------------------------------------------------------
// c. capacity_rejects_bad_intervals
// ---------------------------------------------------------------------------

int capacity_rejects_bad_intervals() {
    int failures = 0;
    for (auto [min_tokens, max_tokens] : {std::pair{1, 0}, std::pair{5, 3}}) {
        failures += expect_invalid_argument(
            ("capacity: bad interval [" + std::to_string(min_tokens) + ',' +
             std::to_string(max_tokens) + "] must throw").c_str(),
            [min_tokens, max_tokens]() {
                (void)ops::linear_workspace_capacity_bytes(QType::TERNARY_PQ2_0, 14336, 5120,
                                                           ops::LinearPolicy::A16Only, min_tokens,
                                                           max_tokens);
            });
    }
    return failures;
}

// ---------------------------------------------------------------------------
// d. weight_validation
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
// d2. metadata_validator_device_safe
//
// A Weight with correct ternary metadata and an all-zero dummy payload (rotation header
// block size 0, not 1024) must PASS the metadata validators (they never dereference
// qdata/rotation bytes) and FAIL the full host-side validators (the content scan reads the
// rotation header and finds block size 0). This is the host-side equivalent of the CUDA
// case, where the payload is device-resident and the content scan SIGSEGVs: the metadata
// validators are the only runtime-safe checks.
// ---------------------------------------------------------------------------

int metadata_validator_device_safe() {
    int failures = 0;
    auto dummy = make_dummy_ternary_weight(5120, 6144);

    // The weight metadata validator succeeds: the dummy content is invalid (rotation block
    // size 0), so any dereference of qdata/rotation bytes would throw; it did not. The view
    // pointers are computed, not read, so they must alias the dummy bytes.
    {
        bool ok = true;
        try {
            const TernaryWeightView view = validate_ternary_weight_metadata(dummy.weight, "test");
            if (reinterpret_cast<const std::byte*>(view.blocks) != dummy.buf.data() ||
                reinterpret_cast<const std::byte*>(view.signs) !=
                    dummy.buf.data() + dummy.rot_offset + kTernaryPq2RotationHeaderBytes) {
                std::cerr << "device-safe: view pointers do not alias the dummy bytes\n";
                ok = false;
            }
        } catch (const std::exception& error) {
            std::cerr << "device-safe: weight metadata validator threw on a content-invalid "
                         "dummy: " << error.what() << '\n';
            ok = false;
        }
        if (!ok) ++failures;
    }

    // The row-view metadata validator also succeeds on a 16-byte row-aligned qdata offset.
    {
        bool ok = true;
        try {
            Weight row = dummy.weight;
            row.qdata = dummy.buf.data() + 16;  // a 16-aligned offset into the dummy payload
            (void)validate_ternary_row_view_metadata(row, "test");
        } catch (const std::exception& error) {
            std::cerr << "device-safe: row-view metadata validator threw on a dummy: "
                      << error.what() << '\n';
            ok = false;
        }
        if (!ok) ++failures;
    }

    // The full host-side validator FAILS on the same dummy: the content scan reads the
    // rotation header and finds block size 0 (not 1024). This proves the content scan is
    // the part that dereferences qdata/rotation (and SIGSEGVs on a device payload).
    failures += expect_invalid_argument(
        "device-safe: full validator must throw on the dummy",
        [&dummy]() { (void)validate_ternary_weight(dummy.weight, "t"); });

    // The full row-view host-side validator fails too (same content scan).
    {
        Weight row = dummy.weight;
        row.qdata = dummy.buf.data() + 16;
        failures += expect_invalid_argument(
            "device-safe: full row-view validator must throw on the dummy",
            [&row]() { (void)validate_ternary_row_view(row, "t"); });
    }

    return failures;
}

// ---------------------------------------------------------------------------
// e. decode_worked_example (spec 2.1 + codebook)
// ---------------------------------------------------------------------------

int decode_worked_example() {
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
                std::cerr << "decode: 0x0000 slot " << slot << " = " << values[slot] << " (expected 0.0)\n";
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
    return failures;
}

// ---------------------------------------------------------------------------
// f. fwt_convention_and_normalization (FP64, the pinned fwt.cuh convention)
// ---------------------------------------------------------------------------

int fwt_convention_and_normalization() {
    int failures = 0;
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
        for (int k = 0; k < N; ++k) acc += H[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] * y_in[k];
        A[j] = kScale * acc;
    }

    // Mirror B: in-place FP64 pinned butterfly over y = s[i]*x[i]*0.03125.
    std::array<double, N> B{};
    for (int i = 0; i < N; ++i) B[static_cast<std::size_t>(i)] = y_in[static_cast<std::size_t>(i)] * kScale;
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
    for (int j = 0; j < N; ++j) sum_a2 += A[static_cast<std::size_t>(j)] * A[static_cast<std::size_t>(j)];
    for (int i = 0; i < N; ++i) sum_in2 += y_in[static_cast<std::size_t>(i)] * y_in[static_cast<std::size_t>(i)];
    if (std::fabs(sum_a2 - sum_in2) > 1e-6 * sum_in2) {
        std::cerr << "fwt: 1/32 normalization not preserved (|sumA^2 - sumin^2| = "
                  << std::fabs(sum_a2 - sum_in2) << ")\n";
        ++failures;
    }

    // Small-case convention pin: x4 = {0.5, -1.25, 3.0, 0.125}, all signs +1, no scaling.
    // After the raw butterfly (h = 1, 2) the output is dyadic-exact.
    {
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

    // Sign-flip case: same x4 with signs {-1, +1, -1, +1} and the 1/32 scale. Compare the
    // raw butterfly + 1/32 on y = s*x against the naive matrix with the signs applied (exact).
    {
        const double a = 0.5, b = -1.25, c = 3.0, d = 0.125;
        const double xv[4] = {a, b, c, d};
        const double ss[4] = {-1.0, 1.0, -1.0, 1.0};
        double yin[4];
        for (int k = 0; k < 4; ++k) yin[k] = ss[k] * xv[k];
        std::array<double, 4> y{{yin[0], yin[1], yin[2], yin[3]}};
        fwt_block_form(y);
        const std::vector<std::vector<double>> H4 = sylvester<4>();
        for (int j = 0; j < 4; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 4; ++k) acc += H4[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] * yin[k];
            const double expected = kScale * acc;
            if (y[static_cast<std::size_t>(j)] * kScale != expected) {
                std::cerr << "fwt: sign-flip slot " << j << " = " << y[static_cast<std::size_t>(j)] * kScale
                          << " (expected " << expected << ")\n";
                ++failures;
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------
// g. cross_route_consistency (host mirror of the shared pinned pipeline)
// ---------------------------------------------------------------------------
//
// For a K=5120 row the gemv (T=1), simt (T=4), and mma (T=33) routes share one
// pinned pipeline: the exact FP32 trit decode ((code - 1) * binary16 scale, exact
// in binary32), the pinned FP32 butterfly FWT (the fused sign flip + 1/32 load and
// the operation order of ternary_fwt_butterfly in fwt.cuh), and the contiguous
// packed plane (one 1024 K block is 8 groups * 34 bytes). The mirrors below
// reproduce each route's byte addressing verbatim (including the post-P3a stride
// expression) so a stride regression in any route breaks the bit-identity
// assertion; the route-private final contraction orders are pinned separately and
// are not part of this shared pipeline.

using TernaryK5120Geometry = TernaryN14336K5120;  // K-driven constants only (N is unused).

// The pinned transform constants (fwt.cuh is a device header; the mirror uses the values
// directly).
constexpr int kTernaryFwtBlock = 1024;
constexpr float kTernaryFwtScale = 0.03125F;  // 1/sqrt(1024), exact in binary.

// FP32 mirror of the fused FWT load: x * sign * (1/32), the kernel's left-associative
// binary32 order.
inline float fused_fwt_load(float value, float sign) { return value * sign * kTernaryFwtScale; }

// Host mirror of ternary_fwt_butterfly (fwt.cuh) with the same per-element FP32
// operation order: 256 lanes x 4 elements, within-warp stages (h < 32),
// across-warp stages (32 <= h < 256), above-block stages (h >= 256).
void fwt_butterfly_f32(std::array<float, kTernaryFwtBlock>& row) {
    constexpr int kThreads = 256;
    constexpr int kElements = kTernaryFwtBlock / kThreads;
    std::array<std::array<float, kElements>, kThreads> reg;
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kElements; ++i) { reg[t][i] = row[static_cast<std::size_t>(i) * kThreads + t]; }
    }
    for (int h = 1; h < 32; h *= 2) {
        // __shfl_xor_sync reads the pre-stage values of every lane atomically; mirror it
        // with a per-stage snapshot.
        std::array<std::array<float, kElements>, kThreads> pre = reg;
        for (int t = 0; t < kThreads; ++t) {
            const int lane    = t & 31;
            const int partner = (t & ~31) + (lane ^ h);  // __shfl_xor_sync partner
            for (int j = 0; j < kElements; ++j) {
                const float val  = reg[t][j];
                const float val2 = pre[partner][j];
                reg[t][j]        = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }
    for (int h = 32; h < kThreads; h *= 2) {
        std::array<std::array<float, kThreads>, kElements> shared;
        for (int t = 0; t < kThreads; ++t) {
            for (int j = 0; j < kElements; ++j) { shared[j][t] = reg[t][j]; }
        }
        for (int t = 0; t < kThreads; ++t) {
            for (int j = 0; j < kElements; ++j) {
                const float val  = reg[t][j];
                const float val2 = shared[j][static_cast<std::size_t>(t ^ h)];
                reg[t][j]        = (t & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }
    for (int h = kThreads; h < kTernaryFwtBlock; h *= 2) {
        const int step = h / kThreads;
        for (int t = 0; t < kThreads; ++t) {
            for (int j = 0; j < kElements; j += 2 * step) {
                for (int k = 0; k < step; ++k) {
                    const float x = reg[t][j + k];
                    const float y = reg[t][j + k + step];
                    reg[t][j + k]         = x + y;
                    reg[t][j + k + step]  = x - y;
                }
            }
        }
    }
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kElements; ++i) { row[static_cast<std::size_t>(i) * kThreads + t] = reg[t][i]; }
    }
}

// Verbatim mirror of the ternary_gemv_kernel / ternary_simt_kernel per-row K-block
// addressing (post-P3a fix: k_block * kGroupsPerKBlock * kTernaryPq2BlockBytes).
const std::uint8_t* gemv_simt_row_block(const std::uint8_t* blocks, int groups_per_row, int row,
                                        int k_block, int group) {
    const std::uint8_t* row_blocks =
        blocks + row * groups_per_row * kTernaryPq2BlockBytes +
        k_block * TernaryK5120Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;
    return row_blocks + group * kTernaryPq2BlockBytes;
}

// Verbatim mirror of the ternary_mma_kernel per-row K-block addressing (per-row base
// plus global group index b * kGroupsPerKBlock + group).
const std::uint8_t* mma_row_block(const std::uint8_t* blocks, int groups_per_row, int row,
                                  int k_block, int group) {
    const std::uint8_t* b_base = blocks + row * groups_per_row * kTernaryPq2BlockBytes;
    const int global_group      = k_block * TernaryK5120Geometry::kGroupsPerKBlock + group;
    return b_base + global_group * kTernaryPq2BlockBytes;
}

// BF16 helpers (round-to-nearest-even, the op_tester.h convention).
float bf16_to_f32(std::uint16_t h) {
    std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) { return std::uint16_t((u >> 16) | 0x0040u); }
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return std::uint16_t(u >> 16);
}

double bf16_to_f64(std::uint16_t h) { return static_cast<double>(bf16_to_f32(h)); }

// One route output for (row, token): decode every 34-byte group of the K block
// exactly ((code - 1) * binary16 scale in binary32), run the pinned FP32 butterfly
// over the fused-loaded row, and accumulate the canonical FP32 FMA chain.
template <class RowBlock>
float route_output(const std::uint8_t* blocks, const float* signs, const std::uint16_t* x_row,
                   int groups_per_row, int row, int token, RowBlock&& row_block) {
    float total = 0.0F;
    for (int k_block = 0; k_block < TernaryK5120Geometry::kKBlocks; ++k_block) {
        const float* sign_block = signs + k_block * kTernaryFwtBlock;
        std::array<float, kTernaryFwtBlock> fwt_row;
        for (int idx = 0; idx < kTernaryFwtBlock; ++idx) {
            fwt_row[static_cast<std::size_t>(idx)] = fused_fwt_load(
                bf16_to_f32(x_row[static_cast<std::size_t>(token) * TernaryK5120Geometry::kInputRows +
                                  k_block * kTernaryFwtBlock + idx]),
                sign_block[idx]);
        }
        fwt_butterfly_f32(fwt_row);
        for (int group = 0; group < TernaryK5120Geometry::kGroupsPerKBlock; ++group) {
            const std::uint8_t* block =
                row_block(blocks, groups_per_row, row, k_block, group);
            const std::uint16_t scale_word =
                static_cast<std::uint16_t>(block[0]) | (static_cast<std::uint16_t>(block[1]) << 8);
            const float scale = ternary_pq2_scale_word_to_float(scale_word);
            const int group0  = group * kTernaryPq2GroupSize;
            for (int lane = 0; lane < kTernaryPq2GroupSize; ++lane) {
                const int code = ternary_pq2_code(block[2 + lane / 4], lane % 4);
                const float weight = static_cast<float>(code - 1) * scale;
                const int feature  = group0 + lane;  // local index into the 1024 block row
                total = fmaf(weight, fwt_row[static_cast<std::size_t>(feature)], total);
            }
        }
    }
    return total;
}

// Deterministic trit + scale fill of the block plane (non-zero so a misaddressed K
// block changes the result).
void fill_ternary_plane(std::byte* plane, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> trit(0, 3);
    // Finite binary16 scale grid: normal + one subnormal word.
    const std::uint16_t scales[] = {0x3C00, 0x3800, 0x4000, 0x3A00, 0x3000, 0x0001};
    const std::int32_t groups_per_row = k / kTernaryPq2GroupSize;
    for (std::int32_t r = 0; r < n; ++r) {
        for (std::int32_t group = 0; group < groups_per_row; ++group) {
            auto* block = plane +
                          static_cast<std::uint64_t>(r * groups_per_row + group) * kTernaryPq2BlockBytes;
            const std::uint16_t scale = scales[(r * 31 + group) % 6];
            block[0] = std::byte(scale & 0xFF);
            block[1] = std::byte(scale >> 8);
            for (int b = 0; b < 32; ++b) {
                std::uint8_t byte = 0;
                for (int slot = 0; slot < 4; ++slot) { byte |= std::uint8_t(trit(g) << (2 * slot)); }
                block[2 + b] = std::byte(byte);
            }
        }
    }
}

int cross_route_consistency() {
    int failures = 0;
    constexpr std::int32_t kRows         = 128;
    constexpr std::int32_t kTokens       = 33;  // the mma route's T
    constexpr std::int32_t kSimtTokens   = 4;   // the simt route's T
    constexpr std::int32_t kGroupsPerRow =
        TernaryK5120Geometry::kInputRows / kTernaryPq2GroupSize;

    auto fixture = make_ternary_fixture(kRows, TernaryK5120Geometry::kInputRows);
    fill_ternary_plane(fixture.buf.data(), kRows, TernaryK5120Geometry::kInputRows, 0x9E3779B9u);
    const auto* blocks = reinterpret_cast<const std::uint8_t*>(fixture.buf.data());
    const auto* signs  = reinterpret_cast<const float*>(fixture.buf.data() + fixture.rot_offset +
                                                         kTernaryPq2RotationHeaderBytes);

    // Deterministic BF16 activations: the exact values the kernels read.
    std::vector<std::uint16_t> x_bf16(
        static_cast<std::size_t>(kTokens) * TernaryK5120Geometry::kInputRows);
    {
        std::mt19937 g(0x1234ABCDu);
        std::uniform_real_distribution<float> d(-4.0F, 4.0F);
        for (auto& word : x_bf16) { word = f32_to_bf16(d(g)); }
    }

    // Route outputs: gemv at T=1 (token 0), simt at T=4 (tokens 0..3), mma at T=33
    // (tokens 0..32), each over every row.
    std::vector<float> gemv(static_cast<std::size_t>(kRows));
    std::vector<float> simt(static_cast<std::size_t>(kRows * kSimtTokens));
    std::vector<float> mma(static_cast<std::size_t>(kRows * kTokens));
    for (std::int32_t r = 0; r < kRows; ++r) {
        gemv[static_cast<std::size_t>(r)] =
            route_output(blocks, signs, x_bf16.data(), kGroupsPerRow, r, 0,
                        [](const std::uint8_t* b, int gpr, int row, int kb, int group) {
                            return gemv_simt_row_block(b, gpr, row, kb, group);
                        });
        for (std::int32_t t = 0; t < kSimtTokens; ++t) {
            simt[static_cast<std::size_t>(r * kSimtTokens + t)] =
                route_output(blocks, signs, x_bf16.data(), kGroupsPerRow, r, t,
                            [](const std::uint8_t* b, int gpr, int row, int kb, int group) {
                                return gemv_simt_row_block(b, gpr, row, kb, group);
                            });
        }
        for (std::int32_t t = 0; t < kTokens; ++t) {
            mma[static_cast<std::size_t>(r * kTokens + t)] =
                route_output(blocks, signs, x_bf16.data(), kGroupsPerRow, r, t,
                            [](const std::uint8_t* b, int gpr, int row, int kb, int group) {
                                return mma_row_block(b, gpr, row, kb, group);
                            });
        }
    }

    // Bit-identity across the routes at every overlapping token (FP32 bit patterns). The gemv
    // route only produces token 0 (T=1); simt and mma overlap on tokens 0..3.
    for (std::int32_t r = 0; r < kRows; ++r) {
        const std::uint32_t g =
            *reinterpret_cast<const std::uint32_t*>(&gemv[static_cast<std::size_t>(r)]);
        const std::uint32_t s0 =
            *reinterpret_cast<const std::uint32_t*>(
                &simt[static_cast<std::size_t>(r * kSimtTokens)]);
        const std::uint32_t m0 =
            *reinterpret_cast<const std::uint32_t*>(&mma[static_cast<std::size_t>(r * kTokens)]);
        if (g != s0 || s0 != m0) {
            std::cerr << "cross-route: row " << r << " token 0 is not bit-identical across the "
                         "routes\n";
            ++failures;
        }
        for (std::int32_t t = 1; t < kSimtTokens; ++t) {
            const std::uint32_t s =
                *reinterpret_cast<const std::uint32_t*>(
                    &simt[static_cast<std::size_t>(r * kSimtTokens + t)]);
            const std::uint32_t m =
                *reinterpret_cast<const std::uint32_t*>(
                    &mma[static_cast<std::size_t>(r * kTokens + t)]);
            if (s != m) {
                std::cerr << "cross-route: row " << r << " token " << t << " is not bit-identical "
                             "across the simt and mma routes\n";
                ++failures;
            }
        }
    }

    // The pre-P3a 2048-byte stride must be detectable: a mirror reading
    // k_block * 2048 diverges from the fixed expression on the random plane.
    {
        float buggy_total = 0.0F;
        for (int k_block = 0; k_block < TernaryK5120Geometry::kKBlocks; ++k_block) {
            const float* sign_block = signs + k_block * kTernaryFwtBlock;
            std::array<float, kTernaryFwtBlock> fwt_row;
            for (int idx = 0; idx < kTernaryFwtBlock; ++idx) {
                fwt_row[static_cast<std::size_t>(idx)] = fused_fwt_load(
                    bf16_to_f32(x_bf16[static_cast<std::size_t>(k_block) * kTernaryFwtBlock + idx]),
                    sign_block[idx]);
            }
            fwt_butterfly_f32(fwt_row);
            for (int group = 0; group < TernaryK5120Geometry::kGroupsPerKBlock; ++group) {
                const std::uint8_t* block = blocks +
                                            k_block * 2048 + group * kTernaryPq2BlockBytes;
                const std::uint16_t scale_word =
                    static_cast<std::uint16_t>(block[0]) |
                    (static_cast<std::uint16_t>(block[1]) << 8);
                const float scale = ternary_pq2_scale_word_to_float(scale_word);
                const int group0  = group * kTernaryPq2GroupSize;
                for (int lane = 0; lane < kTernaryPq2GroupSize; ++lane) {
                    const int code = ternary_pq2_code(block[2 + lane / 4], lane % 4);
                    buggy_total = fmaf(
                        static_cast<float>(code - 1) * scale,
                        fwt_row[static_cast<std::size_t>(group0 + lane)],
                        buggy_total);
                }
            }
        }
        if (buggy_total == gemv[0]) {
            std::cerr << "cross-route: the 2048-byte stride mirror is not detectable\n";
            ++failures;
        }
    }

    // FP64 oracle for a row subset: the naive Sylvester H_1024 (the pinned
    // convention, the 1/32 scale exact) over the exact-decoded weights. Criterion for the FP32
    // pipeline (exact trit decode, FP32 FWT, the 5120-term FP32 FMA chain): a relative 1e-3
    // bound with a finite gross pointwise cap of 1e-3 (op-development.md 6.3), so cancellation
    // does not require strict agreement. Observed against the fixed fixture: max relative about
    // 2e-6, max absolute about 1e-3.
    const std::vector<std::vector<double>> H = sylvester<1024>();
    for (std::int32_t r = 0; r < 16; ++r) {
        for (std::int32_t t = 0; t < kTokens; ++t) {
            double reference = 0.0;
            for (int k_block = 0; k_block < TernaryK5120Geometry::kKBlocks; ++k_block) {
                const std::uint8_t* block0 =
                    blocks + r * kGroupsPerRow * kTernaryPq2BlockBytes +
                    k_block * TernaryK5120Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;
                std::vector<double> fwt64(1024);
                for (int j = 0; j < 1024; ++j) {
                    double acc = 0.0;
                    for (int k = 0; k < 1024; ++k) {
                        const double x = bf16_to_f64(
                            x_bf16[static_cast<std::size_t>(t) *
                                      TernaryK5120Geometry::kInputRows +
                                  k_block * kTernaryFwtBlock + k]);
                        const double s = reinterpret_cast<const std::uint32_t*>(signs)[k_block *
                                                                                      kTernaryFwtBlock +
                                                                                  k] ==
                                             0x3F800000u
                             ? 1.0
                             : -1.0;
                        acc += static_cast<double>(
                                  H[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]) *
                               s * x;
                    }
                    fwt64[static_cast<std::size_t>(j)] = 0.03125 * acc;
                }
                for (int group = 0; group < TernaryK5120Geometry::kGroupsPerKBlock; ++group) {
                    const std::uint8_t* block = block0 + group * kTernaryPq2BlockBytes;
                    const std::uint16_t scale_word =
                        static_cast<std::uint16_t>(block[0]) |
                        (static_cast<std::uint16_t>(block[1]) << 8);
                    const double scale =
                        static_cast<double>(ternary_pq2_scale_word_to_float(scale_word));
                    const int group0 = group * kTernaryPq2GroupSize;
                    for (int lane = 0; lane < kTernaryPq2GroupSize; ++lane) {
                        const int code = ternary_pq2_code(block[2 + lane / 4], lane % 4);
                        reference += static_cast<double>(code - 1) * scale *
                                     fwt64[static_cast<std::size_t>(group0 + lane)];
                    }
                }
            }
            const float got = t == 0
                                  ? gemv[static_cast<std::size_t>(r)]
                                  : t < kSimtTokens
                                        ? simt[static_cast<std::size_t>(r * kSimtTokens + t)]
                                        : mma[static_cast<std::size_t>(r * kTokens + t)];
            const double bound = 1e-3 * std::max(1.0, std::fabs(reference));
            if (std::fabs(static_cast<double>(got) - reference) > bound) {
                std::cerr << "cross-route: FP32 pipeline row " << r << " token " << t << " is "
                          << std::fabs(static_cast<double>(got) - reference)
                          << " from the FP64 oracle\n";
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
        failures += metadata_validator_device_safe();
        failures += decode_worked_example();
        failures += fwt_convention_and_normalization();
        failures += cross_route_consistency();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TERNARY_PQ2_0 Linear (host)\n";
        return failures;
    } catch (const std::exception& error) {
        std::cerr << "TERNARY_PQ2_0 Linear (host): " << error.what() << '\n';
        return 1;
    }
}
