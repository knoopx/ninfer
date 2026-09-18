#include "core/weight.h"
#include "ninfer/ops/gdn_input_proj.h"
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

// Pinned FWT block geometry and the 1/sqrt(1024) normalization, mirroring
// src/ops/common/fwt.cuh (kTernaryFwtBlock = 1024, kTernaryFwtScale = 0.03125F, exact in
// binary). Defined locally because fwt.cuh is a device header the host test does not
// include.
constexpr std::int32_t kTernaryFwtBlock = 1024;
constexpr float kTernaryFwtScale        = 0.03125F;

using Geometry = TernaryN16384K5120;  // K-driven constants only (N is unused).

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
    const std::uint64_t groups     = static_cast<std::uint64_t>(k) / kTernaryPq2GroupSize;
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
    w.qtype            = QType::TERNARY_PQ2_0;
    w.layout           = QuantLayout::TernaryPq2Block;
    w.group_size       = kTernaryPq2GroupSize;
    w.group            = kTernaryPq2GroupSize;
    w.ndim             = 2;
    w.n                = n;
    w.k                = k;
    w.shape[0]         = n;
    w.shape[1]         = k;
    w.shape[2]         = 1;
    w.shape[3]         = 1;
    w.padded_shape[0]  = n;
    w.padded_shape[1]  = k;
    w.padded_shape[2]  = 1;
    w.padded_shape[3]  = 1;
    w.payload          = buf.data();
    w.payload_bytes    = buf.size();
    w.qdata            = buf.data();
    w.rotation         = buf.data() + rot_offset;
    w.rotation_bytes   = rot_bytes;
    return {std::move(buf), w, rot_offset, rot_bytes};
}

// ---------------------------------------------------------------------------
// Host mirrors of the pinned ternary pipeline: the fused FP32 load
// (value * sign * 1/32, the kernel's left-associative binary32 order) and the
// FP32 butterfly of ternary_fwt_butterfly (256 lanes x 4 elements,
// within-warp / across-warp / above-block stages, exact FP32 operation order).
// ---------------------------------------------------------------------------

inline float fused_fwt_load(float value, float sign) { return value * sign * kTernaryFwtScale; }

void fwt_butterfly_f32(std::array<float, kTernaryFwtBlock>& row) {
    constexpr int kThreads  = 256;
    constexpr int kElements = kTernaryFwtBlock / kThreads;
    std::array<std::array<float, kElements>, kThreads> reg;
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kElements; ++i) { reg[t][i] = row[static_cast<std::size_t>(i) * kThreads + t]; }
    }
    for (int h = 1; h < 32; h *= 2) {
        // The __shfl_xor_sync stage reads every lane's pre-stage value atomically; a snapshot
        // keeps the sequential mirror's partner reads on the original values.
        const std::array<std::array<float, kElements>, kThreads> snap = reg;
        for (int t = 0; t < kThreads; ++t) {
            const int lane    = t & 31;
            const int partner = (t & ~31) + (lane ^ h);  // __shfl_xor_sync partner
            for (int j = 0; j < kElements; ++j) {
                const float val  = snap[t][j];
                const float val2 = snap[partner][j];
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

// Deterministic trit + scale fill of the block plane (finite binary16 scale grid, one per
// (row * 31 + group) % 6, with random 2-bit trits; non-zero so a misaddressed K block
// changes the result).
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

// ---------------------------------------------------------------------------
// a. capacity_admits
// ---------------------------------------------------------------------------

int capacity_admits() {
    int failures = 0;
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                        ops::LinearPolicy::AllowA4}) {
        // Every policy resolves to the A16 routes, which need no transient storage.
        const std::size_t capacity =
            ops::gdn_input_proj_workspace_capacity_bytes(QType::TERNARY_PQ2_0, 16384, 5120, policy,
                                                         1, 8);
        if (capacity != 0) {
            std::cerr << "capacity: [16384,5120] policy " << static_cast<int>(policy)
                      << " expected 0 transient bytes, got " << capacity << '\n';
            ++failures;
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------
// b. capacity_rejects
// ---------------------------------------------------------------------------

int capacity_rejects() {
    int failures = 0;

    // Unregistered row profiles.
    const std::pair<std::int32_t, std::int32_t> shapes[] = {{16384, 5121}, {14336, 5120}, {1, 1}};
    for (auto [n, k] : shapes) {
        failures += expect_invalid_argument(
            ("capacity: unregistered [" + std::to_string(n) + ',' + std::to_string(k) +
             "] must throw").c_str(),
            [n, k]() {
                (void)ops::gdn_input_proj_workspace_capacity_bytes(QType::TERNARY_PQ2_0, n, k,
                                                                   ops::LinearPolicy::A16Only, 1, 8);
            });
    }

    // Bad token intervals on the registered profile.
    for (auto [min_tokens, max_tokens] : {std::pair{1, 0}, std::pair{5, 3}}) {
        failures += expect_invalid_argument(
            ("capacity: bad interval [" + std::to_string(min_tokens) + ',' +
             std::to_string(max_tokens) + "] must throw").c_str(),
            [min_tokens, max_tokens]() {
                (void)ops::gdn_input_proj_workspace_capacity_bytes(QType::TERNARY_PQ2_0, 16384, 5120,
                                                                   ops::LinearPolicy::A16Only,
                                                                   min_tokens, max_tokens);
            });
    }
    return failures;
}

// ---------------------------------------------------------------------------
// c. weight_validation
// ---------------------------------------------------------------------------

int weight_validation() {
    int failures = 0;

    // The valid (16384, 5120) fixture must validate, with view.signs pointing just past the
    // rotation header.
    {
        auto f = make_ternary_fixture(16384, 5120);
        bool ok = true;
        try {
            const TernaryWeightView view = validate_ternary_weight(f.weight, "test");
            if (reinterpret_cast<const std::byte*>(view.signs) !=
                f.buf.data() + f.rot_offset + kTernaryPq2RotationHeaderBytes) {
                std::cerr << "weight: view.signs does not point past the rotation header\n";
                ok = false;
            }
        } catch (const std::exception& error) {
            std::cerr << "weight: valid (16384,5120) fixture rejected: " << error.what() << '\n';
        }
        if (!ok) ++failures;
    }

    // Variations that must throw std::invalid_argument.
    {
        auto f = make_ternary_fixture(16384, 5120);
        f.weight.rotation = nullptr;
        failures += expect_invalid_argument("weight: rotation=nullptr must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(16384, 5120);
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
        auto f = make_ternary_fixture(16384, 5120);
        (f.buf.data() + f.rot_offset)[4] = std::byte{0x01};  // transform byte
        failures += expect_invalid_argument("weight: transform byte=1 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(16384, 5120);
        (f.buf.data() + f.rot_offset)[7] = std::byte{0x01};  // reserved byte
        failures += expect_invalid_argument("weight: reserved byte=1 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(16384, 5120);
        auto* signs =
            reinterpret_cast<std::uint32_t*>(f.buf.data() + f.rot_offset +
                                              kTernaryPq2RotationHeaderBytes);
        signs[0] = 0x40000000u;  // not +1.0 / -1.0
        failures += expect_invalid_argument("weight: sign word 0x40000000 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(16384, 5120);
        f.buf[0] = std::byte{0x00};
        f.buf[1] = std::byte{0x7C};  // first block scale word 0x7C00 (nonfinite)
        failures += expect_invalid_argument("weight: scale word 0x7C00 (nonfinite) must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(16384, 129);  // K must be a multiple of the 128 group size.
        failures += expect_invalid_argument("weight: k=129 (K%128!=0) must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    {
        auto f = make_ternary_fixture(0, 5120);  // N must be positive.
        failures += expect_invalid_argument("weight: n=0 must throw",
                                            [&f]() { (void)validate_ternary_weight(f.weight, "t"); });
    }
    return failures;
}

// ---------------------------------------------------------------------------
// d. decode_fwt_fp64_oracle
// ---------------------------------------------------------------------------
//
// One (row, token) projection through the pinned ternary pipeline: the fused FP32
// load (x * sign * 1/32, left-associative), the pinned FP32 butterfly, and the
// canonical FP32 FMA-chain contraction over the 1024 features of every rotation
// block. The FP64 oracle exact-decodes the weights and evaluates
// out = (1/32) * sum_k H[j][k] * s[k] * x[k] per 1024 block with the naive
// Sylvester matrix; the FP32 pipeline stays within a loose relative bound of
// the exact value.

// The canonical FP32 pipeline for one (row, token): fused load, pinned FP32
// butterfly, canonical FP32 FMA chain over the 5 rotation blocks.
float pipeline_output(const std::uint8_t* blocks, const float* signs, const std::uint16_t* x_row,
                      int row, int token) {
    const int groups_per_row = Geometry::kGroupsPerRow;
    float total = 0.0F;
    for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
        const float* sign_block = signs + k_block * kTernaryFwtBlock;
        std::array<float, kTernaryFwtBlock> fwt_row;
        for (int idx = 0; idx < kTernaryFwtBlock; ++idx) {
            fwt_row[static_cast<std::size_t>(idx)] = fused_fwt_load(
                bf16_to_f32(x_row[static_cast<std::size_t>(token) * Geometry::kInputRows +
                                  k_block * kTernaryFwtBlock + idx]),
                sign_block[idx]);
        }
        fwt_butterfly_f32(fwt_row);
        for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
            const std::uint8_t* block = blocks +
                                        static_cast<std::uint64_t>(row) * groups_per_row *
                                            kTernaryPq2BlockBytes +
                                        (k_block * Geometry::kGroupsPerKBlock + group) *
                                            kTernaryPq2BlockBytes;
            const std::uint16_t scale_word =
                static_cast<std::uint16_t>(block[0]) | (static_cast<std::uint16_t>(block[1]) << 8);
            const float scale = ternary_pq2_scale_word_to_float(scale_word);
            const int group0  = group * kTernaryPq2GroupSize;
            for (int lane = 0; lane < kTernaryPq2GroupSize; ++lane) {
                const int code   = ternary_pq2_code(block[2 + lane / 4], lane % 4);
                const float weight = static_cast<float>(code - 1) * scale;
                // fwt_row holds this block's transformed row; the within-block feature is
                // group0 + lane (the k_block offset stays in the global x/weight addressing).
                total = fmaf(weight, fwt_row[static_cast<std::size_t>(group0 + lane)], total);
            }
        }
    }
    return total;
}

int decode_fwt_fp64_oracle() {
    int failures = 0;
    constexpr std::int32_t kRows   = 128;
    constexpr std::int32_t kTokens = 4;

    auto fixture = make_ternary_fixture(kRows, Geometry::kInputRows);
    fill_ternary_plane(fixture.buf.data(), kRows, Geometry::kInputRows, 0x9E3779B9u);
    const auto* blocks = reinterpret_cast<const std::uint8_t*>(fixture.buf.data());
    const auto* signs  = reinterpret_cast<const float*>(fixture.buf.data() + fixture.rot_offset +
                                                         kTernaryPq2RotationHeaderBytes);

    // Deterministic BF16 activations: the exact values the kernels read.
    std::vector<std::uint16_t> x_bf16(static_cast<std::size_t>(kTokens) * Geometry::kInputRows);
    {
        std::mt19937 g(0x1234ABCDu);
        std::uniform_real_distribution<float> d(-4.0F, 4.0F);
        for (auto& word : x_bf16) { word = f32_to_bf16(d(g)); }
    }

    // FP64 oracle: the naive Sylvester H_1024 over the exact-decoded weights.
    const std::vector<std::vector<double>> H = sylvester<1024>();
    for (std::int32_t r = 0; r < 16; ++r) {
        for (std::int32_t t = 0; t < kTokens; ++t) {
            const float got = pipeline_output(blocks, signs, x_bf16.data(), r, t);

            double reference = 0.0;
            for (int k_block = 0; k_block < Geometry::kKBlocks; ++k_block) {
                const std::uint8_t* block0 =
                    blocks + static_cast<std::uint64_t>(r) * Geometry::kGroupsPerRow *
                                kTernaryPq2BlockBytes +
                    k_block * Geometry::kGroupsPerKBlock * kTernaryPq2BlockBytes;
                std::vector<double> fwt64(1024);
                for (int j = 0; j < 1024; ++j) {
                    double acc = 0.0;
                    for (int k = 0; k < 1024; ++k) {
                        const double x =
                            bf16_to_f64(x_bf16[static_cast<std::size_t>(t) *
                                                   Geometry::kInputRows +
                                               k_block * kTernaryFwtBlock + k]);
                        const double s =
                            reinterpret_cast<const std::uint32_t*>(signs)[k_block * kTernaryFwtBlock +
                                                                          k] == 0x3F800000u
                                ? 1.0
                                : -1.0;
                        acc += static_cast<double>(H[static_cast<std::size_t>(j)][
                                                       static_cast<std::size_t>(k)]) *
                               s * x;
                    }
                    fwt64[static_cast<std::size_t>(j)] = 0.03125 * acc;
                }
                for (int group = 0; group < Geometry::kGroupsPerKBlock; ++group) {
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
            const double bound = 1e-4 * std::max(1.0, std::fabs(reference));
            if (std::fabs(static_cast<double>(got) - reference) > bound) {
                std::cerr << "oracle: row " << r << " token " << t << " is "
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
        failures += capacity_admits();
        failures += capacity_rejects();
        failures += weight_validation();
        failures += decode_fwt_fp64_oracle();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TERNARY_PQ2_0 GdnInputProj (host)\n";
        return failures;
    } catch (const std::exception& error) {
        std::cerr << "TERNARY_PQ2_0 GdnInputProj (host): " << error.what() << '\n';
        return 1;
    }
}
