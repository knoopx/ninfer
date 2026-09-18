#include "ops/linear_pair/ternary/ternary_pair_plan.h"

#include "ops/linear/ternary/ternary_format.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// The registered pair problem: N=1024, K in {5120 (independent weights), 2048 (adjacent row
// views of one [6144,2048] parent)}.
constexpr std::int32_t kPairRows = 1024;

bool admits_k(std::int32_t k) {
    return k == 5120 || k == 2048;
}

// P3b T routing (until P4 tuning): T=1 uses the GEMV (decode) route; 2..32 use the 16-token
// SIMT route; T>=33 uses the MMA route. All resolve to A16 compute.
enum class TernaryPairRoute {
    Gemv,
    Simt,
    Mma,
};

TernaryPairRoute resolve_route(std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("ternary linear_pair: T must be positive"); }
    if (tokens == 1) { return TernaryPairRoute::Gemv; }
    if (tokens <= 32) { return TernaryPairRoute::Simt; }
    return TernaryPairRoute::Mma;
}

// The [1024,2048] case: two exact adjacent row views of one [6144,2048] parent. Both views
// share the parent payload (the qdata offsets into it are row * groups * 34) and the parent's
// rotation auxiliary. Mirrors the q8 require_dflash_row_views geometry check.
void require_pair_ternary_views(const Weight& first_weight, const Weight& second_weight) {
    constexpr std::int32_t kParentRows    = 6144;
    constexpr std::int32_t kHidden        = 2048;
    constexpr std::int32_t kFirstRow      = 4096;
    constexpr std::int32_t kSecondRow     = 5120;
    constexpr std::uint64_t kParentGroups = kHidden / kTernaryPq2GroupSize;
    constexpr std::uint64_t kParentBlockPlaneBytes =
        static_cast<std::uint64_t>(kParentRows) * kParentGroups * kTernaryPq2BlockBytes;
    constexpr std::uint64_t kParentRotationBytes =
        kTernaryPq2RotationHeaderBytes +
        static_cast<std::uint64_t>(kHidden) * kTernaryPq2SignWordBytes;
    constexpr std::uint64_t kParentPayloadBytes =
        kParentBlockPlaneBytes + kParentRotationBytes;
    constexpr std::uint64_t kFirstQdataOffset =
        static_cast<std::uint64_t>(kFirstRow) * kParentGroups * kTernaryPq2BlockBytes;
    constexpr std::uint64_t kSecondQdataOffset =
        static_cast<std::uint64_t>(kSecondRow) * kParentGroups * kTernaryPq2BlockBytes;

    const auto* payload = first_weight.payload;
    const auto* first_qdata =
        static_cast<const void*>(static_cast<const std::byte*>(payload) + kFirstQdataOffset);
    const auto* second_qdata =
        static_cast<const void*>(static_cast<const std::byte*>(payload) + kSecondQdataOffset);
    if (payload == nullptr || second_weight.payload != payload ||
        first_weight.payload_bytes < kParentPayloadBytes ||
        second_weight.payload_bytes < kParentPayloadBytes ||
        first_weight.qdata != first_qdata || second_weight.qdata != second_qdata ||
        first_weight.rotation != second_weight.rotation) {
        throw std::invalid_argument(
            "ternary linear_pair: [1024,2048] weights must be exact adjacent parent K/V row "
            "views sharing the parent rotation auxiliary");
    }
}

void require_pair_weights(const Weight& first_weight, const Weight& second_weight,
                          std::int32_t k) {
    if (first_weight.n != kPairRows || second_weight.n != kPairRows || first_weight.k != k ||
        second_weight.k != k) {
        throw std::invalid_argument(
            "ternary linear_pair: weights must be matching [1024,K] ternary matrices");
    }
}

} // namespace

std::size_t ternary_pair_workspace_capacity_bytes(std::int32_t input_rows,
                                                  std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("ternary linear_pair: invalid token interval");
    }
    if (!admits_k(input_rows)) {
        throw std::invalid_argument("ternary linear_pair: unsupported input width");
    }
    (void)resolve_route(min_tokens);
    (void)resolve_route(max_tokens);
    // The A16-only ternary route needs no transient storage (plain writing).
    return 0;
}

void ternary_pair_dispatch(const Tensor& x, const Weight& first_weight,
                           const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                           cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("ternary linear_pair: T must be positive"); }
    const std::int32_t k = x.ne[0];
    if (!admits_k(k)) { throw std::invalid_argument("ternary linear_pair: unsupported input width"); }
    require_pair_weights(first_weight, second_weight, k);
    if (k == 2048) {
        require_pair_ternary_views(first_weight, second_weight);
        (void)validate_ternary_row_view_metadata(first_weight, "ternary linear_pair");
        (void)validate_ternary_row_view_metadata(second_weight, "ternary linear_pair");
    } else {
        (void)validate_ternary_weight_metadata(first_weight, "ternary linear_pair");
        (void)validate_ternary_weight_metadata(second_weight, "ternary linear_pair");
    }

    switch (resolve_route(tokens)) {
    case TernaryPairRoute::Gemv:
        ternary_pair_gemv_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    case TernaryPairRoute::Simt:
        ternary_pair_simt_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    case TernaryPairRoute::Mma:
        ternary_pair_mma_launch(x, first_weight, second_weight, first_out, second_out, stream);
        return;
    }
    throw std::logic_error("ternary linear_pair: unrouted token count");
}

} // namespace ninfer::ops::detail
