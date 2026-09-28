// Qualifies sampling_build_truncated_block_fast (src/ops/kernel/sampling_device.cuh)
// against an independent CPU oracle. The body is a one-pass bounded per-thread
// top-cap gather (each thread inserts its strided items into a sorted cap-entry
// register list) followed by a parallel 2-level warp-ballot head merge of the
// per-thread lists to the block top-cap. The case matrix covers varied
// distributions (peaked, flat, all-ties, -inf scattered/majority, penalties,
// overlay, small top_k).
//
// The host section (runs even without a GPU) mirrors the exact merge pipeline:
// the key encoding (score_id_order.cuh), the per-thread bounded insertion over
// strided vocab entries, and the 2-level head merge (max-key head merge with
// lowest-lane tie-break, the host analog of the ballot idiom). The result is
// compared EXACTLY (bit-exact float + exact index, all cap entries) to a full
// sort of (adjusted desc, index asc) taken to cap.
//
// The device section (only when a CUDA device is available) launches a probe
// kernel with the SAME __shared__ declarations as the production callers that
// runs the real sampling_build_truncated_block_fast and verifies:
//   (i)   cand_val[0..cap-1] bit-exact + cand_idx[0..cap-1] exact vs the CPU
//         reference (the core claim: same top-cap set, same ordering);
//   (ii)  n_support exact vs a host re-computation of the
//         sampling_normalize_support support logic applied to the read-back
//         prob values (the kernel's exp values are taken as given);
//   (iii) renormalized prob[0..support-1] bit-exact vs the host re-computation
//         (ssum then inv = 1/ssum, same accumulation order);
//   (iv)  determinism: two runs into two buffers are byte-identical
//         (graph-safety: no mutable state, no RNG, no host sync).

#include "ops/kernel/sampling_device.cuh"
#include "ninfer/ops/sampling.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kBlock = ops::kSamplerBlock; // 256
constexpr int kFast  = ops::kSamplerFastCandidates;
constexpr int kWarps = ops::kSamplingTileWarps; // 8

// --- host mirrors of the device primitives ----------------------------------

// Mirrors ordered_score_bits in score_id_order.cuh (canonicalizes -0.0f).
std::uint32_t host_ordered_score_bits(float v) {
    if (v == 0.0f) { v = 0.0f; }
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

// Mirrors score_id_order_key: id == INT_MAX -> sentinel 0; higher = better
// (value desc, lower index wins ties).
std::uint64_t host_key(float v, int idx) {
    if (idx == INT_MAX) { return 0; }
    return (static_cast<std::uint64_t>(host_ordered_score_bits(v)) << 32) |
           (0xffffffffu - static_cast<std::uint32_t>(idx));
}

// Mirrors score_from_order_key (exact bit round-trip for finite values and
// -inf; the sentinel key 0 decodes to NaN, which is never observed because
// cap <= vocab keeps the top-cap all real entries).
float host_key_float(std::uint64_t key) {
    const std::uint32_t ordered = static_cast<std::uint32_t>(key >> 32);
    const std::uint32_t bits    = (ordered & 0x80000000u) != 0u ? (ordered ^ 0x80000000u) : ~ordered;
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Mirrors id_from_order_key (key 0 -> INT_MAX).
int host_key_index(std::uint64_t key) {
    if (key == 0) { return INT_MAX; }
    return static_cast<int>(0xffffffffu - static_cast<std::uint32_t>(key));
}

// Host mirror of the SamplingConfig fields the merge path consumes.
struct HostConfig {
    float temperature        = 1.0f;
    int top_k                = 20;
    float top_p              = 1.0f;
    float min_p              = 0.0f;
    float presence_penalty   = 0.0f;
    float frequency_penalty  = 0.0f;
    const std::vector<int>* counts  = nullptr;
    const std::vector<int>* overlay = nullptr;
};

// Mirrors sampling_candidate_cap exactly.
int host_cap(int top_k, int vocab) {
    int cap = kFast;
    if (top_k > 0 && top_k < cap) { cap = top_k; }
    if (vocab < cap) { cap = vocab; }
    return cap;
}

// Mirrors sampling_adjusted_logit (same float operations, same order).
float host_adjusted(float raw, int v, const HostConfig& cfg) {
    float x = raw;
    if (cfg.presence_penalty == 0.0f && cfg.frequency_penalty == 0.0f) { return x; }
    int cnt = cfg.counts != nullptr ? (*cfg.counts)[static_cast<std::size_t>(v)] : 0;
    if (cfg.overlay != nullptr) {
        for (int ov : *cfg.overlay) {
            if (ov == v) { ++cnt; }
        }
    }
    if (cnt > 0) { x -= cfg.presence_penalty; }
    if (cfg.frequency_penalty != 0.0f) { x -= cfg.frequency_penalty * static_cast<float>(cnt); }
    return x;
}

// Mirrors sampling_better / sampling_insert_candidate (bounded descending list).
bool host_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

void host_insert(std::array<float, kFast>& val, std::array<int, kFast>& idx, int cap, float v,
                 int i) {
    if (cap <= 0 || !host_better(v, i, val[cap - 1], idx[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && host_better(v, i, val[pos - 1], idx[pos - 1])) {
        val[pos] = val[pos - 1];
        idx[pos] = idx[pos - 1];
        --pos;
    }
    val[pos] = v;
    idx[pos] = i;
}

struct LaneList {
    std::array<float, kFast> val{};
    std::array<int, kFast> idx{};
};

// Per-thread bounded insertion over the strided entries v = t, t+256, ... < vocab.
std::vector<LaneList> host_lane_lists(const std::vector<float>& logits, int vocab,
                                      const HostConfig& cfg, int cap) {
    std::vector<LaneList> lanes(static_cast<std::size_t>(kBlock));
    for (int t = 0; t < kBlock; ++t) {
        auto& lane = lanes[static_cast<std::size_t>(t)];
        lane.val.fill(-std::numeric_limits<float>::infinity());
        lane.idx.fill(INT_MAX);
        for (int v = t; v < vocab; v += kBlock) {
            host_insert(lane.val, lane.idx, cap,
                        host_adjusted(logits[static_cast<std::size_t>(v)], v, cfg), v);
        }
    }
    return lanes;
}

// Host analog of the 2-level ballot head merge: per rank take the max head key,
// break ties by lowest lane (== __ffs), advance the winner's position.
std::vector<std::pair<float, int>> host_two_level_merge(const std::vector<LaneList>& lanes,
                                                         int cap) {
    std::array<std::vector<std::uint64_t>, kWarps> warp_keys;
    for (int w = 0; w < kWarps; ++w) {
        std::array<int, 32> pos{};
        warp_keys[static_cast<std::size_t>(w)].resize(static_cast<std::size_t>(cap));
        for (int rank = 0; rank < cap; ++rank) {
            std::uint64_t best   = 0;
            int source           = 0;
            for (int l = 0; l < 32; ++l) {
                const auto& lane = lanes[static_cast<std::size_t>(w * 32 + l)];
                const std::uint64_t key = host_key(lane.val[pos[l]], lane.idx[pos[l]]);
                if (l == 0 || key > best) {
                    best   = key;
                    source = l;
                }
            }
            warp_keys[static_cast<std::size_t>(w)][static_cast<std::size_t>(rank)] = best;
            ++pos[source];
        }
    }
    std::array<int, kWarps> wpos{};
    std::vector<std::pair<float, int>> out(static_cast<std::size_t>(cap));
    for (int rank = 0; rank < cap; ++rank) {
        std::uint64_t best   = 0;
        int source           = 0;
        for (int w = 0; w < kWarps; ++w) {
            const std::uint64_t key =
                warp_keys[static_cast<std::size_t>(w)][static_cast<std::size_t>(wpos[w])];
            if (w == 0 || key > best) {
                best   = key;
                source = w;
            }
        }
        out[static_cast<std::size_t>(rank)] = {host_key_float(best), host_key_index(best)};
        ++wpos[source];
    }
    return out;
}

// Independent oracle: full sort of (adjusted desc, index asc), take cap.
// The value is canonicalized exactly as the key contract does (score_id_order
// "Numeric zero is canonicalized so +0 and -0 reach the id tie-break"): a
// -0.0f adjusted logit decodes back as +0.0f, so the reference matches the
// merge output bit-for-bit.
std::vector<std::pair<float, int>> host_reference(const std::vector<float>& logits, int vocab,
                                                  const HostConfig& cfg, int cap) {
    std::vector<std::pair<float, int>> all(static_cast<std::size_t>(vocab));
    for (int v = 0; v < vocab; ++v) {
        float a = host_adjusted(logits[static_cast<std::size_t>(v)], v, cfg);
        if (a == 0.0f) { a = 0.0f; }
        all[static_cast<std::size_t>(v)] = {a, v};
    }
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    all.resize(static_cast<std::size_t>(cap));
    return all;
}

// Mirrors the support/renormalization part of sampling_normalize_support,
// applied to the kernel's read-back prob values (exp values NOT recomputed).
// Returns support; when renorm != nullptr, fills the renormalized values.
// Faithfulness scope: the min_p threshold line is exactly equivalent to the
// device's (the renormalization scale cancels: prob_unnorm[j] < min_p*e0
// iff prob_norm[j] < min_p*prob_norm[0]). The top_p target uses the read-back
// sum over all cap entries, which equals the device's pre-truncation sum only
// while top_p is disabled (support == cap); do not add a top_p < 1 case
// without first making this target bit-faithful.
int host_support(const std::vector<float>& prob, int cap, const HostConfig& cfg,
                 std::vector<float>* renorm) {
    float sum = 0.0f;
    for (int j = 0; j < cap; ++j) { sum += prob[static_cast<std::size_t>(j)]; }
    const float e0           = prob[0];
    const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
    const bool top_p_active  = (cfg.top_p < 1.0f);
    const float top_p_target = cfg.top_p * sum;
    float cum                 = 0.0f;
    int support               = 0;
    for (int j = 0; j < cap; ++j) {
        if (min_p_thresh >= 0.0f && prob[static_cast<std::size_t>(j)] < min_p_thresh) { break; }
        cum += prob[static_cast<std::size_t>(j)];
        support = j + 1;
        if (top_p_active && cum >= top_p_target) { break; }
    }
    if (support < 1) { support = 1; }
    if (renorm != nullptr) {
        float ssum = 0.0f;
        for (int j = 0; j < support; ++j) { ssum += prob[static_cast<std::size_t>(j)]; }
        const float inv = 1.0f / ssum;
        renorm->resize(static_cast<std::size_t>(support));
        for (int j = 0; j < support; ++j) {
            (*renorm)[static_cast<std::size_t>(j)] = prob[static_cast<std::size_t>(j)] * inv;
        }
    }
    return support;
}

// --- cases -------------------------------------------------------------------

struct MergeCase {
    const char* label;
    std::vector<float> logits; // f32, already on the BF16 grid
    int vocab                  = 0;
    HostConfig cfg;
    std::vector<int> counts;   // owned; non-empty only for the penalty case
    std::vector<int> overlay;  // owned; non-empty only for the penalty case
};

std::vector<MergeCase> make_cases() {
    std::vector<MergeCase> cases;
    cases.reserve(9);

    // (a) random logits, real shape, no penalties, full cap.
    {
        MergeCase c;
        c.label        = "merge random vocab=248077 top_k=20";
        c.vocab        = 248077;
        c.logits.resize(static_cast<std::size_t>(c.vocab));
        fill_uniform(c.logits, 0x5eedu, -12.0f, 12.0f);
        round_to_bf16(c.logits);
        c.cfg.temperature = 1.0f;
        c.cfg.top_k       = 20;
        cases.push_back(std::move(c));
    }
    // (b) all ties: top-20 must be exactly indices 0..19.
    {
        MergeCase c;
        c.label        = "merge all-ties vocab=300 top-20 = 0..19";
        c.vocab        = 300;
        c.logits.assign(static_cast<std::size_t>(c.vocab), 1.0f);
        c.cfg.temperature = 1.0f;
        c.cfg.top_k       = 20;
        cases.push_back(std::move(c));
    }
    // (c1) scattered -inf: 25 finite (even indices), 25 real -inf (odd);
    // threads >= 50 are all-sentinel lists.
    {
        MergeCase c;
        c.label        = "merge -inf scattered vocab=50 (25 finite / 25 -inf)";
        c.vocab        = 50;
        c.logits.assign(static_cast<std::size_t>(c.vocab), -std::numeric_limits<float>::infinity());
        std::vector<float> finite(25);
        fill_uniform(finite, 77u, -8.0f, 8.0f);
        round_to_bf16(finite);
        for (int i = 0; i < 25; ++i) {
            c.logits[static_cast<std::size_t>(2 * i)] = finite[static_cast<std::size_t>(i)];
        }
        c.cfg.temperature = 1.0f;
        c.cfg.top_k       = 20;
        cases.push_back(std::move(c));
    }
    // (c2) -inf majority: only 15 finite, so the top-20 includes 5 real -inf
    // entries ordered by ascending index.
    {
        MergeCase c;
        c.label        = "merge -inf majority vocab=50 (15 finite / 35 -inf)";
        c.vocab        = 50;
        c.logits.assign(static_cast<std::size_t>(c.vocab), -std::numeric_limits<float>::infinity());
        std::vector<float> finite(15);
        fill_uniform(finite, 99u, -5.0f, 5.0f);
        round_to_bf16(finite);
        for (int i = 0; i < 15; ++i) {
            c.logits[static_cast<std::size_t>(i)] = finite[static_cast<std::size_t>(i)];
        }
        c.cfg.temperature = 1.0f;
        c.cfg.top_k       = 20;
        cases.push_back(std::move(c));
    }
    // (d) presence/frequency penalties with counts + overlay, real shape.
    {
        MergeCase c;
        c.label        = "merge penalties vocab=248077 (presence=0.7 frequency=0.3)";
        c.vocab        = 248077;
        c.logits.resize(static_cast<std::size_t>(c.vocab));
        fill_uniform(c.logits, 31337u, -12.0f, 12.0f);
        round_to_bf16(c.logits);
        c.counts.resize(static_cast<std::size_t>(c.vocab));
        for (int v = 0; v < c.vocab; ++v) { c.counts[static_cast<std::size_t>(v)] = v % 7; }
        c.overlay = {17, 17, 7919};
        c.cfg.temperature       = 1.0f;
        c.cfg.top_k             = 20;
        c.cfg.presence_penalty  = 0.7f;
        c.cfg.frequency_penalty = 0.3f;
        cases.push_back(std::move(c));
    }
    // (e) cap < 20: top_k = 7 on the real shape.
    {
        MergeCase c;
        c.label        = "merge random vocab=248077 top_k=7";
        c.vocab        = 248077;
        c.logits.resize(static_cast<std::size_t>(c.vocab));
        fill_uniform(c.logits, 555u, -12.0f, 12.0f);
        round_to_bf16(c.logits);
        c.cfg.temperature = 1.0f;
        c.cfg.top_k       = 7;
        cases.push_back(std::move(c));
    }
    // (f) speculative overlay only (counts == nullptr): the round-local count
    // overlay alone drives the penalties for its tokens, exercising the
    // overlay scan in sampling_adjusted_logit. Token 17 sees a count of 2,
    // tokens 7919 and 100000 a count of 1.
    {
        MergeCase c;
        c.label        = "merge overlay-only vocab=248077 (presence=0.5 frequency=0.3)";
        c.vocab        = 248077;
        c.logits.resize(static_cast<std::size_t>(c.vocab));
        fill_uniform(c.logits, 777u, -12.0f, 12.0f);
        round_to_bf16(c.logits);
        c.overlay = {17, 17, 7919, 100000};
        c.cfg.temperature       = 1.0f;
        c.cfg.top_k             = 20;
        c.cfg.presence_penalty  = 0.5f;
        c.cfg.frequency_penalty = 0.3f;
        cases.push_back(std::move(c));
    }
    // (g) peaked distribution: 64 hot values @ 100.0 among zeros. The exact
    // top-20 is the 20 lowest-index hot entries (0..19 at 100.0).
    {
        MergeCase c;
        c.label        = "merge peaked vocab=248077 (64 hot @ 100.0)";
        c.vocab        = 248077;
        c.logits.assign(static_cast<std::size_t>(c.vocab), 0.0f);
        for (int i = 0; i < 64; ++i) { c.logits[static_cast<std::size_t>(i)] = 100.0f; }
        c.cfg.temperature = 1.0f;
        c.cfg.top_k       = 20;
        cases.push_back(std::move(c));
    }
    // (h) flat distribution: a gentle ramp 0.0..1.01 (indices 0..101), 5
    // mild outliers @ 10.0 (indices 200..204), base -1.0. The exact top-20
    // is the 5 outliers plus the 15 highest ramp entries (indices 101..87).
    {
        MergeCase c;
        c.label        = "merge flat vocab=248077 (ramp + 5 mild @ 10.0)";
        c.vocab        = 248077;
        c.logits.assign(static_cast<std::size_t>(c.vocab), -1.0f);
        for (int i = 0; i < 102; ++i) { c.logits[static_cast<std::size_t>(i)] = i * 0.01f; }
        for (int i = 0; i < 5; ++i) { c.logits[static_cast<std::size_t>(200 + i)] = 10.0f; }
        round_to_bf16(c.logits);
        c.cfg.temperature = 1.0f;
        c.cfg.top_k       = 20;
        cases.push_back(std::move(c));
    }

    // Wire the cfg pointers into each case's owned vectors (stable addresses).
    for (auto& k : cases) {
        if (!k.counts.empty()) { k.cfg.counts = &k.counts; }
        if (!k.overlay.empty()) { k.cfg.overlay = &k.overlay; }
    }
    return cases;
}

// --- host verification --------------------------------------------------------

int verify_host_case(const MergeCase& c) {
    const int cap = host_cap(c.cfg.top_k, c.vocab);
    const auto lanes  = host_lane_lists(c.logits, c.vocab, c.cfg, cap);
    const auto merged = host_two_level_merge(lanes, cap);
    const auto ref    = host_reference(c.logits, c.vocab, c.cfg, cap);

    int failures = 0;
    if (merged.size() != ref.size()) {
        std::cerr << c.label << ": host merge size " << merged.size() << " != ref "
                  << ref.size() << '\n';
        return 1;
    }
    for (int rank = 0; rank < cap; ++rank) {
        const auto& got = merged[static_cast<std::size_t>(rank)];
        const auto& exp = ref[static_cast<std::size_t>(rank)];
        if (std::memcmp(&got.first, &exp.first, sizeof(float)) != 0 || got.second != exp.second) {
            std::cerr << c.label << ": host merge mismatch at rank " << rank << " got=("
                      << got.first << ", " << got.second << ") ref=(" << exp.first << ", "
                      << exp.second << ")\n";
            ++failures;
        }
    }
    return failures;
}

// --- device probe --------------------------------------------------------------

__global__ void merge_probe_kernel(const __nv_bfloat16* logits, std::int32_t vocab,
                                   const ops::SamplingConfig* cfg, const std::int32_t* overlay,
                                   int overlay_len, float* out_val, int* out_idx,
                                   float* out_prob, int* out_support) {
    // Same __shared__ declarations as the production callers
    // (sample_row_kernel / speculative_accept_greedy_drafts_kernel).
    __shared__ float cand_val[ops::kSamplerCandidateCap];
    __shared__ int cand_idx[ops::kSamplerCandidateCap];
    __shared__ float prob[ops::kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ float merge_val[ops::kSamplerBlock * ops::kSamplerFastCandidates];
    __shared__ int merge_idx[ops::kSamplerBlock * ops::kSamplerFastCandidates];

    ops::sampling_build_truncated_block_fast(
        logits, 0, vocab, *cfg, merge_val, merge_idx, cand_val, cand_idx, prob, &n_support,
        overlay, overlay_len);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < ops::kSamplerCandidateCap; ++j) {
            out_val[j]  = cand_val[j];
            out_idx[j]  = cand_idx[j];
            out_prob[j] = prob[j];
        }
        *out_support = n_support;
    }
}

struct DeviceOutputs {
    std::vector<float> val;
    std::vector<int> idx;
    std::vector<float> prob;
    int support = -1;
};

DeviceOutputs run_probe(const __nv_bfloat16* d_logits, int vocab,
                        const ops::SamplingConfig* d_cfg, const std::int32_t* d_overlay,
                        int overlay_len, const DeviceBuffer& out_val, const DeviceBuffer& out_idx,
                        const DeviceBuffer& out_prob, const DeviceBuffer& out_support) {
    merge_probe_kernel<<<1, kBlock>>>(d_logits, vocab, d_cfg, d_overlay, overlay_len,
                                      static_cast<float*>(out_val.p),
                                      static_cast<int*>(out_idx.p),
                                      static_cast<float*>(out_prob.p),
                                      static_cast<int*>(out_support.p));
    cuda_check_last_launch("merge probe launch");
    cuda_synchronize();
    DeviceOutputs out;
    out.val     = from_device<float>(out_val, ops::kSamplerCandidateCap);
    out.idx     = from_device<int>(out_idx, ops::kSamplerCandidateCap);
    out.prob    = from_device<float>(out_prob, ops::kSamplerCandidateCap);
    out.support = from_device<int>(out_support, 1)[0];
    return out;
}

int run_device_case(const MergeCase& c, const DeviceBuffer& d_logits_max,
                    const DeviceBuffer& d_counts, const DeviceBuffer& d_overlay) {
    const int cap = host_cap(c.cfg.top_k, c.vocab);
    const auto ref = host_reference(c.logits, c.vocab, c.cfg, cap);

    // Upload this case's BF16 logits into the shared device buffer (sized for
    // the largest vocab; ~= 0.5 MB, reused across all cases).
    std::vector<std::uint16_t> bits(c.logits.size());
    for (std::size_t i = 0; i < c.logits.size(); ++i) {
        bits[i] = f32_to_bf16(c.logits[i]);
    }
    d_logits_max.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));

    // Device config; token_counts points at the device counts buffer for the
    // penalty case, nullptr otherwise.
    ops::SamplingConfig hcfg{};
    hcfg.temperature       = c.cfg.temperature;
    hcfg.top_k             = c.cfg.top_k;
    hcfg.top_p             = c.cfg.top_p;
    hcfg.min_p             = c.cfg.min_p;
    hcfg.presence_penalty  = c.cfg.presence_penalty;
    hcfg.frequency_penalty = c.cfg.frequency_penalty;
    hcfg.seed              = 0;
    if (c.cfg.counts != nullptr) { hcfg.token_counts = static_cast<std::int32_t*>(d_counts.p); }
    DeviceBuffer d_cfg(sizeof(ops::SamplingConfig));
    d_cfg.copy_from_host(&hcfg, sizeof(ops::SamplingConfig));

    const int overlay_len =
        c.cfg.overlay != nullptr ? static_cast<int>(c.cfg.overlay->size()) : 0;
    const std::int32_t* d_overlay =
        overlay_len != 0 ? static_cast<std::int32_t*>(d_overlay.p) : nullptr;

    DeviceBuffer out_val(ops::kSamplerCandidateCap * sizeof(float));
    DeviceBuffer out_idx(ops::kSamplerCandidateCap * sizeof(int));
    DeviceBuffer out_prob(ops::kSamplerCandidateCap * sizeof(float));
    DeviceBuffer out_support(sizeof(int));
    DeviceBuffer out_val2(ops::kSamplerCandidateCap * sizeof(float));
    DeviceBuffer out_idx2(ops::kSamplerCandidateCap * sizeof(int));
    DeviceBuffer out_prob2(ops::kSamplerCandidateCap * sizeof(float));
    DeviceBuffer out_support2(sizeof(int));

    const DeviceOutputs a =
        run_probe(static_cast<const __nv_bfloat16*>(d_logits_max.p), c.vocab,
                  static_cast<const ops::SamplingConfig*>(d_cfg.p), d_overlay, overlay_len, out_val,
                  out_idx, out_prob, out_support);
    const DeviceOutputs b =
        run_probe(static_cast<const __nv_bfloat16*>(d_logits_max.p), c.vocab,
                  static_cast<const ops::SamplingConfig*>(d_cfg.p), d_overlay, overlay_len, out_val2,
                  out_idx2, out_prob2, out_support2);

    int failures = 0;
    // (i) core claim: exact top-cap set and ordering vs the CPU reference.
    for (int rank = 0; rank < cap; ++rank) {
        const float exp_val = ref[static_cast<std::size_t>(rank)].first;
        const int exp_idx   = ref[static_cast<std::size_t>(rank)].second;
        if (std::memcmp(&a.val[static_cast<std::size_t>(rank)], &exp_val, sizeof(float)) != 0 ||
            a.idx[static_cast<std::size_t>(rank)] != exp_idx) {
            std::cerr << c.label << ": device merge mismatch at rank " << rank << " got=("
                      << a.val[static_cast<std::size_t>(rank)] << ", "
                      << a.idx[static_cast<std::size_t>(rank)] << ") ref=(" << exp_val << ", "
                      << exp_idx << ")\n";
            ++failures;
        }
    }
    // (ii) support size: host re-computation over the kernel's prob values.
    const int ref_support = host_support(a.prob, cap, c.cfg, nullptr);
    if (a.support != ref_support) {
        std::cerr << c.label << ": n_support got=" << a.support << " expected=" << ref_support
                  << '\n';
        ++failures;
    }
    // (iii) renormalized support probabilities, bit-exact.
    std::vector<float> ref_renorm;
    host_support(a.prob, cap, c.cfg, &ref_renorm);
    for (std::size_t j = 0; j < ref_renorm.size(); ++j) {
        if (std::memcmp(&a.prob[j], &ref_renorm[j], sizeof(float)) != 0) {
            std::cerr << c.label << ": renormalized prob[" << j << "] got=" << a.prob[j]
                      << " expected=" << ref_renorm[j] << '\n';
            ++failures;
        }
    }
    // (iv) determinism: two runs byte-identical (no mutable state).
    if (a.val != b.val || a.idx != b.idx || a.prob != b.prob || a.support != b.support) {
        std::cerr << c.label << ": non-deterministic probe outputs between two runs\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    const auto cases = make_cases();

    // Host section: always runs (CPU model of the changed merge logic).
    int failures = 0;
    for (const auto& c : cases) {
        failures += verify_host_case(c);
    }
    // Sanity: the all-ties oracle itself must order ties by ascending index.
    {
        const MergeCase& ties = cases[1];
        const int cap         = host_cap(ties.cfg.top_k, ties.vocab);
        const auto ref        = host_reference(ties.logits, ties.vocab, ties.cfg, cap);
        for (int rank = 0; rank < cap; ++rank) {
            if (ref[static_cast<std::size_t>(rank)].second != rank) {
                std::cerr << "all-ties oracle did not order by ascending index at rank " << rank
                          << '\n';
                ++failures;
            }
        }
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return failures == 0 ? 77 : 1;
    }

    try {
        // Shared device buffers, sized for the largest case; one logits buffer
        // (248077 BF16 ~= 0.5 MB) is reused across all cases.
        DeviceBuffer d_logits_max(248077 * sizeof(std::uint16_t));
        DeviceBuffer d_counts(248077 * sizeof(std::int32_t));
        d_counts.copy_from_host(cases[4].counts.data(),
                                cases[4].counts.size() * sizeof(std::int32_t));
        std::size_t max_overlay = 0;
        for (const auto& c : cases) {
            if (c.cfg.overlay != nullptr) {
                max_overlay = std::max(max_overlay, c.cfg.overlay->size());
            }
        }
        DeviceBuffer d_overlay(std::max<std::size_t>(max_overlay, 1) * sizeof(std::int32_t));

        for (const auto& c : cases) {
            if (c.cfg.overlay != nullptr) {
                d_overlay.copy_from_host(c.cfg.overlay->data(),
                                         c.cfg.overlay->size() * sizeof(std::int32_t));
            }
            failures += run_device_case(c, d_logits_max, d_counts, d_overlay);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " sampling merge\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "sampling merge: " << error.what() << '\n';
        return 1;
    }
}
