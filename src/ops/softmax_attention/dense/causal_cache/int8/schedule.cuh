#pragma once

#include "core/paged_kv_cache.h"
#include "ops/softmax_attention/common/causal_geometry.h"

namespace ninfer::ops::detail {

template <int TokenTile, int Warps, int KeyTile, int MinBlocks = 1, bool DynamicArena = true>
struct Int8KvGroupedMmaSchedule {
    static_assert(TokenTile > 0 && Warps > 0 && Warps <= 16 && MinBlocks > 0);
    static_assert(KeyTile == 32 || KeyTile == 64);
    static constexpr int kTokenTile     = TokenTile;
    static constexpr int kWarps         = Warps;
    static constexpr int kThreads       = Warps * 32;
    static constexpr int kKeyRows       = KeyTile;
    static constexpr int kMinBlocks     = MinBlocks;
    static constexpr bool kDynamicArena = DynamicArena;
    static constexpr int kArenaBytes    = 4 * KeyTile * 256;
};

// Fast INT8 prompt schedule: each warp owns 16 query rows of one head for the whole key sweep,
// so scores, probabilities and the D256 output accumulator never leave registers. Eight warps
// fill an SM's register file; four serve launches too narrow to occupy every SM with 128-row
// CTAs. K and V codes are double-buffered as raw INT8 pages (one 64-key page per tile).
inline constexpr int kInt8FastBc        = 64;
inline constexpr int kInt8FastGroupSize = 64; // INT8 G64 codec group
inline constexpr int kInt8FastGroups    = kCausalHeadDim / kInt8FastGroupSize;
inline constexpr int kInt8FastTileBytes   = kInt8FastBc * kCausalHeadDim;
inline constexpr int kInt8FastScaleBytes  = kInt8FastBc * kInt8FastGroups * 2;
inline constexpr int kInt8FastStageBytes  = 2 * kInt8FastTileBytes + 2 * kInt8FastScaleBytes;

template <int Warps>
struct Int8KvFastMmaSchedule {
    static_assert(Warps == 4 || Warps == 8);
    static constexpr int Threads   = Warps * 32;
    static constexpr int Br        = Warps * 16;
    static constexpr int QBytes    = Br * kCausalHeadDim;
    static constexpr int SmemBytes = QBytes + 2 * kInt8FastStageBytes;
};

static_assert(kInt8FastBc == kPagedKVPageSize);
static_assert(kInt8FastGroups == 4);
static_assert(Int8KvFastMmaSchedule<8>::SmemBytes == 100352);

template <int DChunk>
struct Int8KvMergeSchedule {
    static_assert(DChunk > 0 && DChunk <= 256);
    static constexpr int kDChunk  = DChunk;
    static constexpr int kThreads = 256;
};

} // namespace ninfer::ops::detail
