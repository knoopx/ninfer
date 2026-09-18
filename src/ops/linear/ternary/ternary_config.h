#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Every ternary PQ2_0 route transforms the activation tile block by 1024-wide block; K must be a
// multiple of the rotation block (the layout guarantees K%128=0, the sign vectors exist for the
// registered K values only, and every registered K is a multiple of 1024).
template <std::int32_t OutputRows, std::int32_t InputRows>
struct TernaryGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % kTernaryPq2RotationBlockSize) == 0);

    static constexpr std::int32_t kOutputRows    = OutputRows;
    static constexpr std::int32_t kInputRows     = InputRows;
    static constexpr std::int32_t kKBlocks       = InputRows / kTernaryPq2RotationBlockSize;
    static constexpr std::int32_t kGroupsPerRow  = InputRows / kTernaryPq2GroupSize;
    static constexpr std::int32_t kGroupsPerKBlock = kTernaryPq2RotationBlockSize / kTernaryPq2GroupSize;
};

// The FWT butterfly transform (src/ops/common/fwt.cuh) runs on a 256-thread block, one
// 1024-element row at a time; every ternary kernel is a 256-thread CTA.

template <int RowsPerWarp, int AccumulatorChains, int MinBlocksPerSm>
struct TernaryGemvSchedule {
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(AccumulatorChains > 0 && AccumulatorChains <= 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta = kTernaryPq2RotationBlockSize / 256;
    static constexpr int kThreads      = 256;
    static constexpr int kRowsPerWarp  = RowsPerWarp;
    static constexpr int kRowsPerCta   = kWarpsPerCta * RowsPerWarp;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
};

template <int Capacity, int RowsPerWarp, int MinBlocksPerSm>
struct TernarySimtSchedule {
    static_assert(Capacity > 1 && Capacity <= 16);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta = kTernaryPq2RotationBlockSize / 256;
    static constexpr int kThreads     = 256;
    static constexpr int kRowsPerWarp = RowsPerWarp;
    static constexpr int kRowsPerCta  = kWarpsPerCta * RowsPerWarp;
    static constexpr int kTokenTile   = Capacity;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
};

// mma.sync m16n8k16 route: the CTA owns a 16-token FWT slab (the MMA-slab transform of
// fwt.cuh) and BlockN weight rows. One warp owns the full 16-token m-tile and BlockN / 8
// n-tiles of 8.
template <int BlockN, int WarpsN, int MinBlocksPerSm>
struct TernaryMmaSchedule {
    static_assert(BlockN >= 256 && (BlockN % 256) == 0);
    static_assert(WarpsN == 8);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kBlockM      = 16;  // the FWT slab token extent.
    static constexpr int kBlockN      = BlockN;
    static constexpr int kWarpsN      = WarpsN;
    static constexpr int kThreads     = 256;
    static constexpr int kWarpN       = BlockN / WarpsN;
    static constexpr int kMmaN        = kWarpN / 8;
    static constexpr int kK16PerBlock = kTernaryPq2RotationBlockSize / 16;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
};

using TernaryN14336K5120 = TernaryGeometry<14336, 5120>;
using TernaryN16384K5120 = TernaryGeometry<16384, 5120>;
using TernaryN34816K5120 = TernaryGeometry<34816, 5120>;
using TernaryN248320K5120 = TernaryGeometry<248320, 5120>;
using TernaryN5120K6144   = TernaryGeometry<5120, 6144>;
using TernaryN5120K17408  = TernaryGeometry<5120, 17408>;

} // namespace ninfer::ops::detail
