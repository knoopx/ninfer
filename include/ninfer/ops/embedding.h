#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Gathers one embedding row per token:
 *
 *   ideal[d,t] = dequantize(table)[ids[t],d].
 *
 * `ids` is contiguous I32 [T], `out` is contiguous BF16 [D,T], and every id is in
 * [0,vocab). `table` has logical shape [vocab,D] and is contiguous BF16, Q6_G64_FP16
 * RowSplit, Q8_G32_FP16 RowSplit, FP8_E4M3FN_ROW_BF16 RowScale, or TERNARY_PQ2_0
 * TernaryPq2Block. Dense BF16 values are copied
 * bit-exactly. For quantized tables, the oracle independently decodes each code and multiplies it
 * by the exact stored scale in FP64; the BF16 output is promoted and compared directly with that
 * ideal. Final output storage rounding belongs to the quantized embedding criterion, not the
 * oracle. The registered domains are Q6 `[248320,5120]`, Q8 `[248320,2048]` or
 * `[248320,5120]`, FP8 `[248320,5120]`, and TERNARY_PQ2_0 `[248320,5120]`. Q6/Q8 scales are FP16;
 * FP8 has one BF16 multiplier per
 * row and requires 4-byte-aligned output storage. TERNARY_PQ2_0 uses the inverse-transform route:
 * the packed row is gathered, each trit is decoded against its exact binary16 scale in FP32, the
 * inverse D1024 Walsh-Hadamard transform is applied, and the result is emitted as BF16. The
 * inverse route reuses the forward transform and sign vector, because the Hadamard matrix is
 * symmetric; the rotation auxiliary inverse flag must be 1, so a non-inverse ternary embed table
 * is rejected. `out` must not overlap `ids` or any table plane.
 * There is no workspace or persistent state side effect.
 */
void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
