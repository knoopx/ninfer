#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: Candidate slice softmax with optional per-row group max-pooling
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c]. For every row r the pooling
 *   is ACTIVE iff any groups[r,c] >= 0. When active,
 *
 *     pooled[g] = max_{c : groups[r,c] == g} l[r,c],
 *
 *   and each candidate with group g >= 0 reports s[r,c] = pooled[g], while a candidate with
 *   groups[r,c] == -1 in an active row reports s[r,c] = l[r,c]. All-(-1) rows are a plain
 *   slice, s[r,c] = l[r,c]. The temperature floor T = max(temperature, 1e-6) gives
 *   z = s / T, and
 *
 *     out[r,c] = exp(z[r,c] - rowmax(z)) / sum_c exp(z[r,c] - rowmax(z)).
 *
 *   The winner is the argmax of the pre-softmax score s; T never changes the winner apart
 *   from ties.
 *
 * Logical shapes:
 *   logits, ids, groups, and out are all [N,C] with N>0 and 1<=C<=256. ids are token ids in
 *   [0,vocab) (a caller precondition; the op takes no vocabulary bound); each groups element
 *   is -1 or in [0,C).
 *
 * Supported domain:
 *   logits and out are contiguous finite FP32; ids and groups are contiguous I32. Storage
 *   has its dtype's natural alignment.
 *
 * Numeric:
 *   out is the FP32 numerical approximation computed with a max-subtracted exp; each row sums
 *   to 1. The independent oracle evaluates the full formula in FP64 from the represented FP32
 *   inputs (docs/maintainer/op-development.md).
 *
 * Validation:
 *   Throws std::invalid_argument for a non-finite temperature, malformed shapes, dtypes,
 *   contiguity, alignment, or overlap, for a negative id or out-of-domain group element,
 *   and for any non-finite logits element.
 *
 * Effects:
 *   Writes every output element and preserves all inputs. out must not overlap logits, ids,
 *   or groups.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state.
 */
void candidate_slice_softmax(const Tensor& logits, const Tensor& ids, const Tensor& groups,
                             float temperature, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
