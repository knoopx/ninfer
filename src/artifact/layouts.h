#pragma once

#include "core/weight_view.h"

#include <string_view>

namespace ninfer::artifact {

struct TensorObject;

// Interpret and validate a requested tensor's complete encoded parent.
[[nodiscard]] WeightGeometry describe_tensor(const TensorObject& object);

// Validate the actual bytes of a ternary PQ2_0 parent: every binary16 block scale must
// be finite, and the rotation auxiliary must carry the fixed block size, the Sylvester
// transform, and a sign vector of exactly +/-1.0. Refuses a mismatched payload.
void validate_ternary_payload(const std::byte* payload, const WeightGeometry& geometry,
                              std::string_view id);

} // namespace ninfer::artifact
