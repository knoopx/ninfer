#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
using TernaryLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
}
