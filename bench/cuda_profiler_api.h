#pragma once

// CUDA 13 removed cudaProfilerStart/cudaProfilerStop from the runtime API.
// Profiling is now external (ncu/nsys). This shim preserves the legacy include
// and defines the removed functions as no-ops returning cudaSuccess.
#include <cuda_runtime_api.h>

#ifndef cudaProfilerStart
#define cudaProfilerStart() (cudaSuccess)
#endif
#ifndef cudaProfilerStop
#define cudaProfilerStop() (cudaSuccess)
#endif
