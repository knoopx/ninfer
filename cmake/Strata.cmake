# --- Strata source tree -------------------------------------------------------
# Strata (github.com/Niko1221/Strata, pinned by the flake) provides the GGUF v3
# reader (header-only) and the CUDA kernels for the GGUF weight types. The tree is
# external and referenced by path only — no Strata source is copied into NInfer.
# The flake always sets NINFER_STRATA_SRC; out-of-flake dev builds use the
# NINFER_STRATA_SRC environment variable.
set(NINFER_STRATA_SRC "" CACHE PATH
  "Strata source tree (fetched by the flake; GGUF reader + CUDA kernels)")
if(NOT NINFER_STRATA_SRC AND DEFINED ENV{NINFER_STRATA_SRC})
  set(NINFER_STRATA_SRC "$ENV{NINFER_STRATA_SRC}")
endif()
if(NOT NINFER_STRATA_SRC)
  message(FATAL_ERROR
    "NINFER_STRATA_SRC is required (the Strata source tree; always set by the flake). "
    "Provide it via -DNINFER_STRATA_SRC=<dir> or the NINFER_STRATA_SRC environment "
    "variable, pointing at a fetched Niko1221/Strata tree.")
endif()
if(NOT EXISTS "${NINFER_STRATA_SRC}/include/strata/artifact/gguf_reader.hpp"
    OR NOT EXISTS "${NINFER_STRATA_SRC}/src/kernels/cuda/s_gemv.cu")
  message(FATAL_ERROR
    "NINFER_STRATA_SRC='${NINFER_STRATA_SRC}' is not a Strata source tree: it must "
    "contain include/strata/artifact/gguf_reader.hpp and src/kernels/cuda/s_gemv.cu.")
endif()

# Headers: the strata/* API headers plus the llama.cpp ggml quant block-layout
# headers (ggml-common.h, pulled in by the iq kernels).
add_library(ninfer_strata INTERFACE)
target_include_directories(ninfer_strata INTERFACE
  ${NINFER_STRATA_SRC}/include
  ${NINFER_STRATA_SRC}/third_party/ggml)

# The Strata CUDA kernels NInfer needs for the GGUF weight types, compiled for the
# project architecture: CMAKE_CUDA_ARCHITECTURES=120a is inherited from the
# top level (no fallbacks for older archs). The kernel translation units include
# only strata/kernels/*.hpp and ggml-common.h — no other Strata sources (e.g.
# src/core/) are required.
add_library(ninfer_strata_kernels STATIC
  ${NINFER_STRATA_SRC}/src/kernels/cuda/s_gemv.cu
  ${NINFER_STRATA_SRC}/src/kernels/cuda/iq_kernels.cu
  ${NINFER_STRATA_SRC}/src/kernels/cuda/native_mmvq.cu
  ${NINFER_STRATA_SRC}/src/kernels/cuda/quantize_act.cu
  ${NINFER_STRATA_SRC}/src/kernels/cuda/bf16_gemv.cu
  ${NINFER_STRATA_SRC}/src/kernels/cuda/dequant_bf16.cu
  ${NINFER_STRATA_SRC}/src/kernels/cuda/elementwise.cu)
ninfer_cuda_archive(ninfer_strata_kernels)
target_link_libraries(ninfer_strata_kernels PUBLIC ninfer_strata PRIVATE CUDA::cudart)
