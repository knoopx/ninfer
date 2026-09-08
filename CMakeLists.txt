cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Prebuilt webui bundling --------------------------------------------------
# The webui is a required, always-bundled part of the build. NINFER_WEBUI_SRC must always
# point at a built llama.cpp webui tree; the flake always sets it to the fetched ggml-org/
# llama-ui tree. It is NOT optional: a build that provides no NINFER_WEBUI_SRC (no flake,
# no -DNINFER_WEBUI_SRC) is a hard error. The tree is copied into the build tree at
# ${CMAKE_BINARY_DIR}/share/ninfer/webui (landing path <exe-dir>/../share/ninfer/webui).
set(NINFER_WEBUI_SRC "" CACHE PATH
  "Built llama.cpp webui tree to bundle (required; always set by the flake)")
if(NOT NINFER_WEBUI_SRC)
  message(FATAL_ERROR
    "NINFER_WEBUI_SRC is required and always set by the flake (the fetched llama-ui webui "
    "tree). Provide a valid webui tree via -DNINFER_WEBUI_SRC=<dir> containing index.html.")
endif()
if(NOT EXISTS "${NINFER_WEBUI_SRC}/index.html")
  message(FATAL_ERROR
    "NINFER_WEBUI_SRC='${NINFER_WEBUI_SRC}' does not contain index.html; it must be a built llama.cpp webui tree.")
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
