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
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
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
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
find_package(Threads REQUIRED)

# Nixpkgs-provided libraries that were previously vendored in-tree. Discovery relies
# on the devShell/package environment: CMAKE_PREFIX_PATH drives find_path (nlohmann) and
# find_package (spdlog); PKG_CONFIG_PATH drives pkg_check_modules (httplib, utf8proc). Both are set
# from the buildInputs' dev outputs.
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
  message(FATAL_ERROR
    "nlohmann/json.hpp not found. Add nixpkgs nlohmann_json to the build environment.")
endif()
pkg_check_modules(HTTPLIB REQUIRED IMPORTED_TARGET httplib)
pkg_check_modules(UTF8PROC REQUIRED IMPORTED_TARGET libutf8proc)

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  # Product operational logging uses the distribution spdlog (static build, external fmt).
  # Core-only builds do not configure or link it.
  find_package(spdlog CONFIG REQUIRED)
endif()

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
