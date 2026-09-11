find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat libavcodec libavutil libswscale)

# Nixpkgs-provided libraries that were previously vendored in-tree. Discovery relies
# on the devShell/package environment: CMAKE_PREFIX_PATH drives find_path (nlohmann) and
# find_package (spdlog); PKG_CONFIG_PATH drives pkg_check_modules (httplib, utf8proc). Both
# are set from the buildInputs' dev outputs.
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
  message(FATAL_ERROR
    "nlohmann/json.hpp not found. Add nixpkgs nlohmann_json to the build environment.")
endif()
pkg_check_modules(HTTPLIB REQUIRED IMPORTED_TARGET httplib)
pkg_check_modules(UTF8PROC REQUIRED IMPORTED_TARGET libutf8proc)

add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE ${NLOHMANN_JSON_INCLUDE_DIR})

add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
target_link_libraries(ninfer::httplib INTERFACE PkgConfig::HTTPLIB)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  # Product operational logging uses the distribution spdlog (static build, external fmt).
  # Core-only builds do not configure or link it.
  find_package(spdlog CONFIG REQUIRED)
endif()
