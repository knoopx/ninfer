add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging PkgConfig::HTTPLIB)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
target_include_directories(ninfer-perplexity PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${NLOHMANN_JSON_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/perplexity)
target_link_libraries(ninfer-perplexity PRIVATE ninfer_engine ninfer_product_logging)

# --- Bundled prebuilt webui --------------------------------------------------
# NINFER_WEBUI_SRC is required and always set (enforced at the top level; the flake always
# points it at the fetched llama-ui tree), so the webui is always bundled: copy the tree
# into the build tree so a dev build ships it at ${CMAKE_BINARY_DIR}/share/ninfer/webui
# (i.e. <exe-dir>/../share/ninfer/webui). The target is part of the default build and
# re-runs when the source tree changes.
set(NINFER_WEBUI_DST "${CMAKE_BINARY_DIR}/share/ninfer/webui")
add_custom_command(
  OUTPUT "${NINFER_WEBUI_DST}/index.html"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${NINFER_WEBUI_SRC}" "${NINFER_WEBUI_DST}"
  DEPENDS "${NINFER_WEBUI_SRC}/index.html"
  COMMENT "Bundling llama.cpp webui -> ${NINFER_WEBUI_DST}"
  VERBATIM)
add_custom_target(ninfer-webui ALL DEPENDS "${NINFER_WEBUI_DST}/index.html")
