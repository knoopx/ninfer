add_library(ninfer_text STATIC
  unicode.cpp
)
ninfer_internal_includes(ninfer_text)
target_link_libraries(ninfer_text PUBLIC PkgConfig::UTF8PROC)

# Language evaluation reuses the engine's Unicode and JSON dependencies.
target_sources(ninfer_jinja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jinja.cpp)
ninfer_internal_includes(ninfer_jinja)
target_include_directories(ninfer_jinja PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_jinja PRIVATE ninfer_text ninfer::json)
