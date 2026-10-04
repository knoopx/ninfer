add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  gguf.cpp
  binder.cpp
  materializer.cpp
  views.cpp
)
ninfer_internal_includes(ninfer_artifact)
# ninfer_strata is PUBLIC: the GGUF v3 reader header (strata/artifact/gguf_reader.hpp)
# is used by the artifact layer and propagates the Strata include dirs to consumers.
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json ninfer_strata)
