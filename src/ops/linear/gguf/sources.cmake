target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gguf.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gguf_kernels.cu"
)
