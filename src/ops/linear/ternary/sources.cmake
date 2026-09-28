target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/ternary_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ternary_rotation.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ternary_rotation.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ternary_rowsplit_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ternary_a8.cu"
)
