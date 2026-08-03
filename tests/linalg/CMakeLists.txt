add_test_module(linalg
  SOURCES
    cpu/test_matrix_exponential.cpp
    test_async_dispatch.cpp
    test_async_matrix_product.cpp
    test_async_self_adjoint_eigh.cpp
    test_async_svd.cpp
    test_async_truncated_svd.cpp
    test_async_tensor_reductions.cpp
    test_async_tensor_transform.cpp
    test_cpu_tensor_dispatch.cpp
    test_cpu_strided_transform.cpp
    test_dispatch_diagnostics.cpp
    test_dense_ops_dispatch.cpp
    test_linalg_gemv_cpu.cpp
    test_matrix_product.cpp
    test_self_adjoint_eigh.cpp
    test_svd.cpp
    test_truncated_svd.cpp
    test_tensor_conjugate_inplace.cpp
    test_tensor_copy.cpp
    test_tensor_generated.cpp
    test_tensor_reductions.cpp
    test_tensor_reshape.cpp
    test_tensor_transform.cpp
  LIBS uni20_linalg uni20_async
)

if(UNI20_BACKEND_BLAS)
  add_test_module(linalg_blas
    SOURCES
      test_blas_matrix_operands.cpp
      test_blas_vector_operands.cpp
      test_blas_gemm.cpp
      test_blas_gemv.cpp
      test_linalg_gemm_dispatch.cpp
      test_linalg_gemv_dispatch.cpp
    LIBS uni20_linalg uni20_backend_blas
  )
endif()

if(UNI20_ENABLE_MPLAPACK)
  add_executable(uni20_linalg_mplapack_binary128_tests
    test_mplapack_binary128_cpu_ops.cpp
    "${PROJECT_SOURCE_DIR}/tests/uni20_tests_main.cpp"
  )
  target_link_libraries(uni20_linalg_mplapack_binary128_tests
    PRIVATE
      uni20_linalg
      mplapack::mplapack_binary128
      GTest::gtest
  )
  target_compile_definitions(uni20_linalg_mplapack_binary128_tests PRIVATE UNIT_TEST)
  gtest_discover_tests(uni20_linalg_mplapack_binary128_tests)
endif()

if(TARGET uni20_backend_cuda)
  add_test_module(linalg_cuda
    SOURCES
      test_cuda_conjugate_inplace.cpp
      test_cuda_copy.cpp
    LIBS uni20_linalg uni20_async uni20_backend_cuda
  )
  target_sources(uni20_linalg_cuda_tests PRIVATE test_cuda_descriptor_compile.cu)
  set_target_properties(uni20_linalg_cuda_tests PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
endif()
