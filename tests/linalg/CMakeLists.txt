add_test_module(linalg
  SOURCES
    cpu/test_matrix_exponential.cpp
  LIBS uni20_linalg
)

if(UNI20_BACKEND_BLAS)
  add_test_module(linalg_blas
    SOURCES
      test_blas_matrix_operands.cpp
      test_blas_gemm.cpp
      test_linalg_gemm_dispatch.cpp
    LIBS uni20_linalg uni20_backend_blas
  )
endif()

if(UNI20_ENABLE_MPLAPACK)
  add_executable(uni20_linalg_mplapack_binary128_tests test_mplapack_binary128_cpu_ops.cpp)
  target_link_libraries(uni20_linalg_mplapack_binary128_tests
    PRIVATE
      uni20_linalg
      mplapack::mplapack_binary128
      GTest::gtest_main
  )
  target_compile_definitions(uni20_linalg_mplapack_binary128_tests PRIVATE UNIT_TEST)
  gtest_discover_tests(uni20_linalg_mplapack_binary128_tests)
endif()
