set(_uni20_krylov_test_sources
    test_dense_linalg.cpp
    test_dense_subspace.cpp
    test_known_matrices.cpp
    test_krylov_exponential.cpp
    test_matrix_free_interface.cpp
    test_nonsymmetric_arnoldi.cpp
    test_symmetric_lanczos.cpp
    test_tridiagonal.cpp)

add_test_module(krylov
  SOURCES
    ${_uni20_krylov_test_sources}
  LIBS
    uni20_krylov
)

target_compile_definitions(uni20_krylov_tests PRIVATE
    UNI20_KRYLOV_MATRIX_MARKET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/matrix_market")

add_test_module(krylov_block_tensor
  SOURCES
    test_block_tensor_vector.cpp
    test_dmrg_lanczos.cpp
    test_finite_environment_cache.cpp
    test_mpo_environment.cpp
    test_rabc_contraction.cpp
    test_two_site_dmrg_slice.cpp
    test_two_site_dmrg_sweep.cpp
    test_two_site_split.cpp
  LIBS
    uni20_krylov
    uni20_symmetry
    uni20_tensor_network
)

if(TARGET uni20_backend_cublas AND TARGET uni20_backend_cusolver)
  add_test_module(krylov_block_tensor_cuda
    SOURCES
      test_cuda_rabc_contraction.cpp
    LIBS
      uni20_tensor_network
      uni20_backend_cuda
      uni20_backend_cublas
      uni20_backend_cusolver
  )
endif()

if(UNI20_ENABLE_MPLAPACK)
  add_executable(uni20_krylov_mplapack_binary128_tests
    test_mplapack_binary128_krylov_solvers.cpp
    "${PROJECT_SOURCE_DIR}/tests/uni20_tests_main.cpp"
  )
  target_link_libraries(uni20_krylov_mplapack_binary128_tests
    PRIVATE
      uni20_krylov
      GTest::gtest
  )
  target_compile_definitions(uni20_krylov_mplapack_binary128_tests PRIVATE
      UNIT_TEST
      UNI20_KRYLOV_MATRIX_MARKET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/matrix_market")
  gtest_discover_tests(uni20_krylov_mplapack_binary128_tests)
endif()
