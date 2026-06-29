set(_uni20_krylov_test_sources
    test_dense_linalg.cpp
    test_dense_linalg_unused.cpp
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
