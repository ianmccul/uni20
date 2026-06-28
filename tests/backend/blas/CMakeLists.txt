# tests/common/CMakeLists.txt

add_test_module(reference_blas
  SOURCES test_reference_blas_gem.cpp test_reference_lapack.cpp test_vendor_backends.cpp
  LIBS uni20_common uni20_backend_blas
)
