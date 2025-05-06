# tests/common/CMakeLists.txt

add_test_module(reference_blas
  SOURCES test_reference_blas_gem.cpp
  LIBS uni20_common blas
)
