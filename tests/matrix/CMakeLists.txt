# tests/matrix/CMakeLists.txt

add_test_module(matrix
  SOURCES test_sparse_matrix.cpp test_sparse_matrix_ops.cpp
  LIBS uni20_common uni20_core uni20_matrix
)
