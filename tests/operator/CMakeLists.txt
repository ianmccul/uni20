# tests/operator/CMakeLists.txt

add_test_module(operator
  SOURCES test_local_space.cpp test_local_operator.cpp test_operator_component.cpp test_finite_triangular_mpo.cpp
  LIBS uni20_common uni20_core uni20_matrix uni20_symmetry uni20_operator
)
