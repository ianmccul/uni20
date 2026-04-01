# tests/models/CMakeLists.txt

add_test_module(models
  SOURCES test_spin_half.cpp test_heisenberg.cpp
  LIBS uni20_common uni20_core uni20_matrix uni20_symmetry uni20_operator uni20_models
)
