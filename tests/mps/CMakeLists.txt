# tests/mps/CMakeLists.txt

add_test_module(mps
  SOURCES test_finite_mps.cpp
  LIBS uni20_common uni20_core uni20_operator uni20_symmetry uni20_tensorcontraction_runtime uni20_mps
)
