# tests/symmetry/CMakeLists.txt

add_test_module(symmetry
  SOURCES test_qnum.cpp
  LIBS uni20_common uni20_core uni20_symmetry
)
