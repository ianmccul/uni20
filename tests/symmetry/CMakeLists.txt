# tests/symmetry/CMakeLists.txt

add_test_module(symmetry
  SOURCES test_qnum.cpp test_u1.cpp test_block_space.cpp
  LIBS uni20_common uni20_core uni20_symmetry
)
