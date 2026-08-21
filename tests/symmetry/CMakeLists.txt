# tests/symmetry/CMakeLists.txt

add_test_module(symmetry
  SOURCES test_qnum.cpp test_u1.cpp test_block_space.cpp test_space.cpp test_morphism_boundary.cpp
          test_block_tensor.cpp
          test_block_tensor_permute.cpp
          test_block_tensor_repartition.cpp
          test_dual_space.cpp
  LIBS uni20_common uni20_core uni20_symmetry
)
