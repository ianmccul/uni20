# tests/tensor/CMakeLists.txt

add_test_module(tensor
  SOURCES test_basic_tensor.cpp
  LIBS uni20_common uni20_core
)
