# tests/tensor/CMakeLists.txt

add_test_module(tensor
  SOURCES
    test_tensor.cpp
    test_tensor_conjugate.cpp
    test_tensor_output.cpp
  LIBS uni20_common uni20_core
)

add_test_module(tensor_async
  SOURCES test_tensor_async.cpp
  LIBS uni20_common uni20_core uni20_async
)
