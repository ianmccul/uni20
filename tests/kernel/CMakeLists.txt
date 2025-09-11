# tests/kernel/CMakeLists.txt

add_test_module(kernel
  SOURCES test_contract.cpp
  LIBS uni20_common uni20_kernel mdspan
)
