# tests/core/CMakeLists.txt

add_test_module(core
  SOURCES test_types.cpp test_scalar_traits.cpp test_scalar_concepts.cpp
  LIBS uni20_core
)
