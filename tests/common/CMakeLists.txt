# tests/common/CMakeLists.txt

add_test_module(common
  SOURCES test_terminal_color.cpp test_types.cpp
  LIBS uni20_common mdspan
)
