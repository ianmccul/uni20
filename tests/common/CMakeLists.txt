# tests/common/CMakeLists.txt

add_test_module(common
  SOURCES test_terminal_color.cpp test_types.cpp test_trace.cpp test_trace_debug.cpp test_trace_ndebug.cpp
  LIBS uni20_common mdspan
)
