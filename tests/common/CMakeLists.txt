# tests/common/CMakeLists.txt

add_test_module(common
  SOURCES test_terminal_color.cpp test_trace.cpp test_trace_debug.cpp test_trace_ndebug.cpp test_aligned_buffer.cpp
  LIBS uni20_common mdspan
)
