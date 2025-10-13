# tests/common/CMakeLists.txt

add_test_module(common
  SOURCES test_terminal_color.cpp test_trace.cpp test_trace_debug.cpp test_trace_ndebug.cpp
          test_aligned_buffer.cpp test_floating_eq.cpp test_namedenum.cpp test_string_util.cpp
          test_trace_format.cpp test_terminal_utils.cpp
  LIBS uni20_common mdspan
)
