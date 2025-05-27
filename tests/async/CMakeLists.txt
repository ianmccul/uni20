# tests/level1/CMakeLists.txt

add_test_module(async
  SOURCES test_async_basic.cpp test_async_awaiters.cpp
  LIBS uni20_common uni20_async
)
