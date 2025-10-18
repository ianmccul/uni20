# tests/async/CMakeLists.txt

add_test_module(async
  SOURCES test_async_basic.cpp test_async_awaiters.cpp test_async_concepts.cpp
          test_async_task_await.cpp test_async_write_to.cpp test_async_ops.cpp
          test_async_destroy.cpp test_future_value.cpp test_reverse_value.cpp test_dual.cpp
  LIBS uni20_common uni20_async
)

# Put tests that use TBB into a separate module, since thread creation and death tests do not work well together
# https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
add_test_module(async_tbb
  SOURCES test_tbb_scheduler.cpp test_scheduler_stress.cpp
  LIBS uni20_common uni20_async TBB::tbb
)
