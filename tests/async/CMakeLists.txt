# tests/async/CMakeLists.txt

add_test_module(async
  SOURCES test_async_basic.cpp test_async_awaiters.cpp test_async_concepts.cpp
          test_async_task_await.cpp test_async_write_to.cpp test_async_ops.cpp
          test_async_destroy.cpp test_async_task_lifetime.cpp test_future_value.cpp test_reverse_value.cpp test_var.cpp
          test_async_deferred.cpp test_async_emplace.cpp test_async_default_init_threads.cpp
          test_async_move.cpp test_async_alias.cpp test_async_toys.cpp test_shared_storage.cpp
          test_task_registry.cpp
  LIBS uni20_common uni20_async
)

# Put tests that use TBB into a separate module, since thread creation and death tests do not work well together
# https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
add_test_module(async_tbb
  SOURCES test_tbb_numa_scheduler.cpp test_scheduler_stress.cpp test_tbb_scheduler.cpp
  LIBS uni20_common uni20_async TBB::tbb
)

if(UNI20_DEBUG_ASYNC_TASKS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(uni20_task_registry_signal_startup_test test_task_registry_signal_startup.cpp)
  target_link_libraries(uni20_task_registry_signal_startup_test PRIVATE uni20_async)
  add_test(
    NAME TaskRegistryDebugTest.SignalEnvironmentStartsDiagnosticsServiceAtStartup
    COMMAND
      ${CMAKE_COMMAND} -E env
      UNI20_DEBUG_DAG_SIGNAL=SIGUSR1
      UNI20_DEBUG_DAG_POLL_MS=10
      UNI20_DEBUG_DAG_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/task_registry_signal_startup
      UNI20_DEBUG_DAG_FILE_PREFIX=startup
      $<TARGET_FILE:uni20_task_registry_signal_startup_test>
  )
endif()
