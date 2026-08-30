
# src/uni20/async/CMakeLists.txt

if(UNI20_DEBUG_ASYNC_TASKS)
    find_package(Threads REQUIRED)
    add_library(uni20_async)
    target_sources(uni20_async PRIVATE task_registry_debug.cpp)
    target_link_libraries(uni20_async PUBLIC uni20_common Threads::Threads)
    target_compile_definitions(uni20_async PUBLIC TBB_PREVIEW_PARALLEL_PHASE=1)
else()
  add_library(uni20_async INTERFACE)
  target_link_libraries(uni20_async INTERFACE uni20_common)
  target_compile_definitions(uni20_async INTERFACE TBB_PREVIEW_PARALLEL_PHASE=1)
endif()
