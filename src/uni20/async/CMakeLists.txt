
# src/uni20/async/CMakeLists.txt

if(UNI20_DEBUG_ASYNC_TASKS)
    find_package(Threads REQUIRED)
    add_library(uni20_async)
    target_sources(uni20_async PUBLIC task_registry_debug.cpp)
    target_link_libraries(uni20_async PUBLIC uni20_deps Threads::Threads)
else()
  add_library(uni20_async INTERFACE)
  target_link_libraries(uni20_async INTERFACE uni20_deps)
endif()
