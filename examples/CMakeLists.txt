# examples/CMakeLists.txt

# Add an executable target for the dependencies example.
add_executable(mdspan_example mdspan_example.cpp)

# Link the dependencies example with fmt and, if necessary, the uni20 library.
# (If mdspan is fetched via FetchContent, its include directories should already be set.)
target_link_libraries(mdspan_example PRIVATE fmt::fmt mdspan uni20_common)

add_executable(trace_example trace_example.cpp)
target_link_libraries(trace_example PRIVATE fmt::fmt mdspan uni20_common)


if(TARGET uni20_backend_blas)
  add_executable(blas_example blas_example.cpp)
  target_link_libraries(blas_example PRIVATE fmt::fmt uni20_backend_blas)
endif()

add_executable(async_example async_example.cpp)
target_link_libraries(async_example PRIVATE fmt::fmt uni20_common uni20_async)

add_executable(async_example2 async_example2.cpp)
target_link_libraries(async_example2 PRIVATE fmt::fmt uni20_common uni20_async)

add_executable(async_ops_example async_ops_example.cpp)
target_link_libraries(async_ops_example PRIVATE fmt::fmt uni20_common uni20_async)

add_executable(future_example future_example.cpp)
target_link_libraries(future_example PRIVATE fmt::fmt uni20_common uni20_async)

add_executable(async_fib_example async_fib_example.cpp)
target_link_libraries(async_fib_example PRIVATE fmt::fmt uni20_common uni20_async)

add_executable(ad_example ad_example.cpp)
target_link_libraries(ad_example PRIVATE fmt::fmt uni20_common uni20_async)

add_executable(gradient_solver gradient_solver.cpp)
target_link_libraries(gradient_solver PRIVATE fmt::fmt uni20_common uni20_async)

enable_testing()
include(GoogleTest)

add_executable(gtest_floating_eq_example gtest_floating_eq_example.cpp)
target_link_libraries(gtest_floating_eq_example PRIVATE gtest fmt::fmt uni20_common)
