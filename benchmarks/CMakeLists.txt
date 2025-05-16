# benchmarks/CMakeLists.txt

# Create a benchmark executable.
add_executable(uni20_benchmarks
    benchmark_dummy.cpp
)

target_link_libraries(uni20_benchmarks
    PRIVATE uni20
            uni20_common
            uni20_kernel
            benchmark::benchmark
)
target_include_directories(uni20_benchmarks
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src  # 可以找到 contract.hpp 等
)
add_custom_target(run_benchmarks
    COMMAND uni20_benchmarks
    DEPENDS uni20_benchmarks
)
