# benchmarks/CMakeLists.txt

# Create a benchmark executable.
add_executable(uni20_benchmarks
#    benchmark_dummy.cpp
    benchmark_coroutine_overhead.cpp
)

target_link_libraries(uni20_benchmarks
    PRIVATE uni20_common
            mdspan
            benchmark::benchmark
)

# Disable TRACE for benchmark builds
target_compile_definitions(uni20_benchmarks PRIVATE TRACE_DISABLE=1)

add_custom_target(run_benchmarks
    COMMAND uni20_benchmarks
    DEPENDS uni20_benchmarks
)
