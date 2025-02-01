# benchmarks/CMakeLists.txt

# Create a benchmark executable.
add_executable(uni20_benchmarks
    benchmark_dummy.cpp
)

target_link_libraries(uni20_benchmarks
    PRIVATE uni20
            benchmark::benchmark
)

add_custom_target(run_benchmarks
    COMMAND uni20_benchmarks
    DEPENDS uni20_benchmarks
)
