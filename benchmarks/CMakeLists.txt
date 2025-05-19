# benchmarks/CMakeLists.txt

# Create a benchmark executable.

add_executable(uni20_benchmark_contract
  benchmark_contract.cpp
)

add_executable(uni20_benchmark_dummy
  benchmark_dummy.cpp
)
target_link_libraries(uni20_benchmark_contract
  PRIVATE uni20 uni20_common uni20_kernel benchmark::benchmark
)

target_link_libraries(uni20_benchmark_dummy
  PRIVATE uni20 uni20_common uni20_kernel benchmark::benchmark
)

add_custom_target(run_contract_bench
  COMMAND uni20_benchmark_contract
  DEPENDS uni20_benchmark_contract
)

add_custom_target(run_dummy_bench
  COMMAND uni20_benchmark_dummy
  DEPENDS uni20_benchmark_dummy
)