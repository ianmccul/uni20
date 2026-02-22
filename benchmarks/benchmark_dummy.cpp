#include "core/dummy.hpp" // Make sure this header is found via the include directories
#include <benchmark/benchmark.h>

// Benchmark for the heavy computation function.
static void BM_ComputeHeavyOperation(benchmark::State& state)
{
  for (auto _ : state)
  {
    // Using volatile to prevent the compiler from optimizing away the computation.
    volatile double result = uni20::compute_heavy_operation(2.0, 3.0);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_ComputeHeavyOperation);

BENCHMARK_MAIN();
