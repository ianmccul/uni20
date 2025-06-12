#include "async/async.hpp"
#include "async/async_ops.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <benchmark/benchmark.h>
#include <numeric>

using namespace uni20::async;

static void Baseline(benchmark::State& state)
{
  double x = 0;
  for (auto _ : state)
  {
    x += 1;
    benchmark::DoNotOptimize(x);
  }
}

BENCHMARK(Baseline);

static void Sine(benchmark::State& state)
{
  using std::sin;
  double x = 1;
  for (auto _ : state)
  {
    x += sin(x);
    benchmark::DoNotOptimize(x);
  }
}

BENCHMARK(Sine);

static void SimpleAsync(benchmark::State& state)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> x = 0;
  for (auto _ : state)
    x += 1;
  int result = (x.get_wait());
  benchmark::DoNotOptimize(result);
}

BENCHMARK(SimpleAsync);

static void Binary(benchmark::State& state)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<int> x = 0;
  for (auto _ : state)
    x = x + 1;
  int result = (x.get_wait());
  benchmark::DoNotOptimize(result);
}

BENCHMARK(Binary);

BENCHMARK_MAIN();
