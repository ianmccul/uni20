/// \file async_tbb_reduction_example.cpp
/// \brief Example of using uni20 coroutines with the TbbScheduler to parallelize a map–reduce computation.
///
/// This example computes the sum of squares from 1 to N using two coroutine kernels:
///   - `square`: squares a single integer
///   - `sum`: sums two integers
///
/// The computation is structured in two phases:
///   1. **Map stage**: a parallel launch of `square` coroutines over all inputs
///   2. **Reduce stage**: a binary-tree reduction using `sum` coroutines
///
/// The scheduler (`TbbScheduler`) manages coroutine resumption across worker threads,
/// while `Async<int>`, `ReadBuffer<int>`, and `WriteBuffer<int>` handle dataflow and
/// dependency tracking between tasks.
///
/// This demonstrates:
///   - Expressing parallel computations as coroutines
///   - Building a DAG of tasks with explicit dependencies
///   - Executing the DAG in parallel using TBB
///
/// Expected output (for N=1000):
/// \code
/// Sum of squares 1..N = 333833500
/// \endcode
///
/// In principle the main thread can get to the last line of main() before any of the worker computations begin.

#include "async/async.hpp"
#include "async/buffers.hpp"
#include "async/tbb_scheduler.hpp"
#include <iostream>
#include <numeric>
#include <vector>

using namespace uni20::async;

// Coroutine: compute the square of an input
AsyncTask square(ReadBuffer<int> in, WriteBuffer<int> out)
{
  int x = co_await in;
  co_await out = x * x;
  co_return;
}

// Coroutine: sum two inputs
AsyncTask sum(ReadBuffer<int> a, ReadBuffer<int> b, WriteBuffer<int> out)
{
  int x = co_await a;
  int y = co_await b;
  co_await out = x + y;
  co_return;
}

int main()
{
  // Scheduler with 4 worker threads
  TbbScheduler sched{4};
  sched.pause(); // for demonstation purposes

  const int N = 1000;
  std::vector<Async<int>> inputs, result;

  // Initialize inputs
  for (int i = 1; i <= N; ++i)
    inputs.emplace_back(i);

  result.resize(N);

  // Map stage: schedule squaring coroutines (parallel across N)
  for (int i = 0; i < N; ++i)
    sched.schedule(square(inputs[i].read(), result[i].write()));

  // Reduction stage: build a binary tree of sums
  int step = 1;
  while (result.size() > 1)
  {
    std::vector<Async<int>> next;
    for (size_t i = 0; i + 1 < result.size(); i += 2)
    {
      Async<int> partial;
      sched.schedule(sum(result[i].read(), result[i + 1].read(), partial.write()));
      next.push_back(std::move(partial));
    }
    if (result.size() % 2 == 1) // odd tail
      next.push_back(std::move(result.back()));
    result = std::move(next);
    step *= 2;
  }

  // start the scheduler
  sched.resume();

  // Final result
  std::cout << "Sum of squares 1..N = " << result[0].get_wait(sched) << "\n";
}
