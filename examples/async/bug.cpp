#include <iostream>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>

using namespace uni20::async;

int main(int argc, char** argv)
{
  int max_concurrency = 4; // default
  if (argc > 1)
  {
    max_concurrency = std::stoi(argv[1]);
  }

  std::cout << "Running SimpleAsync with TbbScheduler(max_concurrency=" << max_concurrency << ")\n";

  TbbScheduler sched{max_concurrency};
  ScopedScheduler guard(&sched);

  Async<int> x = 0;

  const int iterations = 100;
  for (int i = 0; i < iterations; ++i)
  {
    x += 1;
  }

  // sched.run_all();
  int result = x.get_wait();

  std::cout << "Final result = " << result << " (expected " << iterations << ")\n";
}
