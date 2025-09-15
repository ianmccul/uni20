#include "async/async.hpp"
#include "async/tbb_scheduler.hpp"
#include <iostream>

using namespace uni20::async;

int main(int argc, char** argv)
{
  int threads = 4; // default
  if (argc > 1)
  {
    threads = std::stoi(argv[1]);
  }

  std::cout << "Running SimpleAsync with TbbScheduler(" << threads << " threads)\n";

  TbbScheduler sched{threads};
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
