#include "async/async.hpp"
#include "async/async_ops.hpp"
#include "async/async_toys.hpp"
#include "async/awaiters.hpp"
#include "async/debug_scheduler.hpp"

using namespace uni20::async;

int main()
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Async<double> x;

  async_read("Enter a number: ", x);

  // These two blocks run in parallel...

  auto y = x + 20;
  async_print("Number + 20 = {}\n", y);
  y += 10;
  async_print("Number + 20 + 10 = {}\n", y);

  // Assignment to y starts a new DAG
  y = sin(x);
  async_print("sin(number) = {}\n", y);
  y = sin(y);
  async_print("sin(sin(number)) = {}\n", y);

  TRACE("first round");
  sched.run();
  sched.run();
  TRACE("second round");
  sched.run();
  sched.run();
  TRACE("third round");
  sched.run();
  TRACE("running all remaining...");
  sched.run_all();
}
