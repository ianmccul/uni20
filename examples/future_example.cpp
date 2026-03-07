#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/async_toys.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/future_value.hpp>

using namespace uni20::async;

int main()
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  FutureValue<double> x;

  auto y = x.value() + 20;
  async_print("Number + 20 = {}\n", y);
  y = sin(x.value());
  async_print("sin(number) = {}\n", y);

  sched.run_all(); // won't be able to do much
  fmt::print("Enter a number: ");
  double d;
  std::cin >> d;
  x = d;

  // now we can run something
  sched.run_all();
}
