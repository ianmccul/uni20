#include "async/async_ops.hpp"
#include "async/debug_scheduler.hpp"
#include "async/dual.hpp"
#include "async/dual_toys.hpp"

using namespace uni20::async;

Dual<double> loss_fn(Dual<double> x) { return 0.5 * (x - 3.0) * sin(x - 4.5); }

Async<double> gradient_descent(Async<double> x_in)
{
  async_print("Current x_in: {}\n", x_in);
  Dual<double> x = x_in;
  Dual<double> loss = loss_fn(x);
  loss.grad = 1.0;
  async_print("loss = {}\n", loss.value);
  async_print("loss gradient = {}\n", x.grad.final());
  return x.value - x.grad.final() * 0.1;
}

Async<double> solve(double InitialValue)
{
  Async<double> x = InitialValue;
  for (int i = 0; i < 100; ++i)
  {
    x = gradient_descent(x);
  }
  return x;
}
int main()
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  auto x = solve(10.0);

  TRACE("here");

  fmt::print("Solution is: {}\n", x.get_wait());

  TRACE("finished");
}
