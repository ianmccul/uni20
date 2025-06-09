// tests/test_dual.cpp
#include "async/async_ops.hpp"
#include "async/debug_scheduler.hpp"
#include "async/dual.hpp"
#include <gtest/gtest.h>

using namespace uni20::async;

TEST(Dual, Sin)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  double v = 0.1;
  Dual<double> x = v;

  Dual<double> y = sin(x);

  EXPECT_NEAR(y.value.get_wait(), std::sin(v), 1e-10);

  y.grad = 1.0; // this seeds the backprop chain, accumulating values into x.grad

  //  x.grad = 0.0; // this starts the backprop chain, the initial value for x.grad
  // x.grad.finalize(); // no longer needed

  EXPECT_NEAR(x.grad.final().get_wait(), std::cos(v), 1e-10);

  sched.run_all();
}

TEST(Dual, Cos)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  double v = 0.1;
  Dual<double> x = v;

  Dual<double> y = cos(x);

  EXPECT_NEAR(y.value.get_wait(), std::cos(v), 1e-10);

  y.grad = 1.0; // this seeds the backprop chain, accumulating values into x.grad

  //  x.grad = 0.0; // this starts the backprop chain, the initial value for x.grad
  // x.grad.finalize(); // no longer needed

  EXPECT_NEAR(x.grad.final().get_wait(), -std::sin(v), 1e-10);

  sched.run_all();
}

TEST(Dual, SinUnused)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  double v = 0.1;

  Dual<double> x = v;

  Dual<double> y = sin(x);

  {
    Dual<double> z = sin(x); // unused
    // z.grad = 0.0;            // without this, we get an exception
  }

  EXPECT_NEAR(y.value.get_wait(), std::sin(v), 1e-10);

  y.grad = 1.0; // this seeds the backprop chain, accumulating values into x.grad

  //  x.grad = 0.0; // this starts the backprop chain, the initial value for x.grad
  // x.grad.finalize(); // no longer needed

  EXPECT_NEAR(x.grad.final().get_wait(), std::cos(v), 1e-10);

  sched.run_all();
}
