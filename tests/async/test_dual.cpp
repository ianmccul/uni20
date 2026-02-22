// tests/test_dual.cpp
#include "async/async_ops.hpp"
#include "async/debug_scheduler.hpp"
#include "async/dual.hpp"
#include "async/dual_toys.hpp"
#include <complex>
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

  EXPECT_NEAR(x.grad.backprop().get_wait(), std::cos(v), 1e-10);

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

  EXPECT_NEAR(x.grad.backprop().get_wait(), -std::sin(v), 1e-10);

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
  }

  EXPECT_NEAR(y.value.get_wait(), std::sin(v), 1e-10);

  y.grad = 1.0; // this seeds the backprop chain, accumulating values into x.grad

  EXPECT_NEAR(x.grad.backprop().get_wait(), std::cos(v), 1e-10);

  sched.run_all();
}

TEST(Dual, MultiplyAndScalarCombos)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Dual<double> a = 2.0;
  Dual<double> b = -0.5;

  Dual<double> c = a * b;
  Dual<double> d = a * 3.0;
  Dual<double> e = 4.0 - a;

  EXPECT_NEAR(c.value.get_wait(), -1.0, 1e-12);
  EXPECT_NEAR(d.value.get_wait(), 6.0, 1e-12);
  EXPECT_NEAR(e.value.get_wait(), 2.0, 1e-12);

  c.grad = 1.0;
  d.grad = 1.0;
  e.grad = 1.0;

  sched.run_all();

  EXPECT_NEAR(a.grad.backprop().get_wait(), 1.5, 1e-12);
  EXPECT_NEAR(b.grad.backprop().get_wait(), 2.0, 1e-12);

  sched.run_all();
}

TEST(Dual, CopyAssignment)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Dual<double> source = 0.75;
  Dual<double> target;

  target = source;

  target.grad = 1.0;

  sched.run_all();

  EXPECT_NEAR(source.grad.backprop().get_wait(), 1.0, 1e-12);

  sched.run_all();
}

TEST(Dual, SubtractionOps)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Dual<double> x = 5.0;
  Dual<double> y = -1.5;

  Dual<double> diff_xy = x - y;
  Dual<double> diff_xs = x - 2.0;
  Dual<double> diff_sx = 10.0 - y;

  EXPECT_NEAR(diff_xy.value.get_wait(), 6.5, 1e-12);
  EXPECT_NEAR(diff_xs.value.get_wait(), 3.0, 1e-12);
  EXPECT_NEAR(diff_sx.value.get_wait(), 11.5, 1e-12);

  diff_xy.grad = 1.0;
  diff_xs.grad = 1.0;
  diff_sx.grad = 1.0;

  sched.run_all();

  EXPECT_NEAR(x.grad.backprop().get_wait(), 2.0, 1e-12);
  EXPECT_NEAR(y.grad.backprop().get_wait(), -2.0, 1e-12);

  sched.run_all();
}

TEST(Dual, RealImagGradients)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Dual<std::complex<double>> z = std::complex<double>{1.5, -2.5};

  auto r = real(z);
  auto i = imag(z);

  EXPECT_NEAR(r.value.get_wait(), 1.5, 1e-12);
  EXPECT_NEAR(i.value.get_wait(), -2.5, 1e-12);

  r.grad = 2.0;
  i.grad = 3.0;

  sched.run_all();

  auto z_grad = z.grad.backprop().get_wait();
  EXPECT_NEAR(z_grad.real(), 2.0, 1e-12);
  EXPECT_NEAR(z_grad.imag(), 3.0, 1e-12);

  sched.run_all();
}

TEST(Dual, RealImagGradientSum)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Dual<std::complex<double>> z = std::complex<double>{1.5, -2.5};

  auto f = 2.0 * real(z) + 3.0 * imag(z);

  EXPECT_NEAR(f.value.get_wait(), -4.5, 1e-12);

  f.grad = 1.0;

  sched.run_all();

  auto z_grad = z.grad.backprop().get_wait();
  EXPECT_NEAR(z_grad.real(), 2.0, 1e-12);
  EXPECT_NEAR(z_grad.imag(), 3.0, 1e-12);

  sched.run_all();
}

TEST(Dual, RealImagGradientsImagSeedFirst)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  Dual<std::complex<double>> z = std::complex<double>{1.5, -2.5};

  auto r = real(z);
  auto i = imag(z);

  EXPECT_NEAR(r.value.get_wait(), 1.5, 1e-12);
  EXPECT_NEAR(i.value.get_wait(), -2.5, 1e-12);

  // Seed imag first to exercise the "construct from imag contribution" path.
  i.grad = 3.0;
  r.grad = 2.0;

  sched.run_all();

  auto z_grad = z.grad.backprop().get_wait();
  EXPECT_NEAR(z_grad.real(), 2.0, 1e-12);
  EXPECT_NEAR(z_grad.imag(), 3.0, 1e-12);

  sched.run_all();
}

TEST(Dual, StressBackpropMatchesAnalytic)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  double const base_value = 0.375;
  Dual<double> x = base_value;
  Dual<double> total = 0.0;

  constexpr int kTerms = 128;
  double expected_value = 0.0;
  double expected_grad = 0.0;

  for (int term_index = 0; term_index < kTerms; ++term_index)
  {
    double const shift = static_cast<double>(term_index) * 0.0025;
    Dual<double> term = sin(x + shift) * cos(x - shift);
    Dual<double> new_total = total + term;
    total = new_total; // std::move(new_total);

    double const plus = base_value + shift;
    double const minus = base_value - shift;
    double const term_value = std::sin(plus) * std::cos(minus);
    expected_value += term_value;
    double const derivative = std::cos(plus) * std::cos(minus) - std::sin(plus) * std::sin(minus);
    expected_grad += derivative;
  }

  double const actual_value = total.value.get_wait();
  EXPECT_NEAR(actual_value, expected_value, 1e-12);

  total.grad = 1.0;
  sched.run_all();

  double const actual_grad = x.grad.backprop().get_wait();
  EXPECT_NEAR(actual_grad, expected_grad, 1e-12);

  sched.run_all();
}
