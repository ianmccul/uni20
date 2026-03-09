#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <fmt/core.h>
#include <cmath>

using namespace uni20::async;

namespace
{

double f(double x, double y) { return x + 2.0 * y; }
double g(double y) { return y * y - 1.0; }
double h(double u, double z) { return 0.5 * u + 0.25 * z; }

Async<double> f(Async<double> const& x, Async<double> const& y) { return x + 2.0 * y; }
Async<double> g(Async<double> const& y) { return y * y - 1.0; }
Async<double> h(Async<double> const& u, Async<double> const& z) { return 0.5 * u + 0.25 * z; }

Async<double> compact_expression_form(Async<double> const& x, Async<double> const& y)
{
  Async<double> z = f(x, y);
  Async<double> u = g(y);
  z += h(u, z);
  return z;
}

Async<double> explicit_kernel_form(Async<double> const& x, Async<double> const& y)
{
  Async<double> z;
  Async<double> u;

  schedule([](ReadBuffer<double> x_in, ReadBuffer<double> y_in, WriteBuffer<double> z_out) static->AsyncTask {
    co_await z_out = f(co_await x_in, co_await y_in);
  }(x.read(), y.read(), z.write()));

  schedule([](ReadBuffer<double> y_in, WriteBuffer<double> u_out) static->AsyncTask {
    co_await u_out = g(co_await y_in);
  }(y.read(), u.write()));

  schedule([](ReadBuffer<double> u_in, WriteBuffer<double> z_io) static->AsyncTask {
    auto [uval, zval] = co_await all(u_in, z_io);
    zval += h(uval, zval);
  }(u.read(), z.write()));

  return z;
}

} // namespace

int main()
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<double> x = 2.0;
  Async<double> y = 3.0;

  auto z_compact = compact_expression_form(x, y);
  auto z_explicit = explicit_kernel_form(x, y);

  double const compact_result = z_compact.get_wait();
  double const explicit_result = z_explicit.get_wait();
  double const abs_diff = std::abs(compact_result - explicit_result);

  fmt::print("compact expression result : {:.6f}\n", compact_result);
  fmt::print("explicit kernel result    : {:.6f}\n", explicit_result);
  fmt::print("absolute difference       : {:.3e}\n", abs_diff);

  return 0;
}
