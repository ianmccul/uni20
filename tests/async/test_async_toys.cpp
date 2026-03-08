#include <uni20/async/async_toys.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/scheduler.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <complex>

using namespace uni20::async;

TEST(AsyncToysTest, SinAndCosMatchStd)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  double const x_value = 0.5;
  Async<double> x = x_value;

  auto y = sin(x);
  auto z = cos(x);

  EXPECT_NEAR(y.get_wait(), std::sin(x_value), 1e-12);
  EXPECT_NEAR(z.get_wait(), std::cos(x_value), 1e-12);
}

TEST(AsyncToysTest, ComplexTransformsMatchStd)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  using complex_t = std::complex<double>;
  complex_t const z_value{2.0, -3.0};
  Async<complex_t> z = z_value;

  auto z_conj = conj(z);
  auto z_real = real(z);
  auto z_imag = imag(z);
  auto z_herm = herm(z);

  EXPECT_EQ(z_conj.get_wait(), std::conj(z_value));
  EXPECT_DOUBLE_EQ(z_real.get_wait(), z_value.real());
  EXPECT_DOUBLE_EQ(z_imag.get_wait(), z_value.imag());
  EXPECT_EQ(z_herm.get_wait(), std::conj(z_value));
}

TEST(AsyncToysTest, CoSinReleasesReadBeforeWriteAcquire)
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  double const x_value = 1.25;
  Async<double> x = x_value;

  auto read_epoch = x.read();
  auto write_epoch = x.write();
  schedule(co_sin(std::move(read_epoch), std::move(write_epoch)));

  sched.run_all();
  EXPECT_NEAR(x.get_wait(), std::sin(x_value), 1e-12);
}
