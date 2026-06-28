#include <uni20/backend/lapack/lapack.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

TEST(LAPACKWrapper, GesvSolvesDoubleSystem)
{
  uni20::blas_int const n = 2;
  uni20::blas_int const nrhs = 1;
  std::vector<double> a{
      3.0,
      1.0,
      1.0,
      2.0,
  };
  std::vector<double> b{9.0, 8.0};
  std::vector<uni20::blas_int> ipiv(static_cast<std::size_t>(n));

  uni20::lapack::gesv(n, nrhs, a.data(), n, ipiv.data(), b.data(), n);

  EXPECT_NEAR(b[0], 2.0, 1e-12);
  EXPECT_NEAR(b[1], 3.0, 1e-12);
}
