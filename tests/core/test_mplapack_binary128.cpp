#include <uni20/backend/blas/backend_blas.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_io.hpp>

#include <fmt/format.h>
#include <mplapack_binary128.h>
#include <mplapack_config.h>

#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
#include <mpblas_binary128.h>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <type_traits>
#include <vector>

namespace
{

using Binary128 = uni20::float128;
using ComplexBinary128 = uni20::complex<Binary128>;

Binary128 abs_error(Binary128 actual, Binary128 expected) { return std::abs(actual - expected); }

Binary128 binary_power_of_two(int exponent)
{
  Binary128 result{1};
  if (exponent >= 0)
  {
    for (int i = 0; i < exponent; ++i)
    {
      result *= Binary128{2};
    }
  }
  else
  {
    for (int i = 0; i < -exponent; ++i)
    {
      result /= Binary128{2};
    }
  }
  return result;
}

Binary128 below_double_resolution_gap() { return binary_power_of_two(-80); }

void expect_gap_is_binary128_only(Binary128 gap)
{
  EXPECT_TRUE(Binary128{1} + gap > Binary128{1});
  EXPECT_EQ(static_cast<double>(Binary128{1} + gap), 1.0);
}

} // namespace

TEST(MplapackBinary128Test, Uni20NumericLimitsSeesBackendScalar)
{
  static_assert(std::is_same_v<uni20::float128, mplapack_binary128_t>);
  static_assert(std::is_same_v<uni20::complex256, uni20::complex<uni20::float128>>);
  static_assert(uni20::numeric_limits<Binary128>::is_specialized);
  EXPECT_GT(uni20::numeric_limits<Binary128>::epsilon(), Binary128{0});
}

TEST(MplapackBinary128Test, Uni20ScalarConceptsSeeBackendScalar)
{
  static_assert(uni20::Real<Binary128>);
  static_assert(uni20::LapackReal<Binary128>);
  static_assert(uni20::LapackScalar<Binary128>);
  static_assert(uni20::LapackRealOrComplex<Binary128>);
  static_assert(uni20::LapackRealOrComplex<ComplexBinary128>);
  static_assert(uni20::LapackComplex<ComplexBinary128>);
  static_assert(uni20::LapackComplexReal<Binary128>);
  static_assert(uni20::LapackScalar<ComplexBinary128>);
#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  static_assert(uni20::BlasReal<Binary128>);
  static_assert(uni20::BlasScalar<Binary128>);
  static_assert(uni20::BlasComplex<ComplexBinary128>);
  static_assert(uni20::BlasScalar<ComplexBinary128>);
#else
  static_assert(!uni20::BlasReal<Binary128>);
  static_assert(!uni20::BlasComplex<ComplexBinary128>);
#endif
}

TEST(MplapackBinary128Test, Uni20ScalarIoParsesAndFormatsConfiguredFloat128)
{
  Binary128 const parsed = uni20::parse_real<Binary128>("1.000000000000000000000000000000001");
  Binary128 const gap = parsed - Binary128{1};

#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  expect_gap_is_binary128_only(gap);
#endif

  uni20::scalar_format_options options;
  options.precision = 36;
  std::string const formatted = uni20::format_real(parsed, options);
  EXPECT_TRUE(uni20::parse_real<Binary128>(formatted) == parsed);

  ComplexBinary128 const complex_value{parsed, -gap};
  std::string const complex_formatted = uni20::format_complex(complex_value, options);
  EXPECT_NE(complex_formatted.find("-"), std::string::npos);
  EXPECT_NE(complex_formatted.find("i"), std::string::npos);

  std::string const fmt_formatted = fmt::format("{}", parsed);
  EXPECT_TRUE(uni20::parse_real<Binary128>(fmt_formatted) == parsed);
}

TEST(MplapackBinary128Test, Uni20ScalarIoReadsConfiguredFloat128FromStreamToken)
{
  std::istringstream input("1.000000000000000000000000000000001");
  Binary128 parsed = {};

  uni20::read_real(input, parsed);

  EXPECT_TRUE(input);
#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  expect_gap_is_binary128_only(parsed - Binary128{1});
#endif
}

TEST(MplapackBinary128Test, SolvesTinySymmetricEigenproblem)
{
  mplapackint constexpr n = 2;
  Binary128 a[4] = {
      Binary128{2},
      Binary128{1},
      Binary128{1},
      Binary128{2},
  };
  Binary128 w[2] = {};
  mplapackint info = 0;
  mplapackint lwork = -1;
  Binary128 work_query = {};

  Rsyev("V", "U", n, a, n, w, &work_query, lwork, info);
  ASSERT_EQ(info, 0);

  lwork = static_cast<mplapackint>(work_query);
  ASSERT_GT(lwork, 0);

  std::vector<Binary128> work(static_cast<std::size_t>(lwork));
  Rsyev("V", "U", n, a, n, w, work.data(), lwork, info);
  ASSERT_EQ(info, 0);

  Binary128 const tolerance = static_cast<Binary128>(1.0e-25L);
  Binary128 const error = std::max(abs_error(w[0], Binary128{1}), abs_error(w[1], Binary128{3}));
  EXPECT_TRUE(error <= tolerance);
}

TEST(MplapackBinary128Test, LinksMpblasTransitively)
{
#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  mplapackint constexpr n = 2;
  Binary128 const one = Binary128{1};
  Binary128 const zero = Binary128{0};
  Binary128 a[4] = {
      Binary128{1},
      Binary128{3},
      Binary128{2},
      Binary128{4},
  };
  Binary128 b[4] = {
      Binary128{5},
      Binary128{7},
      Binary128{6},
      Binary128{8},
  };
  Binary128 c[4] = {};

  Rgemm("N", "N", n, n, n, one, a, n, b, n, zero, c, n);

  Binary128 const tolerance = static_cast<Binary128>(1.0e-25L);
  Binary128 const error = std::max({abs_error(c[0], Binary128{19}), abs_error(c[1], Binary128{43}),
                                    abs_error(c[2], Binary128{22}), abs_error(c[3], Binary128{50})});
  EXPECT_TRUE(error <= tolerance);
#else
  GTEST_SKIP() << "configured MPLAPACK binary128 mode aliases long double, so MPBLAS binary128 wrappers are disabled";
#endif
}

TEST(MplapackBinary128Test, Uni20BlasWrappersPreserveBinary128OnlyIncrements)
{
#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const one_plus_delta = Binary128{1} + delta;
  expect_gap_is_binary128_only(delta);

  Binary128 a[2] = {one_plus_delta, Binary128{-1}};
  Binary128 b[2] = {Binary128{1}, Binary128{1}};
  Binary128 c[1] = {};
  uni20::blas::gemm('N', 'N', 1, 1, 2, Binary128{1}, a, 1, b, 2, Binary128{}, c, 1);

  EXPECT_TRUE(c[0] > Binary128{});
  EXPECT_TRUE(abs_error(c[0], delta) <= static_cast<Binary128>(1.0e-25L));
#else
  GTEST_SKIP() << "configured MPLAPACK binary128 mode aliases long double, so MPBLAS binary128 wrappers are disabled";
#endif
}
