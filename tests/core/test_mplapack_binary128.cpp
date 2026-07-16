#include <uni20/backend/blas/backend_blas.hpp>
#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/common/gtest.hpp>
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
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
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

enum class ComplexSvdDriver
{
  standard,
  divide_and_conquer,
  selected
};

void expect_complex_svd_reconstructs(ComplexSvdDriver driver)
{
  uni20::blas_int constexpr n = 2;
  std::vector<ComplexBinary128> const original{
      ComplexBinary128{Binary128{}, Binary128{3}},
      ComplexBinary128{},
      ComplexBinary128{},
      ComplexBinary128{Binary128{2}, Binary128{-1}},
  };
  auto matrix = original;
  std::array<Binary128, n> singular_values{};
  std::array<ComplexBinary128, n * n> left{};
  std::array<ComplexBinary128, n * n> right{};
  std::array<Binary128, 256> real_work{};
  std::array<uni20::blas_int, 32> int_work{};
  ComplexBinary128 work_query{};

  if (driver == ComplexSvdDriver::standard)
  {
    uni20::lapack::gesvd('S', 'S', n, n, matrix.data(), n, singular_values.data(), left.data(), n, right.data(), n,
                         &work_query, -1, real_work.data());
  }
  else if (driver == ComplexSvdDriver::divide_and_conquer)
  {
    uni20::lapack::gesdd('S', n, n, matrix.data(), n, singular_values.data(), left.data(), n, right.data(), n,
                         &work_query, -1, real_work.data(), int_work.data());
  }
  else
  {
    uni20::blas_int selected_count = 0;
    uni20::lapack::gesvdx('V', 'V', 'A', n, n, matrix.data(), n, Binary128{}, Binary128{}, 0, 0, selected_count,
                          singular_values.data(), left.data(), n, right.data(), n, &work_query, -1, real_work.data(),
                          int_work.data());
  }

  uni20::blas_int const lwork = std::max<uni20::blas_int>(1, static_cast<uni20::blas_int>(std::real(work_query)));
  std::vector<ComplexBinary128> work(static_cast<std::size_t>(lwork));
  matrix = original;
  uni20::blas_int selected_count = n;

  if (driver == ComplexSvdDriver::standard)
  {
    uni20::lapack::gesvd('S', 'S', n, n, matrix.data(), n, singular_values.data(), left.data(), n, right.data(), n,
                         work.data(), lwork, real_work.data());
  }
  else if (driver == ComplexSvdDriver::divide_and_conquer)
  {
    uni20::lapack::gesdd('S', n, n, matrix.data(), n, singular_values.data(), left.data(), n, right.data(), n,
                         work.data(), lwork, real_work.data(), int_work.data());
  }
  else
  {
    uni20::lapack::gesvdx('V', 'V', 'A', n, n, matrix.data(), n, Binary128{}, Binary128{}, 0, 0, selected_count,
                          singular_values.data(), left.data(), n, right.data(), n, work.data(), lwork, real_work.data(),
                          int_work.data());
    ASSERT_EQ(selected_count, n);
  }

  Binary128 const tolerance = static_cast<Binary128>(1.0e-25L);
  for (uni20::blas_int column = 0; column < n; ++column)
  {
    for (uni20::blas_int row = 0; row < n; ++row)
    {
      ComplexBinary128 reconstructed{};
      for (uni20::blas_int inner = 0; inner < n; ++inner)
      {
        reconstructed += left[static_cast<std::size_t>(row + inner * n)] * singular_values[inner] *
                         right[static_cast<std::size_t>(inner + column * n)];
      }
      EXPECT_TRUE(std::abs(reconstructed - original[static_cast<std::size_t>(row + column * n)]) <= tolerance);
    }
  }
}

} // namespace

TEST(MplapackBinary128Test, Uni20NumericLimitsSeesBackendScalar)
{
  static_assert(std::is_same_v<uni20::float128, mplapack_binary128_t>);
  static_assert(std::is_same_v<uni20::complex256, uni20::complex<uni20::float128>>);
  static_assert(uni20::numeric_limits<Binary128>::is_specialized);
  EXPECT_GT(uni20::numeric_limits<Binary128>::epsilon(), Binary128{0});
}

TEST(MplapackBinary128Test, StandardTypedPiPreservesBinary128Precision)
{
  Binary128 const pi = std::numbers::pi_v<Binary128>;
  Binary128 const widened_double_pi = static_cast<Binary128>(std::numbers::pi);

  EXPECT_GT(pi, Binary128{3});
  EXPECT_LT(pi, Binary128{4});
  EXPECT_NE(pi, widened_double_pi);
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

  EXPECT_FLOATING_EQ(c[0], Binary128{19});
  EXPECT_FLOATING_EQ(c[1], Binary128{43});
  EXPECT_FLOATING_EQ(c[2], Binary128{22});
  EXPECT_FLOATING_EQ(c[3], Binary128{50});
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
  EXPECT_FLOATING_EQ(c[0], delta);
#else
  GTEST_SKIP() << "configured MPLAPACK binary128 mode aliases long double, so MPBLAS binary128 wrappers are disabled";
#endif
}

TEST(MplapackBinary128Test, ComplexLapackNormAndLuWrappers)
{
  uni20::blas_int constexpr n = 2;
  Binary128 const tolerance = static_cast<Binary128>(1.0e-25L);
  std::vector<ComplexBinary128> matrix{
      ComplexBinary128{Binary128{2}, Binary128{1}},
      ComplexBinary128{Binary128{1}, Binary128{-1}},
      ComplexBinary128{Binary128{1}, Binary128{}},
      ComplexBinary128{Binary128{3}, Binary128{0.5}},
  };
  auto const original = matrix;
  std::array<Binary128, n> norm_work{};
  Binary128 const matrix_norm = uni20::lapack::lange('1', n, n, matrix.data(), n, norm_work.data());
  EXPECT_TRUE(matrix_norm > Binary128{});

  std::vector<ComplexBinary128> hermitian{
      ComplexBinary128{Binary128{2}, Binary128{}},
      ComplexBinary128{Binary128{1}, Binary128{-1}},
      ComplexBinary128{Binary128{1}, Binary128{1}},
      ComplexBinary128{Binary128{3}, Binary128{}},
  };
  Binary128 const hermitian_norm = uni20::lapack::lanhe('F', 'U', n, hermitian.data(), n, norm_work.data());
  EXPECT_FLOATING_EQ(hermitian_norm, std::sqrt(Binary128{17}));

  std::vector<ComplexBinary128> triangular{
      ComplexBinary128{Binary128{1}, Binary128{1}},
      ComplexBinary128{},
      ComplexBinary128{Binary128{2}, Binary128{-1}},
      ComplexBinary128{Binary128{-3}, Binary128{0.5}},
  };
  Binary128 const triangular_norm = uni20::lapack::lantr('F', 'U', 'N', n, n, triangular.data(), n, norm_work.data());
  EXPECT_FLOATING_EQ(triangular_norm, std::sqrt(Binary128{16.25}));

  std::array<ComplexBinary128, n> expected{
      ComplexBinary128{Binary128{1}, Binary128{2}},
      ComplexBinary128{Binary128{-1}, Binary128{0.5}},
  };
  std::array<ComplexBinary128, n> right_hand_side{};
  for (uni20::blas_int row = 0; row < n; ++row)
  {
    for (uni20::blas_int column = 0; column < n; ++column)
      right_hand_side[row] += original[static_cast<std::size_t>(row + column * n)] * expected[column];
  }

  auto gesv_matrix = original;
  auto gesv_right_hand_side = right_hand_side;
  std::array<uni20::blas_int, n> gesv_pivots{};
  uni20::lapack::gesv(n, 1, gesv_matrix.data(), n, gesv_pivots.data(), gesv_right_hand_side.data(), n);
  EXPECT_FLOATING_EQ(gesv_right_hand_side[0], expected[0]);
  EXPECT_FLOATING_EQ(gesv_right_hand_side[1], expected[1]);

  std::array<uni20::blas_int, n> pivots{};
  uni20::lapack::getrf(n, n, matrix.data(), n, pivots.data());
  uni20::lapack::getrs('N', n, 1, matrix.data(), n, pivots.data(), right_hand_side.data(), n);

  EXPECT_FLOATING_EQ(right_hand_side[0], expected[0]);
  EXPECT_FLOATING_EQ(right_hand_side[1], expected[1]);

  std::array<ComplexBinary128, 2 * n> condition_work{};
  std::array<Binary128, 2 * n> condition_real_work{};
  Binary128 const reciprocal_condition =
      uni20::lapack::gecon('1', n, matrix.data(), n, matrix_norm, condition_work.data(), condition_real_work.data());
  EXPECT_TRUE(reciprocal_condition > Binary128{});
  EXPECT_TRUE(reciprocal_condition <= Binary128{1});

  ComplexBinary128 inverse_work_query{};
  uni20::lapack::getri(n, matrix.data(), n, pivots.data(), &inverse_work_query, -1);
  uni20::blas_int const inverse_lwork =
      std::max<uni20::blas_int>(1, static_cast<uni20::blas_int>(std::real(inverse_work_query)));
  std::vector<ComplexBinary128> inverse_work(static_cast<std::size_t>(inverse_lwork));
  uni20::lapack::getri(n, matrix.data(), n, pivots.data(), inverse_work.data(), inverse_lwork);

  for (uni20::blas_int column = 0; column < n; ++column)
  {
    for (uni20::blas_int row = 0; row < n; ++row)
    {
      ComplexBinary128 product{};
      for (uni20::blas_int inner = 0; inner < n; ++inner)
      {
        product +=
            original[static_cast<std::size_t>(row + inner * n)] * matrix[static_cast<std::size_t>(inner + column * n)];
      }
      ComplexBinary128 const expected_product =
          row == column ? ComplexBinary128{Binary128{1}, Binary128{}} : ComplexBinary128{};
      EXPECT_TRUE(std::abs(product - expected_product) <= tolerance);
    }
  }
}

TEST(MplapackBinary128Test, ComplexGesvdReconstructsMatrix)
{
  expect_complex_svd_reconstructs(ComplexSvdDriver::standard);
}

TEST(MplapackBinary128Test, ComplexGesddReconstructsMatrix)
{
  expect_complex_svd_reconstructs(ComplexSvdDriver::divide_and_conquer);
}

TEST(MplapackBinary128Test, ComplexGesvdxReconstructsMatrix)
{
  expect_complex_svd_reconstructs(ComplexSvdDriver::selected);
}
