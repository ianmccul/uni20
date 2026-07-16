#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/common/gtest.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
class ErrorsAbortGuard {
  public:
    explicit ErrorsAbortGuard(bool errors_abort) : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(errors_abort);
    }

    ~ErrorsAbortGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

template <class Complex> using real_type_t = typename Complex::value_type;

template <class Complex> real_type_t<Complex> tolerance()
{
  if constexpr (std::same_as<real_type_t<Complex>, float>)
    return 5.0e-5F;
  else
    return 1.0e-12;
}

template <class Complex>
void expect_complex_near(Complex actual, Complex expected, real_type_t<Complex> error = tolerance<Complex>())
{
  EXPECT_NEAR(actual.real(), expected.real(), error);
  EXPECT_NEAR(actual.imag(), expected.imag(), error);
}

template <class Complex> std::vector<Complex> complex_svd_input()
{
  using Real = real_type_t<Complex>;
  return {
      Complex{Real{1}, Real{1}},  Complex{Real{-2}, Real{0.5}}, Complex{Real{3}, Real{-1.5}},
      Complex{Real{2}, Real{-1}}, Complex{Real{0.5}, Real{2}},  Complex{Real{-1}, Real{-2}},
  };
}

enum class ComplexSvdDriver
{
  standard,
  divide_and_conquer,
  selected
};

template <class Complex> void check_complex_svd_reconstruction(ComplexSvdDriver driver)
{
  using Real = real_type_t<Complex>;
  uni20::blas_int constexpr m = 3;
  uni20::blas_int constexpr n = 2;
  uni20::blas_int constexpr k = 2;
  auto const original = complex_svd_input<Complex>();
  auto matrix = original;
  std::vector<Real> singular_values(k);
  std::vector<Complex> left(static_cast<std::size_t>(m * k));
  std::vector<Complex> right(static_cast<std::size_t>(k * n));
  std::vector<Real> real_work(512);
  std::vector<uni20::blas_int> int_work(64);
  Complex work_query{};

  if (driver == ComplexSvdDriver::standard)
  {
    uni20::lapack::gesvd('S', 'S', m, n, matrix.data(), m, singular_values.data(), left.data(), m, right.data(), k,
                         &work_query, -1, real_work.data());
  }
  else if (driver == ComplexSvdDriver::divide_and_conquer)
  {
    uni20::lapack::gesdd('S', m, n, matrix.data(), m, singular_values.data(), left.data(), m, right.data(), k,
                         &work_query, -1, real_work.data(), int_work.data());
  }
  else
  {
    uni20::blas_int selected_count = 0;
    uni20::lapack::gesvdx('V', 'V', 'A', m, n, matrix.data(), m, Real{}, Real{}, 0, 0, selected_count,
                          singular_values.data(), left.data(), m, right.data(), k, &work_query, -1, real_work.data(),
                          int_work.data());
  }

  uni20::blas_int const lwork = std::max<uni20::blas_int>(1, static_cast<uni20::blas_int>(std::real(work_query)));
  std::vector<Complex> work(static_cast<std::size_t>(lwork));
  matrix = original;
  uni20::blas_int selected_count = k;

  if (driver == ComplexSvdDriver::standard)
  {
    uni20::lapack::gesvd('S', 'S', m, n, matrix.data(), m, singular_values.data(), left.data(), m, right.data(), k,
                         work.data(), lwork, real_work.data());
  }
  else if (driver == ComplexSvdDriver::divide_and_conquer)
  {
    uni20::lapack::gesdd('S', m, n, matrix.data(), m, singular_values.data(), left.data(), m, right.data(), k,
                         work.data(), lwork, real_work.data(), int_work.data());
  }
  else
  {
    uni20::lapack::gesvdx('V', 'V', 'A', m, n, matrix.data(), m, Real{}, Real{}, 0, 0, selected_count,
                          singular_values.data(), left.data(), m, right.data(), k, work.data(), lwork, real_work.data(),
                          int_work.data());
    ASSERT_EQ(selected_count, k);
  }

  EXPECT_GE(singular_values[0], singular_values[1]);
  EXPECT_GE(singular_values[1], Real{});
  for (uni20::blas_int column = 0; column < n; ++column)
  {
    for (uni20::blas_int row = 0; row < m; ++row)
    {
      Complex reconstructed{};
      for (uni20::blas_int inner = 0; inner < k; ++inner)
      {
        reconstructed += left[static_cast<std::size_t>(row + inner * m)] * singular_values[inner] *
                         right[static_cast<std::size_t>(inner + column * k)];
      }
      expect_complex_near(reconstructed, original[static_cast<std::size_t>(row + column * m)],
                          Real{20} * tolerance<Complex>());
    }
  }
}

template <class Complex> class ComplexLAPACKWrapperTest : public ::testing::Test {};

using ComplexTypes = ::testing::Types<uni20::complex<float>, uni20::complex<double>>;
TYPED_TEST_SUITE(ComplexLAPACKWrapperTest, ComplexTypes);
} // namespace

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

  EXPECT_FLOATING_EQ(b[0], 2.0);
  EXPECT_FLOATING_EQ(b[1], 3.0);
}

TEST(LAPACKWrapper, GesvRaisesStructuredErrorForSingularSystem)
{
  uni20::blas_int const n = 2;
  uni20::blas_int const nrhs = 1;
  std::vector<double> a{
      1.0,
      2.0,
      2.0,
      4.0,
  };
  std::vector<double> b{1.0, 2.0};
  std::vector<uni20::blas_int> ipiv(static_cast<std::size_t>(n));

  ErrorsAbortGuard const throw_errors(false);
  std::optional<uni20::lapack::LapackError> captured_error;
  try
  {
    uni20::lapack::gesv(n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
    ADD_FAILURE() << "gesv should have raised LapackError for a singular matrix";
  }
  catch (uni20::lapack::LapackError const& error)
  {
    captured_error = error;
  }
  catch (std::exception const& error)
  {
    ADD_FAILURE() << "gesv raised the wrong exception type: " << error.what();
  }

  ASSERT_TRUE(captured_error.has_value());
  EXPECT_EQ(captured_error->routine(), "gesv");
  EXPECT_EQ(captured_error->info(), 2);
  EXPECT_EQ(captured_error->reason(), "found a singular matrix");
  ASSERT_TRUE(captured_error->source_location().has_value());
  EXPECT_GT(captured_error->source_location()->line(), 0);
#if UNI20_HAS_STACKTRACE
  EXPECT_TRUE(captured_error->stacktrace().has_value());
#endif

  std::string const report = uni20::presentation::render_plain(diagnostic_report(*captured_error));
  EXPECT_NE(report.find("LAPACK 'gesv' failed"), std::string::npos);
  EXPECT_NE(report.find("INFO"), std::string::npos);
  EXPECT_NE(report.find("found a singular matrix"), std::string::npos);
}

TEST(LAPACKWrapper, PositiveFailureAbortsInNativeMode)
{
  EXPECT_DEATH(uni20::lapack::detail::check_singular("gesv", 2), "LAPACK 'gesv' failed");
}

TEST(LAPACKWrapper, NegativeInfoAlwaysAbortsAsProviderContractViolation)
{
  EXPECT_DEATH(
      {
        trace::get_formatting_options().set_errors_abort(false);
        uni20::lapack::detail::check_invalid_argument("test", -2);
      },
      "info >= 0");
}

TEST(LAPACKWrapper, DocumentedPositiveStatusesRemainResults)
{
  EXPECT_FALSE(uni20::lapack::detail::check_expert_solve("gesvx", 3, 0));
  EXPECT_TRUE(uni20::lapack::detail::check_expert_solve("gesvx", 3, 4));
  EXPECT_FALSE(uni20::lapack::detail::check_sylvester("trsyl", 0));
  EXPECT_TRUE(uni20::lapack::detail::check_sylvester("trsyl", 1));
}

TYPED_TEST(ComplexLAPACKWrapperTest, NormsUseUnderlyingRealResults)
{
  using Complex = TypeParam;
  using Real = real_type_t<Complex>;

  std::vector<Complex> general{
      Complex{Real{1}, Real{2}},
      Complex{Real{-2}, Real{1}},
      Complex{Real{3}, Real{-1}},
      Complex{Real{0.5}, Real{2}},
  };
  std::array<Real, 2> work{};
  Real const expected_general = std::sqrt(Real{5} + Real{5} + Real{10} + Real{4.25});
  EXPECT_FLOATING_EQ(uni20::lapack::lange('F', 2, 2, general.data(), 2, work.data()), expected_general);

  std::vector<Complex> hermitian{
      Complex{Real{2}, Real{}},
      Complex{Real{1}, Real{-1}},
      Complex{Real{1}, Real{1}},
      Complex{Real{3}, Real{}},
  };
  EXPECT_FLOATING_EQ(uni20::lapack::lanhe('F', 'U', 2, hermitian.data(), 2, work.data()), std::sqrt(Real{17}));

  std::vector<Complex> triangular{
      Complex{Real{1}, Real{1}},
      Complex{},
      Complex{Real{2}, Real{-1}},
      Complex{Real{-3}, Real{0.5}},
  };
  EXPECT_FLOATING_EQ(uni20::lapack::lantr('F', 'U', 'N', 2, 2, triangular.data(), 2, work.data()),
                     std::sqrt(Real{16.25}));
}

TYPED_TEST(ComplexLAPACKWrapperTest, GeneralLuSolveInverseAndConditionEstimate)
{
  using Complex = TypeParam;
  using Real = real_type_t<Complex>;
  uni20::blas_int constexpr n = 2;
  std::vector<Complex> const original{
      Complex{Real{2}, Real{1}},
      Complex{Real{1}, Real{-1}},
      Complex{Real{1}, Real{}},
      Complex{Real{3}, Real{0.5}},
  };
  std::array<Complex, 2> const expected{
      Complex{Real{1}, Real{2}},
      Complex{Real{-1}, Real{0.5}},
  };
  std::array<Complex, 2> right_hand_side{};
  for (uni20::blas_int row = 0; row < n; ++row)
  {
    for (uni20::blas_int column = 0; column < n; ++column)
      right_hand_side[row] += original[static_cast<std::size_t>(row + column * n)] * expected[column];
  }

  auto solve_matrix = original;
  auto solved = right_hand_side;
  std::array<uni20::blas_int, n> solve_pivots{};
  uni20::lapack::gesv(n, 1, solve_matrix.data(), n, solve_pivots.data(), solved.data(), n);
  EXPECT_FLOATING_EQ(solved[0], expected[0]);
  EXPECT_FLOATING_EQ(solved[1], expected[1]);

  auto factors = original;
  std::array<uni20::blas_int, n> pivots{};
  uni20::lapack::getrf(n, n, factors.data(), n, pivots.data());

  auto solved_from_factors = right_hand_side;
  uni20::lapack::getrs('N', n, 1, factors.data(), n, pivots.data(), solved_from_factors.data(), n);
  EXPECT_FLOATING_EQ(solved_from_factors[0], expected[0]);
  EXPECT_FLOATING_EQ(solved_from_factors[1], expected[1]);

  auto norm_matrix = original;
  std::array<Real, n> norm_work{};
  Real const matrix_norm = uni20::lapack::lange('1', n, n, norm_matrix.data(), n, norm_work.data());
  std::array<Complex, 2 * n> condition_work{};
  std::array<Real, 2 * n> condition_real_work{};
  Real const reciprocal_condition =
      uni20::lapack::gecon('1', n, factors.data(), n, matrix_norm, condition_work.data(), condition_real_work.data());
  EXPECT_GT(reciprocal_condition, Real{});
  EXPECT_LE(reciprocal_condition, Real{1});

  Complex inverse_work_query{};
  uni20::lapack::getri(n, factors.data(), n, pivots.data(), &inverse_work_query, -1);
  uni20::blas_int const inverse_lwork =
      std::max<uni20::blas_int>(1, static_cast<uni20::blas_int>(std::real(inverse_work_query)));
  std::vector<Complex> inverse_work(static_cast<std::size_t>(inverse_lwork));
  uni20::lapack::getri(n, factors.data(), n, pivots.data(), inverse_work.data(), inverse_lwork);

  for (uni20::blas_int column = 0; column < n; ++column)
  {
    for (uni20::blas_int row = 0; row < n; ++row)
    {
      Complex product{};
      for (uni20::blas_int inner = 0; inner < n; ++inner)
      {
        product +=
            original[static_cast<std::size_t>(row + inner * n)] * factors[static_cast<std::size_t>(inner + column * n)];
      }
      expect_complex_near(product, row == column ? Complex{Real{1}, Real{}} : Complex{},
                          Real{10} * tolerance<Complex>());
    }
  }
}

TYPED_TEST(ComplexLAPACKWrapperTest, GesvdReconstructsMatrix)
{
  check_complex_svd_reconstruction<TypeParam>(ComplexSvdDriver::standard);
}

TYPED_TEST(ComplexLAPACKWrapperTest, GesddReconstructsMatrix)
{
  check_complex_svd_reconstruction<TypeParam>(ComplexSvdDriver::divide_and_conquer);
}

TYPED_TEST(ComplexLAPACKWrapperTest, GesvdxReconstructsMatrix)
{
  check_complex_svd_reconstruction<TypeParam>(ComplexSvdDriver::selected);
}
