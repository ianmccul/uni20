#include <uni20/backend/lapack/lapack.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <exception>
#include <optional>
#include <string>
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

  EXPECT_NEAR(b[0], 2.0, 1e-12);
  EXPECT_NEAR(b[1], 3.0, 1e-12);
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
