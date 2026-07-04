#include <gtest/gtest.h>
#include <limits>
#include <uni20/core/numeric_limits.hpp>

namespace
{
struct CustomScalar
{};
} // namespace

namespace uni20
{
template <> struct numeric_limits<CustomScalar>
{
    static constexpr bool is_specialized = true;
    static constexpr int digits = 37;

    static constexpr CustomScalar epsilon() noexcept { return {}; }
};
} // namespace uni20

TEST(NumericLimitsTest, DelegatesToStdNumericLimitsForBuiltins)
{
  static_assert(uni20::numeric_limits<float>::is_specialized);
  static_assert(uni20::numeric_limits<double>::is_specialized);
  static_assert(uni20::numeric_limits<long double>::is_specialized);

  static_assert(uni20::numeric_limits<float>::digits == std::numeric_limits<float>::digits);
  static_assert(uni20::numeric_limits<double>::max_digits10 == std::numeric_limits<double>::max_digits10);
  EXPECT_EQ(uni20::numeric_limits<float>::epsilon(), std::numeric_limits<float>::epsilon());
  EXPECT_EQ(uni20::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
}

TEST(NumericLimitsTest, SupportsUni20CustomSpecializations)
{
  static_assert(uni20::has_numeric_limits_v<CustomScalar>);
  static_assert(uni20::numeric_limits<CustomScalar>::digits == 37);
}

TEST(NumericLimitsTest, ForwardsCvQualifiedTypesToUni20Specialization)
{
  static_assert(uni20::has_numeric_limits_v<CustomScalar const>);
  static_assert(uni20::has_numeric_limits_v<CustomScalar volatile>);
  static_assert(uni20::numeric_limits<CustomScalar const>::digits == 37);
  static_assert(uni20::numeric_limits<CustomScalar volatile>::digits == 37);
}

TEST(NumericLimitsTest, ReportsMissingLimits)
{
  struct NoLimits
  {};

  static_assert(!uni20::has_numeric_limits_v<NoLimits>);
}
