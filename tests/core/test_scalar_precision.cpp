#include <uni20/core/scalar_precision.hpp>

#include <gtest/gtest.h>

namespace
{

TEST(ScalarPrecisionTest, ParsesStablePrecisionNames)
{
  EXPECT_EQ(uni20::parse_scalar_precision("fp32"), uni20::ScalarPrecision::fp32);
  EXPECT_EQ(uni20::parse_scalar_precision("fp64"), uni20::ScalarPrecision::fp64);
  EXPECT_EQ(uni20::parse_scalar_precision("fp128"), uni20::ScalarPrecision::fp128);
  EXPECT_EQ(uni20::parse_scalar_precision("double"), std::nullopt);
}

TEST(ScalarPrecisionTest, ReportsConfiguredPrecisions)
{
  auto const configured = uni20::configured_scalar_precisions();
  ASSERT_GE(configured.size(), 2);
  EXPECT_EQ(configured[0], uni20::ScalarPrecision::fp32);
  EXPECT_EQ(configured[1], uni20::ScalarPrecision::fp64);
  EXPECT_EQ(configured.size(), uni20::has_float128 ? 3 : 2);
  EXPECT_EQ(uni20::scalar_precision_is_available(uni20::ScalarPrecision::fp128), uni20::has_float128);
}

TEST(ScalarPrecisionTest, VisitsConfiguredConcreteTypes)
{
  auto type_size = []<typename Scalar>() { return sizeof(Scalar); };
  EXPECT_EQ(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp32, type_size), sizeof(uni20::float32));
  EXPECT_EQ(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp64, type_size), sizeof(uni20::float64));

  if (uni20::has_float128)
  {
    EXPECT_GT(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp128, type_size), sizeof(uni20::float64));
  }
  else
  {
    EXPECT_THROW(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp128, type_size), std::invalid_argument);
  }
}

} // namespace
