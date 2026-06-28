#include <uni20/core/scalar_io.hpp>

#include <gtest/gtest.h>

#include <sstream>

TEST(ScalarIoTest, FormatsBuiltInRealScalars)
{
  uni20::scalar_format_options options;
  options.precision = 4;

  EXPECT_EQ(uni20::format_real(1.0 / 3.0, options), "0.3333");

  options.notation = uni20::real_format_notation::scientific;
  options.precision = 2;
  EXPECT_EQ(uni20::format_real(12.5, options), "1.25e+01");
}

TEST(ScalarIoTest, FormatsBuiltInComplexScalars)
{
  uni20::scalar_format_options options;
  options.precision = 3;

  EXPECT_EQ(uni20::format_complex(uni20::complex<double>{1.25, -3.5}, options), "1.25-3.5i");

  options.notation = uni20::real_format_notation::fixed;
  options.precision = 2;
  EXPECT_EQ(uni20::format_complex(uni20::complex<double>{1.25, -0.0}, options), "1.25+0.00i");
}

TEST(ScalarIoTest, ParsesBuiltInRealScalars)
{
  EXPECT_FLOAT_EQ(uni20::parse_real<float>("1.25"), 1.25f);
  EXPECT_DOUBLE_EQ(uni20::parse_real<double>("-2.5e1"), -25.0);
  EXPECT_THROW((void)uni20::parse_real<double>("1.0x"), std::invalid_argument);
}

TEST(ScalarIoTest, ReadsBuiltInRealScalarsFromStreamToken)
{
  std::istringstream input("3.5");
  double value = 0.0;

  uni20::read_real(input, value);

  EXPECT_TRUE(input);
  EXPECT_DOUBLE_EQ(value, 3.5);
}
