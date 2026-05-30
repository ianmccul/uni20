#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>

namespace utc = uni20::tensorcontraction;

namespace
{

auto make_vector() -> utc::MatrixFamily
{
  std::array blocks{utc::MatrixFamily::Block{2, 2}, utc::MatrixFamily::Block{1, 3}};
  utc::MatrixFamily x(blocks);
  x.assign(0, std::array{1.0, -2.0, 3.0, -4.0});
  x.assign(1, std::array{5.0, -6.0, 7.0});
  return x;
}

} // namespace

TEST(TensorContractionVectorAlgebraTest, ComputesDotAndNorm)
{
  auto x = make_vector();
  auto y = make_vector();
  y.assign(0, std::array{2.0, 3.0, -1.0, 0.5});
  y.assign(1, std::array{4.0, -2.0, 1.0});

  EXPECT_DOUBLE_EQ(utc::dot(x, y), 30.0);
  EXPECT_DOUBLE_EQ(utc::norm2(x), 140.0);
  EXPECT_DOUBLE_EQ(utc::norm(x), std::sqrt(140.0));
}

TEST(TensorContractionVectorAlgebraTest, ScalesAndAxpy)
{
  auto x = make_vector();
  auto y = utc::make_like(x);
  y.fill(1.0);

  utc::scale(x, 0.5);
  EXPECT_DOUBLE_EQ(x.values(0)[0], 0.5);
  EXPECT_DOUBLE_EQ(x.values(0)[3], -2.0);
  EXPECT_DOUBLE_EQ(x.values(1)[2], 3.5);

  utc::axpy(2.0, x, y);
  EXPECT_DOUBLE_EQ(y.values(0)[0], 2.0);
  EXPECT_DOUBLE_EQ(y.values(0)[1], -1.0);
  EXPECT_DOUBLE_EQ(y.values(0)[2], 4.0);
  EXPECT_DOUBLE_EQ(y.values(0)[3], -3.0);
  EXPECT_DOUBLE_EQ(y.values(1)[0], 6.0);
  EXPECT_DOUBLE_EQ(y.values(1)[1], -5.0);
  EXPECT_DOUBLE_EQ(y.values(1)[2], 8.0);
}

TEST(TensorContractionVectorAlgebraTest, CopiesAndZeros)
{
  auto x = make_vector();
  auto y = utc::make_like(x);

  utc::copy(x, y);
  EXPECT_DOUBLE_EQ(y.values(0)[2], 3.0);
  EXPECT_DOUBLE_EQ(y.values(1)[1], -6.0);

  utc::zero(y);
  for (std::size_t block = 0; block < y.blocks().size(); ++block)
  {
    for (double value : y.values(block))
    {
      EXPECT_DOUBLE_EQ(value, 0.0);
    }
  }
}

TEST(TensorContractionVectorAlgebraTest, NormalizesAndReturnsOriginalNorm)
{
  auto x = make_vector();

  double const original_norm = utc::normalize(x);

  EXPECT_DOUBLE_EQ(original_norm, std::sqrt(140.0));
  EXPECT_NEAR(utc::norm(x), 1.0, 1.0e-14);
}

TEST(TensorContractionVectorAlgebraTest, RejectsZeroNormalizeAndShapeMismatches)
{
  auto x = make_vector();
  auto zero = utc::make_like(x);
  std::array wrong_blocks{utc::MatrixFamily::Block{7, 1}};
  utc::MatrixFamily wrong(wrong_blocks);

  EXPECT_THROW(utc::normalize(zero), std::invalid_argument);
  EXPECT_THROW(utc::dot(x, wrong), std::invalid_argument);
  EXPECT_THROW(utc::axpy(1.0, x, wrong), std::invalid_argument);
  EXPECT_THROW(utc::copy(x, wrong), std::invalid_argument);
}
