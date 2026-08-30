#include <uni20/krylov/dense_linalg.hpp>

#include "krylov_test_types.hpp"

#include <uni20/common/gtest.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace
{

template <typename Scalar> class KrylovDenseLinalgTypedTest : public ::testing::Test {};

using ScalarTypes = uni20::krylov::test::KrylovScalarTestTypes;
TYPED_TEST_SUITE(KrylovDenseLinalgTypedTest, ScalarTypes);

template <typename Scalar> double tolerance()
{
  if constexpr (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, uni20::complex<float>>)
  {
    return 1.0e-5;
  }
  else
  {
    return 1.0e-12;
  }
}

template <typename Scalar> void expect_floating_eq(Scalar const& actual, Scalar const& expected)
{
  EXPECT_FLOATING_EQ(actual, expected);
}

template <typename Scalar>
void expect_vector_floating_eq(std::vector<Scalar> const& actual, std::vector<Scalar> const& expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index)
  {
    expect_floating_eq(actual[index], expected[index]);
  }
}

template <typename Scalar> std::span<Scalar const> const_span(std::vector<Scalar> const& vector)
{
  return std::span<Scalar const>(vector.data(), vector.size());
}

template <typename Scalar> std::span<Scalar> span(std::vector<Scalar>& vector)
{
  return std::span<Scalar>(vector.data(), vector.size());
}

TYPED_TEST(KrylovDenseLinalgTypedTest, CopiesScalesAndAddsVectors)
{
  using Scalar = TypeParam;

  std::vector<Scalar> x{Scalar{1}, Scalar{2}, Scalar{3}};
  std::vector<Scalar> y(3);

  uni20::krylov::copy(span(y), const_span(x));
  expect_vector_floating_eq(y, x);

  uni20::krylov::scal(Scalar{2}, span(y));
  expect_vector_floating_eq(y, std::vector<Scalar>{Scalar{2}, Scalar{4}, Scalar{6}});

  uni20::krylov::axpy(span(y), Scalar{-1}, const_span(x));
  expect_vector_floating_eq(y, x);
}

TYPED_TEST(KrylovDenseLinalgTypedTest, ComputesVectorProducts)
{
  using Scalar = TypeParam;

  std::vector<Scalar> x{Scalar{1}, Scalar{2}, Scalar{3}};
  std::vector<Scalar> y{Scalar{4}, Scalar{5}, Scalar{6}};

  expect_floating_eq(uni20::krylov::dotu(const_span(x), const_span(y)), Scalar{32});
  expect_floating_eq(uni20::krylov::dotc(const_span(x), const_span(y)), Scalar{32});
}

TEST(KrylovDenseLinalg, ComputesComplexConjugatedDotProduct)
{
  using Scalar = uni20::complex<double>;

  std::vector<Scalar> x{Scalar{1.0, 2.0}, Scalar{3.0, -1.0}};
  std::vector<Scalar> y{Scalar{2.0, -1.0}, Scalar{-4.0, 0.5}};

  EXPECT_FLOATING_EQ(uni20::krylov::dotu(const_span(x), const_span(y)), (Scalar{-7.5, 8.5}));
  EXPECT_FLOATING_EQ(uni20::krylov::dotc(const_span(x), const_span(y)), (Scalar{-12.5, -7.5}));
}

TYPED_TEST(KrylovDenseLinalgTypedTest, SetsAndCopiesMatrixRegions)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 3);
  uni20::krylov::laset(matrix, Scalar{1}, Scalar{-2}, uni20::krylov::MatrixFill::All);

  for (uni20::index_type col = 0; col < matrix.cols(); ++col)
  {
    for (uni20::index_type row = 0; row < matrix.rows(); ++row)
    {
      expect_floating_eq(matrix[row, col], row == col ? Scalar{1} : Scalar{-2});
    }
  }

  uni20::krylov::Matrix<Scalar> upper(3, 3);
  uni20::krylov::laset(upper, Scalar{}, Scalar{}, uni20::krylov::MatrixFill::All);
  uni20::krylov::lacpy(upper, matrix, uni20::krylov::MatrixFill::Upper);

  for (uni20::index_type col = 0; col < upper.cols(); ++col)
  {
    for (uni20::index_type row = 0; row < upper.rows(); ++row)
    {
      Scalar const expected = row <= col ? matrix[row, col] : Scalar{};
      expect_floating_eq(upper[row, col], expected);
    }
  }
}

TYPED_TEST(KrylovDenseLinalgTypedTest, StoresRightMatrixRowMajorAndCopiesToColumnMajor)
{
  using Scalar = TypeParam;

  uni20::krylov::RightMatrix<Scalar> right(2, 3);
  right[0, 0] = Scalar{1};
  right[0, 1] = Scalar{2};
  right[0, 2] = Scalar{3};
  right[1, 0] = Scalar{4};
  right[1, 1] = Scalar{5};
  right[1, 2] = Scalar{6};

  expect_vector_floating_eq(std::vector<Scalar>(right.storage().data(), right.storage().data() + right.size()),
                            std::vector<Scalar>{Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}, Scalar{5}, Scalar{6}});

  auto span = uni20::krylov::right_mdspan(right);
  expect_floating_eq((span[0, 2]), Scalar{3});
  expect_floating_eq((span[1, 0]), Scalar{4});

  uni20::krylov::Matrix<Scalar> left = uni20::krylov::copy_right_to_left(right);
  expect_vector_floating_eq(std::vector<Scalar>(left.storage().data(), left.storage().data() + left.size()),
                            std::vector<Scalar>{Scalar{1}, Scalar{4}, Scalar{2}, Scalar{5}, Scalar{3}, Scalar{6}});

  uni20::krylov::RightMatrix<Scalar> round_trip = uni20::krylov::copy_left_to_right(left);
  expect_vector_floating_eq(
      std::vector<Scalar>(round_trip.storage().data(), round_trip.storage().data() + round_trip.size()),
      std::vector<Scalar>(right.storage().data(), right.storage().data() + right.size()));
}

TYPED_TEST(KrylovDenseLinalgTypedTest, ComputesMatrixVectorProducts)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 3);
  matrix[0, 0] = Scalar{1};
  matrix[1, 0] = Scalar{2};
  matrix[0, 1] = Scalar{3};
  matrix[1, 1] = Scalar{4};
  matrix[0, 2] = Scalar{5};
  matrix[1, 2] = Scalar{6};

  std::vector<Scalar> x{Scalar{1}, Scalar{2}, Scalar{3}};
  using Real = uni20::make_real_t<Scalar>;
  Scalar const nan{uni20::numeric_limits<Real>::quiet_NaN()};
  std::vector<Scalar> y{nan, nan};

  uni20::krylov::gemv(span(y), Scalar{1}, matrix, const_span(x), Scalar{0});
  expect_vector_floating_eq(y, std::vector<Scalar>{Scalar{22}, Scalar{28}});

  std::vector<Scalar> xt{Scalar{1}, Scalar{-1}};
  std::vector<Scalar> yt{Scalar{10}, Scalar{20}, Scalar{30}};
  uni20::krylov::gemv(span(yt), Scalar{1}, matrix, const_span(xt), Scalar{0},
                      uni20::krylov::MatrixTranspose::Transpose);
  expect_vector_floating_eq(yt, std::vector<Scalar>{Scalar{-1}, Scalar{-1}, Scalar{-1}});
}

TEST(KrylovDenseLinalg, ComputesComplexConjugateTransposeMatrixVectorProduct)
{
  using Scalar = uni20::complex<double>;

  uni20::krylov::Matrix<Scalar> matrix(2, 1);
  matrix[0, 0] = Scalar{1.0, 2.0};
  matrix[1, 0] = Scalar{3.0, -1.0};

  std::vector<Scalar> x{Scalar{2.0, -1.0}, Scalar{1.0, 4.0}};
  std::vector<Scalar> y{Scalar{}};
  uni20::krylov::gemv(span(y), Scalar{1}, matrix, const_span(x), Scalar{0},
                      uni20::krylov::MatrixTranspose::ConjugateTranspose);

  EXPECT_FLOATING_EQ(y[0], (Scalar{-1.0, 8.0}));
}

TYPED_TEST(KrylovDenseLinalgTypedTest, AppliesRankOneUpdates)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  uni20::krylov::laset(matrix, Scalar{}, Scalar{}, uni20::krylov::MatrixFill::All);
  std::vector<Scalar> x{Scalar{1}, Scalar{2}};
  std::vector<Scalar> y{Scalar{3}, Scalar{4}};

  uni20::krylov::geru(matrix, Scalar{2}, const_span(x), const_span(y));

  expect_floating_eq(matrix[0, 0], Scalar{6});
  expect_floating_eq(matrix[1, 0], Scalar{12});
  expect_floating_eq(matrix[0, 1], Scalar{8});
  expect_floating_eq(matrix[1, 1], Scalar{16});
}

TEST(KrylovDenseLinalg, AppliesComplexConjugatedRankOneUpdate)
{
  using Scalar = uni20::complex<double>;

  uni20::krylov::Matrix<Scalar> matrix(1, 2);
  uni20::krylov::laset(matrix, Scalar{}, Scalar{}, uni20::krylov::MatrixFill::All);
  std::vector<Scalar> x{Scalar{2.0, 1.0}};
  std::vector<Scalar> y{Scalar{1.0, -1.0}, Scalar{3.0, 2.0}};

  uni20::krylov::gerc(matrix, Scalar{1}, const_span(x), const_span(y));

  EXPECT_FLOATING_EQ((matrix[0, 0]), (Scalar{1.0, 3.0}));
  EXPECT_FLOATING_EQ((matrix[0, 1]), (Scalar{8.0, -1.0}));
}

TEST(KrylovDenseLinalg, RejectsMismatchedVectorSizes)
{
  std::vector<double> x{1.0, 2.0};
  std::vector<double> y{0.0};

  EXPECT_THROW(uni20::krylov::copy(span(y), const_span(x)), std::invalid_argument);
  EXPECT_THROW(uni20::krylov::axpy(span(y), 1.0, const_span(x)), std::invalid_argument);
  EXPECT_THROW(uni20::krylov::dot(const_span(x), const_span(y)), std::invalid_argument);
}

} // namespace
