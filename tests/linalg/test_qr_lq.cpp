#include <uni20/linalg/ops/lq.hpp>
#include <uni20/linalg/ops/qr.hpp>
#include <uni20/tensor/tensor.hpp>
#include <uni20/tensor/transform.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>

namespace
{

template <class Scalar> class QrLqTest : public ::testing::Test {};

using QrLqRealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(QrLqTest, QrLqRealTypes);

template <class Scalar> constexpr Scalar tolerance() { return Scalar{200} * std::numeric_limits<Scalar>::epsilon(); }

template <class Matrix> void initialize_matrix(Matrix& matrix)
{
  using scalar_type = typename Matrix::value_type;
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < matrix.extent(1); ++column)
    {
      matrix[row, column] =
          static_cast<scalar_type>((row + 1) * (column + 2)) + (row == column ? scalar_type{1} : scalar_type{});
    }
  }
}

template <class Matrix, class Q, class R, class Scalar>
void expect_qr(Matrix const& matrix, Q const& q, R const& r, Scalar error)
{
  ASSERT_EQ(q.extent(0), matrix.extent(0));
  ASSERT_EQ(q.extent(1), r.extent(0));
  ASSERT_EQ(r.extent(1), matrix.extent(1));
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < matrix.extent(1); ++column)
    {
      Scalar reconstructed{};
      for (uni20::index_type inner = 0; inner < q.extent(1); ++inner)
        reconstructed += q[row, inner] * r[inner, column];
      EXPECT_NEAR(reconstructed, (matrix[row, column]), error);
    }
  }
  for (uni20::index_type lhs = 0; lhs < q.extent(1); ++lhs)
  {
    for (uni20::index_type rhs = 0; rhs < q.extent(1); ++rhs)
    {
      Scalar inner{};
      for (uni20::index_type row = 0; row < q.extent(0); ++row)
        inner += q[row, lhs] * q[row, rhs];
      EXPECT_NEAR(inner, lhs == rhs ? Scalar{1} : Scalar{}, error);
    }
  }
  for (uni20::index_type row = 0; row < r.extent(0); ++row)
    for (uni20::index_type column = 0; column < std::min(row, r.extent(1)); ++column)
      EXPECT_EQ((r[row, column]), Scalar{});
}

template <class Matrix, class L, class Q, class Scalar>
void expect_lq(Matrix const& matrix, L const& l, Q const& q, Scalar error)
{
  ASSERT_EQ(l.extent(0), matrix.extent(0));
  ASSERT_EQ(l.extent(1), q.extent(0));
  ASSERT_EQ(q.extent(1), matrix.extent(1));
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < matrix.extent(1); ++column)
    {
      Scalar reconstructed{};
      for (uni20::index_type inner = 0; inner < q.extent(0); ++inner)
        reconstructed += l[row, inner] * q[inner, column];
      EXPECT_NEAR(reconstructed, (matrix[row, column]), error);
    }
  }
  for (uni20::index_type lhs = 0; lhs < q.extent(0); ++lhs)
  {
    for (uni20::index_type rhs = 0; rhs < q.extent(0); ++rhs)
    {
      Scalar inner{};
      for (uni20::index_type column = 0; column < q.extent(1); ++column)
        inner += q[lhs, column] * q[rhs, column];
      EXPECT_NEAR(inner, lhs == rhs ? Scalar{1} : Scalar{}, error);
    }
  }
  for (uni20::index_type row = 0; row < l.extent(0); ++row)
    for (uni20::index_type column = row + 1; column < l.extent(1); ++column)
      EXPECT_EQ((l[row, column]), Scalar{});
}

TYPED_TEST(QrLqTest, PreservingApisFactorBothRectangularOrientations)
{
  using scalar_type = TypeParam;
  scalar_type const error = tolerance<scalar_type>();

  uni20::DenseMatrix<scalar_type, uni20::RowMajor> tall(4, 3);
  initialize_matrix(tall);
  auto const tall_original = tall;
  auto tall_qr = uni20::linalg::qr(tall);
  auto tall_lq = uni20::linalg::lq(tall);

  static_assert(std::same_as<typename decltype(tall_qr.q)::layout_type, uni20::ColumnMajor>);
  static_assert(std::same_as<typename decltype(tall_lq.l)::layout_type, uni20::ColumnMajor>);
  expect_qr(tall_original, tall_qr.q, tall_qr.r, error);
  expect_lq(tall_original, tall_lq.l, tall_lq.q, error);

  uni20::DenseMatrix<scalar_type, uni20::RowMajor> wide(3, 4);
  initialize_matrix(wide);
  auto const wide_original = wide;
  auto wide_qr = uni20::linalg::qr(wide);
  auto wide_lq = uni20::linalg::lq(wide);

  expect_qr(wide_original, wide_qr.q, wide_qr.r, error);
  expect_lq(wide_original, wide_lq.l, wide_lq.q, error);
  for (uni20::index_type row = 0; row < tall.extent(0); ++row)
    for (uni20::index_type column = 0; column < tall.extent(1); ++column)
      EXPECT_EQ((tall[row, column]), (tall_original[row, column]));
  for (uni20::index_type row = 0; row < wide.extent(0); ++row)
    for (uni20::index_type column = 0; column < wide.extent(1); ++column)
      EXPECT_EQ((wide[row, column]), (wide_original[row, column]));
}

TEST(QrLqTest, DestructiveKernelsAcquireDeferredHostOperands)
{
  uni20::test::DeferredHostTensor<double, 2> qr_work(3, 2);
  uni20::test::DeferredHostTensor<double, 2> qr_q(3, 2);
  uni20::test::DeferredHostTensor<double, 2> qr_r(2, 2);
  {
    auto access = uni20::test::acquire_host_write_access_sync(qr_work);
    initialize_matrix(access.mdspan());
  }
  auto const qr_original = qr_work.storage();
  uni20::linalg::qr_factorization(uni20::linalg::LapackBackend{}, qr_q, qr_r, qr_work);
  auto qr_q_access = uni20::test::acquire_host_read_access_sync(qr_q);
  auto qr_r_access = uni20::test::acquire_host_read_access_sync(qr_r);
  stdex::mdspan<double const, stdex::dextents<uni20::index_type, 2>, stdex::layout_left> qr_original_span(
      qr_original.data(), 3, 2);
  expect_qr(qr_original_span, qr_q_access.mdspan(), qr_r_access.mdspan(), 1.0e-12);

  uni20::test::DeferredHostTensor<double, 2> lq_work(2, 3);
  uni20::test::DeferredHostTensor<double, 2> lq_l(2, 2);
  uni20::test::DeferredHostTensor<double, 2> lq_q(2, 3);
  {
    auto access = uni20::test::acquire_host_write_access_sync(lq_work);
    initialize_matrix(access.mdspan());
  }
  auto const lq_original = lq_work.storage();
  uni20::linalg::lq_factorization(uni20::linalg::LapackBackend{}, lq_l, lq_q, lq_work);
  auto lq_l_access = uni20::test::acquire_host_read_access_sync(lq_l);
  auto lq_q_access = uni20::test::acquire_host_read_access_sync(lq_q);
  stdex::mdspan<double const, stdex::dextents<uni20::index_type, 2>, stdex::layout_left> lq_original_span(
      lq_original.data(), 2, 3);
  expect_lq(lq_original_span, lq_l_access.mdspan(), lq_q_access.mdspan(), 1.0e-12);
}

TEST(QrLqTest, LapackDeclinesRowMajorWorkspaceWithoutMutation)
{
  uni20::DenseMatrix<double, uni20::RowMajor> matrix_work(2, 3);
  initialize_matrix(matrix_work);
  auto const original = matrix_work;
  uni20::DenseMatrix<double> q(2, 2);
  uni20::DenseMatrix<double> r(2, 3);
  uni20::fill(q, 7.0);
  uni20::fill(r, 9.0);
  auto q_descriptor = uni20::mdspec_of(q);
  auto r_descriptor = uni20::mdspec_of(r);
  auto matrix_descriptor = uni20::mdspec_of(matrix_work);

  bool const accepted = uni20::linalg::try_dispatch_kernel(uni20::linalg::LapackBackend{}, uni20::linalg::qr_op{},
                                                           q_descriptor, r_descriptor, matrix_descriptor);

  EXPECT_FALSE(accepted);
  for (uni20::index_type row = 0; row < matrix_work.extent(0); ++row)
    for (uni20::index_type column = 0; column < matrix_work.extent(1); ++column)
      EXPECT_EQ((matrix_work[row, column]), (original[row, column]));
  EXPECT_EQ((q[0, 0]), 7.0);
  EXPECT_EQ((r[0, 0]), 9.0);

  uni20::DenseMatrix<double> l(2, 2);
  uni20::DenseMatrix<double> lq_q(2, 3);
  uni20::fill(l, 11.0);
  uni20::fill(lq_q, 13.0);
  auto l_descriptor = uni20::mdspec_of(l);
  auto lq_q_descriptor = uni20::mdspec_of(lq_q);
  bool const lq_accepted = uni20::linalg::try_dispatch_kernel(uni20::linalg::LapackBackend{}, uni20::linalg::lq_op{},
                                                              l_descriptor, lq_q_descriptor, matrix_descriptor);

  EXPECT_FALSE(lq_accepted);
  for (uni20::index_type row = 0; row < matrix_work.extent(0); ++row)
    for (uni20::index_type column = 0; column < matrix_work.extent(1); ++column)
      EXPECT_EQ((matrix_work[row, column]), (original[row, column]));
  EXPECT_EQ((l[0, 0]), 11.0);
  EXPECT_EQ((lq_q[0, 0]), 13.0);
}

TEST(QrLqTest, ZeroRankReturnsReducedEmptyFactors)
{
  uni20::DenseMatrix<double> no_columns(3, 0);
  auto qr_result = uni20::linalg::qr(no_columns);
  EXPECT_EQ(qr_result.q.extent(0), 3);
  EXPECT_EQ(qr_result.q.extent(1), 0);
  EXPECT_EQ(qr_result.r.extent(0), 0);
  EXPECT_EQ(qr_result.r.extent(1), 0);

  uni20::DenseMatrix<double> no_rows(0, 3);
  auto lq_result = uni20::linalg::lq(no_rows);
  EXPECT_EQ(lq_result.l.extent(0), 0);
  EXPECT_EQ(lq_result.l.extent(1), 0);
  EXPECT_EQ(lq_result.q.extent(0), 0);
  EXPECT_EQ(lq_result.q.extent(1), 3);
}

} // namespace
