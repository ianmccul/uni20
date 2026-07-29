#include <uni20/async/shared_storage.hpp>
#include <uni20/config.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
template <class Matrix, class Scalar> void fill_matrix(Matrix& matrix, std::initializer_list<Scalar> values)
{
  auto value = values.begin();
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.cols(); ++col)
    {
      matrix[row, col] = *value;
      ++value;
    }
  }
}

class ErrorModeGuard {
  public:
    ErrorModeGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorModeGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

template <class Scalar> void check_reference_product()
{
  uni20::DenseMatrix<Scalar> lhs(2, 3);
  uni20::DenseMatrix<Scalar, uni20::RowMajor> rhs(3, 2);
  uni20::DenseMatrix<Scalar> output;
  fill_matrix(lhs, {Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}, Scalar{5}, Scalar{6}});
  fill_matrix(rhs, {Scalar{7}, Scalar{8}, Scalar{9}, Scalar{10}, Scalar{11}, Scalar{12}});

  uni20::linalg::assign_product(output, lhs, rhs);

  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 2);
  EXPECT_TRUE((output[0, 0] == Scalar{58}));
  EXPECT_TRUE((output[0, 1] == Scalar{64}));
  EXPECT_TRUE((output[1, 0] == Scalar{139}));
  EXPECT_TRUE((output[1, 1] == Scalar{154}));
}
} // namespace

TEST(MatrixProductTest, AssignProductResizesAndUsesDefaultSelector)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  std::vector<diagnostics::event> events;
  diagnostics::scoped_sink capture([&](diagnostics::event const& event) { events.push_back(event); });

  check_reference_product<double>();

  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].operation, "assign_product");
  EXPECT_TRUE(events[0].succeeded());
  EXPECT_TRUE(events[0].selected_backend().has_value());
}

TEST(MatrixProductTest, AssignProductRetainsMatchingOutputStorage)
{
  uni20::DenseMatrix<double> lhs(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  uni20::DenseMatrix<double> output(2, 2);
  fill_matrix(lhs, {1.0, 0.0, 0.0, 1.0});
  fill_matrix(rhs, {1.0, 2.0, 3.0, 4.0});
  auto* const original_handle = output.mutable_handle();

  uni20::linalg::assign_product(output, lhs, rhs, 2.0);

  EXPECT_EQ(output.mutable_handle(), original_handle);
  EXPECT_DOUBLE_EQ((output[0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((output[1, 1]), 8.0);
}

TEST(MatrixProductTest, GemmWithZeroBetaRetainsFixedOutputDispatchIdentity)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  std::vector<diagnostics::event> events;
  diagnostics::scoped_sink capture([&](diagnostics::event const& event) { events.push_back(event); });
  uni20::DenseMatrix<double> lhs(1, 1);
  uni20::DenseMatrix<double> rhs(1, 1);
  uni20::DenseMatrix<double> output(1, 1);
  lhs[0, 0] = 2.0;
  rhs[0, 0] = 3.0;

  uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);

  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].operation, "gemm");
  EXPECT_DOUBLE_EQ((output[0, 0]), 6.0);
}

TEST(MatrixProductTest, AddProductUsesFixedOutputAndExistingValues)
{
  uni20::DenseMatrix<double> lhs(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  uni20::DenseMatrix<double, uni20::RowMajor> output(2, 2);
  fill_matrix(lhs, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(rhs, {5.0, 6.0, 7.0, 8.0});
  fill_matrix(output, {10.0, 20.0, 30.0, 40.0});

  uni20::linalg::add_product(output, lhs, rhs, 0.5);

  EXPECT_DOUBLE_EQ((output[0, 0]), 19.5);
  EXPECT_DOUBLE_EQ((output[0, 1]), 31.0);
  EXPECT_DOUBLE_EQ((output[1, 0]), 51.5);
  EXPECT_DOUBLE_EQ((output[1, 1]), 65.0);
}

TEST(MatrixProductTest, ExplicitCpuSelectorSupportsMixedLayoutsAndResize)
{
  uni20::DenseMatrix<double, uni20::RowMajor> lhs(2, 3);
  uni20::DenseMatrix<double> rhs(3, 2);
  uni20::DenseMatrix<double, uni20::RowMajor> output(1, 1);
  fill_matrix(lhs, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(rhs, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  uni20::linalg::assign_product(uni20::linalg::CpuReferenceBackend{}, output, lhs, rhs);

  EXPECT_EQ(output.mapping().stride(0), 2);
  EXPECT_EQ(output.mapping().stride(1), 1);
  EXPECT_DOUBLE_EQ((output[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((output[1, 1]), 154.0);
}

TEST(MatrixProductTest, BlasLayoutDeclineLeavesWrongShapedOutputForCpuFallback)
{
  using strided_matrix = uni20::StridedTensor<double, 2>;
  strided_matrix lhs(strided_matrix::extents_type{2, 2}, std::array<uni20::index_type, 2>{2, 5});
  uni20::DenseMatrix<double> rhs(2, 2);
  uni20::DenseMatrix<double> output(1, 1);
  fill_matrix(lhs, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(rhs, {1.0, 0.0, 0.0, 1.0});
  output[0, 0] = -7.0;

  EXPECT_EQ(uni20::linalg::try_kernel(uni20::linalg::BlasBackend{}, uni20::linalg::assign_product_op{}, output, 1.0,
                                      lhs, rhs),
            uni20::linalg::KernelAttempt::unsupported_layout);
  EXPECT_EQ(output.rows(), 1);
  EXPECT_EQ(output.cols(), 1);
  EXPECT_DOUBLE_EQ((output[0, 0]), -7.0);

  uni20::linalg::assign_product(output, lhs, rhs);
  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 2);
  EXPECT_DOUBLE_EQ((output[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((output[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((output[1, 0]), 3.0);
  EXPECT_DOUBLE_EQ((output[1, 1]), 4.0);
}

TEST(MatrixProductTest, BlasLayoutDeclineLeavesDeferredOutputUnconstructed)
{
  using strided_matrix = uni20::StridedTensor<double, 2>;
  using output_matrix = uni20::DenseMatrix<double>;
  strided_matrix lhs(strided_matrix::extents_type{2, 2}, std::array<uni20::index_type, 2>{2, 5});
  output_matrix rhs(2, 2);
  auto output = uni20::async::make_unconstructed_shared_storage<output_matrix>();

  EXPECT_EQ(uni20::linalg::try_kernel(uni20::linalg::BlasBackend{}, uni20::linalg::assign_product_op{}, output, 1.0,
                                      lhs, rhs),
            uni20::linalg::KernelAttempt::unsupported_layout);
  EXPECT_FALSE(output.constructed());
}

TEST(MatrixProductTest, BlasEmptyInnerDeclineLeavesWrongShapedOutputUntouched)
{
  uni20::DenseMatrix<double> lhs(2, 0);
  uni20::DenseMatrix<double> rhs(0, 3);
  uni20::DenseMatrix<double> output(1, 1);
  output[0, 0] = -7.0;

  EXPECT_EQ(uni20::linalg::try_kernel(uni20::linalg::BlasBackend{}, uni20::linalg::assign_product_op{}, output, 1.0,
                                      lhs, rhs),
            uni20::linalg::KernelAttempt::unsupported_instance);
  EXPECT_EQ(output.rows(), 1);
  EXPECT_EQ(output.cols(), 1);
  EXPECT_DOUBLE_EQ((output[0, 0]), -7.0);
}

TEST(MatrixProductTest, BlasTransformDeclineLeavesWrongShapedOutputUntouched)
{
  using scalar_type = uni20::complex<double>;
  uni20::DenseMatrix<scalar_type> lhs(1, 1);
  uni20::DenseMatrix<scalar_type> rhs(1, 1);
  uni20::DenseMatrix<scalar_type> output(2, 2);
  lhs[0, 0] = scalar_type{2.0, 1.0};
  rhs[0, 0] = scalar_type{3.0, 1.0};
  fill_matrix(output, {scalar_type{-7.0}, scalar_type{-7.0}, scalar_type{-7.0}, scalar_type{-7.0}});
  auto conjugated = uni20::conj(lhs);

  EXPECT_EQ(uni20::linalg::try_kernel(uni20::linalg::BlasBackend{}, uni20::linalg::assign_product_op{}, output,
                                      scalar_type{1.0}, conjugated, rhs),
            uni20::linalg::KernelAttempt::unsupported_transform);
  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 2);
  EXPECT_EQ((output[0, 0]), scalar_type{-7.0});
}

TEST(MatrixProductTest, AlphaZeroDoesNotReadProductOperands)
{
  double const nan = std::numeric_limits<double>::quiet_NaN();
  uni20::DenseMatrix<double> lhs(1, 1);
  uni20::DenseMatrix<double> rhs(1, 1);
  uni20::DenseMatrix<double> output(1, 1);
  lhs[0, 0] = nan;
  rhs[0, 0] = nan;
  output[0, 0] = 7.0;

  uni20::linalg::add_product(uni20::linalg::CpuReferenceBackend{}, output, lhs, rhs, 0.0);
  EXPECT_DOUBLE_EQ((output[0, 0]), 7.0);

  uni20::linalg::assign_product(uni20::linalg::CpuReferenceBackend{}, output, lhs, rhs, 0.0);
  EXPECT_DOUBLE_EQ((output[0, 0]), 0.0);
}

TEST(MatrixProductTest, EmptyInnerDimensionProducesZeroMatrix)
{
  uni20::DenseMatrix<double> lhs(2, 0);
  uni20::DenseMatrix<double> rhs(0, 3);
  uni20::DenseMatrix<double> output;

  uni20::linalg::assign_product(uni20::linalg::CpuReferenceBackend{}, output, lhs, rhs);

  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 3);
  for (uni20::index_type row = 0; row < output.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < output.cols(); ++col)
    {
      EXPECT_DOUBLE_EQ((output[row, col]), 0.0);
    }
  }
}

TEST(MatrixProductTest, AddProductRejectsShapeMismatchWithoutResize)
{
  uni20::DenseMatrix<double> lhs(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  uni20::DenseMatrix<double> output(1, 1);
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::linalg::add_product(output, lhs, rhs), std::runtime_error);
  EXPECT_EQ(output.rows(), 1);
  EXPECT_EQ(output.cols(), 1);
}

TEST(MatrixProductTest, RejectsObviousSameObjectAliasingBeforeResize)
{
  uni20::DenseMatrix<double> lhs(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::linalg::assign_product(lhs, lhs, rhs), std::runtime_error);
  EXPECT_EQ(lhs.rows(), 2);
  EXPECT_EQ(lhs.cols(), 2);
}

#if UNI20_HAS_FLOAT128
TEST(MatrixProductTest, ConfiguredFloat128UsesTensorProductFrontEnd) { check_reference_product<uni20::float128>(); }
#endif
