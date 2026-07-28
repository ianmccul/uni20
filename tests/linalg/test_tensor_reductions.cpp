#include <uni20/common/gtest.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/linalg.hpp>
#include <uni20/tensor/generated.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace
{

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

TEST(TensorReductionTest, InnerProductReturnsScalarTensorOrHostScalar)
{
  uni20::Tensor<double, 2> lhs(2, 3);
  uni20::RowMajorTensor<double, 2> rhs(2, 3);
  double expected = 0.0;
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
    {
      lhs[row, column] = static_cast<double>(row * 3 + column + 1);
      rhs[row, column] = static_cast<double>(2 * row - column + 1);
      expected += lhs[row, column] * rhs[row, column];
    }

  auto result = uni20::inner_product(lhs, rhs);
  auto const host_result = uni20::inner_product_host(lhs, rhs);

  static_assert(std::same_as<decltype(result), uni20::ScalarTensor<double>>);
  EXPECT_DOUBLE_EQ(result[], expected);
  EXPECT_DOUBLE_EQ(host_result, expected);
}

TEST(TensorReductionTest, ExplicitScalarOutputUsesTheSameKernel)
{
  uni20::Tensor<float, 1> lhs(3);
  uni20::Tensor<float, 1> rhs(3);
  uni20::ScalarTensor<float> output;
  lhs[0] = 1.0F;
  lhs[1] = 2.0F;
  lhs[2] = 3.0F;
  rhs[0] = 4.0F;
  rhs[1] = -1.0F;
  rhs[2] = 2.0F;

  uni20::inner_product(output, lhs, rhs);

  EXPECT_FLOAT_EQ(output[], 8.0F);
}

TEST(TensorReductionTest, DeferredInputsAndOutputsUseHostLeases)
{
  uni20::test::DeferredHostTensor<double, 1> lhs(3);
  uni20::test::DeferredHostTensor<double, 1> rhs(3);
  uni20::test::DeferredHostTensor<double, 0> sum_output(
      typename uni20::test::DeferredHostTensor<double, 0>::extents_type{});
  lhs.storage() = {1.0, 2.0, 3.0};
  rhs.storage() = {4.0, -1.0, 2.0};

  uni20::sum(sum_output, lhs);
  auto inner = uni20::inner_product(lhs, rhs);
  auto const norm = uni20::norm_host(lhs);

  EXPECT_DOUBLE_EQ(sum_output.storage()[0], 6.0);
  EXPECT_DOUBLE_EQ(inner[], 8.0);
  EXPECT_DOUBLE_EQ(norm, std::sqrt(14.0));
}

TEST(TensorReductionTest, FullSumReturnsScalarTensorOrHostScalar)
{
  uni20::Tensor<double, 2> input(2, 3);
  double expected = 0.0;
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
    {
      input[row, column] = static_cast<double>(row * 3 + column + 1);
      expected += input[row, column];
    }

  auto result = uni20::sum(input);
  auto const host_result = uni20::sum_host(input);
  auto explicit_axes_result = uni20::sum(input, 1, 0);

  static_assert(std::same_as<decltype(result), uni20::ScalarTensor<double>>);
  static_assert(std::same_as<decltype(explicit_axes_result), uni20::ScalarTensor<double>>);
  EXPECT_DOUBLE_EQ(result[], expected);
  EXPECT_DOUBLE_EQ(explicit_axes_result[], expected);
  EXPECT_DOUBLE_EQ(host_result, expected);
}

TEST(TensorReductionTest, SumAndInnerProductRecoverCancellationWithoutPromotion)
{
  uni20::Tensor<double, 1> values(3);
  uni20::Tensor<double, 1> ones(3);
  values[0] = 1.0e16;
  values[1] = 1.0;
  values[2] = -1.0e16;
  ones[0] = 1.0;
  ones[1] = 1.0;
  ones[2] = 1.0;

  EXPECT_DOUBLE_EQ(uni20::sum_host(values), 1.0);
  EXPECT_DOUBLE_EQ(uni20::inner_product_host(values, ones), 1.0);
}

TEST(TensorReductionTest, PartialSumRemovesAxesAndPreservesCanonicalLayout)
{
  uni20::Tensor<double, 3> column_major(2, 3, 4);
  uni20::RowMajorTensor<double, 3> row_major(2, 3, 4);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      for (uni20::index_type k = 0; k < 4; ++k)
      {
        double const value = static_cast<double>(100 * i + 10 * j + k);
        column_major[i, j, k] = value;
        row_major[i, j, k] = value;
      }

  auto middle = uni20::sum(column_major, 1);
  auto last = uni20::sum(row_major, -1);
  auto outer = uni20::sum(column_major, 2, 0);

  static_assert(std::same_as<decltype(middle), uni20::Tensor<double, 2>>);
  static_assert(std::same_as<decltype(last), uni20::RowMajorTensor<double, 2>>);
  static_assert(std::same_as<decltype(outer), uni20::Tensor<double, 1>>);
  ASSERT_EQ(middle.extent(0), 2);
  ASSERT_EQ(middle.extent(1), 4);
  ASSERT_EQ(last.extent(0), 2);
  ASSERT_EQ(last.extent(1), 3);
  ASSERT_EQ(outer.extent(0), 3);

  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type k = 0; k < 4; ++k)
      EXPECT_DOUBLE_EQ((middle[i, k]), static_cast<double>(300 * i + 30 + 3 * k));

  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ((last[i, j]), static_cast<double>(400 * i + 40 * j + 6));

  for (uni20::index_type j = 0; j < 3; ++j)
    EXPECT_DOUBLE_EQ(outer[j], static_cast<double>(412 + 80 * j));
}

TEST(TensorReductionTest, PartialSumResizesExplicitTensorOutput)
{
  uni20::Tensor<double, 3> input(2, 3, 4);
  uni20::RowMajorTensor<double, 2> output(1, 1);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      for (uni20::index_type k = 0; k < 4; ++k)
        input[i, j, k] = static_cast<double>(100 * i + 10 * j + k);

  uni20::sum(output, input, 1);

  ASSERT_EQ(output.extent(0), 2);
  ASSERT_EQ(output.extent(1), 4);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type k = 0; k < 4; ++k)
      EXPECT_DOUBLE_EQ((output[i, k]), static_cast<double>(300 * i + 30 + 3 * k));
}

TEST(TensorReductionTest, SumObservesGeneratedAndConjugatingAccessors)
{
  auto generated = uni20::ones<double>(2, 3, 4);
  auto generated_result = uni20::sum(generated, 1);

  static_assert(std::same_as<decltype(generated_result), uni20::Tensor<double, 2>>);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type k = 0; k < 4; ++k)
      EXPECT_DOUBLE_EQ((generated_result[i, k]), 3.0);

  using Complex = uni20::complex<double>;
  uni20::Tensor<Complex, 2> input(2, 2);
  input[0, 0] = Complex{1.0, 2.0};
  input[0, 1] = Complex{3.0, -4.0};
  input[1, 0] = Complex{-2.0, 1.0};
  input[1, 1] = Complex{5.0, 3.0};
  auto conjugated = uni20::conj(input);
  auto result = uni20::sum(conjugated, 0);

  EXPECT_FLOATING_EQ(result[0], (uni20::conj(input[0, 0]) + uni20::conj(input[1, 0])));
  EXPECT_FLOATING_EQ(result[1], (uni20::conj(input[0, 1]) + uni20::conj(input[1, 1])));
}

TEST(TensorReductionTest, SumHandlesRankZeroAndZeroExtents)
{
  uni20::ScalarTensor<double> scalar;
  scalar[] = 4.5;
  auto scalar_result = uni20::sum(scalar);
  EXPECT_DOUBLE_EQ(scalar_result[], 4.5);
  EXPECT_DOUBLE_EQ(uni20::sum_host(scalar), 4.5);

  uni20::Tensor<double, 3> empty_reduced_axis(2, 0, 3);
  auto zeros = uni20::sum(empty_reduced_axis, 1);
  ASSERT_EQ(zeros.extent(0), 2);
  ASSERT_EQ(zeros.extent(1), 3);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ((zeros[i, j]), 0.0);

  auto empty_result = uni20::sum(empty_reduced_axis, 0);
  EXPECT_EQ(empty_result.extent(0), 0);
  EXPECT_EQ(empty_result.extent(1), 3);
}

TEST(TensorReductionTest, SumRejectsDuplicateAndOutOfRangeAxes)
{
  uni20::Tensor<double, 2> input(2, 3);
  ErrorModeGuard const error_mode;

  EXPECT_THROW((void)uni20::sum(input, 0, 0), std::runtime_error);
  EXPECT_THROW((void)uni20::sum(input, 2), std::runtime_error);
  EXPECT_THROW((void)uni20::sum(input, -3), std::runtime_error);
}

TEST(TensorReductionTest, ScalarInputsReduceTheirSoleElement)
{
  using Complex = uni20::complex<double>;
  uni20::ScalarTensor<Complex> lhs;
  uni20::ScalarTensor<Complex> rhs;
  lhs[] = Complex{3.0, 4.0};
  rhs[] = Complex{2.0, -1.0};

  EXPECT_FLOATING_EQ(uni20::inner_product_host(lhs, rhs), (Complex{2.0, -11.0}));
  EXPECT_DOUBLE_EQ(uni20::norm_host(lhs), 5.0);
}

TEST(TensorReductionTest, InnerProductObservesConjugatingAccessorSemantics)
{
  using Complex = uni20::complex<double>;
  uni20::Tensor<Complex, 1> lhs(2);
  uni20::Tensor<Complex, 1> rhs(2);
  lhs[0] = Complex{1.0, 2.0};
  lhs[1] = Complex{3.0, -1.0};
  rhs[0] = Complex{2.0, -1.0};
  rhs[1] = Complex{-4.0, 2.0};

  auto const ordinary = uni20::inner_product_host(lhs, rhs);
  auto conjugated_lhs = uni20::conj(lhs);
  auto const accessor_transformed = uni20::inner_product_host(conjugated_lhs, rhs);

  EXPECT_FLOATING_EQ(ordinary, (Complex{-14.0, -3.0}));
  EXPECT_FLOATING_EQ(accessor_transformed, (lhs[0] * rhs[0] + lhs[1] * rhs[1]));
}

TEST(TensorReductionTest, NormUsesScaledSumOfSquares)
{
  uni20::Tensor<double, 1> large(2);
  double const large_value = std::numeric_limits<double>::max() / 4.0;
  large[0] = large_value;
  large[1] = large_value;

  double const large_norm = uni20::norm_host(large);
  EXPECT_TRUE(std::isfinite(large_norm));
  EXPECT_DOUBLE_EQ(large_norm, std::hypot(large_value, large_value));

  uni20::Tensor<double, 1> tiny(2);
  double const tiny_value = std::numeric_limits<double>::min();
  tiny[0] = tiny_value;
  tiny[1] = tiny_value;

  double const tiny_norm = uni20::norm_host(tiny);
  EXPECT_GT(tiny_norm, 0.0);
  EXPECT_DOUBLE_EQ(tiny_norm, std::hypot(tiny_value, tiny_value));
}

TEST(TensorReductionTest, ComplexNormReturnsRealScalarTensor)
{
  using Complex = uni20::complex<double>;
  uni20::Tensor<Complex, 1> input(2);
  input[0] = Complex{3.0, 4.0};
  input[1] = Complex{0.0, 12.0};

  auto result = uni20::norm(input);

  static_assert(std::same_as<decltype(result), uni20::ScalarTensor<double>>);
  EXPECT_DOUBLE_EQ(result[], 13.0);
}

TEST(TensorReductionTest, GeneratedInputUsesAccessorRespectingCpuFallback)
{
  auto input = uni20::eye<double>(2, 3, 4);

  auto result = uni20::norm(input);
  auto const host_result = uni20::norm_host(input);

  static_assert(std::same_as<decltype(result), uni20::ScalarTensor<double>>);
  EXPECT_DOUBLE_EQ(result[], std::sqrt(2.0));
  EXPECT_DOUBLE_EQ(host_result, std::sqrt(2.0));
}

TEST(TensorReductionTest, ZeroExtentInputsReduceToZero)
{
  uni20::Tensor<double, 2> lhs(2, 0);
  uni20::Tensor<double, 2> rhs(2, 0);

  EXPECT_DOUBLE_EQ(uni20::inner_product_host(lhs, rhs), 0.0);
  EXPECT_DOUBLE_EQ(uni20::norm_host(lhs), 0.0);
}

TEST(TensorReductionTest, InnerProductRejectsDifferentExtentsBeforeDispatch)
{
  uni20::Tensor<double, 2> lhs(2, 3);
  uni20::Tensor<double, 2> rhs(3, 2);
  ErrorModeGuard const error_mode;

  EXPECT_THROW((void)uni20::inner_product_host(lhs, rhs), std::runtime_error);
}

TEST(TensorReductionTest, CpuTypeProbeAcceptsHostAndRankZeroOutputs)
{
  uni20::Tensor<double, 1> lhs(2);
  uni20::Tensor<double, 1> rhs(2);
  uni20::ScalarTensor<double> output;
  auto lhs_span = lhs.mdspan();
  auto rhs_span = rhs.mdspan();
  auto output_span = output.mdspan();
  double host_output = 0.0;

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{},
                                                 uni20::linalg::inner_product_op{}, output_span, lhs_span, rhs_span),
            uni20::linalg::KernelTypeAcceptance::yes);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{},
                                                 uni20::linalg::inner_product_op{}, host_output, lhs_span, rhs_span),
            uni20::linalg::KernelTypeAcceptance::yes);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, uni20::linalg::norm_op{},
                                                 output_span, lhs_span),
            uni20::linalg::KernelTypeAcceptance::yes);

  uni20::Tensor<double, 2> matrix(2, 3);
  uni20::Tensor<double, 1> partial_output(3);
  auto matrix_span = matrix.mdspan();
  auto partial_output_span = partial_output.mdspan();
  auto sum_operation = uni20::linalg::sum_reduction_op<2, 1>{.axes = uni20::linalg::make_reduction_axes<2>(0)};
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, sum_operation,
                                                 partial_output_span, matrix_span),
            uni20::linalg::KernelTypeAcceptance::yes);

  uni20::Tensor<int, 2> integer_matrix(2, 3);
  uni20::Tensor<int, 1> integer_output(3);
  auto integer_matrix_span = integer_matrix.mdspan();
  auto integer_output_span = integer_output.mdspan();
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, sum_operation,
                                                 integer_output_span, integer_matrix_span),
            uni20::linalg::KernelTypeAcceptance::no);
}

} // namespace
