#include <uni20/common/gtest.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/linalg.hpp>
#include <uni20/tensor/generated.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

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
}

} // namespace
