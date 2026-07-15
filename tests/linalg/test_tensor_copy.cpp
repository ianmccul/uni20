#include <uni20/core/types.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/tensor/copy.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <stdexcept>
#include <type_traits>

namespace
{
using complex_type = uni20::complex<double>;

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

void fill_complex_matrix(uni20::DenseMatrix<complex_type, uni20::RowMajor>& matrix)
{
  matrix[0, 0] = complex_type{1.0, 2.0};
  matrix[0, 1] = complex_type{3.0, -4.0};
  matrix[1, 0] = complex_type{-5.0, 6.0};
  matrix[1, 1] = complex_type{7.0, 8.0};
}
} // namespace

TEST(TensorCopyTest, CopyResizesOutputAndObservesLazyConjugation)
{
  uni20::DenseMatrix<complex_type, uni20::RowMajor> input(2, 2);
  uni20::DenseMatrix<complex_type> output;
  fill_complex_matrix(input);

  auto conjugated = uni20::conj(input);
  uni20::copy(output, conjugated);

  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 2);
  EXPECT_EQ((output[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((output[0, 1]), (complex_type{3.0, 4.0}));
  EXPECT_EQ((output[1, 0]), (complex_type{-5.0, -6.0}));
  EXPECT_EQ((output[1, 1]), (complex_type{7.0, -8.0}));
}

TEST(TensorCopyTest, MakeTensorInfersScalarExtentsAndSourceLayout)
{
  uni20::DenseMatrix<complex_type, uni20::RowMajor> input(2, 2);
  fill_complex_matrix(input);

  auto result = uni20::make_tensor(uni20::conj(input));

  using expected_type = uni20::Tensor<complex_type, 2, uni20::VectorStorage, uni20::RowMajor>;
  static_assert(std::same_as<decltype(result), expected_type>);
  EXPECT_EQ((result[0, 1]), (complex_type{3.0, 4.0}));
  EXPECT_EQ(result.mapping().stride(0), 2);
  EXPECT_EQ(result.mapping().stride(1), 1);
}

TEST(TensorCopyTest, CtadMaterializesConjugatedViewWithMatchingAliases)
{
  using complex_type = uni20::complex<double>;
  uni20::RowMajorTensor<complex_type, 2> input(2, 2);
  input[0, 0] = complex_type{1.0, 2.0};
  input[0, 1] = complex_type{3.0, 4.0};
  input[1, 0] = complex_type{5.0, 6.0};
  input[1, 1] = complex_type{7.0, 8.0};
  auto conjugated = uni20::conj(input);

  auto inferred = uni20::Tensor(conjugated);
  auto named = uni20::RowMajorTensor(conjugated);
  auto matrix = uni20::DenseMatrix(conjugated);
  auto basic = uni20::BasicTensor(conjugated);

  using expected_type = uni20::RowMajorTensor<complex_type, 2>;
  static_assert(std::same_as<decltype(inferred), expected_type>);
  static_assert(std::same_as<decltype(named), expected_type>);
  static_assert(std::same_as<decltype(matrix), expected_type>);
  static_assert(std::same_as<decltype(basic), expected_type>);
  EXPECT_NE(inferred.storage().data(), input.storage().data());
  EXPECT_EQ((inferred[0, 1]), (complex_type{3.0, -4.0}));
  EXPECT_EQ((named[1, 0]), (complex_type{5.0, -6.0}));
  EXPECT_EQ((matrix[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((basic[1, 1]), (complex_type{7.0, -8.0}));
}

TEST(TensorCopyTest, MakeTensorAcceptsExplicitLayoutAndBareMdspanSelector)
{
  uni20::DenseMatrix<complex_type, uni20::RowMajor> input(2, 2);
  fill_complex_matrix(input);
  auto conjugated = uni20::conj(input.mdspan());

  auto result = uni20::make_tensor<uni20::ColumnMajor>(uni20::linalg::CpuReferenceBackend{}, conjugated);

  static_assert(std::same_as<typename decltype(result)::layout_type, uni20::ColumnMajor>);
  EXPECT_EQ((result[1, 0]), (complex_type{-5.0, -6.0}));
  EXPECT_EQ(result.mapping().stride(0), 1);
  EXPECT_EQ(result.mapping().stride(1), 2);
}

TEST(TensorCopyTest, MakeTensorMaterializesStaticExtentsAsGeneralPurposeTensor)
{
  using fixed_extents = stdex::extents<uni20::index_type, 2, 3>;
  using fixed_tensor = uni20::BasicTensor<double, fixed_extents, uni20::VectorStorage, uni20::RowMajor>;
  fixed_tensor input(fixed_extents{});
  input[0, 0] = 1.0;
  input[1, 2] = 6.0;

  auto result = uni20::make_tensor(input);

  using expected_type = uni20::Tensor<double, 2, uni20::VectorStorage, uni20::RowMajor>;
  static_assert(std::same_as<decltype(result), expected_type>);
  EXPECT_EQ(result.rows(), 2);
  EXPECT_EQ(result.cols(), 3);
  EXPECT_DOUBLE_EQ((result[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((result[1, 2]), 6.0);

  fixed_tensor generated(uni20::ones<double>(2, 3));
  static_assert(std::same_as<decltype(generated), fixed_tensor>);
  EXPECT_DOUBLE_EQ((generated[0, 2]), 1.0);
  EXPECT_DOUBLE_EQ((generated[1, 1]), 1.0);
}

TEST(TensorCopyTest, MakeTensorUsesDefaultLayoutForNoncanonicalSourceType)
{
  using input_type = uni20::StridedTensor<double, 2>;
  input_type input(input_type::extents_type{2, 2}, std::array<uni20::index_type, 2>{2, 1});
  input[0, 0] = 1.0;
  input[0, 1] = 2.0;
  input[1, 0] = 3.0;
  input[1, 1] = 4.0;

  auto result = uni20::make_tensor(input);

  static_assert(std::same_as<typename decltype(result)::layout_type, uni20::ColumnMajor>);
  EXPECT_EQ(result.mapping().stride(0), 1);
  EXPECT_EQ(result.mapping().stride(1), 2);
  EXPECT_DOUBLE_EQ((result[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 3.0);
}

TEST(TensorCopyTest, ExplicitStaticExtentsRejectMismatchedSourceShape)
{
  using fixed_extents = stdex::extents<uni20::index_type, 2, 3>;
  using fixed_tensor = uni20::BasicTensor<double, fixed_extents>;
  ErrorModeGuard const error_mode;

  EXPECT_THROW(static_cast<void>(fixed_tensor(uni20::ones<double>(2, 4))), std::runtime_error);
}

TEST(TensorCopyTest, CpuReferenceCopySupportsScalarConversion)
{
  uni20::Tensor<float, 1> input(3);
  uni20::Tensor<double, 1> output;
  input[0] = 1.25F;
  input[1] = -2.5F;
  input[2] = 4.0F;

  uni20::copy(output, input);

  EXPECT_EQ(output.extent(0), 3);
  EXPECT_DOUBLE_EQ(output[0], 1.25);
  EXPECT_DOUBLE_EQ(output[1], -2.5);
  EXPECT_DOUBLE_EQ(output[2], 4.0);
}
