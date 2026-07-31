#include <uni20/common/trace.hpp>
#include <uni20/core/types.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace
{
using mutable_matrix = uni20::DenseMatrix<double, uni20::RowMajor>;
using mutable_reshape = decltype(uni20::reshape_view(std::declval<mutable_matrix&>(), 3, 2));
using const_reshape = decltype(uni20::reshape_view(std::declval<mutable_matrix const&>(), 3, 2));
using strided_matrix = uni20::StridedTensor<double, 2>;
using generated_matrix = decltype(uni20::ones<double>(2, 3));
using const_strided_matrix = uni20::ConstTensorView<strided_matrix>;
using metadata_extents = stdex::dextents<uni20::index_type, 2>;
using metadata_span = uni20::mdspec<double const, metadata_extents, stdex::layout_stride,
                                    stdex::default_accessor<double const>, std::size_t>;

static_assert(uni20::MutableRankedImmediateTensorView<mutable_reshape, 2>);
static_assert(uni20::RankedImmediateTensorView<const_reshape, 2>);
static_assert(!uni20::MutableImmediateTensorView<const_reshape>);
static_assert(!uni20::OwningTensor<mutable_reshape>);
static_assert(std::same_as<typename mutable_reshape::layout_type, uni20::RowMajor>);

template <class Tensor>
concept HasMutableMatrixElement = requires(Tensor& tensor) { tensor[0, 0] = 1.0; };

static_assert(HasMutableMatrixElement<mutable_reshape>);
static_assert(!HasMutableMatrixElement<mutable_reshape const>);
static_assert(std::same_as<decltype(std::declval<mutable_reshape const&>()[0, 0]), double const&>);

template <class Tensor>
concept CanReshapeViewRvalue = requires(Tensor tensor) { uni20::reshape_view(std::move(tensor), 6); };

static_assert(!CanReshapeViewRvalue<mutable_matrix>);

template <class Tensor>
concept CanAutomaticallyReshapeView = requires(Tensor& tensor) { uni20::reshape_view(tensor, 6); };

static_assert(!CanAutomaticallyReshapeView<strided_matrix>);
static_assert(!CanAutomaticallyReshapeView<generated_matrix>);

template <class Tensor>
concept CanImplicitlyReshapeValue = requires(Tensor const& tensor) { uni20::reshape(tensor, 6); };

template <class Tensor>
concept CanExplicitlyReshapeValue = requires(Tensor const& tensor) { uni20::reshape<uni20::ColumnMajor>(tensor, 6); };

static_assert(!CanImplicitlyReshapeValue<const_strided_matrix>);
static_assert(CanExplicitlyReshapeValue<const_strided_matrix>);

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
} // namespace

TEST(TensorReshapeTest, RowMajorViewAliasesSourceAcrossRanks)
{
  mutable_matrix input(2, 3);
  double value = 1.0;
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type col = 0; col < 3; ++col)
      input[row, col] = value++;

  auto reshaped = uni20::reshape_view(input, 3, 2);

  static_assert(std::same_as<typename decltype(reshaped)::layout_type, uni20::RowMajor>);
  EXPECT_DOUBLE_EQ((reshaped[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((reshaped[1, 0]), 3.0);
  EXPECT_DOUBLE_EQ((reshaped[2, 1]), 6.0);
  reshaped[1, 1] = 40.0;
  EXPECT_DOUBLE_EQ((input[1, 0]), 40.0);
}

TEST(TensorReshapeTest, ColumnMajorViewPreservesColumnMajorSequence)
{
  uni20::DenseMatrix<double> input(2, 3);
  input[0, 0] = 1.0;
  input[0, 1] = 2.0;
  input[0, 2] = 3.0;
  input[1, 0] = 4.0;
  input[1, 1] = 5.0;
  input[1, 2] = 6.0;

  auto reshaped = uni20::reshape_view(input, 3, 2);

  static_assert(std::same_as<typename decltype(reshaped)::layout_type, uni20::ColumnMajor>);
  EXPECT_DOUBLE_EQ((reshaped[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((reshaped[1, 0]), 4.0);
  EXPECT_DOUBLE_EQ((reshaped[2, 0]), 2.0);
  EXPECT_DOUBLE_EQ((reshaped[0, 1]), 5.0);
  EXPECT_DOUBLE_EQ((reshaped[1, 1]), 3.0);
  EXPECT_DOUBLE_EQ((reshaped[2, 1]), 6.0);
}

TEST(TensorReshapeTest, NestedViewPreservesStaticColumnMajorOrderAcrossSingletonExtent)
{
  uni20::DenseMatrix<double> input(2, 2);
  input[0, 0] = 1.0;
  input[0, 1] = 2.0;
  input[1, 0] = 3.0;
  input[1, 1] = 4.0;

  auto flattened = uni20::reshape_view(input, 1, 4);
  auto restored = uni20::reshape_view(flattened, 2, 2);

  EXPECT_DOUBLE_EQ((restored[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((restored[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((restored[1, 0]), 3.0);
  EXPECT_DOUBLE_EQ((restored[1, 1]), 4.0);
}

TEST(TensorReshapeTest, ViewInfersOneExtent)
{
  uni20::Tensor<int, 1> input(12);
  for (uni20::index_type i = 0; i < 12; ++i)
    input[i] = static_cast<int>(i);

  auto reshaped = uni20::reshape_view(input, 3, -1, 2);

  static_assert(decltype(reshaped)::extents_type::rank() == 3);
  EXPECT_EQ(reshaped.extent(0), 3);
  EXPECT_EQ(reshaped.extent(1), 2);
  EXPECT_EQ(reshaped.extent(2), 2);
  EXPECT_EQ((reshaped[2, 1, 1]), 11);
}

TEST(TensorReshapeTest, ViewInfersZeroExtentForEmptySource)
{
  uni20::Tensor<int, 2> input(0, 5);

  auto reshaped = uni20::reshape_view(input, -1, 2, 3);

  EXPECT_EQ(reshaped.extent(0), 0);
  EXPECT_EQ(reshaped.extent(1), 2);
  EXPECT_EQ(reshaped.extent(2), 3);
}

TEST(TensorReshapeTest, ExplicitViewSelectsOrderForRankOneStridedSource)
{
  uni20::StridedTensor<int, 1> input(6);
  for (uni20::index_type i = 0; i < 6; ++i)
    input[i] = static_cast<int>(i + 1);

  auto left = uni20::reshape_view_left(input, 2, 3);
  auto right = uni20::reshape_view_right(input, 2, 3);

  static_assert(std::same_as<typename decltype(left)::layout_type, uni20::ColumnMajor>);
  static_assert(std::same_as<typename decltype(right)::layout_type, uni20::RowMajor>);
  EXPECT_EQ((left[0, 0]), 1);
  EXPECT_EQ((left[1, 0]), 2);
  EXPECT_EQ((right[0, 0]), 1);
  EXPECT_EQ((right[0, 1]), 2);
}

TEST(TensorReshapeTest, ViewAcceptsArbitrarySingletonStrideInContiguousMapping)
{
  using tensor_type = uni20::StridedTensor<int, 2>;
  tensor_type input(typename tensor_type::extents_type{1, 6}, std::array<uni20::index_type, 2>{99, 1});
  for (uni20::index_type i = 0; i < 6; ++i)
    input.storage()[static_cast<std::size_t>(i)] = static_cast<int>(i + 1);

  auto reshaped = uni20::reshape_view_right(input, 2, 3);

  EXPECT_EQ((reshaped[0, 0]), 1);
  EXPECT_EQ((reshaped[1, 2]), 6);
}

TEST(TensorReshapeTest, ContiguousMappingAnalysisDoesNotRequireADataHandle)
{
  metadata_extents const extents{2, 3};
  metadata_span const row_major{0, metadata_span::mapping_type{extents, std::array<uni20::index_type, 2>{3, 1}},
                                metadata_span::accessor_type{}};
  metadata_span const noncontiguous{0, metadata_span::mapping_type{extents, std::array<uni20::index_type, 2>{4, 1}},
                                    metadata_span::accessor_type{}};

  EXPECT_TRUE(uni20::detail::has_canonical_contiguous_mapping<uni20::RowMajor>(row_major));
  EXPECT_FALSE(uni20::detail::has_canonical_contiguous_mapping<uni20::RowMajor>(noncontiguous));
}

TEST(TensorReshapeTest, ViewRejectsInvalidShapeAndNoncontiguousMapping)
{
  ErrorModeGuard const error_mode;
  uni20::Tensor<int, 1> contiguous(6);
  EXPECT_THROW(static_cast<void>(uni20::reshape_view(contiguous, 5)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(uni20::reshape_view(contiguous, -1, -1)), std::runtime_error);

  using tensor_type = uni20::StridedTensor<int, 2>;
  tensor_type noncontiguous(typename tensor_type::extents_type{2, 3}, std::array<uni20::index_type, 2>{4, 1});
  EXPECT_THROW(static_cast<void>(uni20::reshape_view_right(noncontiguous, 6)), std::runtime_error);
}

TEST(TensorReshapeTest, InplaceReshapeKeepsAllocationAndChangesDescriptor)
{
  mutable_matrix input(2, 3);
  double* const original_data = input.storage().data();
  double value = 1.0;
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type col = 0; col < 3; ++col)
      input[row, col] = value++;

  uni20::reshape_inplace(input, 3, 2);

  EXPECT_EQ(input.storage().data(), original_data);
  EXPECT_EQ(input.rows(), 3);
  EXPECT_EQ(input.cols(), 2);
  EXPECT_DOUBLE_EQ((input[1, 0]), 3.0);
  EXPECT_DOUBLE_EQ((input[2, 1]), 6.0);
}

TEST(TensorReshapeTest, OwningReshapeCopiesLvalue)
{
  mutable_matrix input(2, 3);
  double* const original_data = input.storage().data();
  for (uni20::index_type i = 0; i < 6; ++i)
    input.storage()[static_cast<std::size_t>(i)] = double(i + 1);

  auto result = uni20::reshape(input, 6);

  static_assert(uni20::OwningTensor<decltype(result)>);
  static_assert(decltype(result)::rank() == 1);
  EXPECT_NE(result.storage().data(), original_data);
  EXPECT_EQ(input.rows(), 2);
  EXPECT_EQ(input.cols(), 3);
  EXPECT_DOUBLE_EQ(result[5], 6.0);
}

TEST(TensorReshapeTest, OwningReshapeReusesRvalueAllocation)
{
  mutable_matrix input(2, 3);
  double* const original_data = input.storage().data();
  for (uni20::index_type i = 0; i < 6; ++i)
    input.storage()[static_cast<std::size_t>(i)] = double(i + 1);

  auto result = uni20::reshape(std::move(input), 3, 2);

  EXPECT_EQ(result.storage().data(), original_data);
  EXPECT_EQ(result.extent(0), 3);
  EXPECT_EQ(result.extent(1), 2);
  EXPECT_DOUBLE_EQ((result[2, 1]), 6.0);
}

TEST(TensorReshapeTest, OwningReshapeMaterializesGeneratedInput)
{
  auto result = uni20::reshape(uni20::eye<double>(2, 3), 3, 2);

  static_assert(uni20::OwningTensor<decltype(result)>);
  static_assert(std::same_as<typename decltype(result)::layout_type, uni20::ColumnMajor>);
  EXPECT_DOUBLE_EQ((result[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 1.0);
  EXPECT_DOUBLE_EQ((result[2, 0]), 0.0);
}

TEST(TensorReshapeTest, OwningGeneratedReshapeAcceptsExplicitLayout)
{
  auto result = uni20::reshape<uni20::RowMajor>(uni20::eye<double>(2, 3), 3, 2);

  static_assert(std::same_as<typename decltype(result)::layout_type, uni20::RowMajor>);
  EXPECT_DOUBLE_EQ((result[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((result[2, 0]), 1.0);
}
