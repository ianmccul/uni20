#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/blas_matrix.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/mdspan/mdspan.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using extents_2d = stdex::dextents<uni20::index_type, 2>;
using uni20::linalg::blas::MatrixTransform;

template <class Scalar> using left_mdspan = stdex::mdspan<Scalar, extents_2d, stdex::layout_left>;

template <class Scalar> struct ValueTransformAccessor
{
    using element_type = Scalar;
    using data_handle_type = Scalar*;
    using reference = Scalar;
    using offset_policy = ValueTransformAccessor;

    constexpr data_handle_type offset(data_handle_type ptr, std::size_t offset) const { return ptr + offset; }

    constexpr reference access(data_handle_type ptr, std::size_t offset) const { return Scalar{2} * ptr[offset]; }
};

template <class Scalar>
using value_transform_mdspan = stdex::mdspan<Scalar, extents_2d, stdex::layout_left, ValueTransformAccessor<Scalar>>;

template <class Span>
concept can_try_blas_writable_matrix =
    requires(Span const& span) { uni20::linalg::blas::try_blas_writable_matrix(span); };

template <class Span>
concept can_try_blas_readable_matrix =
    requires(Span const& span) { uni20::linalg::blas::try_blas_readable_matrix(span); };

template <class Span>
concept can_try_blas_readable_matrix_with_transform =
    requires(Span const& span) { uni20::linalg::blas::try_blas_readable_matrix(span, MatrixTransform::normal); };

template <class Span>
concept can_try_lapack_writable_matrix =
    requires(Span const& span) { uni20::linalg::blas::try_lapack_writable_matrix(span); };

template <class Scalar>
concept can_lower_blas_trans_char = requires { uni20::linalg::blas::blas_trans_char<Scalar>(MatrixTransform::normal); };
} // namespace

TEST(BlasMatrixTransformTest, ComposeAndTransposeResultTransform)
{
  using uni20::linalg::blas::blas_trans_char;
  using uni20::linalg::blas::blas_trans_char_is_supported;
  using uni20::linalg::blas::compose;
  using uni20::linalg::blas::conjugates_values;
  using uni20::linalg::blas::swaps_axes;
  using uni20::linalg::blas::transpose_result_transform;

  EXPECT_EQ(std::to_underlying(MatrixTransform::normal), 0U);
  EXPECT_EQ(std::to_underlying(MatrixTransform::transpose), 1U);
  EXPECT_EQ(std::to_underlying(MatrixTransform::conjugate), 2U);
  EXPECT_EQ(std::to_underlying(MatrixTransform::conjugate_transpose), 3U);

  EXPECT_FALSE(swaps_axes(MatrixTransform::normal));
  EXPECT_TRUE(swaps_axes(MatrixTransform::transpose));
  EXPECT_FALSE(swaps_axes(MatrixTransform::conjugate));
  EXPECT_TRUE(swaps_axes(MatrixTransform::conjugate_transpose));

  EXPECT_FALSE(conjugates_values(MatrixTransform::normal));
  EXPECT_FALSE(conjugates_values(MatrixTransform::transpose));
  EXPECT_TRUE(conjugates_values(MatrixTransform::conjugate));
  EXPECT_TRUE(conjugates_values(MatrixTransform::conjugate_transpose));

  EXPECT_EQ(compose(MatrixTransform::transpose, MatrixTransform::transpose), MatrixTransform::normal);
  EXPECT_EQ(compose(MatrixTransform::conjugate, MatrixTransform::transpose), MatrixTransform::conjugate_transpose);
  EXPECT_EQ(compose(MatrixTransform::conjugate_transpose, MatrixTransform::conjugate), MatrixTransform::transpose);

  EXPECT_EQ(transpose_result_transform(MatrixTransform::normal), MatrixTransform::transpose);
  EXPECT_EQ(transpose_result_transform(MatrixTransform::transpose), MatrixTransform::normal);
  EXPECT_EQ(transpose_result_transform(MatrixTransform::conjugate_transpose), MatrixTransform::conjugate);
  EXPECT_EQ(transpose_result_transform(MatrixTransform::conjugate), MatrixTransform::conjugate_transpose);

  EXPECT_TRUE(blas_trans_char_is_supported<double>(MatrixTransform::conjugate));
  EXPECT_TRUE(blas_trans_char_is_supported<uni20::complex<double>>(MatrixTransform::conjugate_transpose));
  EXPECT_FALSE(blas_trans_char_is_supported<uni20::complex<double>>(MatrixTransform::conjugate));

  EXPECT_EQ(blas_trans_char<double>(MatrixTransform::conjugate), 'N');
  EXPECT_EQ(blas_trans_char<double>(MatrixTransform::conjugate_transpose), 'T');
  EXPECT_EQ(blas_trans_char<uni20::complex<double>>(MatrixTransform::conjugate), 'R');
  EXPECT_EQ(blas_trans_char<uni20::complex<double>>(MatrixTransform::conjugate_transpose), 'C');
}

TEST(BlasMatrixTransformDeathTest, InvalidTransformPanics)
{
  GTEST_FLAG_SET(death_test_style, "fast");
  auto const invalid = static_cast<MatrixTransform>(4U);

  EXPECT_DEATH((void)uni20::linalg::blas::blas_trans_char<uni20::complex<double>>(invalid), "invalid MatrixTransform");
  EXPECT_DEATH((void)uni20::linalg::blas::blas_trans_char<double>(invalid), "invalid MatrixTransform");
}

TEST(BlasMatrixOperandTest, ConvenienceApisRequireConfiguredScalarBackends)
{
  static_assert(can_lower_blas_trans_char<double>);
  static_assert(!can_lower_blas_trans_char<int>);

  static_assert(can_try_blas_writable_matrix<left_mdspan<double>>);
  static_assert(can_try_blas_readable_matrix<left_mdspan<double>>);
  static_assert(!can_try_blas_readable_matrix_with_transform<left_mdspan<double>>);
  static_assert(can_try_lapack_writable_matrix<left_mdspan<double>>);

  static_assert(!can_try_blas_writable_matrix<left_mdspan<int>>);
  static_assert(!can_try_blas_readable_matrix<left_mdspan<int>>);
  static_assert(!can_try_lapack_writable_matrix<left_mdspan<int>>);

  static_assert(uni20::StridedMdspanLike<value_transform_mdspan<double>>);
  static_assert(std::convertible_to<typename value_transform_mdspan<double>::data_handle_type, double*>);
  static_assert(!uni20::DefaultAccessorMdspanLike<value_transform_mdspan<double>>);
  static_assert(!can_try_blas_writable_matrix<value_transform_mdspan<double>>);
  static_assert(!can_try_blas_readable_matrix<value_transform_mdspan<double>>);
  static_assert(!can_try_lapack_writable_matrix<value_transform_mdspan<double>>);
}

TEST(BlasMatrixOperandTest, StagesColumnMajorMdspan)
{
  std::vector<double> storage(6);
  stdex::mdspan<double, extents_2d, stdex::layout_left> span(storage.data(), 2, 3);

  auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(span);
  ASSERT_TRUE(stage.has_value());
  EXPECT_EQ(stage->extent0, 2);
  EXPECT_EQ(stage->extent1, 3);
  EXPECT_EQ(stage->unit_stride_axis, 0);
  EXPECT_EQ(stage->nonunit_stride, 2);
  EXPECT_FALSE(stage->needs_conjugation);

  auto writable = uni20::linalg::blas::blas_writable_matrix(*stage);
  EXPECT_EQ(writable.rows, 2);
  EXPECT_EQ(writable.cols, 3);
  EXPECT_EQ(writable.leading_dimension, 2);

  auto readable = uni20::linalg::blas::blas_readable_matrix(*stage);
  EXPECT_EQ(readable.rows, 2);
  EXPECT_EQ(readable.cols, 3);
  EXPECT_EQ(readable.leading_dimension, 2);
  EXPECT_EQ(readable.transform, MatrixTransform::normal);

  auto lapack_matrix = uni20::linalg::blas::try_lapack_writable_matrix(span);
  ASSERT_TRUE(lapack_matrix.has_value());
  EXPECT_EQ(lapack_matrix->rows, 2);
  EXPECT_EQ(lapack_matrix->cols, 3);
}

TEST(BlasMatrixOperandTest, StagesRowMajorMdspanAsTransposedProviderMatrix)
{
  std::vector<double> storage(6);
  stdex::mdspan<double, extents_2d, stdex::layout_right> span(storage.data(), 2, 3);

  auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(span);
  ASSERT_TRUE(stage.has_value());
  EXPECT_EQ(stage->extent0, 2);
  EXPECT_EQ(stage->extent1, 3);
  EXPECT_EQ(stage->unit_stride_axis, 1);
  EXPECT_EQ(stage->nonunit_stride, 3);

  auto writable = uni20::linalg::blas::blas_writable_matrix(*stage);
  EXPECT_EQ(writable.rows, 3);
  EXPECT_EQ(writable.cols, 2);
  EXPECT_EQ(writable.leading_dimension, 3);

  auto readable = uni20::linalg::blas::blas_readable_matrix(*stage);
  EXPECT_EQ(readable.rows, 3);
  EXPECT_EQ(readable.cols, 2);
  EXPECT_EQ(readable.transform, MatrixTransform::transpose);

  EXPECT_FALSE(uni20::linalg::blas::try_lapack_writable_matrix(span).has_value());
}

TEST(BlasMatrixOperandTest, NormalizesUnobservedZeroRowProviderStride)
{
  using extents_type = stdex::dextents<uni20::index_type, 2>;
  stdex::mdspan<double, extents_type, stdex::layout_right> matrix(nullptr, 2, 0);

  auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(matrix);

  ASSERT_TRUE(stage.has_value());
  EXPECT_EQ(stage->extent0, 2);
  EXPECT_EQ(stage->extent1, 0);
  EXPECT_EQ(stage->unit_stride_axis, 1);
  EXPECT_EQ(stage->nonunit_stride, 1);
}

TEST(BlasMatrixOperandTest, NormalizesUnobservedSingletonProviderColumnStride)
{
  {
    std::vector<double> storage(3);
    stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{3, 1}, std::array<uni20::index_type, 2>{1, 0});
    stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

    auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(span);
    ASSERT_TRUE(stage.has_value());
    EXPECT_EQ(stage->unit_stride_axis, 0);
    EXPECT_EQ(stage->nonunit_stride, 3);

    auto writable = uni20::linalg::blas::blas_writable_matrix(*stage);
    EXPECT_EQ(writable.rows, 3);
    EXPECT_EQ(writable.cols, 1);
    EXPECT_EQ(writable.leading_dimension, 3);
  }

  {
    std::vector<double> storage(3);
    stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{1, 3}, std::array<uni20::index_type, 2>{0, 1});
    stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

    auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(span);
    ASSERT_TRUE(stage.has_value());
    EXPECT_EQ(stage->unit_stride_axis, 1);
    EXPECT_EQ(stage->nonunit_stride, 3);

    auto writable = uni20::linalg::blas::blas_writable_matrix(*stage);
    EXPECT_EQ(writable.rows, 3);
    EXPECT_EQ(writable.cols, 1);
    EXPECT_EQ(writable.leading_dimension, 3);
  }
}

TEST(BlasMatrixOperandTest, UsesSingletonAxisForOtherwiseStridedMatrix)
{
  {
    std::vector<double> storage(5);
    stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{3, 1}, std::array<uni20::index_type, 2>{2, 37});
    stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

    auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(span);
    ASSERT_TRUE(stage.has_value());
    EXPECT_EQ(stage->unit_stride_axis, 1);
    EXPECT_EQ(stage->nonunit_stride, 2);

    auto writable = uni20::linalg::blas::blas_writable_matrix(*stage);
    EXPECT_EQ(writable.rows, 1);
    EXPECT_EQ(writable.cols, 3);
    EXPECT_EQ(writable.leading_dimension, 2);
  }

  {
    std::vector<double> storage(5);
    stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{1, 3}, std::array<uni20::index_type, 2>{37, 2});
    stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

    auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(span);
    ASSERT_TRUE(stage.has_value());
    EXPECT_EQ(stage->unit_stride_axis, 0);
    EXPECT_EQ(stage->nonunit_stride, 2);

    auto writable = uni20::linalg::blas::blas_writable_matrix(*stage);
    EXPECT_EQ(writable.rows, 1);
    EXPECT_EQ(writable.cols, 3);
    EXPECT_EQ(writable.leading_dimension, 2);
  }
}

TEST(BlasMatrixOperandTest, DeclinesNonBlasMatrixStridePattern)
{
  std::vector<double> storage(16);
  stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{2, 3}, std::array<uni20::index_type, 2>{2, 5});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

  EXPECT_FALSE(uni20::linalg::blas::try_mdspan_matrix_stage(span).has_value());
}

TEST(BlasMatrixOperandTest, DeclinesValuesOutsideBlasIntegerRange)
{
  static_assert(std::is_signed_v<uni20::blas_int>);

  if constexpr (std::numeric_limits<uni20::index_type>::max() > std::numeric_limits<uni20::blas_int>::max())
  {
    std::vector<double> storage(1);
    auto const too_large =
        static_cast<uni20::index_type>(std::numeric_limits<uni20::blas_int>::max()) + uni20::index_type{1};

    {
      stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{too_large, 1},
                                                        std::array<uni20::index_type, 2>{1, 0});
      stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

      EXPECT_FALSE(uni20::linalg::blas::try_mdspan_matrix_stage(span).has_value());
    }

    {
      stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{2, 2},
                                                        std::array<uni20::index_type, 2>{1, too_large});
      stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

      EXPECT_FALSE(uni20::linalg::blas::try_mdspan_matrix_stage(span).has_value());
    }

    {
      stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{3, 1},
                                                        std::array<uni20::index_type, 2>{1, too_large});
      stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

      EXPECT_TRUE(uni20::linalg::blas::try_mdspan_matrix_stage(span).has_value());
    }
  }
  else
  {
    GTEST_SKIP() << "index_type cannot represent values outside the configured BLAS integer range";
  }
}

TEST(BlasMatrixOperandTest, AccessorTraitMarksConjugatingViews)
{
  std::vector<uni20::complex<double>> storage(4);
  stdex::mdspan<uni20::complex<double>, extents_2d, stdex::layout_left> span(storage.data(), 2, 2);
  auto conjugated = uni20::conj(span);
  static_assert(uni20::accessor_applies_conjugation_v<typename decltype(conjugated)::accessor_type>);

  auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(conjugated);
  ASSERT_TRUE(stage.has_value());
  EXPECT_TRUE(stage->needs_conjugation);

  auto readable = uni20::linalg::blas::blas_readable_matrix(*stage);
  EXPECT_EQ(readable.transform, MatrixTransform::conjugate);
}
