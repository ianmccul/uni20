#include <uni20/common/mdspan.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/mdspan_matrix_operand.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
using extents_2d = stdex::dextents<uni20::index_type, 2>;
using uni20::linalg::blas::MatrixTransform;

template <class Scalar> using left_mdspan = stdex::mdspan<Scalar, extents_2d, stdex::layout_left>;

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
  using uni20::linalg::blas::compose;
  using uni20::linalg::blas::standard_blas_trans_char;
  using uni20::linalg::blas::transpose_result_transform;

  EXPECT_EQ(compose(MatrixTransform::transpose, MatrixTransform::transpose), MatrixTransform::normal);
  EXPECT_EQ(compose(MatrixTransform::conjugate, MatrixTransform::transpose), MatrixTransform::conjugate_transpose);
  EXPECT_EQ(compose(MatrixTransform::conjugate_transpose, MatrixTransform::conjugate), MatrixTransform::transpose);

  EXPECT_EQ(transpose_result_transform(MatrixTransform::normal), MatrixTransform::transpose);
  EXPECT_EQ(transpose_result_transform(MatrixTransform::transpose), MatrixTransform::normal);
  EXPECT_EQ(transpose_result_transform(MatrixTransform::conjugate_transpose), MatrixTransform::conjugate);
  EXPECT_EQ(transpose_result_transform(MatrixTransform::conjugate), MatrixTransform::conjugate_transpose);

  EXPECT_EQ(standard_blas_trans_char<double>(MatrixTransform::conjugate), 'N');
  EXPECT_EQ(standard_blas_trans_char<double>(MatrixTransform::conjugate_transpose), 'T');
  EXPECT_FALSE(standard_blas_trans_char<uni20::complex<double>>(MatrixTransform::conjugate));
  EXPECT_EQ(standard_blas_trans_char<uni20::complex<double>>(MatrixTransform::conjugate_transpose), 'C');

  EXPECT_EQ(blas_trans_char<double>(MatrixTransform::conjugate), 'N');
  EXPECT_FALSE(blas_trans_char<uni20::complex<double>>(MatrixTransform::conjugate));
  EXPECT_EQ(blas_trans_char<uni20::complex<double>>(MatrixTransform::conjugate_transpose), 'C');
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
