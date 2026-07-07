#include <uni20/common/mdspan.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/mdspan_matrix_operand.hpp>

#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <type_traits>
#include <vector>

namespace
{
using extents_2d = stdex::dextents<uni20::index_type, 2>;
using uni20::linalg::blas::MatrixTransform;

struct ConjugatingAccessor
{
    using element_type = uni20::complex<double>;
    using data_handle_type = element_type const*;
    using offset_policy = ConjugatingAccessor;
    using reference = element_type;
    using offset_type = uni20::index_type;

    constexpr reference access(data_handle_type ptr, offset_type offset) const { return std::conj(ptr[offset]); }
    constexpr data_handle_type offset(data_handle_type ptr, offset_type offset) const { return ptr + offset; }
};
} // namespace

template <> struct uni20::linalg::blas::accessor_applies_conjugation<ConjugatingAccessor> : std::true_type
{};

TEST(BlasMatrixTransformTest, ComposeAndTransposeResultTransform)
{
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

  auto readable = uni20::linalg::blas::blas_readable_matrix(*stage, MatrixTransform::transpose);
  EXPECT_EQ(readable.rows, 2);
  EXPECT_EQ(readable.cols, 3);
  EXPECT_EQ(readable.leading_dimension, 2);
  EXPECT_EQ(readable.transform, MatrixTransform::transpose);

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

TEST(BlasMatrixOperandTest, DeclinesNonBlasMatrixStridePattern)
{
  std::vector<double> storage(16);
  stdex::layout_stride::mapping<extents_2d> mapping(extents_2d{2, 3}, std::array<uni20::index_type, 2>{2, 5});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> span(storage.data(), mapping);

  EXPECT_FALSE(uni20::linalg::blas::try_mdspan_matrix_stage(span).has_value());
}

TEST(BlasMatrixOperandTest, AccessorTraitMarksConjugatingViews)
{
  static_assert(uni20::linalg::blas::accessor_applies_conjugation_v<ConjugatingAccessor>);

  std::vector<uni20::complex<double>> storage(4);
  stdex::mdspan<uni20::complex<double>, extents_2d, stdex::layout_left, ConjugatingAccessor> span(storage.data(), 2, 2);

  auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(span);
  ASSERT_TRUE(stage.has_value());
  EXPECT_TRUE(stage->needs_conjugation);

  auto readable = uni20::linalg::blas::blas_readable_matrix(*stage);
  EXPECT_EQ(readable.transform, MatrixTransform::conjugate);
}
