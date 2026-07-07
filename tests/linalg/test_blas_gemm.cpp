#include <uni20/common/mdspan.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/gemm.hpp>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <vector>

namespace
{
using extents_2d = stdex::dextents<uni20::index_type, 2>;
using uni20::linalg::blas::MatrixTransform;

template <class Span> void fill_matrix(Span span, std::initializer_list<double> values)
{
  auto it = values.begin();
  for (uni20::index_type row = 0; row < static_cast<uni20::index_type>(span.extent(0)); ++row)
  {
    for (uni20::index_type col = 0; col < static_cast<uni20::index_type>(span.extent(1)); ++col)
    {
      span[row, col] = *it;
      ++it;
    }
  }
}
} // namespace

TEST(BlasGemmTest, MultipliesColumnMajorMdspans)
{
  std::vector<double> a_storage(6);
  std::vector<double> b_storage(6);
  std::vector<double> c_storage(4);

  stdex::mdspan<double, extents_2d, stdex::layout_left> a(a_storage.data(), 2, 3);
  stdex::mdspan<double, extents_2d, stdex::layout_left> b(b_storage.data(), 3, 2);
  stdex::mdspan<double, extents_2d, stdex::layout_left> c(c_storage.data(), 2, 2);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(b, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  EXPECT_TRUE(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 154.0);
}

TEST(BlasGemmTest, RewritesRowMajorOutput)
{
  std::vector<double> a_storage(6);
  std::vector<double> b_storage(6);
  std::vector<double> c_storage(4);

  stdex::mdspan<double, extents_2d, stdex::layout_right> a(a_storage.data(), 2, 3);
  stdex::mdspan<double, extents_2d, stdex::layout_right> b(b_storage.data(), 3, 2);
  stdex::mdspan<double, extents_2d, stdex::layout_right> c(c_storage.data(), 2, 2);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(b, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  uni20::linalg::blas::gemm(c, 1.0, a, b, 0.0);

  EXPECT_DOUBLE_EQ((c[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 154.0);
}

TEST(BlasGemmTest, HandlesRequestedTranspose)
{
  std::vector<double> a_storage(6);
  std::vector<double> b_storage(6);
  std::vector<double> c_storage(4);

  stdex::mdspan<double, extents_2d, stdex::layout_left> a(a_storage.data(), 3, 2);
  stdex::mdspan<double, extents_2d, stdex::layout_left> b(b_storage.data(), 3, 2);
  stdex::mdspan<double, extents_2d, stdex::layout_left> c(c_storage.data(), 2, 2);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(b, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  EXPECT_TRUE(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0, MatrixTransform::transpose));

  EXPECT_DOUBLE_EQ((c[0, 0]), 89.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 98.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 116.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 128.0);
}

TEST(BlasGemmTest, NormalizesUnobservedSingletonProviderColumnStride)
{
  std::vector<double> a_storage(3);
  std::vector<double> b_storage(1, 4.0);
  std::vector<double> c_storage(3);

  stdex::layout_stride::mapping<extents_2d> vector_mapping(extents_2d{3, 1}, std::array<uni20::index_type, 2>{1, 0});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> a(a_storage.data(), vector_mapping);
  stdex::mdspan<double, extents_2d, stdex::layout_left> b(b_storage.data(), 1, 1);
  stdex::mdspan<double, extents_2d, stdex::layout_stride> c(c_storage.data(), vector_mapping);

  fill_matrix(a, {1.0, 2.0, 3.0});

  EXPECT_TRUE(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 4.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 8.0);
  EXPECT_DOUBLE_EQ((c[2, 0]), 12.0);
}

TEST(BlasGemmTest, TryDeclinesUnsupportedStridePattern)
{
  std::vector<double> a_storage(16);
  std::vector<double> b_storage(4, 1.0);
  std::vector<double> c_storage(4, 7.0);

  stdex::layout_stride::mapping<extents_2d> bad_mapping(extents_2d{2, 2}, std::array<uni20::index_type, 2>{2, 5});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> a(a_storage.data(), bad_mapping);
  stdex::mdspan<double, extents_2d, stdex::layout_left> b(b_storage.data(), 2, 2);
  stdex::mdspan<double, extents_2d, stdex::layout_left> c(c_storage.data(), 2, 2);

  EXPECT_FALSE(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0));
  EXPECT_DOUBLE_EQ((c[0, 0]), 7.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 7.0);
}

TEST(BlasGemmTest, CollapsesRealConjugateOnlyTransform)
{
  std::vector<double> a_storage(1, 3.0);
  std::vector<double> b_storage(1, 4.0);
  std::vector<double> c_storage(1);

  stdex::mdspan<double, extents_2d, stdex::layout_left> a(a_storage.data(), 1, 1);
  stdex::mdspan<double, extents_2d, stdex::layout_left> b(b_storage.data(), 1, 1);
  stdex::mdspan<double, extents_2d, stdex::layout_left> c(c_storage.data(), 1, 1);

  EXPECT_TRUE(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0, MatrixTransform::conjugate));
  EXPECT_DOUBLE_EQ((c[0, 0]), 12.0);
}
