#include <uni20/common/mdspan.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/gemm.hpp>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <utility>
#include <vector>

namespace
{
using uni20::linalg::KernelAttempt;

using extents_2d = stdex::dextents<uni20::index_type, 2>;

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

template <class Output, class Scalar, class Lhs, class Rhs>
concept can_call_try_gemm = requires(Output&& output, Lhs&& lhs, Rhs&& rhs) {
  uni20::linalg::blas::try_gemm(std::forward<Output>(output), Scalar{}, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs),
                                Scalar{});
};
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

  EXPECT_EQ(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0), KernelAttempt::success);

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

TEST(BlasGemmTest, HandlesTransposedMdspanView)
{
  std::vector<double> a_storage(6);
  std::vector<double> b_storage(6);
  std::vector<double> c_storage(4);

  stdex::mdspan<double, extents_2d, stdex::layout_left> a_base(a_storage.data(), 3, 2);
  stdex::layout_stride::mapping<extents_2d> a_transposed_mapping(extents_2d{2, 3},
                                                                 std::array<uni20::index_type, 2>{3, 1});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> a(a_storage.data(), a_transposed_mapping);
  stdex::mdspan<double, extents_2d, stdex::layout_left> b(b_storage.data(), 3, 2);
  stdex::mdspan<double, extents_2d, stdex::layout_left> c(c_storage.data(), 2, 2);

  fill_matrix(a_base, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(b, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  EXPECT_EQ(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0), KernelAttempt::success);

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

  EXPECT_EQ(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0), KernelAttempt::success);

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

  EXPECT_EQ(uni20::linalg::blas::try_gemm(c, 1.0, a, b, 0.0), KernelAttempt::unsupported_layout);
  EXPECT_DOUBLE_EQ((c[0, 0]), 7.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 7.0);
}

TEST(BlasGemmTest, AcceptsRealConjIdentityView)
{
  std::vector<double> a_storage(1, 3.0);
  std::vector<double> b_storage(1, 4.0);
  std::vector<double> c_storage(1);

  stdex::mdspan<double, extents_2d, stdex::layout_left> a(a_storage.data(), 1, 1);
  stdex::mdspan<double, extents_2d, stdex::layout_left> b(b_storage.data(), 1, 1);
  stdex::mdspan<double, extents_2d, stdex::layout_left> c(c_storage.data(), 1, 1);

  EXPECT_EQ(uni20::linalg::blas::try_gemm(c, 1.0, uni20::conj(a), b, 0.0), KernelAttempt::success);
  EXPECT_DOUBLE_EQ((c[0, 0]), 12.0);
}

TEST(BlasGemmTest, ConjugatingMdspanIsReadableOnly)
{
  using Complex = uni20::complex<double>;
  using Span = stdex::mdspan<Complex, extents_2d, stdex::layout_left>;
  using ConjugatingSpan = decltype(uni20::conj(std::declval<Span&>()));

  static_assert(can_call_try_gemm<Span&, Complex, ConjugatingSpan, Span&>);
  static_assert(!can_call_try_gemm<ConjugatingSpan, Complex, Span&, Span&>);
  static_assert(!can_call_try_gemm<Span&, Complex, value_transform_mdspan<Complex>&, Span&>);
}

TEST(BlasGemmTest, TryDeclinesConjugateOnlyComplexInput)
{
  using Complex = uni20::complex<double>;

  std::vector<Complex> a_storage(1, Complex{2.0, 1.0});
  std::vector<Complex> b_storage(1, Complex{3.0, 1.0});
  std::vector<Complex> c_storage(1, Complex{-5.0, 2.0});

  stdex::mdspan<Complex, extents_2d, stdex::layout_left> a(a_storage.data(), 1, 1);
  stdex::mdspan<Complex, extents_2d, stdex::layout_left> b(b_storage.data(), 1, 1);
  stdex::mdspan<Complex, extents_2d, stdex::layout_left> c(c_storage.data(), 1, 1);

  EXPECT_EQ(uni20::linalg::blas::try_gemm(c, Complex{1.0, 0.0}, uni20::conj(a), b, Complex{0.0, 0.0}),
            KernelAttempt::unsupported_transform);
  EXPECT_DOUBLE_EQ((c[0, 0]).real(), -5.0);
  EXPECT_DOUBLE_EQ((c[0, 0]).imag(), 2.0);
}

TEST(BlasGemmTest, UsesConjugateTransposedMdspanInput)
{
  using Complex = uni20::complex<double>;

  std::vector<Complex> a_storage{Complex{2.0, 1.0}, Complex{3.0, 2.0}};
  std::vector<Complex> b_storage{Complex{4.0, 0.0}, Complex{5.0, 0.0}};
  std::vector<Complex> c_storage(1);

  stdex::layout_stride::mapping<extents_2d> a_transposed_mapping(extents_2d{1, 2},
                                                                 std::array<uni20::index_type, 2>{2, 1});
  stdex::mdspan<Complex, extents_2d, stdex::layout_stride> a_transposed(a_storage.data(), a_transposed_mapping);
  stdex::mdspan<Complex, extents_2d, stdex::layout_left> b(b_storage.data(), 2, 1);
  stdex::mdspan<Complex, extents_2d, stdex::layout_left> c(c_storage.data(), 1, 1);

  EXPECT_EQ(uni20::linalg::blas::try_gemm(c, Complex{1.0, 0.0}, uni20::conj(a_transposed), b, Complex{0.0, 0.0}),
            KernelAttempt::success);
  EXPECT_DOUBLE_EQ((c[0, 0]).real(), 23.0);
  EXPECT_DOUBLE_EQ((c[0, 0]).imag(), -14.0);
}
