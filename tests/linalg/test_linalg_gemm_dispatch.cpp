#include <uni20/common/mdspan.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/gemm.hpp>
#include <uni20/linalg/ops/gemm.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
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

template <class Scalar> using left_mdspan = stdex::mdspan<Scalar, extents_2d, stdex::layout_left>;

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

using uni20::linalg::backend_list;
using uni20::linalg::BlasBackend;
using uni20::linalg::CpuGenericBackend;
using uni20::linalg::gemm_op;
using uni20::linalg::KernelTypeAcceptance;

static_assert(uni20::linalg::kernel_type_acceptance<BlasBackend, gemm_op, left_mdspan<double>&, double&,
                                                    left_mdspan<double>&, left_mdspan<double>&, double&>() ==
              KernelTypeAcceptance::maybe);

static_assert(uni20::linalg::kernel_type_acceptance<BlasBackend, gemm_op, left_mdspan<double>&, double&,
                                                    value_transform_mdspan<double>&, left_mdspan<double>&, double&>() ==
              KernelTypeAcceptance::no);

static_assert(uni20::linalg::kernel_type_acceptance<CpuGenericBackend, gemm_op, left_mdspan<double>&, double&,
                                                    value_transform_mdspan<double>&, left_mdspan<double>&, double&>() ==
              KernelTypeAcceptance::yes);

static_assert(requires(BlasBackend backend, gemm_op op, left_mdspan<double>& output, double scalar,
                       value_transform_mdspan<double>& lhs, left_mdspan<double>& rhs) {
  { try_kernel(backend, op, output, scalar, lhs, rhs, scalar) } -> std::same_as<bool>;
});
} // namespace

TEST(LinalgGemmDispatchTest, ForcedBlasBackendRunsRepresentableMdspans)
{
  std::vector<double> a_storage(6);
  std::vector<double> b_storage(6);
  std::vector<double> c_storage(4);

  left_mdspan<double> a(a_storage.data(), 2, 3);
  left_mdspan<double> b(b_storage.data(), 3, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(b, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  EXPECT_TRUE(uni20::linalg::try_gemm(BlasBackend{}, c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 154.0);
}

TEST(LinalgGemmDispatchTest, ForcedBlasBackendDeclinesUnsupportedStrideBeforeSideEffects)
{
  std::vector<double> a_storage(8);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4, -7.0);

  stdex::layout_stride::mapping<extents_2d> bad_mapping(extents_2d{2, 2}, std::array<uni20::index_type, 2>{2, 5});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> a(a_storage.data(), bad_mapping);
  left_mdspan<double> b(b_storage.data(), 2, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {5.0, 6.0, 7.0, 8.0});

  EXPECT_FALSE(uni20::linalg::try_gemm(BlasBackend{}, c, 1.0, a, b, 0.0));
  EXPECT_DOUBLE_EQ((c[0, 0]), -7.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), -7.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), -7.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), -7.0);
}

TEST(LinalgGemmDispatchTest, BackendListFallsThroughWhenBlasDeclinesStride)
{
  std::vector<double> a_storage(8);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4, -7.0);

  stdex::layout_stride::mapping<extents_2d> bad_mapping(extents_2d{2, 2}, std::array<uni20::index_type, 2>{2, 5});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> a(a_storage.data(), bad_mapping);
  left_mdspan<double> b(b_storage.data(), 2, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {5.0, 6.0, 7.0, 8.0});

  auto selector = backend_list{BlasBackend{}, CpuGenericBackend{}};
  EXPECT_TRUE(uni20::linalg::try_gemm(selector, c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 19.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 22.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 43.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 50.0);
}

TEST(LinalgGemmDispatchTest, BackendListFallsThroughForAccessorOnlyReadableInput)
{
  std::vector<double> a_storage(4);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4);

  left_mdspan<double> a_raw(a_storage.data(), 2, 2);
  value_transform_mdspan<double> a(a_storage.data(), 2, 2);
  left_mdspan<double> b(b_storage.data(), 2, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  fill_matrix(a_raw, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {1.0, 0.0, 0.0, 1.0});

  auto selector = backend_list{BlasBackend{}, CpuGenericBackend{}};
  EXPECT_TRUE(uni20::linalg::try_gemm(selector, c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 4.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 6.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 8.0);
}

TEST(LinalgGemmDispatchTest, CpuFallbackDoesNotReadOutputWhenBetaIsZero)
{
  std::vector<double> a_storage(1, 3.0);
  std::vector<double> b_storage(1, 4.0);
  std::vector<double> c_storage(1, std::numeric_limits<double>::quiet_NaN());

  left_mdspan<double> a(a_storage.data(), 1, 1);
  left_mdspan<double> b(b_storage.data(), 1, 1);
  left_mdspan<double> c(c_storage.data(), 1, 1);

  EXPECT_TRUE(uni20::linalg::try_gemm(CpuGenericBackend{}, c, 1.0, a, b, 0.0));
  EXPECT_DOUBLE_EQ((c[0, 0]), 12.0);
}
