#include <uni20/common/mdspan.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/mdspan_vector.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using extents_1d = stdex::dextents<uni20::index_type, 1>;

template <class Scalar> using left_mdspan = stdex::mdspan<Scalar, extents_1d, stdex::layout_left>;

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
using value_transform_mdspan = stdex::mdspan<Scalar, extents_1d, stdex::layout_left, ValueTransformAccessor<Scalar>>;

template <class Mdspan>
concept can_try_blas_readable_vector =
    requires(Mdspan const& span) { uni20::linalg::blas::try_blas_readable_vector(span); };

template <class Mdspan>
concept can_try_blas_writable_vector =
    requires(Mdspan const& span) { uni20::linalg::blas::try_blas_writable_vector(span); };

template <class Mdspan>
concept can_try_lapack_writable_vector =
    requires(Mdspan const& span) { uni20::linalg::blas::try_lapack_writable_vector(span); };
} // namespace

TEST(BlasVectorOperandTest, ConstrainsDirectAccessorAndScalarTypes)
{
  static_assert(can_try_blas_readable_vector<left_mdspan<double>>);
  static_assert(can_try_blas_writable_vector<left_mdspan<double>>);
  static_assert(can_try_blas_readable_vector<left_mdspan<double const>>);
  static_assert(!can_try_blas_writable_vector<left_mdspan<double const>>);
  static_assert(!can_try_blas_readable_vector<left_mdspan<int>>);
  static_assert(!can_try_blas_readable_vector<value_transform_mdspan<double>>);
  static_assert(!can_try_blas_writable_vector<value_transform_mdspan<double>>);
  static_assert(can_try_lapack_writable_vector<left_mdspan<double>>);
  static_assert(!can_try_lapack_writable_vector<left_mdspan<double const>>);
  static_assert(!can_try_lapack_writable_vector<left_mdspan<int>>);
  static_assert(!can_try_lapack_writable_vector<value_transform_mdspan<double>>);
}

TEST(BlasVectorOperandTest, StagesPositiveStridedVector)
{
  std::vector<double> storage(5);
  stdex::layout_stride::mapping<extents_1d> mapping(extents_1d{3}, std::array<uni20::index_type, 1>{2});
  stdex::mdspan<double, extents_1d, stdex::layout_stride> span(storage.data(), mapping);

  auto stage = uni20::linalg::blas::try_mdspan_vector_stage(span);
  ASSERT_TRUE(stage.has_value());
  EXPECT_EQ(stage->extent, 3);
  EXPECT_EQ(stage->increment, 2);
  EXPECT_FALSE(stage->needs_conjugation);

  auto writable = uni20::linalg::blas::blas_writable_vector(*stage);
  EXPECT_EQ(writable.data, storage.data());
  EXPECT_EQ(writable.size, 3);
  EXPECT_EQ(writable.increment, 2);
}

TEST(BlasVectorOperandTest, NormalizesUnobservedSingletonStride)
{
  std::vector<double> storage(1);
  stdex::layout_stride::mapping<extents_1d> mapping(extents_1d{1}, std::array<uni20::index_type, 1>{0});
  stdex::mdspan<double, extents_1d, stdex::layout_stride> span(storage.data(), mapping);

  auto stage = uni20::linalg::blas::try_mdspan_vector_stage(span);
  ASSERT_TRUE(stage.has_value());
  EXPECT_EQ(stage->extent, 1);
  EXPECT_EQ(stage->increment, 1);
}

TEST(BlasVectorOperandTest, LapackRequiresContiguousWritableVector)
{
  std::vector<double> contiguous_storage(3);
  left_mdspan<double> contiguous(contiguous_storage.data(), 3);
  auto operand = uni20::linalg::blas::try_lapack_writable_vector(contiguous);
  ASSERT_TRUE(operand.has_value());
  EXPECT_EQ(operand->size, 3);
  EXPECT_EQ(operand->increment, 1);

  std::vector<double> strided_storage(5);
  stdex::layout_stride::mapping<extents_1d> mapping(extents_1d{3}, std::array<uni20::index_type, 1>{2});
  stdex::mdspan<double, extents_1d, stdex::layout_stride> strided(strided_storage.data(), mapping);
  EXPECT_FALSE(uni20::linalg::blas::try_lapack_writable_vector(strided).has_value());
}

TEST(BlasVectorOperandTest, ConjugatingViewRequiresOperationSpecificLowering)
{
  using Scalar = uni20::complex<double>;
  std::vector<Scalar> storage(3);
  left_mdspan<Scalar> span(storage.data(), 3);
  auto conjugated = uni20::conj(span);

  auto stage = uni20::linalg::blas::try_mdspan_vector_stage(conjugated);
  ASSERT_TRUE(stage.has_value());
  EXPECT_TRUE(stage->needs_conjugation);
  EXPECT_FALSE(uni20::linalg::blas::try_blas_readable_vector(conjugated).has_value());
  EXPECT_DEATH((void)uni20::linalg::blas::blas_readable_vector(*stage), "cannot discard accessor conjugation");
}

TEST(BlasVectorOperandTest, DeclinesValuesOutsideBlasIntegerRange)
{
  if constexpr (std::numeric_limits<uni20::index_type>::max() > std::numeric_limits<uni20::blas_int>::max())
  {
    std::vector<double> storage(1);
    auto const too_large =
        static_cast<uni20::index_type>(std::numeric_limits<uni20::blas_int>::max()) + uni20::index_type{1};
    stdex::layout_stride::mapping<extents_1d> mapping(extents_1d{too_large}, std::array<uni20::index_type, 1>{1});
    stdex::mdspan<double, extents_1d, stdex::layout_stride> span(storage.data(), mapping);

    EXPECT_FALSE(uni20::linalg::blas::try_mdspan_vector_stage(span).has_value());
  }
  else
  {
    GTEST_SKIP() << "index_type cannot represent values outside the configured BLAS integer range";
  }
}
