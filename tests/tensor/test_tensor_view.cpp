#include <uni20/level1/assign.hpp>
#include <uni20/level1/zip_transform.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/tensor_view.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <type_traits>
#include <utility>

using namespace uni20;

namespace
{

using index_t = index_type;
using extents_2d = stdex::dextents<index_t, 2>;
using const_traits_type = tensor_traits<extents_2d, VectorStorage>;
using mutable_traits_type = mutable_tensor_traits<extents_2d, VectorStorage>;

template <typename T> constexpr bool has_mutable_handle_v = requires(T&& t) { std::forward<T>(t).mutable_handle(); };

template <typename Span>
constexpr bool can_assign_element_v =
    std::is_assignable_v<typename std::remove_reference_t<Span>::reference,
                         std::remove_const_t<typename std::remove_reference_t<Span>::value_type>>;

TEST(BasicTensorViewTest, ConstructFromConstPointer)
{
  int const data[] = {1, 2, 3, 4, 5, 6};
  BasicTensorView<int const, const_traits_type> view(data, extents_2d{2, 3});

  EXPECT_EQ(view.handle(), data);
  EXPECT_EQ(view.extents().extent(0), 2);
  EXPECT_EQ(view.extents().extent(1), 3);
  EXPECT_EQ(view.size(), 6);
  EXPECT_EQ((view[0, 1]), 2);
}

TEST(BasicTensorViewTest, MutableViewProvidesSeparateMutableHandle)
{
  std::array<int, 6> storage{0, 1, 2, 3, 4, 5};
  auto* ptr = storage.data();

  BasicTensorView<int const, const_traits_type> const_view(ptr, extents_2d{2, 3});
  EXPECT_EQ(const_view.handle(), static_cast<int const*>(ptr));
  EXPECT_EQ((const_view[1, 2]), 5);

  BasicTensorView<int, mutable_traits_type> view(ptr, extents_2d{2, 3});
  EXPECT_EQ(view.handle(), static_cast<int const*>(ptr));
  EXPECT_EQ(view.mutable_handle(), ptr);

  view[1, 2] = 42;
  EXPECT_EQ(storage[5], 42);
  EXPECT_EQ((const_view[1, 2]), 42);

  static_assert(!has_mutable_handle_v<BasicTensorView<int const, const_traits_type> const&>);
  static_assert(has_mutable_handle_v<BasicTensorView<int, mutable_traits_type>&>);
  static_assert(!has_mutable_handle_v<BasicTensorView<int, mutable_traits_type> const&>);
}

TEST(BasicTensorViewTest, MdspanMutabilityFollowsViewElementType)
{
  std::array<int, 6> storage{0, 1, 2, 3, 4, 5};
  BasicTensorView<int, mutable_traits_type> view(storage.data(), extents_2d{2, 3});

  auto span_from_mutable = view.mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mutable)::reference, int&>);
  span_from_mutable[1, 2] = 42;
  EXPECT_EQ(storage[5], 42);

  BasicTensorView<int, mutable_traits_type> const& const_ref = view;
  using const_span_type = decltype(const_ref.mdspan());
  static_assert(std::is_same_v<typename const_span_type::reference, int&>);
  static_assert(can_assign_element_v<const_span_type const&>);

  auto span_from_mdspan = view.mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mdspan)::reference, int&>);
  static_assert(can_assign_element_v<decltype(span_from_mdspan) const&>);
  static_assert(MutableTensorView<BasicTensorView<int, mutable_traits_type> const>);
}

TEST(BasicTensorViewTest, RankTwoTensorProvidesMatrixDimensions)
{
  std::array<int, 6> storage{0, 1, 2, 3, 4, 5};
  BasicTensorView<int const, const_traits_type> const_view(storage.data(), extents_2d{2, 3});
  EXPECT_EQ(const_view.rows(), 2);
  EXPECT_EQ(const_view.cols(), 3);

  BasicTensorView<int, mutable_traits_type> mutable_view(storage.data(), extents_2d{2, 3});
  EXPECT_EQ(mutable_view.rows(), 2);
  EXPECT_EQ(mutable_view.cols(), 3);

  BasicTensorView<int, mutable_traits_type> const& const_ref = mutable_view;
  EXPECT_EQ(const_ref.rows(), 2);
  EXPECT_EQ(const_ref.cols(), 3);
}

TEST(BasicTensorViewTest, ResolvesMdspansForLevel1Kernels)
{
  using const_view_type = BasicTensorView<int const, const_traits_type>;
  using mutable_view_type = BasicTensorView<int, mutable_traits_type>;

  static_assert(!SpanLike<const_view_type>);
  static_assert(!StridedMdspan<const_view_type>);
  static_assert(!SpanLike<mutable_view_type>);
  static_assert(!MutableSpanLike<mutable_view_type>);
  static_assert(!StridedMdspan<mutable_view_type>);
  static_assert(!MutableStridedMdspan<mutable_view_type>);
  static_assert(TensorView<const_view_type>);
  static_assert(RankedTensorView<const_view_type, 2>);
  static_assert(StridedTensorView<const_view_type>);
  static_assert(RankedStridedTensorView<const_view_type, 2>);
  static_assert(!RankedTensorView<const_view_type, 1>);
  static_assert(!MutableTensorView<const_view_type>);
  static_assert(!MutableRankedTensorView<const_view_type, 2>);
  static_assert(TensorView<mutable_view_type>);
  static_assert(RankedTensorView<mutable_view_type, 2>);
  static_assert(StridedTensorView<mutable_view_type>);
  static_assert(MutableTensorView<mutable_view_type>);
  static_assert(MutableRankedTensorView<mutable_view_type, 2>);
  static_assert(MutableStridedTensorView<mutable_view_type>);
  static_assert(MutableRankedStridedTensorView<mutable_view_type, 2>);
  static_assert(!MutableRankedTensorView<mutable_view_type, 1>);
  static_assert(std::is_same_v<typename mutable_view_type::layout_type, stdex::layout_stride>);

  std::array<int, 6> src_storage{1, 2, 3, 4, 5, 6};
  std::array<int, 6> dst_storage{};

  const_view_type src(src_storage.data(), extents_2d{2, 3});
  mutable_view_type dst(dst_storage.data(), extents_2d{2, 3});

  EXPECT_EQ(src.extent(0), 2);
  EXPECT_EQ(src.extent(1), 3);
  EXPECT_TRUE(src.is_strided());

  auto src_span = src.mdspan();
  auto dst_span = dst.mdspan();
  static_assert(StridedMdspan<decltype(src_span)>);
  static_assert(MutableStridedMdspan<decltype(dst_span)>);

  assign(dst_span, src_span);
  EXPECT_EQ(dst_storage, src_storage);

  auto shifted = zip_transform([](int const& value) { return value + 10; }, src_span);
  EXPECT_EQ((shifted[0, 0]), 11);
  EXPECT_EQ((shifted[1, 2]), 16);
}

TEST(BasicTensorViewTest, ExposesDefaultBackendSelector)
{
  using const_view_type = BasicTensorView<int const, const_traits_type>;
  using mutable_view_type = BasicTensorView<int, mutable_traits_type>;

  static_assert(std::is_same_v<typename const_view_type::default_tag, VectorStorage::default_tag>);
  static_assert(std::is_same_v<typename mutable_view_type::default_tag, VectorStorage::default_tag>);
  static_assert(
      std::same_as<typename const_view_type::backend_selector_type, typename VectorStorage::backend_selector_type>);

  EXPECT_TRUE((std::is_same_v<typename const_view_type::default_tag, VectorStorage::default_tag>));
  EXPECT_TRUE((std::is_same_v<typename mutable_view_type::default_tag, VectorStorage::default_tag>));
  EXPECT_EQ(const_view_type{}.backend_selector(), VectorStorage::backend_selector());
  EXPECT_EQ(mutable_view_type{}.backend_selector(), VectorStorage::backend_selector());
}

} // namespace
