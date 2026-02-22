#include "tensor/tensor_view.hpp"

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

template <typename T> constexpr bool has_mutable_mdspan_v = requires(T&& t) { std::forward<T>(t).mutable_mdspan(); };

template <typename Span>
constexpr bool can_assign_element_v =
    std::is_assignable_v<typename std::remove_reference_t<Span>::reference,
                         std::remove_const_t<typename std::remove_reference_t<Span>::value_type>>;

TEST(TensorViewTest, ConstructFromConstPointer)
{
  int const data[] = {1, 2, 3, 4, 5, 6};
  TensorView<int const, const_traits_type> view(data, extents_2d{2, 3});

  EXPECT_EQ(view.handle(), data);
  EXPECT_EQ(view.extents().extent(0), 2);
  EXPECT_EQ(view.extents().extent(1), 3);
  EXPECT_EQ(view.size(), 6);
  EXPECT_EQ((view[0, 1]), 2);
}

TEST(TensorViewTest, MutableViewProvidesSeparateMutableHandle)
{
  std::array<int, 6> storage{0, 1, 2, 3, 4, 5};
  auto* ptr = storage.data();

  TensorView<int const, const_traits_type> const_view(ptr, extents_2d{2, 3});
  EXPECT_EQ(const_view.handle(), static_cast<int const*>(ptr));
  EXPECT_EQ((const_view[1, 2]), 5);

  TensorView<int, mutable_traits_type> view(ptr, extents_2d{2, 3});
  EXPECT_EQ(view.handle(), static_cast<int const*>(ptr));
  EXPECT_EQ(view.mutable_handle(), ptr);

  view[1, 2] = 42;
  EXPECT_EQ(storage[5], 42);
  EXPECT_EQ((const_view[1, 2]), 42);

  static_assert(!has_mutable_handle_v<TensorView<int const, const_traits_type> const&>);
  static_assert(has_mutable_handle_v<TensorView<int, mutable_traits_type>&>);
  static_assert(!has_mutable_handle_v<TensorView<int, mutable_traits_type> const&>);
}

TEST(TensorViewTest, MdspanFromConstViewIsReadOnly)
{
  std::array<int, 6> storage{0, 1, 2, 3, 4, 5};
  TensorView<int, mutable_traits_type> view(storage.data(), extents_2d{2, 3});

  auto span_from_mutable = view.mutable_mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mutable)::reference, int&>);
  span_from_mutable[1, 2] = 42;
  EXPECT_EQ(storage[5], 42);

  TensorView<int, mutable_traits_type> const& const_ref = view;
  using const_span_type = decltype(const_ref.mdspan());
  static_assert(std::is_same_v<typename const_span_type::reference, int const&>);
  static_assert(!can_assign_element_v<const_span_type const&>);

  auto span_from_mdspan = view.mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mdspan)::reference, int const&>);
  static_assert(!can_assign_element_v<decltype(span_from_mdspan) const&>);

  static_assert(!has_mutable_mdspan_v<TensorView<int, mutable_traits_type> const&>);
}

TEST(TensorViewTest, RankTwoTensorProvidesMatrixDimensions)
{
  std::array<int, 6> storage{0, 1, 2, 3, 4, 5};
  TensorView<int const, const_traits_type> const_view(storage.data(), extents_2d{2, 3});
  EXPECT_EQ(const_view.rows(), 2);
  EXPECT_EQ(const_view.cols(), 3);

  TensorView<int, mutable_traits_type> mutable_view(storage.data(), extents_2d{2, 3});
  EXPECT_EQ(mutable_view.rows(), 2);
  EXPECT_EQ(mutable_view.cols(), 3);

  TensorView<int, mutable_traits_type> const& const_ref = mutable_view;
  EXPECT_EQ(const_ref.rows(), 2);
  EXPECT_EQ(const_ref.cols(), 3);
}

TEST(TensorViewTest, ExposesDefaultBackendTag)
{
  using const_view_type = TensorView<int const, const_traits_type>;
  using mutable_view_type = TensorView<int, mutable_traits_type>;

  static_assert(std::is_same_v<typename const_view_type::default_tag, VectorStorage::default_tag>);
  static_assert(std::is_same_v<typename mutable_view_type::default_tag, VectorStorage::default_tag>);

  EXPECT_TRUE((std::is_same_v<typename const_view_type::default_tag, VectorStorage::default_tag>));
  EXPECT_TRUE((std::is_same_v<typename mutable_view_type::default_tag, VectorStorage::default_tag>));
}

} // namespace
