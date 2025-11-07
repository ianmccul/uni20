#include "tensor/tensor_view.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <type_traits>

using namespace uni20;

namespace
{

using index_t = index_type;
using extents_2d = stdex::dextents<index_t, 2>;
using traits_type = tensor_traits<extents_2d, VectorStorage>;

TEST(TensorViewTest, ConstructFromConstPointer)
{
  int const data[] = {1, 2, 3, 4, 5, 6};
  TensorView<int const, traits_type> view(data, extents_2d{2, 3});

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

  TensorView<int const, traits_type> const_view(ptr, extents_2d{2, 3});
  EXPECT_EQ(const_view.handle(), static_cast<int const*>(ptr));
  EXPECT_EQ((const_view[1, 2]), 5);

  TensorView<int, traits_type> view(ptr, extents_2d{2, 3});
  EXPECT_EQ(view.handle(), static_cast<int const*>(ptr));
  EXPECT_EQ(view.mutable_handle(), ptr);

  view[1, 2] = 42;
  EXPECT_EQ(storage[5], 42);
  EXPECT_EQ((const_view[1, 2]), 42);

  static_assert(!requires(TensorView<int const, traits_type> const& tv) { tv.mutable_handle(); });
  static_assert(requires(TensorView<int, traits_type> & tv) { tv.mutable_handle(); });
  static_assert(!requires(TensorView<int, traits_type> const& tv) { tv.mutable_handle(); });
}

TEST(TensorViewTest, MdspanFromConstViewIsReadOnly)
{
  std::array<int, 6> storage{0, 1, 2, 3, 4, 5};
  TensorView<int, traits_type> view(storage.data(), extents_2d{2, 3});

  auto span_from_mutable = view.mutable_mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mutable)::reference, int&>);
  span_from_mutable(1, 2) = 42;
  EXPECT_EQ(storage[5], 42);

  TensorView<int, traits_type> const& const_ref = view;
  using const_span_type = decltype(const_ref.mdspan());
  static_assert(std::is_same_v<typename const_span_type::reference, int const&>);
  static_assert(!requires(const_span_type const& span) { span(0, 0) = 7; });

  auto span_from_mdspan = view.mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mdspan)::reference, int const&>);
  static_assert(!requires(decltype(span_from_mdspan) const& span) { span(0, 0) = 9; });

  static_assert(!requires(TensorView<int, traits_type> const& tv) { tv.mutable_mdspan(); });
}

} // namespace
