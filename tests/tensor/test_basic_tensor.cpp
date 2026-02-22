#include "tensor/basic_tensor.hpp"
#include "tensor/layout.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

using namespace uni20;

namespace
{

using index_t = index_type;
using extents_2d = stdex::dextents<index_t, 2>;
using tensor_type = BasicTensor<int, extents_2d, VectorStorage>;

template <typename T> constexpr bool has_mutable_mdspan_v = requires(T&& t) { std::forward<T>(t).mutable_mdspan(); };

template <typename T> constexpr bool has_mutable_handle_v = requires(T&& t) { std::forward<T>(t).mutable_handle(); };

template <typename Span>
constexpr bool can_assign_element_v =
    std::is_assignable_v<typename std::remove_reference_t<Span>::reference,
                         std::remove_const_t<typename std::remove_reference_t<Span>::value_type>>;

TEST(BasicTensorTest, DefaultMappingUsesVectorStorage)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts);

  EXPECT_EQ(tensor.extents().extent(0), exts.extent(0));
  EXPECT_EQ(tensor.extents().extent(1), exts.extent(1));
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.mapping().required_span_size(), 6);
  EXPECT_EQ(tensor.storage().size(), 6u);

  for (index_t i = 0; i < static_cast<index_t>(exts.extent(0)); ++i)
  {
    for (index_t j = 0; j < static_cast<index_t>(exts.extent(1)); ++j)
    {
      tensor[i, j] = static_cast<int>(i * static_cast<index_t>(exts.extent(1)) + j);
    }
  }

  auto const& storage = tensor.storage();
  std::vector<int> expected{0, 1, 2, 3, 4, 5};
  ASSERT_EQ(storage.size(), expected.size());
  for (std::size_t idx = 0; idx < expected.size(); ++idx)
  {
    EXPECT_EQ(storage[idx], expected[idx]);
  }

  EXPECT_EQ((tensor[1, 2]), expected.back());
  EXPECT_EQ(tensor.mapping().stride(0), 3);
  EXPECT_EQ(tensor.mapping().stride(1), 1);
}

TEST(BasicTensorTest, CustomStridesAllocateFullSpan)
{
  extents_2d exts{2, 2};
  std::array<index_t, 2> strides{3, 1};
  tensor_type tensor(exts, strides);

  EXPECT_EQ(tensor.mapping().stride(0), strides[0]);
  EXPECT_EQ(tensor.mapping().stride(1), strides[1]);
  EXPECT_EQ(tensor.mapping().required_span_size(), 5);
  EXPECT_EQ(tensor.storage().size(), 5u);

  tensor[0, 0] = 10;
  tensor[0, 1] = 11;
  tensor[1, 0] = 12;
  tensor[1, 1] = 13;

  auto const& storage = tensor.storage();
  EXPECT_EQ(storage[0], 10);
  EXPECT_EQ(storage[1], 11);
  EXPECT_EQ(storage[3], 12);
  EXPECT_EQ(storage[4], 13);
  EXPECT_EQ((tensor[1, 1]), 13);
}

TEST(BasicTensorTest, MappingBuilderSupportsLayoutLeft)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts, layout::LayoutLeft());

  EXPECT_EQ(tensor.mapping().stride(0), 1);
  EXPECT_EQ(tensor.mapping().stride(1), 2);
  EXPECT_EQ(tensor.storage().size(), 6u);

  for (index_t j = 0; j < static_cast<index_t>(exts.extent(1)); ++j)
  {
    for (index_t i = 0; i < static_cast<index_t>(exts.extent(0)); ++i)
    {
      tensor[i, j] = static_cast<int>(j * 10 + i);
    }
  }

  auto const& storage = tensor.storage();
  std::vector<int> expected{0, 1, 10, 11, 20, 21};
  ASSERT_EQ(storage.size(), expected.size());
  for (std::size_t idx = 0; idx < expected.size(); ++idx)
  {
    EXPECT_EQ(storage[idx], expected[idx]);
  }

  EXPECT_EQ((tensor[1, 2]), 21);
}

TEST(BasicTensorTest, MdspanFromConstTensorIsReadOnly)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts);

  auto mutable_span = tensor.mutable_mdspan();
  static_assert(std::is_same_v<typename decltype(mutable_span)::reference, int&>);
  mutable_span[0, 0] = 5;
  mutable_span[1, 2] = 17;

  tensor_type const& const_tensor = tensor;
  using const_span_type = decltype(const_tensor.mdspan());
  static_assert(std::is_same_v<typename const_span_type::reference, int const&>);
  static_assert(!can_assign_element_v<const_span_type const&>);

  auto span_from_mdspan = tensor.mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mdspan)::reference, int const&>);
  static_assert(!can_assign_element_v<decltype(span_from_mdspan) const&>);

  EXPECT_EQ((span_from_mdspan[0, 0]), 5);
  EXPECT_EQ((span_from_mdspan[1, 2]), 17);

  static_assert(!has_mutable_mdspan_v<tensor_type const&>);
}

TEST(BasicTensorTest, ViewsShareStorageAndRespectConstness)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts);

  using mutable_view_type =
      TensorView<int, mutable_tensor_traits<extents_2d, VectorStorage, stdex::layout_stride, DefaultAccessorFactory>>;
  using const_view_type =
      TensorView<int const, tensor_traits<extents_2d, VectorStorage, stdex::layout_stride, DefaultAccessorFactory>>;

  auto view = tensor.view();
  static_assert(std::is_same_v<decltype(view), mutable_view_type>);
  static_assert(has_mutable_handle_v<decltype(view)&>);
  static_assert(has_mutable_mdspan_v<decltype(view)&>);

  view[0, 0] = 9;
  view[1, 2] = 42;

  EXPECT_EQ(tensor.storage()[0], 9);
  EXPECT_EQ(tensor.storage()[5], 42);

  auto cview = tensor.const_view();
  static_assert(std::is_same_v<decltype(cview), const_view_type>);
  static_assert(!can_assign_element_v<decltype(cview) const&>);
  static_assert(!has_mutable_handle_v<decltype(cview) const&>);

  EXPECT_EQ((cview[0, 0]), 9);
  EXPECT_EQ((cview[1, 2]), 42);

  tensor_type const& const_tensor = tensor;
  auto const_view_from_const = const_tensor.view();
  static_assert(std::is_same_v<decltype(const_view_from_const), const_view_type>);
  static_assert(!can_assign_element_v<decltype(const_view_from_const) const&>);
  static_assert(!has_mutable_handle_v<decltype(const_view_from_const) const&>);

  EXPECT_EQ(view.handle(), tensor.handle());
  EXPECT_EQ(cview.handle(), static_cast<int const*>(tensor.handle()));
  EXPECT_EQ(const_view_from_const.handle(), static_cast<int const*>(tensor.handle()));
}

} // namespace
