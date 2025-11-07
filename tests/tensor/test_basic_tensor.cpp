#include "tensor/basic_tensor.hpp"
#include "tensor/layout.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace uni20;

namespace
{

using index_t = index_type;
using extents_2d = stdex::dextents<index_t, 2>;
using tensor_type = BasicTensor<int, extents_2d, VectorStorage>;

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

  EXPECT_EQ(tensor[1, 2], expected.back());
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
  EXPECT_EQ(tensor[1, 1], 13);
}

TEST(BasicTensorTest, MappingBuilderSupportsLayoutLeft)
{
  extents_2d exts{2, 3};
  layout::LayoutLeft builder{};
  tensor_type tensor(exts, builder);

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

  EXPECT_EQ(tensor[1, 2], 21);
}

} // namespace
