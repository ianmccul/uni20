#include <uni20/common/trace.hpp>
#include <uni20/tensor/layout.hpp>
#include <uni20/tensor/tensor.hpp>

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
using tensor_type = Tensor<int, 2, VectorStorage>;
using strided_tensor_type = StridedTensor<int, 2, VectorStorage>;

static_assert(std::same_as<tensor_type, BasicTensor<int, extents_2d, VectorStorage, ColumnMajor>>);
static_assert(std::same_as<tensor_type, ColumnMajorTensor<int, 2>>);
static_assert(std::same_as<RowMajorTensor<int, 2>, Tensor<int, 2, VectorStorage, RowMajor>>);
static_assert(std::same_as<strided_tensor_type, Tensor<int, 2, VectorStorage, stdex::layout_stride>>);
static_assert(!std::constructible_from<tensor_type, extents_2d const&, std::array<index_t, 2> const&>);
static_assert(std::constructible_from<strided_tensor_type, extents_2d const&, std::array<index_t, 2> const&>);

static_assert(TensorView<tensor_type>);
static_assert(OwningTensor<tensor_type>);
static_assert(OwningTensor<tensor_type const>);
static_assert(MutableTensorView<tensor_type>);
static_assert(RankedTensorView<tensor_type, 2>);
static_assert(MutableRankedTensorView<tensor_type, 2>);
static_assert(StridedTensorView<tensor_type>);
static_assert(MutableStridedTensorView<tensor_type>);
static_assert(RankedStridedTensorView<tensor_type, 2>);
static_assert(MutableRankedStridedTensorView<tensor_type, 2>);
static_assert(!RankedTensorView<tensor_type, 1>);
static_assert(!MutableRankedTensorView<tensor_type, 1>);
static_assert(!MutableTensorView<tensor_type const>);
static_assert(!SpanLike<tensor_type>);
static_assert(!StridedMdspan<tensor_type>);

using row_major_matrix = DenseMatrix<int, RowMajor>;
using strided_matrix = typename row_major_matrix::template rebind_layout_type<stdex::layout_stride>;
static_assert(std::same_as<strided_matrix, StridedTensor<int, 2>>);

template <typename Span>
constexpr bool can_assign_element_v =
    std::is_assignable_v<typename std::remove_reference_t<Span>::reference,
                         std::remove_const_t<typename std::remove_reference_t<Span>::value_type>>;

TEST(TensorTest, DefaultMappingUsesColumnMajorVectorStorage)
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
  std::vector<int> expected{0, 3, 1, 4, 2, 5};
  ASSERT_EQ(storage.size(), expected.size());
  for (std::size_t idx = 0; idx < expected.size(); ++idx)
  {
    EXPECT_EQ(storage[idx], expected[idx]);
  }

  EXPECT_EQ((tensor[1, 2]), expected.back());
  EXPECT_EQ(tensor.mapping().stride(0), 1);
  EXPECT_EQ(tensor.mapping().stride(1), 2);
}

TEST(TensorTest, DynamicExtentsConstructorAcceptsOneExtentPerAxis)
{
  tensor_type tensor(2, 3);

  EXPECT_EQ(tensor.rows(), 2);
  EXPECT_EQ(tensor.cols(), 3);
  EXPECT_EQ(tensor.size(), 6);
}

TEST(BasicTensorTest, SupportsStaticMdspanExtents)
{
  using fixed_extents = stdex::extents<index_t, 2, 3>;
  BasicTensor<int, fixed_extents> tensor(fixed_extents{});

  static_assert(decltype(tensor)::rank() == 2);
  static_assert(decltype(tensor)::rank_dynamic() == 0);
  EXPECT_EQ(tensor.rows(), 2);
  EXPECT_EQ(tensor.cols(), 3);
  EXPECT_EQ(tensor.size(), 6);
}

TEST(TensorTest, DenseMatrixDefaultsToColumnMajor)
{
  DenseMatrix<int> matrix(2, 3);

  static_assert(std::same_as<typename DenseMatrix<int>::layout_type, ColumnMajor>);
  EXPECT_EQ(matrix.rows(), 2);
  EXPECT_EQ(matrix.cols(), 3);
  EXPECT_EQ(matrix.mapping().stride(0), 1);
  EXPECT_EQ(matrix.mapping().stride(1), 2);
}

TEST(TensorTest, DenseMatrixSupportsRowMajorOwnership)
{
  DenseMatrix<int, RowMajor> matrix(2, 3);

  EXPECT_EQ(matrix.mapping().stride(0), 3);
  EXPECT_EQ(matrix.mapping().stride(1), 1);
}

TEST(TensorTest, CustomStridesAllocateFullSpan)
{
  extents_2d exts{2, 2};
  std::array<index_t, 2> strides{3, 1};
  strided_tensor_type tensor(exts, strides);

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

TEST(TensorTest, AdoptedStorageMayRetainPaddingAndUnusedTail)
{
  extents_2d exts{2, 2};
  strided_tensor_type::mapping_type mapping(exts, std::array<index_t, 2>{1, 3});
  strided_tensor_type::storage_type storage(8, -1);
  int* original_storage = storage.data();
  storage[0] = 10;
  storage[1] = 11;
  storage[3] = 12;
  storage[4] = 13;

  auto tensor = strided_tensor_type::adopt_storage(mapping, std::move(storage));

  EXPECT_EQ(tensor.mutable_handle(), original_storage);
  EXPECT_EQ(tensor.storage().size(), 8u);
  EXPECT_EQ(tensor.size(), 5);
  EXPECT_EQ((tensor[0, 0]), 10);
  EXPECT_EQ((tensor[1, 0]), 11);
  EXPECT_EQ((tensor[0, 1]), 12);
  EXPECT_EQ((tensor[1, 1]), 13);

  auto released = std::move(tensor).release_storage();
  EXPECT_EQ(released.data(), original_storage);
  EXPECT_EQ(released.size(), 8u);
}

TEST(TensorTest, MappingBuilderSupportsLayoutLeft)
{
  extents_2d exts{2, 3};
  strided_tensor_type tensor(exts, layout::LayoutLeft());

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

TEST(TensorTest, CopyConstructionOwnsIndependentStorage)
{
  tensor_type source(extents_2d{2, 3});
  source[0, 0] = 1;
  source[1, 2] = 6;

  tensor_type copy(source);

  EXPECT_EQ(copy.handle(), static_cast<int const*>(copy.storage().data()));
  EXPECT_NE(copy.handle(), source.handle());
  EXPECT_EQ(copy.extents().extent(0), 2);
  EXPECT_EQ(copy.extents().extent(1), 3);
  EXPECT_EQ((copy[0, 0]), 1);
  EXPECT_EQ((copy[1, 2]), 6);

  copy[0, 0] = 99;
  EXPECT_EQ((source[0, 0]), 1);
  EXPECT_EQ((copy[0, 0]), 99);
}

TEST(TensorTest, CopyAssignmentOwnsIndependentStorageAndMapping)
{
  strided_tensor_type source(extents_2d{2, 3}, std::array<index_t, 2>{4, 1});
  source[0, 0] = 3;
  source[1, 2] = 9;

  strided_tensor_type target(extents_2d{1, 1});
  target = source;

  EXPECT_EQ(target.handle(), static_cast<int const*>(target.storage().data()));
  EXPECT_NE(target.handle(), source.handle());
  EXPECT_EQ(target.mapping().stride(0), 4);
  EXPECT_EQ(target.mapping().stride(1), 1);
  EXPECT_EQ(target.storage().size(), source.storage().size());
  EXPECT_EQ((target[0, 0]), 3);
  EXPECT_EQ((target[1, 2]), 9);

  target[1, 2] = 42;
  EXPECT_EQ((source[1, 2]), 9);
  EXPECT_EQ((target[1, 2]), 42);
}

TEST(TensorTest, MoveConstructionRetainsOwnedMapping)
{
  tensor_type source(extents_2d{2, 3});
  source[0, 1] = 7;
  source[1, 2] = 8;

  tensor_type moved(std::move(source));

  EXPECT_EQ(moved.handle(), static_cast<int const*>(moved.storage().data()));
  EXPECT_EQ(source.handle(), static_cast<int const*>(source.storage().data()));
  EXPECT_EQ((moved[0, 1]), 7);
  EXPECT_EQ((moved[1, 2]), 8);
}

TEST(TensorTest, MoveAssignmentRetainsOwnedMapping)
{
  tensor_type source(extents_2d{2, 3});
  source[0, 1] = 11;
  source[1, 2] = 12;

  tensor_type target(extents_2d{1, 1});
  target = std::move(source);

  EXPECT_EQ(target.handle(), static_cast<int const*>(target.storage().data()));
  EXPECT_EQ(source.handle(), static_cast<int const*>(source.storage().data()));
  EXPECT_EQ(target.extents().extent(0), 2);
  EXPECT_EQ(target.extents().extent(1), 3);
  EXPECT_EQ((target[0, 1]), 11);
  EXPECT_EQ((target[1, 2]), 12);
}

TEST(TensorTest, MdspanFromConstTensorIsReadOnly)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts);

  auto mutable_span = tensor.mdspan();
  static_assert(std::is_same_v<typename decltype(mutable_span)::reference, int&>);
  mutable_span[0, 0] = 5;
  mutable_span[1, 2] = 17;

  tensor_type const& const_tensor = tensor;
  using const_span_type = decltype(const_tensor.mdspan());
  static_assert(std::is_same_v<typename const_span_type::reference, int const&>);
  static_assert(!can_assign_element_v<const_span_type const&>);

  auto span_from_mdspan = tensor.mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mdspan)::reference, int&>);
  static_assert(can_assign_element_v<decltype(span_from_mdspan) const&>);

  EXPECT_EQ((span_from_mdspan[0, 0]), 5);
  EXPECT_EQ((span_from_mdspan[1, 2]), 17);
}

TEST(TensorTest, ResolvedMdspansShareOwnedStorage)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts);

  auto span = tensor.mdspan();
  span[0, 0] = 9;
  span[1, 2] = 42;

  EXPECT_EQ(tensor.storage()[0], 9);
  EXPECT_EQ(tensor.storage()[5], 42);

  tensor_type const& const_tensor = tensor;
  auto const_span = const_tensor.mdspan();
  static_assert(!can_assign_element_v<decltype(const_span) const&>);

  EXPECT_EQ(span.data_handle(), tensor.mutable_handle());
  EXPECT_EQ(const_span.data_handle(), tensor.handle());
  EXPECT_EQ((const_span[0, 0]), 9);
  EXPECT_EQ((const_span[1, 2]), 42);
  EXPECT_EQ(tensor.backend_selector(), VectorStorage::backend_selector());
}

TEST(TensorTest, TraceFormattingUsesPresentationTensorArt)
{
  tensor_type tensor(extents_2d{2, 2});
  tensor[0, 0] = 1;
  tensor[0, 1] = 20;
  tensor[1, 0] = 300;
  tensor[1, 1] = 4;

  auto opts = trace::get_formatting_options("tensor-format-test");
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);

  EXPECT_EQ(trace::formatValue(tensor, opts), "shape=(2, 2)\n"
                                              "\xE2\x8E\xA1   1 20 \xE2\x8E\xA4\n"
                                              "\xE2\x8E\xA3 300  4 \xE2\x8E\xA6");
}

} // namespace
