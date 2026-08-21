#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_space_traits.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

using namespace uni20;

static_assert(BlockTensorStorage<SeparateSparseBlockStorage<>>);
static_assert(BlockTensorStorage<PackedSparseBlockStorage<>>);
static_assert(SparseBlockStorage<SeparateSparseBlockStorage<>>);
static_assert(SparseBlockStorage<PackedSparseBlockStorage<>>);
static_assert(!CompleteBlockStorage<SeparateSparseBlockStorage<>>);
static_assert(!CompleteBlockStorage<PackedSparseBlockStorage<>>);
static_assert(!std::same_as<SeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>);
static_assert(BlockTensorStorageFor<SeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<SeparateSparseBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<PackedSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedSparseBlockStorage<>, double, 4, 0>);
static_assert(std::same_as<SeparateSparseBlockStorage<>::backend_selector_type, VectorStorage::backend_selector_type>);
static_assert(std::same_as<PackedSparseBlockStorage<>::backend_selector_type, VectorStorage::backend_selector_type>);
static_assert(BlockTensorSpaceTraits<LocalSpace>::has_block_coordinate);
static_assert(!BlockTensorSpaceTraits<LocalSpace>::has_dense_axis);
static_assert(!BlockTensorSpaceTraits<QNumSpace>::has_block_coordinate);
static_assert(!BlockTensorSpaceTraits<QNumSpace>::has_dense_axis);
static_assert(BlockTensorSpaceTraits<BlockSpace>::has_block_coordinate);
static_assert(BlockTensorSpaceTraits<BlockSpace>::has_dense_axis);
static_assert(BlockTensorSpaceTraits<IrregularSpace>::has_block_coordinate);
static_assert(BlockTensorSpaceTraits<IrregularSpace>::has_dense_axis);
static_assert(!BlockTensorSpaceTraits<DenseSpace>::has_block_coordinate);
static_assert(BlockTensorSpaceTraits<DenseSpace>::has_dense_axis);

template <class Storage> class SparseBlockTensorTest : public ::testing::Test {};

using SparseStorageTypes = ::testing::Types<SeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>;
TYPED_TEST_SUITE(SparseBlockTensorTest, SparseStorageTypes);

TYPED_TEST(SparseBlockTensorTest, OrderTwoBlockSpacesCanonicalizeKeysAndExposeDenseBlocks)
{
  Symmetry const sym{"N:U(1)"};
  BlockSpace const rows(sym,
                        {
                            {make_qnum(sym, {{"N", 0}}), 2},
                            {make_qnum(sym, {{"N", 1}}), 3},
                        },
                        "row");
  BlockSpace const columns(sym,
                           {
                               {make_qnum(sym, {{"N", 0}}), 4},
                               {make_qnum(sym, {{"N", 1}}), 5},
                           },
                           "column");

  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{rows}, Codomain{columns}, {Key{{1, 1}}, Key{{0, 0}}});

  static_assert(Tensor::order() == 2);
  static_assert(Tensor::key_coordinate_count() == 2);
  static_assert(Tensor::dense_block_order() == 2);
  EXPECT_EQ(tensor.stored_block_count(), 2);
  EXPECT_EQ(tensor.legal_block_count(), 2);
  EXPECT_TRUE(tensor.has_all_legal_blocks());
  EXPECT_EQ(tensor.stored_keys()[0], (Key{{0, 0}}));
  EXPECT_EQ(tensor.stored_keys()[1], (Key{{1, 1}}));
  EXPECT_TRUE(tensor.is_legal(Key{{0, 0}}));
  EXPECT_FALSE(tensor.is_legal(Key{{0, 1}}));
  EXPECT_FALSE(tensor.is_legal(Key{{2, 0}}));
  EXPECT_FALSE(tensor.contains(Key{{0, 1}}));

  auto block = tensor.block(Key{{1, 1}});
  static_assert(decltype(block)::rank() == 2);
  EXPECT_EQ(block.extent(0), 3);
  EXPECT_EQ(block.extent(1), 5);
  auto const initial_value = block[0, 0];
  EXPECT_DOUBLE_EQ(initial_value, 0.0);
  block[2, 4] = 7.5;

  Tensor const& const_tensor = tensor;
  auto const const_block = const_tensor.block(Key{{1, 1}});
  auto const stored_value = const_block[2, 4];
  EXPECT_DOUBLE_EQ(stored_value, 7.5);
  EXPECT_FALSE(const_tensor.find_block(Key{{0, 1}}).has_value());
  EXPECT_THROW(static_cast<void>(const_tensor.block(Key{{0, 1}})), std::out_of_range);

  Tensor copy = tensor;
  copy.block(Key{{1, 1}})[2, 4] = 8.5;
  auto const original_value = tensor.block(Key{{1, 1}})[2, 4];
  auto const copied_value = copy.block(Key{{1, 1}})[2, 4];
  EXPECT_DOUBLE_EQ(original_value, 7.5);
  EXPECT_DOUBLE_EQ(copied_value, 8.5);
}

TYPED_TEST(SparseBlockTensorTest, OrderTwoLocalSpacesKeepRepeatedChargeStatesDistinct)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = make_qnum(sym, {{"N", 0}});
  LocalSpace const input(sym, {q0, q0}, "input");
  LocalSpace const output(sym, {q0, q0}, "output");

  using Tensor = BlockTensor<double, Domain<LocalSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{input}, Codomain{output}, {Key{{1, 0}}, Key{{0, 0}}});

  static_assert(Tensor::order() == 2);
  static_assert(Tensor::key_coordinate_count() == 2);
  static_assert(Tensor::dense_block_order() == 0);
  EXPECT_EQ(tensor.legal_block_count(), 4);
  EXPECT_EQ(tensor.stored_block_count(), 2);
  EXPECT_TRUE(tensor.contains(Key{{0, 0}}));
  EXPECT_TRUE(tensor.contains(Key{{1, 0}}));
  EXPECT_FALSE(tensor.contains(Key{{0, 1}}));
  auto first = tensor.block(Key{{0, 0}});
  auto repeated = tensor.block(Key{{1, 0}});
  static_assert(decltype(first)::rank() == 0);
  EXPECT_DOUBLE_EQ(first[], 0.0);
  EXPECT_DOUBLE_EQ(repeated[], 0.0);

  first[] = 1.0;
  repeated[] = 2.0;
  auto const first_value = tensor.block(Key{{0, 0}})[];
  auto const repeated_value = tensor.block(Key{{1, 0}})[];
  EXPECT_DOUBLE_EQ(first_value, 1.0);
  EXPECT_DOUBLE_EQ(repeated_value, 2.0);
}

TYPED_TEST(SparseBlockTensorTest, EmptyBoundaryFactorHasNoLegalOrStoredBlocks)
{
  Symmetry const sym{"N:U(1)"};
  BlockSpace const empty(sym, "empty");
  LocalSpace const scalar(sym, {make_qnum(sym, {{"N", 0}})}, "scalar");

  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{empty}, Codomain{scalar}, {});

  EXPECT_EQ(tensor.legal_block_count(), 0);
  EXPECT_EQ(tensor.stored_block_count(), 0);
  EXPECT_TRUE(tensor.has_all_legal_blocks());
  EXPECT_FALSE(tensor.is_legal(Key{{0, 0}}));
  EXPECT_FALSE(tensor.contains(Key{{0, 0}}));
  EXPECT_FALSE(tensor.find_block(Key{{0, 0}}).has_value());
}

TYPED_TEST(SparseBlockTensorTest, OrderTwoQNumSpacesUseOneCoordinateFreeScalarBlock)
{
  Symmetry const sym{"N:U(1)"};
  QNumSpace const input(make_qnum(sym, {{"N", 1}}), "input-irrep");
  QNumSpace const output(make_qnum(sym, {{"N", 1}}), "output-irrep");

  using Tensor = BlockTensor<double, Domain<QNumSpace>, Codomain<QNumSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{input}, Codomain{output}, {Key{}});

  static_assert(Tensor::order() == 2);
  static_assert(Tensor::key_coordinate_count() == 0);
  static_assert(Tensor::dense_block_order() == 0);
  static_assert(Key::size() == 0);
  EXPECT_EQ(tensor.legal_block_count(), 1);
  EXPECT_EQ(tensor.stored_block_count(), 1);
  EXPECT_TRUE(tensor.is_legal(Key{}));

  auto block = tensor.block(Key{});
  static_assert(decltype(block)::rank() == 0);
  EXPECT_DOUBLE_EQ(block[], 0.0);
  block[] = 4.0;
  auto const stored_value = std::as_const(tensor).block(Key{})[];
  EXPECT_DOUBLE_EQ(stored_value, 4.0);
}

TYPED_TEST(SparseBlockTensorTest, QNumSpaceFixedChargeParticipatesWithoutAKeyCoordinate)
{
  Symmetry const sym{"N:U(1)"};
  LocalSpace const ket(sym,
                       {
                           make_qnum(sym, {{"N", 0}}),
                           make_qnum(sym, {{"N", 1}}),
                           make_qnum(sym, {{"N", 2}}),
                       },
                       "ket");
  QNumSpace const transform(make_qnum(sym, {{"N", 1}}), "operator-charge");
  LocalSpace const bra(sym,
                       {
                           make_qnum(sym, {{"N", 0}}),
                           make_qnum(sym, {{"N", 1}}),
                           make_qnum(sym, {{"N", 2}}),
                       },
                       "bra");

  using Tensor = BlockTensor<double, Domain<LocalSpace, QNumSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{ket, transform}, Codomain{bra}, {Key{{1, 2}}, Key{{0, 1}}});

  static_assert(Tensor::order() == 3);
  static_assert(Tensor::key_coordinate_count() == 2);
  static_assert(Tensor::dense_block_order() == 0);
  EXPECT_EQ(tensor.legal_block_count(), 2);
  EXPECT_TRUE(tensor.is_legal(Key{{0, 1}}));
  EXPECT_TRUE(tensor.is_legal(Key{{1, 2}}));
  EXPECT_FALSE(tensor.is_legal(Key{{0, 0}}));
}

TYPED_TEST(SparseBlockTensorTest, OrderThreeMpsLikeTensorMayOmitLegalBlocks)
{
  Symmetry const sym{"N:U(1)"};
  BlockSpace const left(sym,
                        {
                            {make_qnum(sym, {{"N", 0}}), 2},
                            {make_qnum(sym, {{"N", 1}}), 3},
                        },
                        "left");
  LocalSpace const physical(sym,
                            {
                                make_qnum(sym, {{"N", 0}}),
                                make_qnum(sym, {{"N", 1}}),
                            },
                            "physical");
  BlockSpace const right(sym,
                         {
                             {make_qnum(sym, {{"N", 0}}), 4},
                             {make_qnum(sym, {{"N", 1}}), 5},
                             {make_qnum(sym, {{"N", 2}}), 6},
                         },
                         "right");

  using Tensor = BlockTensor<double, Domain<BlockSpace, LocalSpace>, Codomain<BlockSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{left, physical}, Codomain{right}, {Key{{1, 1, 2}}, Key{{0, 1, 1}}});

  static_assert(Tensor::order() == 3);
  static_assert(Tensor::key_coordinate_count() == 3);
  static_assert(Tensor::dense_block_order() == 2);
  EXPECT_EQ(tensor.stored_block_count(), 2);
  EXPECT_EQ(tensor.legal_block_count(), 4);
  EXPECT_FALSE(tensor.has_all_legal_blocks());
  EXPECT_TRUE(tensor.is_legal(Key{{0, 0, 0}}));
  EXPECT_TRUE(tensor.is_legal(Key{{1, 0, 1}}));
  EXPECT_FALSE(tensor.contains(Key{{0, 0, 0}}));
  EXPECT_TRUE(tensor.contains(Key{{1, 1, 2}}));

  auto block = tensor.block(Key{{1, 1, 2}});
  static_assert(decltype(block)::rank() == 2);
  EXPECT_EQ(block.extent(0), 3);
  EXPECT_EQ(block.extent(1), 6);
  block[2, 5] = -2.25;

  Tensor const& const_tensor = tensor;
  auto const const_block = const_tensor.block(Key{{1, 1, 2}});
  auto const stored_value = const_block[2, 5];
  EXPECT_DOUBLE_EQ(stored_value, -2.25);
}

TYPED_TEST(SparseBlockTensorTest, OrderFourMpoLikeTensorUsesDomainCodomainSelectionRule)
{
  Symmetry const sym{"N:U(1)"};
  LocalSpace const left(sym,
                        {
                            make_qnum(sym, {{"N", 0}}),
                            make_qnum(sym, {{"N", 1}}),
                        },
                        "left-virtual");
  LocalSpace const right(sym,
                         {
                             make_qnum(sym, {{"N", 0}}),
                             make_qnum(sym, {{"N", 1}}),
                         },
                         "right-virtual");
  LocalSpace const ket(sym,
                       {
                           make_qnum(sym, {{"N", 0}}),
                           make_qnum(sym, {{"N", 1}}),
                       },
                       "ket");
  LocalSpace const bra(sym,
                       {
                           make_qnum(sym, {{"N", 0}}),
                           make_qnum(sym, {{"N", 1}}),
                       },
                       "bra");

  using Tensor = BlockTensor<double, Domain<LocalSpace, LocalSpace>, Codomain<LocalSpace, LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{left, ket}, Codomain{right, bra}, {Key{{0, 1, 0, 1}}, Key{{0, 0, 0, 0}}});

  static_assert(Tensor::order() == 4);
  static_assert(Tensor::key_coordinate_count() == 4);
  static_assert(Tensor::dense_block_order() == 0);
  EXPECT_EQ(tensor.stored_block_count(), 2);
  EXPECT_EQ(tensor.legal_block_count(), 6);
  EXPECT_FALSE(tensor.has_all_legal_blocks());
  EXPECT_TRUE(tensor.is_legal(Key{{1, 0, 0, 1}}));
  EXPECT_FALSE(tensor.is_legal(Key{{0, 0, 1, 0}}));

  auto block = tensor.block(Key{{0, 1, 0, 1}});
  static_assert(decltype(block)::rank() == 0);
  EXPECT_DOUBLE_EQ(block[], 0.0);
  block[] = 11.0;

  Tensor const& const_tensor = tensor;
  auto const const_block = const_tensor.block(Key{{0, 1, 0, 1}});
  auto const stored_value = const_block[];
  EXPECT_DOUBLE_EQ(stored_value, 11.0);

  if constexpr (std::same_as<TypeParam, PackedSparseBlockStorage<>>)
  {
    auto const offsets = tensor.storage().offsets();
    ASSERT_EQ(offsets.size(), 3);
    EXPECT_EQ(offsets[0], 0);
    EXPECT_EQ(offsets[1], 1);
    EXPECT_EQ(offsets[2], 2);
    EXPECT_EQ(tensor.storage().buffer().size(), 2);
  }
}

TEST(BlockTensorTest, RejectsDuplicateIllegalAndMismatchedKeys)
{
  Symmetry const sym{"N:U(1)"};
  Symmetry const other_sym{"Sz:U(1)"};
  BlockSpace const rows(sym, {{make_qnum(sym, {{"N", 0}}), 2}});
  BlockSpace const columns(sym, {{make_qnum(sym, {{"N", 0}}), 3}});
  BlockSpace const other_columns(other_sym, {{make_qnum(other_sym, {{"Sz", 0}}), 3}});

  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, SeparateSparseBlockStorage<>>;
  using Key = typename Tensor::key_type;

  auto make_duplicate = [&] { return Tensor(sym, Domain{rows}, Codomain{columns}, {Key{{0, 0}}, Key{{0, 0}}}); };
  auto make_illegal = [&] {
    BlockSpace charged_columns(sym, {{make_qnum(sym, {{"N", 1}}), 3}});
    return Tensor(sym, Domain{rows}, Codomain{charged_columns}, {Key{{0, 0}}});
  };
  auto make_mismatched = [&] { return Tensor(sym, Domain{rows}, Codomain{other_columns}, {}); };

  EXPECT_THROW(static_cast<void>(make_duplicate()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_illegal()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_mismatched()), std::invalid_argument);
}

TEST(BlockTensorTest, PackedStorageUsesOneContiguousBufferAndCanonicalOffsets)
{
  Symmetry const sym{"N:U(1)"};
  BlockSpace const left(sym, {
                                 {make_qnum(sym, {{"N", 0}}), 2},
                                 {make_qnum(sym, {{"N", 1}}), 3},
                             });
  LocalSpace const physical(sym, {
                                     make_qnum(sym, {{"N", 0}}),
                                     make_qnum(sym, {{"N", 1}}),
                                 });
  BlockSpace const right(sym, {
                                  {make_qnum(sym, {{"N", 0}}), 4},
                                  {make_qnum(sym, {{"N", 1}}), 5},
                                  {make_qnum(sym, {{"N", 2}}), 6},
                              });

  using Tensor = BlockTensor<double, Domain<BlockSpace, LocalSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{left, physical}, Codomain{right}, {Key{{1, 1, 2}}, Key{{0, 1, 1}}});

  auto const offsets = tensor.storage().offsets();
  ASSERT_EQ(offsets.size(), 3);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 10);
  EXPECT_EQ(offsets[2], 28);
  EXPECT_EQ(tensor.storage().buffer().size(), 28);

  auto first = tensor.block(Key{{0, 1, 1}});
  auto second = tensor.block(Key{{1, 1, 2}});
  EXPECT_EQ(second.data_handle() - first.data_handle(), 10);
}

TEST(BlockTensorTest, RejectsExtentsWhichCannotFormADenseBlock)
{
  if constexpr (!std::is_signed_v<index_type> || sizeof(index_type) > sizeof(std::size_t))
  {
    GTEST_SKIP() << "index_type represents every size_t extent on this platform";
  }
  else
  {
    Symmetry const sym{"N:U(1)"};
    auto const q0 = make_qnum(sym, {{"N", 0}});
    using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, SeparateSparseBlockStorage<>>;
    using Key = typename Tensor::key_type;

    std::size_t const maximum_index_extent = static_cast<std::size_t>(std::numeric_limits<index_type>::max());
    BlockSpace const scalar_column(sym, {{q0, 1}});
    BlockSpace const oversized_row(sym, {{q0, maximum_index_extent + 1}});
    auto make_oversized_extent = [&] {
      return Tensor(sym, Domain{oversized_row}, Codomain{scalar_column}, {Key{{0, 0}}});
    };
    EXPECT_THROW(static_cast<void>(make_oversized_extent()), std::length_error);

    BlockSpace const huge_row(sym, {{q0, maximum_index_extent}});
    BlockSpace const three_column(sym, {{q0, 3}});
    auto make_overflowing_block = [&] { return Tensor(sym, Domain{huge_row}, Codomain{three_column}, {Key{{0, 0}}}); };
    EXPECT_THROW(static_cast<void>(make_overflowing_block()), std::length_error);
  }
}
