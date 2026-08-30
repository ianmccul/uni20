#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/symmetry/block_tensor_space_traits.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

using namespace uni20;

static_assert(BlockTensorStorage<SeparateSparseBlockStorage<>>);
static_assert(BlockTensorStorage<ParallelSeparateSparseBlockStorage<>>);
static_assert(BlockTensorStorage<PackedSparseBlockStorage<>>);
static_assert(BlockTensorStorage<ParallelPackedSparseBlockStorage<>>);
static_assert(BlockTensorStorage<PackedCompleteBlockStorage<>>);
static_assert(BlockTensorStorage<ParallelPackedCompleteBlockStorage<>>);
static_assert(BlockTensorStorage<PackedDiagonalBlockStorage<>>);
static_assert(BlockTensorStorage<AsyncSeparateSparseBlockStorage<>>);
static_assert(SparseBlockStorage<SeparateSparseBlockStorage<>>);
static_assert(SparseBlockStorage<ParallelSeparateSparseBlockStorage<>>);
static_assert(SparseBlockStorage<PackedSparseBlockStorage<>>);
static_assert(SparseBlockStorage<ParallelPackedSparseBlockStorage<>>);
static_assert(SparseBlockStorage<PackedDiagonalBlockStorage<>>);
static_assert(DiagonalBlockStorage<PackedDiagonalBlockStorage<>>);
static_assert(!DiagonalBlockStorage<PackedSparseBlockStorage<>>);
static_assert(SparseBlockStorage<AsyncSeparateSparseBlockStorage<>>);
static_assert(!CompleteBlockStorage<SeparateSparseBlockStorage<>>);
static_assert(!CompleteBlockStorage<ParallelSeparateSparseBlockStorage<>>);
static_assert(!CompleteBlockStorage<PackedSparseBlockStorage<>>);
static_assert(!CompleteBlockStorage<ParallelPackedSparseBlockStorage<>>);
static_assert(CompleteBlockStorage<PackedCompleteBlockStorage<>>);
static_assert(CompleteBlockStorage<ParallelPackedCompleteBlockStorage<>>);
static_assert(!CompleteBlockStorage<AsyncSeparateSparseBlockStorage<>>);
static_assert(!std::same_as<SeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>);
static_assert(BlockTensorStorageFor<SeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<SeparateSparseBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<ParallelSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<ParallelSeparateSparseBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<PackedSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedSparseBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<ParallelPackedSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<ParallelPackedSparseBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<PackedCompleteBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<ParallelPackedCompleteBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<PackedDiagonalBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedDiagonalBlockStorage<>, double, 0, 3>);
static_assert(BlockTensorStorageFor<AsyncSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<AsyncSeparateSparseBlockStorage<>, double, 4, 0>);
static_assert(LocalBlockStorageFor<PackedCompleteBlockStorage<>, double, 2, 2>);
static_assert(LocalBlockStorageFor<AsyncSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<SeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<ParallelSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<PackedSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<ParallelPackedSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<PackedCompleteBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<ParallelPackedCompleteBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<PackedDiagonalBlockStorage<>, double, 2, 2>);
static_assert(!ImmediateLocalBlockStorageFor<AsyncSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(AsyncLocalBlockStorageFor<AsyncSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(!AsyncLocalBlockStorageFor<SeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(std::same_as<SeparateSparseBlockStorage<>::backend_selector_type, HostStorage::backend_selector_type>);
static_assert(std::same_as<PackedSparseBlockStorage<>::backend_selector_type, HostStorage::backend_selector_type>);

using ImmediateConceptTensor =
    BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, SeparateSparseBlockStorage<>>;
using AsyncConceptTensor =
    BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, AsyncSeparateSparseBlockStorage<>>;
static_assert(BlockTensorView<ImmediateConceptTensor>);
static_assert(MutableBlockTensorView<ImmediateConceptTensor>);
static_assert(ImmediateBlockTensorView<ImmediateConceptTensor>);
static_assert(MutableImmediateBlockTensorView<ImmediateConceptTensor>);
static_assert(!BorrowedBlockTensorView<ImmediateConceptTensor>);
static_assert(!AsyncBlockTensorView<ImmediateConceptTensor>);
static_assert(!MutableBlockTensorView<ImmediateConceptTensor const>);
static_assert(BlockTensorView<AsyncConceptTensor>);
static_assert(MutableBlockTensorView<AsyncConceptTensor>);
static_assert(!ImmediateBlockTensorView<AsyncConceptTensor>);
static_assert(AsyncBlockTensorView<AsyncConceptTensor>);
static_assert(MutableAsyncBlockTensorView<AsyncConceptTensor>);
static_assert(!BorrowedBlockTensorView<AsyncConceptTensor>);

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

using SparseStorageTypes = ::testing::Types<SeparateSparseBlockStorage<>, ParallelSeparateSparseBlockStorage<>,
                                            PackedSparseBlockStorage<>, ParallelPackedSparseBlockStorage<>>;
TYPED_TEST_SUITE(SparseBlockTensorTest, SparseStorageTypes);

template <class Storage> class CompleteBlockTensorTest : public ::testing::Test {};

using CompleteStorageTypes = ::testing::Types<PackedCompleteBlockStorage<>, ParallelPackedCompleteBlockStorage<>>;
TYPED_TEST_SUITE(CompleteBlockTensorTest, CompleteStorageTypes);

TYPED_TEST(CompleteBlockTensorTest, DerivesCanonicalLegalKeysAndPackedOffsets)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const rows(sym, {{q0, 2}, {q1, 3}}, "rows");
  BlockSpace const columns(sym, {{q0, 4}, {q1, 5}}, "columns");

  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{rows}, Codomain{columns});

  EXPECT_EQ(tensor.stored_block_count(), 2);
  EXPECT_EQ(tensor.legal_block_count(), 2);
  EXPECT_TRUE(tensor.has_all_legal_blocks());
  EXPECT_EQ(tensor.stored_keys()[0], (Key{{0, 0}}));
  EXPECT_EQ(tensor.stored_keys()[1], (Key{{1, 1}}));

  auto const offsets = tensor.storage().offsets();
  ASSERT_EQ(offsets.size(), 3);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 8);
  EXPECT_EQ(offsets[2], 23);
  EXPECT_EQ(tensor.storage().buffer().size(), 23);
}

TYPED_TEST(CompleteBlockTensorTest, RepresentsScalarUnitsAndEmptyLegalKeySets)
{
  Symmetry const sym{"N:U(1)"};

  using ScalarTensor = BlockTensor<double, Domain<>, Codomain<>, TypeParam>;
  ScalarTensor scalar(sym, Domain<>{}, Codomain<>{});
  ASSERT_EQ(scalar.stored_block_count(), 1);
  EXPECT_TRUE(scalar.has_all_legal_blocks());
  scalar.block_by_ordinal(0)[] = 1.0;
  EXPECT_DOUBLE_EQ(scalar.block_by_ordinal(0)[], 1.0);

  BlockSpace const empty(sym, {}, "empty");
  using EmptyTensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, TypeParam>;
  EmptyTensor tensor(sym, Domain{empty}, Codomain{empty});
  EXPECT_EQ(tensor.stored_block_count(), 0);
  EXPECT_EQ(tensor.legal_block_count(), 0);
  EXPECT_TRUE(tensor.has_all_legal_blocks());
  ASSERT_EQ(tensor.storage().offsets().size(), 1);
  EXPECT_EQ(tensor.storage().offsets()[0], 0);
  EXPECT_EQ(tensor.storage().buffer().size(), 0);
}

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
  EXPECT_EQ(tensor.legal_block_keys(), (std::vector<Key>{Key{{0, 0}}, Key{{1, 1}}}));
  EXPECT_TRUE(tensor.has_all_legal_blocks());
  EXPECT_EQ(tensor.stored_keys()[0], (Key{{0, 0}}));
  EXPECT_EQ(tensor.stored_keys()[1], (Key{{1, 1}}));
  EXPECT_TRUE(tensor.is_legal(Key{{0, 0}}));
  EXPECT_FALSE(tensor.is_legal(Key{{0, 1}}));
  EXPECT_FALSE(tensor.is_legal(Key{{2, 0}}));
  EXPECT_FALSE(tensor.contains(Key{{0, 1}}));

  auto block = tensor.block(Key{{1, 1}});
  static_assert(decltype(block)::rank() == 2);
  static_assert(MutableRankedImmediateTensorView<decltype(block), 2>);
  static_assert(!MdspecLike<decltype(block)>);
  EXPECT_EQ(block.extent(0), 3);
  EXPECT_EQ(block.extent(1), 5);
  block[2, 4] = 7.5;

  Tensor const& const_tensor = tensor;
  auto const const_block = const_tensor.block(Key{{1, 1}});
  static_assert(RankedImmediateTensorView<decltype(const_block), 2>);
  static_assert(!MutableTensorView<decltype(const_block)>);
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

TYPED_TEST(SparseBlockTensorTest, OrderZeroTensorUnitHasOneLegalScalarBlock)
{
  Symmetry const sym{"N:U(1)"};
  using Tensor = BlockTensor<double, Domain<>, Codomain<>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain<>{}, Codomain<>{}, {Key{}});

  static_assert(Tensor::order() == 0);
  static_assert(Tensor::key_coordinate_count() == 0);
  static_assert(Tensor::dense_block_order() == 0);
  EXPECT_EQ(tensor.legal_block_count(), 1);
  EXPECT_EQ(tensor.stored_block_count(), 1);
  EXPECT_TRUE(tensor.is_legal(Key{}));
  EXPECT_TRUE(tensor.has_all_legal_blocks());

  auto scalar = tensor.block(Key{});
  static_assert(decltype(scalar)::rank() == 0);
  scalar[] = 2.5;
  EXPECT_DOUBLE_EQ(tensor.block(Key{})[], 2.5);

  Tensor structural_zero(sym, Domain<>{}, Codomain<>{}, {});
  EXPECT_EQ(structural_zero.legal_block_count(), 1);
  EXPECT_EQ(structural_zero.stored_block_count(), 0);
  EXPECT_FALSE(structural_zero.has_all_legal_blocks());
}

TYPED_TEST(SparseBlockTensorTest, OrderOneTensorStoresOnlyIdentityChargeOccurrences)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  LocalSpace const basis(sym, {q0, q1, q0}, "basis");

  using DomainTensor = BlockTensor<double, Domain<LocalSpace>, Codomain<>, TypeParam>;
  using DomainKey = typename DomainTensor::key_type;
  DomainTensor domain_vector(sym, Domain{basis}, Codomain<>{}, {DomainKey{{2}}, DomainKey{{0}}});

  static_assert(DomainTensor::order() == 1);
  static_assert(DomainTensor::key_coordinate_count() == 1);
  static_assert(DomainTensor::dense_block_order() == 0);
  EXPECT_EQ(domain_vector.legal_block_count(), 2);
  EXPECT_TRUE(domain_vector.is_legal(DomainKey{{0}}));
  EXPECT_FALSE(domain_vector.is_legal(DomainKey{{1}}));
  EXPECT_TRUE(domain_vector.is_legal(DomainKey{{2}}));

  using CodomainTensor = BlockTensor<double, Domain<>, Codomain<LocalSpace>, TypeParam>;
  using CodomainKey = typename CodomainTensor::key_type;
  CodomainTensor codomain_vector(sym, Domain<>{}, Codomain{basis}, {CodomainKey{{0}}, CodomainKey{{2}}});
  EXPECT_EQ(codomain_vector.legal_block_count(), 2);
  EXPECT_TRUE(codomain_vector.is_legal(CodomainKey{{0}}));
  EXPECT_FALSE(codomain_vector.is_legal(CodomainKey{{1}}));
  EXPECT_TRUE(codomain_vector.is_legal(CodomainKey{{2}}));
}

TYPED_TEST(SparseBlockTensorTest, OrderOneFixedIrrepIsLegalOnlyWhenItIsIdentity)
{
  Symmetry const sym{"N:U(1)"};
  QNumSpace const identity(QNum::identity(sym), "identity");
  QNumSpace const charged(make_qnum(sym, {{"N", 1}}), "charged");

  using Tensor = BlockTensor<double, Domain<QNumSpace>, Codomain<>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor identity_tensor(sym, Domain{identity}, Codomain<>{}, {Key{}});
  Tensor charged_zero(sym, Domain{charged}, Codomain<>{}, {});

  static_assert(Tensor::key_coordinate_count() == 0);
  static_assert(Tensor::dense_block_order() == 0);
  EXPECT_EQ(identity_tensor.legal_block_count(), 1);
  EXPECT_TRUE(identity_tensor.is_legal(Key{}));
  EXPECT_EQ(charged_zero.legal_block_count(), 0);
  EXPECT_FALSE(charged_zero.is_legal(Key{}));
  EXPECT_THROW((Tensor(sym, Domain{charged}, Codomain<>{}, {Key{}})), std::invalid_argument);
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
  EXPECT_EQ(second.mdspan().data_handle() - first.mdspan().data_handle(), 10);
}

TEST(BlockTensorTest, AlignedPackedStoragePadsBlockStartsWithinOneBuffer)
{
  Symmetry const sym{"N:U(1)"};
  BlockSpace const rows(sym, {
                                 {make_qnum(sym, {{"N", 0}}), 3},
                                 {make_qnum(sym, {{"N", 1}}), 2},
                             });
  BlockSpace const columns(sym, {
                                    {make_qnum(sym, {{"N", 0}}), 3},
                                    {make_qnum(sym, {{"N", 1}}), 2},
                                });

  using Storage = ParallelPackedSparseBlockStorage<HostStorage, 64>;
  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{rows}, Codomain{columns}, {Key{{0, 0}}, Key{{1, 1}}});

  auto const offsets = tensor.storage().offsets();
  ASSERT_EQ(offsets.size(), 3);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 16);
  EXPECT_EQ(offsets[2], 20);
  EXPECT_EQ(tensor.storage().buffer().size(), 20);
  EXPECT_TRUE(tensor.storage().has_padding());
  EXPECT_TRUE(std::ranges::equal(tensor.storage().block_ends(), std::array<std::size_t, 2>{9, 20}));
  for (std::size_t offset = 9; offset < 16; ++offset)
    EXPECT_DOUBLE_EQ(tensor.storage().buffer().data()[offset], 0.0);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(tensor.block_by_ordinal(0).mdspan().data_handle()) % 64, 0);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(tensor.block_by_ordinal(1).mdspan().data_handle()) % 64, 0);
}

TEST(BlockTensorTest, AsyncSeparateStorageReturnsMdspecWithStableBlockEpochIdentity)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const rows(sym, {{q0, 2}}, "rows");
  BlockSpace const columns(sym, {{q0, 3}}, "columns");
  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, AsyncSeparateSparseBlockStorage<>>;
  using Key = typename Tensor::key_type;
  Key const key{{0, 0}};
  Tensor tensor(sym, Domain{rows}, Codomain{columns}, {key});

  auto descriptor = tensor.block(key);
  auto const const_descriptor = std::as_const(tensor).block(key);
  static_assert(MutableRankedTensorView<decltype(descriptor), 2>);
  static_assert(RankedTensorView<decltype(const_descriptor), 2>);
  static_assert(!ImmediateTensorView<decltype(descriptor)>);
  static_assert(std::same_as<typename tensor_mdspec_t<decltype(const_descriptor)>::element_type, double const>);
  EXPECT_EQ(descriptor.extent(0), 2);
  EXPECT_EQ(descriptor.extent(1), 3);
  EXPECT_EQ(&mdspec_of(descriptor).data_descriptor().async_block(), &tensor.async_block(key));
  EXPECT_EQ(&mdspec_of(const_descriptor).data_descriptor().async_block(), &std::as_const(tensor).async_block(key));
  EXPECT_EQ(&tensor.async_block_by_ordinal(0), &tensor.async_block(key));
  EXPECT_THROW(static_cast<void>(tensor.block_by_ordinal(1)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(tensor.async_block_by_ordinal(1)), std::out_of_range);

  auto& block_value = tensor.async_block(key).unsafe_value_ref();
  block_value[1, 2] = 4.5;
  EXPECT_DOUBLE_EQ((std::as_const(tensor).async_block(key).unsafe_value_ref()[1, 2]), 4.5);
}

TEST(BlockTensorTest, PackedDiagonalStorageRepresentsRectangularBlocksWithoutStructuralZeros)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const rows(sym, {{q0, 2}}, "rows");
  BlockSpace const columns(sym, {{q0, 4}}, "columns");
  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedDiagonalBlockStorage<>>;
  using Key = Tensor::key_type;
  Key const key{{0, 0}};
  Tensor tensor(sym, Domain{rows}, Codomain{columns}, {key});

  static_assert(DiagonalBlockStorage<typename Tensor::storage_policy>);
  static_assert(MutableImmediateBlockTensorView<Tensor>);
  auto values = tensor.diagonal_values(key);
  ASSERT_EQ(values.size(), 2);
  EXPECT_EQ(tensor.storage().buffer().size(), 2);
  values[0] = 2.0;
  values[1] = -3.0;

  auto block = std::as_const(tensor).block(key);
  EXPECT_EQ(block.extent(0), 2);
  EXPECT_EQ(block.extent(1), 4);
  EXPECT_DOUBLE_EQ((block[0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((block[1, 1]), -3.0);
  EXPECT_DOUBLE_EQ((block[0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((block[1, 3]), 0.0);
}

TEST(BlockTensorTest, PackedDiagonalStorageRepresentsRankThreeCopyBlocks)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const first(sym, {{q0, 2}}, "first");
  BlockSpace const second(sym, {{q0, 3}}, "second");
  BlockSpace const third(sym, {{q0, 4}}, "third");
  using Tensor =
      BlockTensor<double, Domain<BlockSpace, BlockSpace>, Codomain<BlockSpace>, PackedDiagonalBlockStorage<>>;
  using Key = Tensor::key_type;
  Key const key{{0, 0, 0}};
  Tensor tensor(sym, Domain{first, second}, Codomain{third}, {key});
  auto values = tensor.diagonal_values_by_ordinal(0);
  ASSERT_EQ(values.size(), 2);
  values[0] = 5.0;
  values[1] = 7.0;

  auto block = std::as_const(tensor).block(key);
  EXPECT_DOUBLE_EQ((block[0, 0, 0]), 5.0);
  EXPECT_DOUBLE_EQ((block[1, 1, 1]), 7.0);
  EXPECT_DOUBLE_EQ((block[1, 0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((block[0, 2, 0]), 0.0);
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
