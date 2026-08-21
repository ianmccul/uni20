#include <uni20/symmetry/block_tensor_repartition.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <type_traits>
#include <utility>

using namespace uni20;

template <class Storage> class BlockTensorRepartitionTest : public ::testing::Test {};

using RepartitionStorageTypes = ::testing::Types<SeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>;
TYPED_TEST_SUITE(BlockTensorRepartitionTest, RepartitionStorageTypes);

TYPED_TEST(BlockTensorRepartitionTest, RightBendResortsLogicalKeysWithoutMovingScalarPayload)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const domain_left(sym, {q0}, "domain-left");
  LocalSpace const moved(sym, {q0, q0}, "moved");
  LocalSpace const codomain_left(sym, {q0, q0}, "codomain-left");
  LocalSpace const codomain_right(sym, {q0}, "codomain-right");

  using Tensor = BlockTensor<double, Domain<LocalSpace, LocalSpace>, Codomain<LocalSpace, LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const first_source_key{{0, 0, 1, 0}};
  Key const second_source_key{{0, 1, 0, 0}};
  Tensor tensor(sym, Domain{domain_left, moved}, Codomain{codomain_left, codomain_right},
                {second_source_key, first_source_key});
  tensor.block(first_source_key)[] = 10.0;
  tensor.block(second_source_key)[] = 20.0;
  auto* const first_address = tensor.block(first_source_key).data_handle();
  auto* const second_address = tensor.block(second_source_key).data_handle();

  auto bent = repartition<MorphismSide::Domain, BoundaryEnd::Right>(tensor);
  using Bent = decltype(bent);
  using BentMovedSpace = typename Bent::codomain_type::template space_type<2>;
  static_assert(Bent::domain_type::size() == 1);
  static_assert(Bent::codomain_type::size() == 3);
  static_assert(DualSpace<BentMovedSpace>);
  static_assert(std::same_as<primal_space_t<BentMovedSpace>, LocalSpace>);
  static_assert(Bent::dense_block_order() == 0);

  Key const first_bent_key{{0, 0, 0, 1}};
  Key const second_bent_key{{0, 1, 0, 0}};
  ASSERT_EQ(bent.stored_keys().size(), 2);
  EXPECT_EQ(bent.stored_keys()[0], first_bent_key);
  EXPECT_EQ(bent.stored_keys()[1], second_bent_key);
  EXPECT_EQ(bent.block(first_bent_key).data_handle(), second_address);
  EXPECT_EQ(bent.block(second_bent_key).data_handle(), first_address);
  EXPECT_DOUBLE_EQ(bent.block(first_bent_key)[], 20.0);
  EXPECT_DOUBLE_EQ(bent.block(second_bent_key)[], 10.0);

  bent.block(first_bent_key)[] = 21.0;
  EXPECT_DOUBLE_EQ(tensor.block(second_source_key)[], 21.0);

  auto restored = repartition<MorphismSide::Codomain, BoundaryEnd::Right>(bent);
  static_assert(std::same_as<typename decltype(restored)::domain_type, typename Tensor::domain_type>);
  static_assert(std::same_as<typename decltype(restored)::codomain_type, typename Tensor::codomain_type>);
  EXPECT_EQ(restored.domain(), tensor.domain());
  EXPECT_EQ(restored.codomain(), tensor.codomain());
  EXPECT_EQ(restored.stored_keys()[0], first_source_key);
  EXPECT_EQ(restored.stored_keys()[1], second_source_key);
  EXPECT_EQ(restored.block(first_source_key).data_handle(), first_address);
  EXPECT_EQ(restored.block(second_source_key).data_handle(), second_address);
}

TYPED_TEST(BlockTensorRepartitionTest, RightBendPermutesDenseAxesThroughStridedView)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const physical(sym, {q0}, "physical");
  BlockSpace const moved(sym, {{q0, 2}}, "moved-bond");
  BlockSpace const output(sym, {{q0, 3}}, "output-bond");

  using Tensor = BlockTensor<double, Domain<LocalSpace, BlockSpace>, Codomain<BlockSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const source_key{{0, 0, 0}};
  Tensor tensor(sym, Domain{physical, moved}, Codomain{output}, {source_key});
  auto source = tensor.block(source_key);
  for (std::size_t row = 0; row < 2; ++row)
  {
    for (std::size_t column = 0; column < 3; ++column)
    {
      source[row, column] = static_cast<double>(10 * row + column);
    }
  }

  auto bent = repartition<MorphismSide::Domain, BoundaryEnd::Right>(tensor);
  Key const bent_key{{0, 0, 0}};
  auto block = bent.block(bent_key);
  static_assert(decltype(block)::rank() == 2);
  EXPECT_EQ(block.data_handle(), source.data_handle());
  EXPECT_EQ(block.extent(0), 3);
  EXPECT_EQ(block.extent(1), 2);
  EXPECT_EQ(block.stride(0), source.stride(1));
  EXPECT_EQ(block.stride(1), source.stride(0));
  for (std::size_t column = 0; column < 3; ++column)
  {
    for (std::size_t row = 0; row < 2; ++row)
    {
      auto const bent_value = block[column, row];
      auto const source_value = source[row, column];
      EXPECT_DOUBLE_EQ(bent_value, source_value);
    }
  }

  block[2, 1] = 42.0;
  auto const source_value = source[1, 2];
  EXPECT_DOUBLE_EQ(source_value, 42.0);

  auto restored = repartition<MorphismSide::Codomain, BoundaryEnd::Right>(bent);
  auto restored_block = restored.block(source_key);
  EXPECT_EQ(restored_block.data_handle(), source.data_handle());
  EXPECT_EQ(restored_block.extent(0), 2);
  EXPECT_EQ(restored_block.extent(1), 3);
  EXPECT_EQ(restored_block.stride(0), source.stride(0));
  EXPECT_EQ(restored_block.stride(1), source.stride(1));
  auto const restored_value = restored_block[1, 2];
  EXPECT_DOUBLE_EQ(restored_value, 42.0);

  auto const_bent = repartition<MorphismSide::Domain, BoundaryEnd::Right>(std::as_const(tensor));
  auto const const_block = const_bent.block(bent_key);
  static_assert(std::same_as<typename decltype(const_block)::element_type, double const>);
  static_assert(!std::is_copy_assignable_v<decltype(const_bent)>);
  static_assert(!std::is_move_assignable_v<decltype(const_bent)>);
  EXPECT_EQ(const_block.data_handle(), source.data_handle());
  auto const const_value = const_block[2, 1];
  EXPECT_DOUBLE_EQ(const_value, 42.0);

  auto const_restored = repartition<MorphismSide::Codomain, BoundaryEnd::Right>(const_bent);
  auto const const_restored_block = const_restored.block(source_key);
  static_assert(std::same_as<typename decltype(const_restored_block)::element_type, double const>);
  EXPECT_EQ(const_restored_block.data_handle(), source.data_handle());
}

TYPED_TEST(BlockTensorRepartitionTest, ChargedCoordinatePermutationMatchesDualizedSelectionRule)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  auto const q2 = make_qnum(sym, {{"N", 2}});
  LocalSpace const left(sym, {q0}, "left");
  LocalSpace const moved(sym, {q1, q2}, "moved");
  LocalSpace const output(sym, {q2, q1}, "output");

  using Tensor = BlockTensor<double, Domain<LocalSpace, LocalSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const source_q1{{0, 0, 1}};
  Key const source_q2{{0, 1, 0}};
  Tensor tensor(sym, Domain{left, moved}, Codomain{output}, {source_q1, source_q2});
  tensor.block(source_q1)[] = 1.0;
  tensor.block(source_q2)[] = 2.0;

  auto bent = repartition<MorphismSide::Domain, BoundaryEnd::Right>(tensor);
  Key const bent_q2{{0, 0, 1}};
  Key const bent_q1{{0, 1, 0}};
  EXPECT_TRUE(bent.is_legal(bent_q1));
  EXPECT_TRUE(bent.is_legal(bent_q2));
  EXPECT_FALSE(bent.is_legal(Key{{0, 0, 0}}));
  EXPECT_DOUBLE_EQ(bent.block(bent_q1)[], 1.0);
  EXPECT_DOUBLE_EQ(bent.block(bent_q2)[], 2.0);
}

TYPED_TEST(BlockTensorRepartitionTest, LeftBendPermutesChargedDenseAxisWithoutMovingPayload)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const input(sym, {{q1, 2}}, "input");
  BlockSpace const moved(sym, {{q1, 3}}, "moved");
  QNumSpace const tail(q0, "tail");

  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace, QNumSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const key{{0, 0}};
  Tensor tensor(sym, Domain{input}, Codomain{moved, tail}, {key});
  auto source = tensor.block(key);
  source[1, 2] = 12.0;

  auto bent = repartition<MorphismSide::Codomain, BoundaryEnd::Left>(tensor);
  auto block = bent.block(key);
  EXPECT_TRUE(bent.is_legal(key));
  EXPECT_EQ(block.data_handle(), source.data_handle());
  EXPECT_EQ(block.extent(0), 3);
  EXPECT_EQ(block.extent(1), 2);
  EXPECT_EQ(block.stride(0), source.stride(1));
  EXPECT_EQ(block.stride(1), source.stride(0));
  EXPECT_DOUBLE_EQ((block[2, 1]), 12.0);
}

TYPED_TEST(BlockTensorRepartitionTest, LeftBendAndItsInversePreservePlanarFactorOrder)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const domain(sym, {q0}, "domain");
  LocalSpace const moved(sym, {q0, q0}, "moved");
  LocalSpace const tail(sym, {q0}, "tail");

  using Tensor = BlockTensor<double, Domain<LocalSpace>, Codomain<LocalSpace, LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const source_key{{0, 1, 0}};
  Tensor tensor(sym, Domain{domain}, Codomain{moved, tail}, {source_key});
  tensor.block(source_key)[] = 5.0;
  auto* const address = tensor.block(source_key).data_handle();

  auto bent = repartition<MorphismSide::Codomain, BoundaryEnd::Left>(tensor);
  using BentDomainFirst = typename decltype(bent)::domain_type::template space_type<0>;
  static_assert(DualSpace<BentDomainFirst>);
  EXPECT_EQ(bent.domain().template space<0>().label(), "moved");
  EXPECT_EQ(bent.domain().template space<1>().label(), "domain");
  EXPECT_EQ(bent.codomain().template space<0>().label(), "tail");

  Key const bent_key{{1, 0, 0}};
  EXPECT_TRUE(bent.is_legal(bent_key));
  EXPECT_EQ(bent.block(bent_key).data_handle(), address);
  EXPECT_DOUBLE_EQ(bent.block(bent_key)[], 5.0);

  auto restored = repartition<MorphismSide::Domain, BoundaryEnd::Left>(bent);
  EXPECT_EQ(restored.domain(), tensor.domain());
  EXPECT_EQ(restored.codomain(), tensor.codomain());
  EXPECT_EQ(restored.stored_keys()[0], source_key);
  EXPECT_EQ(restored.block(source_key).data_handle(), address);
}

TYPED_TEST(BlockTensorRepartitionTest, ChargedFixedIrrepBendUsesDualChargeWithoutAKeyCoordinate)
{
  Symmetry const sym{"N:U(1)"};
  auto const q1 = make_qnum(sym, {{"N", 1}});
  QNumSpace const charge(q1, "charge");
  LocalSpace const state(sym, {q1}, "state");

  using Tensor = BlockTensor<double, Domain<QNumSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain{charge}, Codomain{state}, {Key{{0}}});
  auto bent = repartition<MorphismSide::Domain, BoundaryEnd::Right>(tensor);

  static_assert(decltype(bent)::key_coordinate_count() == 1);
  static_assert(DualSpace<typename decltype(bent)::codomain_type::template space_type<1>>);
  EXPECT_TRUE(bent.domain().empty());
  EXPECT_EQ(bent.codomain().template space<1>().qnum(), dual(q1));
  EXPECT_TRUE(bent.is_legal(Key{{0}}));
  EXPECT_EQ(bent.legal_block_count(), 1);
}
