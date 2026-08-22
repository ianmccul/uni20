#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <type_traits>

using namespace uni20;

template <class Storage> class BlockTensorPermutationTest : public ::testing::Test {};

using PermutationStorageTypes = ::testing::Types<SeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>;
TYPED_TEST_SUITE(BlockTensorPermutationTest, PermutationStorageTypes);

TYPED_TEST(BlockTensorPermutationTest, PermutesBoundaryTypesLabelsAndDenseAxesWithoutMovingPayload)
{
  Symmetry const sym{"N:U(1)"};
  auto const q1 = make_qnum(sym, {{"N", 1}});
  auto const q2 = make_qnum(sym, {{"N", 2}});
  BlockSpace const left(sym, {{q1, 2}}, "left");
  BlockSpace const right(sym, {{q2, 3}}, "right");
  QNumSpace const output_left(q2, "output-left");
  QNumSpace const output_right(q1, "output-right");

  using Tensor = BlockTensor<double, Domain<BlockSpace, BlockSpace>, Codomain<QNumSpace, QNumSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const key{{0, 0}};
  Tensor tensor(sym, Domain{left, right}, Codomain{output_left, output_right}, {key});
  auto source = tensor.block(key);
  source[1, 2] = 12.0;

  auto view = permute<1, 0, 3, 2>(tensor);
  using View = decltype(view);
  static_assert(std::same_as<typename View::domain_type, Domain<BlockSpace, BlockSpace>>);
  static_assert(std::same_as<typename View::codomain_type, Codomain<QNumSpace, QNumSpace>>);
  static_assert(!requires(Tensor& candidate) { permute<2, 1, 0, 3>(candidate); });

  EXPECT_EQ(view.domain().template space<0>().label(), "right");
  EXPECT_EQ(view.domain().template space<1>().label(), "left");
  EXPECT_EQ(view.codomain().template space<0>().label(), "output-right");
  EXPECT_EQ(view.codomain().template space<1>().label(), "output-left");
  EXPECT_TRUE(view.is_legal(key));

  auto block = view.block(key);
  EXPECT_EQ(block.data_handle(), source.data_handle());
  EXPECT_EQ(block.extent(0), 3);
  EXPECT_EQ(block.extent(1), 2);
  EXPECT_EQ(block.stride(0), source.stride(1));
  EXPECT_EQ(block.stride(1), source.stride(0));
  EXPECT_DOUBLE_EQ((block[2, 1]), 12.0);

  auto restored = permute<1, 0, 3, 2>(view);
  static_assert(std::same_as<typename decltype(restored)::domain_type, typename Tensor::domain_type>);
  static_assert(std::same_as<typename decltype(restored)::codomain_type, typename Tensor::codomain_type>);
  EXPECT_EQ(restored.domain(), tensor.domain());
  EXPECT_EQ(restored.codomain(), tensor.codomain());
  EXPECT_EQ(restored.block(key).data_handle(), source.data_handle());
}

TYPED_TEST(BlockTensorPermutationTest, ResortsLogicalKeysWhileKeepingPhysicalBindings)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const domain_left(sym, {q0}, "domain-left");
  LocalSpace const domain_right(sym, {q0, q0}, "domain-right");
  LocalSpace const codomain_left(sym, {q0, q0}, "codomain-left");
  LocalSpace const codomain_right(sym, {q0}, "codomain-right");

  using Tensor = BlockTensor<double, Domain<LocalSpace, LocalSpace>, Codomain<LocalSpace, LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const first_source{{0, 0, 1, 0}};
  Key const second_source{{0, 1, 0, 0}};
  Tensor tensor(sym, Domain{domain_left, domain_right}, Codomain{codomain_left, codomain_right},
                {first_source, second_source});
  tensor.block(first_source)[] = 10.0;
  tensor.block(second_source)[] = 20.0;
  auto* const first_address = tensor.block(first_source).data_handle();
  auto* const second_address = tensor.block(second_source).data_handle();

  auto view = permute<1, 0, 3, 2>(tensor);
  Key const first_view{{0, 0, 0, 1}};
  Key const second_view{{1, 0, 0, 0}};
  ASSERT_EQ(view.stored_keys().size(), 2);
  EXPECT_EQ(view.stored_keys()[0], first_view);
  EXPECT_EQ(view.stored_keys()[1], second_view);
  EXPECT_EQ(view.block(first_view).data_handle(), first_address);
  EXPECT_EQ(view.block(second_view).data_handle(), second_address);
  EXPECT_DOUBLE_EQ(view.block(first_view)[], 10.0);
  EXPECT_DOUBLE_EQ(view.block(second_view)[], 20.0);
}

TYPED_TEST(BlockTensorPermutationTest, TensorUnitPermutationIsIdentityView)
{
  Symmetry const sym{"N:U(1)"};
  using Tensor = BlockTensor<double, Domain<>, Codomain<>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor tensor(sym, Domain<>{}, Codomain<>{}, {Key{}});
  tensor.block(Key{})[] = 3.0;

  auto view = permute<>(tensor);
  static_assert(decltype(view)::order() == 0);
  EXPECT_EQ(view.domain(), tensor.domain());
  EXPECT_EQ(view.codomain(), tensor.codomain());
  EXPECT_EQ(view.block(Key{}).data_handle(), tensor.block(Key{}).data_handle());
  EXPECT_DOUBLE_EQ(view.block(Key{})[], 3.0);
}

TYPED_TEST(BlockTensorPermutationTest, PermutationAndRepartitionBendAnInteriorFactorExplicitly)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const left(sym, {q0}, "left");
  LocalSpace const moved(sym, {q0, q0}, "moved");
  LocalSpace const right(sym, {q0}, "right");
  LocalSpace const output(sym, {q0}, "output");

  using Tensor = BlockTensor<double, Domain<LocalSpace, LocalSpace, LocalSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const source_key{{0, 1, 0, 0}};
  Tensor tensor(sym, Domain{left, moved, right}, Codomain{output}, {source_key});
  tensor.block(source_key)[] = 6.0;
  auto* const address = tensor.block(source_key).data_handle();

  auto edge = permute<0, 2, 1, 3>(tensor);
  auto bent = repartition<MorphismSide::Domain, BoundaryEnd::Right>(edge);
  using Bent = decltype(bent);
  static_assert(std::same_as<typename Bent::domain_type, Domain<LocalSpace, LocalSpace>>);
  static_assert(std::same_as<typename Bent::codomain_type, Codomain<LocalSpace, Dual<LocalSpace>>>);
  Key const bent_key{{0, 0, 0, 1}};
  EXPECT_EQ(bent.domain().template space<0>().label(), "left");
  EXPECT_EQ(bent.domain().template space<1>().label(), "right");
  EXPECT_EQ(bent.codomain().template space<0>().label(), "output");
  EXPECT_EQ(bent.codomain().template space<1>().label(), "moved");
  EXPECT_EQ(bent.block(bent_key).data_handle(), address);

  auto unbent = repartition<MorphismSide::Codomain, BoundaryEnd::Right>(bent);
  auto restored = permute<0, 2, 1, 3>(unbent);
  EXPECT_EQ(restored.domain(), tensor.domain());
  EXPECT_EQ(restored.codomain(), tensor.codomain());
  EXPECT_EQ(restored.block(source_key).data_handle(), address);
}

TEST(BlockTensorPermutationTest, AsyncMdspecPermutationPreservesEpochIdentityAndPermutesMapping)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const first(sym, {{q0, 2}}, "first");
  BlockSpace const second(sym, {{q0, 3}}, "second");
  using Tensor = BlockTensor<double, Domain<BlockSpace, BlockSpace>, Codomain<>, AsyncSeparateSparseBlockStorage<>>;
  using Key = typename Tensor::key_type;
  Key const key{{0, 0}};
  Tensor tensor(sym, Domain{first, second}, Codomain<>{}, {key});
  auto source = tensor.block(key);

  auto view = permute<1, 0>(tensor);
  auto block = view.block(key);
  static_assert(MutableRankedMdspecLike<decltype(block), 2>);
  static_assert(!MdspanLike<decltype(block)>);
  EXPECT_EQ(block.extent(0), 3);
  EXPECT_EQ(block.extent(1), 2);
  EXPECT_EQ(block.stride(0), source.stride(1));
  EXPECT_EQ(block.stride(1), source.stride(0));
  EXPECT_EQ(&block.data_descriptor().async_block(), &tensor.async_block(key));
}

TEST(BlockTensorPermutationTest, AsyncRankZeroPermutationPreservesScalarEpochIdentity)
{
  Symmetry const sym{"N:U(1)"};
  using Tensor = BlockTensor<double, Domain<>, Codomain<>, AsyncSeparateSparseBlockStorage<>>;
  Tensor::key_type const key{};
  Tensor tensor(sym, Domain<>{}, Codomain<>{}, {key});

  auto view = permute<>(tensor);
  auto block = view.block(key);
  static_assert(MutableRankedMdspecLike<decltype(block), 0>);
  static_assert(!MdspanLike<decltype(block)>);
  EXPECT_EQ(&block.data_descriptor().async_block(), &tensor.async_block(key));
}
