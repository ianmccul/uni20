#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <type_traits>

using namespace uni20;

template <class LeftStorage, class RightStorage> struct ContractionStoragePair
{
    using left_storage = LeftStorage;
    using right_storage = RightStorage;
};

template <class StoragePair> class BlockTensorContractionTest : public ::testing::Test {};

using ContractionStorageTypes =
    ::testing::Types<ContractionStoragePair<SeparateSparseBlockStorage<>, SeparateSparseBlockStorage<>>,
                     ContractionStoragePair<SeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>,
                     ContractionStoragePair<PackedSparseBlockStorage<>, SeparateSparseBlockStorage<>>,
                     ContractionStoragePair<PackedSparseBlockStorage<>, PackedSparseBlockStorage<>>>;
TYPED_TEST_SUITE(BlockTensorContractionTest, ContractionStorageTypes);

TYPED_TEST(BlockTensorContractionTest, ContractsDenseBlockAxisAsMatrixMultiplication)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const rows(sym, {{q0, 2}}, "rows");
  BlockSpace const bond(sym, {{q0, 3}}, "bond");
  BlockSpace const columns(sym, {{q0, 2}}, "columns");

  using Left = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, typename TypeParam::left_storage>;
  using Right = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, typename TypeParam::right_storage>;
  typename Left::key_type const left_key{{0, 0}};
  typename Right::key_type const right_key{{0, 0}};
  Left left(sym, Domain{rows}, Codomain{bond}, {left_key});
  Right right(sym, Domain{bond}, Codomain{columns}, {right_key});

  auto left_block = left.block(left_key);
  left_block[0, 0] = 1.0;
  left_block[0, 1] = 2.0;
  left_block[0, 2] = 3.0;
  left_block[1, 0] = 4.0;
  left_block[1, 1] = 5.0;
  left_block[1, 2] = 6.0;
  auto right_block = right.block(right_key);
  right_block[0, 0] = 7.0;
  right_block[0, 1] = 8.0;
  right_block[1, 0] = 9.0;
  right_block[1, 1] = 10.0;
  right_block[2, 0] = 11.0;
  right_block[2, 1] = 12.0;

  auto result = contract<1, 0>(left, right);
  using Result = decltype(result);
  static_assert(std::same_as<typename Result::domain_type, Domain<BlockSpace>>);
  static_assert(std::same_as<typename Result::codomain_type, Codomain<BlockSpace>>);
  static_assert(std::same_as<typename Result::storage_policy, PackedSparseBlockStorage<>>);
  typename Result::key_type const result_key{{0, 0}};
  ASSERT_EQ(result.stored_block_count(), 1);
  EXPECT_EQ(result.domain().template space<0>(), rows);
  EXPECT_EQ(result.codomain().template space<0>(), columns);
  auto block = result.block(result_key);
  EXPECT_DOUBLE_EQ((block[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((block[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((block[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((block[1, 1]), 154.0);

  auto separate_result = contract<1, 0, SeparateSparseBlockStorage<>>(left, right);
  static_assert(std::same_as<typename decltype(separate_result)::storage_policy, SeparateSparseBlockStorage<>>);
  EXPECT_DOUBLE_EQ((separate_result.block(result_key)[1, 1]), 154.0);
}

TYPED_TEST(BlockTensorContractionTest, DoesNotCreateBlocksFromMismatchedSectors)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const external(sym, {{q0, 1}, {q1, 1}}, "external");
  BlockSpace const bond(sym, {{q0, 1}, {q1, 1}}, "bond");

  using Left = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, typename TypeParam::left_storage>;
  using Right = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, typename TypeParam::right_storage>;
  typename Left::key_type const q0_block{{0, 0}};
  typename Right::key_type const q1_block{{1, 1}};
  Left left(sym, Domain{external}, Codomain{bond}, {q0_block});
  Right right(sym, Domain{bond}, Codomain{external}, {q1_block});
  left.block(q0_block)[0, 0] = 2.0;
  right.block(q1_block)[0, 0] = 3.0;

  auto result = contract<1, 0>(left, right);

  EXPECT_EQ(result.stored_block_count(), 0);
  EXPECT_EQ(result.legal_block_count(), 2);
}

TYPED_TEST(BlockTensorContractionTest, SumsRepeatedLocalOccurrencesIntoRankZeroResult)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const bond(sym, {q0, q0}, "bond");

  using Left = BlockTensor<double, Domain<>, Codomain<LocalSpace>, typename TypeParam::left_storage>;
  using Right = BlockTensor<double, Domain<LocalSpace>, Codomain<>, typename TypeParam::right_storage>;
  typename Left::key_type const left0{{0}};
  typename Left::key_type const left1{{1}};
  typename Right::key_type const right0{{0}};
  typename Right::key_type const right1{{1}};
  Left left(sym, Domain<>{}, Codomain{bond}, {left0, left1});
  Right right(sym, Domain{bond}, Codomain<>{}, {right0, right1});
  left.block(left0)[] = 2.0;
  left.block(left1)[] = 3.0;
  right.block(right0)[] = 5.0;
  right.block(right1)[] = 7.0;

  auto result = contract<0, 0>(left, right);
  using Result = decltype(result);
  static_assert(Result::order() == 0);
  static_assert(Result::key_coordinate_count() == 0);
  static_assert(Result::dense_block_order() == 0);
  typename Result::key_type const result_key{};
  ASSERT_EQ(result.stored_block_count(), 1);
  EXPECT_DOUBLE_EQ(result.block(result_key)[], 31.0);
}

TYPED_TEST(BlockTensorContractionTest, PreservesPlanarExternalBoundaryAndRequiresExactBondValue)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const left_domain(sym, {q0, q0}, "left-domain");
  LocalSpace const left_output(sym, {q0}, "left-output");
  LocalSpace const bond(sym, {q0, q0}, "bond");
  LocalSpace const right_domain(sym, {q0, q0}, "right-domain");
  LocalSpace const right_output(sym, {q0}, "right-output");

  using Left =
      BlockTensor<double, Domain<LocalSpace>, Codomain<LocalSpace, LocalSpace>, typename TypeParam::left_storage>;
  using Right =
      BlockTensor<double, Domain<LocalSpace, LocalSpace>, Codomain<LocalSpace>, typename TypeParam::right_storage>;
  typename Left::key_type const left_key{{1, 0, 1}};
  typename Right::key_type const right_key{{1, 1, 0}};
  Left left(sym, Domain{left_domain}, Codomain{left_output, bond}, {left_key});
  Right right(sym, Domain{bond, right_domain}, Codomain{right_output}, {right_key});
  left.block(left_key)[] = 2.0;
  right.block(right_key)[] = 4.0;

  auto result = contract<2, 0>(left, right);
  using Result = decltype(result);
  static_assert(std::same_as<typename Result::domain_type, Domain<LocalSpace, LocalSpace>>);
  static_assert(std::same_as<typename Result::codomain_type, Codomain<LocalSpace, LocalSpace>>);
  typename Result::key_type const result_key{{1, 1, 0, 0}};
  ASSERT_EQ(result.stored_block_count(), 1);
  EXPECT_EQ(result.domain().template space<0>(), left_domain);
  EXPECT_EQ(result.domain().template space<1>(), right_domain);
  EXPECT_EQ(result.codomain().template space<0>(), left_output);
  EXPECT_EQ(result.codomain().template space<1>(), right_output);
  EXPECT_DOUBLE_EQ(result.block(result_key)[], 8.0);

  LocalSpace const mismatched_bond(sym, {q0, q0}, "other-bond");
  Right mismatched(sym, Domain{mismatched_bond, right_domain}, Codomain{right_output}, {right_key});
  EXPECT_THROW((contract<2, 0>(left, mismatched)), std::invalid_argument);
}

TYPED_TEST(BlockTensorContractionTest, ContractsAPairExposedByExplicitPermutation)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const input(sym, {q0}, "input");
  LocalSpace const bond(sym, {q0}, "bond");
  LocalSpace const retained(sym, {q0}, "retained");

  using Left =
      BlockTensor<double, Domain<LocalSpace>, Codomain<LocalSpace, LocalSpace>, typename TypeParam::left_storage>;
  using Right = BlockTensor<double, Domain<LocalSpace>, Codomain<>, typename TypeParam::right_storage>;
  typename Left::key_type const left_key{{0, 0, 0}};
  typename Right::key_type const right_key{{0}};
  Left left(sym, Domain{input}, Codomain{bond, retained}, {left_key});
  Right right(sym, Domain{bond}, Codomain<>{}, {right_key});
  left.block(left_key)[] = 3.0;
  right.block(right_key)[] = 5.0;

  auto exposed = permute<0, 2, 1>(left);
  auto result = contract<2, 0>(exposed, right);
  using Result = decltype(result);
  static_assert(std::same_as<typename Result::domain_type, Domain<LocalSpace>>);
  static_assert(std::same_as<typename Result::codomain_type, Codomain<LocalSpace>>);
  typename Result::key_type const result_key{{0, 0}};
  EXPECT_EQ(result.domain().template space<0>(), input);
  EXPECT_EQ(result.codomain().template space<0>(), retained);
  EXPECT_DOUBLE_EQ(result.block(result_key)[], 15.0);
}

TYPED_TEST(BlockTensorContractionTest, PlacesExternalDenseAxesInResultBoundaryOrder)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const left_domain(sym, {{q0, 2}}, "left-domain");
  BlockSpace const left_output(sym, {{q0, 2}}, "left-output");
  BlockSpace const bond(sym, {{q0, 2}}, "bond");
  BlockSpace const right_domain(sym, {{q0, 2}}, "right-domain");
  BlockSpace const right_output(sym, {{q0, 2}}, "right-output");

  using Left =
      BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace, BlockSpace>, typename TypeParam::left_storage>;
  using Right =
      BlockTensor<double, Domain<BlockSpace, BlockSpace>, Codomain<BlockSpace>, typename TypeParam::right_storage>;
  typename Left::key_type const left_key{{0, 0, 0}};
  typename Right::key_type const right_key{{0, 0, 0}};
  Left left(sym, Domain{left_domain}, Codomain{left_output, bond}, {left_key});
  Right right(sym, Domain{bond, right_domain}, Codomain{right_output}, {right_key});

  auto left_block = left.block(left_key);
  auto right_block = right.block(right_key);
  for (std::size_t m = 0; m < 2; ++m)
    for (std::size_t p = 0; p < 2; ++p)
      for (std::size_t k = 0; k < 2; ++k)
        left_block[m, p, k] = static_cast<double>(100 * m + 10 * p + k + 1);
  for (std::size_t k = 0; k < 2; ++k)
    for (std::size_t q = 0; q < 2; ++q)
      for (std::size_t n = 0; n < 2; ++n)
        right_block[k, q, n] = static_cast<double>(100 * k + 10 * q + n + 1);

  auto result = contract<2, 0>(left, right);
  using Result = decltype(result);
  static_assert(std::same_as<typename Result::domain_type, Domain<BlockSpace, BlockSpace>>);
  static_assert(std::same_as<typename Result::codomain_type, Codomain<BlockSpace, BlockSpace>>);
  typename Result::key_type const result_key{{0, 0, 0, 0}};
  auto block = result.block(result_key);
  for (std::size_t m = 0; m < 2; ++m)
    for (std::size_t q = 0; q < 2; ++q)
      for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t n = 0; n < 2; ++n)
        {
          double expected = 0.0;
          for (std::size_t k = 0; k < 2; ++k)
            expected += left_block[m, p, k] * right_block[k, q, n];
          EXPECT_DOUBLE_EQ((block[m, q, p, n]), expected);
        }
}

TEST(BlockTensorContractionTest, SupportsComplexRankZeroAccumulation)
{
  using Scalar = uni20::complex<double>;
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const bond(sym, {q0, q0}, "bond");
  using Left = BlockTensor<Scalar, Domain<>, Codomain<LocalSpace>, PackedSparseBlockStorage<>>;
  using Right = BlockTensor<Scalar, Domain<LocalSpace>, Codomain<>, PackedSparseBlockStorage<>>;
  typename Left::key_type const key0{{0}};
  typename Left::key_type const key1{{1}};
  Left left(sym, Domain<>{}, Codomain{bond}, {key0, key1});
  Right right(sym, Domain{bond}, Codomain<>{}, {key0, key1});
  left.block(key0)[] = Scalar{1.0, 2.0};
  left.block(key1)[] = Scalar{2.0, -1.0};
  right.block(key0)[] = Scalar{3.0, -1.0};
  right.block(key1)[] = Scalar{-1.0, 4.0};

  auto result = contract<0, 0>(left, right);
  EXPECT_EQ(result.block(typename decltype(result)::key_type{})[], (Scalar{7.0, 14.0}));
}
