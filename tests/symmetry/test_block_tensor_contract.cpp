#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>

#include <uni20/async/debug_scheduler.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace uni20;

namespace
{
class RecordingBatchScheduler : public async::DebugScheduler {
  public:
    std::size_t batch_calls = 0;
    std::vector<std::size_t> batch_sizes;

  private:
    void execute_batch_impl(async::LightweightTaskBatch const& batch) override
    {
      ++batch_calls;
      batch_sizes.push_back(batch.size());
      for (std::size_t index = 0; index < batch.size(); ++index)
        batch(index);
    }
};

template <class T> async::AsyncTask publish_block(async::WriteBuffer<T> output, T value)
{
  co_await output = std::move(value);
  co_return;
}

template <class T> async::AsyncTask fail_block(async::WriteBuffer<T> output)
{
  static_cast<void>(output);
  throw std::runtime_error("deliberate BlockTensor block failure");
  co_return;
}
} // namespace

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

TYPED_TEST(BlockTensorContractionTest, ContractsAdjacentFactorGroupWithMultipleDenseAxes)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const rows(sym, {{q0, 2}}, "rows");
  BlockSpace const first_bond(sym, {{q0, 2}}, "first-bond");
  BlockSpace const second_bond(sym, {{q0, 3}}, "second-bond");
  BlockSpace const columns(sym, {{q0, 2}}, "columns");

  using Left =
      BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace, BlockSpace>, typename TypeParam::left_storage>;
  using Right =
      BlockTensor<double, Domain<BlockSpace, BlockSpace>, Codomain<BlockSpace>, typename TypeParam::right_storage>;
  typename Left::key_type const left_key{{0, 0, 0}};
  typename Right::key_type const right_key{{0, 0, 0}};
  Left left(sym, Domain{rows}, Codomain{first_bond, second_bond}, {left_key});
  Right right(sym, Domain{first_bond, second_bond}, Codomain{columns}, {right_key});

  auto left_block = left.block(left_key);
  auto right_block = right.block(right_key);
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t first = 0; first < 2; ++first)
      for (std::size_t second = 0; second < 3; ++second)
      {
        left_block[row, first, second] = static_cast<double>(20 * row + 4 * first + second + 1);
        for (std::size_t column = 0; column < 2; ++column)
          right_block[first, second, column] = static_cast<double>(10 * first + 3 * second + column + 1);
      }

  auto result = contract_adjacent<2>(left, right);
  using Result = decltype(result);
  static_assert(std::same_as<typename Result::domain_type, Domain<BlockSpace>>);
  static_assert(std::same_as<typename Result::codomain_type, Codomain<BlockSpace>>);
  typename Result::key_type const result_key{{0, 0}};
  auto result_block = result.block(result_key);
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t column = 0; column < 2; ++column)
    {
      double expected = 0.0;
      for (std::size_t first = 0; first < 2; ++first)
        for (std::size_t second = 0; second < 3; ++second)
          expected += left_block[row, first, second] * right_block[first, second, column];
      EXPECT_DOUBLE_EQ((result_block[row, column]), expected);
    }

  Result fixed(sym, Domain{rows}, Codomain{columns}, {result_key});
  fixed.block(result_key)[0, 0] = -1.0;
  contract_adjacent<2>(fixed, left, right);
  EXPECT_DOUBLE_EQ((fixed.block(result_key)[0, 0]), (result_block[0, 0]));
}

TEST(BlockTensorContractionTest, ContractsRealDiagonalBlocksWithComplexDenseBlocks)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const rows(sym, {{q0, 2}}, "rows");
  BlockSpace const bond(sym, {{q0, 2}}, "bond");
  BlockSpace const columns(sym, {{q0, 2}}, "columns");
  using Diagonal = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedDiagonalBlockStorage<>>;
  using Scalar = uni20::complex<double>;
  using Dense = BlockTensor<Scalar, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  typename Diagonal::key_type const diagonal_key{{0, 0}};
  typename Dense::key_type const dense_key{{0, 0}};
  Diagonal diagonal(sym, Domain{rows}, Codomain{bond}, {diagonal_key});
  Dense dense(sym, Domain{bond}, Codomain{columns}, {dense_key});
  auto diagonal_values = diagonal.diagonal_values(diagonal_key);
  diagonal_values[0] = 2.0;
  diagonal_values[1] = 3.0;
  auto dense_block = dense.block(dense_key);
  dense_block[0, 0] = Scalar{1.0, 1.0};
  dense_block[0, 1] = Scalar{2.0, -1.0};
  dense_block[1, 0] = Scalar{-1.0, 0.5};
  dense_block[1, 1] = Scalar{4.0, 2.0};

  auto result = contract<1, 0>(diagonal, dense);
  static_assert(std::same_as<typename decltype(result)::value_type, Scalar>);
  typename decltype(result)::key_type const result_key{{0, 0}};
  auto block = result.block(result_key);
  EXPECT_EQ((block[0, 0]), Scalar(2.0, 2.0));
  EXPECT_EQ((block[0, 1]), Scalar(4.0, -2.0));
  EXPECT_EQ((block[1, 0]), Scalar(-3.0, 1.5));
  EXPECT_EQ((block[1, 1]), Scalar(12.0, 6.0));
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

TEST(BlockTensorContractionTest, ParallelSeparateStorageBatchesByOutputBlock)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const external(sym, {{q0, 1}, {q1, 1}}, "external");
  LocalSpace const bond(sym, {q0, q0, q1, q1}, "bond");
  using InputStorage = SeparateSparseBlockStorage<>;
  using OutputStorage = ParallelSeparateSparseBlockStorage<>;
  using Left = BlockTensor<double, Domain<BlockSpace>, Codomain<LocalSpace>, InputStorage>;
  using Right = BlockTensor<double, Domain<LocalSpace>, Codomain<BlockSpace>, InputStorage>;
  typename Left::key_type const key00{{0, 0}};
  typename Left::key_type const key01{{0, 1}};
  typename Left::key_type const key12{{1, 2}};
  typename Left::key_type const key13{{1, 3}};
  typename Right::key_type const right_key00{{0, 0}};
  typename Right::key_type const right_key10{{1, 0}};
  typename Right::key_type const right_key21{{2, 1}};
  typename Right::key_type const right_key31{{3, 1}};
  Left left(sym, Domain{external}, Codomain{bond}, {key00, key01, key12, key13});
  Right right(sym, Domain{bond}, Codomain{external}, {right_key00, right_key10, right_key21, right_key31});

  left.block(key00)[0] = 2.0;
  left.block(key01)[0] = 3.0;
  left.block(key12)[0] = 5.0;
  left.block(key13)[0] = 7.0;
  right.block(right_key00)[0] = 11.0;
  right.block(right_key10)[0] = 13.0;
  right.block(right_key21)[0] = 17.0;
  right.block(right_key31)[0] = 19.0;

  RecordingBatchScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  auto result = contract<1, 0, OutputStorage>(left, right);
  static_assert(std::same_as<typename decltype(result)::storage_policy, OutputStorage>);

  EXPECT_EQ(scheduler.batch_calls, 1);
  EXPECT_EQ(scheduler.batch_sizes, (std::vector<std::size_t>{2}));
  EXPECT_DOUBLE_EQ((result.block(typename decltype(result)::key_type{{0, 0}})[0, 0]), 61.0);
  EXPECT_DOUBLE_EQ((result.block(typename decltype(result)::key_type{{1, 1}})[0, 0]), 218.0);
}

TEST(BlockTensorContractionTest, OverwritesFixedOutputAndPreflightsItsSparseStructure)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const external(sym, {{q0, 1}, {q1, 1}}, "external");
  BlockSpace const bond(sym, {{q0, 1}, {q1, 1}}, "bond");
  using Input = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, SeparateSparseBlockStorage<>>;
  using Output = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, ParallelSeparateSparseBlockStorage<>>;
  typename Input::key_type const key0{{0, 0}};
  typename Input::key_type const key1{{1, 1}};
  Input left(sym, Domain{external}, Codomain{bond}, {key0});
  Input right(sym, Domain{bond}, Codomain{external}, {key0});
  Output output(sym, Domain{external}, Codomain{external}, {key0, key1});
  left.block(key0)[0, 0] = 2.0;
  right.block(key0)[0, 0] = 3.0;
  output.block(key0)[0, 0] = 17.0;
  output.block(key1)[0, 0] = 19.0;

  RecordingBatchScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  contract<1, 0>(output, left, right);

  EXPECT_DOUBLE_EQ((output.block(key0)[0, 0]), 6.0);
  EXPECT_DOUBLE_EQ((output.block(key1)[0, 0]), 0.0);
  EXPECT_EQ(scheduler.batch_calls, 2);
  EXPECT_EQ(scheduler.batch_sizes, (std::vector<std::size_t>{1, 1}));

  Output exact(sym, Domain{external}, Codomain{external}, {key0});
  exact.block(key0)[0, 0] = 29.0;
  contract<1, 0>(exact, left, right);
  EXPECT_DOUBLE_EQ((exact.block(key0)[0, 0]), 6.0);
  EXPECT_EQ(scheduler.batch_calls, 3);
  EXPECT_EQ(scheduler.batch_sizes, (std::vector<std::size_t>{1, 1, 1}));

  Output incomplete(sym, Domain{external}, Codomain{external}, {key1});
  incomplete.block(key1)[0, 0] = 23.0;
  EXPECT_THROW((contract<1, 0>(incomplete, left, right)), std::invalid_argument);
  EXPECT_DOUBLE_EQ((incomplete.block(key1)[0, 0]), 23.0);
  EXPECT_THROW((contract<1, 0>(left, left, right)), std::invalid_argument);
  EXPECT_DOUBLE_EQ((left.block(key0)[0, 0]), 2.0);
}

TEST(BlockTensorContractionTest, AsyncStorageSchedulesOneMatrixProductPerDenseBlock)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const rows(sym, {{q0, 2}}, "rows");
  BlockSpace const bond(sym, {{q0, 3}}, "bond");
  BlockSpace const columns(sym, {{q0, 2}}, "columns");
  using Storage = AsyncSeparateSparseBlockStorage<>;
  using Left = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  using Right = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  typename Left::key_type const key{{0, 0}};
  Left left(sym, Domain{rows}, Codomain{bond}, {key});
  Right right(sym, Domain{bond}, Codomain{columns}, {key});
  auto& left_block = left.async_block(key).unsafe_value_ref();
  auto& right_block = right.async_block(key).unsafe_value_ref();
  left_block[0, 0] = 1.0;
  left_block[0, 1] = 2.0;
  left_block[0, 2] = 3.0;
  left_block[1, 0] = 4.0;
  left_block[1, 1] = 5.0;
  left_block[1, 2] = 6.0;
  right_block[0, 0] = 7.0;
  right_block[0, 1] = 8.0;
  right_block[1, 0] = 9.0;
  right_block[1, 1] = 10.0;
  right_block[2, 0] = 11.0;
  right_block[2, 1] = 12.0;

  async::DebugScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  auto result = contract<1, 0>(left, right);
  static_assert(std::same_as<typename decltype(result)::storage_policy, Storage>);
  auto const& block = result.async_block(typename decltype(result)::key_type{{0, 0}}).get_wait(scheduler);
  EXPECT_DOUBLE_EQ((block[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((block[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((block[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((block[1, 1]), 154.0);
}

TEST(BlockTensorContractionTest, AsyncStorageOverwritesFixedOutputInEpochOrder)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  BlockSpace const rows(sym, {{q0, 1}}, "rows");
  BlockSpace const bond(sym, {{q0, 1}}, "bond");
  using Storage = AsyncSeparateSparseBlockStorage<>;
  using Tensor = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  typename Tensor::key_type const key{{0, 0}};
  Tensor left(sym, Domain{rows}, Codomain{bond}, {key});
  Tensor right(sym, Domain{bond}, Codomain{rows}, {key});
  Tensor output(sym, Domain{rows}, Codomain{rows}, {key});
  left.async_block(key).unsafe_value_ref()[0, 0] = 5.0;
  right.async_block(key).unsafe_value_ref()[0, 0] = 7.0;
  output.async_block(key).unsafe_value_ref()[0, 0] = 11.0;

  async::DebugScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  contract<1, 0>(output, left, right);

  EXPECT_DOUBLE_EQ((output.async_block(key).get_wait(scheduler)[0, 0]), 35.0);
}

TEST(BlockTensorContractionTest, AsyncBlocksProgressIndependentlyAcrossOutputSectors)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const external(sym, {{q0, 1}, {q1, 1}}, "external");
  BlockSpace const bond(sym, {{q0, 1}, {q1, 1}}, "bond");
  using Storage = AsyncSeparateSparseBlockStorage<>;
  using Left = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  using Right = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  typename Left::key_type const key0{{0, 0}};
  typename Left::key_type const key1{{1, 1}};
  Left left(sym, Domain{external}, Codomain{bond}, {key0, key1});
  Right right(sym, Domain{bond}, Codomain{external}, {key0, key1});
  left.async_block(key0).unsafe_value_ref()[0, 0] = 2.0;
  right.async_block(key0).unsafe_value_ref()[0, 0] = 3.0;
  right.async_block(key1).unsafe_value_ref()[0, 0] = 7.0;

  async::DebugScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  auto blocked_input = left.async_block(key1).write();
  auto result = contract<1, 0>(left, right);
  typename decltype(result)::key_type const result0{{0, 0}};
  typename decltype(result)::key_type const result1{{1, 1}};
  auto ready_reader = result.async_block(result0).read();
  auto blocked_reader = result.async_block(result1).read();

  scheduler.run();

  EXPECT_TRUE(ready_reader.await_ready());
  EXPECT_FALSE(blocked_reader.await_ready());
  EXPECT_NE(&result.async_block(result0).queue(), &result.async_block(result1).queue());
  EXPECT_DOUBLE_EQ((ready_reader.get_wait(scheduler)[0, 0]), 6.0);

  using block_type = typename Left::storage_type::block_value_type;
  block_type delayed(1, 1);
  delayed[0, 0] = 5.0;
  scheduler.schedule(publish_block(std::move(blocked_input), std::move(delayed)));
  scheduler.run_all();
  EXPECT_DOUBLE_EQ((blocked_reader.get_wait(scheduler)[0, 0]), 35.0);
}

TEST(BlockTensorContractionTest, AsyncInputFailurePropagatesOnlyToDependentOutputBlock)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const external(sym, {{q0, 1}, {q1, 1}}, "external");
  BlockSpace const bond(sym, {{q0, 1}, {q1, 1}}, "bond");
  using Storage = AsyncSeparateSparseBlockStorage<>;
  using Left = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  using Right = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  typename Left::key_type const key0{{0, 0}};
  typename Left::key_type const key1{{1, 1}};
  Left left(sym, Domain{external}, Codomain{bond}, {key0, key1});
  Right right(sym, Domain{bond}, Codomain{external}, {key0, key1});
  left.async_block(key0).unsafe_value_ref()[0, 0] = 2.0;
  right.async_block(key0).unsafe_value_ref()[0, 0] = 3.0;
  right.async_block(key1).unsafe_value_ref()[0, 0] = 7.0;

  async::DebugScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  scheduler.schedule(fail_block(left.async_block(key1).write()));
  auto result = contract<1, 0>(left, right);
  typename decltype(result)::key_type const result0{{0, 0}};
  typename decltype(result)::key_type const result1{{1, 1}};
  scheduler.run_all();

  EXPECT_DOUBLE_EQ((result.async_block(result0).get_wait(scheduler)[0, 0]), 6.0);
  EXPECT_THROW(static_cast<void>(result.async_block(result1).get_wait(scheduler)), std::runtime_error);
}
