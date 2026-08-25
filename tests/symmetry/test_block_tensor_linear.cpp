#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/core/math.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

using namespace uni20;

template <class Storage> class BlockTensorLinearTest : public ::testing::Test {};

using ImmediateLinearStorageTypes =
    ::testing::Types<SeparateSparseBlockStorage<>, ParallelSeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>;
TYPED_TEST_SUITE(BlockTensorLinearTest, ImmediateLinearStorageTypes);

TYPED_TEST(BlockTensorLinearTest, AppliesLinearOperationsWithSparseZeroSemantics)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const rows(sym, {q0, q0}, "rows");
  LocalSpace const columns(sym, {q0, q0}, "columns");
  using Tensor = BlockTensor<double, Domain<LocalSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const key00{{0, 0}};
  Key const key01{{0, 1}};
  Key const key10{{1, 0}};
  Key const key11{{1, 1}};

  Tensor lhs(sym, Domain{rows}, Codomain{columns}, {key00, key10});
  Tensor rhs(sym, Domain{rows}, Codomain{columns}, {key00, key01});
  Tensor output(sym, Domain{rows}, Codomain{columns}, {key00, key01, key10, key11});
  lhs.block(key00)[] = 2.0;
  lhs.block(key10)[] = 3.0;
  rhs.block(key00)[] = 5.0;
  rhs.block(key01)[] = 7.0;
  for (Key const& key : output.stored_keys())
    output.block(key)[] = 9.0;

  copy(output, lhs);
  EXPECT_DOUBLE_EQ(output.block(key00)[], 2.0);
  EXPECT_DOUBLE_EQ(output.block(key01)[], 0.0);
  EXPECT_DOUBLE_EQ(output.block(key10)[], 3.0);
  EXPECT_DOUBLE_EQ(output.block(key11)[], 0.0);

  assign_scale(output, 2.0, lhs);
  EXPECT_DOUBLE_EQ(output.block(key00)[], 4.0);
  EXPECT_DOUBLE_EQ(output.block(key01)[], 0.0);
  EXPECT_DOUBLE_EQ(output.block(key10)[], 6.0);
  EXPECT_DOUBLE_EQ(output.block(key11)[], 0.0);

  add(output, lhs, rhs);
  EXPECT_DOUBLE_EQ(output.block(key00)[], 7.0);
  EXPECT_DOUBLE_EQ(output.block(key01)[], 7.0);
  EXPECT_DOUBLE_EQ(output.block(key10)[], 3.0);
  EXPECT_DOUBLE_EQ(output.block(key11)[], 0.0);

  add_inplace(output, lhs);
  EXPECT_DOUBLE_EQ(output.block(key00)[], 9.0);
  EXPECT_DOUBLE_EQ(output.block(key01)[], 7.0);
  EXPECT_DOUBLE_EQ(output.block(key10)[], 6.0);
  EXPECT_DOUBLE_EQ(output.block(key11)[], 0.0);

  axpy(output, -2.0, rhs);
  EXPECT_DOUBLE_EQ(output.block(key00)[], -1.0);
  EXPECT_DOUBLE_EQ(output.block(key01)[], -7.0);
  EXPECT_DOUBLE_EQ(output.block(key10)[], 6.0);
  EXPECT_DOUBLE_EQ(output.block(key11)[], 0.0);

  scale(output, 0.5);
  EXPECT_DOUBLE_EQ(output.block(key00)[], -0.5);
  EXPECT_DOUBLE_EQ(output.block(key01)[], -3.5);
  EXPECT_DOUBLE_EQ(output.block(key10)[], 3.0);

  auto sum = add(lhs, rhs);
  static_assert(std::same_as<typename decltype(sum)::storage_policy, TypeParam>);
  ASSERT_EQ(sum.stored_block_count(), 3);
  EXPECT_EQ(sum.stored_keys()[0], key00);
  EXPECT_EQ(sum.stored_keys()[1], key01);
  EXPECT_EQ(sum.stored_keys()[2], key10);
  EXPECT_DOUBLE_EQ(sum.block(key00)[], 7.0);
  EXPECT_DOUBLE_EQ(sum.block(key01)[], 7.0);
  EXPECT_DOUBLE_EQ(sum.block(key10)[], 3.0);

  set_zero(output);
  for (Key const& key : output.stored_keys())
    EXPECT_DOUBLE_EQ(output.block(key)[], 0.0);
}

TYPED_TEST(BlockTensorLinearTest, RejectsStructuralChangesBeforeModifyingOutput)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const rows(sym, {q0, q0}, "rows");
  LocalSpace const columns(sym, {q0, q0}, "columns");
  using Tensor = BlockTensor<double, Domain<LocalSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const key00{{0, 0}};
  Key const key01{{0, 1}};

  Tensor output(sym, Domain{rows}, Codomain{columns}, {key00});
  Tensor input(sym, Domain{rows}, Codomain{columns}, {key01});
  output.block(key00)[] = 11.0;
  input.block(key01)[] = 3.0;
  EXPECT_THROW(copy(output, input), std::invalid_argument);
  EXPECT_DOUBLE_EQ(output.block(key00)[], 11.0);

  auto relabelled_columns = columns;
  relabelled_columns.set_label("other-columns");
  Tensor relabelled(sym, Domain{rows}, Codomain{relabelled_columns}, {key00});
  relabelled.block(key00)[] = 4.0;
  EXPECT_THROW(add_inplace(output, relabelled), std::invalid_argument);
  EXPECT_DOUBLE_EQ(output.block(key00)[], 11.0);
}

TYPED_TEST(BlockTensorLinearTest, ComputesComplexInnerProductsAndNormsAcrossDenseBlocks)
{
  using Scalar = uni20::complex<double>;
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const space(sym, {{q0, 2}, {q1, 1}}, "space");
  using Tensor = BlockTensor<Scalar, Domain<BlockSpace>, Codomain<BlockSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const key0{{0, 0}};
  Key const key1{{1, 1}};
  Tensor lhs(sym, Domain{space}, Codomain{space}, {key0, key1});
  Tensor rhs(sym, Domain{space}, Codomain{space}, {key0, key1});

  std::vector<Scalar> const lhs_values{{1.0, 2.0}, {3.0, -1.0}, {-2.0, 0.5}, {4.0, 0.0}, {5.0, -2.0}};
  std::vector<Scalar> const rhs_values{{2.0, -1.0}, {-1.0, 3.0}, {0.5, 2.0}, {3.0, 1.0}, {-2.0, 4.0}};
  auto lhs0 = lhs.block(key0);
  auto rhs0 = rhs.block(key0);
  std::size_t value = 0;
  for (std::size_t column = 0; column < 2; ++column)
    for (std::size_t row = 0; row < 2; ++row)
    {
      lhs0[row, column] = lhs_values[value];
      rhs0[row, column] = rhs_values[value++];
    }
  lhs.block(key1)[0, 0] = lhs_values[value];
  rhs.block(key1)[0, 0] = rhs_values[value];

  Scalar expected_inner{};
  double expected_squared_norm = 0.0;
  for (std::size_t index = 0; index < lhs_values.size(); ++index)
  {
    expected_inner += uni20::conj(lhs_values[index]) * rhs_values[index];
    expected_squared_norm += std::norm(lhs_values[index]);
  }

  EXPECT_NEAR(std::abs(inner_product_host(lhs, rhs) - expected_inner), 0.0, 1e-12);
  EXPECT_NEAR(norm_host(lhs), std::sqrt(expected_squared_norm), 1e-12);
  EXPECT_NEAR(std::abs(inner_product(lhs, rhs)[] - expected_inner), 0.0, 1e-12);
  EXPECT_NEAR(norm(lhs)[], std::sqrt(expected_squared_norm), 1e-12);

  Tensor rhs_sector_one(sym, Domain{space}, Codomain{space}, {key1});
  rhs_sector_one.block(key1)[0, 0] = rhs_values.back();
  EXPECT_NEAR(std::abs(inner_product_host(lhs, rhs_sector_one) - uni20::conj(lhs_values.back()) * rhs_values.back()),
              0.0, 1e-12);
}

TYPED_TEST(BlockTensorLinearTest, OperatesOnWritableAndReadableMappedViews)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const first(sym, {q0, q0}, "first");
  LocalSpace const second(sym, {q0, q0}, "second");
  using Tensor = BlockTensor<double, Domain<LocalSpace, LocalSpace>, Codomain<>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const source_key{{0, 1}};
  Key const permuted_key{{1, 0}};
  Tensor source(sym, Domain{first, second}, Codomain<>{}, {source_key});
  source.block(source_key)[] = 3.0;

  auto view = permute<1, 0>(source);
  Tensor output(sym, Domain{second, first}, Codomain<>{}, {permuted_key});
  copy(output, view);
  EXPECT_DOUBLE_EQ(output.block(permuted_key)[], 3.0);

  scale(view, 2.0);
  EXPECT_DOUBLE_EQ(source.block(source_key)[], 6.0);
  EXPECT_DOUBLE_EQ(norm_host(view), 6.0);
}

TYPED_TEST(BlockTensorLinearTest, PropagatesNonFiniteBlockReductionValues)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const rows(sym, {q0, q0}, "rows");
  LocalSpace const columns(sym, {q0, q0}, "columns");
  using Tensor = BlockTensor<double, Domain<LocalSpace>, Codomain<LocalSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Key const key00{{0, 0}};
  Key const key11{{1, 1}};
  Tensor lhs(sym, Domain{rows}, Codomain{columns}, {key00, key11});
  Tensor rhs(sym, Domain{rows}, Codomain{columns}, {key00, key11});
  lhs.block(key00)[] = std::numeric_limits<double>::infinity();
  lhs.block(key11)[] = 2.0;
  rhs.block(key00)[] = 1.0;
  rhs.block(key11)[] = 3.0;

  EXPECT_TRUE(std::isinf(inner_product_host(lhs, rhs)));
  EXPECT_TRUE(std::isinf(norm_host(lhs)));

  lhs.block(key00)[] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(norm_host(lhs)));
}

TEST(BlockTensorLinearTest, AsyncStorageSchedulesLinearUpdatesAndSupportsBlockingReductions)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  LocalSpace const rows(sym, {q0, q0}, "rows");
  LocalSpace const columns(sym, {q0, q0}, "columns");
  using Storage = AsyncSeparateSparseBlockStorage<>;
  using Tensor = BlockTensor<double, Domain<LocalSpace>, Codomain<LocalSpace>, Storage>;
  using Key = typename Tensor::key_type;
  Key const key00{{0, 0}};
  Key const key01{{0, 1}};
  Key const key10{{1, 0}};
  Tensor lhs(sym, Domain{rows}, Codomain{columns}, {key00, key10});
  Tensor rhs(sym, Domain{rows}, Codomain{columns}, {key00, key01});
  lhs.async_block(key00).unsafe_value_ref()[] = 2.0;
  lhs.async_block(key10).unsafe_value_ref()[] = 3.0;
  rhs.async_block(key00).unsafe_value_ref()[] = 5.0;
  rhs.async_block(key01).unsafe_value_ref()[] = 7.0;

  async::DebugScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  auto result = add(lhs, rhs);
  axpy(result, 2.0, lhs);
  scale(result, 0.5);

  EXPECT_DOUBLE_EQ(result.async_block(key00).get_wait(scheduler)[], 5.5);
  EXPECT_DOUBLE_EQ(result.async_block(key01).get_wait(scheduler)[], 3.5);
  EXPECT_DOUBLE_EQ(result.async_block(key10).get_wait(scheduler)[], 4.5);
  EXPECT_DOUBLE_EQ(inner_product_host(lhs, rhs), 10.0);
  EXPECT_DOUBLE_EQ(norm_host(lhs), std::sqrt(13.0));

  set_zero(result);
  scheduler.run_all();
  EXPECT_DOUBLE_EQ(result.async_block(key00).get_wait(scheduler)[], 0.0);
  EXPECT_DOUBLE_EQ(result.async_block(key01).get_wait(scheduler)[], 0.0);
  EXPECT_DOUBLE_EQ(result.async_block(key10).get_wait(scheduler)[], 0.0);
}
