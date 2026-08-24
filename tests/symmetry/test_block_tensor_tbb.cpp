#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/symmetry/block_tensor_contract.hpp>

#include <gtest/gtest.h>

#include <concepts>

using namespace uni20;

TEST(BlockTensorTbbTest, ParallelSeparateStorageContractsIndependentOutputBlocks)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q1 = make_qnum(symmetry, {{"N", 1}});
  BlockSpace const external(symmetry, {{q0, 1}, {q1, 1}}, "external");
  LocalSpace const bond(symmetry, {q0, q0, q1, q1}, "bond");
  using InputStorage = SeparateSparseBlockStorage<>;
  using OutputStorage = ParallelSeparateSparseBlockStorage<>;
  using Left = BlockTensor<double, Domain<BlockSpace>, Codomain<LocalSpace>, InputStorage>;
  using Right = BlockTensor<double, Domain<LocalSpace>, Codomain<BlockSpace>, InputStorage>;
  typename Left::key_type const left_key00{{0, 0}};
  typename Left::key_type const left_key01{{0, 1}};
  typename Left::key_type const left_key12{{1, 2}};
  typename Left::key_type const left_key13{{1, 3}};
  typename Right::key_type const right_key00{{0, 0}};
  typename Right::key_type const right_key10{{1, 0}};
  typename Right::key_type const right_key21{{2, 1}};
  typename Right::key_type const right_key31{{3, 1}};
  Left left(symmetry, Domain{external}, Codomain{bond}, {left_key00, left_key01, left_key12, left_key13});
  Right right(symmetry, Domain{bond}, Codomain{external}, {right_key00, right_key10, right_key21, right_key31});

  left.block(left_key00)[0] = 2.0;
  left.block(left_key01)[0] = 3.0;
  left.block(left_key12)[0] = 5.0;
  left.block(left_key13)[0] = 7.0;
  right.block(right_key00)[0] = 11.0;
  right.block(right_key10)[0] = 13.0;
  right.block(right_key21)[0] = 17.0;
  right.block(right_key31)[0] = 19.0;

  async::TbbScheduler scheduler{4};
  async::ScopedScheduler scoped(&scheduler);
  auto result = contract<1, 0, OutputStorage>(left, right);
  static_assert(std::same_as<typename decltype(result)::storage_policy, OutputStorage>);

  EXPECT_DOUBLE_EQ((result.block(typename decltype(result)::key_type{{0, 0}})[0, 0]), 61.0);
  EXPECT_DOUBLE_EQ((result.block(typename decltype(result)::key_type{{1, 1}})[0, 0]), 218.0);
}
