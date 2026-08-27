#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/tensor/transform.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

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

TEST(BlockTensorTbbTest, ParallelPackedStorageFactorizesIndependentChargeSectors)
{
  Symmetry const symmetry{"N:U(1)"};
  std::vector<BlockSector> sectors;
  for (int charge = 0; charge < 8; ++charge)
    sectors.push_back({.q = make_qnum(symmetry, {{"N", charge}}), .dim = 24});
  BlockSpace const input(symmetry, sectors, "input");
  BlockSpace const output(symmetry, sectors, "output");

  using Storage = ParallelPackedSparseBlockStorage<>;
  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;
  using Key = typename Matrix::key_type;
  std::vector<Key> keys;
  for (std::size_t sector = 0; sector < sectors.size(); ++sector)
    keys.emplace_back(std::array<std::size_t, 2>{sector, sector});
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, std::move(keys));
  for (std::size_t sector = 0; sector < sectors.size(); ++sector)
  {
    auto block = matrix.block(Key{{sector, sector}});
    uni20::fill(block, 0.0);
    for (uni20::index_type index = 0; index < block.extent(0); ++index)
      block[index, index] = static_cast<double>(sector + 1);
  }

  async::TbbScheduler scheduler{4};
  async::ScopedScheduler scoped(&scheduler);
  auto decomposition = block_svd(matrix);

  ASSERT_EQ(decomposition.sectors().size(), sectors.size());
  ASSERT_EQ(decomposition.spectrum().size(), sectors.size() * 24);
  EXPECT_DOUBLE_EQ(decomposition.spectrum().front().singular_value, 8.0);
  EXPECT_EQ(decomposition.sectors().front().charge, sectors.front().q);
  EXPECT_EQ(decomposition.sectors().back().charge, sectors.back().q);
}
