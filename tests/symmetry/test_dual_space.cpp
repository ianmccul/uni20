#include <uni20/symmetry/block_tensor_space_traits.hpp>
#include <uni20/symmetry/dual_space.hpp>

#include <gtest/gtest.h>

#include <ranges>
#include <type_traits>
#include <vector>

using namespace uni20;

static_assert(!DualSpace<LocalSpace>);
static_assert(DualSpace<Dual<LocalSpace>>);
static_assert(Space<Dual<DenseSpace>>);
static_assert(SymmetrySpace<Dual<LocalSpace>>);
static_assert(!SymmetrySpace<Dual<DenseSpace>>);
static_assert(std::same_as<primal_space_t<Dual<BlockSpace>>, BlockSpace>);
static_assert(BlockTensorSpaceTraits<Dual<LocalSpace>>::has_block_coordinate);
static_assert(!BlockTensorSpaceTraits<Dual<LocalSpace>>::has_dense_axis);
static_assert(BlockTensorSpaceTraits<Dual<BlockSpace>>::has_block_coordinate);
static_assert(BlockTensorSpaceTraits<Dual<BlockSpace>>::has_dense_axis);
static_assert(std::ranges::input_range<Dual<LocalSpace>>);
static_assert(std::ranges::input_range<Dual<BlockSpace>>);

TEST(DualSpaceTest, LocalSpaceDualizesQuantumNumbersWithoutChangingOccurrences)
{
  Symmetry const sym{"N:U(1)"};
  auto const q1 = make_qnum(sym, {{"N", 1}});
  auto const q_minus_2 = make_qnum(sym, {{"N", -2}});
  LocalSpace const primal(sym, {q1, q_minus_2, q1}, "physical");
  auto dual_basis = dual(primal);

  EXPECT_EQ(dual_basis.size(), 3);
  EXPECT_EQ(dual_basis.label(), "physical");
  EXPECT_EQ(dual_basis[0], dual(q1));
  EXPECT_EQ(dual_basis[1], dual(q_minus_2));
  EXPECT_EQ(dual_basis[2], dual(q1));

  auto const qnums = dual_basis.qnums();
  std::vector<QNum> const observed(qnums.begin(), qnums.end());
  EXPECT_EQ(observed, (std::vector<QNum>{dual(q1), dual(q_minus_2), dual(q1)}));
  std::vector<QNum> iterated;
  for (QNum const q : dual_basis)
    iterated.push_back(q);
  EXPECT_EQ(iterated, observed);

  dual_basis.set_label("dual-physical");
  EXPECT_EQ(dual_basis.label(), "dual-physical");
  EXPECT_EQ(dual(std::move(dual_basis)).label(), "dual-physical");
}

TEST(DualSpaceTest, BlockSpacePreservesSectorIndicesAndDimensions)
{
  Symmetry const sym{"N:U(1)"};
  auto const q_minus_1 = make_qnum(sym, {{"N", -1}});
  auto const q2 = make_qnum(sym, {{"N", 2}});
  BlockSpace const primal(sym, {{q_minus_1, 3}, {q2, 5}}, "bond");
  auto const dual_basis = dual(primal);

  ASSERT_EQ(dual_basis.size(), 2);
  EXPECT_EQ(dual_basis[0], (BlockSector{dual(primal[0].q), 3}));
  EXPECT_EQ(dual_basis[1], (BlockSector{dual(primal[1].q), 5}));
  EXPECT_TRUE(dual_basis.contains(dual(primal[0].q)));
  EXPECT_FALSE(dual_basis.contains(primal[0].q));
  EXPECT_THROW(dual_basis.contains(QNum{}), std::invalid_argument);
  Symmetry const other_sym{"M:U(1)"};
  EXPECT_THROW(dual_basis.contains(QNum::identity(other_sym)), std::invalid_argument);

  auto const sectors = dual_basis.sectors();
  std::vector<BlockSector> const observed(sectors.begin(), sectors.end());
  EXPECT_EQ(observed[0], dual_basis[0]);
  EXPECT_EQ(observed[1], dual_basis[1]);
  EXPECT_EQ(dual(dual_basis), primal);
}

TEST(DualSpaceTest, FixedIrrepAndDenseSpacesForwardNonChargeStructure)
{
  Symmetry const sym{"N:U(1)"};
  auto const q3 = make_qnum(sym, {{"N", 3}});
  QNumSpace const irrep(q3, "operator");
  DenseSpace const batch(7, "batch");

  auto const dual_irrep = dual(irrep);
  auto const dual_batch = dual(batch);
  EXPECT_EQ(dual_irrep.qnum(), dual(q3));
  EXPECT_EQ(dual_irrep.symmetry(), sym);
  EXPECT_EQ(dual_batch.extent(), 7);
  EXPECT_EQ(dual_batch.label(), "batch");
}
