#include <uni20/symmetry/block_space.hpp>
#include <uni20/symmetry/dense_space.hpp>
#include <uni20/symmetry/irregular_space.hpp>
#include <uni20/symmetry/local_space.hpp>
#include <uni20/symmetry/qnum_space.hpp>
#include <uni20/symmetry/space.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using namespace uni20;

static_assert(Space<BlockSpace>);
static_assert(Space<IrregularSpace>);
static_assert(Space<LocalSpace>);
static_assert(Space<QNumSpace>);
static_assert(Space<DenseSpace>);

static_assert(SymmetrySpace<BlockSpace>);
static_assert(SymmetrySpace<IrregularSpace>);
static_assert(SymmetrySpace<LocalSpace>);
static_assert(SymmetrySpace<QNumSpace>);
static_assert(!SymmetrySpace<DenseSpace>);

TEST(LocalSpaceTest, PreservesOrderedRepeatedStatesAndExplicitSymmetry)
{
  Symmetry const sym{"N:U(1)"};
  std::vector<QNum> const qnums{
      make_qnum(sym, {{"N", 1}}),
      make_qnum(sym, {{"N", -1}}),
      make_qnum(sym, {{"N", 1}}),
  };
  LocalSpace space(sym, qnums, "spin-1/2");

  ASSERT_EQ(space.size(), 3);
  EXPECT_EQ(space.symmetry(), sym);
  EXPECT_EQ(space.label(), "spin-1/2");
  EXPECT_EQ(space[0], qnums[0]);
  EXPECT_EQ(space[1], qnums[1]);
  EXPECT_EQ(space[2], qnums[2]);

  auto relabeled = space;
  relabeled.set_label("impurity");
  EXPECT_NE(relabeled, space);
  EXPECT_TRUE(std::ranges::equal(relabeled.qnums(), space.qnums()));
}

TEST(LocalSpaceTest, EmptySpaceRetainsSymmetryAndRejectsMismatchedStates)
{
  Symmetry const sym{"N:U(1)"};
  Symmetry const other{"Sz:U(1)"};

  LocalSpace const empty(sym);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.symmetry(), sym);

  EXPECT_THROW(static_cast<void>(LocalSpace(Symmetry{})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(LocalSpace(sym, {make_qnum(other, {{"Sz", 0}})})), std::invalid_argument);
}

TEST(LocalSpaceTest, RegularizationPreservesLabelAndRecordsOffsets)
{
  Symmetry const sym{"N:U(1)"};
  LocalSpace const space(sym,
                         {
                             make_qnum(sym, {{"N", 1}}),
                             make_qnum(sym, {{"N", -1}}),
                             make_qnum(sym, {{"N", 1}}),
                         },
                         "physical");

  auto const regularized = regularize(space);

  EXPECT_EQ(regularized.original, space);
  EXPECT_EQ(regularized.regular.label(), "physical");
  ASSERT_EQ(regularized.regular.size(), 2);
  EXPECT_EQ(u1_component(regularized.regular[0].q, "N"), U1{1});
  EXPECT_EQ(regularized.regular[0].dim, 2);
  EXPECT_EQ(u1_component(regularized.regular[1].q, "N"), U1{-1});
  EXPECT_EQ(regularized.regular[1].dim, 1);
  EXPECT_EQ(regularized.block_index, (std::vector<std::size_t>{0, 1, 0}));
  EXPECT_EQ(regularized.block_offset, (std::vector<std::size_t>{0, 0, 1}));
  EXPECT_EQ(to_block_space(space), regularized.regular);
}

TEST(QNumSpaceTest, CarriesOneIrrepAndMutableLabel)
{
  Symmetry const sym{"N:U(1)"};
  QNum const q = make_qnum(sym, {{"N", 2}});
  QNumSpace space(q, "charge-two");

  EXPECT_EQ(space.symmetry(), sym);
  EXPECT_EQ(space.qnum(), q);
  EXPECT_EQ(space.label(), "charge-two");

  auto relabeled = space;
  relabeled.set_label("operator");
  EXPECT_NE(relabeled, space);
  EXPECT_EQ(relabeled.qnum(), space.qnum());

  EXPECT_THROW(static_cast<void>(QNumSpace(QNum{})), std::invalid_argument);
}

TEST(DenseSpaceTest, CarriesOnlyExtentAndLabel)
{
  DenseSpace space(4, "batch");
  EXPECT_EQ(space.extent(), 4);
  EXPECT_FALSE(space.empty());
  EXPECT_EQ(space.label(), "batch");

  auto relabeled = space;
  relabeled.set_label("vectors");
  EXPECT_NE(relabeled, space);
  EXPECT_EQ(relabeled.extent(), space.extent());

  DenseSpace const empty(0);
  EXPECT_TRUE(empty.empty());
}

TEST(IrregularSpaceTest, PreservesOrderedRepeatedBlocks)
{
  Symmetry const sym{"N:U(1)"};
  IrregularSpace space(sym,
                       {
                           {make_qnum(sym, {{"N", 1}}), 2},
                           {make_qnum(sym, {{"N", -1}}), 1},
                           {make_qnum(sym, {{"N", 1}}), 3},
                           {make_qnum(sym, {{"N", 0}}), 4},
                       },
                       "projected");

  ASSERT_EQ(space.size(), 4);
  EXPECT_EQ(space.total_dim(), 10);
  EXPECT_EQ(space.label(), "projected");
  EXPECT_EQ(u1_component(space[0].q, "N"), U1{1});
  EXPECT_EQ(space[0].dim, 2);
  EXPECT_EQ(u1_component(space[1].q, "N"), U1{-1});
  EXPECT_EQ(space[1].dim, 1);
  EXPECT_EQ(u1_component(space[2].q, "N"), U1{1});
  EXPECT_EQ(space[2].dim, 3);
  EXPECT_EQ(u1_component(space[3].q, "N"), U1{0});
  EXPECT_EQ(space[3].dim, 4);

  auto relabeled = space;
  relabeled.set_label("reduced");
  EXPECT_NE(relabeled, space);
  EXPECT_TRUE(std::ranges::equal(relabeled.sectors(), space.sectors()));
}

TEST(IrregularSpaceTest, ValidatesSymmetryAndPositiveDimensions)
{
  Symmetry const sym{"N:U(1)"};
  Symmetry const other{"Sz:U(1)"};

  IrregularSpace const empty(sym);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.symmetry(), sym);

  EXPECT_THROW(static_cast<void>(IrregularSpace(Symmetry{})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(IrregularSpace(sym, {{make_qnum(other, {{"Sz", 0}}), 1}})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(IrregularSpace(sym, {{make_qnum(sym, {{"N", 0}}), 0}})), std::invalid_argument);
}

TEST(IrregularSpaceTest, RegularizationCoalescesBlocksAndRecordsRanges)
{
  Symmetry const sym{"N:U(1)"};
  IrregularSpace const space(sym,
                             {
                                 {make_qnum(sym, {{"N", 1}}), 2},
                                 {make_qnum(sym, {{"N", -1}}), 1},
                                 {make_qnum(sym, {{"N", 1}}), 3},
                                 {make_qnum(sym, {{"N", 0}}), 4},
                             },
                             "projected");

  auto const regularized = regularize(space);

  EXPECT_EQ(regularized.original, space);
  EXPECT_EQ(regularized.regular.label(), "projected");
  ASSERT_EQ(regularized.regular.size(), 3);
  EXPECT_EQ(u1_component(regularized.regular[0].q, "N"), U1{0});
  EXPECT_EQ(regularized.regular[0].dim, 4);
  EXPECT_EQ(u1_component(regularized.regular[1].q, "N"), U1{1});
  EXPECT_EQ(regularized.regular[1].dim, 5);
  EXPECT_EQ(u1_component(regularized.regular[2].q, "N"), U1{-1});
  EXPECT_EQ(regularized.regular[2].dim, 1);

  EXPECT_EQ(regularized.block_mapping, (std::vector<BlockRangeMapping>{
                                           {.block_index = 1, .first = 0, .last = 2},
                                           {.block_index = 2, .first = 0, .last = 1},
                                           {.block_index = 1, .first = 2, .last = 5},
                                           {.block_index = 0, .first = 0, .last = 4},
                                       }));
  EXPECT_EQ(regularized.block_mapping[2].size(), 3);
  EXPECT_EQ(to_block_space(space), regularized.regular);
}
