#include <uni20/symmetry/block_space.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace uni20;

TEST(BlockSpaceTest, EnforcesSharedSymmetryAndTracksDimensions)
{
  Symmetry const sym{"N:U(1)"};
  Symmetry const other{"Sz:U(1)"};

  BlockSpace space(sym,
                   {
                       {make_qnum(sym, {{"N", 1}}), 2},
                       {make_qnum(sym, {{"N", -1}}), 3},
                   },
                   "bond");

  EXPECT_EQ(space.size(), 2);
  EXPECT_EQ(space.total_dim(), 5);
  EXPECT_FALSE(space.empty());
  EXPECT_EQ(space.label(), "bond");
  EXPECT_TRUE(space.contains(make_qnum(sym, {{"N", 1}})));
  EXPECT_FALSE(space.contains(make_qnum(sym, {{"N", 0}})));

  EXPECT_THROW(static_cast<void>(BlockSpace(sym, {{make_qnum(other, {{"Sz", 0}}), 1}})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(BlockSpace(sym, {{QNum{}, 1}})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(BlockSpace(sym, {{make_qnum(sym, {{"N", 0}}), 0}})), std::invalid_argument);
}

TEST(BlockSpaceTest, CanonicalizesSectorsAndRejectsDuplicates)
{
  Symmetry const sym{"N:U(1)"};

  std::vector<BlockSector> const sectors{
      {make_qnum(sym, {{"N", -1}}), 3},
      {make_qnum(sym, {{"N", 0}}), 1},
      {make_qnum(sym, {{"N", 1}}), 2},
  };
  BlockSpace const space(sym, sectors);

  ASSERT_EQ(space.size(), 3);
  EXPECT_EQ(u1_component(space[0].q, "N"), U1{0});
  EXPECT_EQ(u1_component(space[1].q, "N"), U1{1});
  EXPECT_EQ(u1_component(space[2].q, "N"), U1{-1});

  EXPECT_THROW(static_cast<void>(BlockSpace(sym,
                                            {
                                                {make_qnum(sym, {{"N", 1}}), 2},
                                                {make_qnum(sym, {{"N", 1}}), 3},
                                            })),
               std::invalid_argument);
}

TEST(BlockSpaceTest, EmptySpaceRetainsSymmetryAndLabelRemainsMutable)
{
  Symmetry const sym{"N:U(1)"};
  BlockSpace space(sym, "left");

  EXPECT_TRUE(space.empty());
  EXPECT_EQ(space.symmetry(), sym);
  EXPECT_EQ(space.label(), "left");

  auto same = space;
  EXPECT_EQ(same, space);
  same.set_label("right");
  EXPECT_NE(same, space);
  EXPECT_EQ(same.symmetry(), sym);

  EXPECT_THROW(static_cast<void>(BlockSpace(Symmetry{})), std::invalid_argument);
}

TEST(BlockSpaceTest, QNumListRegularizationProducesCanonicalBlockSpace)
{
  Symmetry const sym{"N:U(1)"};

  QNumList list(sym, {
                         make_qnum(sym, {{"N", 1}}),
                         make_qnum(sym, {{"N", -1}}),
                         make_qnum(sym, {{"N", 1}}),
                         make_qnum(sym, {{"N", 0}}),
                         make_qnum(sym, {{"N", -1}}),
                     });

  auto const regularized = regularize(list);

  ASSERT_EQ(regularized.regular.size(), 3);
  EXPECT_EQ(u1_component(regularized.regular[0].q, "N"), U1{0});
  EXPECT_EQ(regularized.regular[0].dim, 1);
  EXPECT_EQ(u1_component(regularized.regular[1].q, "N"), U1{1});
  EXPECT_EQ(regularized.regular[1].dim, 2);
  EXPECT_EQ(u1_component(regularized.regular[2].q, "N"), U1{-1});
  EXPECT_EQ(regularized.regular[2].dim, 2);

  ASSERT_EQ(regularized.block_index.size(), list.size());
  ASSERT_EQ(regularized.block_offset.size(), list.size());
  EXPECT_EQ(regularized.block_index[0], 1);
  EXPECT_EQ(regularized.block_offset[0], 0);
  EXPECT_EQ(regularized.block_index[1], 2);
  EXPECT_EQ(regularized.block_offset[1], 0);
  EXPECT_EQ(regularized.block_index[2], 1);
  EXPECT_EQ(regularized.block_offset[2], 1);
  EXPECT_EQ(regularized.block_index[3], 0);
  EXPECT_EQ(regularized.block_offset[3], 0);
  EXPECT_EQ(regularized.block_index[4], 2);
  EXPECT_EQ(regularized.block_offset[4], 1);

  auto const as_block_space = to_block_space(list);
  EXPECT_EQ(as_block_space, regularized.regular);
}
