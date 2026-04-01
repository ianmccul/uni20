#include <uni20/operator/local_space.hpp>

#include <gtest/gtest.h>

using namespace uni20;

TEST(LocalSpaceTest, WrapsOrderedSparseQNumList)
{
    Symmetry const sym{"N:U(1)"};

    LocalSpace space(sym);
    space.push_back(make_qnum(sym, {{"N", U1{0}}}));
    space.push_back(make_qnum(sym, {{"N", U1{1}}}));
    space.push_back(make_qnum(sym, {{"N", U1{1}}}));

    EXPECT_EQ(space.symmetry(), sym);
    EXPECT_EQ(space.size(), 3);
    EXPECT_FALSE(space.empty());
    EXPECT_TRUE(space.contains(make_qnum(sym, {{"N", U1{1}}})));
    EXPECT_EQ(space[1], make_qnum(sym, {{"N", U1{1}}}));
    EXPECT_EQ(space.qnums().size(), 3);
}

TEST(LocalSpaceTest, SingletonConstructorUsesIrrepSymmetry)
{
    Symmetry const sym{"N:U(1)"};
    QNum const q = make_qnum(sym, {{"N", U1{2}}});

    LocalSpace space(q);

    EXPECT_EQ(space.symmetry(), sym);
    ASSERT_EQ(space.size(), 1);
    EXPECT_EQ(space[0], q);
}
