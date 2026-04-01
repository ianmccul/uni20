#include <uni20/symmetry/qnum.hpp>

#include <gtest/gtest.h>

using namespace uni20;

TEST(SymmetryTest, ParseCanonicalizesSpacing)
{
    Symmetry const sym_a{"N:U(1),Sz:U(1)"};
    Symmetry const sym_b{"  N:U(1) ,  Sz:U(1) "};

    EXPECT_EQ(sym_a, sym_b);
    EXPECT_EQ(sym_a.factor_count(), 2);
    EXPECT_EQ(to_string(sym_a), "N:U(1),Sz:U(1)");
}

TEST(QNumTest, U1EncodingUsesCanonicalDisplayOrder)
{
    Symmetry const sym{"N:U(1)"};

    auto const zero = make_qnum(sym, {{"N", 0}});
    auto const plus_one = make_qnum(sym, {{"N", 1}});
    auto const minus_one = make_qnum(sym, {{"N", -1}});
    auto const plus_two = make_qnum(sym, {{"N", 2}});

    EXPECT_EQ(zero.raw_code(), 0);
    EXPECT_EQ(plus_one.raw_code(), 1);
    EXPECT_EQ(minus_one.raw_code(), 2);
    EXPECT_EQ(plus_two.raw_code(), 3);
}

TEST(QNumTest, BasicU1OperationsWork)
{
    Symmetry const sym{"N:U(1),Sz:U(1)"};
    auto const q1 = make_qnum(sym, {{"N", 2}, {"Sz", -1}});
    auto const q2 = make_qnum(sym, {{"N", -1}, {"Sz", 1}});
    auto const sum = q1 + q2;

    EXPECT_EQ(u1_component(q1, "N"), 2);
    EXPECT_EQ(u1_component(q1, "Sz"), -1);
    EXPECT_EQ(to_string(q1), "N=2,Sz=-1");
    EXPECT_EQ(u1_component(dual(q1), "N"), -2);
    EXPECT_EQ(u1_component(dual(q1), "Sz"), 1);
    EXPECT_EQ(u1_component(sum, "N"), 1);
    EXPECT_EQ(u1_component(sum, "Sz"), 0);
    EXPECT_TRUE(is_scalar(make_qnum(sym, {{"N", 0}, {"Sz", 0}})));
    EXPECT_DOUBLE_EQ(qdim(q1), 1.0);
    EXPECT_EQ(degree(q1), 1);
}

TEST(QNumTest, CoercionUsesNamedComponents)
{
    Symmetry const particle{"N:U(1)"};
    Symmetry const full{"N:U(1),Sz:U(1)"};

    auto const n_only = make_qnum(particle, {{"N", 3}});
    auto const lifted = coerce(n_only, full);

    EXPECT_EQ(u1_component(lifted, "N"), 3);
    EXPECT_EQ(u1_component(lifted, "Sz"), 0);

    auto const drop_ok = coerce(make_qnum(full, {{"N", 1}, {"Sz", 0}}), particle);
    EXPECT_EQ(u1_component(drop_ok, "N"), 1);

    EXPECT_THROW(static_cast<void>(coerce(make_qnum(full, {{"N", 1}, {"Sz", 1}}), particle)), std::invalid_argument);
}

TEST(QNumListTest, EnforcesSharedSymmetryAndNormalizes)
{
    Symmetry const sym{"N:U(1)"};
    Symmetry const other{"Sz:U(1)"};

    QNumList list(sym);
    list.push_back(make_qnum(sym, {{"N", 2}}));
    list.push_back(make_qnum(sym, {{"N", 1}}));
    list.push_back(make_qnum(sym, {{"N", 2}}));

    EXPECT_TRUE(list.contains(make_qnum(sym, {{"N", 1}})));
    EXPECT_THROW(static_cast<void>(list.contains(make_qnum(other, {{"Sz", 1}}))), std::invalid_argument);

    list.normalize();
    ASSERT_EQ(list.size(), 2);
    EXPECT_EQ(u1_component(list[0], "N"), 1);
    EXPECT_EQ(u1_component(list[1], "N"), 2);

    EXPECT_THROW(list.push_back(make_qnum(other, {{"Sz", 0}})), std::invalid_argument);
}
