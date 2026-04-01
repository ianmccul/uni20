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
    auto const plus_half = make_qnum(sym, {{"N", U1{half_int{0.5}}}});
    auto const minus_half = make_qnum(sym, {{"N", U1{half_int{-0.5}}}});
    auto const plus_one = make_qnum(sym, {{"N", 1}});
    auto const minus_one = make_qnum(sym, {{"N", -1}});

    EXPECT_EQ(zero.raw_code(), 0);
    EXPECT_EQ(plus_half.raw_code(), 1);
    EXPECT_EQ(minus_half.raw_code(), 2);
    EXPECT_EQ(plus_one.raw_code(), 3);
    EXPECT_EQ(minus_one.raw_code(), 4);
}

TEST(QNumTest, BasicU1OperationsWork)
{
    Symmetry const sym{"N:U(1),Sz:U(1)"};
    auto const q1 = make_qnum(sym, {{"N", U1{half_int{2.5}}}, {"Sz", U1{half_int{-0.5}}}});
    auto const q2 = make_qnum(sym, {{"N", U1{half_int{-1.5}}}, {"Sz", U1{half_int{0.5}}}});
    auto const sum = q1 + q2;

    EXPECT_EQ(u1_component(q1, "N"), U1{half_int{2.5}});
    EXPECT_EQ(u1_component(q1, "Sz"), U1{half_int{-0.5}});
    EXPECT_EQ(to_string(q1), "N=2.5,Sz=-0.5");
    EXPECT_EQ(u1_component(dual(q1), "N"), U1{half_int{-2.5}});
    EXPECT_EQ(u1_component(dual(q1), "Sz"), U1{half_int{0.5}});
    EXPECT_EQ(u1_component(sum, "N"), U1{1});
    EXPECT_EQ(u1_component(sum, "Sz"), U1{0});
    EXPECT_TRUE(is_scalar(make_qnum(sym, {{"N", 0}, {"Sz", 0}})));
    EXPECT_DOUBLE_EQ(qdim(q1), 1.0);
    EXPECT_EQ(degree(q1), 1);
}

TEST(QNumTest, IdentityAndMissingComponentsDefaultToZero)
{
    Symmetry const sym{"N:U(1),Sz:U(1)"};

    auto const identity_q = QNum::identity(sym);
    auto const n_only = make_qnum(sym, {{"N", U1{half_int{1.5}}}});

    EXPECT_TRUE(is_scalar(identity_q));
    EXPECT_EQ(to_string(identity_q), "N=0,Sz=0");
    EXPECT_EQ(u1_component(n_only, "N"), U1{half_int{1.5}});
    EXPECT_EQ(u1_component(n_only, "Sz"), U1{0});
}

TEST(QNumTest, InvalidInputsThrowUsefulErrors)
{
    Symmetry const sym{"N:U(1),Sz:U(1)"};
    QNum const invalid;

    EXPECT_FALSE(invalid.valid());
    EXPECT_THROW(static_cast<void>(invalid.symmetry()), std::logic_error);
    EXPECT_THROW(static_cast<void>(invalid.raw_code()), std::logic_error);
    EXPECT_THROW(static_cast<void>(is_scalar(invalid)), std::logic_error);

    EXPECT_THROW(static_cast<void>(make_qnum(sym, {{"Bogus", 0}})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(make_qnum(sym, {{"N", 0}, {"N", 1}})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(u1_component(make_qnum(sym, {{"N", 0}}), "Bogus")), std::invalid_argument);
}

TEST(QNumTest, CoercionUsesNamedComponents)
{
    Symmetry const particle{"N:U(1)"};
    Symmetry const full{"N:U(1),Sz:U(1)"};

    auto const n_only = make_qnum(particle, {{"N", U1{half_int{3.5}}}});
    auto const lifted = coerce(n_only, full);

    EXPECT_EQ(u1_component(lifted, "N"), U1{half_int{3.5}});
    EXPECT_EQ(u1_component(lifted, "Sz"), U1{0});

    auto const drop_ok = coerce(make_qnum(full, {{"N", U1{half_int{1.5}}}, {"Sz", U1{0}}}), particle);
    EXPECT_EQ(u1_component(drop_ok, "N"), U1{half_int{1.5}});

    EXPECT_THROW(static_cast<void>(coerce(make_qnum(full, {{"N", U1{1}}, {"Sz", U1{half_int{0.5}}}}), particle)),
                 std::invalid_argument);
}

TEST(QNumTest, OperatorPlusRequiresMatchingSymmetry)
{
    auto const n = make_qnum(Symmetry{"N:U(1)"}, {{"N", 1}});
    auto const sz = make_qnum(Symmetry{"Sz:U(1)"}, {{"Sz", 1}});

    EXPECT_THROW(static_cast<void>(n + sz), std::invalid_argument);
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
    EXPECT_EQ(u1_component(list[0], "N"), U1{1});
    EXPECT_EQ(u1_component(list[1], "N"), U1{2});

    EXPECT_THROW(list.push_back(make_qnum(other, {{"Sz", 0}})), std::invalid_argument);
}

TEST(QNumListTest, ConstructorRejectsMismatchedInitialValues)
{
    Symmetry const sym{"N:U(1)"};
    Symmetry const other{"Sz:U(1)"};

    EXPECT_THROW(static_cast<void>(QNumList(sym, {make_qnum(sym, {{"N", 0}}), make_qnum(other, {{"Sz", 0}})})),
                 std::invalid_argument);
}
