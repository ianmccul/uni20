#include <uni20/models/spin_half.hpp>

#include <gtest/gtest.h>

using namespace uni20;

TEST(SpinHalfSiteTest, BuildsExpectedU1LocalSpaceAndOperators)
{
    auto const site = make_spin_half_u1_site();

    EXPECT_EQ(site.symmetry, Symmetry{"Sz:U(1)"});
    ASSERT_EQ(site.space.size(), 2);
    EXPECT_EQ(site.space[0], site.up);
    EXPECT_EQ(site.space[1], site.down);
    EXPECT_EQ(u1_component(site.up, "Sz"), U1{0.5});
    EXPECT_EQ(u1_component(site.down, "Sz"), U1{-0.5});

    EXPECT_TRUE(is_scalar(site.identity.transforms_as()));
    EXPECT_TRUE(is_scalar(site.sz.transforms_as()));
    EXPECT_TRUE(is_scalar(site.sigma_z.transforms_as()));
    EXPECT_EQ(u1_component(site.sp.transforms_as(), "Sz"), U1{1});
    EXPECT_EQ(u1_component(site.sm.transforms_as(), "Sz"), U1{-1});

    EXPECT_DOUBLE_EQ(site.identity.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(site.identity.at(1, 1), 1.0);

    EXPECT_DOUBLE_EQ(site.sz.at(0, 0), 0.5);
    EXPECT_DOUBLE_EQ(site.sz.at(1, 1), -0.5);

    EXPECT_DOUBLE_EQ(site.sigma_z.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(site.sigma_z.at(1, 1), -1.0);

    EXPECT_DOUBLE_EQ(site.sp.at(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(site.sm.at(1, 0), 1.0);
}

TEST(SpinHalfSiteTest, SupportsCustomChargeName)
{
    auto const site = make_spin_half_u1_site("Q");

    EXPECT_EQ(site.symmetry, Symmetry{"Q:U(1)"});
    EXPECT_EQ(u1_component(site.up, "Q"), U1{0.5});
    EXPECT_EQ(u1_component(site.sp.transforms_as(), "Q"), U1{1});
}
