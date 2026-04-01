#include <uni20/models/heisenberg.hpp>

#include <gtest/gtest.h>

using namespace uni20;

TEST(HeisenbergModelTest, BuildsExpectedBulkVirtualSpace)
{
    auto const site = make_spin_half_u1_site();
    auto const virtual_space = make_spin_half_heisenberg_virtual_space(site);

    ASSERT_EQ(virtual_space.size(), 5);
    EXPECT_TRUE(is_scalar(virtual_space[0]));
    EXPECT_EQ(u1_component(virtual_space[1], "Sz"), U1{-1});
    EXPECT_EQ(u1_component(virtual_space[2], "Sz"), U1{1});
    EXPECT_TRUE(is_scalar(virtual_space[3]));
    EXPECT_TRUE(is_scalar(virtual_space[4]));
}

TEST(HeisenbergModelTest, BuildsExpectedBulkComponent)
{
    auto const site = make_spin_half_u1_site();
    auto const component = make_spin_half_heisenberg_bulk_component(site, 2.0, 0.25);

    EXPECT_TRUE(is_upper_triangular(component));
    EXPECT_EQ(component.local_bra_space(), site.space);
    EXPECT_EQ(component.local_ket_space(), site.space);
    EXPECT_EQ(component.nnz(), 9);

    EXPECT_DOUBLE_EQ(component.at(0, 0).at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(component.at(0, 1).at(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(component.at(0, 2).at(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(component.at(0, 3).at(0, 0), 0.5);
    EXPECT_DOUBLE_EQ(component.at(0, 3).at(1, 1), -0.5);
    EXPECT_DOUBLE_EQ(component.at(0, 4).at(0, 0), 0.125);
    EXPECT_DOUBLE_EQ(component.at(0, 4).at(1, 1), -0.125);
    EXPECT_DOUBLE_EQ(component.at(1, 4).at(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(component.at(2, 4).at(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(component.at(3, 4).at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(component.at(3, 4).at(1, 1), -1.0);
    EXPECT_DOUBLE_EQ(component.at(4, 4).at(0, 0), 1.0);
}

TEST(HeisenbergModelTest, BuildsUniformFiniteTriangularMpo)
{
    auto const site = make_spin_half_u1_site();
    auto const mpo = make_spin_half_heisenberg_mpo(4, site, 1.5);

    ASSERT_EQ(mpo.size(), 4);
    EXPECT_EQ(mpo.symmetry(), site.symmetry);
    EXPECT_EQ(mpo.left_boundary_virtual_space(), mpo[0].left_virtual_space());
    EXPECT_EQ(mpo.right_boundary_virtual_space(), mpo[3].right_virtual_space());
    EXPECT_TRUE(is_upper_triangular(mpo));

    for (std::size_t i = 0; i < mpo.size(); ++i)
    {
        EXPECT_EQ(mpo[i].local_bra_space(), site.space);
        EXPECT_EQ(mpo[i].local_ket_space(), site.space);
        EXPECT_EQ(mpo[i].left_virtual_space(), mpo[i].right_virtual_space());
        EXPECT_EQ(mpo[i].nnz(), 8);
    }
}
