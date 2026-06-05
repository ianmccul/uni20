#include <uni20/models/fermi_hubbard.hpp>
#include <uni20/mps/block_sparse_mps.hpp>
#include <uni20/mps/sparse_mpo_site.hpp>

#include <gtest/gtest.h>

#include <array>

using namespace uni20;

TEST(FermiHubbardSiteTest, BuildsExpectedU1U1LocalSpaceAndOperators)
{
  auto const site = make_fermi_hubbard_u1u1_site();

  EXPECT_EQ(site.symmetry, Symmetry{"N:U(1),Sz:U(1)"});
  ASSERT_EQ(site.space.size(), 4);
  EXPECT_EQ(site.space[0], site.empty);
  EXPECT_EQ(site.space[1], site.doubly_occupied);
  EXPECT_EQ(site.space[2], site.down);
  EXPECT_EQ(site.space[3], site.up);

  EXPECT_EQ(u1_component(site.empty, "N"), U1{0});
  EXPECT_EQ(u1_component(site.empty, "Sz"), U1{0});
  EXPECT_EQ(u1_component(site.doubly_occupied, "N"), U1{2});
  EXPECT_EQ(u1_component(site.doubly_occupied, "Sz"), U1{0});
  EXPECT_EQ(u1_component(site.down, "N"), U1{1});
  EXPECT_EQ(u1_component(site.down, "Sz"), U1{-0.5});
  EXPECT_EQ(u1_component(site.up, "N"), U1{1});
  EXPECT_EQ(u1_component(site.up, "Sz"), U1{0.5});

  EXPECT_TRUE(is_scalar(site.identity.transforms_as()));
  EXPECT_TRUE(is_scalar(site.parity.transforms_as()));
  EXPECT_EQ(u1_component(site.ch_up.transforms_as(), "N"), U1{1});
  EXPECT_EQ(u1_component(site.ch_up.transforms_as(), "Sz"), U1{0.5});
  EXPECT_EQ(u1_component(site.c_up.transforms_as(), "N"), U1{-1});
  EXPECT_EQ(u1_component(site.c_up.transforms_as(), "Sz"), U1{-0.5});
  EXPECT_EQ(u1_component(site.ch_down.transforms_as(), "N"), U1{1});
  EXPECT_EQ(u1_component(site.ch_down.transforms_as(), "Sz"), U1{-0.5});
  EXPECT_EQ(u1_component(site.c_down.transforms_as(), "N"), U1{-1});
  EXPECT_EQ(u1_component(site.c_down.transforms_as(), "Sz"), U1{0.5});

  EXPECT_DOUBLE_EQ(site.identity.at(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(site.identity.at(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(site.identity.at(2, 2), 1.0);
  EXPECT_DOUBLE_EQ(site.identity.at(3, 3), 1.0);

  EXPECT_DOUBLE_EQ(site.parity.at(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(site.parity.at(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(site.parity.at(2, 2), -1.0);
  EXPECT_DOUBLE_EQ(site.parity.at(3, 3), -1.0);

  EXPECT_DOUBLE_EQ(site.n.at(1, 1), 2.0);
  EXPECT_DOUBLE_EQ(site.n.at(2, 2), 1.0);
  EXPECT_DOUBLE_EQ(site.n.at(3, 3), 1.0);
  EXPECT_DOUBLE_EQ(site.n_up.at(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(site.n_up.at(3, 3), 1.0);
  EXPECT_DOUBLE_EQ(site.n_down.at(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(site.n_down.at(2, 2), 1.0);
  EXPECT_DOUBLE_EQ(site.sz.at(2, 2), -0.5);
  EXPECT_DOUBLE_EQ(site.sz.at(3, 3), 0.5);
  EXPECT_DOUBLE_EQ(site.p_double.at(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(site.hu.at(0, 0), 0.25);
  EXPECT_DOUBLE_EQ(site.hu.at(1, 1), 0.25);
  EXPECT_DOUBLE_EQ(site.hu.at(2, 2), -0.25);
  EXPECT_DOUBLE_EQ(site.hu.at(3, 3), -0.25);
}

TEST(FermiHubbardSiteTest, BuildsMptkFermionSignsAndParityModifiedOperators)
{
  auto const site = make_fermi_hubbard_u1u1_site();

  EXPECT_DOUBLE_EQ(site.ch_up.at(3, 0), 1.0);
  EXPECT_DOUBLE_EQ(site.ch_up.at(1, 2), 1.0);
  EXPECT_DOUBLE_EQ(site.c_up.at(0, 3), 1.0);
  EXPECT_DOUBLE_EQ(site.c_up.at(2, 1), 1.0);

  EXPECT_DOUBLE_EQ(site.ch_down.at(2, 0), 1.0);
  EXPECT_DOUBLE_EQ(site.ch_down.at(1, 3), -1.0);
  EXPECT_DOUBLE_EQ(site.c_down.at(0, 2), 1.0);
  EXPECT_DOUBLE_EQ(site.c_down.at(3, 1), -1.0);

  EXPECT_DOUBLE_EQ(site.ch_up_parity.at(3, 0), 1.0);
  EXPECT_DOUBLE_EQ(site.ch_up_parity.at(1, 2), -1.0);
  EXPECT_DOUBLE_EQ(site.c_up_parity.at(0, 3), -1.0);
  EXPECT_DOUBLE_EQ(site.c_up_parity.at(2, 1), 1.0);

  EXPECT_DOUBLE_EQ(site.ch_down_parity.at(2, 0), 1.0);
  EXPECT_DOUBLE_EQ(site.ch_down_parity.at(1, 3), 1.0);
  EXPECT_DOUBLE_EQ(site.c_down_parity.at(0, 2), -1.0);
  EXPECT_DOUBLE_EQ(site.c_down_parity.at(3, 1), -1.0);
}

TEST(FermiHubbardModelTest, BuildsExpectedBulkVirtualSpace)
{
  auto const site = make_fermi_hubbard_u1u1_site();
  auto const virtual_space = make_fermi_hubbard_virtual_space(site);

  ASSERT_EQ(virtual_space.size(), 6);
  EXPECT_TRUE(is_scalar(virtual_space[0]));
  EXPECT_EQ(virtual_space[1], site.c_up.transforms_as());
  EXPECT_EQ(virtual_space[2], site.ch_up.transforms_as());
  EXPECT_EQ(virtual_space[3], site.c_down.transforms_as());
  EXPECT_EQ(virtual_space[4], site.ch_down.transforms_as());
  EXPECT_TRUE(is_scalar(virtual_space[5]));
}

TEST(FermiHubbardModelTest, BuildsChargeCheckedSparseMpoSite)
{
  auto const site = make_fermi_hubbard_u1u1_site();
  auto const component = make_fermi_hubbard_bulk_component(site, 2.0, 4.0);

  EXPECT_TRUE(is_upper_triangular(component));
  EXPECT_EQ(component.local_bra_space(), site.space);
  EXPECT_EQ(component.local_ket_space(), site.space);
  EXPECT_EQ(component.nnz(), 11);

  EXPECT_DOUBLE_EQ(component.at(0, 1).at(3, 0), -2.0);
  EXPECT_DOUBLE_EQ(component.at(0, 1).at(1, 2), 2.0);
  EXPECT_DOUBLE_EQ(component.at(0, 2).at(0, 3), -2.0);
  EXPECT_DOUBLE_EQ(component.at(0, 2).at(2, 1), 2.0);
  EXPECT_DOUBLE_EQ(component.at(0, 5).at(1, 1), 4.0);
  EXPECT_DOUBLE_EQ(component.at(1, 5).at(0, 3), 1.0);
  EXPECT_DOUBLE_EQ(component.at(2, 5).at(3, 0), 1.0);

  SparseMpoSite const sparse_site(component);
  EXPECT_EQ(sparse_site.nnz(), 25);
  for (auto const& entry : sparse_site.entries())
  {
    EXPECT_TRUE(sparse_mpo_entry_allowed(sparse_site.left_virtual_space(), sparse_site.bra_space(),
                                         sparse_site.ket_space(), sparse_site.right_virtual_space(), entry.key));
  }
}

TEST(FermiHubbardModelTest, BuildsUniformFiniteTriangularMpo)
{
  auto const site = make_fermi_hubbard_u1u1_site();
  auto const mpo = make_fermi_hubbard_mpo(5, site, 1.0, 8.0);

  ASSERT_EQ(mpo.size(), 5);
  EXPECT_EQ(mpo.symmetry(), site.symmetry);
  EXPECT_TRUE(is_upper_triangular(mpo));

  for (std::size_t i = 0; i < mpo.size(); ++i)
  {
    EXPECT_EQ(mpo[i].local_bra_space(), site.space);
    EXPECT_EQ(mpo[i].local_ket_space(), site.space);
    EXPECT_EQ(mpo[i].left_virtual_space(), mpo[i].right_virtual_space());
    EXPECT_EQ(mpo[i].nnz(), 11);
  }
}

TEST(FermiHubbardModelTest, AlternatingHalfFilledProductStateHasSpinZeroBoundarySector)
{
  auto const site = make_fermi_hubbard_u1u1_site();
  std::array<std::size_t, 4> const indices{3, 2, 3, 2};
  auto const psi = make_block_sparse_product_state(site.space, indices);

  ASSERT_EQ(psi.size(), indices.size());
  ASSERT_EQ(psi[0].row_space().size(), 1);
  EXPECT_EQ(psi[0].row_space()[0].q, QNum::identity(site.symmetry));
  EXPECT_EQ(psi[0].row_space()[0].dim, 1);

  auto const& total_sector = psi[psi.size() - 1].col_space();
  ASSERT_EQ(total_sector.size(), 1);
  EXPECT_EQ(total_sector[0].dim, 1);
  EXPECT_EQ(u1_component(total_sector[0].q, "N"), U1{4});
  EXPECT_EQ(u1_component(total_sector[0].q, "Sz"), U1{0});
}
