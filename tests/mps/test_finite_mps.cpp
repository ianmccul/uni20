#include <uni20/models/spin_half.hpp>
#include <uni20/mps/finite_mps.hpp>

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

using namespace uni20;

namespace
{

auto bond_space(Symmetry sym, std::size_t dim) -> BlockSpace
{
  return BlockSpace(sym, {BlockSector{QNum::identity(sym), dim}});
}

} // namespace

TEST(MpsSiteTensorTest, StoresPhysicalBlocks)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor site(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 3));

  EXPECT_EQ(site.physical_dim(), 2);
  EXPECT_EQ(site.left_dim(), 2);
  EXPECT_EQ(site.right_dim(), 3);

  site.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  EXPECT_EQ(site.values(0)[0], 1.0);
  EXPECT_EQ(site.values(0)[5], 6.0);
  EXPECT_THROW(site.assign(1, std::array{1.0, 2.0}), std::invalid_argument);
}

TEST(FiniteMPSTest, ValidatesAdjacentBondSpaces)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 2));
  MpsSiteTensor right(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 1));

  FiniteMPS psi({std::move(left), std::move(right)});
  EXPECT_EQ(psi.size(), 2);

  MpsSiteTensor bad_left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 2));
  MpsSiteTensor bad_right(spin.space, bond_space(spin.symmetry, 3), bond_space(spin.symmetry, 1));
  EXPECT_THROW(FiniteMPS(FiniteMPS::container_type{std::move(bad_left), std::move(bad_right)}), std::invalid_argument);
}

TEST(FiniteMPSTest, ReplacesAdjacentSites)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 2));
  MpsSiteTensor middle(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 3));
  MpsSiteTensor right(spin.space, bond_space(spin.symmetry, 3), bond_space(spin.symmetry, 1));
  FiniteMPS psi({std::move(left), std::move(middle), std::move(right)});

  MpsSiteTensor new_left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 4));
  MpsSiteTensor new_middle(spin.space, bond_space(spin.symmetry, 4), bond_space(spin.symmetry, 3));
  psi.replace_adjacent(0, std::move(new_left), std::move(new_middle));

  EXPECT_EQ(psi[0].right_dim(), 4);
  EXPECT_EQ(psi[1].left_dim(), 4);
  EXPECT_EQ(psi[1].right_dim(), 3);

  MpsSiteTensor bad_new_left(spin.space, bond_space(spin.symmetry, 4), bond_space(spin.symmetry, 2));
  MpsSiteTensor bad_new_right(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 1));
  EXPECT_THROW(psi.replace_adjacent(0, std::move(bad_new_left), std::move(bad_new_right)), std::invalid_argument);
}

TEST(TwoSiteWavefunctionTest, PacksAdjacentSitesIntoSingleMatrixBlock)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor left(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 2));
  MpsSiteTensor right(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 3));

  left.assign(0, std::array{1.0, 2.0, 3.0, 4.0});
  left.assign(1, std::array{5.0, 6.0, 7.0, 8.0});
  right.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  right.assign(1, std::array{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  FiniteMPS psi({std::move(left), std::move(right)});
  auto center = make_two_site_wavefunction(psi, 0);

  EXPECT_EQ(center.left_site(), 0);
  EXPECT_EQ(center.right_site(), 1);
  EXPECT_EQ(center.block(), (uni20::tensorcontraction::MatrixFamily::Block{4, 6}));

  std::vector<double> expected{
      9.0,  12.0, 15.0, 27.0, 30.0, 33.0, 29.0, 40.0, 51.0, 95.0,  106.0, 117.0,
      19.0, 26.0, 33.0, 61.0, 68.0, 75.0, 39.0, 54.0, 69.0, 129.0, 144.0, 159.0,
  };

  ASSERT_EQ(center.values().size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(center.values()[i], expected[i]);
  }
}

TEST(TwoSiteWavefunctionTest, AssignsToCompatibleTensorContractionVector)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  MpsSiteTensor right(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  left.assign(0, std::array{2.0});
  left.assign(1, std::array{3.0});
  right.assign(0, std::array{5.0});
  right.assign(1, std::array{7.0});

  FiniteMPS psi({std::move(left), std::move(right)});
  auto center = make_two_site_wavefunction(psi, 0);
  std::array block{center.block()};
  tensorcontraction::MatrixFamily output(block);

  center.assign_to(output);

  EXPECT_EQ(output.values(0)[0], 10.0);
  EXPECT_EQ(output.values(0)[1], 14.0);
  EXPECT_EQ(output.values(0)[2], 15.0);
  EXPECT_EQ(output.values(0)[3], 21.0);
}
