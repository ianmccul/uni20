#include <uni20/models/spin_half.hpp>
#include <uni20/mps/two_site_split.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace uni20;

namespace
{

auto bond_space(Symmetry sym, std::size_t dim) -> BlockSpace
{
  return BlockSpace(sym, {BlockSector{QNum::identity(sym), dim}});
}

void ensure_mpi_initialized()
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0)
  {
    return;
  }

  MPI_Init(nullptr, nullptr);
  std::atexit([] {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0)
    {
      MPI_Finalize();
    }
  });
}

auto make_center(std::initializer_list<double> values) -> tensorcontraction::MatrixFamily
{
  std::array block{tensorcontraction::MatrixFamily::Block{2, 2}};
  tensorcontraction::MatrixFamily center(block);
  center.assign(0, std::span{values.begin(), values.size()});
  return center;
}

auto repack(MpsSiteTensor const& left, MpsSiteTensor const& right) -> std::vector<double>
{
  FiniteMPS psi({left, right});
  auto theta = make_two_site_wavefunction(psi, 0);
  return std::vector<double>(theta.values().begin(), theta.values().end());
}

void expect_reconstructs(MpsSiteTensor const& left, MpsSiteTensor const& right, std::span<double const> expected)
{
  auto actual = repack(left, right);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(actual[i], expected[i], 1.0e-12);
  }
}

} // namespace

TEST(TwoSiteSplitTest, SplitsLeftToRightAndAbsorbsSingularValuesOnRight)
{
  auto const spin = make_spin_half_u1_site();
  auto center = make_center({2.0, 0.0, 0.0, 1.0});
  TwoSiteEffectiveHamiltonianLayout layout{
      .left_bond_dim = 1, .left_physical_dim = 2, .right_physical_dim = 2, .right_bond_dim = 1};

  auto split = split_two_site_center(center, layout, spin.space, spin.space, bond_space(spin.symmetry, 1),
                                     bond_space(spin.symmetry, 1), TwoSiteSplitDirection::LeftToRight);

  EXPECT_EQ(split.left.right_dim(), 2);
  EXPECT_EQ(split.right.left_dim(), 2);
  EXPECT_NEAR(split.spectrum.singular_values[0], 2.0, 1.0e-12);
  EXPECT_NEAR(split.spectrum.singular_values[1], 1.0, 1.0e-12);
  expect_reconstructs(split.left, split.right, center.values(0));

  double left_col_norm_0 = 0.0;
  double left_col_norm_1 = 0.0;
  for (std::size_t phys = 0; phys < split.left.physical_dim(); ++phys)
  {
    left_col_norm_0 += split.left.values(phys)[0] * split.left.values(phys)[0];
    left_col_norm_1 += split.left.values(phys)[1] * split.left.values(phys)[1];
  }
  EXPECT_NEAR(left_col_norm_0, 1.0, 1.0e-12);
  EXPECT_NEAR(left_col_norm_1, 1.0, 1.0e-12);
}

TEST(TwoSiteSplitTest, SplitsRightToLeftAndAbsorbsSingularValuesOnLeft)
{
  auto const spin = make_spin_half_u1_site();
  auto center = make_center({2.0, 0.0, 0.0, 1.0});
  TwoSiteEffectiveHamiltonianLayout layout{
      .left_bond_dim = 1, .left_physical_dim = 2, .right_physical_dim = 2, .right_bond_dim = 1};

  auto split = split_two_site_center(center, layout, spin.space, spin.space, bond_space(spin.symmetry, 1),
                                     bond_space(spin.symmetry, 1), TwoSiteSplitDirection::RightToLeft);

  expect_reconstructs(split.left, split.right, center.values(0));

  double right_row_norm_0 = 0.0;
  double right_row_norm_1 = 0.0;
  for (std::size_t phys = 0; phys < split.right.physical_dim(); ++phys)
  {
    right_row_norm_0 += split.right.values(phys)[0] * split.right.values(phys)[0];
    right_row_norm_1 += split.right.values(phys)[1] * split.right.values(phys)[1];
  }
  EXPECT_NEAR(right_row_norm_0, 1.0, 1.0e-12);
  EXPECT_NEAR(right_row_norm_1, 1.0, 1.0e-12);
}

TEST(TwoSiteSplitTest, TruncatesAndReportsDiscardedWeight)
{
  auto const spin = make_spin_half_u1_site();
  auto center = make_center({3.0, 0.0, 0.0, 1.0});
  TwoSiteEffectiveHamiltonianLayout layout{
      .left_bond_dim = 1, .left_physical_dim = 2, .right_physical_dim = 2, .right_bond_dim = 1};

  auto split = split_two_site_center(center, layout, spin.space, spin.space, bond_space(spin.symmetry, 1),
                                     bond_space(spin.symmetry, 1), TwoSiteSplitDirection::LeftToRight,
                                     tensorcontraction::SvdOptions{.max_rank = 1});

  EXPECT_EQ(split.spectrum.singular_values.size(), 1);
  EXPECT_NEAR(split.spectrum.discarded_weight, 1.0, 1.0e-12);
  EXPECT_EQ(split.left.right_dim(), 1);
  EXPECT_EQ(split.right.left_dim(), 1);
}

TEST(TwoSiteSplitTest, ReplacesMpsSitesAfterSplit)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  MpsSiteTensor right(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  FiniteMPS psi({std::move(left), std::move(right)});
  auto center = make_center({1.0, 0.0, 0.0, 1.0});
  TwoSiteEffectiveHamiltonianLayout layout{
      .left_bond_dim = 1, .left_physical_dim = 2, .right_physical_dim = 2, .right_bond_dim = 1};
  auto split = split_two_site_center(center, layout, spin.space, spin.space, bond_space(spin.symmetry, 1),
                                     bond_space(spin.symmetry, 1), TwoSiteSplitDirection::LeftToRight);

  replace_two_site_solution(psi, 0, std::move(split));

  EXPECT_EQ(psi[0].right_dim(), 2);
  EXPECT_EQ(psi[1].left_dim(), 2);
}

TEST(TwoSiteSplitTest, UsesRankZeroSplitAuthorityInMpi)
{
  ensure_mpi_initialized();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto const spin = make_spin_half_u1_site();
  auto center = rank == 0 ? make_center({2.0, 0.25, -0.5, 1.0}) : make_center({3.0, -0.75, 0.125, 0.5});
  std::array<double, 4> rank_zero_center{2.0, 0.25, -0.5, 1.0};
  TwoSiteEffectiveHamiltonianLayout layout{
      .left_bond_dim = 1, .left_physical_dim = 2, .right_physical_dim = 2, .right_bond_dim = 1};

  auto split = split_two_site_center(center, layout, spin.space, spin.space, bond_space(spin.symmetry, 1),
                                     bond_space(spin.symmetry, 1), TwoSiteSplitDirection::LeftToRight);

  expect_reconstructs(split.left, split.right, rank_zero_center);
}
