#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/two_site_sweep.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
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

auto alternating_product_state(SpinHalfSite const& spin, std::size_t length) -> FiniteMPS
{
  std::vector<MpsSiteTensor> sites;
  sites.reserve(length);
  for (std::size_t site = 0; site < length; ++site)
  {
    MpsSiteTensor tensor(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
    if (site % 2 == 0)
    {
      tensor.assign(0, std::array{1.0});
      tensor.assign(1, std::array{0.0});
    }
    else
    {
      tensor.assign(0, std::array{0.0});
      tensor.assign(1, std::array{1.0});
    }
    sites.push_back(std::move(tensor));
  }
  return FiniteMPS(std::move(sites));
}

auto sweep_options() -> TwoSiteSweepOptions
{
  return TwoSiteSweepOptions{
      .lanczos = tensorcontraction::LanczosOptions{.max_iterations = 12, .min_iterations = 2, .tolerance = 1.0e-12},
      .svd = tensorcontraction::SvdOptions{},
  };
}

} // namespace

TEST(TwoSiteSweepTest, OptimizesTwoSiteChainLeftToRight)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, 2);
  auto mpo = make_spin_half_heisenberg_mpo(2, spin, 1.0, 0.0);

  auto result = sweep_two_site_left_to_right(psi, mpo, sweep_options());

  ASSERT_EQ(result.updates.size(), 1);
  EXPECT_EQ(result.direction, TwoSiteSweepDirection::LeftToRight);
  EXPECT_EQ(result.updates[0].left_site, 0);
  EXPECT_NEAR(result.updates[0].energy, -0.75, 1.0e-12);
  EXPECT_TRUE(result.updates[0].lanczos.converged());
  EXPECT_EQ(result.updates[0].kept_rank, 2);
  EXPECT_EQ(result.updates[0].full_rank, 2);
  EXPECT_DOUBLE_EQ(result.updates[0].discarded_weight, 0.0);
  EXPECT_EQ(psi[0].right_dim(), 2);
  EXPECT_EQ(psi[1].left_dim(), 2);
}

TEST(TwoSiteSweepTest, SweepsThreeSiteChainInBothDirections)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, 3);
  auto mpo = make_spin_half_heisenberg_mpo(3, spin, 1.0, 0.0);
  auto options = sweep_options();

  auto left_to_right = sweep_two_site(psi, mpo, TwoSiteSweepDirection::LeftToRight, options);
  auto right_to_left = sweep_two_site(psi, mpo, TwoSiteSweepDirection::RightToLeft, options);

  ASSERT_EQ(left_to_right.updates.size(), 2);
  ASSERT_EQ(right_to_left.updates.size(), 2);
  EXPECT_EQ(left_to_right.updates[0].left_site, 0);
  EXPECT_EQ(left_to_right.updates[1].left_site, 1);
  EXPECT_EQ(right_to_left.updates[0].left_site, 1);
  EXPECT_EQ(right_to_left.updates[1].left_site, 0);
  EXPECT_TRUE(left_to_right.updates[0].lanczos.converged());
  EXPECT_TRUE(left_to_right.updates[1].lanczos.converged());
  EXPECT_TRUE(right_to_left.updates[0].lanczos.converged());
  EXPECT_TRUE(right_to_left.updates[1].lanczos.converged());
  EXPECT_EQ(psi[0].right_bond_space(), psi[1].left_bond_space());
  EXPECT_EQ(psi[1].right_bond_space(), psi[2].left_bond_space());
}

TEST(TwoSiteSweepTest, NotifiesObserverAfterBondReplacement)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, 2);
  auto mpo = make_spin_half_heisenberg_mpo(2, spin, 1.0, 0.0);
  auto options = sweep_options();

  bool observed = false;
  options.observer = [&](TwoSiteSweepDirection direction, TwoSiteBondUpdate const& update) {
    observed = true;
    EXPECT_EQ(direction, TwoSiteSweepDirection::LeftToRight);
    EXPECT_EQ(update.left_site, 0);
    EXPECT_EQ(psi[0].right_dim(), update.kept_rank);
    EXPECT_EQ(psi[1].left_dim(), update.kept_rank);
  };

  auto result = sweep_two_site_left_to_right(psi, mpo, options);

  EXPECT_TRUE(observed);
  ASSERT_EQ(result.updates.size(), 1);
  EXPECT_EQ(result.updates[0].left_site, 0);
}

TEST(TwoSiteSweepTest, RejectsInvalidInputs)
{
  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, 1);
  auto mpo = make_spin_half_heisenberg_mpo(1, spin);

  EXPECT_THROW(static_cast<void>(sweep_two_site_left_to_right(psi, mpo)), std::invalid_argument);

  auto longer_mpo = make_spin_half_heisenberg_mpo(2, spin);
  EXPECT_THROW(static_cast<void>(sweep_two_site_right_to_left(psi, longer_mpo)), std::invalid_argument);
}
