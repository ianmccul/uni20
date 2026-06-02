#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/dmrg.hpp>

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

auto make_spin_half_dense_site() -> SpinHalfSite
{
  Symmetry const symmetry{"Trivial:U(1)"};
  QNum const scalar = QNum::identity(symmetry);
  LocalSpace const space(symmetry, {scalar, scalar});

  LocalOperator identity(space, space, scalar);
  identity.insert_or_assign(0, 0, 1.0);
  identity.insert_or_assign(1, 1, 1.0);

  LocalOperator sz(space, space, scalar);
  sz.insert_or_assign(0, 0, 0.5);
  sz.insert_or_assign(1, 1, -0.5);

  LocalOperator sp(space, space, scalar);
  sp.insert_or_assign(0, 1, 1.0);

  LocalOperator sm(space, space, scalar);
  sm.insert_or_assign(1, 0, 1.0);

  LocalOperator sigma_z(space, space, scalar);
  sigma_z.insert_or_assign(0, 0, 1.0);
  sigma_z.insert_or_assign(1, 1, -1.0);

  return SpinHalfSite{
      .symmetry = symmetry,
      .space = space,
      .up = scalar,
      .down = scalar,
      .identity = std::move(identity),
      .sz = std::move(sz),
      .sp = std::move(sp),
      .sm = std::move(sm),
      .sigma_z = std::move(sigma_z),
  };
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

auto dmrg_options(std::size_t sweeps) -> TwoSiteDmrgOptions
{
  return TwoSiteDmrgOptions{
      .sweeps = sweeps,
      .sweep =
          TwoSiteSweepOptions{
              .lanczos =
                  tensorcontraction::LanczosOptions{.max_iterations = 12, .min_iterations = 2, .tolerance = 1.0e-12},
              .svd = tensorcontraction::SvdOptions{},
          },
  };
}

} // namespace

TEST(TwoSiteDmrgTest, RunsOneFullSweepOnTwoSiteChain)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, 2);
  auto mpo = make_spin_half_heisenberg_mpo(2, spin, 1.0, 0.0);

  auto result = run_two_site_dmrg(psi, mpo, dmrg_options(1));

  ASSERT_EQ(result.sweeps.size(), 1);
  EXPECT_EQ(result.sweeps[0].sweep, 0);
  ASSERT_EQ(result.sweeps[0].left_to_right.updates.size(), 1);
  ASSERT_EQ(result.sweeps[0].right_to_left.updates.size(), 1);
  EXPECT_EQ(result.sweeps[0].left_to_right.direction, TwoSiteSweepDirection::LeftToRight);
  EXPECT_EQ(result.sweeps[0].right_to_left.direction, TwoSiteSweepDirection::RightToLeft);
  EXPECT_NEAR(result.sweeps[0].left_to_right.updates[0].energy, -0.75, 1.0e-12);
  EXPECT_NEAR(result.sweeps[0].right_to_left.updates[0].energy, -0.75, 1.0e-12);
  EXPECT_NEAR(final_two_site_energy(result), -0.75, 1.0e-12);
  EXPECT_EQ(psi[0].right_dim(), 2);
  EXPECT_EQ(psi[1].left_dim(), 2);
}

TEST(TwoSiteDmrgTest, AlternatesFullSweepsAndPreservesDiagnostics)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, 3);
  auto mpo = make_spin_half_heisenberg_mpo(3, spin, 1.0, 0.0);

  auto result = run_two_site_dmrg(psi, mpo, dmrg_options(2));

  ASSERT_EQ(result.sweeps.size(), 2);
  for (std::size_t sweep = 0; sweep < result.sweeps.size(); ++sweep)
  {
    EXPECT_EQ(result.sweeps[sweep].sweep, sweep);
    ASSERT_EQ(result.sweeps[sweep].left_to_right.updates.size(), 2);
    ASSERT_EQ(result.sweeps[sweep].right_to_left.updates.size(), 2);
    EXPECT_EQ(result.sweeps[sweep].left_to_right.updates[0].left_site, 0);
    EXPECT_EQ(result.sweeps[sweep].left_to_right.updates[1].left_site, 1);
    EXPECT_EQ(result.sweeps[sweep].right_to_left.updates[0].left_site, 1);
    EXPECT_EQ(result.sweeps[sweep].right_to_left.updates[1].left_site, 0);
    EXPECT_TRUE(result.sweeps[sweep].left_to_right.updates[0].lanczos.converged());
    EXPECT_TRUE(result.sweeps[sweep].left_to_right.updates[1].lanczos.converged());
    EXPECT_TRUE(result.sweeps[sweep].right_to_left.updates[0].lanczos.converged());
    EXPECT_TRUE(result.sweeps[sweep].right_to_left.updates[1].lanczos.converged());
  }
  EXPECT_EQ(psi[0].right_bond_space(), psi[1].left_bond_space());
  EXPECT_EQ(psi[1].right_bond_space(), psi[2].left_bond_space());
}

TEST(TwoSiteDmrgTest, DenseFourSiteChainConvergesToExactEnergy)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_dense_site();
  auto psi = alternating_product_state(spin, 4);
  auto mpo = make_spin_half_heisenberg_mpo(4, spin, 1.0, 0.0);

  auto result = run_two_site_dmrg(psi, mpo, dmrg_options(4));

  EXPECT_NEAR(final_two_site_energy(result), -1.616025403784438, 1.0e-12);
}

TEST(TwoSiteDmrgTest, RejectsInvalidInputs)
{
  auto const spin = make_spin_half_u1_site();
  auto psi = alternating_product_state(spin, 2);
  auto mpo = make_spin_half_heisenberg_mpo(2, spin);

  EXPECT_THROW(static_cast<void>(run_two_site_dmrg(psi, mpo, dmrg_options(0))), std::invalid_argument);

  TwoSiteDmrgResult empty;
  EXPECT_THROW(static_cast<void>(final_two_site_energy(empty)), std::logic_error);
}
