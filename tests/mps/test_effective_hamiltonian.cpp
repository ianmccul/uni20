#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/effective_hamiltonian.hpp>

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

auto scalar_virtual_space(Symmetry sym) -> LocalSpace { return LocalSpace(sym, {QNum::identity(sym)}); }

auto scalar_environment(LocalSpace virtual_space, BlockSpace bond, std::size_t identity_index) -> MpoEnvironment
{
  MpoEnvironment env(std::move(virtual_space), std::move(bond));
  env.set_identity(identity_index);
  return env;
}

auto one_entry_component(SpinHalfSite const& site, LocalOperator op) -> OperatorComponent
{
  auto virtual_space = scalar_virtual_space(site.symmetry);
  OperatorComponent component(site.space, site.space, virtual_space, virtual_space);
  component.insert_or_assign(0, 0, std::move(op));
  return component;
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

} // namespace

TEST(TwoSiteEffectiveHamiltonianTest, AppliesProductOperatorFromScalarEnvironments)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_u1_site();
  auto left_component = one_entry_component(spin, spin.sz);
  auto right_component = one_entry_component(spin, spin.identity);
  auto left_env = scalar_environment(left_component.left_virtual_space(), bond_space(spin.symmetry, 1), 0);
  auto right_env = scalar_environment(right_component.right_virtual_space(), bond_space(spin.symmetry, 1), 0);

  auto local_h = make_two_site_effective_hamiltonian(left_env, left_component, right_component, right_env);
  auto x = local_h.op.make_input_vector();
  auto y = local_h.op.make_output_vector();
  x.assign(0, std::array{1.0, 2.0, 3.0, 4.0});

  local_h.op.apply(x, y);

  std::array expected{0.5, 1.0, -1.5, -2.0};
  ASSERT_EQ(y.values(0).size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(y.values(0)[i], expected[i], 1.0e-12);
  }
}

TEST(TwoSiteEffectiveHamiltonianTest, AppliesTwoSiteHeisenbergBulkPath)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_u1_site();
  auto const left_component = make_spin_half_heisenberg_bulk_component(spin, 1.0, 0.0);
  auto const right_component = make_spin_half_heisenberg_bulk_component(spin, 1.0, 0.0);
  auto left_env = scalar_environment(left_component.left_virtual_space(), bond_space(spin.symmetry, 1), 0);
  auto right_env = scalar_environment(right_component.right_virtual_space(), bond_space(spin.symmetry, 1), 4);

  auto local_h = make_two_site_effective_hamiltonian(left_env, left_component, right_component, right_env);
  auto x = local_h.op.make_input_vector();
  auto y = local_h.op.make_output_vector();
  x.assign(0, std::array{1.0, 2.0, 3.0, 4.0});

  local_h.op.apply(x, y);

  std::array expected{0.25, 1.0, 0.25, 1.0};
  ASSERT_EQ(y.values(0).size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(y.values(0)[i], expected[i], 1.0e-12);
  }
}

TEST(TwoSiteEffectiveHamiltonianTest, VectorizesTwoSiteWavefunction)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  MpsSiteTensor right(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  left.assign(0, std::array{2.0});
  left.assign(1, std::array{3.0});
  right.assign(0, std::array{5.0});
  right.assign(1, std::array{7.0});
  FiniteMPS psi({std::move(left), std::move(right)});
  auto theta = make_two_site_wavefunction(psi, 0);
  TwoSiteEffectiveHamiltonianLayout layout{
      .left_bond_dim = 1, .left_physical_dim = 2, .right_physical_dim = 2, .right_bond_dim = 1};

  auto vector = make_two_site_vector(theta, layout);

  std::array expected{10.0, 14.0, 15.0, 21.0};
  ASSERT_EQ(vector.values(0).size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(vector.values(0)[i], expected[i]);
  }
}

TEST(TwoSiteEffectiveHamiltonianTest, RejectsMismatchedVirtualSpaces)
{
  auto const spin = make_spin_half_u1_site();
  auto left_component = one_entry_component(spin, spin.identity);
  auto right_component = make_spin_half_heisenberg_bulk_component(spin);
  auto left_env = scalar_environment(left_component.left_virtual_space(), bond_space(spin.symmetry, 1), 0);
  auto right_env = scalar_environment(right_component.right_virtual_space(), bond_space(spin.symmetry, 1), 4);

  EXPECT_THROW(
      static_cast<void>(make_two_site_effective_hamiltonian(left_env, left_component, right_component, right_env)),
      std::invalid_argument);
}
