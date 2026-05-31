#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/two_site_solve.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstdlib>

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

auto product_state_mps(SpinHalfSite const& spin) -> FiniteMPS
{
  MpsSiteTensor left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  MpsSiteTensor right(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  left.assign(0, std::array{1.0});
  left.assign(1, std::array{0.0});
  right.assign(0, std::array{0.0});
  right.assign(1, std::array{1.0});
  return FiniteMPS({std::move(left), std::move(right)});
}

} // namespace

TEST(TwoSiteSolveTest, FindsTwoSiteHeisenbergGroundState)
{
  ensure_mpi_initialized();

  auto const spin = make_spin_half_u1_site();
  auto psi = product_state_mps(spin);
  auto mpo = make_spin_half_heisenberg_mpo(2, spin, 1.0, 0.0);
  auto left_envs = build_left_environments(psi, mpo);
  auto right_envs = build_right_environments(psi, mpo);

  auto result =
      solve_two_site(psi, mpo, 0, left_envs[0], right_envs[2],
                     tensorcontraction::LanczosOptions{.max_iterations = 8, .min_iterations = 2, .tolerance = 1.0e-12});

  EXPECT_NEAR(result.lanczos.eigenvalue, -0.75, 1.0e-12);
  EXPECT_TRUE(result.lanczos.converged());
  ASSERT_EQ(result.optimized_vector.size(), 4);
  for (std::size_t i = 0; i < result.optimized_vector.size(); ++i)
  {
    EXPECT_EQ(result.optimized_vector.block(i), (tensorcontraction::MatrixFamily::Block{1, 1}));
  }
  EXPECT_EQ(result.optimized_matrix.block(0), (tensorcontraction::MatrixFamily::Block{2, 2}));
  EXPECT_NEAR(tensorcontraction::norm(result.optimized_vector), 1.0, 1.0e-12);
  EXPECT_NEAR(result.optimized_matrix.values(0)[0], 0.0, 1.0e-12);
  EXPECT_NEAR(result.optimized_matrix.values(0)[3], 0.0, 1.0e-12);
  EXPECT_NEAR(std::abs(result.optimized_matrix.values(0)[1]), 1.0 / std::sqrt(2.0), 1.0e-12);
  EXPECT_NEAR(std::abs(result.optimized_matrix.values(0)[2]), 1.0 / std::sqrt(2.0), 1.0e-12);
  EXPECT_LT(result.optimized_matrix.values(0)[1] * result.optimized_matrix.values(0)[2], 0.0);
}

TEST(TwoSiteSolveTest, ConvertsVectorBackToMatrixLayout)
{
  TwoSiteEffectiveHamiltonianLayout layout{
      .left_bond_dim = 1, .left_physical_dim = 2, .right_physical_dim = 2, .right_bond_dim = 1};
  std::array vector_blocks{tensorcontraction::MatrixFamily::Block{1, 1}, tensorcontraction::MatrixFamily::Block{1, 1},
                           tensorcontraction::MatrixFamily::Block{1, 1}, tensorcontraction::MatrixFamily::Block{1, 1}};
  tensorcontraction::MatrixFamily vector(vector_blocks);
  vector.assign(0, std::array{1.0});
  vector.assign(1, std::array{2.0});
  vector.assign(2, std::array{3.0});
  vector.assign(3, std::array{4.0});

  auto matrix = make_two_site_matrix(vector, layout);

  EXPECT_EQ(matrix.block(0), (tensorcontraction::MatrixFamily::Block{2, 2}));
  EXPECT_DOUBLE_EQ(matrix.values(0)[0], 1.0);
  EXPECT_DOUBLE_EQ(matrix.values(0)[1], 2.0);
  EXPECT_DOUBLE_EQ(matrix.values(0)[2], 3.0);
  EXPECT_DOUBLE_EQ(matrix.values(0)[3], 4.0);
}

TEST(TwoSiteSolveTest, RejectsOutOfRangeSite)
{
  auto const spin = make_spin_half_u1_site();
  auto psi = product_state_mps(spin);
  auto mpo = make_spin_half_heisenberg_mpo(2, spin);
  auto left_envs = build_left_environments(psi, mpo);
  auto right_envs = build_right_environments(psi, mpo);

  EXPECT_THROW(static_cast<void>(solve_two_site(psi, mpo, 1, left_envs[1], right_envs[2])), std::out_of_range);
}
