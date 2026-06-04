#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/block_sparse_dmrg.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace uni20;

namespace
{

auto heisenberg_two_site_product_state() -> BlockSparseFiniteMPS
{
  auto const spin = make_spin_half_u1_site();
  std::array<std::size_t, 2> const indices{0, 1};
  return make_block_sparse_product_state(spin.space, indices);
}

auto heisenberg_product_state(std::size_t length) -> BlockSparseFiniteMPS
{
  auto const spin = make_spin_half_u1_site();
  auto const indices = make_alternating_spin_half_indices(length);
  return make_block_sparse_product_state(spin.space, indices);
}

auto heisenberg_two_site_mpo() -> BlockSparseMpoChain
{
  auto const spin = make_spin_half_u1_site();
  return make_block_sparse_mpo_chain(make_spin_half_heisenberg_mpo(2, spin, 1.0, 0.0));
}

auto heisenberg_mpo(std::size_t length) -> BlockSparseMpoChain
{
  auto const spin = make_spin_half_u1_site();
  return make_block_sparse_mpo_chain(make_spin_half_heisenberg_mpo(length, spin, 1.0, 0.0));
}

auto two_site_layout(BlockSparseFiniteMPS const& psi) -> BlockSparseTwoSiteLayout
{
  return BlockSparseTwoSiteLayout(psi[0].local_space(), psi[1].local_space(), psi[0].row_space(), psi[1].col_space());
}

auto pair_key(std::size_t left_physical, std::size_t right_physical) -> ThreeLegBlockKey
{
  return ThreeLegBlockKey{
      .row_sector = 0, .local = two_site_pair_index(2, left_physical, right_physical), .col_sector = 0};
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

class ResidentTensorContractionBackendGuard {
  public:
    ResidentTensorContractionBackendGuard()
    {
      if (auto const* value = std::getenv("UNI20_TENSORCONTRACTION_BACKEND"); value != nullptr)
      {
        previous_ = value;
        had_previous_ = true;
      }
      unsetenv("UNI20_TENSORCONTRACTION_BACKEND");
    }

    ResidentTensorContractionBackendGuard(ResidentTensorContractionBackendGuard const&) = delete;
    auto operator=(ResidentTensorContractionBackendGuard const&) -> ResidentTensorContractionBackendGuard& = delete;

    ~ResidentTensorContractionBackendGuard()
    {
      if (had_previous_)
      {
        setenv("UNI20_TENSORCONTRACTION_BACKEND", previous_.c_str(), 1);
        return;
      }
      unsetenv("UNI20_TENSORCONTRACTION_BACKEND");
    }

  private:
    bool had_previous_ = false;
    std::string previous_;
};

} // namespace

TEST(BlockSparseEnvironmentTest, ProductStateExpectationUsesSparseU1Blocks)
{
  auto psi = heisenberg_two_site_product_state();
  auto const mpo = heisenberg_two_site_mpo();

  auto const left_envs = build_left_environments(psi, mpo);
  auto const right_envs = build_right_environments(psi, mpo);

  ASSERT_EQ(left_envs.size(), 3);
  ASSERT_EQ(right_envs.size(), 3);
  EXPECT_NEAR(environment_scalar(left_envs.back(), left_envs.back().local_space().size() - 1), -0.25, 1.0e-14);
  EXPECT_NEAR(mps_expectation_value(psi, mpo), -0.25, 1.0e-14);
}

TEST(BlockSparseTwoSiteVectorTest, BuildsOnlyLegalU1PhysicalPairBlocks)
{
  auto psi = heisenberg_two_site_product_state();
  auto const layout = two_site_layout(psi);
  auto const center = make_two_site_vector(psi, 0, layout);

  ASSERT_EQ(layout.block_count(), 2);
  ASSERT_TRUE(layout.contains(pair_key(0, 1)));
  ASSERT_TRUE(layout.contains(pair_key(1, 0)));
  EXPECT_DOUBLE_EQ(center.values(layout.block_index(pair_key(0, 1)))[0], 1.0);
  EXPECT_DOUBLE_EQ(center.values(layout.block_index(pair_key(1, 0)))[0], 0.0);
}

TEST(BlockSparseEffectiveHamiltonianTest, AppliesTwoSiteHeisenbergWithoutDenseFallback)
{
  auto psi = heisenberg_two_site_product_state();
  auto const mpo = heisenberg_two_site_mpo();
  auto const left_env = make_left_boundary_environment(psi, mpo);
  auto const right_env = make_right_boundary_environment(psi, mpo);
  auto const layout = two_site_layout(psi);
  auto const input = make_two_site_vector(psi, 0, layout);
  auto output = make_zero_matrix_family(layout);

  apply_two_site_effective_hamiltonian(left_env, mpo[0], mpo[1], right_env, layout, input, output);

  EXPECT_NEAR(output.values(layout.block_index(pair_key(0, 1)))[0], -0.25, 1.0e-14);
  EXPECT_NEAR(output.values(layout.block_index(pair_key(1, 0)))[0], 0.5, 1.0e-14);
}

TEST(BlockSparseTwoSiteSolveTest, FindsTwoSiteSingletAndSplitsByChargeSector)
{
  ensure_mpi_initialized();
  ResidentTensorContractionBackendGuard const resident_backend;
  auto psi = heisenberg_two_site_product_state();
  auto const mpo = heisenberg_two_site_mpo();
  auto const left_env = make_left_boundary_environment(psi, mpo);
  auto const right_env = make_right_boundary_environment(psi, mpo);

  auto solution =
      solve_two_site(psi, mpo, 0, left_env, right_env,
                     tensorcontraction::LanczosOptions{.max_iterations = 8, .min_iterations = 2, .tolerance = 1.0e-13});
  EXPECT_NEAR(solution.lanczos.eigenvalue, -0.75, 1.0e-12);

  auto split = split_two_site_solution(solution, TwoSiteSplitDirection::LeftToRight,
                                       tensorcontraction::SvdOptions{.max_rank = 4});
  ASSERT_EQ(split.sector_ranks.size(), 2);
  EXPECT_EQ(split.spectrum.singular_values.size(), 2);
  EXPECT_EQ(split.left.col_space().size(), 2);
  EXPECT_EQ(split.right.row_space().size(), 2);

  replace_two_site_solution(psi, 0, std::move(split));
  EXPECT_NEAR(mps_expectation_value(psi, mpo), -0.75, 1.0e-12);
}

TEST(BlockSparseTwoSiteSolveTest, ResidentSplitMaterializesOnlyAtExplicitBoundary)
{
  ensure_mpi_initialized();
  ResidentTensorContractionBackendGuard const resident_backend;
  auto psi = heisenberg_two_site_product_state();
  auto const mpo = heisenberg_two_site_mpo();
  auto const left_env = make_left_boundary_environment(psi, mpo);
  auto const right_env = make_right_boundary_environment(psi, mpo);

  auto solution =
      solve_two_site(psi, mpo, 0, left_env, right_env,
                     tensorcontraction::LanczosOptions{.max_iterations = 8, .min_iterations = 2, .tolerance = 1.0e-13});
  auto device_split = split_two_site_solution_resident(solution, TwoSiteSplitDirection::LeftToRight,
                                                       tensorcontraction::SvdOptions{.max_rank = 4});

  ASSERT_EQ(device_split.sector_ranks.size(), 2);
  EXPECT_EQ(device_split.spectrum.singular_values.size(), 2);
  EXPECT_EQ(device_split.left.col_space().size(), 2);
  EXPECT_EQ(device_split.right.row_space().size(), 2);

  auto split = device_split.materialize_to_host();
  replace_two_site_solution(psi, 0, std::move(split));
  EXPECT_NEAR(mps_expectation_value(psi, mpo), -0.75, 1.0e-12);
}

TEST(BlockSparseTwoSiteDmrgTest, TwoSiteRunConvergesToSinglet)
{
  ensure_mpi_initialized();
  ResidentTensorContractionBackendGuard const resident_backend;
  auto psi = heisenberg_two_site_product_state();
  auto const mpo = heisenberg_two_site_mpo();

  auto result = run_two_site_dmrg(
      psi, mpo,
      BlockSparseTwoSiteDmrgOptions{
          .sweeps = 1,
          .sweep =
              BlockSparseTwoSiteSweepOptions{
                  .lanczos =
                      tensorcontraction::LanczosOptions{.max_iterations = 8, .min_iterations = 2, .tolerance = 1.0e-13},
                  .svd = tensorcontraction::SvdOptions{.max_rank = 4},
              },
      });

  EXPECT_NEAR(final_two_site_energy(result), -0.75, 1.0e-12);
  EXPECT_NEAR(mps_expectation_value(psi, mpo), -0.75, 1.0e-12);
}

TEST(BlockSparseTwoSiteDmrgTest, FourSiteRunMatchesExactOpenChainEnergy)
{
  ensure_mpi_initialized();
  ResidentTensorContractionBackendGuard const resident_backend;
  auto psi = heisenberg_product_state(4);
  auto const mpo = heisenberg_mpo(4);

  auto result = run_two_site_dmrg(psi, mpo,
                                  BlockSparseTwoSiteDmrgOptions{
                                      .sweeps = 4,
                                      .sweep =
                                          BlockSparseTwoSiteSweepOptions{
                                              .lanczos = tensorcontraction::LanczosOptions{.max_iterations = 16,
                                                                                           .min_iterations = 2,
                                                                                           .tolerance = 1.0e-13},
                                              .svd = tensorcontraction::SvdOptions{.max_rank = 16},
                                          },
                                  });

  EXPECT_LT(result.sweeps.back().right_to_left.updates.back().kept_rank, 16);
  EXPECT_NEAR(mps_expectation_value(psi, mpo), -1.6160254037844386, 1.0e-10);
}
