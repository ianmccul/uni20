#include <uni20/async/debug_scheduler.hpp>
#include <uni20/krylov/block_tensor_vector.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>
#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/tensor_network/site_types.hpp>
#include <uni20/tensor_network/two_site_effective_hamiltonian.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <ranges>
#include <stdexcept>

namespace
{

using Storage = uni20::SeparateSparseBlockStorage<>;
using Center = uni20::tensor_network::TwoSiteCenter<double, uni20::BlockSpace, uni20::LocalSpace, uni20::LocalSpace,
                                                    uni20::BlockSpace, Storage>;
using Hamiltonian = uni20::tensor_network::TwoSiteLocalOperator<double, uni20::LocalSpace, uni20::LocalSpace>;
using Environment =
    uni20::tensor_network::MpoEnvironment<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
using Mpo = uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace,
                                           uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
using CenterKey = typename Center::key_type;
using HamiltonianKey = typename Hamiltonian::key_type;
using EnvironmentKey = typename Environment::key_type;
using MpoKey = typename Mpo::key_type;
using MpsSite = uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
using MpoSite =
    uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace>;
using ScalarEnvironment = uni20::tensor_network::ScalarEnvironment<double>;

static_assert(std::same_as<typename MpsSite::domain_type, uni20::Domain<uni20::BlockSpace, uni20::LocalSpace>>);
static_assert(std::same_as<typename MpsSite::codomain_type, uni20::Codomain<uni20::BlockSpace>>);
static_assert(std::same_as<typename MpoSite::domain_type, uni20::Domain<uni20::LocalSpace, uni20::LocalSpace>>);
static_assert(std::same_as<typename MpoSite::codomain_type, uni20::Codomain<uni20::LocalSpace, uni20::LocalSpace>>);
static_assert(ScalarEnvironment::order() == 0);
static_assert(Environment::dense_block_order() == 2);
static_assert(Mpo::dense_block_order() == 0);

auto make_center(uni20::Symmetry const& symmetry, uni20::BlockSpace const& left_bond,
                 uni20::LocalSpace const& left_physical, uni20::LocalSpace const& right_physical,
                 uni20::BlockSpace const& right_bond) -> Center
{
  return Center(symmetry, uni20::Domain{left_bond, left_physical, right_physical}, uni20::Codomain{right_bond},
                {CenterKey{{0, 0, 1, 0}}, CenterKey{{0, 1, 0, 0}}});
}

auto make_heisenberg_hamiltonian(uni20::Symmetry const& symmetry, uni20::LocalSpace const& left_physical,
                                 uni20::LocalSpace const& right_physical) -> Hamiltonian
{
  HamiltonianKey const up_up{{0, 0, 0, 0}};
  HamiltonianKey const up_down{{0, 1, 0, 1}};
  HamiltonianKey const up_down_to_down_up{{0, 1, 1, 0}};
  HamiltonianKey const down_up_to_up_down{{1, 0, 0, 1}};
  HamiltonianKey const down_up{{1, 0, 1, 0}};
  HamiltonianKey const down_down{{1, 1, 1, 1}};
  Hamiltonian result(symmetry, uni20::Domain{uni20::dual(left_physical), uni20::dual(right_physical)},
                     uni20::Codomain{uni20::dual(left_physical), uni20::dual(right_physical)},
                     {up_up, up_down, up_down_to_down_up, down_up_to_up_down, down_up, down_down});
  result.block(up_up)[] = 0.25;
  result.block(up_down)[] = -0.25;
  result.block(up_down_to_down_up)[] = 0.5;
  result.block(down_up_to_up_down)[] = 0.5;
  result.block(down_up)[] = -0.25;
  result.block(down_down)[] = 0.25;
  return result;
}

TEST(TwoSiteDmrgSlice, FindsAndFactorizesU1HeisenbergGroundState)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const left_bond(symmetry, {{q0, 1}}, "left-boundary");
  uni20::BlockSpace const right_bond(symmetry, {{q1, 1}}, "right-boundary");
  uni20::LocalSpace const left_physical(symmetry, {q0, q1}, "left-physical");
  uni20::LocalSpace const right_physical(symmetry, {q0, q1}, "right-physical");

  Center initial = make_center(symmetry, left_bond, left_physical, right_physical, right_bond);
  initial.block(CenterKey{{0, 0, 1, 0}})[0, 0] = 1.0;
  initial.block(CenterKey{{0, 1, 0, 0}})[0, 0] = 0.0;
  Hamiltonian hamiltonian = make_heisenberg_hamiltonian(symmetry, left_physical, right_physical);
  uni20::tensor_network::LocalTwoSiteEffectiveHamiltonian effective_hamiltonian(std::move(hamiltonian));
  using EffectiveHamiltonian = decltype(effective_hamiltonian);
  static_assert(std::invocable<EffectiveHamiltonian&, Center&, Center const&>);

  Center applied = make_center(symmetry, left_bond, left_physical, right_physical, right_bond);
  effective_hamiltonian(applied, initial);
  EXPECT_DOUBLE_EQ((applied.block(CenterKey{{0, 0, 1, 0}})[0, 0]), -0.25);
  EXPECT_DOUBLE_EQ((applied.block(CenterKey{{0, 1, 0, 0}})[0, 0]), 0.5);

  uni20::krylov::BlockTensorMatrixFreeOps ops(initial, std::move(effective_hamiltonian));
  uni20::krylov::SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 2;
  params.spectrum = uni20::krylov::SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = true;
  auto eigensystem = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  ASSERT_EQ(eigensystem.eigenvalues.size(), 1);
  ASSERT_EQ(eigensystem.eigenvectors.size(), 1);
  EXPECT_NEAR(eigensystem.eigenvalues[0], -0.75, 1.0e-12);
  auto const& ground = eigensystem.eigenvectors[0];
  EXPECT_NEAR(ops.norm(ground), 1.0, 1.0e-12);
  double const up_down = ground.block(CenterKey{{0, 0, 1, 0}})[0, 0];
  double const down_up = ground.block(CenterKey{{0, 1, 0, 0}})[0, 0];
  EXPECT_NEAR(up_down + down_up, 0.0, 1.0e-12);
  EXPECT_NEAR(std::abs(up_down), std::sqrt(0.5), 1.0e-12);

  Center h_ground = ops.allocate_like(ground);
  ops.matvec(h_ground, ground);
  EXPECT_NEAR(ops.inner_product(ground, h_ground), -0.75, 1.0e-12);
  EXPECT_TRUE(std::ranges::equal(ground.stored_keys(), initial.stored_keys()));

  auto matrix_view = uni20::repartition<uni20::MorphismSide::Domain, uni20::BoundaryEnd::Right>(ground);
  auto decomposition = uni20::block_svd(matrix_view);
  auto selection = uni20::select_svd_states(decomposition.spectrum());
  auto factors = uni20::materialize_svd(decomposition, selection, {.bond_label = "schmidt"});
  ASSERT_EQ(decomposition.spectrum().size(), 2);
  for (auto const& state : decomposition.spectrum())
    EXPECT_NEAR(state.singular_value, std::sqrt(0.5), 1.0e-12);

  auto right_times_values =
      uni20::contract_adjacent<1>(factors.right_singular_vectors_adjoint, factors.singular_values);
  auto reconstructed_matrix = uni20::contract_adjacent<1>(right_times_values, factors.left_singular_vectors);
  auto reconstructed_center =
      uni20::repartition<uni20::MorphismSide::Codomain, uni20::BoundaryEnd::Right>(reconstructed_matrix);
  Center reconstruction_error = ops.allocate_like(ground);
  uni20::copy(reconstruction_error, reconstructed_center);
  uni20::axpy(reconstruction_error, -1.0, ground);
  EXPECT_LT(uni20::norm_host(reconstruction_error), 1.0e-12);
  EXPECT_EQ(reconstructed_center.domain(), ground.domain());
  EXPECT_EQ(reconstructed_center.codomain(), ground.codomain());
}

TEST(TwoSiteDmrgSlice, CompilesMpoAndEnvironmentsIntoSparseEffectiveHamiltonianPlan)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  auto const qminus1 = uni20::make_qnum(symmetry, {{"N", -1}});
  uni20::BlockSpace const left_bond(symmetry, {{q0, 1}}, "left-boundary");
  uni20::BlockSpace const right_bond(symmetry, {{q1, 1}}, "right-boundary");
  uni20::LocalSpace const left_physical(symmetry, {q0, q1}, "left-physical");
  uni20::LocalSpace const right_physical(symmetry, {q0, q1}, "right-physical");
  uni20::LocalSpace const left_auxiliary(symmetry, {q0}, "left-mpo-bond");
  uni20::LocalSpace const middle_auxiliary(symmetry, {q0, qminus1, q1}, "middle-mpo-bond");
  uni20::LocalSpace const right_auxiliary(symmetry, {q0}, "right-mpo-bond");

  Center initial = make_center(symmetry, left_bond, left_physical, right_physical, right_bond);
  initial.block(CenterKey{{0, 0, 1, 0}})[0, 0] = 1.0;
  initial.block(CenterKey{{0, 1, 0, 0}})[0, 0] = 0.0;

  Environment left_environment(symmetry, uni20::Domain{left_bond, left_auxiliary}, uni20::Codomain{left_bond},
                               {EnvironmentKey{{0, 0, 0}}});
  Environment right_environment(symmetry, uni20::Domain{right_bond, right_auxiliary}, uni20::Codomain{right_bond},
                                {EnvironmentKey{{0, 0, 0}}});
  left_environment.block(EnvironmentKey{{0, 0, 0}})[0, 0] = 1.0;
  right_environment.block(EnvironmentKey{{0, 0, 0}})[0, 0] = 1.0;

  Mpo first_mpo(symmetry, uni20::Domain{left_auxiliary, left_physical},
                uni20::Codomain{middle_auxiliary, left_physical},
                {MpoKey{{0, 0, 0, 0}}, MpoKey{{0, 0, 1, 1}}, MpoKey{{0, 1, 0, 1}}, MpoKey{{0, 1, 2, 0}}});
  first_mpo.block(MpoKey{{0, 0, 0, 0}})[] = 0.5;
  first_mpo.block(MpoKey{{0, 1, 0, 1}})[] = -0.5;
  first_mpo.block(MpoKey{{0, 0, 1, 1}})[] = 0.5;
  first_mpo.block(MpoKey{{0, 1, 2, 0}})[] = 0.5;

  Mpo second_mpo(symmetry, uni20::Domain{middle_auxiliary, right_physical},
                 uni20::Codomain{right_auxiliary, right_physical},
                 {MpoKey{{0, 0, 0, 0}}, MpoKey{{0, 1, 0, 1}}, MpoKey{{1, 1, 0, 0}}, MpoKey{{2, 0, 0, 1}}});
  second_mpo.block(MpoKey{{0, 0, 0, 0}})[] = 0.5;
  second_mpo.block(MpoKey{{0, 1, 0, 1}})[] = -0.5;
  second_mpo.block(MpoKey{{1, 1, 0, 0}})[] = 1.0;
  second_mpo.block(MpoKey{{2, 0, 0, 1}})[] = 1.0;

  Center incomplete(symmetry, uni20::Domain{left_bond, left_physical, right_physical}, uni20::Codomain{right_bond},
                    {CenterKey{{0, 0, 1, 0}}});
  EXPECT_THROW((static_cast<void>(uni20::tensor_network::make_two_site_effective_hamiltonian(
                   incomplete, left_environment, first_mpo, second_mpo, right_environment))),
               std::invalid_argument);

  auto effective_hamiltonian = uni20::tensor_network::make_two_site_effective_hamiltonian(
      initial, left_environment, first_mpo, second_mpo, right_environment);
  EXPECT_EQ(effective_hamiltonian.term_count(), 4);
  EXPECT_GT(effective_hamiltonian.prepared_intermediate_count(), 0);

  Center applied = make_center(symmetry, left_bond, left_physical, right_physical, right_bond);
  effective_hamiltonian(applied, initial);
  EXPECT_NEAR((applied.block(CenterKey{{0, 0, 1, 0}})[0, 0]), -0.25, 1.0e-14);
  EXPECT_NEAR((applied.block(CenterKey{{0, 1, 0, 0}})[0, 0]), 0.5, 1.0e-14);

  uni20::krylov::BlockTensorMatrixFreeOps ops(initial, std::move(effective_hamiltonian));
  uni20::krylov::SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 2;
  params.spectrum = uni20::krylov::SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = true;
  auto eigensystem = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);
  ASSERT_EQ(eigensystem.eigenvalues.size(), 1);
  EXPECT_NEAR(eigensystem.eigenvalues[0], -0.75, 1.0e-12);
  EXPECT_NEAR(ops.norm(eigensystem.eigenvectors[0]), 1.0, 1.0e-12);
}

TEST(TwoSiteDmrgSlice, AppliesDenseEnvironmentBlocksAsATimesBTimesCTranspose)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::BlockSpace const left_bond(symmetry, {{q0, 2}}, "left-boundary");
  uni20::BlockSpace const right_bond(symmetry, {{q0, 2}}, "right-boundary");
  uni20::LocalSpace const left_physical(symmetry, {q0}, "left-physical");
  uni20::LocalSpace const right_physical(symmetry, {q0}, "right-physical");
  uni20::LocalSpace const left_auxiliary(symmetry, {q0}, "left-mpo-bond");
  uni20::LocalSpace const middle_auxiliary(symmetry, {q0}, "middle-mpo-bond");
  uni20::LocalSpace const right_auxiliary(symmetry, {q0}, "right-mpo-bond");

  CenterKey const center_key{{0, 0, 0, 0}};
  Center input(symmetry, uni20::Domain{left_bond, left_physical, right_physical}, uni20::Codomain{right_bond},
               {center_key});
  auto input_block = input.block(center_key);
  input_block[0, 0] = 5.0;
  input_block[0, 1] = 6.0;
  input_block[1, 0] = 7.0;
  input_block[1, 1] = 8.0;

  EnvironmentKey const environment_key{{0, 0, 0}};
  Environment left_environment(symmetry, uni20::Domain{left_bond, left_auxiliary}, uni20::Codomain{left_bond},
                               {environment_key});
  Environment right_environment(symmetry, uni20::Domain{right_bond, right_auxiliary}, uni20::Codomain{right_bond},
                                {environment_key});
  auto left_block = left_environment.block(environment_key);
  left_block[0, 0] = 1.0;
  left_block[0, 1] = 2.0;
  left_block[1, 0] = 3.0;
  left_block[1, 1] = 4.0;
  auto right_block = right_environment.block(environment_key);
  right_block[0, 0] = 9.0;
  right_block[0, 1] = 10.0;
  right_block[1, 0] = 11.0;
  right_block[1, 1] = 12.0;

  MpoKey const mpo_key{{0, 0, 0, 0}};
  Mpo first_mpo(symmetry, uni20::Domain{left_auxiliary, left_physical},
                uni20::Codomain{middle_auxiliary, left_physical}, {mpo_key});
  Mpo second_mpo(symmetry, uni20::Domain{middle_auxiliary, right_physical},
                 uni20::Codomain{right_auxiliary, right_physical}, {mpo_key});
  first_mpo.block(mpo_key)[] = 2.0;
  second_mpo.block(mpo_key)[] = 1.0;

  auto effective_hamiltonian = uni20::tensor_network::make_two_site_effective_hamiltonian(
      input, left_environment, first_mpo, second_mpo, right_environment);
  using ParallelCenter =
      uni20::tensor_network::TwoSiteCenter<double, uni20::BlockSpace, uni20::LocalSpace, uni20::LocalSpace,
                                           uni20::BlockSpace, uni20::ParallelSeparateSparseBlockStorage<>>;
  ParallelCenter output(symmetry, uni20::Domain{left_bond, left_physical, right_physical}, uni20::Codomain{right_bond},
                        {typename ParallelCenter::key_type{{0, 0, 0, 0}}});
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler use_scheduler(&scheduler);
  effective_hamiltonian(output, input);

  auto result = output.block(typename ParallelCenter::key_type{{0, 0, 0, 0}});
  EXPECT_DOUBLE_EQ((result[0, 0]), 782.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 946.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 1774.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 2146.0);

  first_mpo.block(mpo_key)[] = 3.0;
  effective_hamiltonian(output, input);
  EXPECT_DOUBLE_EQ((result[0, 0]), 782.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 946.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 1774.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 2146.0);

  auto recompiled_hamiltonian = uni20::tensor_network::make_two_site_effective_hamiltonian(
      input, left_environment, first_mpo, second_mpo, right_environment);
  recompiled_hamiltonian(output, input);
  EXPECT_DOUBLE_EQ((result[0, 0]), 1173.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 1419.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 2661.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 3219.0);
}

} // namespace
