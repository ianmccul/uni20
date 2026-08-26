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

namespace
{

using Storage = uni20::SeparateSparseBlockStorage<>;
using Center = uni20::tensor_network::TwoSiteCenter<double, uni20::BlockSpace, uni20::LocalSpace, uni20::LocalSpace,
                                                    uni20::BlockSpace, Storage>;
using Hamiltonian = uni20::tensor_network::TwoSiteLocalOperator<double, uni20::LocalSpace, uni20::LocalSpace>;
using CenterKey = typename Center::key_type;
using HamiltonianKey = typename Hamiltonian::key_type;
using MpsSite = uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
using MpoSite =
    uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace>;
using ScalarEnvironment = uni20::tensor_network::ScalarEnvironment<double>;

static_assert(std::same_as<typename MpsSite::domain_type, uni20::Domain<uni20::BlockSpace, uni20::LocalSpace>>);
static_assert(std::same_as<typename MpsSite::codomain_type, uni20::Codomain<uni20::BlockSpace>>);
static_assert(std::same_as<typename MpoSite::domain_type, uni20::Domain<uni20::LocalSpace, uni20::LocalSpace>>);
static_assert(std::same_as<typename MpoSite::codomain_type, uni20::Codomain<uni20::LocalSpace, uni20::LocalSpace>>);
static_assert(ScalarEnvironment::order() == 0);

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
  uni20::tensor_network::TwoSiteEffectiveHamiltonian effective_hamiltonian(std::move(hamiltonian));
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

} // namespace
