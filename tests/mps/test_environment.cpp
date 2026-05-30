#include <uni20/models/spin_half.hpp>
#include <uni20/mps/environment.hpp>

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace uni20;

namespace
{

auto bond_space(Symmetry sym, std::size_t dim) -> BlockSpace
{
  return BlockSpace(sym, {BlockSector{QNum::identity(sym), dim}});
}

auto scalar_virtual_space(Symmetry sym) -> LocalSpace { return LocalSpace(sym, {QNum::identity(sym)}); }

auto identity_component(SpinHalfSite const& site) -> OperatorComponent
{
  auto virtual_space = scalar_virtual_space(site.symmetry);
  OperatorComponent component(site.space, site.space, virtual_space, virtual_space);
  component.insert_or_assign(0, 0, site.identity);
  return component;
}

} // namespace

TEST(MpoEnvironmentTest, LeftUpdateContractsDenseMpsWithSparseIdentityOperator)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor site(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 3));
  site.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  site.assign(1, std::array{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
  auto component = identity_component(spin);
  MpoEnvironment left_env(component.left_virtual_space(), site.left_bond_space());
  left_env.set_identity(0);

  auto next = extend_left_environment(left_env, site, component);

  std::array expected{
      166.0, 188.0, 210.0, 188.0, 214.0, 240.0, 210.0, 240.0, 270.0,
  };
  ASSERT_EQ(next.values(0).size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(next.values(0)[i], expected[i]);
  }
}

TEST(MpoEnvironmentTest, SparseLocalOperatorEntriesContributeToEnvironment)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor site(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  site.assign(0, std::array{2.0});
  site.assign(1, std::array{3.0});
  auto virtual_space = scalar_virtual_space(spin.symmetry);
  OperatorComponent component(spin.space, spin.space, virtual_space, virtual_space);
  component.insert_or_assign(0, 0, spin.sz);
  MpoEnvironment left_env(component.left_virtual_space(), site.left_bond_space());
  left_env.set_identity(0);

  auto next = extend_left_environment(left_env, site, component);

  EXPECT_DOUBLE_EQ(next.values(0)[0], -2.5);
}

TEST(MpoEnvironmentTest, RightUpdateMatchesLeftUpdateForOneSiteScalarEnvironment)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor site(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  site.assign(0, std::array{2.0});
  site.assign(1, std::array{3.0});
  auto virtual_space = scalar_virtual_space(spin.symmetry);
  OperatorComponent component(spin.space, spin.space, virtual_space, virtual_space);
  component.insert_or_assign(0, 0, spin.sz);
  MpoEnvironment right_env(component.right_virtual_space(), site.right_bond_space());
  right_env.set_identity(0);

  auto previous = extend_right_environment(right_env, site, component);

  EXPECT_DOUBLE_EQ(previous.values(0)[0], -2.5);
}

TEST(MpoEnvironmentTest, BuildsLeftAndRightEnvironmentChains)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor left(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  MpsSiteTensor right(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  left.assign(0, std::array{1.0});
  left.assign(1, std::array{2.0});
  right.assign(0, std::array{3.0});
  right.assign(1, std::array{4.0});
  FiniteMPS psi({std::move(left), std::move(right)});

  auto first = identity_component(spin);
  auto second = identity_component(spin);
  FiniteTriangularMPO mpo({std::move(first), std::move(second)});

  auto left_envs = build_left_environments(psi, mpo);
  auto right_envs = build_right_environments(psi, mpo);

  ASSERT_EQ(left_envs.size(), 3);
  ASSERT_EQ(right_envs.size(), 3);
  EXPECT_DOUBLE_EQ(left_envs[0].values(0)[0], 1.0);
  EXPECT_DOUBLE_EQ(left_envs[1].values(0)[0], 5.0);
  EXPECT_DOUBLE_EQ(left_envs[2].values(0)[0], 125.0);
  EXPECT_DOUBLE_EQ(right_envs[2].values(0)[0], 1.0);
  EXPECT_DOUBLE_EQ(right_envs[1].values(0)[0], 25.0);
  EXPECT_DOUBLE_EQ(right_envs[0].values(0)[0], 125.0);
}

TEST(MpoEnvironmentTest, RejectsMismatchedChainLengths)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor site(spin.space, bond_space(spin.symmetry, 1), bond_space(spin.symmetry, 1));
  FiniteMPS psi({std::move(site)});

  auto first = identity_component(spin);
  auto second = identity_component(spin);
  FiniteTriangularMPO mpo({std::move(first), std::move(second)});

  EXPECT_THROW(static_cast<void>(make_left_boundary_environment(psi, mpo)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_right_boundary_environment(psi, mpo)), std::invalid_argument);
}
