#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/environment.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

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

void expect_environment_near(MpoEnvironment const& actual, MpoEnvironment const& expected)
{
  ASSERT_EQ(actual.virtual_dim(), expected.virtual_dim());
  for (std::size_t block = 0; block < expected.virtual_dim(); ++block)
  {
    SCOPED_TRACE(::testing::Message() << "block " << block);
    ASSERT_EQ(actual.values(block).size(), expected.values(block).size());
    for (std::size_t i = 0; i < expected.values(block).size(); ++i)
    {
      EXPECT_NEAR(actual.values(block)[i], expected.values(block)[i], 1.0e-10);
    }
  }
}

} // namespace

TEST(TensorContractionMpsEnvironmentTest, ScalarLeftAndRightUpdatesMatchHostReference)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor site(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 3));
  site.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  site.assign(1, std::array{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
  auto component = identity_component(spin);

  MpoEnvironment left_env(component.left_virtual_space(), site.left_bond_space());
  left_env.set_identity(0);
  expect_environment_near(detail::left_environment_tensorcontraction(left_env, site, component),
                          detail::left_environment_host(left_env, site, component));

  MpoEnvironment right_env(component.right_virtual_space(), site.right_bond_space());
  right_env.set_identity(0);
  expect_environment_near(detail::right_environment_tensorcontraction(right_env, site, component),
                          detail::right_environment_host(right_env, site, component));
}

TEST(TensorContractionMpsEnvironmentTest, HeisenbergLeftAndRightUpdatesMatchHostReference)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor site(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 3));
  site.assign(0, std::array{1.0, -2.0, 3.0, 0.5, -1.5, 2.5});
  site.assign(1, std::array{-0.5, 1.25, 0.75, -2.25, 3.5, -4.0});
  auto component = make_spin_half_heisenberg_bulk_component(spin, 1.0, 0.0);

  MpoEnvironment left_env(component.left_virtual_space(), site.left_bond_space());
  left_env.set_identity(0);
  auto left_expected = detail::left_environment_host(left_env, site, component);
  auto left_actual = detail::left_environment_tensorcontraction(left_env, site, component);
  expect_environment_near(left_actual, left_expected);

  MpoEnvironment right_env(component.right_virtual_space(), site.right_bond_space());
  right_env.set_identity(right_env.virtual_dim() - 1);
  auto right_expected = detail::right_environment_host(right_env, site, component);
  auto right_actual = detail::right_environment_tensorcontraction(right_env, site, component);
  expect_environment_near(right_actual, right_expected);
}

TEST(TensorContractionMpsEnvironmentTest, HeisenbergUpdatesWithNontrivialEnvironmentMatchHostReference)
{
  auto const spin = make_spin_half_u1_site();
  MpsSiteTensor site(spin.space, bond_space(spin.symmetry, 2), bond_space(spin.symmetry, 3));
  site.assign(0, std::array{1.0, -2.0, 3.0, 0.5, -1.5, 2.5});
  site.assign(1, std::array{-0.5, 1.25, 0.75, -2.25, 3.5, -4.0});
  auto component = make_spin_half_heisenberg_bulk_component(spin, 1.0, 0.0);

  MpoEnvironment left_env(component.left_virtual_space(), site.left_bond_space());
  for (std::size_t block = 0; block < left_env.virtual_dim(); ++block)
  {
    auto values = left_env.values(block);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
      values[i] = 0.25 * static_cast<double>((block + 1) * (i + 2));
    }
  }
  expect_environment_near(detail::left_environment_tensorcontraction(left_env, site, component),
                          detail::left_environment_host(left_env, site, component));

  MpoEnvironment right_env(component.right_virtual_space(), site.right_bond_space());
  for (std::size_t block = 0; block < right_env.virtual_dim(); ++block)
  {
    auto values = right_env.values(block);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
      values[i] = -0.125 * static_cast<double>((block + 3) * (i + 1));
    }
  }
  expect_environment_near(detail::right_environment_tensorcontraction(right_env, site, component),
                          detail::right_environment_host(right_env, site, component));
}
