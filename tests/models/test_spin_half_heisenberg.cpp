#include <uni20/models/spin_half_heisenberg.hpp>
#include <uni20/tensor_network/environment_cache.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace
{

TEST(SpinHalfHeisenbergModelTest, BuildsTheOrderedU1LocalSpace)
{
  auto const local = uni20::models::make_spin_half_u1_site();
  ASSERT_EQ(local.space.size(), 2);
  EXPECT_EQ(local.space[0], local.up);
  EXPECT_EQ(local.space[1], local.down);
  EXPECT_EQ(uni20::u1_component(local.up, "Sz"), uni20::U1{uni20::half_int{0.5}});
  EXPECT_EQ(uni20::u1_component(local.down, "Sz"), uni20::U1{uni20::half_int{-0.5}});
  EXPECT_EQ(uni20::models::SpinHalfU1Site::coordinate(uni20::models::SpinHalfState::up), 0);
  EXPECT_EQ(uni20::models::SpinHalfU1Site::coordinate(uni20::models::SpinHalfState::down), 1);
  EXPECT_THROW(static_cast<void>(uni20::models::make_spin_half_u1_site("")), std::invalid_argument);
}

TEST(SpinHalfHeisenbergModelTest, BuildsNormalizedNeelProductMpsWithCumulativeCharges)
{
  auto const local = uni20::models::make_spin_half_u1_site();
  auto const mps = uni20::models::make_neel_product_mps(4, local);

  ASSERT_EQ(mps.size(), 4);
  auto const q0 = uni20::QNum::identity(local.symmetry);
  for (std::size_t site = 0; site < mps.size(); ++site)
  {
    auto const& value = mps.site(site);
    ASSERT_EQ(value.stored_block_count(), 1);
    EXPECT_DOUBLE_EQ((value.block_by_ordinal(0)[0, 0]), 1.0);
    EXPECT_EQ(value.domain().template space<1>(), local.space);
    EXPECT_EQ(value.stored_keys()[0].coordinate(1), site % 2 == 0 ? 0 : 1);
    EXPECT_EQ(value.domain().template space<0>().label(), "neel-mps-bond-" + std::to_string(site));
    EXPECT_EQ(value.codomain().template space<0>().label(), "neel-mps-bond-" + std::to_string(site + 1));
  }
  EXPECT_EQ(mps.site(0).domain().template space<0>()[0].q, q0);
  EXPECT_EQ(mps.site(0).codomain().template space<0>()[0].q, local.up);
  EXPECT_EQ(mps.site(1).codomain().template space<0>()[0].q, q0);
  EXPECT_EQ(mps.site(3).codomain().template space<0>()[0].q, q0);

  auto const down_first = uni20::models::make_neel_product_mps(3, local, uni20::models::SpinHalfState::down);
  EXPECT_EQ(down_first.site(0).stored_keys()[0].coordinate(1), 1);
  EXPECT_EQ(down_first.site(2).codomain().template space<0>()[0].q, local.down);
  EXPECT_THROW(static_cast<void>(uni20::models::make_neel_product_mps(0, local)), std::invalid_argument);
}

TEST(SpinHalfHeisenbergModelTest, BuildsReducedBoundaryHeisenbergMpo)
{
  auto const local = uni20::models::make_spin_half_u1_site();
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo(4, local, 1.5, 0.25);

  ASSERT_EQ(mpo.size(), 4);
  EXPECT_EQ(mpo.site(0).domain().template space<0>().size(), 1);
  EXPECT_EQ(mpo.site(0).codomain().template space<0>().size(), 5);
  EXPECT_EQ(mpo.site(1).domain().template space<0>().size(), 5);
  EXPECT_EQ(mpo.site(1).codomain().template space<0>().size(), 5);
  EXPECT_EQ(mpo.site(3).codomain().template space<0>().size(), 1);
  for (std::size_t site = 0; site < mpo.size(); ++site)
  {
    EXPECT_EQ(mpo.site(site).domain().template space<1>(), local.space);
    EXPECT_EQ(mpo.site(site).codomain().template space<1>(), local.space);
  }

  auto const& bulk_auxiliary = mpo.site(1).domain().template space<0>();
  EXPECT_EQ(uni20::u1_component(bulk_auxiliary[1], "Sz"), uni20::U1{-1});
  EXPECT_EQ(uni20::u1_component(bulk_auxiliary[2], "Sz"), uni20::U1{1});
  EXPECT_TRUE(uni20::is_identity(bulk_auxiliary[0]));
  EXPECT_TRUE(uni20::is_identity(bulk_auxiliary[3]));
  EXPECT_TRUE(uni20::is_identity(bulk_auxiliary[4]));
  EXPECT_THROW(static_cast<void>(uni20::models::make_spin_half_heisenberg_mpo(0, local)), std::invalid_argument);
}

TEST(SpinHalfHeisenbergModelTest, ProducesTheExpectedNeelStateEnergy)
{
  auto const local = uni20::models::make_spin_half_u1_site();
  auto const mps = uni20::models::make_neel_product_mps(4, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo(4, local, 1.5, 0.25);
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);

  auto const& expectation = cache.left_environment(4);
  ASSERT_EQ(expectation.stored_block_count(), 1);
  EXPECT_NEAR((expectation.block_by_ordinal(0)[0, 0]), -3.0 * 1.5 / 4.0, 1.0e-14);
  auto const& reverse_expectation = cache.right_environment(0);
  ASSERT_EQ(reverse_expectation.stored_block_count(), 1);
  EXPECT_NEAR((reverse_expectation.block_by_ordinal(0)[0, 0]), -3.0 * 1.5 / 4.0, 1.0e-14);
}

TEST(SpinHalfHeisenbergModelTest, SupportsTheOneSiteFieldHamiltonian)
{
  auto const local = uni20::models::make_spin_half_u1_site();
  auto const mps = uni20::models::make_neel_product_mps(1, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo(1, local, 7.0, 0.4);
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);

  EXPECT_NEAR((cache.left_environment(1).block_by_ordinal(0)[0, 0]), 0.2, 1.0e-14);
}

TEST(SpinHalfHeisenbergModelTest, SupportsComplexStorageWithRealCouplings)
{
  using value_type = uni20::complex<double>;
  auto const local = uni20::models::make_spin_half_u1_site();
  auto const mps = uni20::models::make_neel_product_mps<value_type>(2, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo<value_type>(2, local, 2.0, 0.3);
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);

  value_type const expectation = cache.left_environment(2).block_by_ordinal(0)[0, 0];
  EXPECT_NEAR(expectation.real(), -0.5, 1.0e-14);
  EXPECT_NEAR(expectation.imag(), 0.0, 1.0e-14);
}

} // namespace
