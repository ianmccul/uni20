#include <uni20/async/debug_scheduler.hpp>
#include <uni20/models/spin_half_heisenberg.hpp>
#include <uni20/tensor_network/environment_cache.hpp>
#include <uni20/tensor_network/two_site_dmrg.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace
{

class RecordingBatchScheduler : public uni20::async::DebugScheduler {
  public:
    std::size_t batch_calls = 0;
    std::size_t maximum_batch_size = 0;

  private:
    void execute_batch_impl(uni20::async::LightweightTaskBatch const& batch) override
    {
      ++batch_calls;
      if (batch.size() > maximum_batch_size) maximum_batch_size = batch.size();
      for (std::size_t index = 0; index < batch.size(); ++index)
        batch(index);
    }
};

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

TEST(SpinHalfHeisenbergModelTest, BuildsNeelProductMpsWithSelectedStorage)
{
  using storage_type = uni20::ParallelSeparateSparseBlockStorage<>;
  auto const local = uni20::models::make_spin_half_u1_site();
  auto const mps = uni20::models::make_neel_product_mps<double, storage_type>(4, local);

  static_assert(std::same_as<typename std::remove_cvref_t<decltype(mps)>::storage_policy, storage_type>);
  ASSERT_EQ(mps.size(), 4);
  for (std::size_t site = 0; site < mps.size(); ++site)
  {
    ASSERT_EQ(mps.site(site).stored_block_count(), 1);
    EXPECT_DOUBLE_EQ((mps.site(site).block_by_ordinal(0)[0, 0]), 1.0);
  }
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

TEST(SpinHalfHeisenbergModelTest, ConvergesAnUntruncatedLengthFourDmrgRun)
{
  using environment_storage = uni20::ParallelPackedSparseBlockStorage<>;
  using center_storage = uni20::ParallelPackedCompleteBlockStorage<>;
  auto const local = uni20::models::make_spin_half_u1_site();
  auto mps = uni20::models::make_neel_product_mps(4, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo(4, local);
  using mps_type = std::remove_cvref_t<decltype(mps)>;
  using mpo_type = std::remove_cvref_t<decltype(mpo)>;
  uni20::tensor_network::MpoEnvironmentCache<mps_type, mpo_type, environment_storage> cache(mps, mpo, 0, 0);
  RecordingBatchScheduler scheduler;
  uni20::async::ScopedScheduler use_scheduler(&scheduler);
  uni20::tensor_network::TwoSiteDmrgRunOptions<double> options;
  options.maximum_sweeps = 8;
  options.energy_tolerance = 1.0e-12;
  options.bond_options.truncation.maximum_retained_extent = 16;

  auto const result = uni20::tensor_network::run_two_site_dmrg(mps, mpo, cache, options, center_storage{});
  ASSERT_TRUE(result.converged);
  ASSERT_GE(result.sweeps.size(), 2);
  ASSERT_LE(result.sweeps.size(), options.maximum_sweeps);
  double const exact_energy = -(3.0 + 2.0 * std::sqrt(3.0)) / 4.0;
  EXPECT_NEAR(result.sweeps.back().terminal_local_energy, exact_energy, 1.0e-12);
  EXPECT_TRUE(result.sweeps.back().terminal_energy_is_global);
  EXPECT_TRUE(result.sweeps.back().energy_change.has_value());
  EXPECT_LE(*result.sweeps.back().energy_change, options.energy_tolerance * std::abs(exact_energy));
  for (std::size_t index = 0; index < result.sweeps.size(); ++index)
  {
    auto const& sweep = result.sweeps[index];
    auto const expected_direction = index % 2 == 0 ? uni20::tensor_network::MpsSweepDirection::left_to_right
                                                   : uni20::tensor_network::MpsSweepDirection::right_to_left;
    EXPECT_EQ(sweep.sweep_index, index);
    EXPECT_EQ(sweep.direction, expected_direction);
    EXPECT_TRUE(sweep.terminal_energy_is_global);
    EXPECT_DOUBLE_EQ(sweep.terminal_discarded_weight, 0.0);
    EXPECT_DOUBLE_EQ(sweep.maximum_discarded_weight, 0.0);
    EXPECT_LE(sweep.maximum_bond_dimension, 4);
    EXPECT_GT(sweep.total_matvec_count, 0);
  }
  EXPECT_GT(scheduler.batch_calls, 0);
  EXPECT_GT(scheduler.maximum_batch_size, 1);
}

TEST(SpinHalfHeisenbergModelTest, DoesNotConvergeFromTruncatedTerminalEnergies)
{
  auto const local = uni20::models::make_spin_half_u1_site();
  auto mps = uni20::models::make_neel_product_mps(2, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo(2, local);
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);
  uni20::tensor_network::TwoSiteDmrgRunOptions<double> options;
  options.maximum_sweeps = 2;
  options.energy_tolerance = 1.0;
  options.bond_options.truncation.maximum_retained_extent = 1;

  auto const result = uni20::tensor_network::run_two_site_dmrg(mps, mpo, cache, options);
  EXPECT_FALSE(result.converged);
  ASSERT_EQ(result.sweeps.size(), 2);
  for (auto const& sweep : result.sweeps)
  {
    EXPECT_FALSE(sweep.terminal_energy_is_global);
    EXPECT_FALSE(sweep.energy_change.has_value());
    EXPECT_GT(sweep.terminal_discarded_weight, 0.0);
    EXPECT_FALSE(sweep.converged);
  }
}

TEST(SpinHalfHeisenbergModelTest, RecordsOptInDmrgPhaseAndSvdBatchMeasurements)
{
  using environment_storage = uni20::ParallelPackedSparseBlockStorage<>;
  using center_storage = uni20::ParallelPackedCompleteBlockStorage<>;
  using event = uni20::tensor_network::TwoSiteDmrgPerformanceEvent;
  auto const local = uni20::models::make_spin_half_u1_site();
  auto mps = uni20::models::make_neel_product_mps(2, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo(2, local);
  using mps_type = std::remove_cvref_t<decltype(mps)>;
  using mpo_type = std::remove_cvref_t<decltype(mpo)>;
  uni20::tensor_network::MpoEnvironmentCache<mps_type, mpo_type, environment_storage> cache(mps, mpo, 0, 0);
  RecordingBatchScheduler scheduler;
  uni20::async::ScopedScheduler use_scheduler(&scheduler);
  uni20::tensor_network::TwoSiteDmrgRunOptions<double> options;
  options.maximum_sweeps = 1;
  uni20::tensor_network::DetailedTwoSiteDmrgPerformanceMeasurements measurements;

  auto const result =
      uni20::tensor_network::run_two_site_dmrg(mps, mpo, cache, options, measurements, center_storage{});

  ASSERT_EQ(result.sweeps.size(), 1U);
  EXPECT_EQ(measurements[event::run].count, 1U);
  EXPECT_EQ(measurements[event::sweep].count, 1U);
  EXPECT_EQ(measurements[event::bond_update].count, 1U);
  EXPECT_EQ(measurements[event::center_construction].count, 1U);
  EXPECT_EQ(measurements[event::local_eigensolver].count, 1U);
  EXPECT_EQ(measurements[event::effective_hamiltonian_application].count,
            static_cast<std::size_t>(result.sweeps.front().total_matvec_count));
  EXPECT_GT(measurements[event::krylov_vector_allocation].count, 0U);
  EXPECT_GT(measurements[event::krylov_vector_update].count, 0U);
  EXPECT_GT(measurements[event::krylov_reduction].count, 0U);
  EXPECT_EQ(measurements[event::block_svd].count, 1U);
  EXPECT_EQ(measurements[event::state_selection].count, 1U);
  EXPECT_EQ(measurements[event::factor_materialization].count, 1U);
  EXPECT_EQ(measurements[event::environment_update].count, 1U);

  ASSERT_EQ(measurements.batches(event::svd_sector_batch).size(), 1U);
  auto const& batch = measurements.batches(event::svd_sector_batch).front();
  EXPECT_GT(batch.requested_items, 0U);
  EXPECT_EQ(batch.started_items, batch.requested_items);
  EXPECT_EQ(batch.completed_items, batch.requested_items);
  EXPECT_EQ(measurements[event::svd_sector_batch].count, 1U);
}

TEST(SpinHalfHeisenbergModelTest, RejectsInvalidRunOptionsBeforeChangingTheMps)
{
  auto const local = uni20::models::make_spin_half_u1_site();
  auto mps = uni20::models::make_neel_product_mps(2, local);
  auto const mpo = uni20::models::make_spin_half_heisenberg_mpo(2, local);
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);
  uni20::tensor_network::TwoSiteDmrgRunOptions<double> options;
  options.maximum_sweeps = 0;

  EXPECT_THROW(static_cast<void>(uni20::tensor_network::run_two_site_dmrg(mps, mpo, cache, options)),
               std::invalid_argument);
  EXPECT_EQ(mps.revision(0), 0);
  EXPECT_EQ(mps.revision(1), 0);

  options.maximum_sweeps = 1;
  options.bond_options.local_solver.matvec_iterations = 0;
  EXPECT_THROW(static_cast<void>(uni20::tensor_network::run_two_site_dmrg(mps, mpo, cache, options)),
               std::invalid_argument);
  EXPECT_EQ(mps.revision(0), 0);
  EXPECT_EQ(mps.revision(1), 0);
}

} // namespace
